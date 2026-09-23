from __future__ import annotations

from pathlib import Path
from string import Template
from subprocess import DEVNULL, Popen, run

from relmedner.clusters import YamlClusterParser
from relmedner.constants import (
    BLOB_SERVER_PORT,
    FLINK_DOCKERFILE,
    FLINK_IMAGE_NAME,
    FLINK_REST_PORT,
    FLINK_VERSION,
    FULLMAP_MOUNT,
    JOBMANAGER_COMPOSE,
    JOBMANAGER_RPC_PORT,
    OUTPUTS_MOUNT,
    PROJECT,
    TASKMANAGER_COMPOSE,
    TASKMANAGER_DATA_PORT,
    WORKER_DOCKERFILE,
)
from relmedner.models import WorkerNode

COMPOSE: list[str] = ["docker", "compose"]
JOBMANAGER_PROJECT: str = f"{PROJECT}-head"
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def run_cmd(
    cmd: list[str],
    dry_run: bool = False,
    stdin: str | None = None,
    cwd: Path | None = None,
) -> None:
    display: str = " ".join(cmd) + (" <<'compose-yaml'" if stdin else "")
    if dry_run:
        print(f"[dry-run] {display}")
        return

    if run(cmd, input=stdin.encode() if stdin else None, check=False, cwd=cwd).returncode != 0:
        raise SystemExit(f"command failed: {display}")

    return


def ssh_to(user: str, host: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", f"{user}@{host}"]


def compose_vars(worker: WorkerNode, flink_image: str, worker_image: str, data_port: int) -> dict[str, str]:
    """every advertised flink address is 127.0.0.1: cross-host ports are firewalled off, so peers
    (and the driver on the head host) reach rpc/blob/data through the loopback tunnels deploy spawns"""
    return {
        "FLINK_IMAGE": flink_image,
        "WORKER_IMAGE": worker_image,
        "JOBMANAGER_HOST": "127.0.0.1",
        "FLINK_REST_PORT": str(FLINK_REST_PORT),
        "JOBMANAGER_RPC_PORT": str(JOBMANAGER_RPC_PORT),
        "BLOB_SERVER_PORT": str(BLOB_SERVER_PORT),
        "SLOTS": str(worker.slots),
        "MEMORY": worker.memory,
        "FULLMAP_DIR": worker.fullmap,
        "FULLMAP_MOUNT": FULLMAP_MOUNT,
        "POLARS_RUNTIME": worker.polars_runtime,
        "OUTPUTS_DIR": worker.outputs,
        "OUTPUTS_MOUNT": OUTPUTS_MOUNT,
        "TASKMANAGER_HOST": "127.0.0.1",
        "TASKMANAGER_DATA_PORT": str(data_port),
        "TASKMANAGER_DATA_BIND_PORT": str(data_port),
    }


def compose_command(user: str, worker: WorkerNode, project: str) -> list[str]:
    """`docker compose` plugin when the host has one, else a standalone docker-compose binary
    (hypatia gets v2 through the nix profile — no sudo for a system plugin install).

    the probe travels as ONE argv element: ssh joins its arguments into a remote shell line, so a
    locally-quoted `sh -c "script"` would lose its quoting and silently run the bare `command`
    builtin (exit 0, no output) instead of the probe; the remote login shell evaluates the line.
    the taskmanager and jobmanager stacks use DIFFERENT projects: both files target the same docker
    host on the head node, and one project's --remove-orphans would reap the other's containers
    """
    prefix: list[str] = ssh_to(user, worker.host)
    if run([*prefix, "docker", "compose", "version"], capture_output=True, check=False).returncode == 0:
        return [*prefix, *COMPOSE, "-p", project]
    probe_script: str = (
        'command -v docker-compose || { test -x "$HOME/.nix-profile/bin/docker-compose" && echo "$HOME/.nix-profile/bin/docker-compose"; }'
    )
    probe = run([*prefix, probe_script], capture_output=True, check=False)
    compose_binary: str = probe.stdout.decode().strip()
    if probe.returncode != 0 or not compose_binary:
        raise SystemExit(f"no docker compose plugin and no docker-compose binary on {worker.host}")
    return [*prefix, compose_binary, "-p", project]


def image_id(prefix: list[str], image: str) -> str:
    """content id of an image, empty when the tag is absent — tag equality is not enough

    the worker tag carries the package version, so a rebuilt wheel at the same version keeps the
    same tag and remotes would silently keep running the previous code
    """
    Result = run([*prefix, "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, check=False)
    return Result.stdout.decode().strip() if Result.returncode == 0 else ""


def data_port(index: int) -> int:
    """per-worker taskmanager data port: the head (index 0) keeps the canonical port"""
    return TASKMANAGER_DATA_PORT + index


def tunnel_spec(forwards: list[int], target: str, user: str) -> list[str]:
    """inner ssh command: advertise target's ports on this host's loopback over the :22-only path"""
    spec: list[str] = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-N",
    ]
    for port in forwards:
        spec += ["-L", f"127.0.0.1:{port}:127.0.0.1:{port}"]
    spec.append(f"{user}@{target}")
    return spec


def tunnel_plan(
    workers: list[WorkerNode],
    jobmanager_host: str,
) -> dict[str, dict[str, list[int]]]:
    """host -> tunnel target -> loopback forwards that host needs

    every taskmanager must reach the jobmanager rpc+blob ports and every OTHER taskmanager's data
    port; each group rides one ssh session to the host that owns the ports
    """
    plan: dict[str, dict[str, list[int]]] = {}
    for worker in workers:
        if worker.host != jobmanager_host:
            plan.setdefault(worker.host, {}).setdefault(jobmanager_host, []).extend([JOBMANAGER_RPC_PORT, BLOB_SERVER_PORT])
        for other_index, other in enumerate(workers):
            if other.host == worker.host:
                continue
            plan.setdefault(worker.host, {}).setdefault(other.host, []).append(data_port(other_index))
    return plan


DYN_WATCHER_REMOTE: str = "/tmp/relmedner-dynwatch.sh"
DYN_WATCHER_LOG: str = "/tmp/relmedner-dynwatch.log"
DYN_CTL_SOCKET: str = "/tmp/relmedner-dyn-ctl"
DYN_SEEN_REMOTE: str = "/tmp/relmedner-dyn-ports"


def dyn_forwarder_script(jobmanager_host: str, user: str) -> str:
    """bash file, installed on a taskmanager host and run DETACHED (setsid nohup): the beam worker
    pool dials the job server's FnAPI endpoints, which the java job server binds on RANDOM ports
    per submission - unfixable from flags. the cross-host firewall means those dials must be
    relayed over ssh, so poll the pool log for every new provision/control/log/artifact endpoint
    and attach a forward for it. one PERSISTENT ssh master carries the listeners (its survival
    profile matches the static tunnels, which outlived everything); each new port is attached
    live via `ssh -O forward`. per-port ssh processes were tried first and died en masse."""
    return f"""#!/bin/bash
seen={DYN_SEEN_REMOTE}
ctl={DYN_CTL_SOCKET}
: > "$seen"
# POLL, don't stream: a long-lived `docker logs -f | grep | grep` pipeline dies silently when any
# member exits (observed mid-job), while a fresh pipeline per tick is self-healing by construction
while true; do
  # ensure the master carrier; -O check pings the control socket
  if ! ssh -O check -S "$ctl" {user}@{jobmanager_host} >/dev/null 2>&1; then
    setsid nohup ssh -M -S "$ctl" -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \\
      -o ControlPersist=no -N {user}@{jobmanager_host} </dev/null >/dev/null 2>&1 &
    sleep 1
  fi
  # 2>&1, never 2>/dev/null: hypatia's docker CLI emits log lines on STDERR — discarding it
  # blinded the watcher from day one (empty seen file, zero forwards, workers dialing dead ports)
  for p in $(docker logs --since 2m relmedner-sdkworker-1 2>&1 \\
      | grep -oE '(provision|control|logging|artifact)_endpoint.{{0,16}}localhost:[0-9]+' \\
      | grep -oE 'localhost:[0-9]+' | grep -oE '[0-9]+' | sort -u); do
    grep -qx "$p" "$seen" 2>/dev/null && continue
    # attach to the master; only remember ports whose forward actually bound
    if ssh -S "$ctl" -O forward -o BatchMode=yes \\
        -L 127.0.0.1:$p:127.0.0.1:$p {user}@{jobmanager_host} >/dev/null 2>&1; then
      echo "$p" >> "$seen"
    fi
  done
  sleep 2
done
"""


def deploy_cluster(teardown: bool = False, dry_run: bool = False) -> None:
    Parser: YamlClusterParser = YamlClusterParser()
    ClusterSpec = Parser.parse_cluster()
    ssh_user: str = ClusterSpec.ssh_user
    jobmanager_host: str = ClusterSpec.jobmanager
    workers: list[WorkerNode] = Parser.remotes()
    jobmanager_worker: WorkerNode = next(worker for worker in workers if worker.host == jobmanager_host)
    flink_image: str = f"{FLINK_IMAGE_NAME}:{FLINK_VERSION}"
    worker_image: str = Parser.worker_image()
    Plan: dict[str, dict[str, list[int]]] = tunnel_plan(workers, jobmanager_host)

    def render_tm(worker: WorkerNode, index: int) -> str:
        # explicit encoding: the compose templates carry non-ascii and ship to remote hosts whose
        # locale is not guaranteed utf-8
        return Template(TASKMANAGER_COMPOSE.read_text(encoding="utf-8")).substitute(compose_vars(worker, flink_image, worker_image, data_port(index)))

    def kill_tunnel_session(host: str, target: str) -> None:
        # one argv element: ssh re-parses separate words on the remote side; the pid file is the
        # handle on a detached tunnel (no tmux - a tmux server dying mid-job once took the relays
        # down with it and killed the job)
        pid_file = f"/tmp/relmedner-tunnel-{target.replace('.', '-')}.pid"
        run_cmd([*ssh_to(ssh_user, host), f"kill $(cat {pid_file}) 2>/dev/null; rm -f {pid_file}; true"], dry_run)

    if teardown:
        for host, targets in Plan.items():
            for target in targets:
                kill_tunnel_session(host, target)
        for worker in workers:
            if worker.host != jobmanager_host:
                run_cmd(
                    [
                        *ssh_to(ssh_user, worker.host),
                        "pkill -f 'relmedner-dynwatch[.]sh' 2>/dev/null; "
                        "ssh -S " + DYN_CTL_SOCKET + " -O exit " + ssh_user + "@" + jobmanager_host + " 2>/dev/null; "
                        "rm -f " + DYN_WATCHER_REMOTE + " " + DYN_SEEN_REMOTE + " " + DYN_CTL_SOCKET + "; true",
                    ],
                    dry_run,
                )
        jm_index: int = workers.index(jobmanager_worker)
        for index, worker in enumerate(workers):
            run_cmd([*compose_command(ssh_user, worker, PROJECT), "-f", "-", "down"], dry_run, stdin=render_tm(worker, index))
        run_cmd(
            [*compose_command(ssh_user, jobmanager_worker, JOBMANAGER_PROJECT), "-f", "-", "down"],
            dry_run,
            stdin=render_tm(jobmanager_worker, jm_index),
        )
        return

    # fail fast on a missing fullmap bundle BEFORE any image build or compose run — a node
    # without its mount would otherwise fail later at sdkworker startup, mid-deploy
    for worker in workers:
        run_cmd([*ssh_to(ssh_user, worker.host), "test", "-d", worker.fullmap], dry_run)
        # the outputs dir is ours to create: docker would otherwise make it root-owned on first mount
        run_cmd([*ssh_to(ssh_user, worker.host), "mkdir", "-p", worker.outputs], dry_run)

    for cmd in (
        ["rm", "-rf", "dist"],
        ["uv", "build", "--wheel", "--out-dir", "dist"],
        ["docker", "build", "-f", str(WORKER_DOCKERFILE), "-t", worker_image, "."],
        ["docker", "build", "-f", str(FLINK_DOCKERFILE), "-t", flink_image, "."],
    ):
        run_cmd(cmd, dry_run, cwd=PROJECT_ROOT)

    local_ids: dict[str, str] = {image: image_id(["docker"], image) for image in (flink_image, worker_image)}
    for worker in workers:
        # no registry — pipe images down the ssh connection we already have, skipping only when the
        # remote already holds the exact same image id (a matching tag can still be stale code)
        for image in (flink_image, worker_image):
            local_id: str = local_ids[image]
            remote_id: str = image_id([*ssh_to(ssh_user, worker.host), "docker"], image)
            if dry_run or local_id != remote_id:
                run_cmd(["sh", "-c", f"docker save {image} | ssh -o BatchMode=yes {ssh_user}@{worker.host} docker load"], dry_run)

    for index, worker in enumerate(workers):
        run_cmd(
            [*compose_command(ssh_user, worker, PROJECT), "-f", "-", "up", "-d", "--force-recreate", "--remove-orphans"],
            dry_run,
            stdin=render_tm(worker, index),
        )

    JobmanagerRendered: str = Template(JOBMANAGER_COMPOSE.read_text(encoding="utf-8")).substitute(
        compose_vars(jobmanager_worker, flink_image, worker_image, data_port(workers.index(jobmanager_worker)))
    )
    run_cmd(
        [*compose_command(ssh_user, jobmanager_worker, JOBMANAGER_PROJECT), "-f", "-", "up", "-d", "--force-recreate", "--remove-orphans"],
        dry_run,
        stdin=JobmanagerRendered,
    )

    # inter-host tunnels over the :22-only path, detached with setsid so they survive both the
    # deploy shell and a tmux server death; flink's fixed-delay restart strategy absorbs the brief
    # drop while a redeploy re-establishes them
    for host, targets in Plan.items():
        for target, forwards in targets.items():
            kill_tunnel_session(host, target)
            if not dry_run:
                inner: str = " ".join(tunnel_spec(forwards, target, ssh_user))
                pid_file = f"/tmp/relmedner-tunnel-{target.replace('.', '-')}.pid"
                Popen(
                    [
                        *ssh_to(ssh_user, host),
                        # inner already starts with `ssh` (see tunnel_spec) — prefixing another one ran
                        # `ssh ssh ...`, which died instantly and left every host tunnel-less
                        f"setsid nohup {inner} </dev/null >/dev/null 2>&1 & echo $! > {pid_file}",
                    ],
                    start_new_session=True,
                    stdout=DEVNULL,
                    stderr=DEVNULL,
                )

    # per-job FnAPI forwarder watcher on every remote taskmanager host — the head's pool dials the
    # job server natively and needs no relay. the script ships as a remote file: inlining it in a
    # remote shell command would let the shell split it at the semicolons
    for worker in workers:
        if worker.host != jobmanager_host:
            run_cmd([*ssh_to(ssh_user, worker.host), "pkill -f 'relmedner-dynwatch[.]sh' 2>/dev/null; true"], dry_run)
            run_cmd([*ssh_to(ssh_user, worker.host), "cat > " + DYN_WATCHER_REMOTE], dry_run, stdin=dyn_forwarder_script(jobmanager_host, ssh_user))
            run_cmd(
                [
                    *ssh_to(ssh_user, worker.host),
                    "setsid nohup bash " + DYN_WATCHER_REMOTE + " > " + DYN_WATCHER_LOG + " 2>&1 < /dev/null &",
                ],
                dry_run,
            )

    return
