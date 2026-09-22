from __future__ import annotations

import os
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

COMPOSE: list[str] = ["docker", "compose", "-p", PROJECT]
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def run_cmd(
    cmd: list[str],
    dry_run: bool = False,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
    cwd: Path | None = None,
) -> None:
    display: str = " ".join(cmd) + (" <<'compose-yaml'" if stdin else "")
    if dry_run:
        print(f"[dry-run] {display}")
        return

    if run(cmd, env=env, input=stdin.encode() if stdin else None, check=False, cwd=cwd).returncode != 0:
        raise SystemExit(f"command failed: {display}")

    return


def compose_vars(worker: WorkerNode, jobmanager: str, flink_image: str, worker_image: str, data_port: int) -> dict[str, str]:
    return {
        "FLINK_IMAGE": flink_image,
        "WORKER_IMAGE": worker_image,
        "JOBMANAGER_HOST": jobmanager,
        "FLINK_REST_PORT": str(FLINK_REST_PORT),
        "JOBMANAGER_RPC_PORT": str(JOBMANAGER_RPC_PORT),
        "BLOB_SERVER_PORT": str(BLOB_SERVER_PORT),
        "SLOTS": str(worker.slots),
        "MEMORY": worker.memory,
        "FULLMAP_DIR": worker.fullmap,
        "FULLMAP_MOUNT": FULLMAP_MOUNT,
        "OUTPUTS_DIR": worker.outputs,
        "OUTPUTS_MOUNT": OUTPUTS_MOUNT,
        "TASKMANAGER_HOST": jobmanager,
        "TASKMANAGER_DATA_PORT": str(data_port),
        "TASKMANAGER_DATA_BIND_PORT": str(TASKMANAGER_DATA_PORT),
        "HOST_UID": str(os.getuid()),
        "HOST_GID": str(os.getgid()),
    }


def image_id(prefix: list[str], image: str, env: dict[str, str] | None = None) -> str:
    """content id of an image, empty when the tag is absent — tag equality is not enough

    the worker tag carries the package version, so a rebuilt wheel at the same version keeps the
    same tag and remotes would silently keep running the previous code
    """
    Result = run([*prefix, "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, check=False, env=env)
    return Result.stdout.decode().strip() if Result.returncode == 0 else ""


def tunnel_spec(port: int, host: str, user: str, jobmanager: str) -> list[str]:
    """remotes accept no inbound shuffle traffic — ride the ssh port instead"""
    return ["ssh", "-o", "BatchMode=yes", "-N", "-L", f"{jobmanager}:{port}:127.0.0.1:{TASKMANAGER_DATA_PORT}", f"{user}@{host}"]


def deploy_cluster(teardown: bool = False, dry_run: bool = False) -> None:
    Parser: YamlClusterParser = YamlClusterParser()
    ClusterSpec = Parser.parse_cluster()
    ssh_user: str = ClusterSpec.ssh_user
    jobmanager: str = Parser.jobmanager()
    flink_image: str = f"{FLINK_IMAGE_NAME}:{FLINK_VERSION}"
    worker_image: str = Parser.worker_image()
    env: dict[str, str] = {
        **os.environ,
        **compose_vars(Parser.local_worker(), jobmanager, flink_image, worker_image, TASKMANAGER_DATA_PORT),
    }
    remotes: list[WorkerNode] = Parser.remotes()
    tunnels: list[list[str]] = [
        tunnel_spec(TASKMANAGER_DATA_PORT + 1 + index, worker.host, ssh_user, jobmanager) for index, worker in enumerate(remotes)
    ]

    def remote(worker: WorkerNode, action: list[str], data_port: int) -> None:
        # explicit encoding: the compose templates carry non-ascii and ship to remote hosts whose
        # locale is not guaranteed utf-8
        rendered: str = Template(TASKMANAGER_COMPOSE.read_text(encoding="utf-8")).substitute(
            compose_vars(worker, jobmanager, flink_image, worker_image, data_port)
        )
        run_cmd(["ssh", "-o", "BatchMode=yes", f"{ssh_user}@{worker.host}", *COMPOSE, "-f", "-", *action], dry_run, stdin=rendered)

    if teardown:
        if not dry_run:
            for tunnel in tunnels:
                run(["pkill", "-f", " ".join(tunnel[1:])], check=False)
        for index, worker in enumerate(remotes):
            remote(worker, ["down"], TASKMANAGER_DATA_PORT + 1 + index)
        run_cmd([*COMPOSE, "-f", str(JOBMANAGER_COMPOSE), "down"], dry_run, env=env)
        return

    # fail fast on a missing fullmap bundle BEFORE any image build or compose run — a node
    # without its mount would otherwise fail later at sdkworker startup, mid-deploy
    LocalWorker = Parser.local_worker()
    if not dry_run and not Path(LocalWorker.fullmap).is_dir():
        raise SystemExit(f"fullmap directory not found on {LocalWorker.host}: {LocalWorker.fullmap}")
    if not dry_run:
        Path(LocalWorker.outputs).mkdir(parents=True, exist_ok=True)
    for worker in remotes:
        run_cmd(["ssh", "-o", "BatchMode=yes", f"{ssh_user}@{worker.host}", "test", "-d", worker.fullmap], dry_run)
        # the outputs dir is ours to create: docker would otherwise make it root-owned on first mount
        run_cmd(["ssh", "-o", "BatchMode=yes", f"{ssh_user}@{worker.host}", "mkdir", "-p", worker.outputs], dry_run)

    for cmd in (
        ["rm", "-rf", "dist"],
        ["uv", "build", "--wheel", "--out-dir", "dist"],
        ["docker", "build", "-f", str(WORKER_DOCKERFILE), "-t", worker_image, "."],
        ["docker", "build", "-f", str(FLINK_DOCKERFILE), "-t", flink_image, "."],
        [*COMPOSE, "-f", str(JOBMANAGER_COMPOSE), "up", "-d", "--force-recreate", "--remove-orphans"],
    ):
        run_cmd(cmd, dry_run, env=env, cwd=PROJECT_ROOT)

    local_ids: dict[str, str] = {image: image_id(["docker"], image, env) for image in (flink_image, worker_image)}
    for index, worker in enumerate(remotes):
        # no registry — pipe images down the ssh connection we already have, skipping only when the
        # remote already holds the exact same image id (a matching tag can still be stale code)
        for image in (flink_image, worker_image):
            local_id: str = local_ids[image]
            remote_id: str = image_id(["ssh", "-o", "BatchMode=yes", f"{ssh_user}@{worker.host}", "docker"], image)
            if dry_run or local_id != remote_id:
                run_cmd(["sh", "-c", f"docker save {image} | ssh -o BatchMode=yes {ssh_user}@{worker.host} docker load"], dry_run)
        remote(worker, ["up", "-d", "--force-recreate", "--remove-orphans"], TASKMANAGER_DATA_PORT + 1 + index)

    if not dry_run:
        for tunnel in tunnels:
            run(["pkill", "-f", " ".join(tunnel[1:])], check=False)
            Popen(tunnel, start_new_session=True, stdout=DEVNULL, stderr=DEVNULL)

    return
