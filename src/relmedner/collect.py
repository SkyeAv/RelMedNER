from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from shutil import copy2
from subprocess import run
from typing import Any

from relmedner.avro_shards import merge_shards, shards_for
from relmedner.models import TrainingExample, WorkerNode

Runner = Callable[[list[str]], None]


def local_shards(outputs: str, artifact: str) -> tuple[Path, ...]:
    """the exact file produced by this run, or its unmerged shards, never a same-stem shard from an
    earlier run"""
    candidate: Path = Path(outputs) / artifact
    if candidate.is_file():
        return (candidate,)
    return shards_for(candidate)


def remote_spec(worker: WorkerNode, ssh_user: str, artifact: str) -> str:
    """remote shard glob: scp's remote shell expands it, shipping every shard in one connection"""
    return f"{ssh_user}@{worker.host}:{Path(worker.outputs) / artifact}.part-*"


def remote_clear_command(worker: WorkerNode, ssh_user: str, artifact: str) -> list[str]:
    """removes a worker's shards once they are safely collected; the glob expands in the remote shell"""
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        f"{ssh_user}@{worker.host}",
        "rm",
        "-f",
        f"{Path(worker.outputs) / artifact}.part-*",
    ]


def remote_has_shards(worker: WorkerNode, ssh_user: str, artifact: str) -> bool:
    """only the node whose sdkworker ran the sink holds this run's artifact, as either the merged
    file or its unmerged shards"""
    Target: str = str(Path(worker.outputs) / artifact)
    Result = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            f"{ssh_user}@{worker.host}",
            "test",
            "-f",
            Target,
            "||",
            "ls",
            f"{Target}.part-*",
        ],
        capture_output=True,
        check=False,
    )
    if Result.returncode == 0:
        return True
    if Result.returncode in (1, 2):
        return False
    raise SystemExit(f"could not inspect remote output mount on {worker.host}")


def collect_outputs(
    workers: tuple[WorkerNode, ...],
    ssh_user: str,
    artifact: str,
    output: str,
    local_host: str,
    command: Runner,
    log: Callable[[str], None] = print,
) -> Path:
    """gather this run's artifact into the exact path requested by --output; shards arriving from any
    node are merged into the final file, and shard mounts are cleaned so a later run never merges
    stale ones"""
    Destination: Path = Path(output).expanduser().resolve().parent
    target: Path = Destination / Path(output).name
    Destination.mkdir(parents=True, exist_ok=True)
    Schema: dict[str, Any] = TrainingExample.avro_schema_to_python()

    for worker in workers:
        if worker.host == local_host:
            for shard in local_shards(worker.outputs, artifact):
                if shard.resolve() == target:
                    continue
                # an already-merged artifact lands under the requested name; unmerged shards keep
                # theirs so merge_shards finds them beside the target
                copy2(shard, target if shard.name == artifact else Destination / shard.name)
                if shard.name != artifact:
                    shard.unlink()
            continue
        if remote_has_shards(worker, ssh_user, artifact):
            command(["scp", "-o", "BatchMode=yes", remote_spec(worker, ssh_user, artifact), str(Destination)])
            command(remote_clear_command(worker, ssh_user, artifact))

    if not target.is_file():
        merge_shards(target, Schema, artifact)

    if not target.is_file():
        raise SystemExit(f"job finished without producing {target}")

    log(f"collected avro shards into {Destination}")
    return Destination
