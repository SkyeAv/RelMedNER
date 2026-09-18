from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from shutil import copy2
from subprocess import run

from relmedner.models import WorkerNode

Runner = Callable[[list[str]], None]


def local_shards(outputs: str, artifact: str) -> tuple[Path, ...]:
    """the exact file produced by this run, never a same-stem shard from an earlier run"""
    candidate: Path = Path(outputs) / artifact
    return (candidate,) if candidate.is_file() else ()


def remote_spec(worker: WorkerNode, ssh_user: str, artifact: str) -> str:
    return f"{ssh_user}@{worker.host}:{Path(worker.outputs) / artifact}"


def remote_has_shards(worker: WorkerNode, ssh_user: str, artifact: str) -> bool:
    """only the node whose sdkworker ran the write sink produces this run's artifact"""
    Result = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            f"{ssh_user}@{worker.host}",
            "test",
            "-f",
            str(Path(worker.outputs) / artifact),
        ],
        capture_output=True,
        check=False,
    )
    if Result.returncode == 0:
        return True
    if Result.returncode == 1:
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
    """gather this run's artifact into the exact path requested by --output"""
    Destination: Path = Path(output).expanduser().resolve().parent
    target: Path = Destination / Path(output).name
    Destination.mkdir(parents=True, exist_ok=True)

    for worker in workers:
        if worker.host == local_host:
            for shard in local_shards(worker.outputs, artifact):
                if shard.resolve() != target:
                    copy2(shard, target)
            continue
        if remote_has_shards(worker, ssh_user, artifact):
            command(["scp", "-o", "BatchMode=yes", remote_spec(worker, ssh_user, artifact), str(target)])

    if not target.is_file():
        raise SystemExit(f"job finished without producing {target}")

    log(f"collected avro shards into {Destination}")
    return Destination
