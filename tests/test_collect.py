from __future__ import annotations

from pathlib import Path

import pytest

from relmedner import collect
from relmedner.collect import collect_outputs, local_shards, remote_clear_command, remote_has_shards, remote_spec
from relmedner.models import RunConfig, WorkerNode


def worker(host: str, outputs: str) -> WorkerNode:
    return WorkerNode(host=host, slots=1, memory="1g", fullmap="/data/fullmap", outputs=outputs)


def test_run_config_artifact_name_is_unique_and_keeps_the_avro_suffix() -> None:
    First: RunConfig = RunConfig(output="results/relmedner.avro", run_id="first")
    Second: RunConfig = RunConfig(output="results/relmedner.avro", run_id="second")

    assert First.artifact_name() == "relmedner-first.avro"
    assert Second.artifact_name() == "relmedner-second.avro"


def test_local_shards_match_only_this_run_artifact(tmp_path: Path) -> None:
    for name in ("relmedner-old.avro", "relmedner-current.avro", "relmedner-current-backup.avro"):
        (tmp_path / name).write_text(name)

    assert local_shards(str(tmp_path), "relmedner-current.avro") == (tmp_path / "relmedner-current.avro",)


def test_local_shards_return_this_run_unmerged_shards_but_never_a_wip_one(tmp_path: Path) -> None:
    (tmp_path / "relmedner-current.avro.part-aa").write_text("one")
    (tmp_path / "relmedner-current.avro.part-bb.wip").write_text("half")
    (tmp_path / "relmedner-current.avro.part-cc").write_text("two")

    assert local_shards(str(tmp_path), "relmedner-current.avro") == (
        tmp_path / "relmedner-current.avro.part-aa",
        tmp_path / "relmedner-current.avro.part-cc",
    )


def test_remote_spec_is_the_remote_shard_glob() -> None:
    assert remote_spec(worker("10.2.9.19", "/data/outputs"), "sgoetz", "relmedner-current.avro") == (
        "sgoetz@10.2.9.19:/data/outputs/relmedner-current.avro.part-*"
    )


def test_remote_clear_command_removes_the_shard_glob_over_ssh() -> None:
    assert remote_clear_command(worker("10.2.9.19", "/data/outputs"), "sgoetz", "relmedner-current.avro") == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "sgoetz@10.2.9.19",
        "rm",
        "-f",
        "/data/outputs/relmedner-current.avro.part-*",
    ]


def test_collect_outputs_copies_this_run_local_artifact_and_scps_remote_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect, "remote_has_shards", lambda *args: True)
    mount: Path = tmp_path / "mount"
    mount.mkdir()
    (mount / "relmedner-old.avro").write_text("stale")
    (mount / "relmedner-current.avro").write_text("fresh")
    destination: Path = tmp_path / "results"
    commands: list[list[str]] = []

    result: Path = collect_outputs(
        (worker("local", str(mount)), worker("10.2.9.11", "/data/outputs")),
        "sgoetz",
        "relmedner-current.avro",
        str(destination / "relmedner.avro"),
        "local",
        commands.append,
        log=lambda _: None,
    )

    assert result == destination.resolve()
    assert (destination / "relmedner.avro").read_text() == "fresh"
    assert commands == [
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "sgoetz@10.2.9.11:/data/outputs/relmedner-current.avro.part-*",
            str(destination.resolve()),
        ],
        remote_clear_command(worker("10.2.9.11", "/data/outputs"), "sgoetz", "relmedner-current.avro"),
    ]


def test_collect_outputs_skips_a_remote_node_without_this_run_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect, "remote_has_shards", lambda *args: False)
    (tmp_path / "relmedner-current.avro").write_text("fresh")
    destination: Path = tmp_path / "results"
    commands: list[list[str]] = []

    collect_outputs(
        (worker("local", str(tmp_path)), worker("10.2.9.11", "/data/outputs")),
        "sgoetz",
        "relmedner-current.avro",
        str(destination / "relmedner.avro"),
        "local",
        commands.append,
        log=lambda _: None,
    )

    assert (destination / "relmedner.avro").read_text() == "fresh"
    assert commands == []


def test_remote_has_shards_requires_the_exact_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCompleted:
        def __init__(self, returncode: int) -> None:
            self.returncode: int = returncode

    monkeypatch.setattr(collect, "run", lambda *args, **kwargs: FakeCompleted(0))
    assert remote_has_shards(worker("10.2.9.11", "/data/outputs"), "sgoetz", "relmedner-current.avro") is True

    for missing in (1, 2):  # test -f miss (1) and ls glob miss (2)
        monkeypatch.setattr(collect, "run", lambda *a, missing=missing, **k: FakeCompleted(missing))
        assert remote_has_shards(worker("10.2.9.11", "/data/outputs"), "sgoetz", "relmedner-current.avro") is False

    monkeypatch.setattr(collect, "run", lambda *args, **kwargs: FakeCompleted(255))
    with pytest.raises(SystemExit, match="could not inspect remote output mount"):
        remote_has_shards(worker("10.2.9.11", "/data/outputs"), "sgoetz", "relmedner-current.avro")


def test_collect_outputs_fails_when_no_node_has_this_run_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect, "remote_has_shards", lambda *args: False)

    with pytest.raises(SystemExit, match="job finished without producing"):
        collect_outputs(
            (worker("local", str(tmp_path)), worker("10.2.9.11", "/data/outputs")),
            "sgoetz",
            "relmedner-current.avro",
            str(tmp_path / "results" / "relmedner.avro"),
            "local",
            lambda _: None,
            log=lambda _: None,
        )


def test_collect_outputs_does_not_copy_onto_itself(tmp_path: Path) -> None:
    shard: Path = tmp_path / "relmedner-current.avro"
    shard.write_text("payload")

    collect_outputs(
        (worker("local", str(tmp_path)),),
        "sgoetz",
        shard.name,
        str(shard),
        "local",
        lambda _: None,
        log=lambda _: None,
    )

    assert shard.read_text() == "payload"


class StubExample:
    """stands in for TrainingExample so the merge writes a tiny known schema"""

    @staticmethod
    def avro_schema_to_python() -> dict:
        return {"name": "r", "type": "record", "fields": [{"name": "a", "type": "int"}]}


def test_collect_outputs_merges_local_shards_and_cleans_the_mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastavro import reader, writer

    monkeypatch.setattr(collect, "TrainingExample", StubExample)
    monkeypatch.setattr(collect, "remote_has_shards", lambda *args: False)
    mount: Path = tmp_path / "mount"
    mount.mkdir()
    schema: dict = StubExample.avro_schema_to_python()
    with (mount / "relmedner-current.avro.part-aa").open("wb") as fo:
        writer(fo, schema, [{"a": 1}])
    with (mount / "relmedner-current.avro.part-cc").open("wb") as fo:
        writer(fo, schema, [{"a": 2}])
    destination: Path = tmp_path / "results"

    collect_outputs(
        (worker("local", str(mount)),),
        "sgoetz",
        "relmedner-current.avro",
        str(destination / "relmedner.avro"),
        "local",
        lambda _: None,
        log=lambda _: None,
    )

    with (destination / "relmedner.avro").open("rb") as fo:
        assert sorted(row["a"] for row in reader(fo)) == [1, 2]
    # no stale shards survive in either the mount or the destination
    assert not list(mount.iterdir())
    assert not list(destination.glob("*.part-*"))
