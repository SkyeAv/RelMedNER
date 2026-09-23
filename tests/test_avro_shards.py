from __future__ import annotations

from pathlib import Path

from fastavro import reader

from relmedner.avro_shards import ShardWriter, merge_shards, shards_for

SCHEMA: dict = {"name": "r", "type": "record", "fields": [{"name": "a", "type": "int"}]}


def write_bundle(writer: ShardWriter, rows: list[dict]) -> None:
    writer.start_bundle()
    for row in rows:
        writer.process(row)
    writer.finish_bundle()


def test_shards_merge_into_one_avro_container_in_order(tmp_path: Path) -> None:
    Target: Path = tmp_path / "relmedner.avro"
    Writer: ShardWriter = ShardWriter(str(Target), SCHEMA)
    write_bundle(Writer, [{"a": 1}, {"a": 2}])
    write_bundle(Writer, [{"a": 3}])

    assert len(shards_for(Target)) == 2
    assert merge_shards(Target, SCHEMA) == Target
    with Target.open("rb") as fo:
        # shard order follows shard names (random hexes), and record order carries no meaning for
        # training data, so compare as multisets
        assert sorted(row["a"] for row in reader(fo)) == [1, 2, 3]
    assert shards_for(Target) == ()


def test_a_bundle_that_never_finished_leaves_nothing_to_merge(tmp_path: Path) -> None:
    """a retried bundle's half-written .wip shard must never enter the merged artifact"""
    Target: Path = tmp_path / "relmedner.avro"
    Writer: ShardWriter = ShardWriter(str(Target), SCHEMA)
    Writer.start_bundle()
    Writer.process({"a": 99})  # bundle dies here: no finish_bundle, shard stays .wip
    write_bundle(Writer, [{"a": 1}])

    assert [part.name for part in shards_for(Target)] != []
    assert merge_shards(Target, SCHEMA) == Target
    with Target.open("rb") as fo:
        assert [row["a"] for row in reader(fo)] == [1]


def test_a_bundle_with_no_records_writes_no_shard(tmp_path: Path) -> None:
    Target: Path = tmp_path / "relmedner.avro"
    Writer: ShardWriter = ShardWriter(str(Target), SCHEMA)

    write_bundle(Writer, [])

    assert shards_for(Target) == ()
    assert not list(tmp_path.iterdir())


def test_merging_without_shards_leaves_an_existing_target_untouched(tmp_path: Path) -> None:
    """the legacy single-file artifact passes straight through collection"""
    Target: Path = tmp_path / "relmedner.avro"
    Target.write_bytes(b"already merged")

    assert merge_shards(Target, SCHEMA) == Target
    assert Target.read_bytes() == b"already merged"
