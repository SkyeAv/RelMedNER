from __future__ import annotations

from pathlib import Path

import pytest
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


def test_merge_copies_blocks_and_keeps_every_record_across_many_blocks(tmp_path: Path) -> None:
    """merge_shards splices encoded blocks instead of re-encoding records; a shard large enough to
    span several avro blocks must still come back record-for-record, so a block-boundary bug
    (dropped tail block, duplicated pending block) cannot hide behind a tiny fixture"""
    Target: Path = tmp_path / "relmedner.avro"
    Writer: ShardWriter = ShardWriter(str(Target), SCHEMA)
    write_bundle(Writer, [{"a": index} for index in range(50_000)])
    write_bundle(Writer, [{"a": -1}])

    merge_shards(Target, SCHEMA)
    with Target.open("rb") as fo:
        assert sorted(row["a"] for row in reader(fo)) == sorted([*range(50_000), -1])


def test_merge_refuses_a_shard_written_under_another_schema(tmp_path: Path) -> None:
    """block copying is only sound when every shard header matches the merge schema and codec;
    a foreign shard must fail loudly rather than splice bytes the merged header mis-describes"""
    from fastavro import writer as avro_writer

    Target: Path = tmp_path / "relmedner.avro"
    other: dict = {"name": "r", "type": "record", "fields": [{"name": "b", "type": "string"}]}
    with (tmp_path / "relmedner.avro.part-foreign").open("wb") as fo:
        avro_writer(fo, other, [{"b": "x"}])

    with pytest.raises(ValueError, match="does not match the merge schema"):
        merge_shards(Target, SCHEMA)


def test_merge_refuses_a_shard_written_with_another_codec(tmp_path: Path) -> None:
    """same schema, different codec: deflate blocks copied under a null-codec header would be
    unreadable, so the codec is part of the header check"""
    from fastavro import writer as avro_writer

    Target: Path = tmp_path / "relmedner.avro"
    with (tmp_path / "relmedner.avro.part-deflate").open("wb") as fo:
        avro_writer(fo, SCHEMA, [{"a": 1}], codec="deflate")

    with pytest.raises(ValueError, match="does not match the merge schema"):
        merge_shards(Target, SCHEMA)
