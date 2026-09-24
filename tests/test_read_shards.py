from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from fastavro import block_reader, parse_schema, reader, writer

from relmedner.enums import DedupMode
from relmedner.hf_json import HuggingFaceJsonDataStream
from relmedner.hf_parquet import HuggingFaceParquetDataStream
from relmedner.huggingface import HuggingFaceDataStream
from relmedner.local import LocalAvroDataStream
from relmedner.models import LocalAvroDataset, RowFilters, RunConfig, ScriptTask, YamlIngests
from relmedner.pipeline import BeamPipeline
from relmedner.registry import build_stream
from relmedner.streams import ZeroYieldError

# ---------------------------------------------------------------------- fixtures --

SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "Row",
    "namespace": "relmedner.tests",
    "fields": [
        {"name": "nct_id", "type": "string"},
        {"name": "intervention_type", "type": "string"},
        {"name": "name", "type": "string"},
        {"name": "description", "type": ["null", "string"], "default": None},
        {"name": "matches", "type": {"type": "array", "items": "string"}, "default": []},
        {"name": "other_names", "type": {"type": "array", "items": "string"}, "default": []},
        {"name": "synonym_curies", "type": {"type": "array", "items": "string"}, "default": []},
        {"name": "unmapped", "type": "boolean", "default": True},
    ],
}

TASK: tuple[Any, ...] = ("script", "CtkpInterventionsScript", ("entities",))


def fixture_records(count: int) -> list[dict[str, Any]]:
    return [
        {
            "nct_id": f"NCT{i:06d}",
            "intervention_type": "DRUG",
            "name": f"ketoconazole {i}% shampoo",
            "description": f"Subjects will wash their hair twice weekly with ketoconazole {i}% shampoo.",
            "matches": [],
            "other_names": [],
            "synonym_curies": [],
            "unmapped": True,
        }
        for i in range(count)
    ]


def write_multiblock_avro(path: Path, records: list[dict[str, Any]], blocks: int) -> Path:
    """one fastavro writer call appends one block per invocation on the same open handle, so
    the fixture carries a known multi-block container (whole blocks are the shard unit of
    the local avro reader)"""
    schema = parse_schema(SCHEMA)
    with path.open("a+b") as handle:
        per_block: int = -(-len(records) // blocks)
        for start in range(0, len(records), per_block):
            writer(handle, schema, records[start : start + per_block])
    return path


def dataset(path: Path, read_shards: int = 1) -> LocalAvroDataset:
    return LocalAvroDataset(
        task=ScriptTask(type="script", name="CtkpInterventionsScript", outputs=["entities"]),
        source="local",
        path=str(path),
        weight=1.0,
        read_shards=read_shards,
    )


# ------------------------------------------------------------ shard invariants --


@pytest.mark.parametrize("shards", [2, 3, 4, 7])
def test_shard_union_equals_the_unsharded_row_list(tmp_path: Path, shards: int) -> None:
    """the transparency contract: sharding only decides WHO reads a row, never WHETHER it is
    read -- the multiset union of all shards' rows must equal the unsharded row list exactly
    (no row lost, no row duplicated), for every shard count"""
    target: Path = write_multiblock_avro(tmp_path / "fixture.avro", fixture_records(25), blocks=3)
    assert len(list(block_reader(target.open("rb")))) == 3  # the shard unit really exists

    unsharded: list[Any] = list(LocalAvroDataStream(TASK, 1.0, str(target)).rows())

    union: Counter[Any] = Counter()
    for shard_index in range(shards):
        stream = LocalAvroDataStream(TASK, 1.0, str(target), read_shards=shards, shard_index=shard_index)
        union.update(repr(row) for row in stream.rows())
    assert union == Counter(repr(row) for row in unsharded)


def test_every_shard_of_a_whole_block_passes_its_records_whole(tmp_path: Path) -> None:
    """block-level selection never splits a record: each shard yields whole records only,
    which the repr-multiset equality above pins at the row level -- this test pins that a
    single shard's yield is a deterministic function of its (read_shards, shard_index) pair,
    so a rerun or a resubmitted job reads the identical shard"""
    target: Path = write_multiblock_avro(tmp_path / "fixture.avro", fixture_records(25), blocks=3)
    first: list[Any] = list(LocalAvroDataStream(TASK, 1.0, str(target), read_shards=2, shard_index=1).rows())
    again: list[Any] = list(LocalAvroDataStream(TASK, 1.0, str(target), read_shards=2, shard_index=1).rows())
    assert [repr(row) for row in first] == [repr(row) for row in again]


def test_stream_args_expands_one_envelope_per_shard(tmp_path: Path) -> None:
    """a source declaring read_shards contributes one Create element per shard index; a
    source at the default contributes exactly one, so the graph stays byte-identical when
    nobody fans out"""
    sharded: LocalAvroDataset = dataset(tmp_path / "a.avro", read_shards=3)
    plain: LocalAvroDataset = dataset(tmp_path / "b.avro")
    assert sharded.to_stream_args() == ("local", sharded.to_tuple()[1], None, 1.0, 3)
    envelopes: tuple[tuple[str, tuple[Any, ...], RowFilters | None, float, int, int], ...] = YamlIngests(datasets=[sharded, plain]).stream_args()
    assert [(source, read_shards, shard_index) for source, _payload, _f, _s, read_shards, shard_index in envelopes] == [
        ("local", 3, 0),
        ("local", 3, 1),
        ("local", 3, 2),
        ("local", 1, 0),
    ]


def test_build_stream_carries_the_shard_envelope(tmp_path: Path) -> None:
    Payload: tuple[Any, ...] = (TASK, 1.0, str(tmp_path / "a.avro"))
    stream = build_stream("local", Payload, read_shards=4, shard_index=3)
    assert isinstance(stream, LocalAvroDataStream)
    assert stream.read_shards == 4 and stream.shard_index == 3
    # the defaults keep every existing positional call valid
    untouched = build_stream("local", Payload)
    assert untouched.read_shards == 1 and untouched.shard_index == 0


def test_shard_envelope_bounds_are_rejected(tmp_path: Path) -> None:
    """fail loud at construction, never mid-run: a shard_index outside the fan-out or a
    zero shard count would silently duplicate or lose rows"""
    with pytest.raises(ValueError):
        LocalAvroDataStream(TASK, 1.0, str(tmp_path / "a.avro"), read_shards=2, shard_index=2)
    with pytest.raises(ValueError):
        LocalAvroDataStream(TASK, 1.0, str(tmp_path / "a.avro"), read_shards=0)


def test_sources_that_cannot_shard_reject_read_shards(tmp_path: Path) -> None:
    """an unshardable source declared with read_shards > 1 would either duplicate every row
    (every shard streaming the whole source) or silently stay serial, so validation raises
    at build time instead"""
    Payload: tuple[Any, ...] = (TASK, 1.0, str(tmp_path / "a.tsv"), ("text",))
    with pytest.raises(ValueError, match="does not shard"):
        build_stream("local_delimited", Payload, read_shards=2)
    # the default stays constructible everywhere
    build_stream("local_delimited", Payload)


# --------------------------------------------------------- stats and the guard --


def test_quality_lines_carry_the_shard_label(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """K shard passes produce K attributable quality lines, tagged with the shard identity"""
    target: Path = write_multiblock_avro(tmp_path / "fixture.avro", fixture_records(6), blocks=2)
    config: RunConfig = RunConfig(sample_limit=None, output=str(tmp_path / "out.avro"), dedup_mode=DedupMode.OFF)
    with caplog.at_level(logging.INFO, logger="relmedner.quality"):
        list(LocalAvroDataStream(TASK, 1.0, str(target), read_shards=2, shard_index=1).stream(config))
        list(LocalAvroDataStream(TASK, 1.0, str(target)).stream(config))
    lines: list[str] = [record.getMessage() for record in caplog.records]
    assert any("[shard 1/2]" in line for line in lines)
    assert any("[shard" not in line for line in lines)


def test_sharded_zero_yield_warns_instead_of_raising(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """one shard of a fanned-out read may legitimately hold zero passing rows (a trailing
    file shard under the declared filter), so the US-008 raise degrades to a WARNING there;
    row-count truth for sharded sources stays with the probe receipts"""
    target: Path = write_multiblock_avro(tmp_path / "fixture.avro", fixture_records(6), blocks=2)
    stream = LocalAvroDataStream(TASK, 1.0, str(target), filters=RowFilters(max_text_len=1), read_shards=2, shard_index=0)
    config: RunConfig = RunConfig(sample_limit=None, output=str(tmp_path / "out.avro"), dedup_mode=DedupMode.OFF)
    with caplog.at_level(logging.WARNING, logger="relmedner.quality"):
        assert list(stream.stream(config)) == []
    assert any("candidate rows, 0 passed" in record.getMessage() for record in caplog.records)


def test_unsharded_zero_yield_still_raises(tmp_path: Path) -> None:
    """the silent-empty-training-set guard keeps its raise on the ordinary single-pass read"""
    target: Path = write_multiblock_avro(tmp_path / "fixture.avro", fixture_records(6), blocks=2)
    stream = LocalAvroDataStream(TASK, 1.0, str(target), filters=RowFilters(max_text_len=1))
    with pytest.raises(ZeroYieldError):
        list(stream.rows())


# ------------------------------------------------------------------ end to end --


def test_directrunner_sharded_matches_unsharded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """the pipeline-level contract: a local-avro source declared with read_shards=4 rides the
    real graph (Create expansion -> build_stream -> stream -> dispatch -> shard writer) and
    produces the same training records as the same source at read_shards=1"""
    target: Path = write_multiblock_avro(tmp_path / "fixture.avro", fixture_records(24), blocks=3)

    def run(shards: int, output: Path) -> list[dict[str, Any]]:
        class StubParser:
            def parse_ingests(self: Any) -> YamlIngests:
                return YamlIngests(datasets=[dataset(target, read_shards=shards)])

        monkeypatch.setattr("relmedner.pipeline.YamlIngestsParser", StubParser)
        config: RunConfig = RunConfig(sample_limit=None, output=str(output), dedup_mode=DedupMode.OFF)
        BeamPipeline().run(config)
        return list(reader(output.open("rb")))

    plain: list[dict[str, Any]] = run(1, tmp_path / "plain.avro")
    sharded: list[dict[str, Any]] = run(4, tmp_path / "sharded.avro")

    assert len(plain) > 0, "the fixture must dispatch into real training records"

    def digest(records: list[dict[str, Any]]) -> Counter[str]:
        return Counter(json.dumps(record, sort_keys=True) for record in records)

    assert digest(sharded) == digest(plain)


# ------------------------------------------------------------------ hub sources --


def streaming_fixture(tmp_path: Path, shards: int, rows_per_shard: int) -> Any:
    """a real streaming IterableDataset over `shards` local parquet files: the same file-shard
    structure a hub IterableDataset reports through num_shards, loaded through the same
    packaged parquet builder the production streaming path uses"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    files: list[str] = []
    for shard in range(shards):
        table = pa.table({"text": [f"row {shard * rows_per_shard + i}" for i in range(rows_per_shard)]})
        path = tmp_path / f"shard-{shard}.parquet"
        pq.write_table(table, path)
        files.append(str(path))
    import datasets

    return datasets.load_dataset("parquet", data_files=files, split="train", streaming=True)


def hf_stream(source: str, monkeypatch: pytest.MonkeyPatch, fixture: Any, read_shards: int, shard_index: int, **kwargs: Any) -> Any:
    module = {"hf": "relmedner.huggingface", "hf_json": "relmedner.hf_json", "hf_parquet": "relmedner.hf_parquet"}[source]
    monkeypatch.setattr(f"{module}.load_dataset", lambda *args, **kwargs: fixture)
    payload: tuple[Any, ...] = (TASK, 1.0, "org/dataset", None, None, None, ("text",))
    stream_class: Any = {"hf": HuggingFaceDataStream, "hf_json": HuggingFaceJsonDataStream, "hf_parquet": HuggingFaceParquetDataStream}[source]
    return stream_class(*payload, read_shards=read_shards, shard_index=shard_index, **kwargs)


@pytest.mark.parametrize("source", ["hf", "hf_json", "hf_parquet"])
def test_hub_shard_union_equals_the_unsharded_row_list(source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """the hub split is transparent too: streaming datasets split at file-shard granularity
    (no row read twice), built datasets split by row index, and either way the union of all
    shards' rows equals the unsharded row list"""
    if source == "hf":
        fixture: Any = streaming_fixture(tmp_path, shards=4, rows_per_shard=6)
    else:
        from datasets import Dataset

        fixture = Dataset.from_dict({"text": [f"row {i}" for i in range(24)]})
    unsharded: list[Any] = [repr(row) for row in hf_stream(source, monkeypatch, fixture, 1, 0).rows()]
    assert len(unsharded) == 24

    union: Counter[str] = Counter()
    for shard_index in range(2):
        union.update(repr(row) for row in hf_stream(source, monkeypatch, fixture, 2, shard_index).rows())
    assert union == Counter(unsharded)


def test_hub_overshard_clamps_to_the_dataset_shard_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """a declared read_shards above what the dataset can split must neither raise nor hand
    whole shards to index 0 (the hub layer's overshard failure modes): every instance clamps
    to the same effective count, extra indices yield nothing, and the union still covers
    every row exactly once"""
    fixture: Any = streaming_fixture(tmp_path, shards=4, rows_per_shard=6)
    union: Counter[str] = Counter()
    for shard_index in range(8):
        rows: list[Any] = list(hf_stream("hf", monkeypatch, fixture, 8, shard_index).rows())
        if shard_index < 4:
            assert rows, "an effective shard must carry its file shards' rows"
        else:
            assert rows == [], "an index beyond the clamp yields nothing without error"
        union.update(repr(row) for row in rows)
    assert len(union) == 24


def test_hub_sharded_zero_yield_warns_instead_of_raising(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """the hf zero-yield guard degrades to a per-shard WARNING on a sharded read and keeps
    its raise on the ordinary single-pass read"""
    from datasets import Dataset

    fixture = Dataset.from_dict({"text": [f"row {i}" for i in range(10)]})
    sharded = hf_stream("hf_json", monkeypatch, fixture, 2, 1, filters=RowFilters(max_text_len=1))
    with caplog.at_level(logging.WARNING, logger="relmedner.quality"):
        assert list(sharded.rows()) == []
    assert any("candidate rows, 0 passed" in record.getMessage() for record in caplog.records)

    whole = hf_stream("hf_json", monkeypatch, fixture, 1, 0, filters=RowFilters(max_text_len=1))
    with pytest.raises(ZeroYieldError):
        list(whole.rows())
