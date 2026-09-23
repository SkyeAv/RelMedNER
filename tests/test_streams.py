from __future__ import annotations

import logging
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from typing import Any, ClassVar, Self

import pytest
from fastavro import parse_schema, writer

from relmedner import hf_json, hf_parquet
from relmedner.hf_json import HuggingFaceJsonDataStream
from relmedner.hf_parquet import HuggingFaceParquetDataStream
from relmedner.huggingface import HuggingFaceDataStream
from relmedner.ingests import YamlIngestsParser
from relmedner.local import LocalAvroDataStream, LocalDelimitedDataStream
from relmedner.models import RowFilters, RunConfig, YamlIngests
from relmedner.registry import SOURCE_REGISTRY, build_stream
from relmedner.streams import DataStream, StreamedRow, StreamStats, ZeroYieldError


class CountingDataStream(DataStream):
    SOURCE: ClassVar[str] = "counting"

    def rows(self: Self) -> Iterator[StreamedRow]:
        for index in count():
            yield ("GlinerBiomedScript", (("entities",), (str(index),)))


class FiniteDataStream(DataStream):
    SOURCE: ClassVar[str] = "finite"

    def rows(self: Self) -> Iterator[StreamedRow]:
        for index in range(20):
            yield ("GlinerBiomedScript", (("entities",), (str(index),)))


def test_stream_truncates_an_unbounded_source() -> None:
    Streamed: list[StreamedRow] = list(CountingDataStream().stream(RunConfig(sample_limit=5)))
    assert Streamed == [("GlinerBiomedScript", (("entities",), (str(index),))) for index in range(5)]


def test_stream_without_a_limit_yields_every_row() -> None:
    Streamed: list[StreamedRow] = list(FiniteDataStream().stream(RunConfig()))
    assert Streamed == [("GlinerBiomedScript", (("entities",), (str(index),))) for index in range(20)]


def test_stream_limit_is_not_shared_between_calls() -> None:
    Stream: CountingDataStream = CountingDataStream()
    assert len(list(Stream.stream(RunConfig(sample_limit=5)))) == 5
    assert len(list(Stream.stream(RunConfig(sample_limit=5)))) == 5


def test_build_stream_constructs_every_declared_ingest() -> None:
    """every declared tuple must unpack into its stream ctor positionally, and the key the stream
    stamps its rows with must equal DatasetBase.row_key, because the pipeline looks the source's
    mixing weight up by that string (drift is a KeyError partway through a run, not a wrong number).

    Building ALL of them is what catches a source whose ctor was never updated for a new DatasetBase
    field: the local stream took (task, path) while the declared payload already carried
    (task, weight, path), so the CTKP ingest raised TypeError at build_stream while every hf-only
    test stayed green. Construction is offline-safe, since no source touches the network or the disk
    until rows() is called.
    """
    Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
    Weights: dict[str, float] = Ingests.weights_by_source()

    Built: list[DataStream] = [build_stream(*dataset.to_tuple()) for dataset in Ingests.datasets]

    assert len(Built) == len(Ingests.datasets)
    # every source kind stays covered, so this invariant cannot silently degrade to hf-only again
    assert {type(stream) for stream in Built} == {
        HuggingFaceDataStream,
        HuggingFaceJsonDataStream,
        HuggingFaceParquetDataStream,
        LocalAvroDataStream,
        LocalDelimitedDataStream,
    }
    for dataset, stream in zip(Ingests.datasets, Built, strict=True):
        assert stream.name == dataset.row_key, f"{type(dataset).__name__} row key drift"
        assert stream.weight == dataset.weight
        assert Weights[stream.name] == dataset.weight

    First = Built[0]
    assert isinstance(First, HuggingFaceDataStream)
    assert First.dataset == "anthonyyazdaniml/gliner-biomed-pre-training"
    assert First.split == "train"
    assert First.columns_out == ("tokenized_text", "ner")
    assert First.weight == 1.0


def test_build_stream_carries_the_declared_weight() -> None:
    """the weight rides the frozen payload tuple (DatasetBase field order) positionally into
    the stream ctor, so a nondefault declaration must land on the stream untouched"""
    Stream: DataStream = build_stream(
        "hf",
        (
            ("script", "GlinerBiomedScript", ("entities",)),
            0.25,
            "some/dataset",
            None,
            "train",
            None,
            ("text",),
        ),
    )

    assert isinstance(Stream, HuggingFaceDataStream)
    assert Stream.weight == 0.25


def test_rebuild_task_round_trips_every_declared_task_type() -> None:
    from relmedner.models import FullmapTask, ScriptTask
    from relmedner.streams import rebuild_task

    for Payload in YamlIngestsParser().generate_tuples():
        Task = rebuild_task(Payload[1][0])
        assert isinstance(Task, ScriptTask | FullmapTask)
        assert Task.to_tuple() == Payload[1][0]

    with pytest.raises(ValueError):
        rebuild_task(("teleport",))


def test_registry_keys_on_the_source_discriminator() -> None:
    assert SOURCE_REGISTRY["hf"] is HuggingFaceDataStream
    assert all(Source == Stream.SOURCE for Source, Stream in SOURCE_REGISTRY.items())


def test_apply_match_keeps_only_declared_values() -> None:
    Stream: HuggingFaceDataStream = HuggingFaceDataStream(
        task=("script", "GlinerBiomedScript", ("entities",)),
        weight=1.0,
        dataset="anthonyyazdaniml/gliner-biomed-pre-training",
        subset=None,
        split="train",
        match_on=(("domain", ("Healthcare",)),),
        columns_out=("tokenized_text",),
    )
    Kept: dict[str, Any] = {"domain": "Healthcare", "tokenized_text": ["a"]}
    Dropped: dict[str, Any] = {"domain": "Finance", "tokenized_text": ["b"]}

    assert Stream.apply_match(Kept) is True
    assert Stream.apply_match(Dropped) is False


def test_build_stream_constructs_the_hf_json_source_positionally() -> None:
    """build_stream splats the payload into __init__ by position, so the payload to_tuple produced must
    land in HuggingFaceJsonDataStream.__init__ in model field order minus source"""
    Source, Payload = (
        "hf_json",
        (
            ("script", "PubmedAbstractsScript", ("entities",)),
            1.0,
            "knowledgator/PubMedAbstractsNER",
            "train.json",
            "train",
            None,
            ("tokenized_text", "ner"),
        ),
    )

    Stream: DataStream = build_stream(Source, Payload)

    assert isinstance(Stream, HuggingFaceJsonDataStream)
    assert Stream.name == "knowledgator/PubMedAbstractsNER"
    assert Stream.task == ("script", "PubmedAbstractsScript", ("entities",))
    assert Stream.weight == 1.0
    assert Stream.file == "train.json"
    assert Stream.columns_out == ("tokenized_text", "ner")


def test_build_stream_constructs_the_hf_parquet_source_positionally() -> None:
    """build_stream splats the payload into __init__ by position, so the payload to_tuple produced must
    land in HuggingFaceParquetDataStream.__init__ in model field order minus source"""
    Source, Payload = (
        "hf_parquet",
        (
            ("script", "ChiaScript", ("entities", "relations")),
            1.0,
            "bigbio/chia",
            "chia_bigbio_kb/train/0000.parquet",
            "refs/convert/parquet",
            "train",
            None,
            ("passages", "entities", "relations"),
        ),
    )

    Stream: DataStream = build_stream(Source, Payload)

    assert isinstance(Stream, HuggingFaceParquetDataStream)
    assert Stream.name == "bigbio/chia"
    assert Stream.task == ("script", "ChiaScript", ("entities", "relations"))
    assert Stream.weight == 1.0
    assert Stream.file == "chia_bigbio_kb/train/0000.parquet"
    assert Stream.revision == "refs/convert/parquet"
    assert Stream.columns_out == ("passages", "entities", "relations")


def test_registry_keys_the_hf_parquet_source() -> None:
    assert SOURCE_REGISTRY["hf_parquet"] is HuggingFaceParquetDataStream


def test_hf_parquet_rows_load_the_parquet_builder_with_the_revision_pinned_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """the revision-pinned hf:// URL is the only load route for a builder-script repo (bigbio/chia
    carries chia.py, which datasets>=3 refuses), so the exact data_files/split/streaming call is
    pinned here; match_on still filters and rows still project onto columns_out"""
    Calls: dict[str, Any] = {}

    def FakeLoadDataset(builder: str, **kwargs: Any) -> list[dict[str, Any]]:
        Calls["builder"] = builder
        Calls["kwargs"] = kwargs
        return [
            {"passages": [{"text": ["criteria"]}], "entities": [1], "relations": [2], "domain": "Healthcare"},
            {"passages": [{"text": ["other"]}], "entities": [3], "relations": [4], "domain": "Finance"},
        ]

    monkeypatch.setattr(hf_parquet, "load_dataset", FakeLoadDataset)
    Stream: HuggingFaceParquetDataStream = HuggingFaceParquetDataStream(
        ("script", "ChiaScript", ("entities", "relations")),
        1.0,
        "bigbio/chia",
        "chia_bigbio_kb/train/0000.parquet",
        "refs/convert/parquet",
        "train",
        (("domain", ("Healthcare",)),),
        ("passages", "entities", "relations"),
    )

    Streamed: list[StreamedRow] = list(Stream.rows())

    assert Calls == {
        "builder": "parquet",
        "kwargs": {
            "data_files": "hf://datasets/bigbio/chia@refs/convert/parquet/chia_bigbio_kb/train/0000.parquet",
            "split": "train",
            "streaming": True,
        },
    }
    assert len(Streamed) == 1
    assert Streamed[0][0] == "bigbio/chia"
    assert Streamed[0][1] == (
        ("script", "ChiaScript", ("entities", "relations")),
        (
            [{"text": ["criteria"]}],
            [1],
            [2],
        ),
    )


def test_hf_parquet_rows_without_a_match_declaration_yield_every_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """no match_on means no filtering: every streamed row ships, projected onto columns_out"""

    def FakeLoadDataset(builder: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"passages": [1], "entities": [2], "relations": [3]},
            {"passages": [4], "entities": [5], "relations": [6]},
        ]

    monkeypatch.setattr(hf_parquet, "load_dataset", FakeLoadDataset)
    Stream: HuggingFaceParquetDataStream = HuggingFaceParquetDataStream(
        ("script", "ChiaScript", ("entities", "relations")),
        1.0,
        "bigbio/chia",
        "chia_bigbio_kb/train/0000.parquet",
        "refs/convert/parquet",
        "train",
        None,
        ("passages", "entities", "relations"),
    )

    Streamed: list[StreamedRow] = list(Stream.rows())

    assert len(Streamed) == 2
    assert Streamed[1][1] == (("script", "ChiaScript", ("entities", "relations")), ([4], [5], [6]))


def test_hf_parquet_load_failure_propagates_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """a hub-side load failure (missing file, bad revision) must surface, never become an empty
    stream: the silent-empty-training-set bug this repo already shipped once"""

    def Boom(builder: str, **kwargs: Any) -> None:
        raise FileNotFoundError("no such parquet")

    monkeypatch.setattr(hf_parquet, "load_dataset", Boom)
    Stream: HuggingFaceParquetDataStream = HuggingFaceParquetDataStream(
        ("script", "ChiaScript", ("entities", "relations")),
        1.0,
        "bigbio/chia",
        "missing/train/0000.parquet",
        "refs/convert/parquet",
        "train",
        None,
        ("passages", "entities", "relations"),
    )

    with pytest.raises(FileNotFoundError):
        list(Stream.rows())


def test_registry_keys_the_hf_json_source() -> None:
    assert SOURCE_REGISTRY["hf_json"] is HuggingFaceJsonDataStream


def test_build_stream_raises_loudly_on_an_unknown_source() -> None:
    """an unregistered source is a declaration bug; the KeyError must surface, not fall through"""
    with pytest.raises(KeyError):
        build_stream("teleport", ())


def test_hf_json_rows_load_the_json_builder_non_streaming_and_honor_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """non-streaming is the fix for the cold-cache hazard: streaming promotes the mixed int/str ner
    cells to utf8 with the label wrapped in literal JSON quotes, so the kwarg must stay absent"""
    Calls: dict[str, Any] = {}

    def FakeLoadDataset(path: str, name: Any = None, **kwargs: Any) -> list[dict[str, Any]]:
        Calls["path"] = path
        Calls["name"] = name
        Calls["kwargs"] = kwargs
        return [
            {"tokenized_text": ["Aspirin"], "ner": [[0, 6, "Drug"]], "domain": "Healthcare"},
            {"tokenized_text": ["Headache"], "ner": [[0, 7, "Disease"]], "domain": "Finance"},
        ]

    monkeypatch.setattr(hf_json, "load_dataset", FakeLoadDataset)
    Stream: HuggingFaceJsonDataStream = HuggingFaceJsonDataStream(
        ("script", "PubmedAbstractsScript", ("entities",)),
        1.0,
        "knowledgator/PubMedAbstractsNER",
        "train.json",
        "train",
        (("domain", ("Healthcare",)),),
        ("tokenized_text", "ner"),
    )

    Streamed: list[StreamedRow] = list(Stream.rows())

    assert Calls == {
        "path": "json",
        "name": None,
        "kwargs": {"data_files": "hf://datasets/knowledgator/PubMedAbstractsNER/train.json", "split": "train"},
    }
    assert Streamed == [("knowledgator/PubMedAbstractsNER", (("script", "PubmedAbstractsScript", ("entities",)), (["Aspirin"], [[0, 6, "Drug"]])))]


def test_hf_json_rows_without_a_match_declaration_yield_every_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """an undeclared match_on is the common case: no row is filtered, payload keeps the declared columns"""

    def FakeLoadDataset(path: str, name: Any = None, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"tokenized_text": ["Aspirin"], "ner": []}, {"tokenized_text": ["Headache"], "ner": []}]

    monkeypatch.setattr(hf_json, "load_dataset", FakeLoadDataset)
    Stream: HuggingFaceJsonDataStream = HuggingFaceJsonDataStream(
        ("script", "PubmedAbstractsScript", ("entities",)),
        1.0,
        "knowledgator/PubMedAbstractsNER",
        "train.json",
        "train",
        None,
        ("tokenized_text",),
    )

    Streamed: list[StreamedRow] = list(Stream.rows())

    assert Streamed == [
        ("knowledgator/PubMedAbstractsNER", (("script", "PubmedAbstractsScript", ("entities",)), (["Aspirin"],))),
        ("knowledgator/PubMedAbstractsNER", (("script", "PubmedAbstractsScript", ("entities",)), (["Headache"],))),
    ]


def test_hf_json_load_failure_propagates_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """a load_dataset error (e.g. the KeyError: 'feature' from a stale dataset_infos.json) is a real
    ingest failure; the stream must surface it unwrapped"""

    def Boom(path: str, name: Any = None, **kwargs: Any) -> Any:
        raise RuntimeError("hub exploded")

    monkeypatch.setattr(hf_json, "load_dataset", Boom)
    Stream: HuggingFaceJsonDataStream = HuggingFaceJsonDataStream(
        ("script", "PubmedAbstractsScript", ("entities",)),
        1.0,
        "knowledgator/PubMedAbstractsNER",
        "train.json",
        "train",
        None,
        ("tokenized_text", "ner"),
    )

    with pytest.raises(RuntimeError, match="hub exploded"):
        list(Stream.rows())


# ------------------------------------------------------------------ US-009 quality accounting --

_QUALITY_TASK: tuple[Any, ...] = ("script", "GlinerBiomedScript", ("entities",))
_QUALITY_SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "QualityRow",
    "namespace": "relmedner.tests",
    "fields": [{"name": "name", "type": "string"}],
}


def _write_avro(path: Path, records: list[dict[str, Any]]) -> Path:
    with path.open("wb") as handle:
        writer(handle, parse_schema(_QUALITY_SCHEMA), records)
    return path


def _hf_stream(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]], **overrides: Any) -> HuggingFaceDataStream:
    """an hf stream over in-memory rows: load_dataset is patched out, so the production rows()
    loop (match_on -> evaluator -> counters) is what the test exercises"""
    monkeypatch.setattr("relmedner.huggingface.load_dataset", lambda *args, **kwargs: rows)
    declared: dict[str, Any] = dict(
        task=_QUALITY_TASK,
        weight=1.0,
        dataset="fake/quality",
        subset=None,
        split="train",
        match_on=None,
        columns_out=("text",),
    )
    declared.update(overrides)
    return HuggingFaceDataStream(**declared)


def test_report_line_is_stable_and_omits_empty_dropped() -> None:
    Stats: StreamStats = StreamStats(rows_in=1000, rows_out=842, dropped_by={"match_on": 150, "min_text_len": 8})
    assert Stats.report() == "rows_in=1000 rows_out=842 dropped={match_on:150, min_text_len:8}"

    Clean: StreamStats = StreamStats(rows_in=20, rows_out=20)
    assert Clean.report() == "rows_in=20 rows_out=20"


def test_every_drop_reason_fires_exactly_once_and_the_stats_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    """crafted rows hit match_on plus all five evaluator reasons exactly once each; the spec
    example math must hold (rows_in - rows_out == sum of dropped) and the report line itself
    is part of the asserted contract"""
    Rows: list[dict[str, Any]] = [
        {"domain": "other", "text": "a fine row dropped only by match_on"},
        {"domain": "ok", "text": ""},  # drop_empty
        {"domain": "ok", "text": "hi"},  # min_text_len (2 < 5)
        {"domain": "ok", "text": "nothing here"},  # include_regex ("world" absent)
        {"domain": "ok", "text": "goodbye cruel world"},  # exclude_regex ("cruel" present)
        {"domain": "ok", "text": "hello world"},  # kept
    ]
    Stream: HuggingFaceDataStream = _hf_stream(
        monkeypatch,
        Rows,
        match_on=(("domain", ("ok",)),),
        filters=RowFilters(drop_empty=True, min_text_len=5, include_regex="world", exclude_regex="cruel"),
    )

    Yielded: list[StreamedRow] = list(Stream.stream(RunConfig()))

    assert Stream.stats.rows_in == 6  # every row read, the match_on drop included
    assert Stream.stats.rows_out == 1
    assert Stream.stats.dropped_by == {
        "match_on": 1,
        "drop_empty": 1,
        "min_text_len": 1,
        "include_regex": 1,
        "exclude_regex": 1,
    }
    assert Stream.stats.rows_in - Stream.stats.rows_out == sum(Stream.stats.dropped_by.values())
    assert Yielded == [("fake/quality", (_QUALITY_TASK, ("hello world",)))]
    assert Stream.stats.report() == ("rows_in=6 rows_out=1 dropped={match_on:1, drop_empty:1, min_text_len:1, include_regex:1, exclude_regex:1}")


def test_quality_line_logs_once_per_pass_on_the_quality_logger(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    Rows: list[dict[str, Any]] = [{"text": "aspirin trial"}, {"text": "x"}]
    Stream: HuggingFaceDataStream = _hf_stream(monkeypatch, Rows, filters=RowFilters(min_text_len=5))

    with caplog.at_level(logging.INFO, logger="relmedner.quality"):
        Yielded: list[StreamedRow] = list(Stream.stream(RunConfig()))

    assert len(Yielded) == 1
    Quality: list[logging.LogRecord] = [record for record in caplog.records if record.name == "relmedner.quality"]
    assert len(Quality) == 1
    assert Quality[0].getMessage() == "ingest quality fake/quality: rows_in=2 rows_out=1 dropped={min_text_len:1}"


def test_quality_line_still_logs_before_zero_yield_error_raises(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """the US-008 guard must not swallow the report: stream()'s finally lands the quality line
    before ZeroYieldError reaches the caller"""
    Rows: list[dict[str, Any]] = [{"text": "short"}, {"text": "tiny"}]
    Stream: HuggingFaceDataStream = _hf_stream(monkeypatch, Rows, filters=RowFilters(min_text_len=1000))

    with caplog.at_level(logging.INFO, logger="relmedner.quality"):
        with pytest.raises(ZeroYieldError, match=r"dropped 100% of 2 rows"):
            list(Stream.stream(RunConfig()))

    Quality: list[logging.LogRecord] = [record for record in caplog.records if record.name == "relmedner.quality"]
    assert len(Quality) == 1
    assert Quality[0].getMessage() == "ingest quality fake/quality: rows_in=2 rows_out=0 dropped={min_text_len:2}"


def test_no_filters_no_match_on_counts_without_touching_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """the no-op guarantee: with neither filters nor match_on the yielded rows stay identical
    to the plain projection while the counters still count and dropped_by stays empty"""
    Rows: list[dict[str, Any]] = [{"text": "alpha"}, {"text": "beta"}]
    Stream: HuggingFaceDataStream = _hf_stream(monkeypatch, Rows)

    Yielded: list[StreamedRow] = list(Stream.stream(RunConfig()))

    assert Yielded == [("fake/quality", (_QUALITY_TASK, ("alpha",))), ("fake/quality", (_QUALITY_TASK, ("beta",)))]
    assert Stream.stats.rows_in == 2 and Stream.stats.rows_out == 2
    assert Stream.stats.dropped_by == {}
    assert Stream.stats.report() == "rows_in=2 rows_out=2"


def test_local_stream_accounts_records_read_and_filter_drops(tmp_path: Path) -> None:
    Target: Path = _write_avro(
        tmp_path / "quality.avro",
        [{"name": "aspirin trial"}, {"name": "x"}, {"name": ""}],
    )
    Stream: LocalAvroDataStream = LocalAvroDataStream(_QUALITY_TASK, 1.0, str(Target), filters=RowFilters(drop_empty=True, min_text_len=5))

    Yielded: list[StreamedRow] = list(Stream.stream(RunConfig()))

    assert [values[0]["name"] for _source, (_task, values) in Yielded] == ["aspirin trial"]
    assert Stream.stats.rows_in == 3 and Stream.stats.rows_out == 1
    assert Stream.stats.dropped_by == {"drop_empty": 1, "min_text_len": 1}


def test_local_unfiltered_pass_counts_without_dropping(tmp_path: Path) -> None:
    Target: Path = _write_avro(tmp_path / "plain.avro", [{"name": "a"}, {"name": "b"}])
    Stream: LocalAvroDataStream = LocalAvroDataStream(_QUALITY_TASK, 1.0, str(Target))

    Yielded: list[StreamedRow] = list(Stream.stream(RunConfig()))

    assert [values[0]["name"] for _source, (_task, values) in Yielded] == ["a", "b"]
    assert Stream.stats.rows_in == 2 and Stream.stats.rows_out == 2
    assert Stream.stats.dropped_by == {}
