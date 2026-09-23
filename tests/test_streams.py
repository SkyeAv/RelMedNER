from __future__ import annotations

import logging
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from typing import Any, ClassVar, Self

import pytest
from fastavro import parse_schema, writer

from relmedner import hf_json
from relmedner.hf_json import HuggingFaceJsonDataStream
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


# ------------------------------------------------- US-002 always-on cap on every source --

_OVER_CAP: str = "word " * 7000  # 35000 joined chars > MAX_TEXT_TOKENS * CHARS_PER_TOKEN (32768)


def test_hf_filters_none_source_still_drops_over_cap_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """the cap is ALWAYS-ON: with no declared filters the stream must still drop an over-cap row
    and attribute it to max_tokens, while a short row yields unchanged -- before effective_filters
    the evaluator only ran when filters were declared, so unfiltered sources leaked over-cap rows"""
    Rows: list[dict[str, Any]] = [{"text": "aspirin trial"}, {"text": _OVER_CAP}]
    Stream: HuggingFaceDataStream = _hf_stream(monkeypatch, Rows)

    Yielded: list[StreamedRow] = list(Stream.rows())

    assert Yielded == [("fake/quality", (_QUALITY_TASK, ("aspirin trial",)))]
    assert Stream.filters is None and Stream.effective_filters == RowFilters()
    assert Stream.stats.dropped_by == {"max_tokens": 1} and Stream.stats.rows_out == 1


def test_hf_json_filters_none_pass_is_counted_and_drops_over_cap_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """the removed fast path: a filters-None hf_json pass must populate stats and apply the cap
    exactly like the other three sources. Before, None skipped the evaluator AND the counters AND
    the quality line, so an over-cap row streamed through with no accounting anywhere"""
    monkeypatch.setattr(hf_json, "load_dataset", lambda *args, **kwargs: [{"text": "aspirin trial"}, {"text": _OVER_CAP}])
    Stream: HuggingFaceJsonDataStream = HuggingFaceJsonDataStream(
        _QUALITY_TASK, 1.0, "knowledgator/PubMedAbstractsNER", "train.json", "train", None, ("text",)
    )

    Yielded: list[StreamedRow] = list(Stream.rows())

    assert Yielded == [("knowledgator/PubMedAbstractsNER", (_QUALITY_TASK, ("aspirin trial",)))]
    assert Stream.filters is None and Stream.effective_filters == RowFilters()
    assert Stream.stats.rows_in == 2 and Stream.stats.rows_out == 1
    assert Stream.stats.dropped_by == {"max_tokens": 1}


def test_local_avro_filters_none_source_still_drops_over_cap_records(tmp_path: Path) -> None:
    """same always-on guarantee for the local avro source: the unfiltered path attributes an
    over-cap record to max_tokens while short records stay byte-identical"""
    Target: Path = _write_avro(tmp_path / "cap.avro", [{"name": "aspirin trial"}, {"name": _OVER_CAP}])
    Stream: LocalAvroDataStream = LocalAvroDataStream(_QUALITY_TASK, 1.0, str(Target))

    Yielded: list[StreamedRow] = list(Stream.stream(RunConfig()))

    assert [values[0]["name"] for _source, (_task, values) in Yielded] == ["aspirin trial"]
    assert Stream.stats.dropped_by == {"max_tokens": 1} and Stream.stats.rows_out == 1


def test_quality_line_counts_max_tokens_when_the_cap_empties_a_source(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """the cap's drops are first-class US-009 counters: a source emptied ONLY by the cap (filters
    None) must report dropped={max_tokens: N} on the quality line, and the guard raise must not
    swallow it (stream()'s finally lands the line first)"""
    Stream: HuggingFaceDataStream = _hf_stream(monkeypatch, [{"text": _OVER_CAP}, {"text": _OVER_CAP + "a"}])

    with caplog.at_level(logging.INFO, logger="relmedner.quality"):
        with pytest.raises(ZeroYieldError, match=r"dropped 100% of 2 rows"):
            list(Stream.stream(RunConfig()))

    Quality: list[logging.LogRecord] = [record for record in caplog.records if record.name == "relmedner.quality"]
    assert len(Quality) == 1
    assert Quality[0].getMessage() == "ingest quality fake/quality: rows_in=2 rows_out=0 dropped={max_tokens:2}"


def test_a_match_on_emptied_source_stays_recorded_not_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """negative test for the now-unconditional guard: it arms on the post-match_on candidate
    count, so a source emptied only by match_on is a recorded drop, never a ZeroYieldError,
    even though the guard no longer checks whether filters were declared"""
    Rows: list[dict[str, Any]] = [{"domain": "Finance", "text": "aspirin trial"}, {"domain": "Finance", "text": "metformin study"}]
    Stream: HuggingFaceDataStream = _hf_stream(monkeypatch, Rows, match_on=(("domain", ("ok",)),))

    Yielded: list[StreamedRow] = list(Stream.stream(RunConfig()))

    assert Yielded == []
    assert Stream.stats.dropped_by == {"match_on": 2}
