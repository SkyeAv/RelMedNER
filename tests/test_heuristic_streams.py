"""end-to-end coverage of the six heuristic RowFilters knobs at the stream chokepoint
(better-denoising US-003): the evaluator-level rules are proven in tests/test_row_filters.py,
so this file proves the WIRING -- real avro/TSV containers streamed through the local
DataStreams attribute heuristic drops to the exact reason strings in StreamStats.dropped_by,
the relmedner.quality line reports them, the zero-yield guard fires when a heuristic kills a
whole source, and an hf declaration carrying heuristic filters reaches HuggingFaceDataStream
intact (construction-only, no hub contact). Same no-network fixture discipline as
tests/test_context_cap.py: every row here is a real file on disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from fastavro import parse_schema, writer

from relmedner.huggingface import HuggingFaceDataStream
from relmedner.local import LocalAvroDataStream, LocalDelimitedDataStream
from relmedner.models import HuggingFaceDataset, RowFilters, RunConfig, ScriptTask
from relmedner.registry import build_stream
from relmedner.streams import ZeroYieldError

TASK: tuple[Any, ...] = ("script", "CtkpInterventionsScript", ("entities",))
SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "Row",
    "namespace": "relmedner.tests",
    "fields": [{"name": "name", "type": ["null", "string"], "default": None}],
}


def write_avro(path: Path, names: list[str | None]) -> Path:
    with path.open("wb") as handle:
        writer(handle, parse_schema(SCHEMA), [{"name": name} for name in names])
    return path


def write_tsv(path: Path, texts: tuple[str, ...]) -> str:
    path.write_text("\n".join(("text", *texts)) + "\n", encoding="utf-8")
    return str(path)


def test_avro_stream_keeps_exactly_at_the_word_floor_and_drops_one_below(tmp_path: Path) -> None:
    """boundary exactness at stream level: min_words=3 keeps the 3-word row and drops the
    2-word row with reason min_words -- the strict < comparison must survive the full
    avro -> rows() -> evaluator path unchanged"""
    Target: Path = write_avro(tmp_path / "words.avro", ["aspirin trial of", "aspirin trial"])
    Stream: LocalAvroDataStream = LocalAvroDataStream(TASK, 1.0, str(Target), filters=RowFilters(min_words=3))

    Yielded: list[Any] = list(Stream.rows())

    assert [values[0]["name"] for _source, (_task, values) in Yielded] == ["aspirin trial of"]
    assert Stream.stats.rows_in == 2 and Stream.stats.rows_out == 1
    assert Stream.stats.dropped_by == {"min_words": 1}


def test_delimited_stream_attributes_symbol_ratio_drops(tmp_path: Path) -> None:
    """the TSV projection rides the same evaluator: a clean prose row passes
    max_symbol_ratio=0.5 and a markup-noise row (0.75 symbol share) drops with its exact
    reason string, so per-source yaml cleanup shows up per-reason in the quality line"""
    Path_: str = write_tsv(tmp_path / "symbols.tsv", ("a b c d", "x!!!"))
    Stream: LocalDelimitedDataStream = LocalDelimitedDataStream(TASK, 1.0, Path_, ("text",), filters=RowFilters(max_symbol_ratio=0.5))

    Rows: list[Any] = list(Stream.rows())

    assert [values[0] for _source, (_task, values) in Rows] == ["a b c d"]
    assert Stream.stats.rows_in == 2 and Stream.stats.rows_out == 1
    assert Stream.stats.dropped_by == {"max_symbol_ratio": 1}


def test_the_quality_line_counts_each_heuristic_reason_separately(tmp_path: Path, caplog: Any) -> None:
    """US-009's per-reason accounting must attribute each heuristic drop to its own reason
    in first-fired order (min_words before max_upper_ratio here), and the survivor keeps the
    line from being swallowed by the zero-yield guard"""
    Target: Path = write_avro(
        tmp_path / "mixed.avro",
        ["aspirin trial", "ASAP NOW ASAP NOW ASAP", "the study of the trial results"],
    )
    Stream: LocalAvroDataStream = LocalAvroDataStream(TASK, 1.0, str(Target), filters=RowFilters(min_words=5, max_upper_ratio=0.5))

    with caplog.at_level(logging.INFO, logger="relmedner.quality"):
        Rows: list[Any] = list(Stream.stream(RunConfig()))

    assert [values[0]["name"] for _source, (_task, values) in Rows] == ["the study of the trial results"]
    assert Stream.stats.dropped_by == {"min_words": 1, "max_upper_ratio": 1}
    Quality: list[Any] = [record for record in caplog.records if record.name == "relmedner.quality"]
    assert len(Quality) == 1
    assert "rows_in=3 rows_out=1" in Quality[0].getMessage()
    assert "dropped={min_words:1, max_upper_ratio:1}" in Quality[0].getMessage()


def test_a_heuristic_filter_emptying_a_source_raises_zero_yield(tmp_path: Path) -> None:
    """fail-loud: a threshold that drops 100% of a non-empty source (a mis-typed word
    floor against a corpus of titles, say) is the silent-empty-training-set bug, so it
    raises ZeroYieldError at exhaustion with the drops attributed, not a quiet empty run"""
    Target: Path = write_avro(tmp_path / "guard.avro", ["BRCA1", "TP53"])
    Stream: LocalAvroDataStream = LocalAvroDataStream(TASK, 1.0, str(Target), filters=RowFilters(min_words=100))

    with pytest.raises(ZeroYieldError, match=r"dropped 100% of 2 rows"):
        list(Stream.rows())
    assert Stream.stats.dropped_by == {"min_words": 2}


def test_a_partial_heuristic_filter_never_raises(tmp_path: Path) -> None:
    """the guard arms only at 100% loss: a 50% cull is the intended behavior of a
    measured threshold, so survivors flow unchanged and nothing raises"""
    Target: Path = write_avro(tmp_path / "partial.avro", ["BRCA1", "a proper trial row"])
    Stream: LocalAvroDataStream = LocalAvroDataStream(TASK, 1.0, str(Target), filters=RowFilters(min_words=4))

    Rows: list[Any] = list(Stream.rows())

    assert [values[0]["name"] for _source, (_task, values) in Rows] == ["a proper trial row"]


def test_an_hf_declaration_carries_heuristic_filters_to_the_stream_intact() -> None:
    """construction-only (no hub contact): a declared hf dataset with heuristic knobs
    builds an HuggingFaceDataStream whose effective_filters IS the declared filters, so the
    evaluator -- and through it every heuristic rule -- applies on the hub-streaming path
    exactly as on the local paths proven above"""
    Filters: RowFilters = RowFilters(min_stop_word_ratio=0.05, max_repeat_ngram_ratio=0.3, max_short_line_ratio=0.5)
    Dataset: HuggingFaceDataset = HuggingFaceDataset(
        task=ScriptTask(type="script", name="GlinerBiomedScript", outputs=["entities"]),
        source="hf",
        dataset="a/b",
        columns_out=["text"],
        filters=Filters,
    )

    Stream = build_stream(*Dataset.to_stream_args())

    assert isinstance(Stream, HuggingFaceDataStream)
    assert Stream.filters is Filters
    assert Stream.effective_filters is Filters
    assert Stream.columns_out == ("text",)
