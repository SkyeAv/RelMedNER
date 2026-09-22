from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from fastavro import parse_schema, writer
from pydantic import ValidationError

from relmedner.constants import CHARS_PER_TOKEN, MAX_TEXT_TOKENS
from relmedner.huggingface import HuggingFaceDataStream
from relmedner.local import LocalAvroDataStream
from relmedner.models import HuggingFaceDataset, LocalAvroDataset, RowFilters, ScriptTask, YamlIngests
from relmedner.registry import build_stream
from relmedner.row_filters import first_drop_reason
from relmedner.streams import ZeroYieldError

TASK: tuple[Any, ...] = ("script", "CtkpInterventionsScript", ("entities",))
SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "Row",
    "namespace": "relmedner.tests",
    "fields": [{"name": "name", "type": "string"}],
}


def write_avro(path: Path, records: list[dict[str, Any]]) -> Path:
    with path.open("wb") as handle:
        writer(handle, parse_schema(SCHEMA), records)
    return path


def local_stream(path: Path, filters: RowFilters | None = None) -> LocalAvroDataStream:
    return LocalAvroDataStream(TASK, 1.0, str(path), filters=filters)


# ------------------------------------------------------------------ model validation --


def test_unknown_filter_key_is_rejected() -> None:
    """extra="forbid" is the tripwire: a typo'd filter name must fail validation, never be ignored"""
    with pytest.raises(ValidationError):
        RowFilters(drop_emtpy=True)  # typo: must not be silently accepted


def test_min_text_len_above_max_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RowFilters(min_text_len=5, max_text_len=2)
    # equal bounds are a legitimate exact-length window
    assert RowFilters(min_text_len=3, max_text_len=3).min_text_len == 3


def test_negative_length_bounds_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RowFilters(min_text_len=-1)
    with pytest.raises(ValidationError):
        RowFilters(max_text_len=-1)


def test_defaults_declare_no_filtering() -> None:
    assert RowFilters() == RowFilters(drop_empty=False, min_text_len=None, max_text_len=None, include_regex=None, exclude_regex=None)


# ------------------------------------------------------------------ pure evaluator --


def test_drop_empty_fires_only_when_every_value_is_empty() -> None:
    Filters: RowFilters = RowFilters(drop_empty=True)

    assert first_drop_reason(("", None), Filters) == "drop_empty"
    assert first_drop_reason(([],), Filters) == "drop_empty"
    assert first_drop_reason(({},), Filters) == "drop_empty"
    assert first_drop_reason(("", "x"), Filters) is None


def test_min_and_max_text_len_measure_joined_text() -> None:
    assert first_drop_reason(("ab",), RowFilters(min_text_len=3)) == "min_text_len"
    assert first_drop_reason(("abc",), RowFilters(min_text_len=3)) is None
    assert first_drop_reason(("abcd",), RowFilters(max_text_len=3)) == "max_text_len"
    assert first_drop_reason(("abc",), RowFilters(max_text_len=3)) is None


def test_token_lists_join_elementwise_and_non_strings_are_skipped() -> None:
    """the text rule: a list/tuple of str contributes its elements joined, other non-str values
    contribute nothing; this is what makes tokenized_text columns filterable"""
    assert first_drop_reason((["a", "b"],), RowFilters(min_text_len=4)) == "min_text_len"
    assert first_drop_reason((["a", "b"],), RowFilters(min_text_len=3)) is None
    # a number contributes no text, so the joined text is "" and a positive min drops it
    assert first_drop_reason((42, "xy"), RowFilters(min_text_len=4)) == "min_text_len"
    assert first_drop_reason((42, "xyz"), RowFilters(min_text_len=3)) is None


def test_regexes_include_then_exclude() -> None:
    Include: RowFilters = RowFilters(include_regex="aspirin")
    assert first_drop_reason(("metformin",), Include) == "include_regex"
    assert first_drop_reason(("aspirin study",), Include) is None

    Exclude: RowFilters = RowFilters(exclude_regex="^withdrawn ")
    assert first_drop_reason(("withdrawn trial",), Exclude) == "exclude_regex"
    assert first_drop_reason(("active trial",), Exclude) is None


def test_the_first_reason_wins_in_the_fixed_order() -> None:
    """drop_empty outranks the length rules, the platform token cap outranks the regexes, and
    the regexes come last: drop_empty -> min_text_len -> max_text_len -> max_tokens ->
    include_regex -> exclude_regex; US-009 counts drops per reason, so the order is part of
    the contract"""
    Both: RowFilters = RowFilters(drop_empty=True, min_text_len=5, include_regex="x", exclude_regex="y")

    assert first_drop_reason(("",), Both) == "drop_empty"
    assert first_drop_reason(("ab",), Both) == "min_text_len"  # include/exclude would also fire; min wins
    OverCap: RowFilters = RowFilters(min_text_len=5, include_regex="x", exclude_regex="y")
    # cap outranks BOTH regexes even though include_regex "x" would also fail on this text
    assert first_drop_reason(("a" * (MAX_TEXT_TOKENS * CHARS_PER_TOKEN + 1),), OverCap) == "max_tokens"
    Window: RowFilters = RowFilters(min_text_len=2, max_text_len=5)
    assert first_drop_reason(("abcdef",), Window) == "max_text_len"  # min passes first, max fires


def test_the_token_cap_boundary_is_strict_over_max_text_tokens_times_chars_per_token() -> None:
    """the cap converts with CHARS_PER_TOKEN: exactly MAX_TEXT_TOKENS * CHARS_PER_TOKEN chars
    (32768, an estimated len // 4 == 8192 tokens) passes and ONE char more drops; strict > on
    chars keeps the boundary exact -- floor-dividing to tokens first would round 32769 back to
    8192 estimated tokens and silently keep the over-cap row"""
    Boundary: str = "a" * (MAX_TEXT_TOKENS * CHARS_PER_TOKEN)
    assert len(Boundary) // CHARS_PER_TOKEN == MAX_TEXT_TOKENS  # the estimate at the boundary
    assert first_drop_reason((Boundary,), RowFilters()) is None
    assert first_drop_reason((Boundary + "a",), RowFilters()) == "max_tokens"


def test_the_cap_fires_with_no_declared_rules() -> None:
    """always-on: default RowFilters() sets no rules at all, yet an over-cap row must attribute
    to max_tokens -- the cap is a platform constant, so a per-dataset config cannot waive it
    by simply not declaring any rule"""
    assert first_drop_reason(("word " * 7000,), RowFilters()) == "max_tokens"  # 35000 chars
    assert first_drop_reason(("word " * 6500,), RowFilters()) is None  # 32500 chars, under the cap


def test_the_cap_sits_between_max_text_len_and_include_regex() -> None:
    """attribution must stay stable for US-009's per-reason counts: a row failing both the
    declared max_text_len and the cap attributes to max_text_len (existing behavior pinned),
    while a row failing both include_regex and the cap attributes to max_tokens (cap precedes
    the regexes)"""
    Big: str = "aspirin" * 5000  # 35000 chars, over the cap
    assert first_drop_reason((Big,), RowFilters(max_text_len=1000)) == "max_text_len"
    assert first_drop_reason((Big,), RowFilters(include_regex="metformin")) == "max_tokens"
    # the cap wins even when the include regex MATCHES: an over-cap row never streams
    assert first_drop_reason((Big,), RowFilters(include_regex="aspirin")) == "max_tokens"


# ------------------------------------------------------------------ stream application --


def test_an_all_dropping_filter_on_a_non_empty_source_raises_at_generator_end(tmp_path: Path) -> None:
    """the fail-loud zero-yield guard: a filter that drops 100% of a non-empty source is the
    historical silent-empty-training-set bug, so it raises only after the source is exhausted"""
    Target: Path = write_avro(tmp_path / "guard.avro", [{"name": "aspirin"}, {"name": "metformin"}])
    Stream: LocalAvroDataStream = local_stream(Target, RowFilters(min_text_len=1000))

    with pytest.raises(ZeroYieldError, match=r"dropped 100% of 2 rows"):
        list(Stream.rows())


def test_a_genuinely_empty_source_is_not_an_error(tmp_path: Path) -> None:
    Target: Path = write_avro(tmp_path / "empty.avro", [])

    assert list(local_stream(Target, RowFilters(min_text_len=3)).rows()) == []


def test_filters_none_is_byte_identical_to_the_unfiltered_path(tmp_path: Path) -> None:
    Records: list[dict[str, Any]] = [{"name": "aspirin"}, {"name": "metformin"}]
    Target: Path = write_avro(tmp_path / "same.avro", Records)
    Defaulted: LocalAvroDataStream = LocalAvroDataStream(TASK, 1.0, str(Target), filters=None)
    Plain: LocalAvroDataStream = LocalAvroDataStream(TASK, 1.0, str(Target))

    assert Defaulted.filters is None and Plain.filters is None
    assert list(Defaulted.rows()) == list(Plain.rows())


def test_a_partial_filter_keeps_the_surviving_rows(tmp_path: Path) -> None:
    Target: Path = write_avro(tmp_path / "partial.avro", [{"name": "aspirin trial"}, {"name": "x"}])

    Rows: list[Any] = list(local_stream(Target, RowFilters(min_text_len=5)).rows())

    assert [values[0]["name"] for _source, (_task, values) in Rows] == ["aspirin trial"]


def test_an_invalid_regex_raises_at_construction() -> None:
    """regexes compile once in __init__ so a bad pattern fails before any row streams, never
    mid-run"""
    with pytest.raises(re.error):
        LocalAvroDataStream(TASK, 1.0, "~/x.avro", filters=RowFilters(include_regex="(["))
    with pytest.raises(re.error):
        HuggingFaceDataStream(
            task=TASK,
            weight=1.0,
            dataset="a/b",
            subset=None,
            split="train",
            match_on=None,
            columns_out=("text",),
            filters=RowFilters(exclude_regex="*not-anchored"),
        )


# ------------------------------------------------------------------ envelope plumbing --


def _hf_dataset(filters: RowFilters | None = None) -> HuggingFaceDataset:
    return HuggingFaceDataset(
        task=ScriptTask(type="script", name="GlinerBiomedScript", outputs=["entities"]),
        source="hf",
        dataset="a/b",
        columns_out=["text"],
        filters=filters,
    )


def test_to_stream_args_appends_filters_after_the_frozen_payload() -> None:
    """the payload stays the 2-tuple the EXPECTED locks pin; filters ride a third envelope slot"""
    Filters: RowFilters = RowFilters(min_text_len=3)

    assert _hf_dataset().to_stream_args() == ("hf", _hf_dataset().to_tuple()[1], None)
    assert _hf_dataset(Filters).to_stream_args() == ("hf", _hf_dataset().to_tuple()[1], Filters)


def test_yaml_ingests_stream_args_round_every_declared_dataset() -> None:
    Ingests: YamlIngests = YamlIngests(datasets=[_hf_dataset(RowFilters(drop_empty=True))])

    assert Ingests.stream_args() == (("hf", _hf_dataset().to_tuple()[1], RowFilters(drop_empty=True)),)
    # generate_tuples keeps the frozen 2-tuple shape the cli parallelism count depends on
    assert Ingests.generate_tuples() == (("hf", _hf_dataset().to_tuple()[1]),)


def test_build_stream_wires_filters_keyword_only() -> None:
    Payload: tuple[Any, ...] = (TASK, 1.0, "interventions/interventions.avro")
    Filters: RowFilters = RowFilters(drop_empty=True)

    Stream: LocalAvroDataStream = build_stream("local", Payload, Filters)
    assert isinstance(Stream, LocalAvroDataStream)
    assert Stream.filters == Filters
    # the filters default keeps every existing positional 2-arg call valid
    assert build_stream("local", Payload).filters is None


def test_local_dataset_declares_filters_through_yaml_shape() -> None:
    Dataset: LocalAvroDataset = LocalAvroDataset(
        task=ScriptTask(type="script", name="CtkpInterventionsScript", outputs=["entities"]),
        source="local",
        path="~/x.avro",
        filters=RowFilters(max_text_len=10_000),
    )

    assert Dataset.filters is not None and Dataset.filters.max_text_len == 10_000
    assert Dataset.to_stream_args()[2] == Dataset.filters
