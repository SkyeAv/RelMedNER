from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from fastavro import parse_schema, writer
from pydantic import ValidationError

from relmedner import row_filters as row_filters_module
from relmedner.constants import CHARS_PER_TOKEN, MAX_TEXT_TOKENS
from relmedner.huggingface import HuggingFaceDataStream
from relmedner.local import LocalAvroDataStream
from relmedner.models import HuggingFaceDataset, LocalAvroDataset, RowFilters, ScriptTask, YamlIngests
from relmedner.registry import build_stream
from relmedner.row_filters import (
    first_drop_reason,
    repeat_ngram_ratio,
    short_line_ratio,
    stop_word_ratio,
    symbol_ratio,
    upper_ratio,
    word_count,
)
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


def test_a_cap_only_emptying_source_raises_even_without_declared_filters(tmp_path: Path) -> None:
    """the zero-yield guard now covers the ALWAYS-ON cap: a source whose every record exceeds
    MAX_TEXT_TOKENS * CHARS_PER_TOKEN empties silently under the old guard (it only armed when
    filters were declared), so the raise must fire with filters=None too, same message shape"""
    Target: Path = write_avro(tmp_path / "cap-guard.avro", [{"name": "word " * 7000}, {"name": "word " * 7001}])
    Stream: LocalAvroDataStream = local_stream(Target)

    assert Stream.filters is None
    with pytest.raises(ZeroYieldError, match=r"dropped 100% of 2 rows"):
        list(Stream.rows())
    # and the drops are attributed, not silent
    assert Stream.stats.dropped_by == {"max_tokens": 2}


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


# ------------------------------------------------------------------ quality heuristics --
# The web-corpus QC primitive family (C4 / Gopher document-level heuristics): pure ratio
# functions the evaluator composes into opt-in drop rules. Each test pins the exact ratio
# semantics on hand-built texts because every future threshold in ingests.yaml is set
# against these numbers -- a silent semantics change here would silently change what ships
# as training data.


def test_word_count_counts_whitespace_tokens() -> None:
    """min_words will drop low-word-count rows, so its counting rule must be plain
    whitespace tokenization: no punctuation stripping, no case folding, empty is zero"""
    assert word_count("") == 0
    assert word_count("   ") == 0
    assert word_count("aspirin trial") == 2
    assert word_count("  spaced   out  ") == 2
    assert word_count("BRCA1-related.") == 1


def test_stop_word_ratio_uses_the_function_word_stoplist_case_insensitively() -> None:
    """the language proxy must measure the share of closed-class English function words
    (constants.FUNCTION_WORDS, the pipeline's ONLY stoplist) over lowercase-folded tokens:
    a 100%-function-word text is 1.0, jargon-only text (gene-symbol lists, tables) is 0.0,
    and case must not hide function words ("The" at a sentence start still counts)"""
    assert stop_word_ratio("") == 0.0
    assert stop_word_ratio("   ") == 0.0
    assert stop_word_ratio("the of and") == 1.0
    assert stop_word_ratio("The Of") == 1.0
    assert stop_word_ratio("BRCA1 mutation analysis") == 0.0
    # 1 function word of 3 tokens
    assert abs(stop_word_ratio("the BRCA1 mutation") - 1 / 3) < 1e-9


def test_symbol_ratio_counts_non_alphanumeric_non_space_chars() -> None:
    """the markup/code-junk guard: over NON-SPACE chars, anything str.isalnum() rejects
    counts as symbol noise; whitespace is neutral (never inflates the ratio) and an
    all-whitespace text is 0.0, not a crash; unicode letters stay alphanumeric so a
    non-Latin script is NOT a symbol-ratio signal (that is the stop-word check's job)"""
    assert symbol_ratio("") == 0.0
    assert symbol_ratio("     ") == 0.0
    assert symbol_ratio("abc") == 0.0
    assert symbol_ratio("!!!") == 1.0
    # non-space chars a, b, ! -> one third
    assert abs(symbol_ratio("a b!") - 1 / 3) < 1e-9
    assert symbol_ratio("癌") == 0.0
    assert abs(symbol_ratio("a b!!") - 0.5) < 1e-9


def test_upper_ratio_counts_all_caps_alpha_words_over_alpha_words() -> None:
    """the weird-capitalization guard: the denominator is purely alphabetic words (digits
    and punctuation make a token like BRCA1 or @TODO neither shouted nor normal), the
    numerator is those that are entirely uppercase; no alpha words at all is 0.0"""
    assert upper_ratio("") == 0.0
    assert upper_ratio("123 !!!") == 0.0
    assert upper_ratio("aspirin trial") == 0.0
    assert upper_ratio("ASAP NOW") == 1.0
    assert upper_ratio("aspirin ASAP") == 0.5
    # BRCA1 is not purely alphabetic, so it leaves both numerator and denominator
    assert upper_ratio("BRCA1 ASAP") == 1.0


def test_repeat_ngram_ratio_flags_repeated_windows_and_passes_short_texts() -> None:
    """the repeated-words/phrases guard: sliding windows of REPEAT_NGRAM_WORDS words
    (case-folded) counted in a plain dict -- no builtin hash(); texts shorter than one
    window pass with 0.0, all-identical windows ("word " * 20 spam) approach 1.0, and a
    unique prose text is exactly 0.0"""
    assert repeat_ngram_ratio("") == 0.0
    assert repeat_ngram_ratio("short text") == 0.0
    assert repeat_ngram_ratio("one two three four five six seven eight nine") == 0.0
    # 20 identical words -> 11 windows of 10, the first is novel, the other 10 repeat
    Spam: str = " ".join(["word"] * 20)
    assert abs(repeat_ngram_ratio(Spam) - 10 / 11) < 1e-9
    # case-folded: the same phrase in different casing still repeats
    assert repeat_ngram_ratio(" ".join(["Word"] * 20)) > 0.9
    # 21 unique-ish words with the SAME 10-gram appearing twice: windows 0 and 11 match
    A: list[str] = [f"w{i}" for i in range(10)]
    Twice: str = " ".join(A + ["gap"] + A)
    assert abs(repeat_ngram_ratio(Twice) - 1 / 12) < 1e-9
    # ordinary prose: distinct 10-grams throughout
    assert repeat_ngram_ratio("the quick brown fox jumps over the lazy dog near") == 0.0


def test_short_line_ratio_measures_short_lines_over_multi_line_texts() -> None:
    """the abnormal-line-breaks guard: share of lines under SHORT_LINE_CHARS; a single-line
    text always passes 0.0 (there is no line structure to be abnormal), blank lines count
    as short (newline spam), and a trailing newline does not manufacture a phantom line"""
    assert short_line_ratio("") == 0.0
    assert short_line_ratio("one single line of ordinary prose length") == 0.0
    assert short_line_ratio("short\nlines here") == 1.0
    Mixed: str = "this line is definitely longer than thirty characters\nshort\nshort"
    assert abs(short_line_ratio(Mixed) - 2 / 3) < 1e-9
    # blank lines are zero-length, hence short
    assert short_line_ratio("\n\n") == 1.0
    # trailing newline: splitlines yields no phantom empty tail line
    assert short_line_ratio("this line is definitely longer than thirty characters\n") == 0.0


# ---------------------------------------------------- heuristic drop rules (US-002) --
# The six opt-in RowFilters knobs compose the US-001 primitives into drop rules at the
# single evaluator chokepoint. Every boundary test here pins strict comparisons (< for the
# min-* rules, > for the max-* rules): exactly-at-threshold rows must KEEP, because every
# dropped record is supervised signal and an off-by-one at a threshold silently changes
# what ships as training data.


def test_heuristic_fields_validate_bounds() -> None:
    """ratio knobs are fractions in [0.0, 1.0] and min_words is a non-negative count: a
    declaration outside those bounds is a typo and must fail validation, never clamp"""
    Knobs = ("min_stop_word_ratio", "max_symbol_ratio", "max_upper_ratio", "max_repeat_ngram_ratio", "max_short_line_ratio")
    for knob in Knobs:
        with pytest.raises(ValidationError):
            RowFilters(**{knob: -0.1})
        with pytest.raises(ValidationError):
            RowFilters(**{knob: 1.1})
    assert RowFilters(min_stop_word_ratio=0.0).min_stop_word_ratio == 0.0
    assert RowFilters(max_symbol_ratio=1.0).max_symbol_ratio == 1.0
    assert RowFilters(min_words=0).min_words == 0
    with pytest.raises(ValidationError):
        RowFilters(min_words=-1)


def test_min_words_rule_drops_low_word_count_rows() -> None:
    assert first_drop_reason(("aspirin trial",), RowFilters(min_words=3)) == "min_words"
    assert first_drop_reason(("aspirin trial of",), RowFilters(min_words=3)) is None


def test_min_stop_word_ratio_rule_drops_language_degenerate_rows() -> None:
    """the language proxy: a jargon-only row (gene symbols, table fragments) has a ~0
    English function-word share and drops under a positive threshold; the strict < keeps a
    row exactly at the threshold"""
    Filters: RowFilters = RowFilters(min_stop_word_ratio=0.5)
    assert first_drop_reason(("BRCA1 mutation analysis",), Filters) == "min_stop_word_ratio"
    assert first_drop_reason(("the of",), Filters) is None
    assert first_drop_reason(("the BRCA1",), Filters) is None  # 0.5 == threshold keeps


def test_max_symbol_ratio_rule_drops_markup_noise_rows() -> None:
    Filters: RowFilters = RowFilters(max_symbol_ratio=0.5)
    assert first_drop_reason(("a!!",), Filters) == "max_symbol_ratio"
    assert first_drop_reason(("a!",), Filters) is None  # exactly 0.5 keeps


def test_max_upper_ratio_rule_drops_shouting_rows() -> None:
    Filters: RowFilters = RowFilters(max_upper_ratio=0.5)
    assert first_drop_reason(("ASAP NOW",), Filters) == "max_upper_ratio"
    assert first_drop_reason(("aspirin ASAP",), Filters) is None  # exactly 0.5 keeps


def test_max_repeat_ngram_ratio_rule_drops_repetition_spam_rows() -> None:
    Filters: RowFilters = RowFilters(max_repeat_ngram_ratio=0.5)
    assert first_drop_reason((" ".join(["word"] * 20),), Filters) == "max_repeat_ngram_ratio"
    assert first_drop_reason(("the quick brown fox jumps over the lazy dog near",), Filters) is None


def test_max_short_line_ratio_rule_drops_line_break_spam_rows() -> None:
    Filters: RowFilters = RowFilters(max_short_line_ratio=0.5)
    assert first_drop_reason(("short\nlines",), Filters) == "max_short_line_ratio"
    Long: str = "this line is definitely longer than thirty characters\nshort"
    assert first_drop_reason((Long,), Filters) is None  # exactly 0.5 keeps


def test_the_heuristic_reasons_evaluate_in_the_fixed_order() -> None:
    """the first failing heuristic wins the attribution: min_words -> min_stop_word_ratio ->
    max_symbol_ratio -> max_upper_ratio -> max_repeat_ngram_ratio -> max_short_line_ratio;
    the always-on cap still precedes ALL of them and the regexes still follow, so the
    US-009 per-reason counts stay deterministic for a fixed input"""
    All: RowFilters = RowFilters(
        min_words=100,
        min_stop_word_ratio=0.9,
        max_symbol_ratio=0.1,
        max_upper_ratio=0.1,
        max_repeat_ngram_ratio=0.1,
        max_short_line_ratio=0.1,
    )
    assert first_drop_reason(("short row",), All) == "min_words"
    Stop: RowFilters = RowFilters(min_stop_word_ratio=0.9, max_symbol_ratio=0.1, max_upper_ratio=0.1)
    assert first_drop_reason(("BRCA1 !!!",), Stop) == "min_stop_word_ratio"
    Symbol: RowFilters = RowFilters(max_symbol_ratio=0.1, max_upper_ratio=0.1)
    assert first_drop_reason(("the of ASAP !!!",), Symbol) == "max_symbol_ratio"
    Upper: RowFilters = RowFilters(max_upper_ratio=0.1, max_repeat_ngram_ratio=0.1)
    assert first_drop_reason((" ".join(["ASAP"] * 20),), Upper) == "max_upper_ratio"
    Repeat: RowFilters = RowFilters(max_repeat_ngram_ratio=0.1, max_short_line_ratio=0.1)
    assert first_drop_reason((" ".join(["word"] * 20),), Repeat) == "max_repeat_ngram_ratio"
    Last: RowFilters = RowFilters(max_short_line_ratio=0.5)
    assert first_drop_reason(("the of and to in\non at by for with",), Last) == "max_short_line_ratio"
    # the cap outranks the heuristics: an over-cap row never attributes to a heuristic
    assert first_drop_reason(("a" * (MAX_TEXT_TOKENS * CHARS_PER_TOKEN + 1),), RowFilters(min_words=1)) == "max_tokens"
    # and a failing heuristic outranks the regexes: an over-threshold row never attributes
    # to include/exclude even when the regex would also decide the row
    assert first_drop_reason(("aspirin",), RowFilters(min_words=5, include_regex="aspirin")) == "min_words"
    assert first_drop_reason(("BRCA1 mutation analysis",), RowFilters(min_stop_word_ratio=0.5, exclude_regex="zzz")) == "min_stop_word_ratio"


def test_unset_heuristics_never_tokenize_the_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """the efficiency contract: every declared ingest leaves the six knobs unset today, so
    the default path must do ZERO tokenization work (no split, no lines) -- _tokenize is the
    single tokenize chokepoint and a spy proves the unset path never reaches it while a set
    knob does"""
    Calls: list[str] = []
    Original = row_filters_module._tokenize

    def Spy(text: str) -> list[str]:
        Calls.append(text)
        return Original(text)

    monkeypatch.setattr(row_filters_module, "_tokenize", Spy)
    assert first_drop_reason(("aspirin trial",), RowFilters()) is None
    assert first_drop_reason(("aspirin trial",), RowFilters(min_text_len=5)) is None
    assert first_drop_reason(("aspirin trial",), RowFilters(exclude_regex="metformin")) is None
    assert Calls == []
    assert first_drop_reason(("aspirin trial",), RowFilters(min_words=2)) is None
    assert Calls == ["aspirin trial"]


# ------------------------------------------------------------------ envelope plumbing --


def _hf_dataset(filters: RowFilters | None = None, sample_rate: float = 1.0) -> HuggingFaceDataset:
    return HuggingFaceDataset(
        task=ScriptTask(type="script", name="GlinerBiomedScript", outputs=["entities"]),
        source="hf",
        dataset="a/b",
        columns_out=["text"],
        filters=filters,
        sample_rate=sample_rate,
    )


def test_to_stream_args_appends_filters_and_sample_rate_after_the_frozen_payload() -> None:
    """the payload stays the 2-tuple the EXPECTED locks pin; filters and sample_rate ride
    envelope slots that cannot shift a frozen position"""
    Filters: RowFilters = RowFilters(min_text_len=3)

    assert _hf_dataset().to_stream_args() == ("hf", _hf_dataset().to_tuple()[1], None, 1.0)
    assert _hf_dataset(Filters).to_stream_args() == ("hf", _hf_dataset().to_tuple()[1], Filters, 1.0)
    assert _hf_dataset(sample_rate=0.25).to_stream_args() == ("hf", _hf_dataset().to_tuple()[1], None, 0.25)
    # the frozen payload itself never moves, whatever the envelope carries
    assert _hf_dataset(Filters, 0.5).to_tuple() == _hf_dataset().to_tuple()


def test_yaml_ingests_stream_args_round_every_declared_dataset() -> None:
    Ingests: YamlIngests = YamlIngests(datasets=[_hf_dataset(RowFilters(drop_empty=True), 0.5)])

    assert Ingests.stream_args() == (("hf", _hf_dataset().to_tuple()[1], RowFilters(drop_empty=True), 0.5),)
    # generate_tuples keeps the frozen 2-tuple shape the cli parallelism count depends on
    assert Ingests.generate_tuples() == (("hf", _hf_dataset().to_tuple()[1]),)


def test_build_stream_wires_filters_and_sample_rate_keyword_only() -> None:
    Payload: tuple[Any, ...] = (TASK, 1.0, "interventions/interventions.avro")
    Filters: RowFilters = RowFilters(drop_empty=True)

    Stream: LocalAvroDataStream = build_stream("local", Payload, Filters, 0.3)
    assert isinstance(Stream, LocalAvroDataStream)
    assert Stream.filters == Filters
    assert Stream.sample_rate == 0.3
    # the defaults keep every existing positional 2-arg call valid
    assert build_stream("local", Payload).filters is None
    assert build_stream("local", Payload).sample_rate == 1.0


def test_local_dataset_declares_filters_through_yaml_shape() -> None:
    Dataset: LocalAvroDataset = LocalAvroDataset(
        task=ScriptTask(type="script", name="CtkpInterventionsScript", outputs=["entities"]),
        source="local",
        path="~/x.avro",
        filters=RowFilters(max_text_len=10_000),
    )

    assert Dataset.filters is not None and Dataset.filters.max_text_len == 10_000
    assert Dataset.to_stream_args()[2] == Dataset.filters


def test_sample_rate_declared_through_yaml_shape_is_validated_and_out_of_the_payload() -> None:
    """the mixing-ratio knob parses off the declared yaml, bounds-checks, and never enters the
    frozen payload tuple the EXPECTED locks pin"""
    import pytest
    from pydantic import ValidationError

    assert _hf_dataset(sample_rate=0.25).sample_rate == 0.25
    with pytest.raises(ValidationError):
        _hf_dataset(sample_rate=0.0)  # zero would silently drop the whole source; drop the entry instead
    with pytest.raises(ValidationError):
        _hf_dataset(sample_rate=1.5)
