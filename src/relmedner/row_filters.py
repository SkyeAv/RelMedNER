"""pure row-filter evaluation, shared by every DataStream rows() implementation

The evaluator is deliberately a free function over plain values (no I/O, no stream state) so the
drop decision is unit-testable in isolation and US-009 can count drops per reason without rework.
"""

from __future__ import annotations

import re
from typing import Any

from relmedner.constants import CHARS_PER_TOKEN, FUNCTION_WORDS, MAX_TEXT_TOKENS, REPEAT_NGRAM_WORDS, SHORT_LINE_CHARS
from relmedner.models import RowFilters


def joined_text(values: tuple[Any, ...]) -> str:
    """the text rule every length/regex filter applies to: " ".join over the projected payload
    values in order, where a value contributes as itself when it is a str, as its elements joined
    when it is a list/tuple of str (tokenized_text columns), and not at all otherwise. Local avro
    records apply the same rule over the record's values (see LocalAvroDataStream.rows)."""
    parts: list[str] = []
    for value in values:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            parts.append(" ".join(value))
    return " ".join(parts)


def is_empty_value(value: Any) -> bool:
    return value is None or value == "" or (isinstance(value, (list, tuple, dict)) and len(value) == 0)


def first_drop_reason(values: tuple[Any, ...], filters: RowFilters) -> str | None:
    """the FIRST matching drop reason in the fixed order drop_empty -> min_text_len ->
    max_text_len -> max_tokens -> include_regex -> exclude_regex, or None when the row passes.
    match_on is NOT evaluated here: it is exact-value membership applied by the streams before
    this evaluator.

    The text every length/regex rule measures is joined_text(values) (see its docstring for the
    exact contribution rule). It is always computed (once per call) because the max_tokens cap
    is ALWAYS on -- an over-cap row drops even when the declared RowFilters sets no rules; the
    cap sits AFTER the declared length rules so any row that already attributed to max_text_len
    keeps that attribution, and BEFORE the regexes so an over-cap row never attributes to a
    regex."""
    if filters.drop_empty and all(is_empty_value(value) for value in values):
        return "drop_empty"
    text = joined_text(values)
    if filters.min_text_len is not None and len(text) < filters.min_text_len:
        return "min_text_len"
    if filters.max_text_len is not None and len(text) > filters.max_text_len:
        return "max_text_len"
    if len(text) > MAX_TEXT_TOKENS * CHARS_PER_TOKEN:
        return "max_tokens"
    if (
        filters.min_words is not None
        or filters.min_stop_word_ratio is not None
        or filters.max_symbol_ratio is not None
        or filters.max_upper_ratio is not None
        or filters.max_repeat_ngram_ratio is not None
        or filters.max_short_line_ratio is not None
    ):
        # the heuristic block is fully opt-in: when no knob is set (every declared ingest
        # today) the row pays ZERO tokenization cost; when any knob is set the words are
        # tokenized ONCE and shared by every rule (single pass over the _*_core helpers)
        words = _tokenize(text)
        lower: list[str] | None = None
        if filters.min_words is not None and len(words) < filters.min_words:
            return "min_words"
        if filters.min_stop_word_ratio is not None:
            lower = [word.lower() for word in words]
            if _stop_word_ratio_core(lower) < filters.min_stop_word_ratio:
                return "min_stop_word_ratio"
        if filters.max_symbol_ratio is not None and _symbol_ratio_core(text) > filters.max_symbol_ratio:
            return "max_symbol_ratio"
        if filters.max_upper_ratio is not None and _upper_ratio_core(words) > filters.max_upper_ratio:
            return "max_upper_ratio"
        if filters.max_repeat_ngram_ratio is not None:
            lower = [word.lower() for word in words] if lower is None else lower
            if _repeat_ngram_ratio_core(lower, REPEAT_NGRAM_WORDS) > filters.max_repeat_ngram_ratio:
                return "max_repeat_ngram_ratio"
        if filters.max_short_line_ratio is not None and _short_line_ratio_core(text.splitlines()) > filters.max_short_line_ratio:
            return "max_short_line_ratio"
    if filters.include_regex is not None and re.search(filters.include_regex, text) is None:
        return "include_regex"
    if filters.exclude_regex is not None and re.search(filters.exclude_regex, text) is not None:
        return "exclude_regex"
    return None


# ------------------------------------------------------- web-corpus QC heuristics ----
# Pure ratio primitives in the C4 / Gopher document-level filter family. The evaluator
# (first_drop_reason) composes them into opt-in RowFilters drop rules; they are free
# functions over plain text so every threshold in ingests.yaml is set against numbers
# these functions return, and each rule stays independently unit-testable. Each public
# function is a thin text wrapper over a word/line-list core: the evaluator tokenizes a
# row ONCE and feeds the shared lists to every enabled rule's core.


def _tokenize(text: str) -> list[str]:
    """the single tokenize chokepoint of the heuristic rules: the spy-tested guarantee is
    that a row with no heuristic knob set never reaches this function"""
    return text.split()


def word_count(text: str) -> int:
    """plain whitespace tokenization: no punctuation stripping, no case folding"""
    return len(_tokenize(text))


def stop_word_ratio(text: str) -> float:
    """share of lowercase-folded whitespace tokens that are closed-class English function
    words (constants.FUNCTION_WORDS, the pipeline's ONLY stoplist). The cheap deterministic
    language proxy: a non-English or degenerate (gene-symbol list, table) text has almost no
    English function words. FastText language ID was REJECTED for this role: it needs a
    ~130MB pretrained model download, a native-lib dependency, and per-row inference cost."""
    return _stop_word_ratio_core([word.lower() for word in _tokenize(text)])


def _stop_word_ratio_core(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(word in FUNCTION_WORDS for word in words) / len(words)


def symbol_ratio(text: str) -> float:
    """share of NON-SPACE chars that str.isalnum() rejects (markup, code, punctuation noise);
    whitespace is neutral and unicode letters stay alphanumeric, so a non-Latin script is
    NOT a symbol signal here (that is stop_word_ratio's job)"""
    return _symbol_ratio_core(text)


def _symbol_ratio_core(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    return sum(not char.isalnum() for char in chars) / len(chars)


def upper_ratio(text: str) -> float:
    """share of purely alphabetic words that are entirely uppercase (the "shouting caps"
    signal). Tokens with digits or symbols (BRCA1, @TODO) are neither shouted nor normal:
    they leave both numerator and denominator."""
    return _upper_ratio_core(_tokenize(text))


def _upper_ratio_core(words: list[str]) -> float:
    alphabetic = [word for word in words if word.isalpha()]
    if not alphabetic:
        return 0.0
    return sum(word.isupper() for word in alphabetic) / len(alphabetic)


def repeat_ngram_ratio(text: str, n: int = REPEAT_NGRAM_WORDS) -> float:
    """share of case-folded n-word sliding windows that appeared earlier in the text (the
    Gopher-style duplicate-ngram signal). Texts shorter than one window pass with 0.0. The
    counts live in a plain dict keyed by word tuples: dict hashing is deterministic for str,
    so the ratio is byte-stable across processes (builtin hash() stays forbidden exactly as
    in dedup)."""
    return _repeat_ngram_ratio_core([word.lower() for word in _tokenize(text)], n)


def _repeat_ngram_ratio_core(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    windows: dict[tuple[str, ...], int] = {}
    repeated = 0
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        if windows.get(gram):
            repeated += 1
        else:
            windows[gram] = 1
    return repeated / (len(words) - n + 1)


def short_line_ratio(text: str) -> float:
    """share of lines shorter than SHORT_LINE_CHARS (the abnormal-line-breaks signal:
    newline spam, OCR fragments, bullet walls). A single-line text always passes 0.0 --
    there is no line structure to be abnormal -- and splitlines never manufactures a
    phantom empty tail line for a trailing newline."""
    return _short_line_ratio_core(text.splitlines())


def _short_line_ratio_core(lines: list[str]) -> float:
    if len(lines) < 2:
        return 0.0
    return sum(len(line) < SHORT_LINE_CHARS for line in lines) / len(lines)
