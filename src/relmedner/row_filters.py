"""pure row-filter evaluation, shared by every DataStream rows() implementation

The evaluator is deliberately a free function over plain values (no I/O, no stream state) so the
drop decision is unit-testable in isolation and US-009 can count drops per reason without rework.
"""

from __future__ import annotations

import re
from typing import Any

from relmedner.constants import CHARS_PER_TOKEN, MAX_TEXT_TOKENS
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
    if filters.include_regex is not None and re.search(filters.include_regex, text) is None:
        return "include_regex"
    if filters.exclude_regex is not None and re.search(filters.exclude_regex, text) is not None:
        return "exclude_regex"
    return None
