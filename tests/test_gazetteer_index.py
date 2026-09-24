"""the first-token phrase index behind gazetteer._scan must reproduce the exhaustive scan"""

from __future__ import annotations

import random

from relmedner import gazetteer
from relmedner.gazetteer import PREDICATE_TRIGGERS, QUALIFIER_TRIGGERS, _scan


def reference_scan(table, tokens):
    """the pre-index implementation verbatim: every phrase tried at every position"""
    lowered = [token.lower() for token in tokens]
    triggers = []
    start = 0
    while start < len(lowered):
        best_length, best_key = 0, None
        for key, phrases in table.items():
            for phrase in phrases:
                length = len(phrase)
                if length > best_length and lowered[start : start + length] == list(phrase):
                    best_length, best_key = length, key
        if best_key is None:
            start += 1
            continue
        triggers.append((start, start + best_length - 1, best_key))
        start += best_length
    return triggers


def test_index_scan_equals_the_exhaustive_scan_on_randomized_trigger_dense_text() -> None:
    """random token streams built from trigger vocabulary (so overlaps, nested and prefix phrases,
    and truncated phrases at the stream end all occur) must scan identically for both tables"""
    rng = random.Random(7)
    for table in (PREDICATE_TRIGGERS, QUALIFIER_TRIGGERS):
        vocab = sorted({token for phrases in table.values() for phrase in phrases for token in phrase} | {"aspirin", "pain", "the"})
        for _ in range(400):
            tokens = [rng.choice(vocab).upper() if rng.random() < 0.1 else rng.choice(vocab) for _ in range(rng.randint(0, 60))]
            assert _scan(table, tokens) == reference_scan(table, tokens)


def test_equal_length_ties_resolve_to_table_order() -> None:
    """two keys owning the same-length phrase at one position: the earlier key wins, as before"""
    table = {"first": (("a", "b"),), "second": (("a", "b"), ("a", "b", "c"))}
    assert _scan(table, ["a", "b"]) == reference_scan(table, ["a", "b"]) == [(0, 1, "first")]
    assert _scan(table, ["a", "b", "c"]) == [(0, 2, "second")]


def test_a_rebuilt_table_gets_a_fresh_index() -> None:
    """configure_gazetteer swaps in a new dict; the index is keyed by table identity, so the new
    table's phrases are live immediately and the old table's index is not reused"""
    old = {"treats": (("treats",),)}
    new = {"treats": (("treats",), ("cures",))}
    assert _scan(old, ["x", "cures", "y"]) == []
    assert _scan(new, ["x", "cures", "y"]) == [(1, 1, "treats")]
    assert gazetteer._PHRASE_INDEXES[id(new)][0] is new


def test_empty_phrase_tuples_never_match() -> None:
    table = {"anatomical_context_qualifier": (), "frequency_qualifier": (("twice", "daily"),)}
    assert _scan(table, ["take", "twice", "daily"]) == [(1, 2, "frequency_qualifier")]
