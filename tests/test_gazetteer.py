from __future__ import annotations

import pytest
from tablassert.biolink import Predicates

from relmedner.gazetteer import MAX_TRIGGER_DISTANCE, PREDICATE_TRIGGERS, SENTENCE_BREAKS, find_triggers, validate_trigger_table


def test_the_vocabulary_covers_exactly_the_six_starter_predicates() -> None:
    """locks the predicate surface so US-002 consumers see exactly the planned starters and nothing else"""
    assert set(PREDICATE_TRIGGERS) == {"treats", "associated_with", "interacts_with", "causes", "biomarker_for", "expressed_in"}


def test_every_predicate_key_is_a_biolink_predicates_member() -> None:
    """KGX edges built from these keys must validate against the pinned Biolink model"""
    Values: frozenset[str] = frozenset(predicate.value for predicate in Predicates)
    assert all(predicate in Values for predicate in PREDICATE_TRIGGERS)


def test_the_starter_phrases_are_pinned_exactly() -> None:
    """silent phrase drift would change trigger recall with no other test noticing"""
    assert PREDICATE_TRIGGERS["treats"] == (
        ("treats",),
        ("to", "treat"),
        ("is", "used", "to", "treat"),
        ("for", "the", "treatment", "of"),
        ("in", "the", "treatment", "of"),
    )
    assert PREDICATE_TRIGGERS["associated_with"] == (("associated", "with"),)
    assert PREDICATE_TRIGGERS["interacts_with"] == (("interacts", "with"), ("interaction", "with"))
    assert PREDICATE_TRIGGERS["causes"] == (("causes",), ("caused", "by"), ("cause", "of"))
    assert PREDICATE_TRIGGERS["biomarker_for"] == (("biomarker", "for"),)
    assert PREDICATE_TRIGGERS["expressed_in"] == (("expressed", "in"),)


def test_find_triggers_matches_single_and_multi_token_phrases_with_inclusive_ends() -> None:
    """end index is inclusive so US-002 can slice mention spans the same way GLiNER spans do"""
    Tokens: list[str] = ["Aspirin", "treats", "headache", ";", "smoking", "is", "associated", "with", "cancer"]
    assert find_triggers(Tokens) == [(1, 1, "treats"), (6, 7, "associated_with")]


def test_find_triggers_is_case_insensitive() -> None:
    """real GLiNER token streams mix case, so matching must compare token.lower()"""
    assert find_triggers(["WARFARIN", "INTERACTS", "With", "NSAIDs"]) == [(1, 2, "interacts_with")]


def test_find_triggers_returns_every_occurrence_in_left_to_right_order() -> None:
    """US-002 pairs each trigger with nearby mentions, so no occurrence may be dropped after the first"""
    Tokens: list[str] = ["statins", "cause", "of", "myalgia", "?", "injury", "caused", "by", "statins"]
    assert find_triggers(Tokens) == [(1, 2, "causes"), (6, 7, "causes")]


def test_find_triggers_prefers_the_longest_phrase_and_consumes_nested_matches() -> None:
    """ "is used to treat" must swallow its nested "to treat" so one surface fires one trigger"""
    assert find_triggers(["Aspirin", "is", "used", "to", "treat", "pain"]) == [(1, 4, "treats")]


def test_find_triggers_ignores_partial_phrases_at_the_token_stream_end() -> None:
    """a phrase truncated by the sentence end must not fire a bogus short trigger"""
    assert find_triggers(["aspirin", "is", "used", "to"]) == []
    assert find_triggers(["expressed"]) == []


def test_find_triggers_handles_empty_and_triggerless_token_lists() -> None:
    """US-002 runs this on every sentence; empty input must yield an empty list, not an error"""
    assert find_triggers([]) == []
    assert find_triggers(["The", "patient", "improved"]) == []


def test_the_constants_hold_the_contract_values() -> None:
    """US-002's sentence-window logic depends on the exact break set and distance cap"""
    assert isinstance(SENTENCE_BREAKS, frozenset)
    assert SENTENCE_BREAKS == frozenset({".", ";"})
    assert MAX_TRIGGER_DISTANCE == 15


def test_validate_trigger_table_rejects_a_non_biolink_predicate() -> None:
    """a typo'd predicate would emit KGX edges that fail Biolink validation downstream"""
    with pytest.raises(ValueError, match="Predicates member"):
        validate_trigger_table({"heals": (("heals",),)})


def test_validate_trigger_table_rejects_an_empty_phrase() -> None:
    """an empty phrase can never match and only masks a table-authoring bug"""
    with pytest.raises(ValueError, match="empty phrase"):
        validate_trigger_table({"treats": ((), ("treats",))})


def test_validate_trigger_table_rejects_empty_tokens_inside_a_phrase() -> None:
    """empty tokens would slice phantom spans out of the token stream"""
    with pytest.raises(ValueError, match="empty token"):
        validate_trigger_table({"treats": (("to", ""),)})


def test_validate_trigger_table_rejects_uppercase_tokens() -> None:
    """uppercase table entries can never match the lowercased scan input, so they are bugs"""
    with pytest.raises(ValueError, match="uppercase token"):
        validate_trigger_table({"treats": (("Treats",),)})


def test_validate_trigger_table_rejects_one_phrase_claimed_by_two_predicates() -> None:
    """a shared phrase would make the longest-match winner depend on table order"""
    Table = {"treats": (("associated", "with"),), "associated_with": (("associated", "with"),)}
    with pytest.raises(ValueError, match="claimed by both"):
        validate_trigger_table(Table)


def test_validate_trigger_table_accepts_a_minimal_valid_table() -> None:
    """the checker must pass legitimate tables; an always-raising check would be worse than none"""
    assert validate_trigger_table({"treats": (("treats",), ("to", "treat"))}) is None


def test_the_shipped_table_passes_its_own_validation() -> None:
    """the module already ran this at import time; asserting it pins the fail-loudly contract"""
    assert validate_trigger_table(PREDICATE_TRIGGERS) is None
