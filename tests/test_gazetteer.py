from __future__ import annotations

import pytest
from tablassert.biolink import Predicates

from relmedner import gazetteer
from relmedner.gazetteer import MAX_TRIGGER_DISTANCE, PREDICATE_TRIGGERS, SENTENCE_BREAKS, extract_relations, find_triggers, validate_trigger_table
from relmedner.models import Relation, RelationField
from relmedner.utils import ScriptUtils

EXPECTED_PREDICATES: frozenset[str] = frozenset(
    {
        "treats",
        "treated_by",
        "preventative_for_condition",
        "causes",
        "caused_by",
        "associated_with",
        "correlated_with",
        "interacts_with",
        "binds",
        "biomarker_for",
        "expressed_in",
        "located_in",
        "decreases_amount_or_activity_of",
        "increases_amount_or_activity_of",
        "has_adverse_event",
        "diagnoses",
        "has_phenotype",
        "part_of",
        "in_taxon",
        "superclass_of",
        "participates_in",
        "precedes",
        "occurs_in",
    }
)


def test_the_vocabulary_covers_exactly_the_planned_predicates() -> None:
    """locks the predicate surface so consumers see exactly the planned starters and nothing else"""
    assert set(PREDICATE_TRIGGERS) == EXPECTED_PREDICATES


def test_the_table_stays_at_the_planned_phrase_count() -> None:
    """the lexicon was mined corpus-wide; silent phrase drift would change trigger recall unnoticed"""
    assert sum(len(phrases) for phrases in PREDICATE_TRIGGERS.values()) == 129


def test_every_predicate_key_is_a_biolink_predicates_member() -> None:
    """KGX edges built from these keys must validate against the pinned Biolink model"""
    Values: frozenset[str] = frozenset(predicate.value for predicate in Predicates)
    assert all(predicate in Values for predicate in PREDICATE_TRIGGERS)


def test_the_starter_phrases_are_pinned_exactly() -> None:
    """silent phrase drift would change trigger recall with no other test noticing;
    direction-bearing phrases are pinned hardest: caused_by and superclass_of were corrected
    away from causes/subclass_of and must never drift back"""
    assert PREDICATE_TRIGGERS["caused_by"] == (
        ("caused", "by"),
        ("induced", "by"),
        ("due", "to"),
        ("secondary", "to"),
        ("resulting", "from"),
        ("attributable", "to"),
    )
    assert (
        (  # and no other predicate may claim the direction-bearing "caused by" phrase
            "caused by",
            "induced by",
            "due to",
        )
        == tuple(" ".join(p) for p in PREDICATE_TRIGGERS["caused_by"][:3])
    )
    assert PREDICATE_TRIGGERS["superclass_of"] == (
        ("such", "as"),
        ("including",),
        ("includes",),
        ("include",),
        ("a", "type", "of"),
        ("a", "form", "of"),
        ("classified", "as"),
    )
    assert PREDICATE_TRIGGERS["treats"] == (
        ("treats",),
        ("to", "treat"),
        ("is", "used", "to", "treat"),
        ("used", "to", "treat"),
        ("for", "the", "treatment", "of"),
        ("in", "the", "treatment", "of"),
        ("therapy", "for"),
        ("effective", "against"),
    )
    assert PREDICATE_TRIGGERS["associated_with"] == (
        ("associated", "with"),
        ("is", "associated", "with"),
        ("are", "associated", "with"),
        ("was", "associated", "with"),
        ("were", "associated", "with"),
        ("linked", "to"),
        ("in", "association", "with"),
    )
    assert PREDICATE_TRIGGERS["interacts_with"] == (
        ("interacts", "with"),
        ("interaction", "with"),
        ("interacting", "with"),
        ("interactions", "with"),
    )
    assert PREDICATE_TRIGGERS["causes"] == (
        ("causes",),
        ("cause", "of"),
        ("induces",),
        ("leads", "to"),
        ("leading", "to"),
        ("results", "in"),
        ("resulting", "in"),
        ("triggers",),
    )
    assert PREDICATE_TRIGGERS["biomarker_for"] == (
        ("biomarker", "for"),
        ("biomarkers", "for"),
        ("marker", "for"),
        ("predictor", "of"),
        ("indicative", "of"),
    )
    assert PREDICATE_TRIGGERS["expressed_in"] == (
        ("expressed", "in"),
        ("expression", "in"),
        ("is", "expressed", "in"),
        ("are", "expressed", "in"),
        ("overexpressed", "in"),
    )


def test_find_triggers_matches_single_and_multi_token_phrases_with_inclusive_ends() -> None:
    """end index is inclusive so US-002 can slice mention spans the same way GLiNER spans do;
    the longest phrase wins, so "is associated with" beats its nested "associated with"""
    Tokens: list[str] = ["Aspirin", "treats", "headache", ";", "smoking", "is", "associated", "with", "cancer"]
    assert find_triggers(Tokens) == [(1, 1, "treats"), (5, 7, "associated_with")]


def test_find_triggers_is_case_insensitive() -> None:
    """real GLiNER token streams mix case, so matching must compare token.lower()"""
    assert find_triggers(["WARFARIN", "INTERACTS", "With", "NSAIDs"]) == [(1, 2, "interacts_with")]


def test_find_triggers_returns_every_occurrence_in_left_to_right_order() -> None:
    """US-002 pairs each trigger with nearby mentions, so no occurrence may be dropped after the first;
    "caused by" is the reversed direction of "causes" and must map to caused_by, never causes"""
    Tokens: list[str] = ["statins", "cause", "of", "myalgia", "?", "injury", "caused", "by", "statins"]
    assert find_triggers(Tokens) == [(1, 2, "causes"), (6, 7, "caused_by")]


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


def test_extract_relations_emits_a_basic_treats_pair() -> None:
    """the happy path must produce one treats relation joining the nearest mentions by surface"""
    Tokens: list[str] = ["Aspirin", "is", "used", "to", "treat", "migraine"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (5, 5, "Disease")]
    assert extract_relations(Tokens, Spans) == [expected_relation("treats", "Aspirin", "migraine")]


def test_extract_relations_skips_triggers_with_no_head_mention() -> None:
    """a trigger before any mention has no argument to bind, so it must emit nothing"""
    Tokens: list[str] = ["treats", "migraine"]
    Spans: list[tuple[int, int, str]] = [(1, 1, "Disease")]
    assert extract_relations(Tokens, Spans) == []


def test_extract_relations_skips_triggers_with_no_tail_mention() -> None:
    """a trigger after every mention has no second argument, so it must emit nothing"""
    Tokens: list[str] = ["Aspirin", "treats"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity")]
    assert extract_relations(Tokens, Spans) == []


def test_extract_relations_skips_when_a_break_sits_between_head_and_trigger() -> None:
    """a sentence break severs the head from its trigger, so the pair must not cross it"""
    Tokens: list[str] = ["Aspirin", ".", "treats", "migraine"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (3, 3, "Disease")]
    assert extract_relations(Tokens, Spans) == []


def test_extract_relations_skips_when_a_break_sits_between_trigger_and_tail() -> None:
    """a sentence break severs the trigger from its tail, so the pair must not cross it"""
    Tokens: list[str] = ["Aspirin", "treats", ".", "migraine"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (3, 3, "Disease")]
    assert extract_relations(Tokens, Spans) == []


def test_extract_relations_skips_when_the_head_window_exceeds_max_trigger_distance() -> None:
    """a head starting more than fifteen tokens before its trigger is too far to be its argument"""
    Tokens: list[str] = ["Aspirin", *["fill"] * 15, "treats", "migraine"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (17, 17, "Disease")]
    assert extract_relations(Tokens, Spans) == []


def test_extract_relations_skips_when_the_tail_window_exceeds_max_trigger_distance() -> None:
    """a tail starting more than fifteen tokens after its trigger is too far to be its argument"""
    Tokens: list[str] = ["Aspirin", "treats", *["fill"] * 16, "migraine"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (18, 18, "Disease")]
    assert extract_relations(Tokens, Spans) == []


def test_extract_relations_keeps_mentions_exactly_at_the_window_boundary() -> None:
    """the distance cap is inclusive: a gap of exactly fifteen tokens on either side still binds"""
    Tokens: list[str] = ["Aspirin", *["fill"] * 14, "treats", "migraine"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (16, 16, "Disease")]
    assert extract_relations(Tokens, Spans) == [expected_relation("treats", "Aspirin", "migraine")]


def test_extract_relations_keeps_a_tail_exactly_at_the_window_boundary() -> None:
    """the tail cap is inclusive: a gap of exactly fifteen tokens after the trigger end still binds"""
    Tokens: list[str] = ["Aspirin", "treats", *["fill"] * 15, "migraine"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (17, 17, "Disease")]
    assert extract_relations(Tokens, Spans) == [expected_relation("treats", "Aspirin", "migraine")]


def test_extract_relations_skips_a_tail_one_past_the_window_boundary() -> None:
    """a tail starting sixteen tokens after the trigger end is dropped, mirroring the head guard"""
    Tokens: list[str] = ["Aspirin", "treats", *["fill"] * 16, "migraine"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (18, 18, "Disease")]
    assert extract_relations(Tokens, Spans) == []


def test_extract_relations_skips_punctuation_only_head_surfaces() -> None:
    """a period span is a tokenizer artifact, not an entity, so it must never become a head"""
    Tokens: list[str] = ["Aspirin", ".", "treats", "migraine"]
    Spans: list[tuple[int, int, str]] = [(1, 1, "Disease"), (3, 3, "Disease")]
    assert extract_relations(Tokens, Spans) == []


def test_extract_relations_skips_punctuation_only_head_adjacent_to_trigger() -> None:
    """an adjacent punctuation-only head has no break token in the gap, so only the surface guard can reject it"""
    Tokens: list[str] = ["Aspirin", "...", "treats", "migraine"]
    Spans: list[tuple[int, int, str]] = [(1, 1, "Disease"), (3, 3, "Disease")]
    assert extract_relations(Tokens, Spans) == []


def test_extract_relations_skips_punctuation_only_tail_surfaces() -> None:
    """a period span must never become a tail either, mirroring the head guard"""
    Tokens: list[str] = ["Aspirin", "treats", ".", "migraine"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (2, 2, "Disease")]
    assert extract_relations(Tokens, Spans) == []


def test_extract_relations_skips_punctuation_only_tail_adjacent_to_trigger() -> None:
    """an adjacent punctuation-only tail has no break token in the gap, so only the surface guard can reject it"""
    Tokens: list[str] = ["Aspirin", "treats", "(", "migraine"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (2, 2, "Disease")]
    assert extract_relations(Tokens, Spans) == []


def test_extract_relations_deduplicates_identical_predicate_head_tail_triples() -> None:
    """two identical triggers binding the same spans describe one fact, so only the first survives"""
    Tokens: list[str] = ["smoking", "causes", "causes", "cancer"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "Behavior"), (3, 3, "Disease")]
    assert extract_relations(Tokens, Spans) == [expected_relation("causes", "smoking", "cancer")]


def test_extract_relations_deduplicates_same_endpoint_coordinates_across_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolver category changes must not create duplicate facts for identical endpoint coordinates"""
    Tokens: list[str] = ["aspirin", "treats", "treats", "migraine"]
    Heads = iter([(0, 0, "Drug"), (0, 0, "ChemicalEntity")])
    Tails = iter([(3, 3, "Disease"), (3, 3, "PhenotypicFeature")])
    monkeypatch.setattr(gazetteer, "_nearest_before", lambda spans, trigger_start: next(Heads))
    monkeypatch.setattr(gazetteer, "_nearest_after", lambda spans, trigger_end: next(Tails))

    assert extract_relations(Tokens, []) == [expected_relation("treats", "aspirin", "migraine")]


def test_extract_relations_picks_the_nearest_overlapping_mention_on_each_side() -> None:
    """among overlapping candidates the closest span wins: greatest end before, smallest start after;
    labels respect biolink domain/range (treats needs a chemical-ish head and a disease-ish tail)"""
    Tokens: list[str] = ["alpha", "fill", "beta", "gamma", "treats", "delta", "fill", "epsilon"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "Disease"), (2, 3, "ChemicalEntity"), (5, 5, "Disease"), (7, 7, "ChemicalEntity")]
    assert extract_relations(Tokens, Spans) == [expected_relation("treats", "beta gamma", "delta")]


def test_extract_relations_matches_triggers_case_insensitively_and_preserves_surfaces() -> None:
    """trigger matching ignores case while the emitted surfaces keep the original casing"""
    Tokens: list[str] = ["ASPIRIN", "TREATS", "MIGRAINE"]
    Spans: list[tuple[int, int, str]] = [(0, 0, "ChemicalEntity"), (2, 2, "Disease")]
    assert extract_relations(Tokens, Spans) == [expected_relation("treats", "ASPIRIN", "MIGRAINE")]


def expected_relation(name: str, head: str, tail: str) -> Relation:
    """emitted relations carry their biolink slot description; build the matching expectation"""
    return Relation(
        name=name,
        fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
        description=ScriptUtils.predicate_description(name),
    )
