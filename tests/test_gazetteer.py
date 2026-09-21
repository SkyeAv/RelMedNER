from __future__ import annotations

import pytest
from tablassert.biolink import Predicates, Qualifiers

from relmedner import gazetteer
from relmedner.gazetteer import (
    DISABLED_QUALIFIERS,
    MAX_TRIGGER_DISTANCE,
    NEGATION_CUES,
    PREDICATE_TRIGGERS,
    QUALIFIER_RANGES,
    QUALIFIER_TRIGGERS,
    SENTENCE_BREAKS,
    extract_relations,
    find_triggers,
    validate_qualifier_table,
    validate_trigger_table,
)
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


def expected_negated_relation(name: str, head: str, tail: str) -> Relation:
    return Relation(
        name=name,
        fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
        negated=True,
        description=ScriptUtils.predicate_description(name.removeprefix(gazetteer.NEGATION_NAME_PREFIX)),
    )


# ------------------------------------------------------------------------ qualifier subset --

DAKP_QUALIFIERS: frozenset[str] = frozenset(
    {
        "disease_context_qualifier",
        "anatomical_context_qualifier",
        "sex_qualifier",
        "population_context_qualifier",
        "frequency_qualifier",
        "temporal_context_qualifier",
    }
)


def test_the_qualifier_subset_is_exactly_dakps_declared_slots() -> None:
    """DAKP tables/*.yaml declare six nullable qualifiers; the gazetteer ships the same subset"""
    assert set(QUALIFIER_TRIGGERS) == DAKP_QUALIFIERS
    assert set(QUALIFIER_RANGES) == DAKP_QUALIFIERS


def test_every_qualifier_slot_is_a_qualifiers_member() -> None:
    """KGX edges built from these slots must validate against the installed tablassert enum,
    the same enum DAKP's own qualifier tests pin"""
    Values: frozenset[str] = frozenset(slot.value for slot in Qualifiers)
    assert all(slot in Values for slot in QUALIFIER_TRIGGERS)


def test_the_species_slot_is_deliberately_absent() -> None:
    """tablassert marks species_context_qualifier DISABLED_EDGE_FIELDS (never emittable; v12
    disabled its derivation), so the gazetteer must never claim it"""
    assert DISABLED_QUALIFIERS == frozenset({"species_context_qualifier"})
    assert "species_context_qualifier" not in QUALIFIER_TRIGGERS
    Table = dict(QUALIFIER_TRIGGERS) | {"species_context_qualifier": (())}
    with pytest.raises(ValueError, match="tablassert-disabled"):
        validate_qualifier_table(Table)


def test_the_qualifier_phrase_table_is_pinned() -> None:
    """silent phrase drift would change qualifier recall unnoticed, mirroring the predicate pins"""
    assert sum(len(phrases) for phrases in QUALIFIER_TRIGGERS.values()) == 17
    assert QUALIFIER_TRIGGERS["disease_context_qualifier"] == (
        ("in", "patients", "with"),
        ("among", "patients", "with"),
        ("in", "those", "with"),
        ("in", "people", "with"),
        ("in", "subjects", "with"),
        ("in", "individuals", "with"),
        ("in", "patients", "who", "have"),
        ("in", "patients", "having"),
        ("in", "patients", "diagnosed", "with"),
        ("in", "patients", "suffering", "from"),
    )
    assert QUALIFIER_TRIGGERS["anatomical_context_qualifier"] == ()
    assert QUALIFIER_TRIGGERS["sex_qualifier"] == ()
    assert QUALIFIER_TRIGGERS["population_context_qualifier"] == ()
    assert QUALIFIER_TRIGGERS["frequency_qualifier"] == (("twice", "daily"), ("once", "daily"), ("three", "times", "daily"), ("once", "a", "week"))
    assert QUALIFIER_TRIGGERS["temporal_context_qualifier"] == (("during",), ("after", "surgery"), ("following", "surgery"))


def test_the_qualifier_ranges_match_the_dakp_field_map() -> None:
    """DAKP maps AnatomicalEntity/BiologicalSex/PopulationOfIndividualOrganisms mentions to
    context fields; the range groups encode the same typing with biolink ancestors"""
    assert QUALIFIER_RANGES["anatomical_context_qualifier"] == "ANAT"
    assert QUALIFIER_RANGES["sex_qualifier"] == "SEX"
    assert QUALIFIER_RANGES["population_context_qualifier"] == "POP"
    assert QUALIFIER_RANGES["disease_context_qualifier"] == "DIS"
    assert QUALIFIER_RANGES["frequency_qualifier"] is None  # value-style slots carry literal text
    assert QUALIFIER_RANGES["temporal_context_qualifier"] is None


def test_the_negation_cue_table_is_pinned() -> None:
    """closed, word-bounded cue list in DAKP PREVENTION_CUE style; longest phrase wins at scan time"""
    assert len(NEGATION_CUES) == 15
    assert ("no", "evidence", "that") in NEGATION_CUES
    assert ("not", "been", "shown", "to") in NEGATION_CUES
    assert ("failed", "to") in NEGATION_CUES
    assert ("did", "not") in NEGATION_CUES
    assert ("without",) in NEGATION_CUES
    assert ("not",) in NEGATION_CUES


# ------------------------------------------------------------------------- qualifier extraction --


def test_a_negation_cue_reencodes_the_statement_as_not_predicate() -> None:
    """biolink's negated slot: 'if set to true, then the association is negated'; the gazetteer
    reuses RelationFamily's gliner2-safe not_<predicate> encoding instead of a third field"""
    Tokens: list[str] = "Aspirin failed to prevent stroke in women".split()
    Spans: list[tuple[int, int, str]] = [(0, 0, "Drug"), (4, 4, "Disease"), (6, 6, "BiologicalSex")]
    assert extract_relations(Tokens, Spans) == [
        expected_negated_relation("not_preventative_for_condition", "Aspirin", "stroke"),
        expected_relation("sex_qualifier", "stroke", "women"),
    ]


def test_a_statement_without_a_cue_stays_positive() -> None:
    Tokens: list[str] = "Aspirin protects against stroke".split()
    Spans: list[tuple[int, int, str]] = [(0, 0, "Drug"), (3, 3, "Disease")]
    assert extract_relations(Tokens, Spans) == [expected_relation("preventative_for_condition", "Aspirin", "stroke")]


def test_the_longest_cue_wins_over_its_nested_bare_not() -> None:
    """greedy longest-match at scan time means 'did not' is one cue, never two"""
    Tokens: list[str] = "Aspirin is not shown to treat stroke".split()
    Spans: list[tuple[int, int, str]] = [(0, 0, "Drug"), (6, 6, "Disease")]
    Relations = extract_relations(Tokens, Spans)

    assert [relation.name for relation in Relations] == ["not_treats"]
    assert [relation.negated for relation in Relations] == [True]


def test_a_cue_never_crosses_a_sentence_break() -> None:
    """negation is a same-sentence scope; a cue in the previous sentence must not leak"""
    Tokens: list[str] = ["No", "evidence", "that", "aspirin", ".", "statins", "treats", "migraine"]
    Spans: list[tuple[int, int, str]] = [(3, 3, "Drug"), (5, 5, "Drug"), (7, 7, "Disease")]
    assert [relation.name for relation in extract_relations(Tokens, Spans)] == ["treats"]


def test_disease_context_attaches_to_the_statement_tail_host() -> None:
    """DAKP hosts qualifiers on the object/disease mention; tail preference encodes that here"""
    Tokens: list[str] = "Metformin treats type 2 diabetes in patients with chronic kidney disease".split()
    Spans: list[tuple[int, int, str]] = [(0, 0, "Drug"), (2, 4, "Disease"), (9, 10, "Disease"), (12, 14, "AnatomicalEntity")]
    assert extract_relations(Tokens, Spans) == [
        expected_relation("treats", "Metformin", "type 2 diabetes"),
        expected_relation("disease_context_qualifier", "type 2 diabetes", "kidney disease"),
    ]


def test_type_gazetteer_attaches_anatomy_and_sex_contexts() -> None:
    """DAKP's field map is type-driven for anatomy/sex/population (no cue regex); the same
    typed-mention gazetteer applies here whenever a statement already fired"""
    Tokens: list[str] = "Aspirin treats stroke in the myocardium of women".split()
    Spans: list[tuple[int, int, str]] = [(0, 0, "Drug"), (2, 2, "Disease"), (5, 5, "AnatomicalEntity"), (7, 7, "BiologicalSex")]
    assert extract_relations(Tokens, Spans) == [
        expected_relation("treats", "Aspirin", "stroke"),
        expected_relation("anatomical_context_qualifier", "stroke", "myocardium"),
        expected_relation("sex_qualifier", "stroke", "women"),
    ]


def test_qualifiers_only_fire_where_a_statement_already_fired() -> None:
    """biolink: a qualifier is a statement qualifier; without a statement there is nothing to qualify"""
    Tokens: list[str] = "The myocardium and the women were examined".split()
    Spans: list[tuple[int, int, str]] = [(1, 1, "AnatomicalEntity"), (4, 4, "BiologicalSex")]
    assert extract_relations(Tokens, Spans) == []


def test_a_typed_disease_mention_never_becomes_a_bare_context() -> None:
    """DAKP excludes disease mentions from generic qualifier attachment (the object path owns
    them); disease contexts only fire through the patient-template phrases"""
    Tokens: list[str] = "Metformin treats diabetes and comorbid hypertension".split()
    Spans: list[tuple[int, int, str]] = [(0, 0, "Drug"), (2, 2, "Disease"), (5, 5, "Disease")]
    assert extract_relations(Tokens, Spans) == [expected_relation("treats", "Metformin", "diabetes")]


def test_the_restatement_guard_rejects_endpoint_contexts() -> None:
    """DAKP qualifier_restarts_object: a context overlapping a statement endpoint is a
    restatement, not a qualifier"""
    Tokens: list[str] = "Aspirin is associated with women".split()
    Spans: list[tuple[int, int, str]] = [(0, 0, "Drug"), (4, 4, "BiologicalSex")]
    Relations = extract_relations(Tokens, Spans)

    assert [relation.name for relation in Relations] == ["associated_with"]
    assert all(relation.name != "sex_qualifier" for relation in Relations)


def test_value_slots_fall_back_to_the_phrase_surface() -> None:
    """DAKP emits frequency/temporal qualifiers as the cell's literal text; when no entity
    mention follows the phrase, the phrase surface itself is the qualifier value"""
    Tokens: list[str] = "This drug treats diabetes twice daily".split()
    Spans: list[tuple[int, int, str]] = [(1, 1, "Drug"), (3, 3, "Disease")]
    assert extract_relations(Tokens, Spans) == [
        expected_relation("treats", "drug", "diabetes"),
        expected_relation("frequency_qualifier", "diabetes", "twice daily"),
    ]


def test_validate_qualifier_table_rejects_a_non_qualifier_slot() -> None:
    with pytest.raises(ValueError, match="Qualifiers member"):
        validate_qualifier_table({"flavor_qualifier": (("sweet",),)})


def test_validate_qualifier_table_rejects_uppercase_and_duplicate_phrases() -> None:
    with pytest.raises(ValueError, match="uppercase token"):
        validate_qualifier_table({"sex_qualifier": (("Women",),)})
    Table = {"frequency_qualifier": (("twice", "daily"),), "temporal_context_qualifier": (("twice", "daily"),)}
    with pytest.raises(ValueError, match="claimed by both"):
        validate_qualifier_table(Table)


def test_validate_qualifier_table_accepts_empty_phrase_tuples_for_type_gazetteer_slots() -> None:
    """anatomy/sex/population attach by DAKP's type map with no cue regex, so empty is correct"""
    assert validate_qualifier_table(dict.fromkeys(("anatomical_context_qualifier", "sex_qualifier", "population_context_qualifier"), ())) is None
