"""pure trust-evaluator coverage: query strings, grading thresholds, record/source scoring,
weight adjustment (band + soft drop), and per-predicate edge extrapolation"""

from __future__ import annotations

import pytest

from relmedner.constants import TRUST_RANGE, TRUST_RELATION_PARTIAL_HITS, TRUST_RELATION_VERIFIED_HITS
from relmedner.models import Entity, Relation, RelationField, TrainingExample
from relmedner.validators import (
    PARTIAL,
    UNVERIFIED,
    VERIFIED,
    adjust_weight,
    example_queries,
    example_weight,
    grade_relation,
    grade_span,
    record_edge_factor,
    record_trust,
    relation_query,
    source_trust,
    span_query,
)

# ------------------------------------------------------------------------ query strings --


def test_span_query_quotes_both_terms() -> None:
    assert span_query("TNF", "Gene") == '"TNF"[All Fields] AND "Gene"[All Fields]'


def test_relation_query_quotes_head_and_tail() -> None:
    assert relation_query("aspirin", "cancer", "treats") == '"aspirin"[All Fields] AND "cancer"[All Fields] AND "treats"[All Fields]'


def test_relation_query_without_predicate_stays_co_occurrence_only() -> None:
    assert relation_query("aspirin", "cancer", None) == '"aspirin"[All Fields] AND "cancer"[All Fields]'


# ------------------------------------------------------------------------ grading --


def test_span_grading_is_binary() -> None:
    assert grade_span(0) == UNVERIFIED
    assert grade_span(1) == VERIFIED


@pytest.mark.parametrize("hits", [0])
def test_relation_grading_zero_hits_is_unverified(hits: int) -> None:
    assert grade_relation(hits) == UNVERIFIED


def test_relation_grading_partial_band_is_half_credit() -> None:
    assert grade_relation(TRUST_RELATION_PARTIAL_HITS) == PARTIAL
    assert grade_relation(TRUST_RELATION_VERIFIED_HITS - 1) == PARTIAL


def test_relation_grading_at_verified_threshold() -> None:
    assert grade_relation(TRUST_RELATION_VERIFIED_HITS) == VERIFIED


# ------------------------------------------------------------------------ record scoring --


def example_with(entities: list[Entity], relations: list[Relation]) -> TrainingExample:
    return TrainingExample(text="a text", entities=entities, relations=relations)


def test_example_queries_cover_first_mention_and_every_relation() -> None:
    example = example_with(
        [Entity(label="Gene", mentions=["TNF", "tumor necrosis factor"])],
        [Relation(name="treats", fields=[RelationField(name="head", value="aspirin"), RelationField(name="tail", value="cancer")])],
    )
    queries = example_queries(example)
    assert queries == [
        ("span", '"TNF"[All Fields] AND "Gene"[All Fields]', "Gene"),
        ("relation", '"aspirin"[All Fields] AND "cancer"[All Fields] AND "treats"[All Fields]', "treats"),
    ]


def test_entities_without_mentions_contribute_no_query() -> None:
    example = example_with([Entity(label="Gene", mentions=[])], [])
    assert example_queries(example) == []


def test_relations_without_head_or_tail_contribute_no_query() -> None:
    example = example_with([], [Relation(name="treats", fields=[RelationField(name="head", value="x")])])
    assert example_queries(example) == []


def test_classification_only_examples_contribute_no_query() -> None:
    assert example_queries(TrainingExample(text="plain")) == []


def test_record_trust_is_the_mean_verdict_score() -> None:
    assert record_trust([VERIFIED, PARTIAL]) == 0.75


def test_record_trust_none_when_nothing_validatable() -> None:
    assert record_trust([]) is None


def test_source_trust_averages_scored_records_only() -> None:
    assert source_trust([1.0, None, 0.5]) == pytest.approx(0.75)


def test_source_trust_none_when_everything_unscored() -> None:
    assert source_trust([None, None]) is None


# ------------------------------------------------------------------------ weight adjustment --


def test_adjust_weight_full_trust_stamps_the_declared_weight() -> None:
    assert adjust_weight(0.7, 1.0) == pytest.approx(0.7)


def test_adjust_weight_stays_inside_the_lower_band() -> None:
    # trust 0.5 drags below the band -> clamps to the -20% floor
    assert adjust_weight(1.0, 0.5) == pytest.approx(1.0 * (1 - TRUST_RANGE))


def test_adjust_weight_honors_trust_inside_the_band() -> None:
    assert adjust_weight(1.0, 0.9) == pytest.approx(0.9)


def test_adjust_weight_clamps_at_one() -> None:
    # 1.0 * (1 + TRUST_RANGE) = 1.2 would exceed the ceiling
    assert adjust_weight(1.0, 1.0) == pytest.approx(1.0)


def test_adjust_weight_zero_trust_bypasses_the_band_soft_drop() -> None:
    assert adjust_weight(1.0, 0.0) == 0.0


# ------------------------------------------------------------------------ edge extrapolation --


def example_with_relations(*names: str) -> TrainingExample:
    return example_with(
        [],
        [Relation(name=name, fields=[RelationField(name="head", value="h"), RelationField(name="tail", value="t")]) for name in names],
    )


def test_edge_factor_is_one_without_relations() -> None:
    assert record_edge_factor(TrainingExample(text="x"), {"treats": 0.3}) == 1.0


def test_edge_factor_is_one_when_no_predicate_is_flagged() -> None:
    assert record_edge_factor(example_with_relations("precedes"), {"treats": 0.3}) == 1.0


def test_edge_factor_takes_the_weakest_flagged_predicate() -> None:
    assert record_edge_factor(example_with_relations("precedes", "treats"), {"treats": 0.3, "precedes": 0.9}) == 0.3


def test_edge_factor_ignores_cleared_predicates() -> None:
    assert record_edge_factor(example_with_relations("treats", "causes"), {"treats": 0.0}) == 0.0


def test_example_weight_untouched_without_flagged_edges() -> None:
    assert example_weight(example_with_relations("precedes"), 0.7, {"treats": 0.3}) == 0.7


def test_example_weight_scales_directly_no_band() -> None:
    # 0.3 would be BELOW the source band [0.8, 1.0] -- edge trust must NOT be band-clamped
    assert example_weight(example_with_relations("treats"), 1.0, {"treats": 0.3}) == pytest.approx(0.3)


def test_example_weight_zero_factor_soft_drops() -> None:
    assert example_weight(example_with_relations("treats"), 1.0, {"treats": 0.0}) == 0.0


def test_example_weight_never_exceeds_one() -> None:
    assert example_weight(example_with_relations("treats"), 1.0, {"treats": 1.0}) == pytest.approx(1.0)
