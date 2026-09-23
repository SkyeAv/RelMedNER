"""trust + trust_edges declaration semantics: parse bounds, shared-key agreement, and the
weight composition the pipeline stamps (source band nudge x per-predicate edge scaling)"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from relmedner.models import (
    HuggingFaceDataset,
    Relation,
    RelationField,
    TrainingExample,
    ValidateTrustConfig,
    YamlIngests,
)
from relmedner.pipeline import weighted
from relmedner.validate import resolve_trust_settings, suggested_yaml
from relmedner.validators import adjust_weight

TASK: dict[str, Any] = {"type": "script", "name": "CtkpInterventionsScript", "outputs": ["entities"]}


def hf(payload: dict) -> HuggingFaceDataset:
    return HuggingFaceDataset.model_validate({"source": "hf", "dataset": "org/repo", "columns_out": ["text"], "task": TASK, **payload})


# ------------------------------------------------------------------------ model validation --


def test_trust_defaults_to_one_and_parses_bounds() -> None:
    assert hf({}).trust == 1.0
    assert hf({"trust": 0.62}).trust == 0.62


@pytest.mark.parametrize("value", [-0.1, 1.01])
def test_trust_outside_the_unit_interval_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError, match="less_than_equal|greater_than_equal"):
        hf({"trust": value})


def test_trust_edges_parses_and_bounds_every_value() -> None:
    assert hf({"trust_edges": {"treats": 0.3}}).trust_edges == {"treats": 0.3}
    with pytest.raises(ValidationError, match="less_than_equal|greater_than_equal"):
        hf({"trust_edges": {"treats": 1.3}})


def test_weight_zero_is_legal_soft_drop() -> None:
    assert hf({"weight": 0.0}).weight == 0.0
    assert TrainingExample(text="x", weight=0.0).weight == 0.0


def test_weight_negative_is_still_rejected() -> None:
    with pytest.raises(ValidationError, match="greater_than_equal"):
        hf({"weight": -0.1})


# ------------------------------------------------------------------------ shared-key rules --


def test_conflicting_source_trust_raises() -> None:
    Ingests = YamlIngests.model_validate(
        {
            "datasets": [
                {"source": "hf", "dataset": "org/repo", "split": "train", "columns_out": ["text"], "task": TASK, "trust": 0.8},
                {"source": "hf", "dataset": "org/repo", "split": "test", "columns_out": ["text"], "task": TASK, "trust": 0.9},
            ]
        }
    )
    with pytest.raises(ValueError, match="conflicting trusts"):
        Ingests.trusts_by_source()


def test_conflicting_edge_trust_for_one_predicate_raises() -> None:
    Ingests = YamlIngests.model_validate(
        {
            "datasets": [
                {"source": "hf", "dataset": "org/repo", "split": "train", "columns_out": ["text"], "task": TASK, "trust_edges": {"treats": 0.3}},
                {"source": "hf", "dataset": "org/repo", "split": "test", "columns_out": ["text"], "task": TASK, "trust_edges": {"treats": 0.4}},
            ]
        }
    )
    with pytest.raises(ValueError, match="conflicting trust for predicate 'treats'"):
        Ingests.trust_edges_by_source()


def test_edge_maps_from_split_siblings_merge() -> None:
    Ingests = YamlIngests.model_validate(
        {
            "datasets": [
                {"source": "hf", "dataset": "org/repo", "split": "train", "columns_out": ["text"], "task": TASK, "trust_edges": {"treats": 0.3}},
                {"source": "hf", "dataset": "org/repo", "split": "test", "columns_out": ["text"], "task": TASK, "trust_edges": {"precedes": 0.9}},
            ]
        }
    )
    assert Ingests.trust_edges_by_source() == {"org/repo": {"treats": 0.3, "precedes": 0.9}}


def test_sources_without_trust_fields_stay_absent_from_the_maps() -> None:
    Ingests = YamlIngests.model_validate({"datasets": [{"source": "hf", "dataset": "org/repo", "columns_out": ["text"], "task": TASK}]})
    assert Ingests.trusts_by_source() == {"org/repo": 1.0}
    assert Ingests.trust_edges_by_source() == {}


# ------------------------------------------------------------------------ stamp composition --


def example_with_treats() -> TrainingExample:
    return TrainingExample(
        text="aspirin treats pain",
        relations=[Relation(name="treats", fields=[RelationField(name="head", value="aspirin"), RelationField(name="tail", value="pain")])],
    )


def test_weighted_untouched_without_edge_trusts() -> None:
    assert weighted(example_with_treats(), 1.0).weight == 1.0
    assert weighted(example_with_treats(), 1.0, {"treats": 1.0}).weight == 1.0


def test_weighted_scales_only_the_flagged_edge() -> None:
    assert weighted(example_with_treats(), 1.0, {"treats": 0.3}).weight == pytest.approx(0.3)


def test_weighted_composes_source_band_with_edge_scaling() -> None:
    # pipeline composition: source trust first (band), then edge factor (direct scale)
    base: float = adjust_weight(1.0, 0.9)
    assert weighted(example_with_treats(), base, {"treats": 0.5}).weight == pytest.approx(0.45)


def test_weighted_edge_zero_soft_drops_the_record() -> None:
    assert weighted(example_with_treats(), 1.0, {"treats": 0.0}).weight == 0.0


# ------------------------------------------------------------------------ suggested yaml --


def test_suggested_yaml_lists_flagged_predicates_only() -> None:
    snippet: str = suggested_yaml(
        [
            {"source": "org/repo", "sampled": 10, "trust": 0.82, "edge_trusts": {"treats": 0.3, "precedes": 1.0}},
        ]
    )
    assert "trust: 0.82" in snippet
    assert "trust_edges:" in snippet
    assert "treats: 0.30" in snippet
    assert "precedes" not in snippet


def test_suggested_yaml_omits_the_edges_block_when_nothing_is_flagged() -> None:
    snippet: str = suggested_yaml([{"source": "org/repo", "sampled": 5, "trust": None, "edge_trusts": {}}])
    assert "trust_edges:" not in snippet
    assert "n/a" in snippet


def test_suggested_yaml_blames_the_failed_queries_when_every_request_errored() -> None:
    """trust is None for two opposite reasons, and the printed snippet must name the right one.
    A real run with a quoted NCBI_API_KEY got `HTTP Error 400` on all 12 queries and the old
    wording told the operator the corpus had nothing to validate, which points at the dataset
    instead of the credential. Verified with the CLI on main, 2026-09-23."""
    snippet: str = suggested_yaml([{"source": "org/repo", "sampled": 2, "trust": None, "edge_trusts": {}, "queries": 12, "errored": 12}])
    assert "all 12 queries errored" in snippet
    assert "NCBI_API_KEY" in snippet
    assert "no entities/relations" not in snippet


def test_suggested_yaml_reports_a_partial_error_storm_that_scored_nothing() -> None:
    """Some queries errored and the rest produced no record score: the snippet must not claim a
    clean nothing-to-validate, because re-running with a working key could still yield a number."""
    snippet: str = suggested_yaml([{"source": "org/repo", "sampled": 2, "trust": None, "edge_trusts": {}, "queries": 12, "errored": 3}])
    assert "3 of 12 queries errored" in snippet


def test_suggested_yaml_keeps_the_corpus_reason_when_nothing_errored() -> None:
    """The classification-only source: zero queries, zero errors, so the honest reason is that
    the sample carried nothing to validate. Also proves summaries without the counters (older
    reports, hand-built dicts) fall back to this wording instead of raising KeyError."""
    with_counters: str = suggested_yaml([{"source": "org/repo", "sampled": 2, "trust": None, "edge_trusts": {}, "queries": 0, "errored": 0}])
    without_counters: str = suggested_yaml([{"source": "org/repo", "sampled": 5, "trust": None, "edge_trusts": {}}])
    for snippet in (with_counters, without_counters):
        assert "no entities/relations to validate" in snippet
        assert "errored" not in snippet


def test_suggested_yaml_is_parseable_yaml_per_source_block() -> None:
    import yaml

    snippet: str = suggested_yaml([{"source": "org/repo", "sampled": 5, "trust": 0.5, "edge_trusts": {"treats": 0.3}}])
    parsed: dict = yaml.safe_load(snippet)
    assert parsed == {"trust": 0.5, "trust_edges": {"treats": 0.3}}


# ------------------------------------------------------------------ x-trust yaml section --


def test_x_trust_section_parses_and_defaults_to_the_constants() -> None:
    Config = ValidateTrustConfig.model_validate({"sample_size": 10, "backend": "firecrawl", "report": "out.jsonl"})
    assert Config.sample_size == 10
    assert Config.backend == "firecrawl"
    assert Config.report == "out.jsonl"
    assert ValidateTrustConfig().sample_size >= 1


def test_unknown_backend_is_a_validation_error() -> None:
    with pytest.raises(ValidationError, match="is not 'pubmed' or 'firecrawl'"):
        ValidateTrustConfig.model_validate({"backend": "duckduckgo"})


def test_trust_settings_precedence_flag_over_yaml_over_constants() -> None:
    Config = ValidateTrustConfig(sample_size=10, backend="firecrawl", report="x.jsonl")
    assert resolve_trust_settings(None, None, None, Config) == (10, "firecrawl", "x.jsonl")
    assert resolve_trust_settings(7, "pubmed", "y.jsonl", Config) == (7, "pubmed", "y.jsonl")
    assert resolve_trust_settings(7, None, None, Config) == (7, "firecrawl", "x.jsonl")
    assert resolve_trust_settings(None, None, None, None) >= (1, "pubmed", "")


def test_x_trust_attaches_to_yaml_ingests_under_the_aliased_key() -> None:
    Ingests = YamlIngests.model_validate(
        {
            "datasets": [],
            "x-trust": {"sample_size": 3, "backend": "pubmed", "report": "r.jsonl"},
        }
    )
    assert Ingests.x_trust is not None
    assert Ingests.x_trust.sample_size == 3
    assert Ingests.generate_tuples() == ()  # x-trust never shifts the dataset payload
