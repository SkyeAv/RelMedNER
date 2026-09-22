from __future__ import annotations

import apache_beam as beam
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.runners.runner import PipelineResult
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from relmedner.dedup import DEDUP_METRICS_NAMESPACE, KeepPriorityWinnerByKey, exact_key, normalize_text, priority
from relmedner.models import Entity, TrainingExample


def a_distinguishable_example(text: str, weight: float = 1.0) -> TrainingExample:
    """a record with a payload, so equal-key records are selectable by priority rather than identical"""
    return TrainingExample(text=text, weight=weight, entities=[Entity(label="Drug", mentions=["aspirin"])])


def normalized_text(example: TrainingExample) -> str:
    return normalize_text(example.text)


def dedup_counters(result: PipelineResult) -> dict[str, int]:
    """read the three dedup counters out of a finished pipeline result (absent means zero)"""
    query = result.metrics().query(MetricsFilter().with_namespace(DEDUP_METRICS_NAMESPACE))
    found = {counter.key.metric.name: counter.committed for counter in query["counters"]}
    return {name: found.get(name, 0) for name in ("exact_in", "exact_dropped", "exact_kept")}


def test_exact_key_folds_whitespace_and_preserves_case() -> None:
    """REQ-EXACT-1: normalization collapses whitespace runs and strips the ends while preserving
    case; biomedical acronym casing (TNF vs tnf) is signal, so it must change the key"""
    assert normalize_text("  TNF alpha   levels\nrise\t") == "TNF alpha levels rise"
    assert normalize_text("BRCA1") == "BRCA1"
    assert normalize_text("") == ""
    assert normalize_text("   \n\t ") == ""
    assert exact_key("TNF alpha") == exact_key("  TNF \t\n alpha  ")
    assert exact_key("TNF alpha") != exact_key("tnf alpha")
    assert len(exact_key("TNF alpha")) == 32  # blake2b digest_size=16 bytes -> 32 hex chars


def test_exact_stage_keeps_priority_winner_per_key() -> None:
    """REQ-EXACT-2: mixed-weight duplicates collapse to the highest-weight record regardless of
    arrival order; equal-weight ties break on canonical json, never on first-seen"""
    curated = a_distinguishable_example(" aspirin ", weight=3.0)
    mined = a_distinguishable_example("aspirin", weight=1.0)

    assert priority(curated) < priority(mined)
    assert min([mined, curated], key=priority) is curated
    assert min([curated, mined], key=priority) is curated  # order-independent

    tied_with_payload = a_distinguishable_example("aspirin", weight=2.0)
    tied_bare = TrainingExample(text="aspirin", weight=2.0)
    assert priority(tied_with_payload) != priority(tied_bare)  # canonical json gives a total order
    assert min([tied_bare, tied_with_payload], key=priority) is min([tied_with_payload, tied_bare], key=priority)

    with TestPipeline() as pipeline:
        survivors = pipeline | beam.Create([mined, curated, mined]) | KeepPriorityWinnerByKey()
        assert_that(survivors, equal_to([curated]))


def test_exact_stage_passes_through_blank_text_without_collapsing() -> None:
    """REQ-EXACT-3: blank-normalized records must never collapse into one survivor: every blank
    passes through untouched and the counters reconcile (exact_in == exact_kept, drops == 0)"""
    blank = TrainingExample(text="")
    whitespace_only = TrainingExample(text="  \n\t ")

    pipeline = TestPipeline()
    survivors = pipeline | beam.Create([blank, whitespace_only]) | KeepPriorityWinnerByKey()
    assert_that(survivors, equal_to([blank, whitespace_only]))

    result = pipeline.run()
    result.wait_until_finish()
    assert dedup_counters(result) == {"exact_in": 2, "exact_dropped": 0, "exact_kept": 2}


def test_exact_stage_on_direct_runner_drops_only_duplicates() -> None:
    """REQ-EXACT-4/5: end-to-end on the in-process DirectRunner: whitespace-equal duplicates
    collapse to their priority winner, the casing variant and the blank survive untouched, and
    the counters reconcile exactly to the input count"""
    duplicate_heavy = TrainingExample(text="Patient shows IL2  elevation", weight=2.5)
    duplicate_light = TrainingExample(text="  Patient shows IL2 elevation\n", weight=0.5)
    casing_variant = TrainingExample(text="patient shows IL2 elevation", weight=1.0)
    blank = TrainingExample(text="   ", weight=1.0)

    pipeline = TestPipeline()
    survivors = pipeline | beam.Create([duplicate_light, casing_variant, duplicate_heavy, blank]) | KeepPriorityWinnerByKey()
    assert_that(survivors | beam.Map(normalized_text), equal_to(["", "Patient shows IL2 elevation", "patient shows IL2 elevation"]))

    result = pipeline.run()
    result.wait_until_finish()
    counters = dedup_counters(result)
    assert counters == {"exact_in": 4, "exact_dropped": 1, "exact_kept": 3}
    assert counters["exact_in"] == counters["exact_dropped"] + counters["exact_kept"]
