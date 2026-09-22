from __future__ import annotations

import json
import os
import subprocess
import sys

import apache_beam as beam
import pytest
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.runners.runner import PipelineResult
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from relmedner.constants import DEDUP_BANDS, DEDUP_NUM_PERM, MIN_NEAR_TOKENS
from relmedner.dedup import (
    DEDUP_METRICS_NAMESPACE,
    KeepPriorityWinnerByKey,
    band_keys,
    base_hash,
    exact_key,
    normalize_text,
    priority,
    shingles,
    signature,
)
from relmedner.models import Entity, TrainingExample

# ~100-word toy abstract and a same-length unrelated abstract with zero shared 5-gram shingles.
# Sized so the one-word-swap pair has measured Jaccard 0.979: at seed 42 with 8 bands x 16 rows
# it shares 5 of 8 band keys (S-curve predicts >= 0.999 for s = 0.979), giving the REQ-NEAR-3
# assertions a large deterministic margin instead of a coin flip.
NEAR_BASE_WORDS: tuple[str, ...] = (
    "patients with chronic obstructive pulmonary disease show reduced forced expiratory volume alongside "
    "progressive dyspnea chronic cough and frequent lower respiratory tract infections that worsen during "
    "winter months when viral coverage increases among older adults with a long history of tobacco exposure "
    "and occupational dust inhalation in urban industrial settings according to pulmonary function testing "
    "performed at baseline and repeated annually by the treating clinical team despite inhaled bronchodilator "
    "therapy pulmonary rehabilitation and smoking cessation counseling offered through the regional chest "
    "clinic where arterial blood gases deteriorate gradually and exacerbation rates double within three years"
).split()
UNRELATED_WORDS: tuple[str, ...] = (
    "researchers deployed a transformer encoder pretrained on bibliographic corpora to extract mechanistic "
    "relations between kinase inhibitors and their downstream signaling targets from full text oncology "
    "reports annotating each mention with normalized ontology identifiers confidence scores and sentence "
    "level evidence spans before evaluating the pipeline against a manually curated benchmark of annotated "
    "pathway diagrams where inter annotator agreement exceeded the threshold set during pilot calibration "
    "and ablation studies confirmed that domain adaptive pretraining improved macro averaged extraction "
    "quality across tumor immunology abstracts without any additional labeled training examples whatsoever"
).split()
NEAR_BASE_TEXT: str = " ".join(NEAR_BASE_WORDS)
NEAR_VARIANT_TEXT: str = " ".join(NEAR_BASE_WORDS[:-1] + ["immunotherapy"])  # exactly one shingle changes
UNRELATED_TEXT: str = " ".join(UNRELATED_WORDS)


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


def test_shingles_are_lowercased_word_ngrams() -> None:
    """REQ-NEAR-1: shingles are lowercased word 5-grams of text.split() in document order, and a
    text with fewer than n tokens yields the empty tuple -- which is exactly what makes
    sub-MIN_NEAR_TOKENS texts structurally unable to emit band keys later (they bypass
    near-dedup because a tiny shingle set makes the MinHash Jaccard estimate noise)"""
    assert shingles("The BRCA1 Gene MUTATES", n=2) == ("the brca1", "brca1 gene", "gene mutates")
    assert shingles("TNF Alpha Signaling") == ()  # 3 tokens < default n=5
    assert shingles("one two three four five") == ("one two three four five",)
    assert shingles("one two three four") == ()
    assert shingles("") == ()
    tokens = MIN_NEAR_TOKENS  # a text at the gate has exactly 6 word 5-grams, never 0
    assert len(shingles(" ".join(f"w{index}" for index in range(tokens)))) == tokens - 5 + 1


def test_signature_deterministic_and_identical_text_equal() -> None:
    """REQ-NEAR-2: signature() is byte-identical across calls AND processes: shingle hashes come
    from blake2b and permutations from random.Random(DEDUP_SEED) drawn once at module load,
    never from builtin hash() (CPython salts it per process). The subprocess check with two
    different PYTHONHASHSEED values guards the cross-worker invariant that two Beam workers
    computing the same text must land in the same LSH buckets or near-dedup silently breaks"""
    first = signature(NEAR_BASE_TEXT)
    assert first == signature(NEAR_BASE_TEXT)
    assert len(first) == DEDUP_NUM_PERM == 128
    assert all(0 <= value < 2**61 - 1 for value in first)
    assert base_hash("tnf alpha signaling") == base_hash("tnf alpha signaling")

    probe_code = "import json, sys\nfrom relmedner.dedup import signature\nprint(json.dumps(signature(sys.argv[1])))\n"
    remote = [
        json.loads(
            subprocess.run(
                [sys.executable, "-c", probe_code, NEAR_BASE_TEXT],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                check=True,
                timeout=300,
            ).stdout
        )
        for seed in ("0", "12345")
    ]
    assert remote[0] == remote[1] == list(first)


def test_near_identical_texts_share_a_band_and_dissimilar_do_not() -> None:
    """REQ-NEAR-3: at seed 42, a one-word-swap pair with shingle Jaccard 0.979 collides in at
    least one LSH band (measured: 5 of 8), while an unrelated abstract with zero shared
    shingles shares no band key. This is the precision contract of near-dedup: only "super
    super similar" texts may merge, because every dropped record is supervised signal. Also
    pins band_keys() shape: one key per band, indices 0..DEDUP_BANDS-1, distinct digests,
    loud ValueError on a malformed signature"""
    base_shingles, variant_shingles, unrelated_shingles = (set(shingles(text)) for text in (NEAR_BASE_TEXT, NEAR_VARIANT_TEXT, UNRELATED_TEXT))
    near_jaccard = len(base_shingles & variant_shingles) / len(base_shingles | variant_shingles)
    assert near_jaccard >= 0.9
    assert not (base_shingles & unrelated_shingles)

    base_keys = band_keys(signature(NEAR_BASE_TEXT))
    assert [band for band, _ in base_keys] == list(range(DEDUP_BANDS))
    assert len({digest for _, digest in base_keys}) == DEDUP_BANDS
    variant_keys = band_keys(signature(NEAR_VARIANT_TEXT))
    unrelated_keys = band_keys(signature(UNRELATED_TEXT))
    assert set(base_keys) & set(variant_keys), "near-identical texts must collide in >= 1 band at seed 42"
    assert not (set(base_keys) & set(unrelated_keys)), "base and dissimilar texts must share no band key"
    assert not (set(variant_keys) & set(unrelated_keys)), "variant and dissimilar texts must share no band key"

    with pytest.raises(ValueError, match="rows"):
        band_keys(signature(NEAR_BASE_TEXT)[:10])
