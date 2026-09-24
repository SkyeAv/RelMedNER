from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import apache_beam as beam
import pytest
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.runners.runner import PipelineResult
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from relmedner.constants import DEDUP_BANDS, DEDUP_NUM_PERM, MIN_NEAR_TOKENS
from relmedner.dedup import (
    DEDUP_METRICS_NAMESPACE,
    KeepPriorityWinnerByKey,
    NearDeduplicate,
    _EmitBandKeys,
    _JoinPayloads,
    _TagBandLosers,
    apply_dedup,
    band_keys,
    base_hash,
    content_id,
    exact_key,
    format_dedup_summary,
    normalize_text,
    priority,
    shingles,
    signature,
)
from relmedner.enums import DedupMode
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


def near_counters(result: PipelineResult) -> dict[str, int]:
    """read the four near-dedup counters out of a finished pipeline result (absent means zero)"""
    query = result.metrics().query(MetricsFilter().with_namespace(DEDUP_METRICS_NAMESPACE))
    found = {counter.key.metric.name: counter.committed for counter in query["counters"]}
    names = ("near_in", "near_dropped", "near_kept", "near_buckets_nontrivial")
    return {name: found.get(name, 0) for name in names}


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
    arrival order; equal-weight ties break on content_id, never on first-seen"""
    curated = a_distinguishable_example(" aspirin ", weight=3.0)
    mined = a_distinguishable_example("aspirin", weight=1.0)

    assert priority(curated) < priority(mined)
    assert min([mined, curated], key=priority) is curated
    assert min([curated, mined], key=priority) is curated  # order-independent

    tied_with_payload = a_distinguishable_example("aspirin", weight=2.0)
    tied_bare = TrainingExample(text="aspirin", weight=2.0)
    assert priority(tied_with_payload) != priority(tied_bare)  # content_id gives a total order
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


def test_near_stage_drops_lower_priority_and_survives_dissimilar() -> None:
    """REQ-NEAR-4: of three records where A~B (shingle Jaccard 0.979, sharing 5 of 8 band keys
    at seed 42) with different weights and C dissimilar (sharing no band key with either), the
    lower-priority of A/B drops exactly once and C survives untouched"""
    base_low = a_distinguishable_example(NEAR_BASE_TEXT, weight=1.0)
    variant_high = a_distinguishable_example(NEAR_VARIANT_TEXT, weight=3.0)
    unrelated = a_distinguishable_example(UNRELATED_TEXT, weight=2.0)

    pipeline = TestPipeline()
    survivors = pipeline | beam.Create([base_low, variant_high, unrelated]) | NearDeduplicate()
    assert_that(survivors | beam.Map(normalized_text), equal_to([NEAR_VARIANT_TEXT, UNRELATED_TEXT]))

    result = pipeline.run()
    result.wait_until_finish()
    counters = near_counters(result)
    assert counters == {"near_in": 3, "near_dropped": 1, "near_kept": 2, "near_buckets_nontrivial": 5}
    assert counters["near_in"] == counters["near_dropped"] + counters["near_kept"]


def test_near_stage_counters_reconcile_including_all_bypass_input() -> None:
    """REQ-NEAR-5/6: near_in == near_dropped + near_kept on every input. All-bypass input
    (texts under MIN_NEAR_TOKENS tokens, including blanks) yields near_dropped == 0 and
    near_buckets_nontrivial == 0: short texts never reach a band bucket. A mixed input of
    short + near-pair + dissimilar reconciles to the full count too"""
    short_one = a_distinguishable_example("aspirin", weight=1.0)
    blank = TrainingExample(text="")
    whitespace_only = TrainingExample(text="  \n\t ")

    pipeline = TestPipeline()
    survivors = pipeline | beam.Create([short_one, blank, whitespace_only]) | NearDeduplicate()
    assert_that(survivors | beam.Map(normalized_text), equal_to(["aspirin", "", ""]))

    result = pipeline.run()
    result.wait_until_finish()
    counters = near_counters(result)
    assert counters == {"near_in": 3, "near_dropped": 0, "near_kept": 3, "near_buckets_nontrivial": 0}
    assert counters["near_in"] == counters["near_dropped"] + counters["near_kept"]

    mixed_pipeline = TestPipeline()
    mixed = (
        mixed_pipeline
        | beam.Create(
            [short_one, blank, a_distinguishable_example(NEAR_BASE_TEXT, weight=1.0), a_distinguishable_example(NEAR_VARIANT_TEXT, weight=3.0)]
        )
        | NearDeduplicate()
    )
    assert_that(mixed | beam.Map(normalized_text), equal_to(["aspirin", "", NEAR_VARIANT_TEXT]))

    mixed_result = mixed_pipeline.run()
    mixed_result.wait_until_finish()
    mixed_counters = near_counters(mixed_result)
    assert mixed_counters["near_in"] == 4
    assert mixed_counters["near_dropped"] == 1
    assert mixed_counters["near_kept"] == 3
    assert mixed_counters["near_in"] == mixed_counters["near_dropped"] + mixed_counters["near_kept"]


def test_near_stage_winner_that_loses_another_band_does_not_resurrect() -> None:
    """REQ-NEAR-6: base (weight 1) loses the 5 bands it shares with variant (weight 3) but WINS
    its 3 remaining singleton bands -- if per-band winners were unioned without the global
    exact-key collapse it would reappear. The collapse sees its lose tags and suppresses it;
    variant, which wins all 8 of its bands, is emitted exactly once (near_kept == 1, not 8)"""
    base_low = a_distinguishable_example(NEAR_BASE_TEXT, weight=1.0)
    variant_high = a_distinguishable_example(NEAR_VARIANT_TEXT, weight=3.0)

    pipeline = TestPipeline()
    survivors = pipeline | beam.Create([base_low, variant_high]) | NearDeduplicate()
    assert_that(survivors, equal_to([variant_high]))

    result = pipeline.run()
    result.wait_until_finish()
    counters = near_counters(result)
    assert counters == {"near_in": 2, "near_dropped": 1, "near_kept": 1, "near_buckets_nontrivial": 5}
    assert counters["near_in"] == counters["near_dropped"] + counters["near_kept"]


# ---------------------------------------------------------------- mode switch (REQ-INT-2) --


def test_apply_dedup_off_is_identity() -> None:
    """REQ-INT-2: OFF returns the pcollection UNCHANGED, so no dedup transform enters the
    graph at all: exact dups AND the Jaccard-0.979 near-pair all survive, and every dedup
    counter stays at zero (absent from the finished run's metrics)"""
    heavy = a_distinguishable_example("aspirin", weight=3.0)
    light = a_distinguishable_example("  aspirin ", weight=1.0)
    base = a_distinguishable_example(NEAR_BASE_TEXT, weight=1.0)
    variant = a_distinguishable_example(NEAR_VARIANT_TEXT, weight=3.0)
    examples = [heavy, light, base, variant]

    pipeline = TestPipeline()
    created = pipeline | beam.Create(examples)
    deduped = apply_dedup(created, DedupMode.OFF)
    assert deduped is created  # identity: the same PCollection object, no stage applied
    assert_that(deduped, equal_to(examples))

    result = pipeline.run()
    result.wait_until_finish()
    assert dedup_counters(result) == {"exact_in": 0, "exact_dropped": 0, "exact_kept": 0}
    assert near_counters(result)["near_in"] == 0


def test_apply_dedup_exact_runs_exact_stage_only() -> None:
    """REQ-INT-2: EXACT adds the exact stage only: the whitespace dup collapses to its
    priority winner, while the near-pair survives as TWO records and near_in stays 0 (the
    near stage never entered the graph)"""
    heavy = a_distinguishable_example("aspirin", weight=3.0)
    light = a_distinguishable_example("  aspirin ", weight=1.0)
    base = a_distinguishable_example(NEAR_BASE_TEXT, weight=1.0)
    variant = a_distinguishable_example(NEAR_VARIANT_TEXT, weight=3.0)

    pipeline = TestPipeline()
    created = pipeline | beam.Create([light, heavy, base, variant])
    survivors = apply_dedup(created, DedupMode.EXACT)
    assert_that(survivors | beam.Map(normalized_text), equal_to(["aspirin", NEAR_BASE_TEXT, NEAR_VARIANT_TEXT]))

    result = pipeline.run()
    result.wait_until_finish()
    assert dedup_counters(result) == {"exact_in": 4, "exact_dropped": 1, "exact_kept": 3}
    assert near_counters(result)["near_in"] == 0


def test_apply_dedup_near_runs_exact_then_near() -> None:
    """REQ-INT-2: NEAR chains exact then near end-to-end: the whitespace dup collapses in the
    exact stage, then the near-pair collapses to its higher-weight winner, and the dissimilar
    record survives untouched; every counter reconciles (in == dropped + kept)"""
    heavy = a_distinguishable_example("aspirin", weight=3.0)
    light = a_distinguishable_example("  aspirin ", weight=1.0)
    base_low = a_distinguishable_example(NEAR_BASE_TEXT, weight=1.0)
    variant_high = a_distinguishable_example(NEAR_VARIANT_TEXT, weight=3.0)
    unrelated = a_distinguishable_example(UNRELATED_TEXT, weight=2.0)

    pipeline = TestPipeline()
    created = pipeline | beam.Create([light, heavy, base_low, variant_high, unrelated])
    survivors = apply_dedup(created, DedupMode.NEAR)
    assert_that(survivors | beam.Map(normalized_text), equal_to(["aspirin", NEAR_VARIANT_TEXT, UNRELATED_TEXT]))

    result = pipeline.run()
    result.wait_until_finish()
    exact = dedup_counters(result)
    assert exact == {"exact_in": 5, "exact_dropped": 1, "exact_kept": 4}
    near = near_counters(result)
    assert near["near_in"] == 4
    assert near["near_dropped"] == 1
    assert near["near_kept"] == 3
    assert near["near_in"] == near["near_dropped"] + near["near_kept"]


# ---------------------------------------------------------------- summary line (REQ-INT-4) --


def test_summary_line_reports_counts() -> None:
    """REQ-INT-4: a finished NEAR run renders exactly the spec's line from the retained
    PipelineResult metrics: one exact drop, one near drop, and total = exact_in (the count
    entering dedup -- near_in is deliberately NOT the total because the near stage only sees
    the exact stage's survivors). The same figures the counter readers above report, so the
    line cannot drift from the counters"""
    heavy = a_distinguishable_example("aspirin", weight=3.0)
    light = a_distinguishable_example("  aspirin ", weight=1.0)
    base_low = a_distinguishable_example(NEAR_BASE_TEXT, weight=1.0)
    variant_high = a_distinguishable_example(NEAR_VARIANT_TEXT, weight=3.0)
    unrelated = a_distinguishable_example(UNRELATED_TEXT, weight=2.0)

    pipeline = TestPipeline()
    created = pipeline | beam.Create([light, heavy, base_low, variant_high, unrelated])
    survivors = apply_dedup(created, DedupMode.NEAR)
    assert_that(survivors | beam.Map(normalized_text), equal_to(["aspirin", NEAR_VARIANT_TEXT, UNRELATED_TEXT]))

    result = pipeline.run()
    result.wait_until_finish()
    assert format_dedup_summary(result) == "dedup: exact -1 near -1 of 5 records"


def test_summary_line_total_is_exact_in_not_near_in() -> None:
    """REQ-INT-4: the `of <total>` figure must be the count entering dedup, not the near
    stage's own input -- otherwise an input where the exact stage already dropped something
    would report a total smaller than the record count. EXACT mode here drops one dup before
    near ever runs, so exact_in == 4 while near_in == 3, and the line must say 4"""
    heavy = a_distinguishable_example("aspirin", weight=3.0)
    light = a_distinguishable_example("  aspirin ", weight=1.0)
    base_low = a_distinguishable_example(NEAR_BASE_TEXT, weight=1.0)
    variant_high = a_distinguishable_example(NEAR_VARIANT_TEXT, weight=3.0)
    unrelated = a_distinguishable_example(UNRELATED_TEXT, weight=2.0)

    pipeline = TestPipeline()
    created = pipeline | beam.Create([light, heavy, base_low, variant_high, unrelated])
    survivors = apply_dedup(created, DedupMode.NEAR)
    assert_that(survivors | beam.Map(normalized_text), equal_to(["aspirin", NEAR_VARIANT_TEXT, UNRELATED_TEXT]))

    result = pipeline.run()
    result.wait_until_finish()
    assert dedup_counters(result)["exact_in"] == 5
    assert near_counters(result)["near_in"] == 4  # one exact dup already gone
    assert format_dedup_summary(result) == "dedup: exact -1 near -1 of 5 records"


def test_summary_line_absent_when_dedup_off() -> None:
    """REQ-INT-4: OFF mode puts no dedup stage in the graph, so no relmedner.dedup counter
    exists in the finished run's metrics and the helper returns None -- the caller then logs
    no line at all (no zeros line), matching the spec's 'zeros or no line' allowance. A
    missing result (None) reports the same way"""
    heavy = a_distinguishable_example("aspirin", weight=3.0)
    light = a_distinguishable_example("  aspirin ", weight=1.0)

    pipeline = TestPipeline()
    created = pipeline | beam.Create([light, heavy])
    deduped = apply_dedup(created, DedupMode.OFF)
    assert_that(deduped, equal_to([light, heavy]))

    result = pipeline.run()
    result.wait_until_finish()
    assert format_dedup_summary(result) is None
    assert format_dedup_summary(None) is None


def _reference_signature(text: str) -> tuple[int, ...]:
    """the pre-numpy implementation verbatim: python ints, exact modular arithmetic"""
    from relmedner.dedup import _MERSENNE_PRIME, _PERM_A, _PERM_B, base_hash

    hashes = [base_hash(shingle) for shingle in shingles(text)]
    if not hashes:
        return signature("")
    return tuple(min((a * h + b) % _MERSENNE_PRIME for h in hashes) for a, b in zip(_PERM_A, _PERM_B, strict=True))


def test_vectorized_signature_is_byte_identical_to_the_python_reference() -> None:
    """signatures decide every near-dedup drop, and a uint64 overflow in the limb arithmetic
    would change them silently; random texts (short, long, repeated shingles) must match the
    python-int reference exactly, value for value"""
    import random

    rng = random.Random(11)
    vocab = [f"w{index}" for index in range(300)]
    texts = ["", "one two three", " ".join(["same"] * 40)]
    texts += [" ".join(rng.choice(vocab) for _ in range(rng.randint(5, 1600))) for _ in range(60)]
    for text in texts:
        assert signature(text) == _reference_signature(text)


def test_affine_mod_is_exact_at_the_modulus_edges() -> None:
    """hashes near 2**64 and 2**61 exercise every fold and the conditional subtract"""
    import numpy as np

    from relmedner.dedup import _MERSENNE_PRIME, _PERM_A, _PERM_B, _affine_mod

    edges = [0, 1, _MERSENNE_PRIME - 1, _MERSENNE_PRIME, _MERSENNE_PRIME + 1, (1 << 62) + 5, (1 << 64) - 1, (1 << 64) - 2]
    got = _affine_mod(np.array(edges, dtype=np.uint64))
    for j, (a, b) in enumerate(zip(_PERM_A, _PERM_B, strict=True)):
        assert [int(value) for value in got[j]] == [(a * h + b) % _MERSENNE_PRIME for h in edges]


# ------------------------------------------- compact band shuffle (content_id + payload join) ----


def test_content_id_is_a_stable_fingerprint_of_the_whole_payload() -> None:
    """content_id names a record in the band shuffle without shipping it, so it must be a pure
    function of the canonical json: equal payloads share an id, any payload difference does not,
    and it is stable across processes (two workers must agree or the join drops records)"""
    base = a_distinguishable_example(NEAR_BASE_TEXT)
    same = a_distinguishable_example(NEAR_BASE_TEXT)
    other = a_distinguishable_example(NEAR_BASE_TEXT, weight=0.5)

    assert content_id(base) == content_id(same)
    assert content_id(base) != content_id(other)
    assert len(content_id(base)) == 32  # blake2b digest_size=16 -> 32 hex chars
    assert priority(base) == (-base.weight, content_id(base))

    probe_code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from tests.test_dedup import a_distinguishable_example, NEAR_BASE_TEXT\n"
        "from relmedner.dedup import content_id\n"
        "print(content_id(a_distinguishable_example(NEAR_BASE_TEXT)))\n"
    )
    repo_root = str(Path(__file__).resolve().parents[1])
    remote = subprocess.run(
        [sys.executable, "-c", probe_code, repo_root],
        env={**os.environ, "PYTHONHASHSEED": "999"},
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    ).stdout.strip()
    assert remote == content_id(base)


def test_the_band_stream_carries_priorities_and_the_payload_rides_one_tagged_output() -> None:
    """the whole point of the restructure: a long record emits DEDUP_BANDS compact
    (band_key, priority) elements and exactly ONE payload element, so its TrainingExample crosses
    the shuffle once instead of twice per band; a short record emits no band element at all"""
    emit = _EmitBandKeys()
    long_record = a_distinguishable_example(NEAR_BASE_TEXT)
    outputs = list(emit.process(long_record))

    bands = [item for item in outputs if not isinstance(item, beam.pvalue.TaggedOutput)]
    payloads = [item for item in outputs if isinstance(item, beam.pvalue.TaggedOutput) and item.tag == "payload"]
    assert len(bands) == DEDUP_BANDS
    assert all(value == priority(long_record) for _key, value in bands)
    assert {key for key, _ in bands} == set(band_keys(signature(NEAR_BASE_TEXT)))
    assert [(item.tag, item.value) for item in payloads] == [("payload", (content_id(long_record), long_record))]

    short_record = a_distinguishable_example("aspirin")
    short_outputs = list(emit.process(short_record))
    assert [(item.tag, item.value) for item in short_outputs] == [("bypass", short_record)]


def test_band_losers_are_marked_once_per_losing_bucket_and_winners_emit_nothing() -> None:
    """a bucket winner ships no marker, so a bucket whose records all survive costs nothing
    downstream; every non-winner ships exactly one marker per bucket it loses"""
    high = priority(a_distinguishable_example(NEAR_BASE_TEXT, weight=3.0))
    low = priority(a_distinguishable_example(NEAR_VARIANT_TEXT, weight=1.0))
    tagger = _TagBandLosers()

    assert list(tagger.process(((0, "k"), [high, low]))) == [(low[1], None)]
    assert list(tagger.process(((0, "k"), [low, high]))) == [(low[1], None)]  # order-independent
    assert list(tagger.process(((0, "k"), [high]))) == []
    assert list(tagger.process(((0, "k"), [high, high]))) == []  # identical priorities tie, no loser


def test_the_payload_join_drops_a_record_that_lost_any_band_and_emits_survivors_once() -> None:
    """REQ-NEAR-4/6 at the join: one loser marker is enough to drop the record no matter how many
    bands it won, a record with no marker is emitted exactly once, and a marker with no payload
    neither emits nor counts"""
    survivor = a_distinguishable_example(NEAR_BASE_TEXT, weight=3.0)
    loser = a_distinguishable_example(NEAR_VARIANT_TEXT, weight=1.0)
    join = _JoinPayloads()

    assert list(join.process((content_id(survivor), {"payload": [survivor], "lost": []}))) == [survivor]
    assert list(join.process((content_id(loser), {"payload": [loser], "lost": [None, None, None]}))) == []
    assert list(join.process(("orphan", {"payload": [], "lost": [None]}))) == []


def test_equal_weight_near_duplicates_pick_one_survivor_regardless_of_arrival_order() -> None:
    """the content_id tie-break replaces the canonical-json one, so determinism is the contract to
    pin: two equal-weight near-duplicates must yield the SAME survivor under every arrival order
    and every bundle split, which is what keeps a Flink run reproducible across shard counts"""
    first = a_distinguishable_example(NEAR_BASE_TEXT, weight=1.0)
    second = a_distinguishable_example(NEAR_VARIANT_TEXT, weight=1.0)
    expected = min([first, second], key=priority)

    for order in ([first, second], [second, first]):
        pipeline = TestPipeline()
        survivors = pipeline | beam.Create(order) | NearDeduplicate()
        assert_that(survivors, equal_to([expected]))
        result = pipeline.run()
        result.wait_until_finish()
        assert near_counters(result) == {"near_in": 2, "near_dropped": 1, "near_kept": 1, "near_buckets_nontrivial": 5}

    # two direct workers: the join must not depend on both records landing in one bundle
    split_pipeline = TestPipeline(options=PipelineOptions(["--direct_num_workers=2"]))
    split = split_pipeline | "create" >> beam.Create([first, second]) | "near" >> NearDeduplicate()
    assert_that(split, equal_to([expected]))
