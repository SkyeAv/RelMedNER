from __future__ import annotations

import hashlib
import random
import struct
import uuid
from collections.abc import Iterable, Iterator
from typing import Any

import apache_beam as beam
import numpy as np
from apache_beam.metrics.metric import Metrics, MetricsFilter
from apache_beam.runners.runner import PipelineResult

from relmedner.constants import (
    DEDUP_BANDS,
    DEDUP_NUM_PERM,
    DEDUP_ROWS_PER_BAND,
    DEDUP_SEED,
    MIN_NEAR_TOKENS,
)
from relmedner.enums import DedupMode
from relmedner.models import TrainingExample

DEDUP_METRICS_NAMESPACE = "relmedner.dedup"


def normalize_text(text: str) -> str:
    """collapse whitespace runs and strip the ends; case is preserved because biomedical acronym
    casing (TNF vs tnf) is signal, never folded for the exact key"""
    return " ".join(text.split())


def exact_key(text: str) -> str:
    """128-bit blake2b fingerprint (hex) of the normalized text"""
    return hashlib.blake2b(normalize_text(text).encode("utf-8"), digest_size=16).hexdigest()


def content_id(example: TrainingExample) -> str:
    """128-bit blake2b fingerprint (hex) of the record's canonical json: a compact, stable name
    for the whole payload, so a record can be referred to in a shuffle without shipping it"""
    return hashlib.blake2b(example.model_dump_json().encode("utf-8"), digest_size=16).hexdigest()


def priority(example: TrainingExample) -> tuple[float, str]:
    """total order over records sharing one key: highest weight wins, content_id breaks ties,
    so the survivor is deterministic across runners, shard counts, and arrival orders.

    The tie-break is a content hash rather than the canonical json string itself (docs/weighting.md):
    both are total orders over content, and only the hash is small enough to ship through the
    near-dedup band shuffle, where every record used to travel once per band with its full payload
    attached. Which of two EQUAL-WEIGHT near-duplicates survives can therefore differ from the
    pre-hash pipeline; determinism, the drop rule, and every weight-bearing decision are unchanged."""
    return (-example.weight, content_id(example))


# ---------------------------------------------------------------- near-dedup math ----
# MinHash + LSH banding over lowercased word 5-gram shingles, following the datasketch /
# text-dedup affine-permutation construction: (a_j * h + b_j) mod p with a_j in [1, p),
# b_j in [0, p), and the minimum over the document's shingle hashes per permutation j.
# text-dedup uses uint32 / (2**32 - 5) / xxh3; here it is a 61-bit Mersenne prime + blake2b
# for headroom, stdlib only. Everything derives from DEDUP_SEED: builtin hash() is salted
# per process and FORBIDDEN here, or two workers would never agree on a signature.

if DEDUP_BANDS * DEDUP_ROWS_PER_BAND != DEDUP_NUM_PERM:  # banding must tile the signature
    raise ValueError("DEDUP_BANDS * DEDUP_ROWS_PER_BAND must equal DEDUP_NUM_PERM")

_MERSENNE_PRIME: int = (1 << 61) - 1
_EMPTY_TEXT_SENTINEL: int = _MERSENNE_PRIME - 1  # largest value (a * h + b) mod p can take

_rng = random.Random(DEDUP_SEED)
_PERM_A: tuple[int, ...] = tuple(_rng.randint(1, _MERSENNE_PRIME - 1) for _ in range(DEDUP_NUM_PERM))
_PERM_B: tuple[int, ...] = tuple(_rng.randint(0, _MERSENNE_PRIME - 1) for _ in range(DEDUP_NUM_PERM))

# ------------------------------------------------ vectorized exact mod-(2**61 - 1) math ----
# numpy has no 128-bit integers, so (a * h) mod p is computed exactly from 32-bit limbs, every
# intermediate proven to fit uint64 (bounds noted per line). Mersenne reduction: 2**61 == 1 mod p,
# so v mod p folds as (v & p) + (v >> 61). The result is byte-identical to the python int
# arithmetic it replaces -- a wraparound would silently change every near-dedup decision, which is
# why tests/test_dedup.py compares against the pure-python reference on random and long texts.

_P64 = np.uint64(_MERSENNE_PRIME)
_LOW32 = np.uint64(0xFFFFFFFF)
_S29, _S32, _S61 = np.uint64(29), np.uint64(32), np.uint64(61)
_LOW29 = np.uint64((1 << 29) - 1)
_A_ARRAY = np.array(_PERM_A, dtype=np.uint64)[:, None]
_B_ARRAY = np.array(_PERM_B, dtype=np.uint64)[:, None]
_A_HI, _A_LO = _A_ARRAY >> _S32, _A_ARRAY & _LOW32  # a < 2**61: a_hi < 2**29, a_lo < 2**32


def _fold(values: np.ndarray) -> np.ndarray:
    """one Mersenne fold: any v < 2**64 -> (v & p) + (v >> 61) < 2**61 + 8"""
    return (values & _P64) + (values >> _S61)


def _reduce(values: np.ndarray) -> np.ndarray:
    """full reduction into [0, p) for values < 2**64: two folds then one conditional subtract"""
    folded = _fold(_fold(values))
    return np.where(folded >= _P64, folded - _P64, folded)


def _affine_mod(hashes: np.ndarray) -> np.ndarray:
    """(a_j * h + b_j) mod p for every permutation j (rows) and shingle hash h (columns)"""
    x = _reduce(hashes)[None, :]  # h mod p, so x < 2**61: x_hi < 2**29, x_lo < 2**32
    x_hi, x_lo = x >> _S32, x & _LOW32
    high = (_A_HI * x_hi) << np.uint64(3)  # a_hi*x_hi < 2**58, times 2**64 == 8 (mod p): < 2**61
    mid = _A_HI * x_lo + _A_LO * x_hi  # < 2 * 2**61 = 2**62
    # mid * 2**32 = (mid >> 29) * 2**61 + (mid & low29) * 2**32 == (mid >> 29) + (mid & low29) << 32
    mid_mod = (mid >> _S29) + ((mid & _LOW29) << _S32)  # < 2**33 + 2**61
    low = _fold(_A_LO * x_lo)  # a_lo*x_lo < 2**64 -> < 2**61 + 8
    total = high + mid_mod + low + _B_ARRAY  # each < 2**61 + 2**33: sum < 2**63
    return _reduce(total)


def shingles(text: str, n: int = 5) -> tuple[str, ...]:
    """lowercased word n-grams (5-grams by default) of text.split(), in document order; empty
    tuple when the text has fewer than n tokens. Lowercasing is for similarity only -- the
    surviving record's original text is never altered -- and the empty tuple is what makes
    sub-MIN_NEAR_TOKENS texts structurally unable to produce band keys (REQ-NEAR-1)."""
    words = text.lower().split()
    return tuple(" ".join(words[index : index + n]) for index in range(len(words) - n + 1))


def base_hash(shingle: str) -> int:
    """stable unsigned 64-bit fingerprint of one shingle (big-endian blake2b, digest_size=8);
    stdlib and cross-process stable where builtin hash() is salted per process"""
    return int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")


def signature(text: str) -> tuple[int, ...]:
    """DEDUP_NUM_PERM MinHash values: the minimum of (a_j * h + b_j) mod (2**61 - 1) over the
    text's shingle hashes h, one value per fixed-seed permutation (a_j, b_j) drawn once at
    module load. Deterministic across calls AND processes. A text with no shingles (fewer
    than n tokens) returns the all-sentinel signature; the near-dedup stage must gate such
    texts on MIN_NEAR_TOKENS before banding, short texts give noisy MinHash estimates.

    Vectorized over a (num_perm x distinct shingles) uint64 matrix with exact limb arithmetic
    (see _affine_mod); repeated shingles are hashed once since they cannot change a minimum."""
    distinct = set(shingles(text))  # min over a multiset equals min over its set
    if not distinct:
        return (_EMPTY_TEXT_SENTINEL,) * DEDUP_NUM_PERM
    hashes = np.fromiter((base_hash(shingle) for shingle in distinct), dtype=np.uint64, count=len(distinct))
    return tuple(int(value) for value in _affine_mod(hashes).min(axis=1))


def band_keys(sig: tuple[int, ...]) -> tuple[tuple[int, str], ...]:
    """DEDUP_BANDS LSH keys: (band index, blake2b hex digest of that band's DEDUP_ROWS_PER_BAND
    signature rows packed little-endian uint64). Two texts share a band key iff every row of
    that band agrees; by the S-curve (see constants.py) texts with true Jaccard >= ~0.88
    share at least one key with high probability, dissimilar ones almost never."""
    if len(sig) != DEDUP_NUM_PERM:
        raise ValueError(f"signature must have exactly {DEDUP_NUM_PERM} rows, got {len(sig)}")
    return tuple(
        (
            band,
            hashlib.blake2b(
                struct.pack(f"<{DEDUP_ROWS_PER_BAND}Q", *sig[band * DEDUP_ROWS_PER_BAND : (band + 1) * DEDUP_ROWS_PER_BAND]),
                digest_size=16,
            ).hexdigest(),
        )
        for band in range(DEDUP_BANDS)
    )


class _KeyByExample(beam.DoFn):
    """stamps every record with its exact-dedup key and counts it as entered (exact_in)

    blank-normalized records get a unique throwaway key so they form singleton groups and pass
    through without collapsing into one survivor (the matches_declared_outputs safety valve)
    """

    def process(self, example: TrainingExample) -> Iterator[tuple[str, TrainingExample]]:
        Metrics.counter(DEDUP_METRICS_NAMESPACE, "exact_in").inc()
        normalized = normalize_text(example.text)
        yield ((exact_key(example.text) if normalized else uuid.uuid4().hex), example)


class _KeepPriorityWinner(beam.DoFn):
    """per key-group emit min(priority); count one kept and one per non-winner dropped

    GroupByKey already materialized the group, and the winner is computed from the whole group
    (not from a first-seen scan), so the result never depends on element order
    """

    def process(self, keyed_group: tuple[str, Iterable[TrainingExample]]) -> Iterator[TrainingExample]:
        _, group = keyed_group
        examples = list(group)
        Metrics.counter(DEDUP_METRICS_NAMESPACE, "exact_kept").inc()
        Metrics.counter(DEDUP_METRICS_NAMESPACE, "exact_dropped").inc(len(examples) - 1)
        yield min(examples, key=priority)


class KeepPriorityWinnerByKey(beam.PTransform):
    """exact-text dedup over TrainingExample values: key, group, keep one priority winner per key

    pure Beam composition (ParDo + GroupByKey, no runner-specific state or timers), so it runs
    unchanged on the DirectRunner and the Flink runner; the stage selects, it never unions or
    mutates: a dropped duplicate's payload is gone, weight multiplicity is untouched. Counters
    under namespace relmedner.dedup always reconcile: exact_in == exact_dropped + exact_kept
    """

    def expand(self, examples: beam.PCollection[TrainingExample]) -> beam.PCollection[TrainingExample]:
        keyed = examples | "assign exact key" >> beam.ParDo(_KeyByExample())
        grouped = keyed | "group by exact key" >> beam.GroupByKey()
        return grouped | "keep priority winner" >> beam.ParDo(_KeepPriorityWinner())


class _EmitBandKeys(beam.DoFn):
    """stamp every record into near_in, then split it into a compact band stream and one
    payload stream.

    Long texts emit one (band_key, priority) per band -- priority is (-weight, content_id), about
    40 bytes -- and the record itself rides a SEPARATE tagged output keyed by content_id, so the
    full payload crosses the shuffle exactly once instead of once per band. Short texts
    (< MIN_NEAR_TOKENS tokens) bypass near-dedup entirely (REQ-NEAR-1) on the "bypass" tag: they
    never reach a band bucket, so they can never be near_dropped and never collapse with anything.

    near_kept is NOT counted here: Beam keys user counters by (step, namespace, name), so one name
    incremented in two different steps is returned by a by-name query as one value PER STEP, not
    their sum (see _CountNearKept)."""

    def process(self, example: TrainingExample) -> Iterator[tuple[tuple[int, str], tuple[float, str]] | beam.pvalue.TaggedOutput]:
        Metrics.counter(DEDUP_METRICS_NAMESPACE, "near_in").inc()
        if len(example.text.split()) < MIN_NEAR_TOKENS:
            yield beam.pvalue.TaggedOutput("bypass", example)
            return
        sig = signature(example.text)
        rank = priority(example)
        for band_key in band_keys(sig):
            yield (band_key, rank)
        yield beam.pvalue.TaggedOutput("payload", (rank[1], example))


class _TagBandLosers(beam.DoFn):
    """per band bucket, emit one loser marker for every record that is not the bucket's
    min-priority winner.

    The winner is computed from the whole bucket (never from arrival order), and losing is a MARKER
    rather than a drop decision: the same record can win band 3 and lose band 5, and only the join
    below may decide its fate -- that is what keeps near_dropped counting one per RECORD instead of
    one per lost bucket (a 0.98-similar pair shares ~5 of 8 bands; counting losses per bucket would
    count the same record 5 times and break the near_in reconciliation). Only losers ship anything,
    so a bucket of records that all survive costs no downstream elements at all."""

    def process(self, keyed_group: tuple[tuple[int, str], Iterable[tuple[float, str]]]) -> Iterator[tuple[str, None]]:
        _, ranks = keyed_group
        ranks = list(ranks)
        if len(ranks) > 1:
            Metrics.counter(DEDUP_METRICS_NAMESPACE, "near_buckets_nontrivial").inc()
        winner = min(ranks)
        for rank in ranks:
            if rank != winner:
                yield (rank[1], None)


class _JoinPayloads(beam.DoFn):
    """CoGroupByKey join of the payload stream with the loser markers: emit a record unless it lost
    at least one band bucket.

    Grouping key is the record's content_id, so a record that lost k of its 8 bands arrives with
    k markers and is dropped EXACTLY ONCE (REQ-NEAR-4/6): winning band 3 cannot resurrect a record
    that lost band 5 to a higher-priority record, because "dropped iff it shares at least one band
    bucket with a strictly higher-priority record" is a global rule. near_dropped is counted here.

    Two records sharing one content_id have byte-identical payloads (the id is a hash of the
    canonical json), so emitting one of them is not a choice: the payload stream carries each
    surviving record exactly once."""

    def process(self, joined: tuple[str, dict[str, list[Any]]]) -> Iterator[TrainingExample]:
        _content_id, groups = joined
        payloads: list[TrainingExample] = groups.get("payload", [])
        if not payloads:
            return  # a loser marker with no payload cannot happen; skip rather than count a phantom
        if groups.get("lost"):
            Metrics.counter(DEDUP_METRICS_NAMESPACE, "near_dropped").inc()
            return
        yield payloads[0]


class _CountNearKept(beam.DoFn):
    """count one near_kept per merged survivor, in a SINGLE step, after the Flatten

    near_kept must be incremented in exactly one step: Beam keys user counters by
    (step, namespace, name), so a name incremented in both the bypass emit step and the join step
    comes back from a by-name metrics query as one value PER STEP. The name-keyed dict in tests
    (and dashboards) then keeps only one step's value instead of the sum, and near_in stops
    reconciling on mixed inputs -- bypass survivors and band survivors must be counted together,
    here, where both paths have merged."""

    def process(self, example: TrainingExample) -> Iterator[TrainingExample]:
        Metrics.counter(DEDUP_METRICS_NAMESPACE, "near_kept").inc()
        yield example


class NearDeduplicate(beam.PTransform):
    """near-text dedup over TrainingExample values: LSH band grouping, priority drop, payload join

    pure Beam composition (ParDo + GroupByKey + CoGroupByKey + Flatten, no runner-specific state,
    timers, or uuid keying), so it runs unchanged on the DirectRunner and the Flink runner and is
    fully deterministic: signatures come from DEDUP_SEED, winners from min(priority) over whole
    buckets. Semantics (REQ-NEAR-4): a record is dropped iff it shares at least one band bucket
    with a strictly higher-priority record; short texts (< MIN_NEAR_TOKENS tokens) bypass untouched.
    Counters under namespace relmedner.dedup always reconcile: near_in == near_dropped + near_kept.

    Shuffle shape: 8 compact (band_key, priority) elements plus ONE full payload per record, and
    the band stream never carries a record. The previous shape shipped the whole TrainingExample
    once per band and again once per band on the collapse step -- 16 full copies per record, which
    is what made near-dedup the pipeline's dominant shuffle cost."""

    def expand(self, examples: beam.PCollection[TrainingExample]) -> beam.PCollection[TrainingExample]:
        emitted = examples | "near: emit band keys" >> beam.ParDo(_EmitBandKeys()).with_outputs("bypass", "payload", main="bands")
        losers = emitted.bands | "near: group bands" >> beam.GroupByKey() | "near: tag band losers" >> beam.ParDo(_TagBandLosers())
        survivors = (
            {"payload": emitted.payload, "lost": losers}
            | "near: join payloads with loser markers" >> beam.CoGroupByKey()
            | "near: drop records that lost a band" >> beam.ParDo(_JoinPayloads())
        )
        merged = (survivors, emitted.bypass) | "near: merge survivors" >> beam.Flatten()
        return merged | "near: count kept survivors" >> beam.ParDo(_CountNearKept())


def format_dedup_summary(result: PipelineResult | None) -> str | None:
    """DirectRunner-only reporting (REQ-INT-4): read the relmedner.dedup counters out of a
    finished run and render the one summary line `dedup: exact -<n> near -<n> of <total>
    records`. exact/near report each stage's dropped counts; total is exact_in, the count
    entering the exact stage -- which is the full record count, because near_in only sees the
    exact stage's survivors and summing them would double-count. Returns None when there is
    nothing to report (OFF mode ran no dedup stage, so no counters exist) so the caller
    simply skips the log line instead of printing zeros"""
    if result is None:
        return None
    query = result.metrics().query(MetricsFilter().with_namespace(DEDUP_METRICS_NAMESPACE))
    totals: dict[str, int] = {}
    for counter in query["counters"]:
        # Beam keys counters by (step, namespace, name): one name incremented in several steps
        # comes back once per step, so a by-name dict must sum across steps
        name = counter.key.metric.name
        totals[name] = totals.get(name, 0) + counter.committed
    if not totals:
        return None
    exact_dropped = totals.get("exact_dropped", 0)
    near_dropped = totals.get("near_dropped", 0)
    total = totals.get("exact_in", 0)
    return f"dedup: exact -{exact_dropped} near -{near_dropped} of {total} records"


def apply_dedup(examples: beam.PCollection[TrainingExample], mode: DedupMode) -> beam.PCollection[TrainingExample]:
    """switch the dedup stage chain per mode (REQ-INT-2): OFF returns the pcollection
    UNCHANGED so no dedup transform enters the graph at all; EXACT adds the exact stage
    only; NEAR chains exact then near. mode is normalized through DedupMode() so the string
    stored by RunConfig's use_enum_values validates loudly instead of silently skipping"""
    selected: DedupMode = DedupMode(mode)
    if selected is DedupMode.OFF:
        return examples
    exacted = examples | "exact dedup" >> KeepPriorityWinnerByKey()
    if selected is DedupMode.EXACT:
        return exacted
    return exacted | "near dedup" >> NearDeduplicate()
