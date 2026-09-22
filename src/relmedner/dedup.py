from __future__ import annotations

import hashlib
import random
import struct
import uuid
from collections.abc import Iterable, Iterator

import apache_beam as beam
from apache_beam.metrics.metric import Metrics

from relmedner.constants import DEDUP_BANDS, DEDUP_NUM_PERM, DEDUP_ROWS_PER_BAND, DEDUP_SEED
from relmedner.models import TrainingExample

DEDUP_METRICS_NAMESPACE = "relmedner.dedup"


def normalize_text(text: str) -> str:
    """collapse whitespace runs and strip the ends; case is preserved because biomedical acronym
    casing (TNF vs tnf) is signal, never folded for the exact key"""
    return " ".join(text.split())


def exact_key(text: str) -> str:
    """128-bit blake2b fingerprint (hex) of the normalized text"""
    return hashlib.blake2b(normalize_text(text).encode("utf-8"), digest_size=16).hexdigest()


def priority(example: TrainingExample) -> tuple[float, str]:
    """total order over records sharing one key: highest weight wins, canonical json breaks ties,
    so the survivor is deterministic across runners, shard counts, and arrival orders"""
    return (-example.weight, example.model_dump_json())


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
    texts on MIN_NEAR_TOKENS before banding, short texts give noisy MinHash estimates."""
    shingle_hashes = [base_hash(shingle) for shingle in shingles(text)]
    if not shingle_hashes:
        return (_EMPTY_TEXT_SENTINEL,) * DEDUP_NUM_PERM
    return tuple(min((a * h + b) % _MERSENNE_PRIME for h in shingle_hashes) for a, b in zip(_PERM_A, _PERM_B, strict=True))


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
