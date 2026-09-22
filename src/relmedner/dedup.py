from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable, Iterator

import apache_beam as beam
from apache_beam.metrics.metric import Metrics

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
