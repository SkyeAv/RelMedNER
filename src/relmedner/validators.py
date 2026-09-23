"""pure trust-validation logic: query construction, hit grading, weight adjustment

Every function here is a free function over plain values with no I/O and no network, the same
contract row_filters establishes: the validation decision is unit-testable in isolation, and
only validate.py (the offline sampling driver) performs HTTP. Nothing in this module may run
inside the Beam graph -- trust is derived offline, declared in ingests.yaml, and applied at
weight-stamp time in pipeline.run.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final, Literal

from relmedner.constants import TRUST_RANGE, TRUST_RELATION_PARTIAL_HITS, TRUST_RELATION_VERIFIED_HITS
from relmedner.models import TrainingExample

Verdict = Literal["verified", "partial", "unverified"]

VERIFIED: Final[Verdict] = "verified"
PARTIAL: Final[Verdict] = "partial"
UNVERIFIED: Final[Verdict] = "unverified"

VERDICT_SCORE: dict[str, float] = {VERIFIED: 1.0, PARTIAL: 0.5, UNVERIFIED: 0.0}
"""fractional credit a verdict contributes to a record's trust; partial counts half so a
source whose relations are attested only weakly (1-4 hits) is trusted less than one whose
relations are well attested, not zero"""


def span_query(mention: str, label: str) -> str:
    """pubmed esearch term asserting the surface form AND the concept label both occur in one
    document. [All Fields] (not [Title/Abstract]) because relation/span attestation anywhere
    in the indexed record counts as evidence the pair is real biomedical text, not that the
    annotation was correct in context."""
    return f'"{mention}"[All Fields] AND "{label}"[All Fields]'


def relation_query(head: str, tail: str, predicate: str | None = None) -> str:
    """pubmed esearch term asserting head and tail co-occur; the predicate name (a biolink
    predicate like 'treats') is added when declared, but co-occurrence alone already grades:
    two entities appearing in one abstract is the minimum bar for a real relation mention"""
    term: str = f'"{head}"[All Fields] AND "{tail}"[All Fields]'
    if predicate:
        term += f' AND "{predicate}"[All Fields]'
    return term


def grade_span(hits: int) -> Verdict:
    """one attested occurrence is enough for a surface+label pair: NER gold either names a
    real concept or it does not, there is no weak-middle case the way there is for relations"""
    return VERIFIED if hits > 0 else UNVERIFIED


def grade_relation(hits: int) -> Verdict:
    """graded on volume: >= TRUST_RELATION_VERIFIED_HITS co-occurring documents is strong
    attestation, 1..partial is a real but possibly coincidental co-occurrence, 0 is
    unverified. Coincidence is the reason relations do NOT use grade_span's binary rule --
    'patient' and 'cancer' co-occur in thousands of abstracts without any asserted relation"""
    if hits >= TRUST_RELATION_VERIFIED_HITS:
        return VERIFIED
    if hits >= TRUST_RELATION_PARTIAL_HITS:
        return PARTIAL
    return UNVERIFIED


def example_queries(example: TrainingExample) -> list[tuple[str, str, str]]:
    """every validation query one emitted example carries, as (kind, query, feature) triples:
    kind is 'span' or 'relation'; feature is the entity LABEL or the relation PREDICATE name --
    the grouping key the sampler aggregates verdicts under, so a sample can say 'treats edges
    in this source are unreliable' instead of only 'this source is unreliable'. One query per
    entity (first mention is the canonical surface; the remaining mentions are variants of the
    same concept, querying them all would multiply-count one concept) and one per relation
    (head/tail field values plus the predicate name). Classifications and structures are NOT
    literature-validatable and contribute no queries, which is what makes record_trust return
    None for them."""
    queries: list[tuple[str, str, str]] = []
    for entity in example.entities:
        if entity.mentions:
            queries.append(("span", span_query(entity.mentions[0], entity.label), entity.label))
    for relation in example.relations:
        fields: dict[str, str] = {field.name: field.value for field in relation.fields}
        if fields.get("head") and fields.get("tail"):
            queries.append(("relation", relation_query(fields["head"], fields["tail"], relation.name), relation.name))
    return queries


def record_trust(verdicts: Iterable[Verdict]) -> float | None:
    """mean verdict score over one record's queries, or None when the record has no
    validatable shapes (entity/relation-free rows neither earn nor spend trust; scoring them
    0 would punish classification-only sources for having nothing to validate)"""
    scores: list[float] = [VERDICT_SCORE[verdict] for verdict in verdicts]
    if not scores:
        return None
    return sum(scores) / len(scores)


def source_trust(record_trusts: Iterable[float | None]) -> float | None:
    """mean over a source's SCORED records only; None when every sampled record had nothing
    to validate, so the caller prints no suggestion instead of a misleading 1.0"""
    scored: list[float] = [trust for trust in record_trusts if trust is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)


def query_outcomes(results: Iterable[dict[str, Any]]) -> tuple[int, int]:
    """(verdicted, errored) over one validated record's query results.

    `validate_record` stores either a graded verdict or an error with `hits=None`, never both,
    so the two counts partition the record's queries. The split matters because a source's trust
    is `None` in two very different situations: the sampled records carried nothing to query, or
    every request failed. Only the error count says which, and the two need opposite responses
    (accept the source as not literature-validatable, versus fix the network or the rejected
    `NCBI_API_KEY` and re-run).
    """
    verdicted = 0
    errored = 0
    for result in results:
        if result.get("verdict") is None:
            errored += 1
        else:
            verdicted += 1
    return verdicted, errored


def record_edge_factor(example: TrainingExample, edge_trusts: dict[str, float]) -> float:
    """heuristic extrapolation of sampled verdicts to one record: the WEAKEST trust among the
    record's relation predicates that the sample flagged (1.0 when the record carries no
    flagged edge). The min is the pessimist's AND -- one unreliable asserted edge poisons the
    record's training value more than several reliable ones redeem it. Records with no
    relations, or relations the sample cleared, are untouched: the edge weighting applies to
    a SUBSET of the dataset by design"""
    if not example.relations or not edge_trusts:
        return 1.0
    flagged: list[float] = [edge_trusts[relation.name] for relation in example.relations if relation.name in edge_trusts]
    return min(flagged) if flagged else 1.0


def example_weight(example: TrainingExample, base_weight: float, edge_trusts: dict[str, float]) -> float:
    """the stamped weight: base_weight (already source-trust-adjusted) scaled by the record's
    edge factor, clamped to [0, 1.0]. Unlike the source-level nudge, edge trust SCALES
    directly -- a targeted heuristic on a known-bad predicate subset is evidence, not a
    mixing-intent question, so the +-TRUST_RANGE band does not apply to it. factor 0 soft-drops
    the record exactly like source trust 0"""
    factor: float = record_edge_factor(example, edge_trusts)
    if factor == 1.0:
        return base_weight
    return min(1.0, base_weight * factor)


def adjust_weight(weight: float, trust: float, band: float = TRUST_RANGE) -> float:
    """effective stamped weight: declared weight x trust, clamped to the fixed symmetric band
    [weight*(1-band), min(1.0, weight*(1+band))]. Trust only NUDGES inside the band -- moving
    beyond it is a declared-weight decision, never a trust decision. THE ONE EXCEPTION:
    trust == 0 bypasses the clamp and yields 0.0, the explicit soft drop (record flows to the
    avro provenance but duplicates zero times at export); it exists for the rare required
    case and the guide discourages it everywhere else."""
    if trust == 0:
        return 0.0
    adjusted: float = weight * trust
    lower: float = weight * (1 - band)
    upper: float = min(1.0, weight * (1 + band))
    return min(max(adjusted, lower), upper)
