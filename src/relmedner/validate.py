"""offline trust-validation driver (the validate-trust CLI backend)

Samples records per declared source, validates the emitted entities/relations against
literature, and writes a reviewable JSONL report plus a suggested-trust yaml snippet. This
module is deliberately OUTSIDE the Beam graph: validation is sampled, rate-limited, and
network-bound, none of which belongs on a worker. The pipeline itself stays deterministic
and offline -- it only consumes the trust: values a human commits to ingests.yaml after
reading the report this driver produces.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from os import environ
from pathlib import Path
from typing import Any, Protocol, Self

from relmedner.constants import (
    PUBMED_ESEARCH_URL,
    PUBMED_THROTTLE_SECONDS,
    PUBMED_THROTTLE_SECONDS_KEYED,
)
from relmedner.fullmap_mine import FullmapMiner
from relmedner.gazetteer import configure_gazetteer
from relmedner.models import RunConfig, TrainingExample, ValidateTrustConfig, YamlIngests
from relmedner.registry import build_stream
from relmedner.streams import StreamedRow, rebuild_task
from relmedner.types import Script
from relmedner.validators import (
    VERDICT_SCORE,
    Verdict,
    example_queries,
    grade_relation,
    grade_span,
    query_outcomes,
    record_trust,
    source_trust,
)

logger = logging.getLogger(__name__)

FULLMAP_TYPE: str = "fullmap"
TEXT_EXCERPT_CHARS: int = 200
"""report text preview length; the full text stays in the corpus, the report only needs enough
to eyeball whether a verdict was right"""

# ------------------------------------------------------------------------ hit clients --


class HitClient(Protocol):
    """one validation backend: a query string in, an evidence-document count out. validate_source
    depends on this protocol rather than a concrete client, so tests substitute fakes without
    touching the network"""

    def hits(self: Self, query: str) -> int: ...


class ThrottledClient:
    """base for rate-limited clients: one shared last-request timestamp so sequential query
    loops self-throttle regardless of which backend answers"""

    def __init__(self: Self, delay_seconds: float) -> None:
        self.delay_seconds: float = delay_seconds
        self._last_request: float = 0.0

    def _throttle(self: Self) -> None:
        elapsed: float = time.monotonic() - self._last_request
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self._last_request = time.monotonic()


class PubMedClient(ThrottledClient):
    """NCBI E-utilities esearch, retmode=json. No key: 3 rps (PUBMED_THROTTLE_SECONDS);
    NCBI_API_KEY env raises the ceiling to ~9 rps. stdlib urllib only -- the pipeline adds no
    dependency for an offline admin tool"""

    def __init__(self: Self) -> None:
        self.api_key: str | None = environ.get("NCBI_API_KEY")
        super().__init__(PUBMED_THROTTLE_SECONDS_KEYED if self.api_key else PUBMED_THROTTLE_SECONDS)

    def hits(self: Self, query: str) -> int:
        self._throttle()
        params: dict[str, str] = {"db": "pubmed", "term": query, "retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        url: str = f"{PUBMED_ESEARCH_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 (fixed NCBI host)
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        return int(payload["esearchresult"]["count"])


# ------------------------------------------------------------------------ sampling --


def sampled_examples(ingests: YamlIngests, config: RunConfig, sample_size: int, only: str | None = None) -> Iterator[tuple[str, TrainingExample]]:
    """up to sample_size dispatched TrainingExamples per row key, reusing the registry and the
    pipeline's dispatch/fullmap-resolution paths so validation sees EXACTLY what training
    would see (same scripts, same gazetteer, same filters). fullmap rows resolve through one
    batched FullmapMiner.resolve_batch per stream, mirroring resolve_rows"""
    configure_gazetteer(ingests.gazetteer)
    # two entries sharing one row key (e.g. train/test splits of one repo) each contribute up
    # to sample_size rows; validate_sources merges them under the shared key, matching how
    # weights_by_source treats the pair as one slot
    for dataset in ingests.datasets:
        # the sampling pass walks each source ONCE as a whole pass, so shard envelopes never
        # reach this loop and a read_shards source is sampled in full, not one shard of it
        source, payload, filters, sample_rate, _read_shards = dataset.to_stream_args()
        stream = build_stream(source, payload, filters=filters, sample_rate=sample_rate)
        key: str = stream.name
        if only is not None and key != only:
            continue
        rows: list[StreamedRow] = []
        for index, row in enumerate(stream.stream(config)):
            if index >= sample_size:
                break
            rows.append(row)
        script_rows: list[StreamedRow] = [row for row in rows if row[1][0][0] != FULLMAP_TYPE]
        fullmap_rows: list[StreamedRow] = [row for row in rows if row[1][0][0] == FULLMAP_TYPE]
        for _source, (task, values) in script_rows:
            task_model = rebuild_task(task)
            _outputs, example = Script.dispatch(task_model.name, (tuple(task_model.outputs), values))
            yield (key, example)
        if fullmap_rows:
            tasks = [rebuild_task(task) for _source, (task, _values) in fullmap_rows]
            pairs = [(values[0], task) for (_source, (_task, values)), task in zip(fullmap_rows, tasks, strict=True)]
            for example in FullmapMiner.resolve_batch(pairs):
                yield (key, example)


# ------------------------------------------------------------------------ validation --


def validate_record(example: TrainingExample, client: HitClient) -> dict[str, Any]:
    """one record -> its query results. network errors are recorded with hits=None and carry
    NO verdict: a failed request must not score against the source (that would punish the
    dataset for the network, not for its labels). feature is the entity label / relation
    predicate, the grouping key the source-level aggregation extrapolates from"""
    results: list[dict[str, Any]] = []
    verdicts: list[Verdict] = []
    for kind, query, feature in example_queries(example):
        try:
            count: int = client.hits(query)
            verdict: Verdict = grade_span(count) if kind == "span" else grade_relation(count)
            results.append({"kind": kind, "feature": feature, "query": query, "hits": count, "verdict": verdict})
            verdicts.append(verdict)
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError) as error:
            results.append({"kind": kind, "feature": feature, "query": query, "hits": None, "error": str(error)})
    return {
        "text": example.text[:TEXT_EXCERPT_CHARS],
        "queries": results,
        "trust": record_trust(verdicts),
    }


def validate_sources(
    ingests: YamlIngests,
    config: RunConfig,
    client: HitClient,
    sample_size: int,
    only: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """every declared source -> (summaries, report records). summaries carry {source, sampled,
    trust}; report records carry one line per sampled record (text excerpt, per-query hits and
    verdicts, record trust) for the JSONL artifact and the review pass"""
    per_source: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for source, example in sampled_examples(ingests, config, sample_size, only):
        entry: dict[str, Any] = per_source.setdefault(
            source,
            {"source": source, "sampled": 0, "record_trusts": [], "edge_verdicts": {}, "verdicted": 0, "errored": 0},
        )
        record: dict[str, Any] = validate_record(example, client)
        entry["sampled"] += 1
        entry["record_trusts"].append(record["trust"])
        # the verdict/error split is what lets the printed suggestion tell "nothing to validate"
        # apart from "every request failed"; both leave trust None
        scored, failed = query_outcomes(record["queries"])
        entry["verdicted"] += scored
        entry["errored"] += failed
        records.append({"source": source, **record})
        logger.info("trust report %s: %s", source, json.dumps({"source": source, **record}))
        # per-predicate edge verdicts: the heuristic bridge from sample to whole dataset --
        # every relation verdict lands under its predicate name, and the suggested yaml
        # down-weights that predicate on ALL of the source's records, not just sampled ones
        for result in record["queries"]:
            if result["kind"] == "relation" and result.get("verdict") is not None:
                entry["edge_verdicts"].setdefault(result["feature"], []).append(result["verdict"])
    summaries: list[dict[str, Any]] = []
    for source in per_source:
        entry = per_source[source]
        trust: float | None = source_trust(entry["record_trusts"])
        edge_trusts: dict[str, float] = {
            predicate: sum(VERDICT_SCORE[verdict] for verdict in verdicts) / len(verdicts) for predicate, verdicts in entry["edge_verdicts"].items()
        }
        summary: dict[str, Any] = {
            "source": source,
            "sampled": entry["sampled"],
            "trust": trust,
            "edge_trusts": edge_trusts,
            "queries": entry["verdicted"] + entry["errored"],
            "errored": entry["errored"],
        }
        summaries.append(summary)
        logger.info(
            "trust summary %s: sampled=%d scored=%d queries=%d errored=%d trust=%s edges=%s",
            source,
            entry["sampled"],
            sum(1 for value in entry["record_trusts"] if value is not None),
            summary["queries"],
            summary["errored"],
            f"{trust:.2f}" if trust is not None else f"n/a ({_unscored_reason(summary)})",
            json.dumps({k: round(v, 2) for k, v in edge_trusts.items()}),
        )
    return summaries, records


def _unscored_reason(summary: dict[str, Any]) -> str:
    """Why one source produced no trust suggestion.

    `source_trust` returns None both when the sampled records carried nothing to query and when
    every query errored, and the two need opposite responses: accept that the source is not
    literature-validatable, or fix the network / rejected `NCBI_API_KEY` and re-run. Printing the
    first reason for the second case is what sent a real run chasing a corpus that was fine
    (12 of 12 queries answered `HTTP Error 400: Bad Request`). Summaries that predate the
    `queries`/`errored` counters keep the old wording rather than raising.
    """
    errored: int = int(summary.get("errored") or 0)
    total: int = int(summary.get("queries") or 0)
    if errored and errored >= total:
        return f"all {errored} queries errored (network, rate limit, or a rejected NCBI_API_KEY); see the report"
    if errored:
        return f"{errored} of {total} queries errored and nothing else scored; see the report"
    return "sampled records had no entities/relations to validate"


def suggested_yaml(summaries: list[dict[str, Any]]) -> str:
    """the hand-editable snippet: source-level `trust:` plus a `trust_edges:` block listing
    every predicate the sample scored below 0.99 (a fully-verified predicate needs no line).
    Deliberately not auto-applied -- the human reads the JSONL report first, then commits"""
    lines: list[str] = []
    for summary in summaries:
        trust: float | None = summary["trust"]
        lines.append(f"  # {summary['source']} (sampled {summary['sampled']})")
        lines.append(f"  trust: {trust:.2f}" if trust is not None else f"  # trust: n/a -- {_unscored_reason(summary)}")
        flagged: list[tuple[str, float]] = sorted((predicate, score) for predicate, score in summary.get("edge_trusts", {}).items() if score < 0.99)
        if flagged:
            lines.append("  trust_edges:")
            lines.extend(f"    {predicate}: {score:.2f}" for predicate, score in flagged)
    return "\n".join(lines)


def resolve_trust_settings(
    sample_size: int | None,
    report: str | None,
    config: ValidateTrustConfig | None,
) -> tuple[int, str]:
    """flag > yaml > constants: CLI flags are Optional so an absent flag defers to the x-trust
    yaml section, whose own defaults are the constants. Secrets never participate -- env only"""
    Settings: ValidateTrustConfig = config or ValidateTrustConfig()
    return (sample_size or Settings.sample_size, report or Settings.report)


def write_report(path: Path, summaries: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    """one JSONL artifact: per-record lines first, one summary line per source last"""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        for summary in summaries:
            handle.write(json.dumps({"summary": summary}) + "\n")
