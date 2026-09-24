from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import blake2b
from itertools import islice
from typing import Any, ClassVar, Self

from relmedner.models import RowFilters, RunConfig
from relmedner.types import ScriptPayload

StreamedRow = tuple[str, ScriptPayload]

# stdlib logging with NO basicConfig on purpose: the Beam DirectRunner and the flink sdkworkers
# capture logged records through their own handlers, and tests attach caplog; this logger is
# the entire US-009 surface (collect.py's log callable stays as-is)
_QUALITY_LOG = logging.getLogger("relmedner.quality")


class ZeroYieldError(RuntimeError):
    """a source dropped 100% of its candidate rows: the historical silent zero-yield ingest bug
    (a declaration typo quietly shipping an empty training set) made impossible to recreate
    through filters OR the ALWAYS-ON token cap. Raised only when the source had candidate rows
    (post-match_on) and none survived; match_on-only emptiness is recorded, not guarded"""


@dataclass
class StreamStats:
    """per-pass data-quality counters for one DataStream (US-009)

    WHY a dataclass and not a StrictBase model: the counters are mutated row by row
    mid-iteration, a plain mutable-record job rather than validation (the models imported
    here arrive already-validated; StrictBase stays the schema boundary at parse time).
    dropped_by keys
    are evaluator reason strings consumed verbatim from row_filters.first_drop_reason plus
    "match_on" attributed by the streams themselves; insertion order is the order reasons first
    fired, which keeps the report line stable for a fixed input.
    """

    rows_in: int = 0
    """every row read from the source (pre-match_on for hf sources, records read for local
    avro), so rows_in - rows_out always equals the total dropped"""

    rows_out: int = 0
    """rows actually yielded to the declared task"""

    dropped_by: dict[str, int] = field(default_factory=dict)
    """one counter per drop reason, keyed by the US-008 evaluator's reason strings"""

    def drop(self: Self, reason: str) -> None:
        """attribute one dropped row to its reason; the caller keeps owning the actual drop
        (its own continue), this only counts"""
        self.dropped_by[reason] = self.dropped_by.get(reason, 0) + 1

    def report(self: Self) -> str:
        """one stable line, e.g. rows_in=1000 rows_out=842 dropped={match_on:150, min_text_len:8};
        the dropped= part is omitted entirely when nothing was dropped"""
        line: str = f"rows_in={self.rows_in} rows_out={self.rows_out}"
        if self.dropped_by:
            dropped: str = ", ".join(f"{reason}:{count}" for reason, count in self.dropped_by.items())
            line += f" dropped={{{dropped}}}"
        return line


def shard_of(datastream: Any, read_shards: int, shard_index: int) -> Any:
    """select one reader's shard of a loaded hub dataset, clamped to what the dataset can
    actually split. The clamp is the load-bearing part: oversharding is broken at the hub
    layer (a streaming IterableDataset with fewer file shards than requested raises or, for
    generator-backed sources, silently hands whole shards to index 0), so every shard
    instance derives the SAME effective count from the dataset itself -- min(declared,
    available) -- and an index beyond it yields nothing. Streaming IterableDatasets shard at
    file granularity (no duplicated bytes); built map-style Datasets shard by row index
    (views over the same arrow table). Must run BEFORE any operator touches the dataset, so
    the split lands at the source"""
    total: int | None = getattr(datastream, "num_shards", None)
    if total is None:
        total = len(datastream) if hasattr(datastream, "__len__") else 1
    effective: int = min(read_shards, max(total, 1))
    if shard_index >= effective:
        return iter(())
    return datastream.shard(num_shards=effective, index=shard_index)


def rebuild_task(task: tuple[Any, ...]) -> Any:
    """rebuild the pydantic task model from its frozen field-order tuple (to_tuple round trip)"""
    from relmedner.models import FullmapTask, ScriptTask

    kind = task[0]
    model = {"script": ScriptTask, "fullmap": FullmapTask}.get(str(kind))
    if model is None:
        raise ValueError(f"unknown task type {task!r}")
    return model(**dict(zip(model.model_fields, task, strict=True)))


def select_declared_columns(dataset: Any, columns_out: tuple[str, ...], match_on: tuple[tuple[str, Any], ...]) -> Any:
    """prune a hub dataset to the columns this declaration actually reads.

    Every hf source iterated whole rows, so the hub decoder materialized every column of every
    row into a python dict and the stream then read two or three of them: the seven reddit
    fullmap ingests declare `text` + `communityName` out of ~30 columns, and they are the
    largest sources in the registry. Projecting first makes the decoder skip the rest
    (on-the-fly for a streaming IterableDataset, at the arrow level for a built Dataset).

    Returns the dataset unchanged when there is nothing to prune (every column is needed) or when
    the schema is unknown (`features` falsy), so a source whose features only arrive with the
    first row keeps working. A declared column the dataset does not have is left out rather than
    raised for: `row.get(column)` already yields None for it, which is how the docred test split
    (no `labels` key) ships entities only.
    """
    features = getattr(dataset, "features", None)
    if not features:
        return dataset
    needed: list[str] = list(dict.fromkeys([*columns_out, *(column for column, _values in match_on)]))
    wanted: list[str] = [name for name in needed if name in features]
    return dataset if len(wanted) == len(features) else dataset.select_columns(wanted)


class DataStream(ABC):
    SOURCE: ClassVar[str]

    SHARDED_READS: ClassVar[bool] = False
    """whether rows() knows how to select only its shard of the source; a subclass that
    flips this to True must make the (read_shards, shard_index) pair deterministic and
    non-overlapping, with the union of all shards' rows equal to the unsharded row list"""

    read_shards: int = 1
    """class-level default so bare test doubles that never call the shared __init__ still
    stream() and report; real sources get the declared value through the base __init__"""

    shard_index: int = 0
    """which shard of read_shards this instance streams; see SHARDED_READS"""

    task: tuple[Any, ...]
    """the whole frozen task tuple (discriminated by its leading type value), carried
    positionally in the frozen payload tuple; the pipeline rebuilds the task model from it
    so script dispatch and fullmap mining share one stream shape"""

    name: str
    """the source key every yielded row is stamped with. It must equal the declared dataset's
    DatasetBase.row_key exactly: the pipeline looks the source's mixing weight up by this string,
    so any drift is a KeyError partway through a run rather than a wrong number"""

    weight: float
    """the declared per-source mixing weight, carried positionally in the frozen payload tuple"""

    filters: RowFilters | None
    """the DECLARED declarative row filters (models.RowFilters); None stays None, so this slot
    keeps answering "what did the dataset declare" -- the evaluator reads effective_filters"""

    effective_filters: RowFilters
    """the filters rows() actually applies: the declared value, else a default RowFilters()
    computed once in __init__, so the ALWAYS-ON max_tokens cap (row_filters.first_drop_reason)
    reaches even unfiltered sources; rows under the cap stay byte-identical, only over-cap rows
    are new drops on the declared-None path"""

    def __init__(
        self,
        task: tuple[Any, ...] = (),
        weight: float = 1.0,
        *,
        filters: RowFilters | None = None,
        sample_rate: float = 1.0,
        read_shards: int = 1,
        shard_index: int = 0,
    ) -> None:
        """the single shared entry point every DataStream subclass builds on: it owns the
        frozen payload's leading fields, so each subclass ctor only adds its source-specific
        ones after super().__init__(task, weight).

        Parameter order must match DatasetBase.to_tuple's field order, because
        registry.build_stream unpacks the declared payload positionally: task, weight, then
        the source-specific fields. US-008 added that keyword-only filters parameter (the
        defaults keep no-arg test doubles constructible); it rides here rather than each subclass
        ctor so every source shares one filter slot. sample_rate rides the same way: the
        declared mixing-ratio fraction stream() keeps, defaulting to everything. read_shards
        and shard_index ride the stream_args envelope the same way: the declared fan-out of
        this source's read over whole-bundle shard instances, defaulting to one full pass.
        """
        self.task: tuple[Any, ...] = tuple(task)
        self.weight: float = weight
        self.filters: RowFilters | None = filters
        self.sample_rate: float = sample_rate
        # fail loud BEFORE the pipeline submits: an unshardable source declared with
        # read_shards > 1 would otherwise either duplicate every row (every shard streaming
        # the whole source) or silently stay serial
        if read_shards < 1 or not 0 <= shard_index < read_shards:
            raise ValueError(f"invalid shard envelope: read_shards={read_shards}, shard_index={shard_index}")
        if read_shards > 1 and not type(self).SHARDED_READS:
            raise ValueError(f"{type(self).__name__} does not shard its read yet; declare read_shards only on sources that support it")
        self.read_shards: int = read_shards
        self.shard_index: int = shard_index
        # computed ONCE: the evaluator always gets a RowFilters, so the ALWAYS-ON token cap
        # applies even when the dataset declared none; self.filters keeps the DECLARED value
        self.effective_filters: RowFilters = filters if filters is not None else RowFilters()
        # US-009 quality counters; rows() resets them at the start of every pass so a direct
        # rows() call and one going through stream() report identically
        self.stats: StreamStats = StreamStats()
        # default label so bare test doubles (which stamp a constant source key in their rows()
        # but never declare self.name) still surface a quality line instead of AttributeError-ing;
        # real sources overwrite it with the declared row key right after super().__init__
        self.name: str = type(self).__name__
        # compile at construction so an invalid pattern raises re.error HERE, never mid-stream;
        # the evaluator re-searches the pattern strings, which hits re's internal pattern cache
        if filters is not None and filters.include_regex is not None:
            re.compile(filters.include_regex)
        if filters is not None and filters.exclude_regex is not None:
            re.compile(filters.exclude_regex)

    @abstractmethod
    def rows(self: Self) -> Iterator[StreamedRow]:
        """yields every row this source declares, unbounded"""

    def _sampled(self: Self, rows: Iterator[StreamedRow]) -> Iterator[StreamedRow]:
        """keep a row iff blake2b(source key + row repr) lands under sample_rate * 2**64.
        Content-addressed so a resumed or re-parallelized pass makes the identical keep/drop
        decision per row, and two sources never sample each other's rows; repr over the plain
        str/int/tuple payload is deterministic across processes. Only wired when the declared
        rate is under 1.0, so the default path pays nothing"""
        cutoff = int(self.sample_rate * 2**64)
        for row in rows:
            digest = blake2b(f"{self.name}\x00{row!r}".encode(), digest_size=8).digest()
            if int.from_bytes(digest, "big") < cutoff:
                yield row
            else:
                self.stats.drop("sample_rate")

    def stream(self: Self, config: RunConfig) -> Iterator[StreamedRow]:
        """wraps rows() with content-hash sampling (declared sample_rate) and the sample limit,
        and surfaces the quality line exactly once per pass, at generator exhaustion. Sampling
        sits BEFORE the islice so sample_limit bounds KEPT rows (a test-run slice of a sampled
        source is a slice of the sample, not of the raw stream); the counters themselves are
        populated by rows(), which owns the match_on and evaluator call sites where a drop's
        reason is known, plus _sampled for its own reason. The finally (not a bare tail call)
        makes the line land even when rows() raises ZeroYieldError at source exhaustion, so the
        quality report is never swallowed by the guard, and before the exception reaches the
        caller"""
        rows: Iterator[StreamedRow] = self.rows()
        if self.sample_rate < 1.0:
            rows = self._sampled(rows)
        if config.sample_limit is not None:
            rows = islice(rows, config.sample_limit)
        try:
            yield from rows
        finally:
            _QUALITY_LOG.info("ingest quality %s: %s", self._stats_label(), self.stats.report())

    def _stats_label(self: Self) -> str:
        """the quality-line label: the declared row key, suffix-tagged when this pass is one
        shard of a fanned-out read so K shard lines stay attributable"""
        return self.name if self.read_shards == 1 else f"{self.name}[shard {self.shard_index}/{self.read_shards}]"

    def zero_yield_guard(self: Self, candidates: int) -> None:
        """fail loud when a WHOLE pass kept zero candidate rows (US-008: the silent-empty-
        training-set bug made impossible to recreate through filters OR the ALWAYS-ON cap).
        The caller owns what counts as a candidate (local avro counts records read, the hf
        streams count post-match_on rows). On a sharded read one shard may legitimately hold
        zero passing rows (a trailing file shard under the cap), so the raise degrades to a
        WARNING carrying the shard label; row-count truth for sharded sources stays with the
        probe receipts and the downstream census"""
        if candidates <= 0 or self.stats.rows_out > 0:
            return
        if self.read_shards > 1:
            _QUALITY_LOG.warning("ingest quality %s: %d candidate rows, 0 passed", self._stats_label(), candidates)
            return
        raise ZeroYieldError(f"filters {self.filters} dropped 100% of {candidates} rows from {self.name}")
