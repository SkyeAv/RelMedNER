from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
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
    """a filtered source dropped 100% of its rows: the historical silent zero-yield ingest bug
    (a declaration typo quietly shipping an empty training set) made impossible to recreate
    through filters. Only raised when filters are declared AND the source was non-empty"""


@dataclass
class StreamStats:
    """per-pass data-quality counters for one DataStream (US-009)

    WHY a dataclass and not a StrictBase model: this file is stdlib-only (models.py owns the
    pydantic imports; streams.py must stay importable without them) and the counters are mutated
    row by row mid-iteration, a plain mutable-record job rather than validation. dropped_by keys
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


def rebuild_task(task: tuple[Any, ...]) -> Any:
    """rebuild the pydantic task model from its frozen field-order tuple (to_tuple round trip)"""
    from relmedner.models import FullmapTask, ScriptTask

    kind = task[0]
    model = {"script": ScriptTask, "fullmap": FullmapTask}.get(str(kind))
    if model is None:
        raise ValueError(f"unknown task type {task!r}")
    return model(**dict(zip(model.model_fields, task, strict=True)))


class DataStream(ABC):
    SOURCE: ClassVar[str]

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
    """declarative row filters (models.RowFilters), applied in rows() after match_on; None means
    the unfiltered path, which stays byte-identical"""

    def __init__(self, task: tuple[Any, ...] = (), weight: float = 1.0, *, filters: RowFilters | None = None) -> None:
        """the single shared entry point every DataStream subclass builds on: it owns the
        frozen payload's leading fields, so each subclass ctor only adds its source-specific
        ones after super().__init__(task, weight).

        Parameter order must match DatasetBase.to_tuple's field order, because
        registry.build_stream unpacks the declared payload positionally: task, weight, then
        the source-specific fields. US-008 added that keyword-only filters parameter (the
        defaults keep no-arg test doubles constructible); it rides here rather than each subclass
        ctor so every source shares one filter slot.
        """
        self.task: tuple[Any, ...] = tuple(task)
        self.weight: float = weight
        self.filters: RowFilters | None = filters
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

    def stream(self: Self, config: RunConfig) -> Iterator[StreamedRow]:
        """wraps rows() with the sample limit and surfaces the quality line exactly once per
        pass, at generator exhaustion. The counters themselves are populated by rows(), which
        owns the match_on and evaluator call sites where a drop's reason is known; stream() only
        hands rows out. The finally (not a bare tail call) makes the line land even when rows()
        raises ZeroYieldError at source exhaustion, so the quality report is never swallowed by
        the guard, and before the exception reaches the caller"""
        rows: Iterator[StreamedRow] = self.rows() if config.sample_limit is None else islice(self.rows(), config.sample_limit)
        try:
            yield from rows
        finally:
            _QUALITY_LOG.info("ingest quality %s: %s", self.name, self.stats.report())
