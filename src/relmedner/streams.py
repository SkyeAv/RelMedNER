from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from itertools import islice
from typing import Any, ClassVar, Self

from relmedner.models import RowFilters, RunConfig
from relmedner.types import ScriptPayload

StreamedRow = tuple[str, ScriptPayload]


class ZeroYieldError(RuntimeError):
    """a filtered source dropped 100% of its rows: the historical silent zero-yield ingest bug
    (a declaration typo quietly shipping an empty training set) made impossible to recreate
    through filters. Only raised when filters are declared AND the source was non-empty"""


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
        if config.sample_limit is None:
            return self.rows()
        return islice(self.rows(), config.sample_limit)
