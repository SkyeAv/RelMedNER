from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from itertools import islice
from typing import Any, ClassVar, Self

from relmedner.models import RunConfig
from relmedner.types import ScriptPayload

StreamedRow = tuple[str, ScriptPayload]


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

    def __init__(self, task: tuple[Any, ...] = (), weight: float = 1.0) -> None:
        """the single shared entry point every DataStream subclass builds on: it owns the
        frozen payload's leading fields, so each subclass ctor only adds its source-specific
        ones after super().__init__(task, weight).

        Parameter order must match DatasetBase.to_tuple's field order, because
        registry.build_stream unpacks the declared payload positionally: task, weight, then
        the source-specific fields. US-008 will extend this signature with a keyword-only
        filters parameter (the defaults keep no-arg test doubles constructible).
        """
        self.task: tuple[Any, ...] = tuple(task)
        self.weight: float = weight

    @abstractmethod
    def rows(self: Self) -> Iterator[StreamedRow]:
        """yields every row this source declares, unbounded"""

    def stream(self: Self, config: RunConfig) -> Iterator[StreamedRow]:
        if config.sample_limit is None:
            return self.rows()
        return islice(self.rows(), config.sample_limit)
