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

    @abstractmethod
    def rows(self: Self) -> Iterator[StreamedRow]:
        """yields every row this source declares, unbounded"""

    def stream(self: Self, config: RunConfig) -> Iterator[StreamedRow]:
        if config.sample_limit is None:
            return self.rows()
        return islice(self.rows(), config.sample_limit)
