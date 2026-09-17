from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from itertools import islice
from typing import ClassVar, Self

from relmedner.models import RunConfig
from relmedner.types import ScriptPayload

StreamedRow = tuple[str, ScriptPayload]


class DataStream(ABC):
    SOURCE: ClassVar[str]

    @abstractmethod
    def rows(self: Self) -> Iterator[StreamedRow]:
        """yields every row this source declares, unbounded"""

    def stream(self: Self, config: RunConfig) -> Iterator[StreamedRow]:
        if config.sample_limit is None:
            return self.rows()
        return islice(self.rows(), config.sample_limit)
