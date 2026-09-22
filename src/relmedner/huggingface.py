from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, Self

from datasets import load_dataset

from relmedner.streams import DataStream, StreamedRow


class HuggingFaceDataStream(DataStream):
    SOURCE: ClassVar[str] = "hf"

    def __init__(
        self: Self,
        task: tuple[Any, ...],
        weight: float,
        dataset: str,
        subset: str | None,
        split: str | None,
        match_on: tuple[tuple[str, tuple[str, ...]], ...] | None,
        columns_out: tuple[str, ...],
    ) -> None:
        # parameter order must match DatasetBase.to_tuple's field order (build_stream unpacks
        # the declared payload positionally): task, weight, then the hf-specific columns
        super().__init__(task, weight)
        self.name: str = dataset
        self.dataset: str = dataset
        self.subset: str | None = subset
        self.split: str | None = split
        self.match_on: tuple[tuple[str, frozenset[str]], ...] = tuple((column, frozenset(values)) for column, values in match_on) if match_on else ()
        self.columns_out: tuple[str, ...] = columns_out

    def apply_match(self: Self, row: dict[str, Any]) -> bool:
        return all(row.get(column) in values for column, values in self.match_on)

    def rows(self: Self) -> Iterator[StreamedRow]:
        datastream = load_dataset(self.dataset, self.subset, split=self.split, streaming=True)

        for row in datastream:
            if self.apply_match(row):
                yield (self.name, (self.task, tuple(row.get(column) for column in self.columns_out)))
