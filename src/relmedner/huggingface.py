from __future__ import annotations

from typing import Self, Optional, Any
from collections.abc import Iterator

from datasets import load_dataset


class HuggingFaceDataStream:
    def __init__(
        self: Self,
        type: str,
        dataset: str,
        subset: Optional[str],
        split: Optional[str],
        match_on: Optional[tuple[tuple[str, tuple[str, ...]], ...]],
        columns_out: tuple[str, ...],
    ) -> None:
        self.type: str = type
        self.dataset: str = dataset
        self.subset: Optional[str] = subset
        self.split: Optional[str] = split
        self.match_on: tuple[tuple[str, frozenset[str]], ...] = tuple((column, frozenset(values)) for column, values in match_on) if match_on else ()
        self.columns_out: tuple[str, ...] = columns_out

    def apply_match(self: Self, row: dict[str, Any]) -> bool:
        return all(row.get(column) in values for column, values in self.match_on)

    def generate_rows(self: Self) -> Iterator[tuple[str, tuple[Optional[str], ...]]]:
        datastream = load_dataset(self.dataset, self.subset, split=self.split, streaming=True)

        for row in datastream:
            if self.apply_match(row):
                yield (self.type, tuple(row.get(column) for column in self.columns_out))
