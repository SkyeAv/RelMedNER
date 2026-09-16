from __future__ import annotations

from typing import Self, Optional, Union
from collections.abc import Iterator

from datasets import load_dataset

class HuggingFaceDataStream:
    def __init__(
        self: Self,
        type: str,
        dataset: str,
        subset: Optional[str],
        split: Optional[str],
        match_on: Optional[list[dict[str, Union[str, list[str]]]]],
        columns_out: list[str]
    ) -> None:
        self.type: str = type
        self.dataset: str = dataset
        self.subset: Optional[str] = subset
        self.split: Optional[str] = split
        self.match_on: Optional[list[dict[str, Union[str, list[str]]]]] = match_on
        self.columns_out: list[str] = columns_out

    def apply_match(self: Self, row) -> bool:
        return any(any(transform.get("values") in row.get(column) for column in transform.get("column")) for transform in self.match_on)

    def generate_rows(self: Self) -> Iterator[tuple[str, tuple[Optional[str], ...]]]:
        datastream = load_dataset(self.dataset, self.subset, split=self.split, streaming=True)

        for row in datastream:
            if apply_match:
                yield (type, (row.get(column) for column in columns_out))
