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

    def row_generator(self: Self) -> Iterator[tuple[str, tuple[tuple[Union[Optional[str], str], ...], ...], tuple[Optional[str], ...]]]:
        datastream = load_dataset(self.dataset, self.subset, split=self.split, streaming=True)

        if self.match_on:
            values: tuple[str, ...] = self.match_on.values()

        for row in datastream:
            outstream: tuple[Optional[str], ...] = tuple(row.get(column) for column in self.columns_out)
            matchstream: tuple[tuple[Union[Optional[str], str], ...], ...] = ()

            if self.match_on:
                matchstream = ((row.get(column) for column in self.match_on.keys()), values)

            yield (type, matchstream, outstream)
