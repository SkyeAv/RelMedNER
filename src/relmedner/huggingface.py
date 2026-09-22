from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, Self

from datasets import load_dataset

from relmedner.models import RowFilters
from relmedner.row_filters import first_drop_reason
from relmedner.streams import DataStream, StreamedRow, ZeroYieldError


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
        *,
        filters: RowFilters | None = None,
    ) -> None:
        # parameter order must match DatasetBase.to_tuple's field order (build_stream unpacks
        # the declared payload positionally): task, weight, then the hf-specific columns;
        # filters is keyword-only and rides the shared base __init__ (US-008)
        super().__init__(task, weight, filters=filters)
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

        # filters is None: the historical unfiltered path, byte-identical (no counting, no guard)
        if self.filters is None:
            for row in datastream:
                if self.apply_match(row):
                    yield (self.name, (self.task, tuple(row.get(column) for column in self.columns_out)))
            return

        rows_in = 0
        rows_out = 0
        for row in datastream:
            if not self.apply_match(row):
                continue
            rows_in += 1
            values: tuple[Any, ...] = tuple(row.get(column) for column in self.columns_out)
            if first_drop_reason(values, self.filters) is not None:
                continue
            rows_out += 1
            yield (self.name, (self.task, values))
        # fail-loud zero-yield guard: a filter that drops every row of a non-empty source is the
        # silent-empty-training-set bug; a genuinely empty source (rows_in == 0) is recorded, not guarded
        if rows_in > 0 and rows_out == 0:
            raise ZeroYieldError(f"filters {self.filters} dropped 100% of {rows_in} rows from {self.name}")
