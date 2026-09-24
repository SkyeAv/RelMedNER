from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, Self

from datasets import load_dataset

from relmedner.models import RowFilters
from relmedner.row_filters import first_drop_reason
from relmedner.streams import DataStream, StreamedRow, StreamStats, ZeroYieldError, select_declared_columns


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
        datastream = select_declared_columns(datastream, self.columns_out, self.match_on)

        # US-009: one stats record per pass over the source, reset here (not in stream()) so a
        # direct rows() call is accounted identically; counting rides the same loop for the
        # historical unfiltered path -- the rows it yields stay byte-identical, only the
        # counters (and the match_on attribution, silent before US-009) are new. rows_in counts
        # EVERY row read (pre-match_on) so rows_in - rows_out equals the sum of dropped_by,
        # per the US-009 spec example; the zero-yield guard keeps its own post-match_on
        # candidate count so its US-008 semantics (a match_on-emptied source is recorded, not
        # guarded) are untouched
        self.stats = StreamStats()
        candidates = 0
        for row in datastream:
            self.stats.rows_in += 1
            if not self.apply_match(row):
                self.stats.drop("match_on")
                continue
            candidates += 1
            values: tuple[Any, ...] = tuple(row.get(column) for column in self.columns_out)
            reason: str | None = first_drop_reason(values, self.effective_filters)
            if reason is not None:
                self.stats.drop(reason)
                continue
            self.stats.rows_out += 1
            yield (self.name, (self.task, values))
        # fail-loud zero-yield guard (US-008, now covering the ALWAYS-ON cap): a declared filter
        # OR the token cap that drops every candidate row of a non-empty source is the
        # silent-empty-training-set bug; a genuinely empty source (0 candidates, e.g. everything
        # match_on-dropped) is recorded, not guarded
        if candidates > 0 and self.stats.rows_out == 0:
            raise ZeroYieldError(f"filters {self.filters} dropped 100% of {candidates} rows from {self.name}")
