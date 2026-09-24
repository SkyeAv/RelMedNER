from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, Self

from datasets import load_dataset

from relmedner.models import RowFilters
from relmedner.row_filters import first_drop_reason
from relmedner.streams import DataStream, StreamedRow, ZeroYieldError, select_declared_columns


class HuggingFaceParquetDataStream(DataStream):
    """parquet-builder ingest over one per-config file on the hub's refs/convert/parquet branch.

    WHY this source kind exists (script-era hub repos like bigbio/ehr_rel):
    - the repo's main branch carries only loading-script files, and current datasets refuses
      script datasets outright, so load_dataset("bigbio/ehr_rel") cannot work at all;
    - loading the whole refs/convert/parquet revision in one call fails on mixed schemas: the
      hub auto-conversion publishes one parquet schema per config, so the union across configs
      is not a single table;
    - one per-config parquet file is therefore the only working route. The scheduler's
      wenceslaus probes proved the exact call this class makes loads end to end:
      load_dataset("parquet",
      data_files="hf://datasets/bigbio/ehr_rel@refs/convert/parquet/ehr_rel_bigbio_pairs/train/0000.parquet",
      split="train").
    Unlike the json builder (relmedner.hf_json) there is no cold-cache streaming hazard: the
    auto-convert files are already typed arrow, so a plain non-streaming load has no type
    promotion step to get wrong; the streaming kwarg stays absent for symmetry, not necessity.
    """

    SOURCE: ClassVar[str] = "hf_parquet"

    def __init__(
        self: Self,
        task: tuple[Any, ...],
        weight: float,
        dataset: str,
        file: str,
        split: str | None,
        match_on: tuple[tuple[str, tuple[str, ...]], ...] | None,
        columns_out: tuple[str, ...],
        *,
        filters: RowFilters | None = None,
        sample_rate: float = 1.0,
    ) -> None:
        # positional contract: the payload DatasetBase.to_tuple() produces for HuggingFaceParquetDataset
        # (model field order minus source); registry.build_stream splats it into this __init__;
        # filters and sample_rate are keyword-only and ride the shared base __init__
        super().__init__(task, weight, filters=filters, sample_rate=sample_rate)
        self.name: str = dataset
        self.dataset: str = dataset
        self.file: str = file
        self.split: str | None = split
        self.match_on: tuple[tuple[str, frozenset[str]], ...] = tuple((column, frozenset(values)) for column, values in match_on) if match_on else ()
        self.columns_out: tuple[str, ...] = columns_out

    def apply_match(self: Self, row: dict[str, Any]) -> bool:
        return all(row.get(column) in values for column, values in self.match_on)

    def rows(self: Self) -> Iterator[StreamedRow]:
        # no streaming kwarg, on purpose: the per-config parquet file is already typed arrow,
        # so the plain load returns the declared dtypes with no promotion step
        datastream = load_dataset("parquet", data_files=f"hf://datasets/{self.dataset}@refs/convert/parquet/{self.file}", split=self.split)
        datastream = select_declared_columns(datastream, self.columns_out, self.match_on)

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
        # silent-empty-training-set bug; a genuinely empty source (rows_in == 0) is not an error
        if rows_in > 0 and rows_out == 0:
            raise ZeroYieldError(f"filters {self.filters} dropped 100% of {rows_in} rows from {self.name}")
