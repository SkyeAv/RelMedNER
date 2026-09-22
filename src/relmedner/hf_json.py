from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, Self

from datasets import load_dataset

from relmedner.models import RowFilters
from relmedner.row_filters import first_drop_reason
from relmedner.streams import DataStream, StreamedRow, ZeroYieldError


class HuggingFaceJsonDataStream(DataStream):
    """json-builder ingest over an hf:// URL instead of the hub's parquet conversion.

    Two measured blockers force this route rather than the "hf" source on
    knowledgator/PubMedAbstractsNER (datasets 5.0.1):
    - load_dataset("knowledgator/PubMedAbstractsNER", split="train", streaming=True) dies inside
      Features.from_dict with KeyError: 'feature': the repo's dataset_infos.json declares
      tokenized_text in the old style {"dtype": "string", "_type": "Sequence"} with no "feature"
      key; passing data_files= to the repo path does not help, and a features= override cannot
      cast the list<item: list<item: string>> ner column.
    - the json builder MUST read non-streaming: on a cold cache streaming promotes every ner
      cell to utf8 straight from the raw JSON text (int offsets and the label all come back as
      strings, the label wrapped in literal quotes), while a full arrow build returns the
      correct int/int/str; a warm streaming read only looks clean because it re-reads that
      arrow cache. Cold cost: 152MB download + ~8s arrow build, then ~0.5s warm reopen.
    """

    SOURCE: ClassVar[str] = "hf_json"

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
    ) -> None:
        # positional contract: the payload DatasetBase.to_tuple() produces for HuggingFaceJsonDataset
        # (model field order minus source); registry.build_stream splats it into this __init__;
        # filters is keyword-only and rides the shared base __init__ (US-008)
        super().__init__(task, weight, filters=filters)
        self.name: str = dataset
        self.dataset: str = dataset
        self.file: str = file
        self.split: str | None = split
        self.match_on: tuple[tuple[str, frozenset[str]], ...] = tuple((column, frozenset(values)) for column, values in match_on) if match_on else ()
        self.columns_out: tuple[str, ...] = columns_out

    def apply_match(self: Self, row: dict[str, Any]) -> bool:
        return all(row.get(column) in values for column, values in self.match_on)

    def rows(self: Self) -> Iterator[StreamedRow]:
        # no streaming kwarg, on purpose: see the class docstring cold-cache hazard
        datastream = load_dataset("json", data_files=f"hf://datasets/{self.dataset}/{self.file}", split=self.split)

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
