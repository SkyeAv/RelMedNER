from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, Self

from datasets import load_dataset

from relmedner.models import RowFilters
from relmedner.row_filters import first_drop_reason
from relmedner.streams import DataStream, StreamedRow, StreamStats, ZeroYieldError, select_declared_columns


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
        sample_rate: float = 1.0,
    ) -> None:
        # positional contract: the payload DatasetBase.to_tuple() produces for HuggingFaceJsonDataset
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
        # no streaming kwarg, on purpose: see the class docstring cold-cache hazard
        datastream = load_dataset("json", data_files=f"hf://datasets/{self.dataset}/{self.file}", split=self.split)
        datastream = select_declared_columns(datastream, self.columns_out, self.match_on)

        # every pass counts (US-009 shape), INCLUDING the declared-filters-None one: the
        # historical None fast path skipped stats, the evaluator, and the quality line, which
        # let over-cap rows stream through unfiltered sources with no accounting. The ALWAYS-ON
        # token cap must reach this source like every other, so the fast path is gone; rows
        # under the cap yield byte-identically, only over-cap rows are new drops (log/stats-only
        # difference otherwise)
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
        # fail-loud zero-yield guard, matching the other hf stream: a declared filter OR the
        # ALWAYS-ON token cap that drops every candidate row of a non-empty source is the
        # silent-empty-training-set bug; a genuinely empty source (0 candidates, e.g. everything
        # match_on-dropped) is recorded, not guarded
        if candidates > 0 and self.stats.rows_out == 0:
            raise ZeroYieldError(f"filters {self.filters} dropped 100% of {candidates} rows from {self.name}")
