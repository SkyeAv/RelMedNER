from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from typing import Any, ClassVar, Self
from urllib.parse import urlencode

from datasets import load_dataset

from relmedner.models import RowFilters
from relmedner.row_filters import first_drop_reason
from relmedner.streams import DataStream, StreamedRow, StreamStats, ZeroYieldError


def parquet_shard_urls(dataset: str, subset: str, split: str) -> list[str]:
    """shard URLs for one (config, split) of a hub repo's auto parquet conversion branch.

    WHY the datasets-server listing instead of one hardcoded 0000.parquet URL: a conversion can
    hold several shards per split, and hardcoding the first shard would silently ship a prefix of
    the split -- the exact silent-partial-data failure this pipeline refuses. The listing is the
    hub's own inventory of the conversion (no auth for public repos); measured working for
    bigbio/gad on 2026-09-23 (43 files listed, one per config-split pair). An empty match is a
    declaration bug (typo'd subset/split) and fails loud here, at stream construction time.
    """
    endpoint = "https://datasets-server.huggingface.co/parquet?" + urlencode({"dataset": dataset})
    with urllib.request.urlopen(endpoint) as response:
        listing = json.load(response)
    shards: list[str] = [entry["url"] for entry in listing.get("parquet_files", []) if entry.get("config") == subset and entry.get("split") == split]
    if not shards:
        raise ValueError(f"no parquet shards listed for {dataset} config {subset!r} split {split!r}")
    return shards


class HuggingFaceParquetDataStream(DataStream):
    """parquet-conversion ingest for hub repos whose own builder is a loading script.

    bigbio/gad (the landing case) ships gad.py as its builder; datasets 5.x refuses script
    builders outright (RuntimeError: Dataset scripts are no longer supported), so load_dataset on
    the repo path cannot stream it. The hub's auto conversion branch (refs/convert/parquet) holds
    the same rows as typed parquet, one shard set per (config, split): measured 43 files /
    10,717,252 bytes for bigbio/gad, every config-split a single shard. A data_files list of the
    listing's resolve URLs generates exactly one builder split named "train", which is what
    rows() requests; the builder download is a small parquet fetch (columnar typed, so the
    cold-cache streaming corruption that forced hf_json non-streaming does not exist here), and
    the split stays small enough that a non-streaming build is the honest simple choice.
    """

    SOURCE: ClassVar[str] = "hf_parquet"

    def __init__(
        self: Self,
        task: tuple[Any, ...],
        weight: float,
        dataset: str,
        subset: str,
        split: str,
        match_on: tuple[tuple[str, tuple[str, ...]], ...] | None,
        columns_out: tuple[str, ...],
        *,
        filters: RowFilters | None = None,
    ) -> None:
        # positional contract: the payload HuggingFaceParquetDataset.to_tuple() produces (model
        # field order minus source); registry.build_stream splats it into this __init__;
        # filters is keyword-only and rides the shared base __init__
        super().__init__(task, weight, filters=filters)
        self.name: str = dataset
        self.dataset: str = dataset
        self.subset: str = subset
        self.split: str = split
        self.match_on: tuple[tuple[str, frozenset[str]], ...] = tuple((column, frozenset(values)) for column, values in match_on) if match_on else ()
        self.columns_out: tuple[str, ...] = columns_out

    def apply_match(self: Self, row: dict[str, Any]) -> bool:
        return all(row.get(column) in values for column, values in self.match_on)

    def rows(self: Self) -> Iterator[StreamedRow]:
        datastream = load_dataset(
            "parquet",
            data_files=parquet_shard_urls(self.dataset, self.subset, self.split),
            split="train",
        )

        # one stats record per pass over the source, matching the hf stream's accounting:
        # rows_in counts EVERY row read (pre-match_on) so rows_in - rows_out equals the sum of
        # dropped_by; the zero-yield guard keeps its own post-match_on candidate count
        self.stats = StreamStats()
        candidates = 0
        for row in datastream:
            self.stats.rows_in += 1
            if not self.apply_match(row):
                self.stats.drop("match_on")
                continue
            candidates += 1
            values: tuple[Any, ...] = tuple(row.get(column) for column in self.columns_out)
            if self.filters is not None:
                reason: str | None = first_drop_reason(values, self.filters)
                if reason is not None:
                    self.stats.drop(reason)
                    continue
            self.stats.rows_out += 1
            yield (self.name, (self.task, values))
        # fail-loud zero-yield guard: a filter that drops every candidate row of a non-empty
        # source is the silent-empty-training-set bug; a genuinely empty source (0 candidates,
        # e.g. everything match_on-dropped) is recorded, not guarded
        if self.filters is not None and candidates > 0 and self.stats.rows_out == 0:
            raise ZeroYieldError(f"filters {self.filters} dropped 100% of {candidates} rows from {self.name}")
