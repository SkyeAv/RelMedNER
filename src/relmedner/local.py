from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Self

from fastavro import reader

from relmedner.constants import DATA
from relmedner.models import RowFilters
from relmedner.row_filters import first_drop_reason
from relmedner.streams import DataStream, StreamedRow, ZeroYieldError


class LocalAvroDataStream(DataStream):
    """streams records out of a local avro container declared by path in ingests.yaml

    The hf streams pull their rows from the hub; this one reads a file the operator built
    out-of-band (the CTKP interventions extract), so a dataset that cannot live on the hub --
    licensing, size, or because it is rebuilt per AACT snapshot -- still rides the same
    declarative ingest path as everything else.
    """

    SOURCE: ClassVar[str] = "local"

    def __init__(self: Self, task: tuple[Any, ...], weight: float, path: str, *, filters: RowFilters | None = None) -> None:
        # parameter order must match DatasetBase.to_tuple's field order, because build_stream
        # unpacks the declared payload positionally: task, weight, then the local-specific path;
        # filters is keyword-only and rides the shared base __init__ (US-008)
        super().__init__(task, weight, filters=filters)
        # the pipeline stamps every row with weights[source], so this key is LocalAvroDataset.row_key
        # verbatim: the declared path, not its basename (two distinct files may share a name and
        # must still be able to declare different weights)
        self.name: str = path
        self.path: str = path

    def rows(self: Self) -> Iterator[StreamedRow]:
        # the whole record ships as a single value so the receiving script owns the shape;
        # avro's reader is already lazy, so a 1M-record container never lands in memory at once
        with Path(self.path).expanduser().open("rb") as handle:
            # filters is None: the historical unfiltered path, byte-identical (no counting, no guard)
            if self.filters is None:
                for record in reader(handle):
                    yield (self.name, (self.task, (record,)))
                return

            rows_in = 0
            rows_out = 0
            for record in reader(handle):
                rows_in += 1
                # the text rule applies over the record's own values (same rule as the hf projection)
                if first_drop_reason(tuple(record.values()), self.filters) is not None:
                    continue
                rows_out += 1
                yield (self.name, (self.task, (record,)))
            # fail-loud zero-yield guard: a filter that drops every row of a non-empty source is
            # the silent-empty-training-set bug; an empty file (rows_in == 0) is not an error
            if rows_in > 0 and rows_out == 0:
                raise ZeroYieldError(f"filters {self.filters} dropped 100% of {rows_in} rows from {self.name}")

class LocalDelimitedDataStream(DataStream):
    """streams rows from a header-delimited local file (TSV/CSV picked by suffix, tab default);
    mirrors HuggingFaceDataStream's shape so script dispatch and fullmap mining share one
    stream shape downstream. Relative paths resolve as-is when they exist on disk (caller's
    CWD), else against the package data dir so in-repo corpora work from any CWD and inside
    the worker container where the package is installed.

    This is a separate source kind from "local" rather than a suffix switch inside it: the avro
    stream ships whole records with no projection, while a delimited file has a header row that
    makes columns_out a real contract.
    """

    SOURCE: ClassVar[str] = "local_delimited"
    DELIMITERS: ClassVar[dict[str, str]] = {".tsv": "\t", ".csv": ","}

    def __init__(
        self: Self,
        task: tuple[Any, ...],
        weight: float,
        path: str | Path,
        columns_out: tuple[str, ...],
        match_on: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
        *,
        filters: RowFilters | None = None,
    ) -> None:
        # positional contract: the payload LocalDelimitedDataset.to_tuple produces (model field
        # order minus source); registry.build_stream splats it into this __init__;
        # filters is keyword-only and rides the shared base __init__ (US-008)
        super().__init__(task, weight, filters=filters)
        # the DECLARED path, not the resolved one and not its stem: the pipeline stamps every row
        # with weights[source], so this key is LocalDelimitedDataset.row_key verbatim, and two
        # distinct files sharing a basename must still be able to declare different weights
        self.name: str = str(path)
        candidate: Path = Path(path)
        try:
            resolved: Path = candidate if candidate.is_file() else Path(str(DATA)) / path
        except OSError:
            # python 3.13 pathlib propagates PermissionError from is_file() probes; an
            # unreadable parent means the caller's path is not a usable file either way
            resolved = Path(str(DATA)) / path
        self.path: Path = resolved
        self.columns_out: tuple[str, ...] = columns_out
        self.match_on: tuple[tuple[str, frozenset[str]], ...] = tuple((column, frozenset(values)) for column, values in match_on) if match_on else ()

    def apply_match(self: Self, row: dict[str, Any]) -> bool:
        return all(row.get(column) in values for column, values in self.match_on)

    def rows(self: Self) -> Iterator[StreamedRow]:
        try:
            exists: bool = self.path.is_file()
        except OSError:
            exists = False
        if not exists:
            raise FileNotFoundError(f"local source file not found: {self.path}")
        delimiter: str = self.DELIMITERS.get(self.path.suffix, "\t")
        with self.path.open(newline="", encoding="utf-8") as handle:
            # filters is None: the historical unfiltered path, byte-identical (no counting, no guard)
            if self.filters is None:
                for row in csv.DictReader(handle, delimiter=delimiter):
                    if self.apply_match(row):
                        yield (self.name, (self.task, tuple(row.get(column) for column in self.columns_out)))
                return

            rows_in = 0
            rows_out = 0
            for row in csv.DictReader(handle, delimiter=delimiter):
                if not self.apply_match(row):
                    continue
                rows_in += 1
                values: tuple[Any, ...] = tuple(row.get(column) for column in self.columns_out)
                if first_drop_reason(values, self.filters) is not None:
                    continue
                rows_out += 1
                yield (self.name, (self.task, values))
            # fail-loud zero-yield guard: a filter that drops every row of a non-empty source is the
            # silent-empty-training-set bug; a genuinely empty file (rows_in == 0) is not an error
            if rows_in > 0 and rows_out == 0:
                raise ZeroYieldError(f"filters {self.filters} dropped 100% of {rows_in} rows from {self.name}")
