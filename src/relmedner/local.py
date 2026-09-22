from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Self

from relmedner.constants import DATA
from relmedner.streams import DataStream, StreamedRow


class LocalDataStream(DataStream):
    """streams rows from a header-delimited local file (TSV/CSV picked by suffix, tab default);
    mirrors HuggingFaceDataStream's shape so script dispatch and fullmap mining share one
    stream shape downstream. Relative paths resolve as-is when they exist on disk (caller's
    CWD), else against the package data dir so in-repo corpora work from any CWD and inside
    the worker container where the package is installed."""

    SOURCE: ClassVar[str] = "local"
    DELIMITERS: ClassVar[dict[str, str]] = {".tsv": "\t", ".csv": ","}

    def __init__(
        self: Self,
        task: tuple[Any, ...],
        path: str | Path,
        columns_out: tuple[str, ...],
        match_on: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
    ) -> None:
        self.task: tuple[Any, ...] = tuple(task)
        candidate: Path = Path(path)
        try:
            resolved: Path = candidate if candidate.is_file() else Path(str(DATA)) / path
        except OSError:
            # python 3.13 pathlib propagates PermissionError from is_file() probes; an
            # unreadable parent means the caller's path is not a usable file either way
            resolved = Path(str(DATA)) / path
        self.path: Path = resolved
        self.name: str = self.path.stem
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
            for row in csv.DictReader(handle, delimiter=delimiter):
                if self.apply_match(row):
                    yield (self.name, (self.task, tuple(row.get(column) for column in self.columns_out)))
