from __future__ import annotations

from pathlib import Path
from typing import Any, Self
from uuid import uuid4

import apache_beam as beam
from fastavro import reader
from fastavro._write_py import Writer as AvroWriter


def shards_for(target: Path, prefix: str | None = None) -> tuple[Path, ...]:
    """complete shards beside this target in stable name order; .wip shards are in-progress bundles
    and never merge. shards carry the artifact name they were written under, which collection keeps
    even when the requested output file is named differently"""
    Stem: str = prefix or target.name
    return tuple(sorted(part for part in target.parent.glob(f"{Stem}.part-*") if not part.name.endswith(".wip")))


def merge_shards(target: Path, schema: dict[str, Any], prefix: str | None = None) -> Path:
    """concatenate the shards into the final avro file and delete them; with no shards the target is
    left exactly as found, so an already-merged artifact passes through untouched"""
    parts: tuple[Path, ...] = shards_for(target, prefix)
    if not parts:
        return target
    with target.open("wb") as out:
        sink: AvroWriter = AvroWriter(out, schema)
        for part in parts:
            with part.open("rb") as fo:
                for record in reader(fo):
                    sink.write(record)
        sink.flush()
    for part in parts:
        part.unlink()
    return target


class ShardWriter(beam.DoFn):
    """streams records straight into per-bundle avro shards beside the target — bypasses WriteFiles,
    whose streaming finalization stamps internal results at TIMESTAMP_MAX_VALUE and trips Flink's
    watermark-hold invariant ("TimestampCombiner moved element ... to earlier time (end of global
    window)"), killing the job on every attempt. a shard is only revealed under its final name on a
    clean bundle finish, so a retried bundle never leaves a half-written shard for the merge"""

    def __init__(self: Self, target: str, schema: dict[str, Any]) -> None:
        self.target: str = target
        self.schema: dict[str, Any] = schema
        # (part path, open file, writer) once the bundle has written its first record
        self._open: tuple[Path, Any, AvroWriter] | None = None

    def start_bundle(self) -> None:
        self._open = None

    def process(self, record: dict[str, Any]) -> None:
        if self._open is None:
            Part: Path = Path(f"{self.target}.part-{uuid4().hex}.wip")
            File: Any = Part.open("wb")
            self._open = (Part, File, AvroWriter(File, self.schema))
        self._open[2].write(record)

    def finish_bundle(self) -> None:
        if self._open is None:
            return
        Part, File, Sink = self._open
        Sink.flush()
        File.close()
        Part.rename(Part.with_suffix(""))
        self._open = None
