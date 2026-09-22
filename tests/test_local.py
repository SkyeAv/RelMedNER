from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fastavro import parse_schema, writer

from relmedner.local import LocalAvroDataStream
from relmedner.models import RunConfig
from relmedner.registry import build_stream

SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "Row",
    "namespace": "relmedner.tests",
    "fields": [
        {"name": "name", "type": "string"},
        {"name": "description", "type": ["null", "string"], "default": None},
    ],
}

TASK: tuple[Any, ...] = ("script", "CtkpInterventionsScript", ("entities",))
WEIGHT: float = 1.0


def write_avro(path: Path, records: list[dict[str, Any]]) -> Path:
    with path.open("wb") as handle:
        writer(handle, parse_schema(SCHEMA), records)
    return path


def test_rows_yield_the_whole_record_keyed_on_the_declared_path(tmp_path: Path) -> None:
    """the row key is the declared path verbatim, not its basename: the pipeline looks the source's
    mixing weight up by this string against DatasetBase.row_key, and two distinct files may share a
    basename while declaring different weights"""
    Target: Path = write_avro(tmp_path / "interventions.avro", [{"name": "aspirin", "description": "81 mg"}])
    Rows: list[Any] = list(LocalAvroDataStream(TASK, WEIGHT, str(Target)).rows())

    assert len(Rows) == 1
    Source, (Task, Values) = Rows[0]
    assert Source == str(Target)
    assert Task == TASK
    assert Values == ({"name": "aspirin", "description": "81 mg"},)


def test_the_registry_builds_the_local_stream_from_its_declared_tuple(tmp_path: Path) -> None:
    Target: Path = write_avro(tmp_path / "ctkp.avro", [{"name": "ALG-055009", "description": None}])
    Stream = build_stream("local", (TASK, 0.75, str(Target)))

    assert isinstance(Stream, LocalAvroDataStream)
    # 0.75 in the middle slot proves the weight rides positionally between task and path
    assert Stream.weight == 0.75
    assert [values[0]["name"] for _source, (_task, values) in Stream.rows()] == ["ALG-055009"]


def test_the_sample_limit_bounds_a_local_stream_like_every_other_source(tmp_path: Path) -> None:
    Target: Path = write_avro(tmp_path / "many.avro", [{"name": f"drug {index}", "description": None} for index in range(10)])
    Streamed: list[Any] = list(LocalAvroDataStream(TASK, WEIGHT, str(Target)).stream(RunConfig(sample_limit=3)))

    assert len(Streamed) == 3


def test_the_path_expands_a_user_relative_declaration(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    write_avro(tmp_path / "home.avro", [{"name": "metformin", "description": None}])
    Streamed: list[Any] = list(LocalAvroDataStream(TASK, WEIGHT, "~/home.avro").rows())

    assert [values[0]["name"] for _source, (_task, values) in Streamed] == ["metformin"]


def test_records_stream_lazily_rather_than_materializing_the_container(tmp_path: Path) -> None:
    """the declared avro is ~1M records, so rows() must stay a generator over the reader"""
    Target: Path = write_avro(tmp_path / "lazy.avro", [{"name": f"drug {index}", "description": None} for index in range(100)])
    Rows = LocalAvroDataStream(TASK, WEIGHT, str(Target)).rows()

    assert isinstance(Rows, io.IOBase) is False
    assert next(iter(Rows))[1][1][0]["name"] == "drug 0"
