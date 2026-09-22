from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from fastavro import parse_schema, writer

from relmedner.constants import DATA
from relmedner.local import LocalAvroDataStream, LocalDelimitedDataStream
from relmedner.models import RunConfig
from relmedner.registry import SOURCE_REGISTRY, build_stream

# --------------------------------------------------------------------------- local (avro) --

SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "Row",
    "namespace": "relmedner.tests",
    "fields": [
        {"name": "name", "type": "string"},
        {"name": "description", "type": ["null", "string"], "default": None},
    ],
}

AVRO_TASK: tuple[Any, ...] = ("script", "CtkpInterventionsScript", ("entities",))
WEIGHT: float = 1.0
# the operator-built CTKP corpus declared in ingests.yaml; the container itself is gitignored
PACKAGED_CORPUS: str = "interventions/interventions.avro"


def write_avro(path: Path, records: list[dict[str, Any]]) -> Path:
    with path.open("wb") as handle:
        writer(handle, parse_schema(SCHEMA), records)
    return path


def test_rows_yield_the_whole_record_keyed_on_the_declared_path(tmp_path: Path) -> None:
    """the row key is the declared path verbatim, not its basename: the pipeline looks the source's
    mixing weight up by this string against DatasetBase.row_key, and two distinct files may share a
    basename while declaring different weights"""
    Target: Path = write_avro(tmp_path / "interventions.avro", [{"name": "aspirin", "description": "81 mg"}])
    Rows: list[Any] = list(LocalAvroDataStream(AVRO_TASK, WEIGHT, str(Target)).rows())

    assert len(Rows) == 1
    Source, (Task, Values) = Rows[0]
    assert Source == str(Target)
    assert Task == AVRO_TASK
    assert Values == ({"name": "aspirin", "description": "81 mg"},)


def test_the_registry_builds_the_local_stream_from_its_declared_tuple(tmp_path: Path) -> None:
    Target: Path = write_avro(tmp_path / "ctkp.avro", [{"name": "ALG-055009", "description": None}])
    Stream = build_stream("local", (AVRO_TASK, 0.75, str(Target)))

    assert isinstance(Stream, LocalAvroDataStream)
    # 0.75 in the middle slot proves the weight rides positionally between task and path
    assert Stream.weight == 0.75
    assert [values[0]["name"] for _source, (_task, values) in Stream.rows()] == ["ALG-055009"]


def test_the_sample_limit_bounds_a_local_stream_like_every_other_source(tmp_path: Path) -> None:
    Target: Path = write_avro(tmp_path / "many.avro", [{"name": f"drug {index}", "description": None} for index in range(10)])
    Streamed: list[Any] = list(LocalAvroDataStream(AVRO_TASK, WEIGHT, str(Target)).stream(RunConfig(sample_limit=3)))

    assert len(Streamed) == 3


def test_the_path_expands_a_user_relative_declaration(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    write_avro(tmp_path / "home.avro", [{"name": "metformin", "description": None}])
    Streamed: list[Any] = list(LocalAvroDataStream(AVRO_TASK, WEIGHT, "~/home.avro").rows())

    assert [values[0]["name"] for _source, (_task, values) in Streamed] == ["metformin"]


def test_cwd_relative_and_absolute_paths_that_exist_win_over_the_package_data_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """an existing file beats the package data dir however the path is spelled; the tilde spelling
    is covered by test_the_path_expands_a_user_relative_declaration, so this pins the CWD-relative
    winner (kept as declared, not absolutized) and the absolute winner"""
    write_avro(tmp_path / "mine.avro", [{"name": "warfarin", "description": None}])
    monkeypatch.chdir(tmp_path)

    Relative = LocalAvroDataStream(AVRO_TASK, WEIGHT, "mine.avro")

    assert Relative.path == Path("mine.avro")
    assert [values[0]["name"] for _source, (_task, values) in Relative.rows()] == ["warfarin"]

    Absolute: Path = write_avro(tmp_path / "absolute.avro", [{"name": "digoxin", "description": None}])
    FromRoot = LocalAvroDataStream(AVRO_TASK, WEIGHT, str(Absolute))

    assert FromRoot.path == Absolute
    assert [values[0]["name"] for _source, (_task, values) in FromRoot.rows()] == ["digoxin"]


def test_the_declared_relative_path_falls_back_to_the_package_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """the resolution rule the delimited stream keeps: a declared relative path with no file under
    the caller's CWD resolves against the package data dir, while the row key stays the DECLARED
    string because that is what the weight lookup keys on"""
    monkeypatch.chdir(tmp_path)
    Corpus: Path = tmp_path / "packaged" / "corpora" / "thing.avro"
    Corpus.parent.mkdir(parents=True)
    write_avro(Corpus, [{"name": "ondansetron", "description": None}])
    monkeypatch.setattr("relmedner.local.DATA", tmp_path / "packaged")

    Stream = LocalAvroDataStream(AVRO_TASK, WEIGHT, "corpora/thing.avro")

    assert Stream.name == "corpora/thing.avro"
    assert Stream.path == Corpus
    assert [source for source, _payload in Stream.rows()] == ["corpora/thing.avro"]


def test_records_stream_lazily_rather_than_materializing_the_container(tmp_path: Path) -> None:
    """the declared avro is ~1M records, so rows() must stay a generator over the reader"""
    Target: Path = write_avro(tmp_path / "lazy.avro", [{"name": f"drug {index}", "description": None} for index in range(100)])
    Rows = LocalAvroDataStream(AVRO_TASK, WEIGHT, str(Target)).rows()

    assert isinstance(Rows, io.IOBase) is False
    assert next(iter(Rows))[1][1][0]["name"] == "drug 0"


def test_missing_containers_raise_file_not_found() -> None:
    """a fresh clone has no gitignored corpus blob, so a declaration resolving to nothing must
    fail loud at stream time instead of silently yielding zero training rows"""
    Missing = LocalAvroDataStream(AVRO_TASK, WEIGHT, "/nonexistent/definitely-missing.avro")

    with pytest.raises(FileNotFoundError):
        list(Missing.rows())


def test_the_container_probe_survives_an_unreadable_parent_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """hermetic twin of test_missing_containers_raise_file_not_found: on the remote box
    /nonexistent exists as an unsearchable directory, so python 3.13 pathlib raises
    PermissionError from the is_file() probe instead of returning False, and the host
    filesystem cannot be relied on to produce the failure. The probe is forced to raise
    here so the OSError-guard in rows() stays pinned: an unreadable path is not a usable
    file either way, and the declared error is FileNotFoundError naming the resolved path
    (the same contract LocalDelimitedDataStream.rows keeps)"""
    Missing = LocalAvroDataStream(AVRO_TASK, WEIGHT, "/nonexistent/definitely-missing.avro")

    def raise_permission_denied(self: Path) -> bool:
        raise PermissionError(f"parent directory not searchable: {self}")

    monkeypatch.setattr(Path, "is_file", raise_permission_denied)

    with pytest.raises(FileNotFoundError):
        list(Missing.rows())


def test_the_packaged_interventions_declaration_resolves_against_the_package_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """runs with or without the blob on disk: from a CWD that lacks the declared corpus the
    packaged relative path must resolve into the package data dir under its declared name"""
    monkeypatch.chdir(tmp_path)

    Stream = LocalAvroDataStream(AVRO_TASK, WEIGHT, PACKAGED_CORPUS)

    assert Stream.path == Path(str(DATA)) / PACKAGED_CORPUS
    assert Stream.name == PACKAGED_CORPUS


@pytest.mark.skipif(
    not (Path(str(DATA)) / PACKAGED_CORPUS).is_file(),
    reason="the interventions corpus is a gitignored out-of-band 92MB operator artifact",
)
def test_the_packaged_interventions_corpus_streams_its_first_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """pins the deployed corpus behind the declared ingest. Skips on a fresh clone: the container
    is a gitignored out-of-band operator artifact (92MB, incompressible), the same convention as
    the fullmap-dependent skips documented in README.md; the resolution half of the contract is
    asserted unconditionally by
    test_the_packaged_interventions_declaration_resolves_against_the_package_data_dir. Reads
    exactly one record lazily -- list() over the ~1M-record container would be the
    materialization bug test_records_stream_lazily_rather_than_materializing_the_container
    guards against."""
    monkeypatch.chdir(tmp_path)
    Stream = LocalAvroDataStream(AVRO_TASK, WEIGHT, PACKAGED_CORPUS)

    assert Stream.path == Path(str(DATA)) / PACKAGED_CORPUS
    First: dict[str, Any] = next(iter(Stream.rows()))[1][1][0]

    assert First["nct_id"] == "NCT04295564"


# ------------------------------------------------------------------- local_delimited (tsv) --

DELIMITED_TASK: tuple[Any, ...] = ("fullmap", 6, "9606", True, ("entities", "relations"))


def write_tsv(path: Path, header: tuple[str, ...], rows: tuple[tuple[str, ...], ...], delimiter: str = "\t") -> str:
    path.write_text("\n".join(delimiter.join(header + row) for row in (tuple(header), *rows)) + "\n", encoding="utf-8")
    return str(path)


def test_tsv_rows_stream_with_header_selected_columns(tmp_path: Path) -> None:
    Path_ = Path(write_tsv(tmp_path / "rows.tsv", ("text", "extra"), (("first row", "a"), ("second row", "b"))))
    Stream = LocalDelimitedDataStream(DELIMITED_TASK, WEIGHT, Path_, ("text",))

    # keyed on the declared path, the same row-key contract the avro stream keeps
    assert list(Stream.rows()) == [
        (str(Path_), (DELIMITED_TASK, ("first row",))),
        (str(Path_), (DELIMITED_TASK, ("second row",))),
    ]


def test_csv_suffix_picks_the_comma_delimiter(tmp_path: Path) -> None:
    Path_ = Path(write_tsv(tmp_path / "rows.csv", ("text",), (("one",), ("two",)), delimiter=","))
    assert [row[1][1][0] for row in LocalDelimitedDataStream(DELIMITED_TASK, WEIGHT, Path_, ("text",)).rows()] == ["one", "two"]


def test_a_tab_delimiter_survives_inside_a_csv_named_file(tmp_path: Path) -> None:
    """the delimiter comes from the suffix contract, not content sniffing"""
    Path_ = Path(write_tsv(tmp_path / "rows.tsv", ("text",), (("a,b",),)))
    assert [row[1][1][0] for row in LocalDelimitedDataStream(DELIMITED_TASK, WEIGHT, Path_, ("text",)).rows()] == ["a,b"]


def test_cwd_relative_paths_that_exist_win_over_the_package_data_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """a declared relative path resolves against the caller's CWD when the file is there, and only
    falls back to the package data dir otherwise; the row key stays the DECLARED string, because
    that is what LocalDelimitedDataset.row_key reports for the weight lookup"""
    write_tsv(tmp_path / "mine.tsv", ("text",), (("local",),))
    monkeypatch.chdir(tmp_path)
    Stream = LocalDelimitedDataStream(DELIMITED_TASK, WEIGHT, "mine.tsv", ("text",))

    assert Stream.name == "mine.tsv"
    # the CWD-relative candidate wins outright, so the resolved path stays the declared relative
    # one rather than an absolutized copy of it
    assert Stream.path == Path("mine.tsv")
    assert [row[1][1][0] for row in Stream.rows()] == ["local"]


def test_missing_files_raise_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        list(LocalDelimitedDataStream(DELIMITED_TASK, WEIGHT, "/nonexistent/definitely-missing.tsv", ("text",)).rows())


def test_match_on_filters_rows_like_the_hf_stream(tmp_path: Path) -> None:
    Path_ = Path(write_tsv(tmp_path / "rows.tsv", ("text", "kind"), (("keep", "x"), ("drop", "y"))))
    Stream = LocalDelimitedDataStream(DELIMITED_TASK, WEIGHT, Path_, ("text",), match_on=(("kind", ("x",)),))
    assert [row[1][1][0] for row in Stream.rows()] == ["keep"]


def test_the_registry_keys_the_delimited_source_separately_from_avro() -> None:
    """both local kinds are declared ingests, so they must not share one SOURCE_REGISTRY key: the
    avro stream ships whole records and takes no columns_out, the delimited one projects them"""
    assert SOURCE_REGISTRY["local"] is LocalAvroDataStream
    assert SOURCE_REGISTRY["local_delimited"] is LocalDelimitedDataStream

    Payload = (DELIMITED_TASK, WEIGHT, "/nonexistent/x.tsv", ("text",), None)
    Stream = build_stream("local_delimited", Payload)

    assert isinstance(Stream, LocalDelimitedDataStream)
    assert Stream.weight == WEIGHT


def test_the_package_data_qualifier_corpus_streams() -> None:
    """the committed corpus resolves via the package data dir from any CWD (worker container
    included); the row count is pinned so silent corpus drift is noticed"""
    Stream = LocalDelimitedDataStream(DELIMITED_TASK, WEIGHT, "qualifiers/qualifier_corpus.tsv", ("text",))
    Rows = list(Stream.rows())

    assert len(Rows) == 24
    assert all(row[1][1][0].strip() for row in Rows)
    assert Rows[0][1][1][0].startswith("Metformin is used to treat")
    # the declared path, not the resolved package-data path: weights_by_source keys on the former
    assert Stream.name == "qualifiers/qualifier_corpus.tsv"
    assert Rows[0][0] == Stream.name
