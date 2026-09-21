from __future__ import annotations

from pathlib import Path

import pytest

from relmedner.local import LocalDataStream
from relmedner.registry import SOURCE_REGISTRY, build_stream

TASK: tuple[object, ...] = ("fullmap", 6, "9606", True, ("entities", "relations"))


def write_tsv(path: Path, header: tuple[str, ...], rows: tuple[tuple[str, ...], ...], delimiter: str = "\t") -> str:
    path.write_text("\n".join(delimiter.join(header + row) for row in (tuple(header), *rows)) + "\n", encoding="utf-8")
    return str(path)


def test_tsv_rows_stream_with_header_selected_columns() -> None:
    Path_ = Path(write_tsv(Path(__file__).parent / "_local_fixture.tsv", ("text", "extra"), (("first row", "a"), ("second row", "b"))))
    try:
        Stream = LocalDataStream(TASK, Path_, ("text",))
        assert list(Stream.rows()) == [
            (Path_.stem, (TASK, ("first row",))),
            (Path_.stem, (TASK, ("second row",))),
        ]
    finally:
        Path_.unlink()


def test_csv_suffix_picks_the_comma_delimiter(tmp_path: Path) -> None:
    Path_ = Path(write_tsv(tmp_path / "rows.csv", ("text",), (("one",), ("two",)), delimiter=","))
    assert [row[1][1][0] for row in LocalDataStream(TASK, Path_, ("text",)).rows()] == ["one", "two"]


def test_a_tab_delimiter_survives_inside_a_csv_named_file(tmp_path: Path) -> None:
    """the delimiter comes from the suffix contract, not content sniffing"""
    Path_ = Path(write_tsv(tmp_path / "rows.tsv", ("text",), (("a,b",),)))
    assert [row[1][1][0] for row in LocalDataStream(TASK, Path_, ("text",)).rows()] == ["a,b"]


def test_cwd_relative_paths_that_exist_win_over_the_package_data_fallback(tmp_path: Path) -> None:
    Path_ = Path(write_tsv(tmp_path / "mine.tsv", ("text",), (("local",),)))
    Stream = LocalDataStream(TASK, Path_, ("text",))
    assert Stream.name == "mine"
    assert [row[1][1][0] for row in Stream.rows()] == ["local"]


def test_missing_files_raise_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        list(LocalDataStream(TASK, "/nonexistent/definitely-missing.tsv", ("text",)).rows())


def test_match_on_filters_rows_like_the_hf_stream(tmp_path: Path) -> None:
    Path_ = Path(write_tsv(tmp_path / "rows.tsv", ("text", "kind"), (("keep", "x"), ("drop", "y"))))
    Stream = LocalDataStream(TASK, Path_, ("text",), match_on=(("kind", ("x",)),))
    assert [row[1][1][0] for row in Stream.rows()] == ["keep"]


def test_the_registry_dispatches_the_local_source() -> None:
    assert SOURCE_REGISTRY["local"] is LocalDataStream
    Payload = (TASK, "/nonexistent/x.tsv", ("text",), None)
    assert isinstance(build_stream("local", Payload), LocalDataStream)


def test_the_package_data_qualifier_corpus_streams() -> None:
    """the committed corpus resolves via the package data dir from any CWD (worker container
    included); the row count is pinned so silent corpus drift is noticed"""
    Stream = LocalDataStream(TASK, "qualifiers/qualifier_corpus.tsv", ("text",))
    Rows = list(Stream.rows())

    assert len(Rows) == 24
    assert all(row[1][1][0].strip() for row in Rows)
    assert Rows[0][1][1][0].startswith("Metformin is used to treat")
