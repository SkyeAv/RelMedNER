"""guards the packaged-data reads against a non-utf-8 locale

On the compute box, bash resolves LANG=en_US to ISO-8859-1. A locale-default read of
cluster.yaml then mis-decodes its em-dashes into control characters that CSafeLoader rejects --
19 tests failed there while the same commit passed everywhere else. The production reads are now
explicitly utf-8.

These guards are deliberately locale-independent: Python resolves the default encoding at C level,
so monkeypatching locale.getpreferredencoding/getencoding does NOT affect open(), and a
locale-forcing test would only reproduce on a host that happens to have a non-utf-8 locale
installed. Instead they assert the two things that actually matter -- that production passes an
explicit encoding, and that the failure mode is real at the byte level.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import pytest
import yaml

from relmedner.constants import CLUSTER_YAML, COMPOSE_DIR, INGESTS_YAML
from relmedner.parsers import YamlParser

LATIN1_LOCALE: str = "ISO-8859-1"
SRC: Path = Path(__file__).resolve().parent.parent / "src" / "relmedner"


class RecordingTraversable:
    """stands in for an importlib Traversable and records how it was opened"""

    RECORDED: ClassVar[list[tuple[str, str | None]]] = []

    def __init__(self: Self, payload: str) -> None:
        self.payload: str = payload

    def open(self: Self, mode: str = "r", encoding: str | None = None, **kwargs: Any) -> io.StringIO:
        RecordingTraversable.RECORDED.append((mode, encoding))
        return io.StringIO(self.payload)


def packaged_files() -> list[Path]:
    """every packaged text data file that production code reads and a human might extend"""
    root: Path = Path(str(COMPOSE_DIR))
    located: list[Path] = [Path(str(CLUSTER_YAML)), Path(str(INGESTS_YAML))]
    located.extend(root.glob("docker-compose.*.yml"))
    located.extend(root.glob("Dockerfile.*"))
    return sorted(located)


@pytest.mark.parametrize("Target", packaged_files(), ids=lambda item: item.name)
def test_every_packaged_data_file_is_valid_utf8(Target: Path) -> None:
    """strict decoding: a latin-1-only save parses fine on the author's box and breaks every
    utf-8 host, so the guard is on the bytes rather than on anyone's locale"""
    assert Target.read_bytes().decode("utf-8")


def test_a_latin1_read_of_cluster_yaml_is_rejected_by_the_loader() -> None:
    """the failure mode, pinned at the byte level: em-dashes decoded as latin-1 yield exactly the
    control characters CSafeLoader refuses -- this is what the compute box hit"""
    mangled: str = Path(str(CLUSTER_YAML)).read_bytes().decode(LATIN1_LOCALE)

    assert any(0x80 <= ord(char) <= 0x9F for char in mangled), "expected C1 control chars from the mis-decode"
    with pytest.raises(yaml.reader.ReaderError):
        yaml.load(mangled, Loader=yaml.CSafeLoader)


def test_a_utf8_read_of_the_same_bytes_parses() -> None:
    """the contrast that proves explicit encoding is the whole fix"""
    parsed: Any = yaml.load(Path(str(CLUSTER_YAML)).read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)

    assert parsed["workers"]


def test_the_yaml_parser_requests_utf8_from_the_resource() -> None:
    """production path: YamlParser must pass an explicit encoding instead of taking the locale"""
    RecordingTraversable.RECORDED = []

    # duck-typed against importlib's Traversable: YamlParser only ever calls .open on it
    Parsed: Any = YamlParser(cast(Any, RecordingTraversable("workers: []\nssh_user: sgoetz\n"))).parse()

    assert RecordingTraversable.RECORDED == [("r", "utf-8")], f"locale-default read: {RecordingTraversable.RECORDED}"
    assert Parsed["ssh_user"] == "sgoetz"


def test_cluster_yaml_keeps_its_em_dash_comments() -> None:
    """the non-ascii that triggered this; if someone rewrites the comments the guards above go
    moot, so lock the character in until then"""
    assert "\u2014" in Path(str(CLUSTER_YAML)).read_text(encoding="utf-8")


def _mode_of(node: ast.Call, is_bare_open: bool) -> str:
    """the open mode, from the keyword or the positional slot that carries it for this call shape"""
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
    slot: int = 1 if is_bare_open else 0  # open(file, mode, ...) vs Traversable.open(mode, ...)
    if len(node.args) > slot:
        positional = node.args[slot]
        if isinstance(positional, ast.Constant) and isinstance(positional.value, str):
            return positional.value
    return ""


def offenders_in(source_file: Path) -> list[str]:
    """every call in one file that opens TEXT without pinning an encoding"""
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_bare_open: bool = isinstance(func, ast.Name) and func.id == "open"
        is_open_attr: bool = isinstance(func, ast.Attribute) and func.attr == "open"
        is_read_text: bool = isinstance(func, ast.Attribute) and func.attr == "read_text"
        if not (is_bare_open or is_open_attr or is_read_text):
            continue
        if not is_read_text and "b" in _mode_of(node, is_bare_open):
            continue  # binary reads carry bytes; no encoding is correct there
        pinned: bool = any(keyword.arg == "encoding" for keyword in node.keywords)
        positional: int = 3 if is_bare_open else 0  # open(f, mode, buffering, encoding) vs .read_text(encoding)
        pinned = pinned or (not is_open_attr and len(node.args) > positional)
        if not pinned:
            offenders.append(f"{source_file.name}:{node.lineno}")
    return offenders


def text_reads_without_explicit_encoding() -> list[str]:
    return [offender for source_file in sorted(SRC.rglob("*.py")) for offender in offenders_in(source_file)]


def test_no_text_read_in_src_relies_on_the_ambient_locale() -> None:
    """the regression guard: dropping encoding= from any of these reads reintroduces the bug"""
    assert text_reads_without_explicit_encoding() == [], "locale-dependent reads found"


def test_the_guard_catches_every_shape_of_locale_dependent_read(tmp_path: Path) -> None:
    """meta-guard: the scanner must catch the bug in all three call shapes -- and must NOT flag
    binary reads, which legitimately take no encoding"""
    Target: Path = tmp_path / "leaky.py"
    Target.write_text(
        "from pathlib import Path\n"
        "Path('a.yaml').read_text()\n"  # line 2: bare-ish attribute read_text
        "open('b.yaml', 'r').read()\n"  # line 3: builtin open, text mode
        "Path('c.avro').open('rb').read()\n"  # line 4: binary, must NOT be flagged
        "Path('d.yaml').read_text(encoding='utf-8')\n",  # line 5: pinned, must NOT be flagged
        encoding="utf-8",
    )

    assert sorted(offenders_in(Target)) == ["leaky.py:2", "leaky.py:3"]
