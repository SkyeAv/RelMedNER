"""Offline docs-integrity guards (REQ-DOCS-1..8): README and docs/ are agent-facing
references that rot silently, so the suite fails loudly on broken relative links, dead
anchors, orphan docs pages, re-duplicated bodies, a lost quick start, and non-ASCII prose.

TDD staging (by design, do not "fix" by editing README here): the fan-out stories move
each FANOUT body out of README.md into its docs/ page, which flips
test_moved_bodies_are_not_duplicated_in_readme from RED to GREEN, and later stories make
test_every_docs_page_is_linked_from_readme go RED the moment a new docs page lands
unlinked. Everything else is green from this story onward.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from relmedner.ingests import YamlIngestsParser

README = pathlib.Path("README.md")
DOCS_DIR = pathlib.Path("docs")
WEIGHTING_DOC = pathlib.Path("docs/weighting.md")

# the section of docs/weighting.md holding the per-dataset tier/weight/trust table; the
# coverage guard reads the table out of this section only, so prose mentions elsewhere in
# the page cannot satisfy it (REQ-CUR-2)
_PRIORS_HEADING = "## Reweighting the existing datasets"
_BACKTICKED_RE = re.compile(r"`([^`]+)`")

# (docs page path, sentinel string taken verbatim from the body that must MOVE into that
# page): while the string still appears in README.md the fan-out has not happened and the
# body is duplicated verbatim (REQ-DOCS-5). Data lives here so the later stories cannot
# quietly redefine what "moved" means.
FANOUT: tuple[tuple[str, str], ...] = (
    ("docs/ingests.md", "All script tasks except the multilingual ingest"),
    ("docs/qualifiers.md", "Gazetteer relations also carry the six DAKP-declared nullable qualifier slots"),
    ("docs/output.md", "Output is Avro records of"),
    ("docs/deduplication.md", "passes through a dedup stage"),
    ("docs/ctkp-interventions.md", "The one `source: local` ingest"),
    ("docs/trialpanorama-database.md", "ships 11 subsets"),
    ("docs/super-glue-record.md", "The one ingest whose script ships general-domain text"),
    ("docs/nemotron-pii.md", "The one general-domain ingest"),
    ("docs/post-training-families.md", "splits the multi-task corpus into disjoint families"),
    ("docs/fullmap-mining.md", "Per batch of documents (Beam"),
)

# the quick start is the only on-ramp for a new user and for CI reproduction
# (REQ-DOCS-6); losing any of these headings or commands breaks setup silently
_QUICK_START_HEADINGS = ("## Install", "## Build the dataset", "## Testing", "## Cluster")
_QUICK_START_COMMANDS = ("uv sync", "build-dataset", "make test")

# markdown inline links/images: the target is everything between "(" and the first
# whitespace or ")"; link titles `](url "title")` are tolerated
_LINK_RE = re.compile(r"\]\(([^)\s]+)")
_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)")
# schemes that can never resolve to a repo file are skipped outright (REQ-DOCS-2)
_EXTERNAL_SCHEMES = ("http://", "https://", "mailto:")

# a prose repeat is this many consecutive non-blank, non-code lines (REQ-CUR-1): four is
# long enough that two identical runs cannot arise by coincidence in prose, and short
# enough to catch a copy-pasted paragraph or a squash merge that re-applied a section
_DUP_WINDOW = 4


def _prose_lines(path: pathlib.Path) -> list[tuple[int, str]]:
    """(line number, text) for every prose line of one markdown file, in order.

    Code is excluded on purpose: fenced blocks toggle on ``` or ~~~ and four-space
    indented blocks are skipped, because repeated shell or yaml snippets across a page are
    normal (the same `uv run pytest` invocation can legitimately appear twice) while
    repeated PROSE means the page was pasted onto itself."""
    prose: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not line.strip() or line.startswith(("    ", "\t")):
            continue
        prose.append((lineno, line.rstrip()))
    return prose


def _headings_in_order(path: pathlib.Path) -> list[tuple[int, str]]:
    """(line number, slug) for every ATX heading outside code fences, in document order.
    Ordered rather than de-duplicated like `_heading_slugs`, so a repeated heading can be
    reported with both of its line numbers."""
    headings: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _ATX_HEADING_RE.match(line)
        if match:
            headings.append((lineno, _slug(match.group(2))))
    return headings


def _slug(heading_text: str) -> str:
    """GitHub-style heading slug: strip inline markdown (backticks, emphasis markers),
    lowercase, spaces become hyphens, every other whitespace/punctuation character is
    removed, and the ends are trimmed (REQ-DOCS-3)."""
    text = re.sub(r"[`*_~]", "", heading_text.strip())
    text = text.lower().replace(" ", "-")
    return re.sub(r"[^\w-]", "", text).strip("-")


def _heading_slugs(path: pathlib.Path) -> set[str]:
    """Slugs of every ATX heading in one markdown file; fenced code blocks are skipped so
    a `#` comment inside a code fence cannot fake a heading."""
    slugs: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _ATX_HEADING_RE.match(line)
        if match:
            slugs.add(_slug(match.group(2)))
    return slugs


def _collect_links(path: pathlib.Path) -> list[str]:
    """Every markdown `](target)` URL in one file, in order of appearance."""
    return _LINK_RE.findall(path.read_text(encoding="utf-8"))


def _broken_links(path: pathlib.Path) -> list[str]:
    """Human-readable description of every relative link in `path` that either does not
    resolve to an existing file (relative to the linking file's directory) or whose
    `#anchor` misses every heading slug of the target file (REQ-DOCS-2/3)."""
    problems: list[str] = []
    for link in _collect_links(path):
        if link.startswith(_EXTERNAL_SCHEMES):
            continue
        target, _, anchor = link.partition("#")
        target_path = path if not target else path.parent / target
        if not target_path.is_file():
            problems.append(f"{path}: link {link!r} does not resolve (expected file {target_path} to exist)")
            continue
        if anchor and anchor not in _heading_slugs(target_path):
            problems.append(f"{path}: link {link!r} anchors to no heading in {target_path}")
    return problems


def test_readme_relative_links_resolve() -> None:
    """README.md is the front door every agent and human enters through (REQ-DOCS-2/3):
    one dead relative link or dead anchor silently severs the path to the schema
    reference or the cluster docs, so each one must resolve to a real file and a real
    heading right now."""
    problems = _broken_links(README)
    assert not problems, f"README.md has {len(problems)} broken relative link(s):\n" + "\n".join(problems)


@pytest.mark.parametrize("doc", sorted(str(p) for p in DOCS_DIR.glob("*.md")))
def test_docs_relative_links_resolve(doc: str) -> None:
    """Each docs/ page is a standalone agent reference (REQ-DOCS-2/3): links are
    parametrized per page so one broken link names its own file instead of drowning in a
    whole-directory dump, and a future page is checked the moment it exists."""
    problems = _broken_links(pathlib.Path(doc))
    assert not problems, f"{doc} has {len(problems)} broken relative link(s):\n" + "\n".join(problems)


def test_every_docs_page_is_linked_from_readme() -> None:
    """Orphan guard (REQ-DOCS-4): the file list is enumerated at run time, not copied, so
    adding a future docs page without linking it from README.md fails CI instead of
    leaving an invisible page no reader can find."""
    readme_targets = {link.partition("#")[0].lstrip("./") or "." for link in _collect_links(README)}
    orphans = sorted(str(p) for p in DOCS_DIR.glob("*.md") if str(p) not in readme_targets)
    assert not orphans, f"docs page(s) exist but are never linked from README.md: {orphans}"


def test_moved_bodies_are_not_duplicated_in_readme() -> None:
    """Fan-out duplication guard (REQ-DOCS-5): each sentinel is a verbatim slice of a body
    that must MOVE from README.md into its docs/ page; while the sentinel still appears in
    README.md the body is duplicated in two places and the copy will drift. RED until the
    fan-out stories land -- that is the intended signal, not a bug."""
    readme_text = README.read_text(encoding="utf-8")
    duplicated = [(page, sentinel) for page, sentinel in FANOUT if sentinel in readme_text]
    assert not duplicated, "README.md still duplicates moved bodies verbatim (each must live only in its docs page):\n" + "\n".join(
        f"  {sentinel!r} belongs in {page}" for page, sentinel in duplicated
    )


def test_readme_keeps_quick_start() -> None:
    """Quick-start guard (REQ-DOCS-6): the install/build/test/cluster headings and the
    `uv sync`, `build-dataset`, `make test` commands are the only on-ramp for a new user
    and for CI reproduction; prose refactors must not be able to delete them."""
    readme_text = README.read_text(encoding="utf-8")
    missing_headings = [h for h in _QUICK_START_HEADINGS if h not in readme_text.splitlines()]
    missing_commands = [c for c in _QUICK_START_COMMANDS if c not in readme_text]
    assert not missing_headings, f"README.md lost quick-start heading(s): {missing_headings}"
    assert not missing_commands, f"README.md lost quick-start command(s): {missing_commands}"


def _priors_table_rows(path: pathlib.Path = WEIGHTING_DOC) -> list[str]:
    """The markdown rows of the priors table: the lines between the reweighting heading and
    the next `## ` heading that begin with a pipe and a backtick, which excludes the header
    row, the separator row, and any non-table prose in the section."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(_PRIORS_HEADING)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return [line for line in lines[start:end] if line.startswith("| `")]


def _names_row_key(row_key: str, spans: list[str]) -> bool:
    """True when one backticked span in the priors table names `row_key`, either exactly or
    through a trailing-`*` wildcard. The wildcard exists because the seven reddit corpora
    share one row on purpose, exactly as they share one `match_on` anchor."""
    return any(span == row_key or (span.endswith("*") and row_key.startswith(span[:-1])) for span in spans)


@pytest.mark.parametrize("doc", [str(README), *sorted(str(p) for p in DOCS_DIR.glob("*.md"))])
def test_docs_have_no_duplicate_headings(doc: str) -> None:
    """Repeated-heading guard (REQ-CUR-1): a squash merge that resolves a conflict by
    keeping both sides re-applies a whole section, and the heading is the only visible
    symptom. README.md shipped `## Reddit corpora` twice after #44 and #48 landed, so a
    repeated slug in one file now fails instead of shipping a page that says the same
    thing twice with two copies free to drift apart."""
    headings = _headings_in_order(pathlib.Path(doc))
    seen: dict[str, int] = {}
    repeats: list[str] = []
    for lineno, slug in headings:
        if slug in seen:
            repeats.append(f"{doc}:{lineno}: heading {slug!r} repeats line {seen[slug]}")
        else:
            seen[slug] = lineno
    assert not repeats, "duplicate heading(s) in one file:\n" + "\n".join(repeats)


@pytest.mark.parametrize("doc", [str(README), *sorted(str(p) for p in DOCS_DIR.glob("*.md"))])
def test_docs_have_no_duplicate_prose_blocks(doc: str) -> None:
    """Repeated-prose guard (REQ-CUR-1): catches the part a heading check misses, namely a
    section pasted twice under a DIFFERENT heading, or a body duplicated without its
    heading. Any window of four consecutive prose lines that appears twice in one file is
    a duplication, reported with both line numbers so the fix is a deletion, not a guess."""
    prose = _prose_lines(pathlib.Path(doc))
    seen: dict[tuple[str, ...], int] = {}
    repeats: list[str] = []
    for start in range(len(prose) - _DUP_WINDOW + 1):
        window = prose[start : start + _DUP_WINDOW]
        key = tuple(text for _, text in window)
        if key in seen:
            repeats.append(f"{doc}:{window[0][0]}: {_DUP_WINDOW} prose lines repeat line {seen[key]}: {key[0][:70]!r}")
        else:
            seen[key] = window[0][0]
    assert not repeats, "duplicate prose block(s) in one file:\n" + "\n".join(repeats)


def test_every_declared_ingest_has_a_weighting_prior() -> None:
    """Weighting-coverage guard (REQ-CUR-2): the row keys come from the real parser
    (`weights_by_source`, the same key space the pipeline stamps weights on), not from a
    copied list, so declaring a new ingest without giving it a tier in docs/weighting.md
    fails CI. That drift is not hypothetical: the guide claimed 25 entries and had no row
    for bioleaflets, Medical-Entity-JSON-Extraction, or either synthetic ADE artifact while
    ingests.yaml declared 29 entries over 26 row keys."""
    row_keys = sorted(YamlIngestsParser().parse_ingests().weights_by_source())
    spans = [span for line in _priors_table_rows() for span in _BACKTICKED_RE.findall(line)]
    missing = [key for key in row_keys if not _names_row_key(key, spans)]
    assert not missing, (
        f"docs/weighting.md priors table has no row for {len(missing)} declared row key(s): {missing}. "
        "Give each one a tier, weight, and trust prior in the '## Reweighting the existing datasets' table."
    )


@pytest.mark.parametrize("doc", [str(README), *sorted(str(p) for p in DOCS_DIR.glob("*.md"))])
def test_docs_prose_is_ascii(doc: str) -> None:
    """ASCII guard (REQ-DOCS-7): the house style forbids em dashes, unicode arrows,
    check marks, bullets, and smart quotes because they break terminal rendering and diff
    tooling; per-line reporting names the exact file:line and character so the fix is
    mechanical."""
    offenders: list[str] = []
    for lineno, line in enumerate(pathlib.Path(doc).read_text(encoding="utf-8").splitlines(), start=1):
        try:
            line.encode("ascii")
        except UnicodeEncodeError:
            bad = sorted({f"{ch!r} (U+{ord(ch):04X})" for ch in line if ord(ch) > 127})
            offenders.append(f"{doc}:{lineno}: {', '.join(bad)}")
    assert not offenders, "non-ASCII character(s) found:\n" + "\n".join(offenders)
