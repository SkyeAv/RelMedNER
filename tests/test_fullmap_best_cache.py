"""ScriptUtils._fullmap_best memoizes best fullmap rows per normalized term across rows.

These tests stub the uncached fetch (_fetch_best) so they run without the fullmap mount and
count exactly which terms reach redb."""

from __future__ import annotations

from collections import OrderedDict

import pytest

from relmedner import utils
from relmedner.utils import ScriptUtils


def row(term: str, curie: str, name: str, category: str) -> dict[str, object]:
    """the canonical lookup_rows row shape (tablassert.fullmap.lookup_rows contract)"""
    return {
        "term": term,
        "CURIE": curie,
        "PREFERRED_NAME": name,
        "CATEGORY_NAME": category,
        "TAXON_ID": 0,
        "SOURCE_NAME": "t",
        "SOURCE_VERSION": "t",
    }


ROW: dict[str, object] = {"CATEGORY_NAME": "biolink:Drug", "CURIE": "CHEBI:15365", "PREFERRED_NAME": "aspirin"}


@pytest.fixture
def fetches(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """a fresh cache per test plus a recording fake fetch: 'aspirin' resolves, everything else misses"""
    calls: list[list[str]] = []
    monkeypatch.setattr(ScriptUtils, "_best_cache", OrderedDict())

    def fake(cls: type[ScriptUtils], terms: list[str]) -> dict[str, dict[str, object]]:
        calls.append(list(terms))
        return {term: ROW for term in terms if term == "aspirin"}

    monkeypatch.setattr(ScriptUtils, "_fetch_best", classmethod(fake))
    return calls


def test_a_term_seen_in_an_earlier_row_never_reaches_redb_again(fetches: list[list[str]]) -> None:
    """the whole point of the cache: recurring vocabulary costs one lookup per process, and the
    answer from the cache equals the answer from the first fetch"""
    first = ScriptUtils._fullmap_best({"Aspirin": "aspirin", "foo": "foo"})
    second = ScriptUtils._fullmap_best({"aspirin": "aspirin", "foo": "foo", "bar": "bar"})
    assert first == second == {"aspirin": ROW}
    assert fetches == [["aspirin", "foo"], ["bar"]]


def test_known_misses_are_cached_too(fetches: list[list[str]]) -> None:
    """a term with no accepted row is remembered as None, otherwise every unresolvable surface
    (most of them) would pay the round trip on every row"""
    ScriptUtils._fullmap_best({"x": "zzz"})
    ScriptUtils._fullmap_best({"x": "zzz"})
    assert fetches == [["zzz"]]


def test_empty_normalizations_never_fetch(fetches: list[list[str]]) -> None:
    assert ScriptUtils._fullmap_best({"!!": ""}) == {}
    assert fetches == []


def test_the_cache_is_bounded_and_evicts_least_recently_used(fetches: list[list[str]], monkeypatch: pytest.MonkeyPatch) -> None:
    """memory must stay flat on long runs: the LRU holds at most the bound, and a re-touched term
    survives eviction while an untouched one is refetched"""
    monkeypatch.setattr(utils, "FULLMAP_BEST_CACHE_TERMS", 2)
    ScriptUtils._fullmap_best({"a": "aspirin"})
    ScriptUtils._fullmap_best({"b": "b"})
    ScriptUtils._fullmap_best({"a": "aspirin"})  # refresh aspirin
    ScriptUtils._fullmap_best({"c": "c"})  # evicts b
    assert list(ScriptUtils._best_cache) == ["aspirin", "c"]
    ScriptUtils._fullmap_best({"b": "b"})
    assert fetches[-1] == ["b"]


def test_a_miss_batch_larger_than_the_bound_still_answers_every_term(fetches: list[list[str]], monkeypatch: pytest.MonkeyPatch) -> None:
    """eviction during the store must not turn a resolved term into a silent miss"""
    monkeypatch.setattr(utils, "FULLMAP_BEST_CACHE_TERMS", 1)
    assert ScriptUtils._fullmap_best({"a": "aspirin", "b": "b", "c": "c"}) == {"aspirin": ROW}
    assert len(ScriptUtils._best_cache) <= 1


def test_cached_rows_carry_only_the_fields_resolve_reads(fetches: list[list[str]]) -> None:
    """_resolve reads CATEGORY_NAME, CURIE, PREFERRED_NAME; the resolved mention is unchanged"""
    resolved = ScriptUtils.resolve_mentions([("aspirin", "drug")])
    assert resolved[0].origin == "fullmap" and resolved[0].curie == "CHEBI:15365" and resolved[0].category == "Drug"


def _polars_best(rows: list[dict[str, object]], terms: list[str]) -> dict[str, dict[str, object]]:
    """the pre-change reference: filter_and_rank then keep the first row per term"""
    import polars as pl
    from tablassert.fullmap import filter_and_rank

    matches = filter_and_rank(pl.DataFrame(rows), pl.DataFrame({"term": terms, "nlp_level": [1] * len(terms)}), "9606", None, None, False)
    best: dict[str, dict[str, object]] = {}
    for match in matches.select("term", "CATEGORY_NAME", "CURIE", "PREFERRED_NAME").iter_rows(named=True):
        best.setdefault(str(match["term"]), {field: match[field] for field in ("CATEGORY_NAME", "CURIE", "PREFERRED_NAME")})
    return best


def test_the_python_ranking_matches_the_polars_reference_on_a_hand_built_table() -> None:
    """taxon filtering, the exact/normalized/other PR tiers, and the CURIE tie-break must all
    agree with filter_and_rank + first-row-per-term, or every resolved mention could shift"""
    rows = [
        row("aspirin", "CHEBI:15365", "aspirin", "ChemicalEntity"),  # PR 1: exact name
        row("aspirin", "NCBIGene:1", "Aspirin Gene", "Gene"),  # PR 10, loses on rank
        row("head ache", "SNOMED:25064002", "Headache", "Disease"),  # PR 5: normalized agreement
        row("head ache", "SNOMED:999", "Headache", "Disease"),  # same PR, CURIE tie-break
        row("flu", "FB:FBgn0004015", "flu", "Gene"),  # taxon 0 is retained
        row("drosophila", "FB:FBgn0004015", "drosophila", "Gene"),  # non-human taxon row
    ]
    rows[-1] = {**rows[-1], "TAXON_ID": 7227}
    terms = ["aspirin", "head ache", "flu", "drosophila"]
    assert ScriptUtils._rank_best(rows) == _polars_best(rows, terms)
    assert set(ScriptUtils._rank_best(rows)) == {"aspirin", "head ache", "flu"}
    assert ScriptUtils._rank_best(rows)["aspirin"]["CURIE"] == "CHEBI:15365"
    assert ScriptUtils._rank_best(rows)["head ache"]["CURIE"] == "SNOMED:25064002"


def test_a_table_that_survives_no_taxon_filter_resolves_to_nothing() -> None:
    rows = [{**row("mouse", "MGI:1", "mouse", "Gene"), "TAXON_ID": 10090}]
    assert ScriptUtils._rank_best(rows) == {} == _polars_best(rows, ["mouse"])


def test_an_empty_row_set_resolves_to_nothing() -> None:
    assert ScriptUtils._rank_best([]) == {}
