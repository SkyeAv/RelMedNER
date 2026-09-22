from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from tablassert import rs

from relmedner.constants import FUNCTION_WORDS, JUNKY_CATEGORIES, NONHUMAN_PREFIXES
from relmedner.fullmap_mine import FullmapMiner
from relmedner.models import FullmapTask, TrainingExample
from relmedner.utils import ScriptUtils

TASK: FullmapTask = FullmapTask(type="fullmap", outputs=["entities", "relations"])


# canonical lookup_rows row shape (tablassert.fullmap.lookup_rows contract)
def row(term: str, curie: str, name: str, category: str) -> dict[str, object]:
    return {"term": term, "CURIE": curie, "PREFERRED_NAME": name, "CATEGORY_NAME": category, "TAXON_ID": 0, "SOURCE_NAME": "t", "SOURCE_VERSION": "t"}


def exact_row(term: str, curie: str, name: str, category: str) -> dict[str, object]:
    """a row the EXACT gate accepts: normalize(PREFERRED_NAME) == term"""
    assert rs.normalize_terms([name])[0] == term, f"fixture bug: {name!r} does not normalize to {term!r}"
    return row(term, curie, name, category)


@pytest.fixture
def fake_rows(monkeypatch: pytest.MonkeyPatch):
    """inject lookup_rows results keyed by term so tests run without the fullmap mount"""

    def install(terms: list[dict[str, object]]) -> None:
        by_term: dict[str, list[dict[str, object]]] = {}
        for entry in terms:
            by_term.setdefault(str(entry["term"]), []).append(entry)
        monkeypatch.setattr("relmedner.fullmap_mine.lookup_rows", lambda _db, distinct: [r for term in distinct for r in by_term.get(term, [])])

    return install


# ------------------------------------------------------------------ candidate generation --


def test_candidates_enumerate_contiguous_ngrams_with_cleaned_surfaces() -> None:
    Text = "Asthma severity, COPD."
    candidates = FullmapMiner.ngram_candidates(0, Text, 2)
    surfaces = {candidate.surface for candidate in candidates}

    assert "Asthma" in surfaces and "Asthma severity" in surfaces and "COPD" in surfaces
    # trailing punctuation is stripped per token, and tokens that clean to empty are dropped
    assert "severity," not in surfaces and "COPD." not in surfaces


def test_candidates_can_never_cross_a_sentence() -> None:
    """the splitter emits '.' as its own token, which cleans to empty and kills the gram"""
    candidates = FullmapMiner.ngram_candidates(0, "Aspirin treats headache. The dose varies.", 6)

    assert candidates
    for candidate in candidates:
        assert "." not in candidate.surface
        assert not ("headache" in candidate.surface and "The" in candidate.surface)


def test_all_function_word_ngrams_are_dropped() -> None:
    candidates = FullmapMiner.ngram_candidates(0, "of the and in it", 6)

    assert candidates == []
    assert all(word in FUNCTION_WORDS for word in "of the and in it".split())


def test_hyphen_and_greek_folds_produce_extra_lookup_keys() -> None:
    candidates = FullmapMiner.ngram_candidates(0, "interleukin-1β and ampicillin", 2)
    keys = {candidate.key for candidate in candidates}

    # greek replacement precedes ascii folding (NFKD alone would strip 'β' to nothing)
    assert "interleukin 1beta" in keys
    # the splitter splits '/' on its own, so slash folding needs no extra key
    assert "ampicillin" in keys


def test_acronym_bridge_is_one_directional_and_flagged() -> None:
    candidates = FullmapMiner.ngram_candidates(0, "Body mass index (BMI) was measured.", 6)
    bridged = [candidate for candidate in candidates if candidate.bridged]

    assert [(c.surface, c.key) for c in bridged] == [("BMI", "Body mass index")]

    # an expansion without an in-document definition never bridges
    assert not [c for c in FullmapMiner.ngram_candidates(0, "BMI was measured.", 6) if c.bridged]


# ------------------------------------------------------------------------- resolution gates --


def test_exact_gate_requires_the_normalized_preferred_name_to_match(fake_rows) -> None:
    """PR is NOT the quality dial: normalize(PREFERRED_NAME)==term is; PR-free fixtures prove it"""
    fake_rows([exact_row("aspirin", "CHEBI:15365", "aspirin", "SmallMolecule")])
    example = FullmapMiner.resolve_batch([("Aspirin was given.", TASK)], db=None)[0]

    assert {mention for entity in example.entities for mention in entity.mentions} == {"Aspirin"}


def test_junk_category_gate_drops_qualifier_concepts(fake_rows) -> None:
    fake_rows([exact_row("aspirin", "UMLS:C0000000", "aspirin", next(iter(JUNKY_CATEGORIES)))])
    example = FullmapMiner.resolve_batch([("Aspirin was given.", TASK)], db=None)[0]

    assert example.entities == []


def test_unigram_minimum_length_with_digit_exemption(fake_rows) -> None:
    fake_rows(
        [
            exact_row("r", "CHEBI:8735", "r", "ChemicalEntity"),
            exact_row("cd63", "NCBIGene:967", "cd63", "Gene"),
        ]
    )
    example = FullmapMiner.resolve_batch([("R and CD63 were measured.", TASK)], db=None)[0]
    mentions = {mention for entity in example.entities for mention in entity.mentions}

    assert "CD63" in mentions and "R" not in mentions


def test_genelike_casing_rule_blocks_lowercase_english_collisions(fake_rows) -> None:
    """'was' -> NCBIGene WAS is exactly the collision the casing rule exists for"""
    fake_rows([exact_row("was", "NCBIGene:7454", "was", "Gene")])
    example = FullmapMiner.resolve_batch([("was recorded here.", TASK)], db=None)[0]

    assert example.entities == []

    # mixed-case gene surfaces survive the same rule
    fake_rows([exact_row("wtap", "NCBIGene:9589", "wtap", "Gene")])
    example = FullmapMiner.resolve_batch([("Wtap was measured.", TASK)], db=None)[0]

    assert [mention for entity in example.entities for mention in entity.mentions] == ["Wtap"]


def test_strict_unigram_agreement_keeps_inflection_but_kills_derivation(fake_rows) -> None:
    fake_rows([exact_row("cancer", "MONDO:0004992", "cancer", "Disease")])
    example = FullmapMiner.resolve_batch([("cancers were staged.", TASK)], db=None)[0]

    assert any("cancers" in entity.mentions for entity in example.entities)

    # 'oxidative' normalizes to the same key as 'oxide' but is not a plural form: rejected
    assert not FullmapMiner.strict_unigram_agrees("oxidative", "oxide")
    assert not FullmapMiner.strict_unigram_agrees("enters", "enteritis")
    assert FullmapMiner.strict_unigram_agrees("gliomas", "glioma")


def test_nonhuman_prefixes_are_excluded_before_ranking(fake_rows, monkeypatch: pytest.MonkeyPatch) -> None:
    seen_prefixes: list[str] = []
    real_rank = __import__("relmedner.fullmap_mine", fromlist=["filter_and_rank"]).filter_and_rank

    def spy_rank(frame: Any, *args: Any, **kwargs: Any) -> Any:
        seen_prefixes.extend(frame.select("CURIE").get_column("CURIE").to_list())
        return real_rank(frame, *args, **kwargs)

    monkeypatch.setattr("relmedner.fullmap_mine.filter_and_rank", spy_rank)
    fake_rows(
        [
            {
                "term": "aspirin",
                "CURIE": "FB:FBgn0004015",
                "PREFERRED_NAME": "aspirin",
                "CATEGORY_NAME": "Gene",
                "TAXON_ID": 0,
                "SOURCE_NAME": "t",
                "SOURCE_VERSION": "t",
            },
            exact_row("aspirin", "CHEBI:15365", "aspirin", "SmallMolecule"),
        ]
    )
    FullmapMiner.resolve_batch([("Aspirin was given.", TASK)], db=None)

    assert seen_prefixes and all(curie.split(":")[0] not in NONHUMAN_PREFIXES for curie in seen_prefixes)

    # the FlyBase collision must not leak a Gene mention into any example
    example = FullmapMiner.resolve_batch([("Aspirin was given.", TASK)], db=None)[0]
    assert [entity.label for entity in example.entities] == ["SmallMolecule"]


def test_bridged_acronym_bypasses_the_strict_unigram_gate(fake_rows) -> None:
    fake_rows([exact_row(rs.normalize_terms(["body mass index"])[0], "EFO:0004340", "body mass index", "PhenotypicFeature")])
    example = FullmapMiner.resolve_batch([("Body mass index (BMI) was recorded.", TASK)], db=None)[0]
    mentions = {mention for entity in example.entities for mention in entity.mentions}

    assert {"Body mass index", "BMI"} <= mentions


# ------------------------------------------------------- selection, relations, task model --


def test_greedy_longest_match_resolves_overlaps_and_rejects_function_word_boundaries(fake_rows) -> None:
    fake_rows(
        [
            exact_row(rs.normalize_terms(["breast cancer"])[0], "MONDO:0007254", "breast cancer", "Disease"),
            exact_row(rs.normalize_terms(["cancer"])[0], "MONDO:0004992", "cancer", "Disease"),
        ]
    )
    example = FullmapMiner.resolve_batch([("in breast cancer in", TASK)], db=None)[0]
    mentions = {mention for entity in example.entities for mention in entity.mentions}

    assert mentions == {"breast cancer"}  # longest match wins; 'in ... in' padding never emitted


def test_relations_carry_distant_evidence_and_drop_self_loops(fake_rows) -> None:
    fake_rows([exact_row("aspirin", "CHEBI:15365", "aspirin", "SmallMolecule")])
    example = FullmapMiner.resolve_batch([("Aspirin is associated with Aspirin.", TASK)], db=None)[0]

    assert example.relations == []  # head == tail mined pair dropped

    fake_rows([exact_row("aspirin", "CHEBI:15365", "aspirin", "SmallMolecule"), exact_row("headach", "HP:0000001", "headache", "Disease")])
    example = FullmapMiner.resolve_batch([("Aspirin is associated with headache.", TASK)], db=None)[0]

    assert [(r.name, {f.name: f.value for f in r.fields}, r.evidence, r.negated) for r in example.relations] == [
        ("associated_with", {"head": "Aspirin", "tail": "headache"}, "distant", False)
    ]


def test_negated_mined_relations_carry_the_not_name_distant_evidence_and_true_flag(fake_rows) -> None:
    """the distant path mirrors the gazetteer's gliner2-safe negative encoding: not_<predicate>
    name with negated=True, evidence upgraded to distant"""
    fake_rows([exact_row("aspirin", "CHEBI:15365", "aspirin", "SmallMolecule"), exact_row("headach", "HP:0000001", "headache", "Disease")])
    example = FullmapMiner.resolve_batch([("Aspirin is not associated with headache.", TASK)], db=None)[0]

    assert [(r.name, {f.name: f.value for f in r.fields}, r.evidence, r.negated) for r in example.relations] == [
        ("not_associated_with", {"head": "Aspirin", "tail": "headache"}, "distant", True)
    ]


def test_relations_false_task_suppresses_gazetteer(fake_rows) -> None:
    fake_rows([exact_row("aspirin", "CHEBI:15365", "aspirin", "SmallMolecule"), exact_row("headach", "HP:0000001", "headache", "Disease")])
    example = FullmapMiner.resolve_batch([("Aspirin is associated with headache.", TASK.model_copy(update={"relations": False}))], db=None)[0]

    assert example.entities and example.relations == []
    assert example.populated() == frozenset({"entities"})


def test_fullmap_task_model_validates_and_rejects_bad_knobs() -> None:
    Task = FullmapTask(type="fullmap", outputs=["entities"])

    assert Task.max_ngram == 6 and Task.taxon == "9606" and Task.relations is True

    with pytest.raises(ValidationError):
        FullmapTask(type="fullmap", max_ngram=0, outputs=["entities"])
    with pytest.raises(ValidationError):
        FullmapTask(type="fullmap")  # outputs required


def test_empty_and_textless_rows_degrade_to_bare_examples() -> None:
    example = FullmapMiner.resolve_batch([("of the and", TASK)], db=None)[0]

    assert example == TrainingExample(text="of the and")
    assert FullmapMiner.resolve_batch([], db=None) == []


@pytest.mark.skipif(not FullmapMiner.available(), reason="fullmap database is not mounted")
def test_live_fullmap_resolves_real_mentions_through_the_miner() -> None:
    """live redb smoke for the miner path; unit tests above stay un-gated via fake rows"""
    Text: str = "Statins have been associated with liver function. Ferinject treats iron deficiency and chronic kidney disease."
    Example: TrainingExample = FullmapMiner.resolve_batch([(Text, TASK)], db=FullmapMiner.db())[0]

    Labels: dict[str, list[str]] = {entity.label: entity.mentions for entity in Example.entities}
    assert "Disease" in Labels and "chronic kidney disease" in Labels["Disease"]
    assert any("liver function" in mentions for mentions in Labels.values())
    # every mined label is a real biolink class with fullmap-evidence descriptions
    for entity in Example.entities:
        assert ScriptUtils.is_biolink_category(entity.label)
        assert entity.description and "[fullmap:" in (entity.description or "")
    for relation in Example.relations:
        Fields: dict[str, str] = {field.name: field.value for field in relation.fields}
        assert Fields["head"].lower() != Fields["tail"].lower()
        assert relation.evidence == "distant"
