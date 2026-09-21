from __future__ import annotations

from typing import Any, ClassVar, Self

import pytest

from relmedner.models import Entity, Relation, RelationField, TrainingExample
from relmedner.scripts import GlinerBiomedScript
from relmedner.types import DispatchedExample, Script, ScriptValues
from relmedner.utils import ResolvedMention, ScriptUtils, strip_biolink_prefix


class StubScript(Script):
    NAME: ClassVar[str] = "StubScript"

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        text, label = values
        return TrainingExample(text=text or "", entities=[Entity(label=label or "", mentions=[text or ""])])


def test_subclasses_self_register_on_import() -> None:
    assert Script.REGISTRY["GlinerBiomedScript"] is not None
    assert isinstance(Script.REGISTRY["GlinerBiomedScript"], GlinerBiomedScript)
    assert isinstance(Script.REGISTRY["StubScript"], StubScript)


def test_the_registry_keys_on_the_declared_name() -> None:
    assert all(Name == type(Instance).NAME for Name, Instance in Script.REGISTRY.items())


def test_dispatch_routes_by_name_and_preserves_declared_outputs() -> None:
    Dispatched: DispatchedExample = Script.dispatch("StubScript", (("entities",), ("Alice", "person")))
    Outputs, Example = Dispatched

    assert Outputs == ("entities",)
    assert Example.text == "Alice"
    assert Example.entities == [Entity(label="person", mentions=["Alice"])]


def test_dispatch_on_an_unregistered_name_raises() -> None:
    with pytest.raises(KeyError):
        Script.dispatch("NoSuchScript", (("entities",), ("Alice", "person")))


def test_join_tokens_glues_with_single_spaces() -> None:
    assert ScriptUtils.join_tokens(["Aspirin", "treats", "headache", "."]) == "Aspirin treats headache ."


def test_mentions_slice_spans_with_inclusive_ends_and_skip_out_of_bounds() -> None:
    Tokens: list[str] = ["Ankle", "sprain", "is", "common", "."]
    Ner: list[list[Any]] = [[0, 1, "Condition"], [4, 4, "Identifier"], [0, 9, "Broken"], [-1, 1, "Broken"], [3, 2, "Broken"]]

    assert ScriptUtils.mentions(Tokens, Ner) == [("Ankle sprain", "Condition"), (".", "Identifier")]


def test_biolink_categories_have_no_prefix_and_membership_checks() -> None:
    Categories: frozenset[str] = ScriptUtils.biolink_categories()

    assert "Gene" in Categories and "Drug" in Categories and "Disease" in Categories
    assert all(not category.startswith("biolink:") for category in Categories)
    assert ScriptUtils.is_biolink_category("Gene") is True
    assert ScriptUtils.is_biolink_category("Identifier") is False
    assert strip_biolink_prefix("biolink:Gene") == "Gene"
    assert strip_biolink_prefix("Gene") == "Gene"


def test_every_fallback_label_maps_to_a_biolink_category() -> None:
    for raw_label, category in ScriptUtils.FALLBACK_LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"fallback {raw_label!r} -> {category!r} is not a biolink class"


def test_the_script_groups_mentions_by_resolved_category(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resolve(spans: list[tuple[str, str]]) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="Aspirin", category="Drug", curie="CHEBI:15365", preferred_name="Acetylsalicylic acid", origin="fullmap"),
            ResolvedMention(mention="headache", category="Disease", origin="fallback"),
            ResolvedMention(mention=".", category="Identifier", origin="raw"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens: list[str] = ["Aspirin", "treats", "headache", "."]
    Ner: list[list[Any]] = [[0, 0, "Drug"], [2, 2, "Condition"], [3, 3, "Identifier"]]
    _, Example = Script.dispatch("GlinerBiomedScript", (("entities",), (Tokens, Ner)))

    assert Example.text == "Aspirin treats headache ."
    assert Example.populated() == frozenset({"entities", "relations"})
    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "Drug": ["Aspirin"],
        "Disease": ["headache"],
        "Identifier": ["."],
    }
    by_label = {entity.label: entity for entity in Example.entities}
    assert by_label["Drug"].description is not None and "[fullmap: CHEBI:15365 | Acetylsalicylic acid]" in by_label["Drug"].description
    assert by_label["Disease"].description is not None and "fullmap" not in by_label["Disease"].description
    assert by_label["Identifier"].description is None


def test_the_script_emits_nothing_for_empty_rows() -> None:
    _, Example = Script.dispatch("GlinerBiomedScript", (("entities",), ([], [])))
    assert Example.text == ""
    assert Example.populated() == frozenset()


def test_mention_spans_keeps_in_bounds_triples_and_drops_the_rest() -> None:
    """US-003 relation wiring consumes (start, end_inclusive, label) triples with mentions' bounds contract"""
    Tokens: list[str] = ["Ankle", "sprain", "is", "common", "."]
    Ner: list[list[Any]] = [[0, 1, "Condition"], [4, 4, "Identifier"], [0, 9, "Broken"], [-1, 1, "Broken"], [3, 2, "Broken"]]

    assert ScriptUtils.mention_spans(Tokens, Ner) == [(0, 1, "Condition"), (4, 4, "Identifier")]


def test_mention_spans_skips_malformed_entries_without_coercion() -> None:
    """malformed external NER rows must not crash or become silently coerced mentions"""
    Tokens: list[str] = ["Aspirin", "treats", "migraine"]
    Ner: list[Any] = [
        [0, 0, "Drug"],
        (2, 2, "Condition"),
        [],
        [0, 0],
        [0, 0, "Drug", "extra"],
        None,
        3,
        "span",
        [True, 0, "Drug"],
        [0, False, "Drug"],
        ["0", 0, "Drug"],
        [0, 0.0, "Drug"],
        [None, 0, "Drug"],
        [0, 0, None],
        [3, 3, "Broken"],
    ]

    assert ScriptUtils.mention_spans(Tokens, Ner) == [(0, 0, "Drug"), (2, 2, "Condition")]
    assert ScriptUtils.mentions(Tokens, Ner) == [("Aspirin", "Drug"), ("migraine", "Condition")]


def test_mentions_rejoins_each_mention_span_slice() -> None:
    """mentions is mention_spans with each triple rejoined, so script wiring can pair the two positionally"""
    Tokens: list[str] = ["Ankle", "sprain", "is", "common", "."]
    Ner: list[list[Any]] = [[0, 1, "Condition"], [4, 4, "Identifier"], [0, 9, "Broken"]]
    Spans: list[tuple[int, int, str]] = ScriptUtils.mention_spans(Tokens, Ner)

    assert ScriptUtils.mentions(Tokens, Ner) == [(" ".join(Tokens[start : end + 1]), label) for start, end, label in Spans]


def test_the_script_skips_malformed_ner_entries_before_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """mixed valid and malformed NER rows must preserve valid entity and relation alignment"""

    def fake_resolve(spans: list[tuple[str, str]]) -> list[ResolvedMention]:
        assert spans == [("Aspirin", "Drug"), ("migraine", "Condition")]
        return [
            ResolvedMention(mention="Aspirin", category="Drug", origin="fallback"),
            ResolvedMention(mention="migraine", category="Disease", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens: list[str] = ["Aspirin", "treats", "migraine"]
    Ner: list[Any] = [[0, 0, "Drug"], None, (2, 2, "Condition"), [True, 1, "Drug"], [0, 0, None]]
    _, Example = Script.dispatch("GlinerBiomedScript", (("entities",), (Tokens, Ner)))

    assert {entity.label: entity.mentions for entity in Example.entities} == {"Drug": ["Aspirin"], "Disease": ["migraine"]}
    assert Example.relations == [expected_relation("treats", "Aspirin", "migraine")]


def test_the_script_emits_gazetteer_relations_between_resolved_mentions(monkeypatch: pytest.MonkeyPatch) -> None:
    """a trigger phrase between two mentions must surface a treats relation over resolved categories"""

    def fake_resolve(spans: list[tuple[str, str]]) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="Aspirin", category="Drug", origin="fallback"),
            ResolvedMention(mention="migraine", category="Disease", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens: list[str] = ["Aspirin", "is", "used", "to", "treat", "migraine"]
    Ner: list[list[Any]] = [[0, 0, "Drug"], [5, 5, "Condition"]]
    _, Example = Script.dispatch("GlinerBiomedScript", (("entities",), (Tokens, Ner)))

    assert Example.relations == [expected_relation("treats", "Aspirin", "migraine")]


def test_the_script_emits_no_relations_on_rows_without_a_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    """mentions with no predicate phrase between them carry no relation signal"""

    def fake_resolve(spans: list[tuple[str, str]]) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="Aspirin", category="Drug", origin="fallback"),
            ResolvedMention(mention="migraine", category="Disease", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens: list[str] = ["Aspirin", "and", "migraine", "coexist", "."]
    Ner: list[list[Any]] = [[0, 0, "Drug"], [2, 2, "Condition"]]
    _, Example = Script.dispatch("GlinerBiomedScript", (("entities",), (Tokens, Ner)))

    assert Example.relations == []


@pytest.mark.skipif(not ScriptUtils.fullmap_available(), reason="fullmap database is not mounted")
def test_fullmap_resolves_real_mentions_to_biolink_categories() -> None:
    Resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions([("Aspirin", "Drug"), ("totally-unresolvable-xyz", "Drug")])

    assert Resolved[0].origin == "fullmap"
    assert ScriptUtils.is_biolink_category(Resolved[0].category)
    assert Resolved[0].curie is not None and ":" in Resolved[0].curie
    assert Resolved[1].origin in {"fallback", "raw"}


# ---------------------------------------------------------------------------
# PileNerBiomedScript + the shared decode/casing/gate primitives (Pile-NER-biomed-IOB ingest)
# ---------------------------------------------------------------------------


def test_parse_literal_list_decodes_python_repr_string_columns() -> None:
    """the pile-ner parquet stores tokens/ner_tags as python-repr strings, not arrays"""
    assert ScriptUtils.parse_literal_list("['a', 'b']") == ["a", "b"]
    assert ScriptUtils.parse_literal_list("['O', 'B-organism', 'I-organism']") == ["O", "B-organism", "I-organism"]
    assert ScriptUtils.parse_literal_list(["already", "a", "list"]) == ["already", "a", "list"]


def test_parse_literal_list_skips_malformed_rows_without_raising() -> None:
    """skip-don't-coerce: a malformed external row becomes an empty example, never a crash"""
    assert ScriptUtils.parse_literal_list("'just a string'") == []
    assert ScriptUtils.parse_literal_list("['a', 1]") == []
    assert ScriptUtils.parse_literal_list("garbage [") == []
    assert ScriptUtils.parse_literal_list(None) == []
    assert ScriptUtils.parse_literal_list(3) == []


def test_iob_spans_decodes_iob2_with_inclusive_ends() -> None:
    assert ScriptUtils.iob_spans(["O", "B-organism", "I-organism", "O", "B-disease", "I-disease", "I-disease", "O"]) == [
        (1, 2, "organism"),
        (4, 6, "disease"),
    ]
    assert ScriptUtils.iob_spans([]) == []
    assert ScriptUtils.iob_spans(["O", "O"]) == []


def test_iob_spans_promotes_orphan_i_tags_to_single_token_spans() -> None:
    """1,158 orphan I- tags exist in the corpus; each carries a real mention that must survive"""
    assert ScriptUtils.iob_spans(["I-orphan", "O", "I-x", "I-x"]) == [(0, 0, "orphan"), (2, 3, "x")]
    assert ScriptUtils.iob_spans(["B-a", "I-b", "I-a"]) == [(0, 0, "a"), (1, 1, "b"), (2, 2, "a")]
    assert ScriptUtils.iob_spans(["E-garbage", "O"]) == []


def test_normalize_and_pascal_label_collapse_corpus_label_variants() -> None:
    """97 underscore-variant labels collapse through normalize; pascal_label names raw tail labels
    the way biolink classes are named so the whole 3,896-type vocabulary stays uniform"""
    assert ScriptUtils.normalize_iob_label("Anatomical_Structure") == "anatomical structure"
    assert ScriptUtils.normalize_iob_label("Medical_Condition ") == "medical condition"
    assert ScriptUtils.pascal_label("medical condition") == "MedicalCondition"
    assert ScriptUtils.pascal_label("anatomical_structure") == "AnatomicalStructure"
    assert ScriptUtils.pascal_label("gene/protein") == "GeneProtein"
    assert ScriptUtils.pascal_label("cell type") == "CellType"
    assert ScriptUtils.pascal_label("") == ""


def test_every_pile_ner_fallback_label_maps_to_a_biolink_category() -> None:
    """dataset-local vocabulary values stay real biolink classes"""
    from relmedner.scripts import PileNerBiomedScript

    for raw_label, category in PileNerBiomedScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"fallback {raw_label!r} -> {category!r} is not a biolink class"


def test_the_fallback_lookup_normalizes_pile_ner_labels() -> None:
    """pile-ner labels use a dataset-local map merged over the shared map"""
    Resolved: list[ResolvedMention] = ScriptUtils.resolve_mentions([])
    assert Resolved == []
    from relmedner.scripts import PileNerBiomedScript

    assert PileNerBiomedScript.LABEL_MAP["medical condition"] == "Disease"


def test_the_pile_ner_script_decodes_real_row_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """end-to-end: python-repr columns, IOB decode, resolution, grouping, and relation extraction"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("Trypanosoma cruzi", "organism"), ("Chagas disease", "disease"), ("DTUs", "abbreviation")]
        return [
            ResolvedMention(
                mention="Trypanosoma cruzi",
                category="OrganismTaxon",
                curie="NCBITaxon:5693",
                preferred_name="Trypanosoma cruzi",
                origin="fullmap",
            ),
            ResolvedMention(
                mention="Chagas disease",
                category="Disease",
                curie="MONDO:0001444",
                preferred_name="Chagas disease",
                origin="fullmap",
            ),
            ResolvedMention(mention="DTUs", category="Abbreviation", origin="raw"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens = "['Trypanosoma', 'cruzi', ',', 'the', 'agent', 'of', 'Chagas', 'disease', '.', 'DTUs', '.']"
    Tags = "['B-organism', 'I-organism', 'O', 'O', 'O', 'O', 'B-disease', 'I-disease', 'O', 'B-abbreviation', 'O']"
    _, Example = Script.dispatch("PileNerBiomedScript", (("entities",), (Tokens, Tags)))

    assert Example.text == "Trypanosoma cruzi , the agent of Chagas disease . DTUs ."
    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "OrganismTaxon": ["Trypanosoma cruzi"],
        "Disease": ["Chagas disease"],
        "Abbreviation": ["DTUs"],
    }
    assert Example.entities[0].description is not None and "[fullmap: NCBITaxon:5693 | Trypanosoma cruzi]" in Example.entities[0].description
    assert Example.entities[2].description is None  # raw tail labels carry no description


def test_the_pile_ner_script_pascalcases_raw_labels_but_keeps_fallback_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    """raw labels surface biolink-cased (decision: keep the tail, case it like biolink);
    fallback-resolved labels already name a biolink class and must stay untouched"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="ibuprofen", category="Drug", origin="fallback"),
            ResolvedMention(mention="some widget", category="job title", origin="raw"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens = "['ibuprofen', 'is', 'a', 'some', 'widget', '.']"
    Tags = "['B-drug', 'O', 'O', 'B-job title', 'I-job title', 'O']"
    _, Example = Script.dispatch("PileNerBiomedScript", (("entities",), (Tokens, Tags)))

    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "Drug": ["ibuprofen"],
        "JobTitle": ["some widget"],
    }


def test_the_pile_ner_script_emits_nothing_for_empty_or_mismatched_rows() -> None:
    _, Empty = Script.dispatch("PileNerBiomedScript", (("entities",), ("[]", "[]")))
    assert Empty.text == ""
    assert Empty.populated() == frozenset()
    _, Malformed = Script.dispatch("PileNerBiomedScript", (("entities",), ("garbage [", "'not a list'")))
    assert Malformed.text == ""
    _, Mismatched = Script.dispatch("PileNerBiomedScript", (("entities",), ("['a', 'b']", "['O']")))
    assert Mismatched.text == "a b"
    assert Mismatched.populated() == frozenset()


def test_the_pile_ner_script_extracts_relations_over_resolved_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    """gates live inside extract_relations over span categories; a compatible pair emits an edge"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="dexamethasone", category="SmallMolecule", curie="CHEBI:41180", preferred_name="dexamethasone", origin="fullmap"),
            ResolvedMention(mention="COPD", category="Disease", curie="MONDO:0005002", preferred_name="COPD", origin="fullmap"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens = "['dexamethasone', 'in', 'the', 'treatment', 'of', 'COPD', '.']"
    Tags = "['B-drug', 'O', 'O', 'O', 'O', 'B-disease', 'O']"
    _, Example = Script.dispatch("PileNerBiomedScript", (("entities",), (Tokens, Tags)))

    assert Example.relations == [expected_relation("treats", "dexamethasone", "COPD")]


def test_the_pile_ner_script_relation_gate_rejects_incompatible_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    """biolink domain/range: expressed_in needs a gene-ish head; a chemical head must not emit"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="benzene", category="ChemicalEntity", curie="CHEBI:167164", preferred_name="benzene", origin="fullmap"),
            ResolvedMention(mention="epithelial cells", category="Cell", curie="CL:0000066", preferred_name="epithelial cell", origin="fullmap"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens = "['benzene', 'expressed', 'in', 'epithelial', 'cells', '.']"
    Tags = "['B-chemical', 'O', 'O', 'B-cell type', 'I-cell type', 'O']"
    _, Example = Script.dispatch("PileNerBiomedScript", (("entities",), (Tokens, Tags)))

    assert Example.relations == []


def expected_relation(name: str, head: str, tail: str) -> Relation:
    """emitted relations carry their biolink slot description; build the matching expectation"""
    return Relation(
        name=name,
        fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
        description=ScriptUtils.predicate_description(name),
    )
