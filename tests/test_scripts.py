from __future__ import annotations

import json
from typing import Any, ClassVar, Self

import pytest

from relmedner.fullmap_mine import FullmapMiner
from relmedner.models import Entity, Relation, RelationField, TrainingExample
from relmedner.scripts import BioleafletsScript, GlinerBiomedScript, KnowledgatorBiomedScript, SentenceRexScript
from relmedner.scripts.sentence_rex import parse_tagged_sentence
from relmedner.types import DispatchedExample, Script, ScriptValues
from relmedner.utils import ResolutionGate, ResolvedMention, ScriptUtils, strip_biolink_prefix


class StubScript(Script):
    NAME: ClassVar[str] = "StubScript"

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        text, label = values
        return TrainingExample(text=text or "", entities=[Entity(label=label or "", mentions=[text or ""])])


def test_subclasses_self_register_on_import() -> None:
    assert Script.REGISTRY["GlinerBiomedScript"] is not None
    assert isinstance(Script.REGISTRY["GlinerBiomedScript"], GlinerBiomedScript)
    assert isinstance(Script.REGISTRY["StubScript"], StubScript)
    assert isinstance(Script.REGISTRY["SentenceRexScript"], SentenceRexScript)


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


def test_lookup_label_probes_the_lowercased_then_the_normalized_key() -> None:
    """one map may key either form: the raw lowercase first, then the IOB-normalized variant, so
    'Anatomical_Structure' reaches an 'anatomical structure' entry without every caller spelling
    out both probes (the resolution chain's fallback map and the script LABEL_MAPs share the rule)"""
    Lowered: dict[str, str] = {"disease": "Disease"}
    Normalized: dict[str, str] = {"anatomical structure": "AnatomicalEntity"}

    assert ScriptUtils.lookup_label(Lowered, "DISEASE") == "Disease"
    assert ScriptUtils.lookup_label(Normalized, "Anatomical_Structure") == "AnatomicalEntity"
    assert ScriptUtils.lookup_label(Normalized, "anatomical structure") == "AnatomicalEntity"
    # a miss stays a miss rather than raising; the `or` chain also means a map must never carry an
    # empty target, which validate_label_map already rejects because "" is not a biolink class
    assert ScriptUtils.lookup_label(Lowered, "missing") is None


def test_pascal_raw_labels_folds_only_the_raw_origin() -> None:
    """the shared raw-tail rule every script rides: fullmap and fallback categories already name a
    biolink class and survive verbatim, while a raw label is dataset vocabulary and trains under a
    biolink-shaped name. No parallel label sequence is needed because _resolve copies the raw label
    into category on the raw path"""
    Resolved: list[ResolvedMention] = [
        ResolvedMention(mention="aspirin", category="Drug", curie="CHEBI:15365", preferred_name="aspirin", origin="fullmap"),
        ResolvedMention(mention="headache", category="Disease", origin="fallback"),
        ResolvedMention(mention="wrist", category="anatomical_structure", origin="raw"),
    ]

    Labeled: list[ResolvedMention] = ScriptUtils.pascal_raw_labels(Resolved)

    assert [item.category for item in Labeled] == ["Drug", "Disease", "AnatomicalStructure"]
    # provenance and fullmap evidence ride through the fold, and the caller's list is not mutated
    assert [item.origin for item in Labeled] == ["fullmap", "fallback", "raw"]
    assert (Labeled[0].curie, Labeled[0].preferred_name) == ("CHEBI:15365", "aspirin")
    assert Resolved[2].category == "anatomical_structure"


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


# ---------------------------------------------------------------------------
# SentenceRexScript + parse_tagged_sentence (knowledgator/sentence_rex ingest)
# ---------------------------------------------------------------------------


def test_parse_tagged_sentence_extracts_both_surfaces_from_real_rows() -> None:
    """row shapes verified verbatim on the knowledgator/sentence_rex dataset card; e1/e2 inner
    whitespace ('<e1> Myristica fragrans </e1>') is lost by the surfaces but must stay in the text
    (43,044/43,044 well-formed rows keep both stripped surfaces verbatim in the tag-stripped text)"""
    Pope = (
        "<e1>Pope Pius XII</e1> re - opened the cause on 7 December 1954 , and Pope John Paul II proclaimed him <e2> Venerable </e2> on 6 July 1985 ."
    )
    Nutmeg = (
        'It is sometimes called the " nutmeg family " , after its most famous member , '
        "<e1> Myristica fragrans </e1> , the source of the spices <e2> nutmeg </e2> and mace ."
    )

    assert parse_tagged_sentence(Pope) == ("Pope Pius XII", "Venerable")
    assert parse_tagged_sentence(Nutmeg) == ("Myristica fragrans", "nutmeg")


def test_parse_tagged_sentence_requires_exactly_one_of_each_tag() -> None:
    """521 measured rows break the (1, 1, 1, 1) tag-count contract; each returns None instead of a
    partially coerced parse (skip-don't-coerce for external data), and tag order is irrelevant"""
    assert parse_tagged_sentence("<e2> B </e2> then <e1> A </e1>") == ("A", "B")

    for Broken in (
        "no tags at all",
        "<e1> A </e1> but no e2 tags",
        "<e1> A </e1> <e1> B </e1> <e2> C </e2>",
        "<e1> A </e1> <e2> B </e2> </e2>",
        "<e1> A </e1> <e2> B </e2> <e2> C </e2> </e2>",
        "<e1></e1> <e2> B </e2> <e1> C </e1>",
    ):
        assert parse_tagged_sentence(Broken) is None


def test_parse_tagged_sentence_guards_nested_empty_and_self_loop_surfaces() -> None:
    """18 measured rows carry '<' inside a stripped surface (the real nested shape below), 0 empty
    surfaces, and 48 case-insensitive self-loops; all return None so no impossible example ships"""
    Nested = (
        "Pharmacodynamics=== Doxylamine acts primarily as an antagonist or inverse agonist of the "
        "<e1> histamine </e1> <e2> H < sub>1</sub > receptor </e2> ."
    )

    assert parse_tagged_sentence(Nested) is None
    assert parse_tagged_sentence("<e1> H <br> </e1> x <e2> T </e2>") is None
    assert parse_tagged_sentence("<e1> </e1> x <e2> T </e2>") is None
    assert parse_tagged_sentence("<e1></e1> x <e2> T </e2>") is None
    assert parse_tagged_sentence("<e1> H </e1> x <e2> H </e2>") is None
    assert parse_tagged_sentence("<e1> H </e1> x <e2> h </e2>") is None
    assert parse_tagged_sentence("<e1> H </e1> x <e2> T </e2>") == ("H", "T")


def test_the_sentence_rex_script_emits_asserted_relations_for_real_rows() -> None:
    """end-to-end over the card's rows 0 and 1: text strips ONLY the four tag literals, so the
    doubled spaces around '<e2> Venerable </e2>' survive (gliner2 whitespace-tokenizes them inert);
    native labels keep snake_case names with description None (17 of 837 measured labels are
    biolink members, so None is the common case and must be handled)"""
    Pope = (
        "<e1>Pope Pius XII</e1> re - opened the cause on 7 December 1954 , and Pope John Paul II proclaimed him <e2> Venerable </e2> on 6 July 1985 ."
    )
    Nutmeg = (
        'It is sometimes called the " nutmeg family " , after its most famous member , '
        "<e1> Myristica fragrans </e1> , the source of the spices <e2> nutmeg </e2> and mace ."
    )
    Expected = "Pope Pius XII re - opened the cause on 7 December 1954 , and Pope John Paul II proclaimed him  Venerable  on 6 July 1985 ."
    _, Example = Script.dispatch("SentenceRexScript", (("relations",), (Pope, "canonization status")))

    assert Example.text == Expected
    assert Example.populated() == frozenset({"relations"})
    assert Example.relations == [expected_relation("canonization_status", "Pope Pius XII", "Venerable")]
    assert Example.relations[0].description is None and Example.relations[0].negated is False
    _, NutmegExample = Script.dispatch("SentenceRexScript", (("relations",), (Nutmeg, "this taxon is source of")))
    assert NutmegExample.relations == [expected_relation("this_taxon_is_source_of", "Myristica fragrans", "nutmeg")]


def test_the_sentence_rex_script_resolves_biolink_member_labels_with_descriptions() -> None:
    """'expressed in' (a real card row) is one of the 17 measured biolink-member labels: the emitted
    predicate is the snake_case member and its biolink slot definition rides as the description"""
    Expressed = "No <e1> UDP glucuronosyltransferase 1 - A1 </e1> expression can be detected in the <e2> liver </e2> tissue ."
    assert ScriptUtils.resolve_predicate("expressed in") == ("expressed_in", True)
    _, Example = Script.dispatch("SentenceRexScript", (("relations",), (Expressed, "expressed in")))

    assert Example.relations == [expected_relation("expressed_in", "UDP glucuronosyltransferase 1 - A1", "liver")]
    assert Example.relations[0].description is not None


def test_the_sentence_rex_script_emits_nothing_for_bad_rows() -> None:
    """every drop rule lands on the empty example downstream filtering removes: 550 null/blank
    sentences, 521 malformed tag counts, 18 nested, 0 empty surfaces, 48 self-loops, plus blank or
    null labels on otherwise well-formed rows"""
    Good = "<e1> H </e1> x <e2> T </e2>"

    for Values in (
        (None, "canonization status"),
        ("", "canonization status"),
        ("   ", "canonization status"),
        (Good, None),
        (Good, ""),
        (Good, "   "),
        ("no tags at all", "canonization status"),
        ("<e1> H </e1> <e1> H </e1> x <e2> T </e2>", "canonization status"),
        ("<e1> H <br> </e1> x <e2> T </e2>", "canonization status"),
        ("<e1> </e1> x <e2> T </e2>", "canonization status"),
        ("<e1> H </e1> x <e2> H </e2>", "canonization status"),
    ):
        _, Example = Script.dispatch("SentenceRexScript", (("relations",), Values))
        assert Example.text == ""
        assert Example.populated() == frozenset()


# ---------------------------------------------------------------------------
# PileNerTypeScript (Universal-NER/Pile-NER-type ingest: conversation-QA rows)
# ---------------------------------------------------------------------------


def conversations(text: str, *answered: tuple[str, str]) -> list[dict[str, str]]:
    """build a Pile-NER-type conversations column: the 'Text: ' turn, the corpus's fixed
    acknowledgement, then one templated question per entity type with its JSON answer turn"""
    turns: list[dict[str, str]] = [
        {"from": "human", "value": f"Text: {text}"},
        {"from": "gpt", "value": "I've read this text."},
    ]
    for raw_label, answer in answered:
        turns.append({"from": "human", "value": f"What describes {raw_label} in the text?"})
        turns.append({"from": "gpt", "value": answer})
    return turns


def test_parse_conversations_decodes_the_real_row_shape() -> None:
    """row ner_1 shape: a 'Text: ' turn plus one question/JSON-answer pair per entity type"""
    Turns = conversations(
        "Juho Haapoja was a Finnish boxer who held the European Union cruiserweight title .",
        ("Person", '["Juho Haapoja"]'),
        ("Organization", '["European Union"]'),
        ("Date", "[]"),
    )

    assert ScriptUtils.parse_conversations(Turns) == (
        "Juho Haapoja was a Finnish boxer who held the European Union cruiserweight title .",
        [("Person", ["Juho Haapoja"]), ("Organization", ["European Union"]), ("Date", [])],
    )


def test_parse_conversations_decodes_python_repr_columns_too() -> None:
    """same skip-don't-coerce decode posture as the python-repr pile-ner-biomed columns"""
    Turns = conversations("Aspirin treats migraine .", ("drug", '["Aspirin"]'))

    assert ScriptUtils.parse_conversations(str(Turns)) == ("Aspirin treats migraine .", [("drug", ["Aspirin"])])


def test_parse_conversations_skips_malformed_rows_without_raising() -> None:
    """a malformed external row becomes an empty example, never a crash"""
    assert ScriptUtils.parse_conversations("garbage [") == ("", [])
    assert ScriptUtils.parse_conversations(None) == ("", [])
    assert ScriptUtils.parse_conversations([]) == ("", [])
    assert ScriptUtils.parse_conversations([{"from": "gpt", "value": "I've read this text."}]) == ("", [])
    assert ScriptUtils.parse_conversations([{"from": "human", "value": "no prefix here"}]) == ("", [])
    assert ScriptUtils.parse_conversations([None, 3, {"from": "human", "value": "Text: kept"}]) == ("kept", [])


def test_parse_mention_list_keeps_strings_and_drops_everything_else() -> None:
    """answers are JSON arrays; python-repr is accepted too and non-string items are dropped"""
    assert ScriptUtils.parse_mention_list('["Aspirin", " migraine "]') == ["Aspirin", "migraine"]
    assert ScriptUtils.parse_mention_list("['Aspirin']") == ["Aspirin"]
    assert ScriptUtils.parse_mention_list('["Aspirin", 3, null, ""]') == ["Aspirin"]
    assert ScriptUtils.parse_mention_list("[]") == []
    assert ScriptUtils.parse_mention_list("I've read this text.") == []


def test_token_occurrences_finds_every_hit_case_insensitively() -> None:
    """the corpus answers with surfaces, so spans are recovered by token-subsequence match"""
    Tokens: list[str] = ["Aspirin", "treats", "migraine", ".", "aspirin", "again"]

    assert ScriptUtils.token_occurrences(Tokens, "aspirin") == [(0, 0), (4, 4)]
    assert ScriptUtils.token_occurrences(Tokens, "Aspirin treats") == [(0, 1)]
    assert ScriptUtils.token_occurrences(Tokens, "never appears") == []
    assert ScriptUtils.token_occurrences(Tokens, "") == []
    assert ScriptUtils.token_occurrences([], "Aspirin") == []


def test_token_occurrences_accepts_a_precomputed_haystack() -> None:
    """the hoisted form must be identical to folding per call. A row matches dozens of mentions
    against one document, so the script folds lowered_tokens(tokens) once and passes it in; a
    haystack that disagreed with tokens would silently misplace every span in that row, which is
    why the equivalence is pinned rather than assumed."""
    Tokens: list[str] = ["Aspirin", "treats", "Migraine", "and", "migraine", "again"]
    Lowered: list[str] = ScriptUtils.lowered_tokens(Tokens)

    assert Lowered == ["aspirin", "treats", "migraine", "and", "migraine", "again"]
    for Mention in ("migraine", "Aspirin treats", "MIGRAINE AND migraine", "never appears", ""):
        assert ScriptUtils.token_occurrences(Tokens, Mention, Lowered) == ScriptUtils.token_occurrences(Tokens, Mention)


def test_every_pile_ner_type_fallback_label_maps_to_a_biolink_category() -> None:
    """dataset-local vocabulary values stay real biolink classes"""
    from relmedner.scripts import PileNerTypeScript

    for raw_label, category in PileNerTypeScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"fallback {raw_label!r} -> {category!r} is not a biolink class"


def test_the_pile_ner_type_script_decodes_real_row_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """end-to-end: conversation decode, surface->span recovery, resolution, and grouping"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("Trypanosoma cruzi", "species"), ("Chagas disease", "disease")]
        return [
            ResolvedMention(
                mention="Trypanosoma cruzi",
                category="OrganismTaxon",
                curie="NCBITaxon:5693",
                preferred_name="Trypanosoma cruzi",
                origin="fullmap",
            ),
            ResolvedMention(mention="Chagas disease", category="Disease", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Turns = conversations(
        "Trypanosoma cruzi , the agent of Chagas disease , presents a clonal structure .",
        ("species", '["Trypanosoma cruzi"]'),
        ("disease", '["Chagas disease"]'),
        ("Date", "[]"),
    )
    _, Example = Script.dispatch("PileNerTypeScript", (("entities",), (Turns,)))

    assert Example.text == "Trypanosoma cruzi , the agent of Chagas disease , presents a clonal structure ."
    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "OrganismTaxon": ["Trypanosoma cruzi"],
        "Disease": ["Chagas disease"],
    }
    assert Example.entities[0].description is not None and "[fullmap: NCBITaxon:5693 | Trypanosoma cruzi]" in Example.entities[0].description


def test_the_pile_ner_type_script_drops_mentions_absent_from_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """gpt answers occasionally name surfaces the document never contains; those must not train"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("Aspirin", "drug")]
        return [ResolvedMention(mention="Aspirin", category="Drug", origin="fallback")]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Turns = conversations("Aspirin eases the pain .", ("drug", '["Aspirin", "ibuprofen"]'))
    _, Example = Script.dispatch("PileNerTypeScript", (("entities",), (Turns,)))

    assert {entity.label: entity.mentions for entity in Example.entities} == {"Drug": ["Aspirin"]}


def test_the_pile_ner_type_script_pascalcases_raw_labels_and_keeps_document_casing(monkeypatch: pytest.MonkeyPatch) -> None:
    """raw tails surface biolink-cased while fallback categories stay untouched; the emitted mention
    is the document's own token slice so gliner2's sanitizer can always find it in the text"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("Aspirin", "drug"), ("Contoso", "job title")]
        return [
            ResolvedMention(mention="Aspirin", category="Drug", origin="fallback"),
            ResolvedMention(mention="Contoso", category="job title", origin="raw"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Turns = conversations("Aspirin ships from Contoso .", ("drug", '["aspirin"]'), ("job title", '["contoso"]'))
    _, Example = Script.dispatch("PileNerTypeScript", (("entities",), (Turns,)))

    assert {entity.label: entity.mentions for entity in Example.entities} == {"Drug": ["Aspirin"], "JobTitle": ["Contoso"]}


def test_the_pile_ner_type_script_emits_nothing_for_empty_negative_or_malformed_rows() -> None:
    """negative-sampled rows (every answer '[]') and malformed rows fall out on the outputs gate"""
    _, Negative = Script.dispatch("PileNerTypeScript", (("entities",), (conversations("REDIRECT Athiyandal , Tiruvannamalai", ("age", "[]")),)))
    assert Negative.text == "REDIRECT Athiyandal , Tiruvannamalai"
    assert Negative.populated() == frozenset()
    _, Malformed = Script.dispatch("PileNerTypeScript", (("entities",), ("garbage [",)))
    assert Malformed.text == ""
    assert Malformed.populated() == frozenset()
    _, Unprefixed = Script.dispatch("PileNerTypeScript", (("entities",), ([{"from": "human", "value": "no prefix"}],)))
    assert Unprefixed.text == ""


def test_the_pile_ner_type_script_extracts_relations_over_resolved_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    """the shared gazetteer runs over whitespace tokens; a compatible pair emits an edge"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="dexamethasone", category="SmallMolecule", curie="CHEBI:41180", preferred_name="dexamethasone", origin="fullmap"),
            ResolvedMention(mention="COPD", category="Disease", curie="MONDO:0005002", preferred_name="COPD", origin="fullmap"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Turns = conversations(
        "dexamethasone in the treatment of COPD .",
        ("chemical", '["dexamethasone"]'),
        ("medical condition", '["COPD"]'),
    )
    _, Example = Script.dispatch("PileNerTypeScript", (("entities",), (Turns,)))

    assert Example.relations == [expected_relation("treats", "dexamethasone", "COPD")]


def test_the_pile_ner_type_script_relation_gate_rejects_incompatible_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    """biolink domain/range: expressed_in needs a gene-ish head; a chemical head must not emit"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="benzene", category="ChemicalEntity", curie="CHEBI:167164", preferred_name="benzene", origin="fullmap"),
            ResolvedMention(mention="epithelial cells", category="Cell", curie="CL:0000066", preferred_name="epithelial cell", origin="fullmap"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Turns = conversations(
        "benzene expressed in epithelial cells .",
        ("chemical", '["benzene"]'),
        ("cell type", '["epithelial cells"]'),
    )
    _, Example = Script.dispatch("PileNerTypeScript", (("entities",), (Turns,)))

    assert Example.relations == []


def test_the_pile_ner_type_script_spans_every_occurrence_of_a_repeated_mention(monkeypatch: pytest.MonkeyPatch) -> None:
    """a surface answer maps to all of its occurrences, so the gazetteer can bracket the nearest one
    while the entity list still carries the mention once"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("Aspirin", "drug"), ("migraine", "medical condition")]
        return [
            ResolvedMention(mention="Aspirin", category="Drug", origin="fallback"),
            ResolvedMention(mention="migraine", category="Disease", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Turns = conversations(
        "Aspirin is cheap . Aspirin is used to treat migraine .",
        ("drug", '["Aspirin"]'),
        ("medical condition", '["migraine"]'),
    )
    _, Example = Script.dispatch("PileNerTypeScript", (("entities",), (Turns,)))

    assert {entity.label: entity.mentions for entity in Example.entities} == {"Drug": ["Aspirin"], "Disease": ["migraine"]}
    assert Example.relations == [expected_relation("treats", "Aspirin", "migraine")]


# ---------------------------------------------------------------------------
# char-span -> token-span bridge (knowledgator-biomed raw-text ingest)
# ---------------------------------------------------------------------------


def test_char_spans_to_token_spans_maps_exclusive_char_ends_to_inclusive_token_ends() -> None:
    """char ends are exclusive, token ends inclusive: end_char == a token's start_char excludes it,
    while end_char == a token's end_char keeps that token"""
    Triples: list[tuple[str, int, int]] = [("Aspirin", 0, 7), ("treats", 8, 14), ("migraine", 15, 23)]

    assert ScriptUtils.char_spans_to_token_spans(Triples, [(0, 7, "Drug")]) == [(0, 0, "Drug")]
    assert ScriptUtils.char_spans_to_token_spans(Triples, [(0, 8, "Drug")]) == [(0, 0, "Drug")]
    assert ScriptUtils.char_spans_to_token_spans(Triples, [(0, 14, "Drug")]) == [(0, 1, "Drug")]
    assert ScriptUtils.char_spans_to_token_spans(Triples, [(0, 15, "Drug")]) == [(0, 1, "Drug")]
    assert ScriptUtils.char_spans_to_token_spans(Triples, [(0, 23, "Drug")]) == [(0, 2, "Drug")]


def test_char_spans_to_token_spans_drops_out_of_bounds_spans() -> None:
    """OOB annotations are dataset defects (0.04-0.24% measured); negative starts and ends past the
    text extent drop while in-bounds survivors keep their input order"""
    Triples: list[tuple[str, int, int]] = [("Aspirin", 0, 7), ("treats", 8, 14), ("migraine", 15, 23)]

    assert ScriptUtils.char_spans_to_token_spans(Triples, [(-1, 5, "Drug"), (0, 7, "Drug"), (18, 30, "Disease"), (15, 23, "Disease")]) == [
        (0, 0, "Drug"),
        (2, 2, "Disease"),
    ]


def test_char_spans_to_token_spans_checks_bounds_before_normalizing_slop() -> None:
    """drop-then-snap order: the bounds check runs on the raw span first, so a negative start drops
    even though whitespace normalization would have clamped it back into the text"""
    Triples: list[tuple[str, int, int]] = [("Aspirin", 0, 7), ("treats", 8, 14)]

    assert ScriptUtils.char_spans_to_token_spans(Triples, [(-1, 2, "Drug")]) == []


def test_char_spans_to_token_spans_normalizes_whitespace_boundary_slop() -> None:
    """~0.4% of spans carry boundary slop; start advances past and end retreats across the
    whitespace runs the splitter leaves between tokens, and a whitespace-only span drops"""
    Triples: list[tuple[str, int, int]] = [("Aspirin", 0, 7), ("treats", 8, 14), ("migraine", 15, 23)]

    assert ScriptUtils.char_spans_to_token_spans(Triples, [(7, 14, "Drug")]) == [(1, 1, "Drug")]
    assert ScriptUtils.char_spans_to_token_spans(Triples, [(7, 15, "Drug")]) == [(1, 1, "Drug")]
    assert ScriptUtils.char_spans_to_token_spans(Triples, [(14, 15, "Drug")]) == []


def test_char_spans_to_token_spans_snaps_mid_token_boundaries_to_full_tokens() -> None:
    """3.43% of spans (351/10,237) land mid-token; the span widens to the full tokens it overlaps so
    surfaces always rejoin from whole tokens"""
    Triples: list[tuple[str, int, int]] = [("Aspirin", 0, 7), ("treats", 8, 14), ("migraine", 15, 23)]

    assert ScriptUtils.char_spans_to_token_spans(Triples, [(2, 5, "Drug")]) == [(0, 0, "Drug")]
    assert ScriptUtils.char_spans_to_token_spans(Triples, [(16, 21, "Disease")]) == [(2, 2, "Disease")]
    assert ScriptUtils.char_spans_to_token_spans(Triples, [(5, 16, "Drug")]) == [(0, 2, "Drug")]


def test_char_spans_to_token_spans_drops_degenerate_spans() -> None:
    """zero-length (start == end, even mid-token) and inverted (end < start) spans drop at the
    degenerate check after normalization"""
    Triples: list[tuple[str, int, int]] = [("Aspirin", 0, 7), ("treats", 8, 14), ("migraine", 15, 23)]

    assert ScriptUtils.char_spans_to_token_spans(Triples, [(3, 3, "Drug")]) == []
    assert ScriptUtils.char_spans_to_token_spans(Triples, [(20, 5, "Drug")]) == []


def test_char_spans_to_token_spans_returns_empty_for_empty_char_spans() -> None:
    Triples: list[tuple[str, int, int]] = [("Aspirin", 0, 7), ("treats", 8, 14), ("migraine", 15, 23)]

    assert ScriptUtils.char_spans_to_token_spans(Triples, []) == []


def test_char_spans_to_token_spans_returns_empty_for_empty_triples() -> None:
    """empty text: the extent derives to 0 and no token can overlap, so every span drops"""
    assert ScriptUtils.char_spans_to_token_spans([], [(0, 1, "Drug")]) == []


def test_char_spans_to_token_spans_preserves_order_without_dedup_or_merge() -> None:
    """the dataset is measured overlap-free (0/18,685) so the bridge carries no overlap policy:
    duplicates and overlaps pass through in input order and downstream mention grouping absorbs them"""
    Triples: list[tuple[str, int, int]] = [("Aspirin", 0, 7), ("treats", 8, 14), ("migraine", 15, 23)]
    Spans: list[tuple[int, int, str]] = [(15, 23, "Disease"), (0, 7, "Drug"), (15, 23, "Disease"), (0, 14, "Drug")]

    assert ScriptUtils.char_spans_to_token_spans(Triples, Spans) == [(2, 2, "Disease"), (0, 0, "Drug"), (2, 2, "Disease"), (0, 1, "Drug")]


# ---------------------------------------------------------------------------
# KnowledgatorBiomedScript + char-offset entity structs (knowledgator/biomed_NER ingest)
# ---------------------------------------------------------------------------


def test_the_knowledgator_script_bridges_char_spans_through_the_real_splitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """end-to-end happy path: raw text tokenized by the real gliner2 splitter (pure python, offline),
    char offsets bridged to token spans, resolution via the dataset label_map, grouping, and a
    gazetteer treats relation when a trigger phrase sits between the two mentions"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("Aspirin", "CHEMICALS"), ("migraine", "DISORDERS")]
        return [
            ResolvedMention(mention="Aspirin", category="Drug", curie="CHEBI:15365", preferred_name="Acetylsalicylic acid", origin="fullmap"),
            ResolvedMention(mention="migraine", category="Disease", curie="MONDO:0005002", preferred_name="migraine disorder", origin="fullmap"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Text = "Aspirin is used to treat migraine"
    Entities = [
        {"start": Text.index("Aspirin"), "end": Text.index("Aspirin") + len("Aspirin"), "class": "CHEMICALS"},
        {"start": Text.index("migraine"), "end": Text.index("migraine") + len("migraine"), "class": "DISORDERS"},
    ]
    _, Example = Script.dispatch("KnowledgatorBiomedScript", (("entities",), (Text, Entities)))

    Tokens: list[str] = FullmapMiner.tokenize(Text)
    assert Example.text == ScriptUtils.join_tokens(Tokens)
    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "Drug": ["Aspirin"],
        "Disease": ["migraine"],
    }
    assert Example.entities[0].description is not None and "[fullmap: CHEBI:15365 | Acetylsalicylic acid]" in Example.entities[0].description
    assert Example.relations == [expected_relation("treats", "Aspirin", "migraine")]
    assert Example.populated() == frozenset({"entities", "relations"})


def test_the_knowledgator_script_rejoins_tokens_not_raw_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """the splitter detaches punctuation (26.7% of char-slice surfaces are absent from raw text), so
    TrainingExample.text must be the whitespace-joined token stream -- here the trailing comma
    detaches and the surface stays a substring of the emitted text"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        return [ResolvedMention(mention="seeds", category="Gene", origin="fallback")]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Text = "seeds, the plant's gene products, accumulate oil"
    Entities = [{"start": 0, "end": 5, "class": "GENES"}]
    _, Example = Script.dispatch("KnowledgatorBiomedScript", (("entities",), (Text, Entities)))

    assert Example.text.startswith("seeds ") and "," in Example.text
    assert {entity.label: entity.mentions for entity in Example.entities} == {"Gene": ["seeds"]}


def test_the_knowledgator_script_skips_malformed_text_and_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    """skip-don't-coerce: non-str text yields an empty example, a non-list entities column yields a
    text-only row, and malformed entries (not dicts, missing or mistyped start/end/class) drop while
    surviving entries keep their alignment -- mirrors mention_spans' defense"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("Aspirin", "CHEMICALS")]
        return [ResolvedMention(mention="Aspirin", category="ChemicalEntity", origin="fallback")]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))

    _, Blank = Script.dispatch("KnowledgatorBiomedScript", (("entities",), ("", [])))
    assert Blank.text == ""
    assert Blank.populated() == frozenset()
    _, NotText = Script.dispatch("KnowledgatorBiomedScript", (("entities",), (None, [{"start": 0, "end": 7, "class": "CHEMICALS"}])))
    assert NotText.text == ""
    assert NotText.populated() == frozenset()
    _, NotList = Script.dispatch("KnowledgatorBiomedScript", (("entities",), ("Aspirin helps.", "not a list")))
    assert NotList.text == ""
    assert NotList.populated() == frozenset()
    _, EmptyList = Script.dispatch("KnowledgatorBiomedScript", (("entities",), ("Aspirin helps.", [])))
    assert EmptyList.text == ""
    assert EmptyList.populated() == frozenset()

    Text = "Aspirin helps."
    Malformed: list[Any] = [
        {"start": 0, "end": 7, "class": "CHEMICALS"},  # the survivor
        "not a dict",
        None,
        3,
        [],
        {"start": 0, "end": 7},
        {"start": 0, "class": "CHEMICALS"},
        {"end": 7, "class": "CHEMICALS"},
        {"start": True, "end": 7, "class": "CHEMICALS"},
        {"start": 0, "end": 7.0, "class": "CHEMICALS"},
        {"start": "0", "end": 7, "class": "CHEMICALS"},
        {"start": 0, "end": 7, "class": None},
        {"start": 0, "end": 7, "class": 3},
    ]
    _, Example = Script.dispatch("KnowledgatorBiomedScript", (("entities",), (Text, Malformed)))

    assert Example.text == ScriptUtils.join_tokens(FullmapMiner.tokenize(Text))
    assert {entity.label: entity.mentions for entity in Example.entities} == {"ChemicalEntity": ["Aspirin"]}


def test_the_knowledgator_script_emits_text_only_row_when_no_spans_survive() -> None:
    """all spans dropped (here: end overshoots the text extent, the measured OOB defect) leaves the
    row text-only, matching the sibling scripts' empty-out contract"""
    Text = "Aspirin helps."
    _, Example = Script.dispatch("KnowledgatorBiomedScript", (("entities",), (Text, [{"start": 0, "end": 10_000, "class": "CHEMICALS"}])))

    assert Example.text == ScriptUtils.join_tokens(FullmapMiner.tokenize(Text))
    assert Example.populated() == frozenset()


def test_the_knowledgator_script_pascalcases_unmapped_labels_as_raw_tails(monkeypatch: pytest.MonkeyPatch) -> None:
    """classes with no honest biolink target (LANGUAGE, REGULATION OR LAW) stay unmapped and surface
    as PascalCased raw tails, the Pile-NER zero-shot convention"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="English", category="LANGUAGE", origin="raw"),
            ResolvedMention(mention="HIPAA", category="REGULATION OR LAW", origin="raw"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Text = "English privacy law HIPAA"
    Entities = [
        {"start": Text.index("English"), "end": Text.index("English") + len("English"), "class": "LANGUAGE"},
        {"start": Text.index("HIPAA"), "end": Text.index("HIPAA") + len("HIPAA"), "class": "REGULATION OR LAW"},
    ]
    _, Example = Script.dispatch("KnowledgatorBiomedScript", (("entities",), (Text, Entities)))

    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "Language": ["English"],
        "RegulationOrLaw": ["HIPAA"],
    }


def test_the_knowledgator_script_resolves_plural_variants_through_the_label_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """the data's 35 distinct raw class strings include plural/legacy variants the HF card never
    documents; the run() call must hand resolve_mentions the dataset map so PRODUCTS/ORGANISMS resolve"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert label_map is KnowledgatorBiomedScript.LABEL_MAP
        return [
            ResolvedMention(mention="ventilator", category=label_map["products"], origin="fallback"),
            ResolvedMention(mention="mice", category=label_map["organisms"], origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Text = "ventilator tested in mice"
    Entities = [
        {"start": 0, "end": 10, "class": "PRODUCTS"},
        {"start": 18, "end": 22, "class": "ORGANISMS"},
    ]
    _, Example = Script.dispatch("KnowledgatorBiomedScript", (("entities",), (Text, Entities)))

    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "Device": ["ventilator"],
        "OrganismTaxon": ["mice"],
    }


def test_every_knowledgator_label_map_value_is_a_biolink_category() -> None:
    """dataset-local vocabulary values stay real biolink classes (import-time validate_label_map
    already raises; this pins the invariant for the whole 29-entry map)"""
    for raw_label, category in KnowledgatorBiomedScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"fallback {raw_label!r} -> {category!r} is not a biolink class"


def test_every_knowledgator_label_map_entry_is_measured_vocabulary() -> None:
    """the map is exactly the measured vocabulary: the 21 canonical classes that have an honest biolink
    target (LANGUAGE, REGULATION OR LAW and MONEY have none) plus the 8 plural/legacy variants counted
    on the full 4,840-row train split (GENES, LOCATIONS, ORGANISMS, ORGANIZATIONS, PRODUCTS,
    FINDINGS/PHENOTYPES, DISORDERS, EVENTS); the catch-alls stay deliberately absent so they fall
    through to raw PascalCase tails"""
    assert sorted(KnowledgatorBiomedScript.LABEL_MAP) == [
        "activity",
        "anatomical structure",
        "body substance",
        "cells and their components",
        "chemicals",
        "clinical drug",
        "disorder",
        "disorders",
        "event",
        "events",
        "findings/phenotypes",
        "function",
        "gene and gene products",
        "genes",
        "geographical areas",
        "group",
        "intellectual property",
        "location",
        "locations",
        "medical procedure",
        "organism",
        "organisms",
        "organization",
        "organizations",
        "person",
        "phenotype",
        "product",
        "products",
        "signaling molecules",
    ]
    assert len(KnowledgatorBiomedScript.LABEL_MAP) == 29
    for raw_label in ("language", "regulation or law", "money", "unlabelled", "intellectual"):
        assert raw_label not in KnowledgatorBiomedScript.LABEL_MAP


def test_the_measured_whitespace_variant_needs_no_map_entry() -> None:
    """the full train split carries 'ORGANISMS ' with a trailing space (253 spans) alongside the clean
    'ORGANISMS' (502). It resolves without its own entry because the shared fallback lookup retries
    through normalize_iob_label, which strips and case-folds -- pinning that here is what keeps the map
    free of whitespace-padded keys, and EVENTS is the contrast: a variant with no normalizing rule that
    saves it, so it does need an entry (23 spans would otherwise ship as a raw 'Events' tail)."""
    LabelMap: dict[str, str] = KnowledgatorBiomedScript.LABEL_MAP

    assert "organisms " not in LabelMap
    assert LabelMap.get(ScriptUtils.normalize_iob_label("ORGANISMS ")) == "OrganismTaxon"
    assert ScriptUtils.normalize_iob_label("EVENTS") == "events"
    assert LabelMap["events"] == "Event"


# ---------------------------------------------------------------------------
# US-004 verbatim row-0 offline smoke: the real dataset row embedded VERBATIM (no file IO, no
# network) locks the whole chain -- dispatch -> tokenize -> bridge -> resolve -> group -- against
# format drift of the real dataset
# ---------------------------------------------------------------------------

ROW0_TEXT: str = (
    "Weed seed inactivation in soil mesocosms via biosolarization with mature compost and tomato processing waste amendments Biosolariz"
    "ation is a fumigation alternative that combines passive solar heating with amendment-driven soil microbial activity to temporarily"
    " create antagonistic soil conditions, such as elevated temperature and acidity, that can inactivate weed seeds and other pest prop"
    "agules. The aim of this study was to use a mesocosm -based field trial to assess soil heating, pH, volatile fatty acid accumulatio"
    "n and weed seed inactivation during biosolarization. Biosolarization for 8 days using 2% mature green waste compost and 2 or 5% to"
    "mato processing residues in the soil resulted in accumulation of volatile fatty acids in the soil, particularly acetic acid, and> "
    "95% inactivation of Brassica nigra and Solanum nigrum seeds. Inactivation kinetics data showed that near complete weed seed inacti"
    "vation in soil was achieved within the first 5 days of biosolarization. This was significantly greater than the inactivation achie"
    "ved in control soils that were solar heated without amendment or were amended but not solar heated. The composition and concentrat"
    "ion of organic matter amendments in soil significantly affected volatile fatty acid accumulation at various soil depths during bio"
    "solarization. Combining solar heating with organic matter amendment resulted in accelerated weed seed inactivation compared with e"
    "ither approach alone. © 2016 Society of Chemical Industry. "
)

ROW0_ENTITIES_JSON: str = """[
 {
  "start": 0,
  "end": 4,
  "class": "ORGANISM"
 },
 {
  "start": 5,
  "end": 9,
  "class": "ORGANISM"
 },
 {
  "start": 26,
  "end": 30,
  "class": "CHEMICALS"
 },
 {
  "start": 31,
  "end": 40,
  "class": "LOCATION"
 },
 {
  "start": 45,
  "end": 60,
  "class": "ACTIVITY"
 },
 {
  "start": 66,
  "end": 80,
  "class": "CHEMICALS"
 },
 {
  "start": 85,
  "end": 91,
  "class": "ORGANISM"
 },
 {
  "start": 103,
  "end": 119,
  "class": "CHEMICALS"
 },
 {
  "start": 120,
  "end": 135,
  "class": "ACTIVITY"
 },
 {
  "start": 141,
  "end": 151,
  "class": "ACTIVITY"
 },
 {
  "start": 222,
  "end": 226,
  "class": "CHEMICALS"
 },
 {
  "start": 227,
  "end": 245,
  "class": "FUNCTION"
 },
 {
  "start": 281,
  "end": 285,
  "class": "CHEMICALS"
 },
 {
  "start": 360,
  "end": 364,
  "class": "ORGANISM"
 },
 {
  "start": 365,
  "end": 370,
  "class": "ORGANISM"
 },
 {
  "start": 433,
  "end": 441,
  "class": "LOCATION"
 },
 {
  "start": 449,
  "end": 454,
  "class": "LOCATION"
 },
 {
  "start": 471,
  "end": 475,
  "class": "CHEMICALS"
 },
 {
  "start": 487,
  "end": 488,
  "class": "CHEMICALS"
 },
 {
  "start": 489,
  "end": 508,
  "class": "CHEMICALS"
 },
 {
  "start": 526,
  "end": 530,
  "class": "ORGANISM"
 },
 {
  "start": 531,
  "end": 535,
  "class": "ORGANISM"
 },
 {
  "start": 556,
  "end": 572,
  "class": "ACTIVITY"
 },
 {
  "start": 573,
  "end": 588,
  "class": "ACTIVITY"
 },
 {
  "start": 622,
  "end": 635,
  "class": "CHEMICALS"
 },
 {
  "start": 648,
  "end": 654,
  "class": "ORGANISM"
 },
 {
  "start": 666,
  "end": 674,
  "class": "CHEMICALS"
 },
 {
  "start": 682,
  "end": 686,
  "class": "CHEMICALS"
 },
 {
  "start": 715,
  "end": 735,
  "class": "CHEMICALS"
 },
 {
  "start": 743,
  "end": 748,
  "class": "CHEMICALS"
 },
 {
  "start": 762,
  "end": 774,
  "class": "CHEMICALS"
 },
 {
  "start": 800,
  "end": 814,
  "class": "ORGANISM"
 },
 {
  "start": 819,
  "end": 833,
  "class": "ORGANISM"
 },
 {
  "start": 834,
  "end": 840,
  "class": "ORGANISM"
 },
 {
  "start": 863,
  "end": 867,
  "class": "INTELLECTUAL PROPERTY"
 },
 {
  "start": 894,
  "end": 898,
  "class": "ORGANISM"
 },
 {
  "start": 899,
  "end": 903,
  "class": "ORGANISM"
 },
 {
  "start": 920,
  "end": 924,
  "class": "CHEMICALS"
 },
 {
  "start": 965,
  "end": 981,
  "class": "ACTIVITY"
 },
 {
  "start": 1055,
  "end": 1060,
  "class": "CHEMICALS"
 },
 {
  "start": 1177,
  "end": 1202,
  "class": "CHEMICALS"
 },
 {
  "start": 1206,
  "end": 1210,
  "class": "CHEMICALS"
 },
 {
  "start": 1234,
  "end": 1253,
  "class": "CHEMICALS"
 },
 {
  "start": 1278,
  "end": 1282,
  "class": "CHEMICALS"
 },
 {
  "start": 1297,
  "end": 1313,
  "class": "ACTIVITY"
 },
 {
  "start": 1343,
  "end": 1357,
  "class": "CHEMICALS"
 },
 {
  "start": 1358,
  "end": 1367,
  "class": "ACTIVITY"
 },
 {
  "start": 1392,
  "end": 1396,
  "class": "ORGANISM"
 },
 {
  "start": 1397,
  "end": 1401,
  "class": "ORGANISM"
 },
 {
  "start": 1459,
  "end": 1488,
  "class": "ORGANIZATION"
 }
]
"""

ROW0_ENTITIES: list[Any] = json.loads(ROW0_ENTITIES_JSON)


def test_the_knowledgator_script_survives_verbatim_row0_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """US-004 smoke: the verbatim real row locks the whole chain (dispatch -> tokenize -> bridge ->
    resolve -> group) against format drift of the real dataset; the fixtures are embedded constants
    and resolve_mentions is faked, so the test runs with no file IO, no fullmap mount, no network"""

    def fake_resolve(spans: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert label_map is KnowledgatorBiomedScript.LABEL_MAP
        assert len(spans) == 50 == len(ROW0_ENTITIES)  # every char span of row 0 survives the bridge
        return [
            ResolvedMention(mention=mention, category="OrganismTaxon", origin="fallback")
            if mention == "Weed"
            else ResolvedMention(mention=mention, category=raw_label, origin="raw")
            for mention, raw_label in spans
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    _, Example = Script.dispatch("KnowledgatorBiomedScript", (("entities",), (ROW0_TEXT, ROW0_ENTITIES)))

    Tokens: list[str] = [token for token, _start, _end in FullmapMiner.splitter()(ROW0_TEXT, lower=False)]
    assert Example.text == ScriptUtils.join_tokens(Tokens)
    by_label = {entity.label: entity for entity in Example.entities}
    assert "Weed" in by_label["OrganismTaxon"].mentions  # ORGANISM -> OrganismTaxon via the label map
    assert {"entities"} <= Example.populated() <= {"entities", "relations"}


def expected_relation(name: str, head: str, tail: str) -> Relation:
    """emitted relations carry their biolink slot description; build the matching expectation"""
    return Relation(
        name=name,
        fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
        description=ScriptUtils.predicate_description(name),
    )


# ---------------------------------------------------------------------------
# PubmedAbstractsScript (PubMedAbstractsNER ingest)
# ---------------------------------------------------------------------------


def test_every_pubmed_heading_maps_to_a_biolink_category() -> None:
    """dataset-local MeSH-heading vocabulary values stay real biolink classes"""
    from relmedner.scripts import PubmedAbstractsScript

    for heading, category in PubmedAbstractsScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"fallback {heading!r} -> {category!r} is not a biolink class"


def test_pubmed_seed_anchors_are_present() -> None:
    """the measured raw-origin anchors the spec pins; only clearly-faithful headings are seeded
    in this story (force-fitting 'Diagnosis'/'Population Characteristics'/'Blood' etc. to the
    nearest-sounding class would silently mislabel training data -- coverage push is US-004)"""
    from relmedner.scripts import PubmedAbstractsScript

    assert PubmedAbstractsScript.LABEL_MAP["pathologic processes"] == "PathologicalProcess"
    assert PubmedAbstractsScript.LABEL_MAP["persons"] == "Human"
    assert PubmedAbstractsScript.LABEL_MAP["publication formats"] == "Publication"
    assert PubmedAbstractsScript.LABEL_MAP["age groups"] == "PopulationOfIndividualOrganisms"


def test_the_pubmed_script_splits_headings_before_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """the definition tail must never reach resolve_mentions: the shared gate keys buckets on
    label words, so definition words ('region', 'leg', 'process', 'measure') would flip fullmap
    outcomes (measured live: 1,782/19,972 sample spans, 10.6%, lost to false rejections, e.g.
    'Lower Extremity - ... the BUTTOCKS; HIP; and LEG.' rejecting a correct Disease hit for
    'ankle') -- the script is pinned to pass bare headings only, with the dataset map merged"""
    from relmedner.scripts import PubmedAbstractsScript

    received: list[list[tuple[str, str]]] = []

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        received.append(list(mentions))
        assert label_map is PubmedAbstractsScript.LABEL_MAP
        return [ResolvedMention(mention=mention, category="Disease", origin="fallback") for mention, _ in mentions]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens: list[str] = ["Ankle", "sprains", "are", "common", "in", "women", "athletes", "."]
    Definition = " - The region of the lower limb, including the BUTTOCKS; HIP; and LEG."
    Ner: list[list[Any]] = [[0, 1, "Pathologic Processes" + Definition], [5, 5, "Persons" + Definition]]
    _, Example = Script.dispatch("PubmedAbstractsScript", (("entities",), (Tokens, Ner)))

    assert received == [[("Ankle sprains", "Pathologic Processes"), ("women", "Persons")]]
    assert Example.populated() == frozenset({"entities"})


def test_pubmed_definition_words_do_not_flip_the_resolution_gate() -> None:
    """the gate half of the split hazard, pinned directly so it runs without the fullmap db:
    the full 'heading - definition' label fires the anatomy bucket on the definition words
    'region'/'leg' and rejects a correct Disease-resolved mention that the bare heading accepts
    (this is the measured 10.6% false-rejection loss the heading split removes)"""
    Full = "Lower Extremity - The region of the lower limb, including the BUTTOCKS; HIP; and LEG."

    assert ResolutionGate.accept("ankle", Full, "MONDO:0001234", "Disease") is False
    assert ResolutionGate.accept("ankle", "Lower Extremity", "MONDO:0001234", "Disease") is True


def test_the_pubmed_script_emits_nothing_for_rows_without_surviving_spans() -> None:
    """the 13 measured empty-ner rows (of 35,000) exit as text-only examples whose empty
    populated() pipeline.matches_declared_outputs drops; a row whose every span fails the
    mention_spans shape/bounds guard (0 in this corpus, guard stays) is text-only too"""
    _, EmptyNer = Script.dispatch("PubmedAbstractsScript", (("entities",), (["Breast", "anatomy", "review", "."], [])))
    assert EmptyNer.text == "Breast anatomy review ."
    assert EmptyNer.populated() == frozenset()
    _, AllMalformed = Script.dispatch(
        "PubmedAbstractsScript",
        (("entities",), (["Breast", "anatomy"], [[9, 9, "Cells - the basic structural unit"], [-1, 0, "Persons"], [1, 0, "Disease"]])),
    )
    assert AllMalformed.populated() == frozenset()


def test_the_pubmed_script_decodes_real_row_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """end-to-end over the corpus's real shape: end-inclusive spans, 'heading - definition'
    labels split to bare headings, and mixed fullmap/fallback/raw origins grouped by category"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [
            ("Malaria", "Pathologic Processes"),
            ("patients", "Persons"),
            ("questionnaires", "Investigative Techniques"),
        ]
        return [
            ResolvedMention(mention="malaria", category="Disease", curie="MONDO:0005388", preferred_name="malaria", origin="fullmap"),
            ResolvedMention(mention="patients", category="Human", origin="fallback"),
            ResolvedMention(mention="questionnaires", category="Investigative Techniques", origin="raw"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens: list[str] = ["Malaria", "is", "endemic", "among", "patients", "surveyed", "with", "questionnaires", "."]
    Ner: list[list[Any]] = [
        [0, 0, "Pathologic Processes - Parasitic diseases characterized by fever and chills."],
        [4, 4, "Persons - Individuals grouped by age, sex, or occupation."],
        [7, 7, "Investigative Techniques - Procedures and methods used in research."],
    ]
    _, Example = Script.dispatch("PubmedAbstractsScript", (("entities",), (Tokens, Ner)))

    assert Example.text == "Malaria is endemic among patients surveyed with questionnaires ."
    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "Disease": ["malaria"],
        "Human": ["patients"],
        "InvestigativeTechniques": ["questionnaires"],
    }
    assert Example.entities[0].description is not None and "[fullmap: MONDO:0005388 | malaria]" in Example.entities[0].description


def test_the_pubmed_script_pascalcases_raw_headings_but_keeps_mapped_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    """unmapped MeSH headings surface biolink-cased raw categories exactly like
    PileNerBiomedScript's raw tail ('Abdominal Core' -> 'AbdominalCore'); LABEL_MAP/fallback
    hits already name a biolink class and must stay untouched"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("Adults", "Age Groups"), ("abdominal core", "Abdominal Core")]
        return [
            ResolvedMention(mention="adults", category="PopulationOfIndividualOrganisms", origin="fallback"),
            ResolvedMention(mention="abdominal core", category="Abdominal Core", origin="raw"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens: list[str] = ["Adults", "show", "increased", "activity", "in", "the", "abdominal", "core", "."]
    Ner: list[list[Any]] = [[0, 0, "Age Groups - Age classifications of humans."], [6, 7, "Abdominal Core - The central abdominal region."]]
    _, Example = Script.dispatch("PubmedAbstractsScript", (("entities",), (Tokens, Ner)))

    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "PopulationOfIndividualOrganisms": ["adults"],
        "AbdominalCore": ["abdominal core"],
    }


def test_the_pubmed_script_emits_gazetteer_relations_over_resolved_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    """a trigger phrase between two resolved mentions emits an edge; the ingest declares
    outputs: [entities, relations] (US-003) so both shapes must come off one row (measured
    on a 5,000-row sample through the production run() path with the expanded map: 18.4% of
    rows carry >=1 relation, 1,107 relations across 22 distinct predicates)"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="Aspirin", category="Drug", origin="fallback"),
            ResolvedMention(mention="migraine", category="Disease", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens: list[str] = ["Aspirin", "is", "used", "to", "treat", "migraine"]
    Ner: list[list[Any]] = [
        [0, 0, "Organic Chemical - a carbon-based substance with medicinal use."],
        [5, 5, "Disease - a disorder of structure or function."],
    ]
    _, Example = Script.dispatch("PubmedAbstractsScript", (("entities",), (Tokens, Ner)))

    assert Example.relations == [expected_relation("treats", "Aspirin", "migraine")]


def test_pubmed_coverage_push_pins_the_measured_raw_headings() -> None:
    """US-004 raw-origin coverage rule: with the US-002 seed map 51,684 of 383,721 full-corpus
    spans land in raw origin over 1,637 headings, and every heading with >=30 raw-origin spans
    whose actual mention surfaces fit one real biolink class is mapped (measured coverage 44.2%
    after the US-004 expansion: the residual raw
    tail is dominated by headings with NO faithful class -- 'Investigative Techniques' is 94%
    the surfaces 'methods'/'METHODS', 'Group Processes' is 99.7% 'role' with no biolink Role
    class, 'Chemical Phenomena'/'Genetic Phenomena' are mixed-category -- which stay unmapped
    because a wrong bucket silently mislabels training data). Pins the top measured raw
    headings to their chosen classes AND the monsters' absence, so a future edit can neither
    silently gut the coverage rule nor force-fit an unfaithful mapping; pure-data, no DB
    (import-time validate_label_map separately pins every value to a real biolink class)"""
    from relmedner.scripts import PubmedAbstractsScript

    label_map = PubmedAbstractsScript.LABEL_MAP
    assert label_map["metabolism"] == "BiologicalProcess"
    assert label_map["genetic variation"] == "SequenceVariant"
    assert label_map["food and beverages"] == "Food"
    assert label_map["vertebrates"] == "Vertebrate"
    assert label_map["neoplasms, glandular and epithelial"] == "Disease"
    assert label_map["body temperature changes"] == "PhenotypicFeature"
    assert label_map["white people"] == "PopulationOfIndividualOrganisms"
    assert label_map["drug resistance"] == "PhenotypicFeature"
    assert label_map["brain"] == "AnatomicalEntity"
    assert label_map["epithelial cells"] == "CellLine"
    assert label_map["dna repair"] == "BiologicalProcess"
    assert label_map["butyrates"] == "ChemicalEntity"
    assert len(label_map) == 211
    for unmapped in ("investigative techniques", "group processes", "chemical phenomena", "genetic phenomena"):
        assert unmapped not in label_map, f"{unmapped!r} has no faithful class and must stay unmapped"


# NemotronPiiScript + the PII span decode policy (nvidia/Nemotron-PII ingest)
# ---------------------------------------------------------------------------


def test_the_nemotron_pii_parse_literal_spans_decodes_python_repr_strings_and_keeps_parsed_lists() -> None:
    """datasets-server serves the spans column as python-repr strings, not arrays; an
    already-parsed list (local/avro shape) must decode to the same thing"""
    from relmedner.scripts.nemotron_pii import parse_literal_spans

    Parsed: list[dict[str, Any]] = [{"start": 0, "end": 4, "text": "Mara", "label": "first_name"}]
    assert parse_literal_spans("[{'start': 0, 'end': 4, 'text': 'Mara', 'label': 'first_name'}]") == Parsed
    assert parse_literal_spans(Parsed) == Parsed
    assert parse_literal_spans("[]") == []


def test_the_nemotron_pii_parse_literal_spans_skips_values_that_are_not_lists_of_dicts() -> None:
    """skip-don't-coerce: a malformed external column must never crash a row or become silently coerced spans"""
    from relmedner.scripts.nemotron_pii import parse_literal_spans

    assert parse_literal_spans("garbage [") == []
    assert parse_literal_spans("'just a string'") == []
    assert parse_literal_spans("['a', 'b']") == []
    # a non-dict entry drops individually; the dict-shaped spans on the same row must survive
    assert parse_literal_spans("[{'start': 0}, 3]") == [{"start": 0}]
    assert parse_literal_spans(None) == []
    assert parse_literal_spans(3) == []


def test_the_nemotron_pii_script_yields_entities_on_structured_markdown_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """real-shaped structured fixture (markdown form, spans as a python-repr string): a nonzero
    populated yield proves the decode -> slice -> resolve -> group chain works end to end"""
    from relmedner.scripts import NemotronPiiScript

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert label_map is NemotronPiiScript.LABEL_MAP
        assert mentions == [("Mara", "first_name"), ("Voss", "last_name"), ("Duluth", "city")]
        return [
            ResolvedMention(mention="Mara", category="Human", origin="fallback"),
            ResolvedMention(mention="Voss", category="Human", origin="fallback"),
            ResolvedMention(mention="Duluth", category="GeographicLocation", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Text = "**Loan Application**\nFirst Name: Mara\nLast Name: Voss\nCity: Duluth\nCountry: USA"
    Spans = (
        "[{'start': 33, 'end': 37, 'text': 'Mara', 'label': 'first_name'}, "
        "{'start': 49, 'end': 53, 'text': 'Voss', 'label': 'last_name'}, "
        "{'start': 60, 'end': 66, 'text': 'Duluth', 'label': 'city'}]"
    )
    _, Example = Script.dispatch("NemotronPiiScript", (("entities",), (Text, Spans)))

    assert Example.text == Text
    assert Example.populated() == frozenset({"entities"})
    assert {entity.label: entity.mentions for entity in Example.entities} == {"Human": ["Mara", "Voss"], "GeographicLocation": ["Duluth"]}


def test_the_nemotron_pii_script_yields_entities_on_unstructured_prose_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """the second document_format must decode identically (no format-specific branch exists), so a
    prose row yields through the same path as the form-shaped one"""
    from relmedner.scripts import NemotronPiiScript

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert label_map is NemotronPiiScript.LABEL_MAP
        assert mentions == [("Mara", "first_name"), ("Duluth", "city"), ("full-time", "employment_status"), ("machinist", "occupation")]
        return [
            ResolvedMention(mention="Mara", category="Human", origin="fallback"),
            ResolvedMention(mention="Duluth", category="GeographicLocation", origin="fallback"),
            ResolvedMention(mention="full-time", category="SocioeconomicAttribute", origin="fallback"),
            ResolvedMention(mention="machinist", category="SocioeconomicAttribute", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Text = "Mara Voss lives in Duluth and works full-time as a machinist."
    Spans = (
        "[{'start': 0, 'end': 4, 'text': 'Mara', 'label': 'first_name'}, "
        "{'start': 19, 'end': 25, 'text': 'Duluth', 'label': 'city'}, "
        "{'start': 36, 'end': 45, 'text': 'full-time', 'label': 'employment_status'}, "
        "{'start': 51, 'end': 60, 'text': 'machinist', 'label': 'occupation'}]"
    )
    _, Example = Script.dispatch("NemotronPiiScript", (("entities",), (Text, Spans)))

    assert Example.populated() == frozenset({"entities"})
    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "Human": ["Mara"],
        "GeographicLocation": ["Duluth"],
        "SocioeconomicAttribute": ["full-time", "machinist"],
    }


def test_the_nemotron_pii_script_accepts_already_parsed_span_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    """an already-parsed list of span dicts (non-datasets-server shape) must decode without the string path"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("Bo", "first_name"), ("Chen", "last_name"), ("12 Oak St", "street_address"), ("55501", "postcode")]
        return [
            ResolvedMention(mention="Bo", category="Human", origin="fallback"),
            ResolvedMention(mention="Chen", category="Human", origin="fallback"),
            ResolvedMention(mention="12 Oak St", category="GeographicLocation", origin="fallback"),
            ResolvedMention(mention="55501", category="GeographicLocation", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Text = "Contact Bo Chen at 12 Oak St, 55501."
    Spans: list[dict[str, Any]] = [
        {"start": 8, "end": 10, "text": "Bo", "label": "first_name"},
        {"start": 11, "end": 15, "text": "Chen", "label": "last_name"},
        {"start": 19, "end": 28, "text": "12 Oak St", "label": "street_address"},
        {"start": 30, "end": 35, "text": "55501", "label": "postcode"},
    ]
    _, Example = Script.dispatch("NemotronPiiScript", (("entities",), (Text, Spans)))

    assert Example.populated() == frozenset({"entities"})
    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "Human": ["Bo", "Chen"],
        "GeographicLocation": ["12 Oak St", "55501"],
    }


def test_the_nemotron_pii_script_slices_surfaces_over_quirked_span_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """measured quirks: span dicts carry int-typed text (age/cvv) and case-drifted values ('spanish'
    against 'Spanish'); decoding must survive both and the surface must be the text slice, never the field"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("44", "age"), ("Spanish", "race_ethnicity"), ("spanish", "language"), ("441", "cvv")]
        return [ResolvedMention(mention=surface, category=label, origin="raw") for surface, label in mentions]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Text = "As a 44 years old Spanish national, she listed language spanish and card cvv 441."
    Spans = (
        "[{'start': 5, 'end': 7, 'text': 44, 'label': 'age'}, "
        "{'start': 18, 'end': 25, 'text': 'spanish', 'label': 'race_ethnicity'}, "
        "{'start': 56, 'end': 63, 'text': 'Spanish', 'label': 'language'}, "
        "{'start': 77, 'end': 80, 'text': 441, 'label': 'cvv'}]"
    )
    _, Example = Script.dispatch("NemotronPiiScript", (("entities",), (Text, Spans)))

    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "Age": ["44"],
        "RaceEthnicity": ["Spanish"],
        "Language": ["spanish"],
        "Cvv": ["441"],
    }


def test_the_nemotron_pii_script_drops_malformed_spans_without_killing_the_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """every malformed span variant measured in the wild (bool start, string end, negative start,
    start==end, end past the text, missing/empty label, non-dict entries) drops individually while
    good spans on the same row still ship"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("Bo", "first_name"), ("12 Oak St", "street_address")]
        return [
            ResolvedMention(mention="Bo", category="Human", origin="fallback"),
            ResolvedMention(mention="12 Oak St", category="GeographicLocation", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Text = "Contact Bo Chen at 12 Oak St, 55501."
    Spans: list[Any] = [
        {"start": 8, "end": 10, "text": "Bo", "label": "first_name"},
        {"start": True, "end": 15, "label": "last_name"},
        {"start": 11, "end": "15", "label": "last_name"},
        {"start": -1, "end": 3, "label": "city"},
        {"start": 30, "end": 30, "label": "postcode"},
        {"start": 30, "end": 99, "label": "postcode"},
        {"start": 30, "end": 35},
        {"start": 30, "end": 35, "label": ""},
        "not-a-dict",
        7,
        None,
        {"start": 19, "end": 28, "text": "12 Oak St", "label": "street_address"},
    ]
    _, Example = Script.dispatch("NemotronPiiScript", (("entities",), (Text, Spans)))

    assert Example.populated() == frozenset({"entities"})
    assert {entity.label: entity.mentions for entity in Example.entities} == {"Human": ["Bo"], "GeographicLocation": ["12 Oak St"]}


def test_the_nemotron_pii_script_ships_text_only_when_every_span_is_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    """a row whose spans all fail validation must not crash or fabricate entities: it ships text-only,
    populated() comes up empty, and the declared-outputs filter drops it downstream"""

    def fail_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        raise AssertionError("no resolution may run when every span is malformed")

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fail_resolve))
    Text = "Contact Bo Chen at 12 Oak St, 55501."
    Spans: list[Any] = [
        {"start": True, "end": 10, "label": "first_name"},
        {"start": -2, "end": 3, "label": "city"},
        {"start": 30, "end": 30, "label": "postcode"},
        "not-a-dict",
    ]
    _, Example = Script.dispatch("NemotronPiiScript", (("entities",), (Text, Spans)))

    assert Example.text == Text
    assert Example.entities == []
    assert Example.populated() == frozenset()


def test_the_nemotron_pii_script_ships_text_only_for_blank_or_empty_text() -> None:
    """blank/empty text must never crash the script; it ships as an empty example the filter drops"""
    _, Empty = Script.dispatch("NemotronPiiScript", (("entities",), ("", "[]")))
    assert Empty.text == ""
    assert Empty.populated() == frozenset()
    _, Blank = Script.dispatch("NemotronPiiScript", (("entities",), ("   \n\t", "[]")))
    assert Blank.text == ""
    assert Blank.populated() == frozenset()


def test_the_nemotron_pii_script_maps_known_labels_and_pascalcases_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    """the two-tier label policy through the real resolution chain (fullmap patched out):
    mapped labels yield their biolink category via the merged fallback map; unmapped tail labels
    (Ssn, Ipv4, ...) stay in training as raw PascalCased labels instead of being dropped"""

    monkeypatch.setattr(ScriptUtils, "_fullmap_best", classmethod(lambda cls, normalized: {}))
    Text = "My first name is Peggy and my SSN is 250-38-8116."
    Spans = "[{'start': 17, 'end': 22, 'text': 'Peggy', 'label': 'first_name'}, {'start': 37, 'end': 48, 'text': '250-38-8116', 'label': 'ssn'}]"
    _, Example = Script.dispatch("NemotronPiiScript", (("entities",), (Text, Spans)))

    assert {entity.label: entity.mentions for entity in Example.entities} == {"Human": ["Peggy"], "Ssn": ["250-38-8116"]}
    assert ScriptUtils.is_biolink_category("Human")
    assert not ScriptUtils.is_biolink_category("Ssn")
    assert ScriptUtils.pascal_label("medical_record_number") == "MedicalRecordNumber"


def test_the_nemotron_pii_script_never_emits_relations(monkeypatch: pytest.MonkeyPatch) -> None:
    """gazetteer predicates are biomedical-mined, so relation extraction would fabricate edges between
    PII mentions under unbounded predicates; the script must stay entity-only in shape and output"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="Mara", category="Human", origin="fallback"),
            ResolvedMention(mention="Duluth", category="GeographicLocation", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Text = "Mara Voss lives in Duluth and works full-time as a machinist."
    Spans = "[{'start': 0, 'end': 4, 'text': 'Mara', 'label': 'first_name'}, {'start': 19, 'end': 25, 'text': 'Duluth', 'label': 'city'}]"
    _, Example = Script.dispatch("NemotronPiiScript", (("entities",), (Text, Spans)))

    assert Example.relations == []
    assert "relations" not in Example.populated()
    assert "relations" not in Example.to_output()["output"]


def test_the_nemotron_pii_script_registers_under_its_declared_name() -> None:
    """self-registration via Script.__init_subclass__ is the dispatch contract; the ingest yaml
    resolves scripts by NAME, so a mismatched key would silently break routing"""
    from relmedner.scripts import NemotronPiiScript

    assert isinstance(Script.REGISTRY["NemotronPiiScript"], NemotronPiiScript)
    assert Script.REGISTRY["NemotronPiiScript"].NAME == "NemotronPiiScript"


# ---------------------------------------------------------------------------
# BioleafletsScript + ScriptUtils.parse_literal_dict (Bioleaflets ingest)
# ---------------------------------------------------------------------------


def bioleaflets_section(content: str | None, entities: list[Any] | None, title: str = "Overview") -> str:
    """one Section_N parquet cell: a python-repr dict string carrying exactly the measured keys
    Title / Section_Content / Entity_Recognition (468-row probe shape)"""
    return repr({"Title": title, "Section_Content": content, "Entity_Recognition": entities})


def comprehend_entry(text: str, label: str, begin: int, end: int) -> dict[str, Any]:
    """a real-probe-shaped Comprehend entry: the 4 base keys plus Id/Score/Category/Traits/Attributes"""
    return {
        "Text": text,
        "Type": label,
        "BeginOffset": begin,
        "EndOffset": end,
        "Id": 0,
        "Score": 0.99,
        "Category": "MEDICATION",
        "Traits": [],
        "Attributes": [],
    }


def stanza_entry(text: str, label: str, begin: int, end: int) -> dict[str, Any]:
    """a real-probe-shaped Stanza entry: only the 4 base keys"""
    return {"Text": text, "Type": label, "BeginOffset": begin, "EndOffset": end}


def test_parse_literal_dict_decodes_python_repr_dict_columns() -> None:
    """the bioleaflets parquet stores Section_N cells as python-repr dict strings, not dict objects:
    a dict-in-string decodes via ast.literal_eval (no code execution) and a real dict passes through"""
    Section = "{'Title': 'Overview', 'Section_Content': 'aspirin treats migraine .', 'Entity_Recognition': None}"

    assert ScriptUtils.parse_literal_dict(Section) == {
        "Title": "Overview",
        "Section_Content": "aspirin treats migraine .",
        "Entity_Recognition": None,
    }
    Real = {"Title": "Overview", "Section_Content": "c", "Entity_Recognition": []}
    assert ScriptUtils.parse_literal_dict(Real) == Real


def test_parse_literal_dict_skips_non_dict_rows_without_raising() -> None:
    """skip-don't-coerce: None, junk strings, lists, quoted strings, and malformed reprs all yield {}
    so callers skip the cell instead of crashing or coercing it into a section"""
    assert ScriptUtils.parse_literal_dict(None) == {}
    assert ScriptUtils.parse_literal_dict("garbage [") == {}
    assert ScriptUtils.parse_literal_dict("{'unbalanced': ") == {}
    assert ScriptUtils.parse_literal_dict(["not", "a", "dict"]) == {}
    assert ScriptUtils.parse_literal_dict("'a quoted string'") == {}
    assert ScriptUtils.parse_literal_dict(3) == {}


def test_every_bioleaflets_label_maps_to_a_biolink_category() -> None:
    """dataset-local vocabulary values stay real biolink classes (import-time validate_label_map
    raises first; this pins the invariant for the whole map)"""
    from relmedner.scripts import BioleafletsScript

    for raw_label, category in BioleafletsScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"fallback {raw_label!r} -> {category!r} is not a biolink class"


def test_the_bioleaflets_label_map_is_exactly_the_measured_vocabulary() -> None:
    """the measured 30-type vocabulary splits into the 10 faithful mappings (Comprehend and Stanza
    name one concept two ways: dx_name/problem, generic_name vs brand_name, test_name vs test,
    treatment_name vs treatment) and the PHI/noise tail that must stay UNMAPPED: AGE, ADDRESS, DATE,
    ID, NAME, PHONE_OR_FAX, PROFESSION, NUMBER, PRODUCT_NAME and TIME_TO_* are not biomedical
    concepts, so forcing them into a class would silently mislabel training data -- they surface as
    PascalCased raw tails instead"""
    from relmedner.scripts import BioleafletsScript

    assert sorted(BioleafletsScript.LABEL_MAP) == [
        "brand_name",
        "dx_name",
        "generic_name",
        "problem",
        "procedure_name",
        "system_organ_site",
        "test",
        "test_name",
        "treatment",
        "treatment_name",
    ]
    assert BioleafletsScript.LABEL_MAP["dx_name"] == "Disease" and BioleafletsScript.LABEL_MAP["problem"] == "Disease"
    assert BioleafletsScript.LABEL_MAP["generic_name"] == "Drug" and BioleafletsScript.LABEL_MAP["brand_name"] == "Drug"
    assert BioleafletsScript.LABEL_MAP["procedure_name"] == "Procedure"
    assert BioleafletsScript.LABEL_MAP["test_name"] == "ClinicalMeasurement" and BioleafletsScript.LABEL_MAP["test"] == "ClinicalMeasurement"
    assert BioleafletsScript.LABEL_MAP["treatment_name"] == "Treatment" and BioleafletsScript.LABEL_MAP["treatment"] == "Treatment"
    assert BioleafletsScript.LABEL_MAP["system_organ_site"] == "AnatomicalEntity"
    for noise in (
        "age",
        "address",
        "date",
        "id",
        "name",
        "phone_or_fax",
        "profession",
        "number",
        "product_name",
        "time_to_execution",
        "time_to_onset",
    ):
        assert noise not in BioleafletsScript.LABEL_MAP


def test_the_bioleaflets_script_decodes_real_row_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """end-to-end over the measured row shape: six python-repr section cells, char-offset
    Comprehend+Stanza entries bridged through content.split() triples, dataset-map resolution and
    grouping, and a gazetteer relation emitted only inside the section that carries the trigger"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert label_map is BioleafletsScript.LABEL_MAP
        # the Comprehend DX_NAME and Stanza PROBLEM spans overlap on 'migraine' with different types:
        # both survive into resolution (only exact duplicates collapse); mentions carry the RAW types
        # (resolution lowercases for the label-map lookup, so the fakes must assert them verbatim)
        assert mentions == [("aspirin", "GENERIC_NAME"), ("migraine", "DX_NAME"), ("migraine", "PROBLEM")]
        return [
            ResolvedMention(mention="aspirin", category="Drug", curie="CHEBI:15365", preferred_name="Acetylsalicylic acid", origin="fullmap"),
            ResolvedMention(mention="migraine", category="Disease", curie="MONDO:0005002", preferred_name="migraine disorder", origin="fullmap"),
            ResolvedMention(mention="migraine", category="Disease", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Content = "aspirin treats migraine ."
    Entities: list[Any] = [
        comprehend_entry("aspirin", "GENERIC_NAME", 0, 7),
        comprehend_entry("migraine", "DX_NAME", 15, 23),
        stanza_entry("migraine", "PROBLEM", 15, 23),
    ]
    Sections = (bioleaflets_section(Content, Entities),) + tuple(bioleaflets_section("empty section .", None) for _ in range(5))
    _, Example = Script.dispatch("BioleafletsScript", (("entities",), Sections))

    assert Example.text == " ".join(["aspirin treats migraine ."] + ["empty section ."] * 5)
    assert {entity.label: entity.mentions for entity in Example.entities} == {"Drug": ["aspirin"], "Disease": ["migraine"]}
    assert Example.entities[0].description is not None and "[fullmap: CHEBI:15365 | Acetylsalicylic acid]" in Example.entities[0].description
    assert Example.relations == [expected_relation("treats", "aspirin", "migraine")]
    assert Example.populated() == frozenset({"entities", "relations"})


def test_the_bioleaflets_script_collapses_exact_duplicates_but_keeps_different_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Comprehend re-annotates the same span the Stanza pass annotated: exact (begin, end, type)
    duplicates collapse -- a repeated mention would skew entity grouping -- while the same span under
    two different types survives, which is the measured Comprehend+Stanza overlap, real signal"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        # four entries decode to two mentions: the three identical DX_NAME entries collapsed to one
        assert mentions == [("migraine", "DX_NAME"), ("migraine", "PROBLEM")]
        return [
            ResolvedMention(mention="migraine", category="Disease", origin="fallback"),
            ResolvedMention(mention="migraine", category="Disease", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Entries: list[Any] = [
        comprehend_entry("migraine", "DX_NAME", 15, 23),
        comprehend_entry("migraine", "DX_NAME", 15, 23),
        stanza_entry("migraine", "PROBLEM", 15, 23),
        comprehend_entry("migraine", "DX_NAME", 15, 23),
    ]
    _, Example = Script.dispatch("BioleafletsScript", (("entities",), (bioleaflets_section("aspirin treats migraine .", Entries),)))

    assert {entity.label: entity.mentions for entity in Example.entities} == {"Disease": ["migraine"]}


def test_the_bioleaflets_script_skips_none_malformed_and_empty_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    """skip-don't-coerce per section: None cells, junk reprs, list cells, a None/empty/missing
    Section_Content, and a None or non-list-decoding Entity_Recognition each drop their section and
    never raise; the one good section still yields its entities and its relation"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("aspirin", "GENERIC_NAME"), ("migraine", "DX_NAME")]
        return [
            ResolvedMention(mention="aspirin", category="Drug", origin="fallback"),
            ResolvedMention(mention="migraine", category="Disease", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Content = "aspirin treats migraine ."
    Good: list[Any] = [
        comprehend_entry("aspirin", "GENERIC_NAME", 0, 7),
        stanza_entry("migraine", "DX_NAME", 15, 23),
    ]
    Sections: tuple[Any, ...] = (
        None,
        "garbage [",
        ["not", "a", "section"],
        bioleaflets_section(None, []),
        bioleaflets_section("", []),
        repr({"Title": "Overview", "Entity_Recognition": []}),
        bioleaflets_section(Content, None),
        bioleaflets_section(Content, "not a list"),
        bioleaflets_section(Content, Good),
    )
    _, Example = Script.dispatch("BioleafletsScript", (("entities",), Sections))

    assert Example.text == " ".join([Content] * 3)
    assert {entity.label: entity.mentions for entity in Example.entities} == {"Drug": ["aspirin"], "Disease": ["migraine"]}
    assert Example.relations == [expected_relation("treats", "aspirin", "migraine")]


def test_the_bioleaflets_script_drops_text_mismatched_and_malformed_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """offsets are authoritative and the Text check is the stale-annotation guard: an entry whose
    Text differs from content[begin:end] drops (the slice also catches inverted and negative
    offsets); so do non-dict entries, missing/mistyped offsets (bool is not int, floats are not
    coerced), and a missing Type -- Id/Score/Category/Traits/Attributes are ignored on survivors"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        assert mentions == [("aspirin", "GENERIC_NAME"), ("migraine", "PROBLEM")]
        return [
            ResolvedMention(mention="aspirin", category="Drug", origin="fallback"),
            ResolvedMention(mention="migraine", category="Disease", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Content = "aspirin treats migraine ."
    Entries: list[Any] = [
        comprehend_entry("aspirin", "GENERIC_NAME", 0, 7),
        comprehend_entry("aspirin", "GENERIC_NAME", 0, 6),
        comprehend_entry("treats", "GENERIC_NAME", 0, 7),
        {"Text": "migraine", "Type": "PROBLEM", "BeginOffset": 15},
        stanza_entry("migraine", "PROBLEM", 15, "23"),
        stanza_entry("migraine", "PROBLEM", True, 23),
        stanza_entry("migraine", "PROBLEM", 15, 23.0),
        {"Text": "migraine", "BeginOffset": 15, "EndOffset": 23},
        {"Text": "migraine", "Type": "PROBLEM", "BeginOffset": 15, "EndOffset": 23, "Id": None, "Score": None},
        stanza_entry("aspirin", "GENERIC_NAME", 7, 0),
        stanza_entry("migraine", "PROBLEM", -8, -1),
        "not a dict",
        None,
        3,
    ]
    _, Example = Script.dispatch("BioleafletsScript", (("entities",), (bioleaflets_section(Content, Entries),)))

    assert {entity.label: entity.mentions for entity in Example.entities} == {"Drug": ["aspirin"], "Disease": ["migraine"]}


def test_the_bioleaflets_script_emits_text_only_rows_when_nothing_yields() -> None:
    """all-None sections emit the empty example (never raise); content with no surviving
    Entity_Recognition emits a text-only example with no entity/relation keys, whose empty
    populated() the pipeline's matches_declared_outputs gate drops"""
    _, Empty = Script.dispatch("BioleafletsScript", (("entities",), tuple(None for _ in range(6))))
    assert Empty.text == ""
    assert Empty.populated() == frozenset()
    _, NoEntities = Script.dispatch("BioleafletsScript", (("entities",), (bioleaflets_section("no entities here .", None),)))
    assert NoEntities.text == "no entities here ."
    assert NoEntities.entities == [] and NoEntities.relations == []
    assert NoEntities.populated() == frozenset()


def test_the_bioleaflets_script_never_pairs_mentions_across_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    """sections are independent documents: a trigger at the end of one section must not bracket a
    mention living in another section (a merged token stream would fabricate treats(aspirin,
    migraine) here, which no single section asserts)"""

    def fake_resolve(mentions: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention=mention, category="Drug" if mention == "aspirin" else "Disease", origin="fallback") for mention, _ in mentions
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Sections = (
        bioleaflets_section("aspirin is used to treat", [comprehend_entry("aspirin", "GENERIC_NAME", 0, 7)]),
        bioleaflets_section("migraine rarely", [stanza_entry("migraine", "PROBLEM", 0, 8)]),
    )
    _, Example = Script.dispatch("BioleafletsScript", (("entities",), Sections))

    assert {entity.label: entity.mentions for entity in Example.entities} == {"Drug": ["aspirin"], "Disease": ["migraine"]}
    assert Example.relations == []
    assert Example.populated() == frozenset({"entities"})


def test_the_synthetic_ner_ade_tweets_script_registers_under_its_declared_name() -> None:
    """self-registration via Script.__init_subclass__ is the dispatch contract; the ingest yaml
    resolves scripts by NAME, so a mismatched key would silently break routing (US-001)"""
    from relmedner.scripts import SyntheticNerAdeTweetsScript

    assert isinstance(Script.REGISTRY["SyntheticNerAdeTweetsScript"], SyntheticNerAdeTweetsScript)
    assert Script.REGISTRY["SyntheticNerAdeTweetsScript"].NAME == "SyntheticNerAdeTweetsScript"


def test_the_scripts_package_exports_every_script_sorted_and_script_only() -> None:
    """__all__ is the package's public surface: alphabetical so additions have one right place,
    and Script instances only so a stray helper cannot leak into registry-driven dispatch (US-001)"""
    import relmedner.scripts as scripts

    assert scripts.__all__ == sorted(scripts.__all__)
    assert all(issubclass(getattr(scripts, name), Script) for name in scripts.__all__)
