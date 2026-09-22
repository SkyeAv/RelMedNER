from __future__ import annotations

from typing import Any, ClassVar, Self

import pytest

from relmedner.models import Entity, Relation, RelationField, TrainingExample
from relmedner.scripts import GlinerBiomedScript
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


def expected_relation(name: str, head: str, tail: str) -> Relation:
    """emitted relations carry their biolink slot description; build the matching expectation"""
    return Relation(
        name=name,
        fields=[RelationField(name="head", value=head), RelationField(name="tail", value=tail)],
        description=ScriptUtils.predicate_description(name),
    )
