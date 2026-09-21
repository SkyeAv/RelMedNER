from __future__ import annotations

import json
from typing import Any, ClassVar, Self

import pytest

from relmedner.fullmap_mine import FullmapMiner
from relmedner.models import Entity, Relation, RelationField, TrainingExample
from relmedner.scripts import GlinerBiomedScript, KnowledgatorBiomedScript
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
    """the data's 32-label vocabulary includes plural/legacy variants the HF card never documents;
    the run() call must hand resolve_mentions the dataset map so PRODUCTS/ORGANISMS resolve"""

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
    """the map matches REQ-sc-3's exact block: the 24 canonical HF-card classes minus the three with
    no honest biolink target (LANGUAGE, REGULATION OR LAW, MONEY) plus the 7 measured plural/legacy
    variants (GENES, LOCATIONS, ORGANISMS, ORGANIZATIONS, PRODUCTS, FINDINGS/PHENOTYPES, DISORDERS);
    the catch-alls stay deliberately absent so they fall through to raw PascalCase tails"""
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
    assert len(KnowledgatorBiomedScript.LABEL_MAP) == 28
    for raw_label in ("language", "regulation or law", "money", "unlabelled", "intellectual"):
        assert raw_label not in KnowledgatorBiomedScript.LABEL_MAP


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
