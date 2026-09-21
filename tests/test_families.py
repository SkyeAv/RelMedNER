from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from relmedner.families import (
    CLASSIFICATION_LABELS,
    EXTRACTION_LABELS,
    RELATION_DELIMITER,
    RowFamily,
    validate_label_map,
)
from relmedner.models import Relation, RelationField, TrainingExample
from relmedner.scripts import GlinerBiomedPostScript
from relmedner.types import Script
from relmedner.utils import ResolvedMention, ScriptUtils

GLINER_DATA_MODULE: str = "gliner2_training_data_families"


def load_gliner_data() -> ModuleType:
    """loads gliner2.training.data by path because importing the package pulls torch"""

    if GLINER_DATA_MODULE in sys.modules:
        return sys.modules[GLINER_DATA_MODULE]

    Located: ModuleSpec | None = importlib.util.find_spec("gliner2")
    if Located is None or not Located.submodule_search_locations:
        pytest.skip("gliner2 is not installed")

    Source: Path = Path(next(iter(Located.submodule_search_locations))) / "training" / "data.py"
    Spec: ModuleSpec | None = importlib.util.spec_from_file_location(GLINER_DATA_MODULE, Source)
    if Spec is None or Spec.loader is None:
        pytest.skip("gliner2 training data module is unavailable")

    Module: ModuleType = importlib.util.module_from_spec(Spec)
    sys.modules[GLINER_DATA_MODULE] = Module
    Spec.loader.exec_module(Module)
    return Module


def dispatch_row(ner: list[Any], negatives: list[str] | None = None) -> TrainingExample:
    Tokens: list[str] = [
        "Identify",
        "relations",
        "between",
        "entities",
        "based",
        "on",
        "provided",
        "source",
        "entity",
        "and",
        "relation",
        ":",
        "kynurenine",
        "pathway",
        "KP",
        "metabolites",
        "are",
        "associated",
        "with",
        "accelerated",
        "atherosclerosis",
        "in",
        "chronic",
        "kidney",
        "disease",
        "CKD",
        "patients",
        ".",
    ]
    _, Example = Script.dispatch("GlinerBiomedPostScript", (("relations",), (Tokens, ner, negatives)))
    return Example


def test_families_self_register_in_priority_order() -> None:
    Names: list[str] = [type(family).__name__ for family in RowFamily.REGISTRY]

    assert Names == ["RelationFamily", "ClassificationFamily", "ExtractionFamily", "EntityFamily"]
    assert all(RowFamily.REGISTRY[index].PRIORITY <= RowFamily.REGISTRY[index + 1].PRIORITY for index in range(len(Names) - 1))


def test_label_sets_partition_the_dispatch() -> None:
    RelationLabels: tuple[str, ...] = ("chemokines <> associated with",)
    ClassificationLabels: tuple[str, ...] = ("label", "tag", "category", "class")
    ExtractionLabels: tuple[str, ...] = ("match",)

    assert RowFamily.REGISTRY[0].matches(RelationLabels)
    assert not RowFamily.REGISTRY[1].matches(RelationLabels) and not RowFamily.REGISTRY[2].matches(RelationLabels)
    assert RowFamily.REGISTRY[1].matches(ClassificationLabels) and not RowFamily.REGISTRY[2].matches(ClassificationLabels)
    assert RowFamily.REGISTRY[2].matches(ExtractionLabels)
    # EntityFamily is the catch-all: it also matches the label sets above, but it is only consulted
    # last, so precedence -- not exclusivity -- is what routes a row to its family
    assert RowFamily.REGISTRY[3].matches(ClassificationLabels) and RowFamily.REGISTRY[3].matches(ExtractionLabels)
    assert not any(family.matches(()) for family in RowFamily.REGISTRY)


def test_the_family_label_sets_stay_disjoint() -> None:
    assert not (CLASSIFICATION_LABELS & EXTRACTION_LABELS)
    assert all(RELATION_DELIMITER not in label for label in CLASSIFICATION_LABELS | EXTRACTION_LABELS)


def test_validate_label_map_rejects_non_biolink_targets_loudly() -> None:
    with pytest.raises(ValueError, match="is not a biolink class"):
        validate_label_map({"person": "Person"}, "NowhereScript")


def test_the_post_training_label_map_targets_biolink_and_stays_off_the_shared_base() -> None:
    validate_label_map(GlinerBiomedPostScript.LABEL_MAP, GlinerBiomedPostScript.NAME)
    for raw_label, category in GlinerBiomedPostScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"{raw_label!r} -> {category!r} is not a biolink class"
    assert "person" not in ScriptUtils.FALLBACK_LABEL_MAP  # dataset vocabulary lives with its dataset


def test_predicate_resolution_maps_biolink_and_shapes_the_rest() -> None:
    assert ScriptUtils.resolve_predicate("associated with") == ("associated_with", True)
    assert ScriptUtils.resolve_predicate("Part-Of") == ("part_of", True)
    assert ScriptUtils.resolve_predicate("was attacked by") == ("was_attacked_by", False)
    assert ScriptUtils.resolve_predicate("HMAS-Shropshire") == ("hmas_shropshire", False)


def test_relation_family_keeps_gold_positives_and_shapes_their_predicates() -> None:
    # the span marks the tail entity; the label prefix carries the head -- exactly the real rows' geometry
    Example: TrainingExample = dispatch_row(
        [[12, 15, "accelerated atherosclerosis <> associated with"], [22, 26, "accelerated atherosclerosis <> occurs in"]]
    )

    assert Example.populated() == frozenset({"relations"})
    assert Example.relations == [
        Relation(
            name="associated_with",
            fields=[
                RelationField(name="head", value="accelerated atherosclerosis"),
                RelationField(name="tail", value="kynurenine pathway KP metabolites"),
            ],
            evidence="asserted",
        ),
        Relation(
            name="occurs_in",
            fields=[
                RelationField(name="head", value="accelerated atherosclerosis"),
                RelationField(name="tail", value="chronic kidney disease CKD patients"),
            ],
            evidence="asserted",
        ),
    ]


def test_relation_family_normalizes_native_predicates_and_dedups_repeated_spans() -> None:
    Example: TrainingExample = dispatch_row(
        [
            [19, 20, "accelerated atherosclerosis <> was attacked by"],
            [19, 20, "accelerated atherosclerosis <> was-attacked-by"],
            [19, 20, "accelerated atherosclerosis <> was attacked by"],
            [5, 5, "match"],  # non-delimiter spans ride along but contribute nothing to the relation family
        ]
    )

    assert [relation.name for relation in Example.relations] == ["was_attacked_by"]


def test_relation_family_drops_heads_tokenization_tore_away_from_the_text() -> None:
    """the dataset labels carry detokenized surfaces ('CC-chemokines' against tokens 'CC', '-', 'chemokines');
    gliner2's sanitizer drops such relations, so extraction filters them instead"""
    Example: TrainingExample = dispatch_row([[0, 3, "absent-from-text entity <> associated with"]])

    assert Example.relations == []


def test_sampled_negatives_are_guarded_and_capped() -> None:
    # ordered so every guard branch executes before the 2x cap breaks the loop
    Negatives: list[str] = [
        "broken negative",  # malformed
        "ghost entity <> associated with <> phantom tail",  # not in text
        "accelerated atherosclerosis <> associated with <> kynurenine pathway KP metabolites",  # duplicate of the positive
        "accelerated atherosclerosis <> occurs in <> accelerated atherosclerosis",  # self-loop
        "kynurenine pathway KP metabolites <> associated with <> accelerated atherosclerosis",  # kept (a reversed triple is a distinct candidate)
        "accelerated atherosclerosis <> associated with <> chronic kidney disease CKD patients",  # kept (cap reached)
        "chronic kidney disease CKD patients <> occurs in <> kynurenine pathway KP metabolites",  # over cap
    ]
    Example: TrainingExample = dispatch_row([[12, 15, "accelerated atherosclerosis <> associated with"]], Negatives)

    assert [(relation.name, relation.negated, relation.evidence) for relation in Example.relations] == [
        ("associated_with", False, "asserted"),
        ("not_associated_with", True, "sampled_negative"),
        ("not_associated_with", True, "sampled_negative"),
    ]


def test_sampled_negatives_never_exceed_twice_the_positives() -> None:
    Negatives: list[str] = [
        "accelerated atherosclerosis <> was attacked by <> kynurenine pathway KP metabolites",
        "accelerated atherosclerosis <> was attacked by <> chronic kidney disease CKD patients",
        "chronic kidney disease CKD patients <> was attacked by <> kynurenine pathway KP metabolites",
        "kynurenine pathway KP metabolites <> was attacked by <> accelerated atherosclerosis",
    ]
    Example: TrainingExample = dispatch_row([[12, 15, "kynurenine pathway KP metabolites <> associated with"]], Negatives)

    Sampled: list[Relation] = [relation for relation in Example.relations if relation.evidence == "sampled_negative"]
    Asserted: list[Relation] = [relation for relation in Example.relations if relation.evidence == "asserted"]

    assert len(Asserted) == 1
    assert len(Sampled) == 2


def test_emitted_relation_fields_stay_exactly_head_and_tail() -> None:
    """gliner2's Relation.__init__ swallows head/tail as named params, so any third field silently
    vanishes from the record, and its sanitizer drops relations whose string values miss the text --
    this lock keeps negated/evidence model-level and the training record gliner2-safe"""
    Example: TrainingExample = dispatch_row(
        [[12, 15, "accelerated atherosclerosis <> associated with"]],
        ["chronic kidney disease CKD patients <> associated with <> accelerated atherosclerosis"],
    )

    assert Example.relations_out() == {
        "relations": [
            {"associated_with": {"head": "accelerated atherosclerosis", "tail": "kynurenine pathway KP metabolites"}},
            {"not_associated_with": {"head": "chronic kidney disease CKD patients", "tail": "accelerated atherosclerosis"}},
        ]
    }
    assert all(relation.negated is False for relation in Example.relations if relation.evidence == "asserted")


def test_the_gliner_sanitizer_keeps_our_relations_and_round_trips_them() -> None:
    GlinerData: ModuleType = load_gliner_data()
    Example: TrainingExample = dispatch_row(
        [[12, 15, "accelerated atherosclerosis <> associated with"], [22, 26, "accelerated atherosclerosis <> occurs in"]],
        ["chronic kidney disease CKD patients <> associated with <> kynurenine pathway KP metabolites"],
    )
    Output: dict[str, Any] = Example.to_output()

    Restored: Any = GlinerData.InputExample.from_dict(Output)
    Warnings: list[str]
    Valid: bool
    Warnings, Valid = Restored.sanitize()

    assert Valid is True
    assert len(Restored.relations) == len(Output["output"]["relations"])
    assert Warnings == []
    assert {key: value for key, value in Restored.to_dict()["output"].items() if key != "record_metadata"} == Output["output"]


def test_classification_family_collects_the_true_labels_in_span_order() -> None:
    Example: TrainingExample = dispatch_row([[13, 14, "tag"], [20, 21, "category"], [13, 14, "tag"], [24, 25, "class"]])

    assert Example.populated() == frozenset({"classifications"})
    assert Example.classifications[0].task == "topic classification"
    assert Example.classifications[0].multi_label is True
    assert Example.classifications[0].labels == Example.classifications[0].true_label
    assert Example.classifications[0].labels == ["pathway KP", "atherosclerosis in", "disease CKD"]


def test_a_single_classification_label_is_not_multi_label() -> None:
    Example: TrainingExample = dispatch_row([[13, 14, "label"]])

    assert Example.classifications[0].multi_label is False


def test_extraction_family_collects_match_surfaces() -> None:
    Example: TrainingExample = dispatch_row([[12, 15, "match"], [19, 20, "match"]])

    assert Example.populated() == frozenset({"structures"})
    assert Example.structures[0].name == "extraction"
    assert Example.structures[0].fields[0].name == "span"
    assert Example.structures[0].fields[0].value == ["kynurenine pathway KP metabolites", "accelerated atherosclerosis"]


def test_entity_family_resolves_groups_and_runs_the_gazetteer(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_resolve(spans: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        seen["spans"] = list(spans)
        seen["label_map"] = label_map
        return [
            ResolvedMention(
                mention="kynurenine pathway KP metabolites", category="ChemicalEntity", curie="CHEBI:1", preferred_name="Kynurenine", origin="fullmap"
            ),
            ResolvedMention(mention="accelerated atherosclerosis", category="PhenotypicFeature", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Example: TrainingExample = dispatch_row([[12, 15, "protein"], [19, 20, "medical condition"]])

    assert seen["spans"] == [("kynurenine pathway KP metabolites", "protein"), ("accelerated atherosclerosis", "medical condition")]
    assert seen["label_map"] == GlinerBiomedPostScript.LABEL_MAP  # the dataset vocabulary rides along
    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "ChemicalEntity": ["kynurenine pathway KP metabolites"],
        "PhenotypicFeature": ["accelerated atherosclerosis"],
    }
    # the gazetteer still runs over resolved categories: the 'associated with' trigger brackets the
    # pair, nearest-before is the head and nearest-after is the tail
    assert Example.relations == [
        Relation(
            name="associated_with",
            fields=[
                RelationField(name="head", value="kynurenine pathway KP metabolites"),
                RelationField(name="tail", value="accelerated atherosclerosis"),
            ],
            description=ScriptUtils.predicate_description("associated_with"),
        )
    ]


def test_entity_family_dedups_mentions_and_keeps_first_fullmap_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resolve(spans: list[tuple[str, str]], label_map: dict[str, str] | None = None) -> list[ResolvedMention]:
        return [
            ResolvedMention(
                mention="pancreas", category="GrossAnatomicalStructure", curie="UBERON:0001264", preferred_name="pancreas", origin="fullmap"
            ),
            ResolvedMention(mention="pancreas", category="GrossAnatomicalStructure", curie="UBERON:OTHER", preferred_name="other", origin="fullmap"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens: list[str] = ["the", "pancreas", "and", "pancreas", "."]
    _, Example = Script.dispatch("GlinerBiomedPostScript", (("entities",), (Tokens, [[1, 1, "anatomy"], [3, 3, "anatomy"]], None)))

    assert len(Example.entities) == 1
    assert Example.entities[0].mentions == ["pancreas"]
    assert "[fullmap: UBERON:0001264 | pancreas]" in (Example.entities[0].description or "")


def test_a_row_without_valid_spans_stays_bare_for_the_declared_outputs_filter() -> None:
    Example: TrainingExample = dispatch_row([])

    assert Example.populated() == frozenset()
    assert Example.relations == [] and Example.entities == [] and Example.classifications == [] and Example.structures == []


def test_the_script_tolerates_degenerate_row_values() -> None:
    _, Blank = Script.dispatch("GlinerBiomedPostScript", (("relations",), (None, None, None)))

    assert Blank.text == ""
    assert Blank.populated() == frozenset()

    _, MissingNer = Script.dispatch("GlinerBiomedPostScript", (("entities",), (["Aspirin"], None, None)))

    assert MissingNer.populated() == frozenset()
    assert MissingNer.entities == []
