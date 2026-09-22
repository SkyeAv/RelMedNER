from __future__ import annotations

from typing import Any

import pytest

from relmedner.models import Entity, TrainingExample
from relmedner.scripts.gliner_multilingual import GlinerMultilingualScript, coerced_span
from relmedner.types import Script, ScriptValues
from relmedner.utils import ScriptUtils

# the multilingual loader ships two NER span shapes; the second is what a lossy export round-trip
# produces: stringified indices and a label wrapped in literal quote characters
REAL_INT_SHAPE: list[Any] = [18, 21, "organization"]
STRINGIFIED_SHAPE: list[Any] = ["18", "21", '"organization"']
TOKEN_COUNT: int = 25


def run_script(values: ScriptValues) -> TrainingExample:
    _, example = Script.dispatch("GlinerMultilingualScript", (("entities",), values))
    return example


def test_the_script_self_registers_under_its_declared_name() -> None:
    """Script.__init_subclass__ keys the registry on NAME; dispatch depends on that exact key"""
    assert isinstance(Script.REGISTRY["GlinerMultilingualScript"], GlinerMultilingualScript)
    assert GlinerMultilingualScript.NAME == "GlinerMultilingualScript"


def test_the_coercion_shim_accepts_both_loader_span_shapes() -> None:
    """the corpus mixes real-int spans with stringified quote-wrapped spans; both must collapse
    into the same (start, end_inclusive, label) triple or downstream code sees two vocabularies"""
    assert coerced_span(REAL_INT_SHAPE, TOKEN_COUNT) == (18, 21, "organization")
    assert coerced_span(STRINGIFIED_SHAPE, TOKEN_COUNT) == (18, 21, "organization")
    assert coerced_span(["18", 21, "'organization'"], TOKEN_COUNT) == (18, 21, "organization")
    # int() is lenient about padding and underscore separators, and the shim documents rather than
    # fights that: the loader only ever emits plain decimal strings, so tightening would add a rule
    # no row can exercise while the docstring claim stays untested
    assert coerced_span([" 18 ", "2_1", "organization"], TOKEN_COUNT) == (18, 21, "organization")


def test_the_coercion_shim_drops_malformed_spans_without_crashing() -> None:
    """external rows must never crash the ingest and never become silently coerced garbage:
    wrong arity, non-numeric or fractional indices, boolean indices, and non-string labels
    (even quote-wrapped ones) are all dropped, not repaired"""
    tokens: list[str] = ["Aspirin", "treats", "Kopfschmerzen"]
    malformed: list[Any] = [
        [0, 0],
        [0, 0, "drug", "extra"],
        [],
        None,
        3,
        "span",
        [0.0, 0, "drug"],
        [0, 0.0, "drug"],
        [True, 0, "drug"],
        [0, False, "drug"],
        [None, 0, "drug"],
        [0, 0, None],
        [0, 0, 7],
        [0, 0, ""],
        [0, 0, '""'],
        ["zero", 0, "drug"],
        [0, "2.0", "drug"],
    ]
    assert [coerced_span(entry, len(tokens)) for entry in malformed] == [None] * len(malformed)
    assert coerced_span([0, 0, "drug"], len(tokens)) == (0, 0, "drug")


def test_the_coercion_shim_enforces_end_inclusive_bounds_against_the_row_tokens() -> None:
    """mirror ScriptUtils.mention_spans semantics: 0 <= start <= end < len(tokens); end is
    inclusive, so the last valid index is len(tokens) - 1 and anything past it is dropped"""
    tokens: list[str] = ["a", "b", "c"]
    assert coerced_span([0, 2, "drug"], len(tokens)) == (0, 2, "drug")
    assert coerced_span([2, 2, "drug"], len(tokens)) == (2, 2, "drug")
    assert coerced_span([2, 3, "drug"], len(tokens)) is None
    assert coerced_span([-1, 1, "drug"], len(tokens)) is None
    assert coerced_span([1, 0, "drug"], len(tokens)) is None


def test_the_script_resolves_labels_directly_without_the_fullmap_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """US-001's decision: multilingual synthetic surfaces have no fullmap term entries, so the
    labeling chain must be a pure LABEL_MAP/pascal_label lookup -- any accidental call into
    ScriptUtils.resolve_mentions or the fullmap database fails this test loudly"""

    def forbid_resolution(*args: Any) -> Any:
        raise AssertionError("GlinerMultilingualScript must not consult resolve_mentions/fullmap")

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(forbid_resolution))
    monkeypatch.setattr(ScriptUtils, "fullmap_available", classmethod(lambda cls: forbid_resolution()))
    Example: TrainingExample = run_script((["Krankheit", "und", "Aspirin"], [[0, 0, "krankheit"], [2, 2, "arzneimittel"]]))

    assert {entity.label: entity.mentions for entity in Example.entities} == {"Disease": ["Krankheit"], "Drug": ["Aspirin"]}


def test_the_script_maps_head_labels_to_biolink_classes_and_pascalcases_the_rest() -> None:
    """cross-lingual head labels collapse to biolink classes; unmapped surfaces (time/date/event
    breadth labels) ride pascal_label so the zero-shot vocabulary stays biolink-cased, not lost"""
    Example: TrainingExample = run_script(
        (
            ["Der", "Patient", "besucht", "die", "Veranstaltung", "wegen", "seiner", "Krankheit"],
            [[1, 1, "person"], [7, 7, "krankheit"], [4, 4, "Event"]],
        )
    )

    assert {entity.label: entity.mentions for entity in Example.entities} == {
        "Human": ["Patient"],
        "Disease": ["Krankheit"],
        "Event": ["Veranstaltung"],
    }


def test_the_script_groups_repeated_labels_into_one_entity() -> None:
    """group_entities is the shared shape every Script emits: one Entity per category carrying
    every mention of that category in first-occurrence order"""
    Example: TrainingExample = run_script(
        (
            ["Der", "Mann", "und", "die", "Frau", "wohnen", "in", "Berlin"],
            [[1, 1, "person"], [4, 4, "persona"], [7, 7, "location"]],
        )
    )

    # descriptions ride biolink_category_description; some classes (Human) carry no schema
    # docstring, so only grouping shape and ordering are this test's subject
    assert Example.entities == [
        Entity(label="Human", mentions=["Mann", "Frau"], description=Example.entities[0].description),
        Entity(label="GeographicLocation", mentions=["Berlin"], description=Example.entities[1].description),
    ]


def test_the_script_emits_entities_only_and_relations_stay_empty() -> None:
    """this ingest declares the entities output shape alone; a populated relations list would
    leak into downstream dispatch even though no gazetteer pass runs over this corpus"""
    Example: TrainingExample = run_script((["Aspirin", "lindert", "Kopfschmerzen"], [[0, 0, "arzneimittel"], [2, 2, "schmerzen"]]))

    assert Example.populated() == frozenset({"entities"})
    assert Example.relations == []
    assert Example.classifications == [] and Example.structures == []


def test_the_script_yields_an_example_for_rows_with_no_usable_text() -> None:
    """skip-don't-crash: empty, non-list, or all-malformed rows still yield a TrainingExample
    (text-only, possibly empty) so the stream never crashes on a bad export row"""
    _, Empty = Script.dispatch("GlinerMultilingualScript", (("entities",), ([], [])))
    assert Empty.text == "" and Empty.populated() == frozenset()
    _, NotAList = Script.dispatch("GlinerMultilingualScript", (("entities",), ("garbage", "not a list")))
    assert NotAList.text == "" and NotAList.populated() == frozenset()
    _, AllMalformed = Script.dispatch("GlinerMultilingualScript", (("entities",), (["a", "b"], [[0, 0], [0, 0, "drug", "extra"]])))
    assert AllMalformed.text == "a b" and AllMalformed.populated() == frozenset()
    _, Mismatched = Script.dispatch("GlinerMultilingualScript", (("entities",), (["a"], [[0, 9, "drug"]])))
    assert Mismatched.text == "a" and Mismatched.populated() == frozenset()


def test_every_label_map_entry_maps_to_a_biolink_category() -> None:
    """dataset-local vocabulary values must be real biolink classes; validate_label_map already
    fails at import, and this test pins the invariant for future map edits"""
    for raw_label, category in GlinerMultilingualScript.LABEL_MAP.items():
        assert ScriptUtils.is_biolink_category(category), f"fallback {raw_label!r} -> {category!r} is not a biolink class"


def test_the_script_processes_german_and_polish_rows_end_to_end() -> None:
    """the corpus's whole point is cross-lingual breadth: German and Polish surfaces must both
    land on biolink classes, including a stringified quote-wrapped span shape on the Polish row"""
    German: TrainingExample = run_script(
        (
            ["Der", "Patient", "nimmt", "Aspirin", "gegen", "die", "Krankheit"],
            [[1, 1, "person"], [3, 3, "arzneimittel"], [6, 6, "krankheit"]],
        )
    )
    assert German.text == "Der Patient nimmt Aspirin gegen die Krankheit"
    assert {entity.label: entity.mentions for entity in German.entities} == {
        "Human": ["Patient"],
        "Drug": ["Aspirin"],
        "Disease": ["Krankheit"],
    }

    Polish: TrainingExample = run_script(
        (
            ["Choroba", "Parkinsona", "dotyka", "pacjenta", "w", "Polsce"],
            [["0", "1", '"choroba"'], ["3", "3", '"osoba"'], ["5", "5", '"miejsce"']],
        )
    )
    assert Polish.text == "Choroba Parkinsona dotyka pacjenta w Polsce"
    assert {entity.label: entity.mentions for entity in Polish.entities} == {
        "Disease": ["Choroba Parkinsona"],
        "Human": ["pacjenta"],
        "GeographicLocation": ["Polsce"],
    }
