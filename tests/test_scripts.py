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
    assert Example.relations == [
        Relation(name="treats", fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="migraine")])
    ]


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

    assert Example.relations == [
        Relation(name="treats", fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="migraine")])
    ]


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
