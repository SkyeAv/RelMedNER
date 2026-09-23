from __future__ import annotations

from typing import Any

import pytest

from relmedner.families import validate_category
from relmedner.models import TrainingExample
from relmedner.scripts import CtkpInterventionsScript  # registry must stay populated alongside the new script
from relmedner.scripts.docred import DocredScript, _flatten_sents, _mention_span, _sentence_offsets
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: DocredScript = DocredScript()

# real-shaped fixture modeled on the measured dev[0] document (probe DOC2_SAMPLE_DEV, wenceslaus
# 2026-09-22): two sentences, five clusters with multi-mention and cross-sentence entities, and
# gold labels keyed h/t/r exactly as the raw json.gz files deliver them
SENTS: list[list[str]] = [
    ["Skai", "TV", "is", "a", "Greek", "free", "-", "to", "-", "air", "television", "network", "based", "in", "Piraeus", "."],
    ["It", "is", "part", "of", "the", "Skai", "Group", ",", "one", "of", "the", "largest", "media", "groups", "in", "Greece", "."],
]
VERTEX: list[list[dict[str, Any]]] = [
    [{"name": "Skai TV", "pos": [0, 2], "sent_id": 0, "type": "ORG"}],
    [{"name": "Greek", "pos": [4, 5], "sent_id": 0, "type": "MISC"}],
    [{"name": "Piraeus", "pos": [14, 15], "sent_id": 0, "type": "LOC"}],
    [{"name": "Skai Group", "pos": [5, 7], "sent_id": 1, "type": "ORG"}],
    [{"name": "Greece", "pos": [15, 16], "sent_id": 1, "type": "LOC"}],
]
LABELS: list[dict[str, Any]] = [
    {"r": "P127", "h": 0, "t": 3, "evidence": [0, 1]},
    {"r": "P17", "h": 3, "t": 4, "evidence": [1]},
]


def row(**overrides: Any) -> tuple[Any, ...]:
    """one DocRED row as the columns_out projection delivers it to ScriptValues"""
    merged: dict[str, Any] = {"sents": SENTS, "vertexSet": VERTEX, "labels": LABELS}
    merged |= overrides
    return (merged["sents"], merged["vertexSet"], merged["labels"])


def mentions_of(example: TrainingExample) -> list[str]:
    return [mention for entity in example.entities for mention in entity.mentions]


def test_the_script_self_registers_under_its_declared_name() -> None:
    """importing the module must install the instance in the shared registry (US-002 wires the
    ingest around Script.dispatch, which resolves by this exact NAME key)"""
    assert isinstance(Script.REGISTRY["DocredScript"], DocredScript)


def test_the_registry_addition_is_additive() -> None:
    """a broken import of the new module must not evict the twelve landed scripts; the ingest
    table and the registry are one contract, so the pre-existing entries must all survive"""
    assert isinstance(Script.REGISTRY["CtkpInterventionsScript"], CtkpInterventionsScript)
    assert len(Script.REGISTRY) >= 13


def test_the_mapped_type_categories_are_biolink_classes() -> None:
    """PER/ORG/LOC/MISC must stay real tablassert Categories members; the import-time guard only
    protects THIS class's constant, so the contract is re-asserted here for drift"""
    assert DocredScript.NAME == "DocredScript"
    for category in DocredScript.TYPE_MAP.values():
        assert ScriptUtils.is_biolink_category(category)


def test_the_import_time_category_guard_rejects_a_non_biolink_class() -> None:
    """a typo'd TYPE_MAP value must fail loudly at import, not silently train garbage labels;
    families.validate_category is the shared guard the module calls on its own constants"""
    with pytest.raises(ValueError, match="NotAClass"):
        validate_category("NotAClass", "BrokenDocredScript")


def test_the_relation_map_is_the_full_measured_vocabulary() -> None:
    """RELATION_MAP is frozen verbatim from data/rel_info.json.gz; a partial copy would silently
    drop gold relations whose P-id simply failed to be pasted (measured: 96 entries, 96/96
    label census coverage)"""
    assert len(DocredScript.RELATION_MAP) == 96
    assert all(isinstance(name, str) and name for name in DocredScript.RELATION_MAP.values())


def test_a_well_formed_row_emits_entities_and_relations() -> None:
    """the nonzero-yield guard: a realistic gold row must produce both declared shapes with the
    mapped biolink categories and snake_case predicates (the silent-zero-yield bug is this
    repo's most expensive landed failure)"""
    example = SCRIPT.run(row())

    assert {entity.label for entity in example.entities} == {"Agent", "GeographicLocation", "NamedThing"}
    # mentions flatten per entity label in first-occurrence order: Agent (both ORG clusters),
    # then NamedThing, then GeographicLocation
    assert mentions_of(example) == ["Skai TV", "Skai Group", "Greek", "Piraeus", "Greece"]
    assert [(relation.name, relation.fields[0].value, relation.fields[1].value) for relation in example.relations] == [
        ("owned_by", "Skai TV", "Skai Group"),
        ("country", "Skai Group", "Greece"),
    ]
    assert all(relation.evidence == "asserted" and relation.negated is False for relation in example.relations)


def test_the_emitted_text_is_the_flattened_token_stream_and_contains_every_surface() -> None:
    """gliner2's InputExample.validate rejects any relation field value outside the text; the
    flattened token join guarantees containment by construction, so this asserts the invariant
    on every mention and relation surface"""
    example = SCRIPT.run(row())

    assert example.text == ScriptUtils.join_tokens([token for sentence in SENTS for token in sentence])
    assert "Skai TV is a Greek free - to - air television network based in Piraeus ." in example.text
    for mention in mentions_of(example):
        assert mention in example.text
    for relation in example.relations:
        for field in relation.fields:
            assert field.value in example.text


def test_end_exclusive_pos_yields_the_inclusive_token_span() -> None:
    """the measured convention is END-EXCLUSIVE (77,515 vs 145 surface matches): pos [14, 15]
    must cover exactly token 14 ('Piraeus'), never bleed into token 15 ('.')"""
    example = SCRIPT.run(row())

    assert "Piraeus" in mentions_of(example)
    assert "Piraeus ." not in mentions_of(example)


def test_a_multi_sentence_cluster_uses_global_offsets() -> None:
    """pos is sentence-relative, so cluster 4's pos [15, 16] on sent_id 1 must resolve against
    sentence 1's tokens ('Greece'), not sentence 0's -- a wrong offset base fabricates surfaces"""
    example = SCRIPT.run(row())

    assert "Greece" in mentions_of(example)
    assert "largest" not in mentions_of(example)


def test_relations_cite_each_clusters_first_valid_mention() -> None:
    """DocRED relations hold between clusters, and a relation field must cite one surface; the
    earliest surviving mention is the canonical reference, so a later duplicate mention must
    appear in entities but never as a relation endpoint"""
    vertex = [cluster for cluster in VERTEX]
    vertex[3] = [
        {"name": "Skai Group", "pos": [5, 7], "sent_id": 1, "type": "ORG"},
        {"name": "Skai Group", "pos": [0, 2], "sent_id": 1, "type": "ORG"},  # fake second mention: 'It is'
    ]
    example = SCRIPT.run(row(vertexSet=vertex))

    assert example.relations[0].fields[0].value == "Skai TV"
    assert example.relations[0].fields[1].value == "Skai Group"  # first valid, not the fake later one


def test_an_out_of_bounds_end_drops_only_the_offending_mention() -> None:
    """pos [15, 17] on a 16-token sentence points past the end (the measured 0.19-0.56% oob
    class); it must drop per mention while the cluster's sibling and the other clusters ship"""
    vertex = [cluster for cluster in VERTEX]
    vertex[4] = [
        {"name": "Greece", "pos": [15, 18], "sent_id": 1, "type": "LOC"},
        {"name": "Greece", "pos": [15, 16], "sent_id": 1, "type": "LOC"},
    ]
    example = SCRIPT.run(row(vertexSet=vertex))

    assert mentions_of(example).count("Greece") == 1
    assert len(example.relations) == 2  # the cluster still has a valid mention, so relations survive


def test_a_cluster_with_no_surviving_mention_drops_its_relations() -> None:
    """a relation whose endpoint cluster lost every mention would cite a surface the text never
    contained; the relation drops while the rest of the row ships (measured consequence of the
    oob rule)"""
    vertex = [cluster for cluster in VERTEX]
    vertex[4] = [{"name": "Greece", "pos": [15, 99], "sent_id": 1, "type": "LOC"}]
    example = SCRIPT.run(row(vertexSet=vertex))

    assert "Greece" not in mentions_of(example)
    assert [relation.name for relation in example.relations] == ["owned_by"]


def test_degenerate_zero_length_and_reversed_spans_are_dropped() -> None:
    """pos with b <= a points at nothing; trusting it would emit an empty or backwards span that
    the validator rejects"""
    assert _mention_span({"pos": [3, 3], "sent_id": 0}, _sentence_offsets(SENTS), 33) is None
    assert _mention_span({"pos": [5, 2], "sent_id": 0}, _sentence_offsets(SENTS), 33) is None
    assert _mention_span({"pos": [-1, 2], "sent_id": 0}, _sentence_offsets(SENTS), 33) is None


def test_non_int_and_bool_pos_components_are_rejected() -> None:
    """bool is an int subclass in python (True == 1) and hub-side schema drift can deliver
    stringified or fractional indices; every non-plain-int component must drop, not coerce"""
    offsets = _sentence_offsets(SENTS)
    assert _mention_span({"pos": [True, 2], "sent_id": 0}, offsets, 33) is None
    assert _mention_span({"pos": [0, False], "sent_id": 0}, offsets, 33) is None
    assert _mention_span({"pos": ["0", 2], "sent_id": 0}, offsets, 33) is None
    assert _mention_span({"pos": [0, 2.5], "sent_id": 0}, offsets, 33) is None


def test_wrong_arity_pos_is_rejected() -> None:
    """a one- or three-element pos has no span semantics; the measured corpus always delivers
    exactly two, so anything else is drift"""
    offsets = _sentence_offsets(SENTS)
    assert _mention_span({"pos": [0], "sent_id": 0}, offsets, 33) is None
    assert _mention_span({"pos": [0, 1, 2], "sent_id": 0}, offsets, 33) is None
    assert _mention_span({"pos": 2, "sent_id": 0}, offsets, 33) is None


def test_out_of_range_or_non_int_sent_id_is_rejected() -> None:
    """sent_id indexes the sentence table; a bool or a two-element-table index 2 would resolve a
    span against the wrong (or no) sentence"""
    offsets = _sentence_offsets(SENTS)
    assert _mention_span({"pos": [0, 2], "sent_id": 2}, offsets, 33) is None
    assert _mention_span({"pos": [0, 2], "sent_id": True}, offsets, 33) is None
    assert _mention_span({"pos": [0, 2], "sent_id": "0"}, offsets, 33) is None
    assert _mention_span({"pos": [0, 2]}, offsets, 33) is None


def test_a_malformed_mention_or_cluster_drops_without_crashing_the_row() -> None:
    """hub-side drift can deliver a non-dict mention or a non-list cluster; the row must keep
    every well-formed sibling instead of crashing the stream worker"""
    vertex: list[Any] = ["not-a-cluster", [{"name": "Greek", "pos": [4, 5], "sent_id": 0, "type": "MISC"}], ["not-a-dict"]]
    example = SCRIPT.run(row(vertexSet=vertex))

    assert mentions_of(example) == ["Greek"]


def test_a_non_string_token_corrupts_the_whole_row_to_empty() -> None:
    """a non-string token shifts every later offset, so any span built past it would be a
    fabricated surface; the row ships as the empty example the pipeline filters instead"""
    example = SCRIPT.run(row(sents=[["Skai", "TV"], ["is", 5]]))

    assert example == TrainingExample(text="")


def test_the_name_field_never_drives_a_drop() -> None:
    """the measured 2.3-5.3% name-vs-token-join mismatch ("Worker-Peasant" vs
    ['Worker', '-', 'Peasant']) is annotation punctuation style, not corruption: the emitted
    surface is the token join and the mention must survive"""
    vertex = [[{"name": "Skai-TV (airline)", "pos": [0, 2], "sent_id": 0, "type": "ORG"}]]
    example = SCRIPT.run(row(vertexSet=vertex))

    assert mentions_of(example) == ["Skai TV"]


def test_an_unmapped_type_surfaces_as_a_titlecase_raw_category() -> None:
    """TIME and NUM have no honest biolink class, so they stay raw (the deliberate-omission
    stance); an unknown future type must degrade the same way instead of dropping gold spans.
    Surfaces are token joins: pos [4, 5] on sent 0 is 'Greek', pos [14, 15] is 'Piraeus'."""
    vertex = [
        [{"name": "2006", "pos": [4, 5], "sent_id": 0, "type": "TIME"}],
        [{"name": "unknown", "pos": [14, 15], "sent_id": 0, "type": "WIDGET"}],
        [{"name": "no type", "pos": [5, 7], "sent_id": 1}],
    ]
    example = SCRIPT.run(row(vertexSet=vertex))

    by_label = {entity.label: entity.mentions for entity in example.entities}
    assert by_label["Time"] == ["Greek"]
    assert by_label["Widget"] == ["Piraeus"]
    # the surface is the token join (pos [5, 7] on sent 1), not the annotator name string
    assert by_label["NamedThing"] == ["Skai Group"]


def test_a_self_loop_label_is_dropped() -> None:
    """h == t asserts a relation from a cluster to itself; the measured census found zero, so
    this guard exists purely for drift and must never fire on real data"""
    example = SCRIPT.run(row(labels=[{"r": "P17", "h": 3, "t": 3, "evidence": [1]}]))

    assert example.relations == []


def test_out_of_range_relation_indices_drop_the_relation_only() -> None:
    """an h or t beyond the cluster table has no endpoint surface; the malformed label drops
    individually while sibling labels ship"""
    example = SCRIPT.run(row(labels=LABELS + [{"r": "P17", "h": 0, "t": 99, "evidence": [0]}]))

    assert len(example.relations) == 2


def test_a_bool_or_non_int_relation_index_is_rejected() -> None:
    """True == 1 in python, so a bool h would silently point at cluster 1; the guard rejects it
    before the index ever reaches the cluster table"""
    example = SCRIPT.run(row(labels=[{"r": "P17", "h": True, "t": 4, "evidence": [1]}, {"r": "P17", "h": 3, "t": False, "evidence": [1]}]))

    assert example.relations == []


def test_an_unmapped_relation_id_drops_the_relation() -> None:
    """a P-id outside the frozen 96-entry map has no English name to resolve; the measured
    census says all 96 occur, so this drops only on drift (a bare P-id must never become a
    fabricated predicate)"""
    example = SCRIPT.run(row(labels=[{"r": "P9999", "h": 0, "t": 3, "evidence": [0]}]))

    assert example.relations == []


def test_a_malformed_label_entry_drops_the_relation_only() -> None:
    """non-dict labels and missing/non-string r keys are per-label drift; the row's other gold
    relations must still ship"""
    example = SCRIPT.run(row(labels=LABELS + ["not-a-dict", {"h": 0, "t": 3}, {"r": 17, "h": 0, "t": 3}]))

    assert len(example.relations) == 2


def test_labels_less_rows_ship_entities_only() -> None:
    """test-split rows carry NO labels key at all (the projection hands None down) and 26/293/13
    labeled-split docs have none: the entity half must still train and the declared-outputs
    filter decides what ships downstream"""
    for labels_value in (None, []):
        example = SCRIPT.run(row(labels=labels_value))

        assert len(example.entities) == 3
        assert example.relations == []
        assert example.populated() == frozenset({"entities"})


def test_flatten_rejects_non_list_sentences() -> None:
    """the offset base for every mention comes from this flatten, so a drifted sentence shape
    (a bare string is iterable and would silently produce character tokens) must poison the row"""
    assert _flatten_sents([["a", "b"], ["c"]]) == [["a", "b"], ["c"]]
    assert _flatten_sents(["abc"]) is None
    assert _flatten_sents([None]) is None


def test_empty_input_ships_an_empty_example() -> None:
    """a row with no sentences, clusters, or labels has no training signal; the empty example is
    what the pipeline's declared-outputs filter drops"""
    assert SCRIPT.run(row(sents=[], vertexSet=[], labels=[])) == TrainingExample(text="")
