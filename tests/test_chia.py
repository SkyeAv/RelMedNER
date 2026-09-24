from __future__ import annotations

from typing import Any

from relmedner.scripts.chia import ChiaScript

Chia = ChiaScript.REGISTRY["ChiaScript"]

TEXT: str = "Patients with metastatic carcinoid tumors may receive lanreotide. Women aged at least 18 years are eligible."


def ent(entity_id: str, entity_type: str, surface: str, text: str = TEXT) -> dict[str, Any]:
    """a real-shaped chia entity dict: char offsets are end-exclusive and computed from the text so
    the fixture can never drift out of bounds"""
    start = text.index(surface)
    return {"id": entity_id, "type": entity_type, "offsets": [[start, start + len(surface)]], "text": [surface]}


def rel(relation_type: str, arg1: str, arg2: str) -> dict[str, Any]:
    """a real-shaped chia relation dict: the gold link is by entity id, not by surface"""
    return {"type": relation_type, "arg1_id": arg1, "arg2_id": arg2}


BASE_ENTITIES: list[dict[str, Any]] = [
    ent("E1", "Condition", "metastatic carcinoid tumors"),
    ent("E2", "Drug", "lanreotide"),
    ent("E3", "Person", "Women"),
    ent("E4", "Value", "at least 18 years"),
]
BASE_RELATIONS: list[dict[str, Any]] = [
    rel("Has_temporal", "E2", "E4"),
    rel("Has_value", "E1", "E4"),
]


def mention_surfaces(example: Any) -> list[str]:
    return [mention for entity in example.entities for mention in entity.mentions]


def relation_surfaces(example: Any) -> list[str]:
    return [field.value for relation in example.relations for field in relation.fields]


def test_flat_rows_ship_entities_and_relations_with_every_surface_in_the_text() -> None:
    """nonzero-yield over a real-shaped flat row: four entities and two gold relations survive the
    bridge, and every emitted mention and relation surface is a substring of the emitted text
    (gliner2's InputExample.validate() rule)"""
    example = Chia.run((TEXT, BASE_ENTITIES, BASE_RELATIONS))

    assert "entities" in example.populated() and "relations" in example.populated()
    assert len(example.entities) == 4
    assert len(example.relations) == 2
    for surface in mention_surfaces(example) + relation_surfaces(example):
        assert surface in example.text, surface


def test_bigbio_passage_rows_reconstruct_the_text_and_ignore_events_and_coreferences() -> None:
    """the kb subset projects a one-passage passages list; the row must yield the same shapes, and
    the always-empty events/coreferences lists (measured 0 non-empty over the full split) ride in
    the row without touching the bridge"""
    passages = [{"offsets": [[0, len(TEXT)]], "text": [TEXT]}]
    example = Chia.run((passages, BASE_ENTITIES, BASE_RELATIONS))

    assert "entities" in example.populated() and "relations" in example.populated()
    assert len(example.entities) == 4
    assert len(example.relations) == 2
    for surface in mention_surfaces(example) + relation_surfaces(example):
        assert surface in example.text, surface


def test_multi_passage_rows_concatenate_in_offset_order() -> None:
    """the measured kb rows carry exactly one gapless passage; a two-passage row is the
    reconstruction guard: parts concatenate in offset order and entity offsets (computed against
    the reconstructed text, end-exclusive) still bridge"""
    first = "Patients with metastatic carcinoid tumors may receive lanreotide."
    second = " Women aged at least 18 years are eligible."
    row_text = first + second
    passages = [
        {"offsets": [[0, len(first)]], "text": [first]},
        {"offsets": [[len(first), len(row_text)]], "text": [second]},
    ]
    example = Chia.run((passages, [ent("E2", "Drug", "lanreotide", text=row_text)], []))

    # passages concatenate in offset order, so the reconstructed text tiles the row
    assert example.text.startswith("Patients")
    assert "lanreotide" in example.text
    assert any("lanreotide" in entity.mentions for entity in example.entities)


def test_mapped_labels_resolve_to_biolink_and_unmapped_labels_surface_as_raw_tails() -> None:
    """the label-map edges, pinned deterministically on a surface the fullmap redb does not resolve
    (so the fallback tier fires): a mapped type (condition) resolves to its biolink class with
    origin fallback, and a deliberately unmapped criterion-structure type (value, 4,002 measured
    spans) stays raw and surfaces PascalCased through the script"""
    from relmedner.utils import ScriptUtils

    resolved = ScriptUtils.resolve_mentions(
        [("at least 18 years", "condition"), ("at least 18 years", "value")],
        label_map=ChiaScript.LABEL_MAP,
    )

    assert resolved[0].category == "DiseaseOrPhenotypicFeature"
    assert resolved[0].origin == "fallback"
    assert resolved[1].category == "value"
    assert resolved[1].origin == "raw"

    example = Chia.run((TEXT, [ent("E4", "Value", "at least 18 years")], []))
    assert {entity.label for entity in example.entities} == {"Value"}


def test_multi_part_entities_emit_one_span_per_part_with_the_same_label() -> None:
    """1,752 measured entities carry two offset parts; each part becomes its own token span with
    the entity's label, and both parts' surfaces stay in the emitted text"""
    part_one = "metastatic carcinoid tumors"
    part_two = "Women"
    start = TEXT.index(part_one)
    multipart = {
        "id": "E9",
        "type": "Condition",
        "offsets": [[start, start + len(part_one)], [TEXT.index(part_two), TEXT.index(part_two) + len(part_two)]],
        "text": [part_one, part_two],
    }
    example = Chia.run((TEXT, [multipart], []))

    all_mentions = mention_surfaces(example)
    assert part_one in all_mentions
    assert part_two in all_mentions


def test_text_offsets_length_mismatch_drops_the_whole_entity() -> None:
    """the part-to-surface alignment is only trustworthy when the text list aligns with the offsets
    list (measured 0 mismatches on the fixed subsets), so a mismatched entity drops whole instead
    of shipping a half-aligned part"""
    broken = {"id": "E1", "type": "Condition", "offsets": [[12, 39]], "text": ["metastatic carcinoid tumors", "a second surface"]}
    example = Chia.run((TEXT, [broken], []))

    assert example.entities == []


def test_malformed_entity_shapes_drop_and_the_well_formed_sibling_survives() -> None:
    """skip-don't-coerce: every malformed shape vanishes while the healthy entity still ships"""
    malformed: list[Any] = [
        "not a dict",
        {"id": "E5", "offsets": [[0, 5]], "text": ["no type"]},
        {"id": "E6", "type": 42, "offsets": [[0, 5]], "text": ["numeric type"]},
        {"id": "E7", "type": "Condition", "offsets": ["not a list"], "text": ["offsets"]},
        {"id": "E8", "type": "Condition", "offsets": [[1, 2, 3]], "text": ["arity three"]},
        {"id": "E10", "type": "Condition", "offsets": [[True, 5]], "text": ["bool start"]},
        {"id": "E11", "type": "Condition", "offsets": [[0, 1.5]], "text": ["float end"]},
    ]
    example = Chia.run((TEXT, [*malformed, ent("E1", "Condition", "metastatic carcinoid tumors")], []))

    assert len(example.entities) == 1
    assert example.entities[0].mentions == ["metastatic carcinoid tumors"]


def test_out_of_bounds_and_degenerate_char_spans_drop_through_the_bridge() -> None:
    """the bridge's measured drop rules: past-the-text ends, negative starts, and zero-length
    spans vanish; the boundary cases at both ends are pinned"""
    oob_end = {"id": "E1", "type": "Condition", "offsets": [[len(TEXT) - 2, len(TEXT) + 10]], "text": ["le."]}
    oob_start = {"id": "E2", "type": "Condition", "offsets": [[-5, 5]], "text": ["Patie"]}
    zero_length = {"id": "E3", "type": "Condition", "offsets": [[10, 10]], "text": [""]}
    example = Chia.run((TEXT, [oob_end, oob_start, zero_length], []))

    assert example.entities == []


def test_integrator_relations_drop_while_a_well_formed_sibling_survives() -> None:
    """AND (2,631 scope / 3,677 without_scope measured) and OR (7) are sentence combinatorics, not
    relation semantics: they drop, and a gold has_qualifier on the same row still ships"""
    relations = [
        rel("AND", "E1", "E2"),
        rel("OR", "E1", "E3"),
        rel("Has_qualifier", "E1", "E3"),
    ]
    example = Chia.run((TEXT, BASE_ENTITIES, relations))

    assert len(example.relations) == 1
    assert example.relations[0].name == "has_qualifier"


def test_dangling_and_selfloop_relations_drop() -> None:
    """dangling arg ids and id self-loops measured 0 but the guards stay; the surface self-loop
    covers the measured 33/48 rows where two ids carry the same surface"""
    relations = [
        rel("Has_value", "E1", "MISSING"),
        rel("Has_value", "E1", "E1"),
        rel("Has_value", "E3", "E3"),
    ]
    example = Chia.run((TEXT, BASE_ENTITIES, relations))

    assert example.relations == []


def test_surface_selfloop_relations_drop() -> None:
    """two distinct entity ids whose first surviving surfaces are case-insensitively equal cannot
    anchor a relation (the measured surface self-loop rule)"""
    entities = [
        {"id": "A", "type": "Condition", "offsets": [[0, 8]], "text": ["Patients"]},
        {"id": "B", "type": "Person", "offsets": [[0, 8]], "text": ["Patients"]},
    ]
    example = Chia.run((TEXT, entities, [rel("Has_value", "A", "B")]))

    assert example.relations == []


def test_relation_surface_not_in_the_emitted_text_drops() -> None:
    """the in-text backstop: a surface dict that escapes the token stream (only reachable through
    the unit seam) must still drop instead of shipping an unverifiable triple"""
    out = ChiaScript.gold_relations(
        [rel("Has_value", "A", "B")],
        {"A": "ghost surface one", "B": "ghost surface two"},
        TEXT,
    )

    assert out == []


def test_predicate_mapping_biolsink_and_native() -> None:
    """Subsumes maps to the biolink superclass_of with arg1 as head (arg1 is the superclass),
    Has_temporal maps to the symmetric temporally_related_to, and the has_* family keeps native
    snake_case names through resolve_predicate"""
    example = Chia.run((TEXT, BASE_ENTITIES, [rel("Subsumes", "E1", "E3"), rel("Has_temporal", "E2", "E4"), rel("Has_value", "E1", "E4")]))

    by_name = {relation.name: relation for relation in example.relations}
    assert set(by_name) == {"superclass_of", "temporally_related_to", "has_value"}
    assert by_name["superclass_of"].fields[0].value == "metastatic carcinoid tumors"
    assert by_name["superclass_of"].fields[1].value == "Women"
    assert by_name["temporally_related_to"].fields[0].value == "lanreotide"
    assert by_name["has_value"].fields[0].value == "metastatic carcinoid tumors"


def test_empty_entity_row_ships_text_only() -> None:
    """the measured 3.4% empty-entity rows ship as text-only examples, which the declared
    entities+relations outputs filter drops downstream (US-003)"""
    example = Chia.run((TEXT, [], []))

    assert example.populated() == frozenset()
    assert example.text
