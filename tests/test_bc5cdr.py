from __future__ import annotations

from typing import Any

from relmedner.models import Relation, TrainingExample
from relmedner.scripts import Bc5CdrScript
from relmedner.types import Script
from relmedner.utils import ResolvedMention, ScriptUtils

SCRIPT: Bc5CdrScript = Bc5CdrScript()

TITLE: str = "Aspirin treatment"
ABSTRACT: str = "Aspirin is a nonsteroidal anti-inflammatory drug. Asthma severity worsens after acetylsalicylic acid exposure."
TEXT: str = f"{TITLE}\n{ABSTRACT}"

ASPIRIN_MESH: str = "D001241"
ASTHMA_MESH: str = "D003888"


def span(surface: str, text: str = TEXT, offset: int | None = None, length: int | None = None) -> tuple[int, int]:
    """document-absolute [start, start + length) over title + '\\n' + abstract, as the builder writes"""
    start: int = text.index(surface) if offset is None else offset
    return (start, len(surface) if length is None else length)


def annotation(surface: str, mesh: Any = ASPIRIN_MESH, etype: str = "Chemical", text: str = TEXT, **overrides: Any) -> dict[str, Any]:
    start, size = span(surface, text=text)
    base: dict[str, Any] = {"type": etype, "mesh": mesh, "offset": start, "length": size, "text": surface}
    return {**base, **overrides}


def avro_record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pmid": "1000000",
        "title": TITLE,
        "abstract": ABSTRACT,
        "entities": [annotation("Aspirin"), annotation("Asthma", mesh=ASTHMA_MESH, etype="Disease"), annotation("acetylsalicylic acid")],
        "relations": [{"chemical_mesh": ASPIRIN_MESH, "disease_mesh": ASTHMA_MESH}],
    }
    return {**base, **overrides}


def relation_fields(relation: Relation) -> dict[str, str]:
    return {field.name: field.value for field in relation.fields}


def test_the_script_self_registers_under_its_declared_name() -> None:
    """importing the module must install the instance in the shared registry (the ingests.yaml
    entries resolve by this exact NAME key)"""
    assert isinstance(Script.REGISTRY["Bc5CdrScript"], Bc5CdrScript)


def test_every_label_map_target_is_a_biolink_category() -> None:
    """the import-time validate_label_map call already guards this; re-asserted here for drift"""
    assert Bc5CdrScript.NAME == "Bc5CdrScript"
    assert all(ScriptUtils.is_biolink_category(category) for category in Bc5CdrScript.LABEL_MAP.values())


def test_text_is_the_title_a_newline_and_the_abstract() -> None:
    """the builder's document text contract: entity offsets are document-absolute over exactly
    this string, so rewriting it would desync every surviving span"""
    Example: TrainingExample = SCRIPT.run((avro_record(),))

    assert Example.text == f"{TITLE}\n{ABSTRACT}"


def test_a_gold_chemical_groups_under_chemicalentity() -> None:
    Example: TrainingExample = SCRIPT.run((avro_record(),))

    labels: dict[str, list[str]] = {entity.label: entity.mentions for entity in Example.entities}
    assert labels["ChemicalEntity"] == ["Aspirin", "acetylsalicylic acid"]
    assert all(entity.description for entity in Example.entities)


def test_a_gold_disease_groups_under_disease() -> None:
    Example: TrainingExample = SCRIPT.run((avro_record(),))

    labels: dict[str, list[str]] = {entity.label: entity.mentions for entity in Example.entities}
    assert labels["Disease"] == ["Asthma"]


def test_a_normalized_mesh_rides_as_a_mesh_curie_without_fullmap_resolution() -> None:
    """gold annotations are trusted as-is: the annotation's own mesh becomes the curie and the
    origin is fullmap only because a curie exists, never because anything was re-resolved"""
    Resolved: list[ResolvedMention] = SCRIPT.mentions_of(avro_record(), TEXT)

    aspirin: ResolvedMention = next(mention for mention in Resolved if mention.mention == "Aspirin")
    assert aspirin.category == "ChemicalEntity"
    assert aspirin.curie == "MESH:D001241"
    assert aspirin.origin == "fullmap"


def test_the_unnormalized_mesh_sentinel_stays_raw() -> None:
    """measured 76 / 60 / 91 MESH:-1 annotations across train / dev / test: the sentinel never
    becomes a curie, but the span itself still ships under its biolink label"""
    Record: dict[str, Any] = avro_record(entities=[annotation("Aspirin", mesh="-1"), annotation("Asthma", mesh="-1", etype="Disease")])

    Resolved: list[ResolvedMention] = SCRIPT.mentions_of(Record, TEXT)
    assert [(mention.mention, mention.curie, mention.origin) for mention in Resolved] == [
        ("Aspirin", None, "raw"),
        ("Asthma", None, "raw"),
    ]

    Example: TrainingExample = SCRIPT.run((Record,))
    assert sorted(entity.label for entity in Example.entities) == ["ChemicalEntity", "Disease"]


def test_an_offset_mismatch_drops_its_span_and_keeps_the_text() -> None:
    """the builder's census keeps corpus noise in the containers (74 / 43 / 37 slice mismatches
    across train / dev / test); this script drops exactly those instead of shipping spans the
    downstream gliner2 sanitizer would silently discard"""
    Shifted: dict[str, Any] = annotation("Asthma", mesh=ASTHMA_MESH, etype="Disease", offset=span("Asthma")[0] + 1)
    Example: TrainingExample = SCRIPT.run((avro_record(entities=[Shifted]),))

    assert Example.text == TEXT
    assert Example.entities == []
    assert Example.relations == []


def test_an_out_of_bounds_offset_drops_its_span() -> None:
    Past: dict[str, Any] = annotation("Asthma", mesh=ASTHMA_MESH, etype="Disease", offset=len(TEXT) + 7, length=6)
    Example: TrainingExample = SCRIPT.run((avro_record(entities=[Past]),))

    assert Example.entities == []
    assert Example.relations == []


def test_an_unknown_annotation_type_drops_its_span() -> None:
    Gene: dict[str, Any] = annotation("Aspirin", etype="Gene")
    Example: TrainingExample = SCRIPT.run((avro_record(entities=[Gene]),))

    assert Example.entities == []


def test_malformed_annotations_drop_without_raising() -> None:
    """bool offsets, string lengths, empty meshes, non-dict entries, and non-list tables all
    drop quietly; a gold table is never coerced into spans it does not actually contain"""
    Bad: list[Any] = [
        annotation("Aspirin", offset=True),
        annotation("Aspirin", length="7"),
        annotation("Aspirin", mesh=None),
        {"type": "Chemical", "mesh": ASPIRIN_MESH, "offset": 0, "length": 7, "text": ""},
        "not an annotation dict",
    ]
    assert SCRIPT.run((avro_record(entities=Bad),)).entities == []
    assert SCRIPT.mentions_of(avro_record(entities="not a list"), TEXT) == []


def test_a_non_dict_record_produces_an_empty_example() -> None:
    Example: TrainingExample = SCRIPT.run(("not a record",))

    assert Example.text == ""
    assert Example.populated() == frozenset()


def test_a_record_without_text_produces_an_empty_example() -> None:
    """None title/abstract coerce to empty strings, leaving only the contract newline; the
    whitespace-only text ships empty like any other label-less row"""
    Blank: dict[str, Any] = avro_record(title=None, abstract=None)
    Example: TrainingExample = SCRIPT.run((Blank,))

    assert Example.text == ""


def test_same_mesh_annotations_cross_product_and_dedupe_into_relations() -> None:
    """two chemical annotations share ASPIRIN_MESH, so the single CID pair expands to the
    cross-product of the surviving surfaces, and the duplicated record relation dedupes"""
    Duplicated: list[dict[str, str]] = [{"chemical_mesh": ASPIRIN_MESH, "disease_mesh": ASTHMA_MESH}] * 2
    Example: TrainingExample = SCRIPT.run((avro_record(relations=Duplicated),))

    assert len(Example.relations) == 2
    Pairs: set[tuple[str, str]] = {(fields["head"], fields["tail"]) for fields in (relation_fields(relation) for relation in Example.relations)}
    assert Pairs == {("Aspirin", "Asthma"), ("acetylsalicylic acid", "Asthma")}
    assert all(relation.name == "causes" for relation in Example.relations)


def test_a_relation_whose_mesh_never_survives_is_unresolved_and_drops() -> None:
    """the mismatched annotation's mesh resolves to nothing, so the CID pair that names it drops
    rather than shipping a relation whose surface is absent from the text"""
    Shifted: dict[str, Any] = annotation("Asthma", mesh=ASTHMA_MESH, etype="Disease", offset=span("Asthma")[0] + 1)
    Example: TrainingExample = SCRIPT.run(
        (avro_record(entities=[Shifted], relations=[{"chemical_mesh": "NOSUCHMESH", "disease_mesh": ASTHMA_MESH}]),)
    )

    assert Example.relations == []


def test_a_self_loop_relation_is_dropped() -> None:
    """a chemical mesh equal to the disease mesh cross-products one surface set against itself;
    the head == tail pairs that follow are dropped, never shipped"""
    Example: TrainingExample = SCRIPT.run(
        (avro_record(entities=[annotation("Aspirin")], relations=[{"chemical_mesh": ASPIRIN_MESH, "disease_mesh": ASPIRIN_MESH}]),)
    )

    assert Example.relations == []


def test_a_malformed_relation_table_drops_without_raising() -> None:
    Bad: list[Any] = ["not a relation", {"chemical_mesh": None, "disease_mesh": ASTHMA_MESH}, {}]
    Example: TrainingExample = SCRIPT.run((avro_record(relations=Bad),))

    assert Example.relations == []
    assert SCRIPT.relations_of(avro_record(relations="not a list"), []) == []


def test_every_relation_surface_occurs_in_the_text() -> None:
    """gliner2's InputExample.validate requires every relation value to occur in the text; gold
    spans are char-exact against title + '\\n' + abstract, so this holds by construction"""
    Example: TrainingExample = SCRIPT.run((avro_record(),))

    assert Example.relations
    for relation in Example.relations:
        for value in relation_fields(relation).values():
            assert value in Example.text


def test_relations_carry_the_biolink_causes_description() -> None:
    Example: TrainingExample = SCRIPT.run((avro_record(),))

    assert Example.relations[0].description == ScriptUtils.predicate_description("causes")
    assert Example.relations[0].description
    assert Example.relations[0].negated is False
    assert Example.relations[0].evidence == "asserted"


def test_dispatch_routes_through_the_registry_and_keeps_declared_outputs() -> None:
    Outputs, Example = Script.dispatch("Bc5CdrScript", (("entities", "relations"), (avro_record(),)))

    assert Outputs == ("entities", "relations")
    assert Example.entities
    assert Example.relations


def test_a_batch_of_rows_yields_nonzero() -> None:
    """a malformed row degrades to the empty example, but real containers stream 500 documents
    per split; a sample must keep yielding populated examples"""
    Rows: list[Any] = [
        avro_record(),
        avro_record(entities=[]),
        avro_record(title="", abstract=""),
        "not a record",
        avro_record(relations=[{"chemical_mesh": "NOSUCHMESH", "disease_mesh": ASTHMA_MESH}]),
    ]
    Dispatched: list[TrainingExample] = [Script.dispatch("Bc5CdrScript", (("entities", "relations"), (row,)))[1] for row in Rows]

    assert len(Dispatched) == len(Rows)
    assert sum(1 for example in Dispatched if example.populated()) >= 2
