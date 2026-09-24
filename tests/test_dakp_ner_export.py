from __future__ import annotations

from typing import Any

from relmedner.models import Relation, TrainingExample
from relmedner.scripts import DakpNerExportScript
from relmedner.types import Script
from relmedner.utils import ScriptUtils

SCRIPT: DakpNerExportScript = DakpNerExportScript()

# the measured failure mode: DAKP's subject guard is case-insensitive, so the asserted relation
# head ('FLUCONAZOLE') arrives in a case the text never spells ('Fluconazole')
TEXT: str = "Hepatic injury: Fluconazole should be administered with caution to patients with liver dysfunction for 14 days."
CONTEXT_LABELS: list[str] = ["indication", "contraindication", "prevention", "observed_prevention"]


def relation(name: str, head: str, tail: str, *, evidence: str = "asserted", negated: bool = False) -> dict[str, Any]:
    return {"name": name, "fields": [{"name": "head", "value": head}, {"name": "tail", "value": tail}], "negated": negated, "evidence": evidence}


def classification(true_label: Any = "contraindication", **overrides: Any) -> dict[str, Any]:
    """a str true_label wraps into the exporter's one-element list; a list passes through raw so
    the malformed-shape tests can hand in exactly what the avro reader could surface"""
    base: dict[str, Any] = {
        "task": "indication context classification",
        "labels": CONTEXT_LABELS,
        "true_label": [true_label] if isinstance(true_label, str) else true_label,
        "multi_label": False,
        "prompt": None,
        "label_descriptions": None,
    }
    return {**base, **overrides}


def export_record(**overrides: Any) -> dict[str, Any]:
    """one record exactly as dakp_pipeline.ner_export writes it (no weight, no relation description)"""
    base: dict[str, Any] = {
        "text": TEXT,
        "entities": [
            {"label": "Disease", "mentions": ["liver dysfunction"], "description": None},
            {"label": "AnatomicalEntity", "mentions": ["Hepatic"], "description": None},
            {"label": "temporal_interval_qualifier", "mentions": ["14 days"], "description": None},
        ],
        "classifications": [classification()],
        "structures": [],
        "relations": [
            relation("contraindicated_in", "FLUCONAZOLE", "liver dysfunction"),
            relation("temporal_context_qualifier", "liver dysfunction", "14 days", evidence="mined"),
        ],
    }
    return {**base, **overrides}


def run(record: Any) -> TrainingExample:
    return SCRIPT.run((record,))


def fields(rel: Relation) -> dict[str, str]:
    return {field.name: field.value for field in rel.fields}


def test_the_script_self_registers_under_its_declared_name() -> None:
    """the yaml names the script by string; a registry miss would KeyError on the first row"""
    assert isinstance(Script.REGISTRY["DakpNerExportScript"], DakpNerExportScript)


def test_text_ships_verbatim() -> None:
    """every mention and relation value is checked against this exact string, so any rewrite
    of the text would silently invalidate the spans the export measured against it"""
    assert run(export_record()).text == TEXT


def test_a_case_only_relation_head_realigns_to_the_text_surface() -> None:
    """the dominant measured defect (2,136,992 of 2,136,998 out-of-text values): realigning keeps
    the corpus's relation signal instead of dropping most asserted relations"""
    example: TrainingExample = run(export_record())
    asserted: Relation = next(rel for rel in example.relations if rel.name == "contraindicated_in")
    assert fields(asserted) == {"head": "Fluconazole", "tail": "liver dysfunction"}


def test_every_emitted_relation_value_occurs_verbatim_in_the_text() -> None:
    """gliner2's InputExample.validate rejects any relation value that is not an in-text surface"""
    example: TrainingExample = run(export_record())
    assert example.relations
    assert all(field.value in example.text for rel in example.relations for field in rel.fields)


def test_a_relation_value_absent_even_case_insensitively_drops_the_relation() -> None:
    """the 6 measured values with no case-insensitive match cannot be realigned; shipping them
    would fail gliner2 validation, so the relation drops and the rest of the row survives"""
    example: TrainingExample = run(export_record(relations=[relation("contraindicated_in", "ketoconazole", "liver dysfunction")]))
    assert example.relations == []
    assert example.entities


def test_exact_duplicate_relations_collapse_to_one() -> None:
    """30,236 measured duplicates: repeated identical triples would overweight one statement"""
    duplicate: dict[str, Any] = relation("contraindicated_in", "FLUCONAZOLE", "liver dysfunction")
    example: TrainingExample = run(
        export_record(relations=[duplicate, duplicate, relation("contraindicated_in", "Fluconazole", "liver dysfunction")])
    )
    assert len(example.relations) == 1


def test_a_head_equal_to_its_tail_is_a_self_loop_and_drops() -> None:
    """96 measured head == tail loops carry no relational signal"""
    example: TrainingExample = run(export_record(relations=[relation("contraindicated_in", "liver dysfunction", "Liver Dysfunction")]))
    assert example.relations == []


def test_species_context_qualifier_relations_drop() -> None:
    """tablassert DISABLED_EDGE_FIELDS: the repo never emits this slot (gazetteer.DISABLED_QUALIFIERS)"""
    example: TrainingExample = run(
        export_record(relations=[relation("species_context_qualifier", "liver dysfunction", "patients", evidence="mined")])
    )
    assert example.relations == []


def test_negated_and_evidence_ride_through_verbatim() -> None:
    """provenance: asserted drug -> object relations vs mined qualifier attachments must stay
    distinguishable in the avro records"""
    example: TrainingExample = run(export_record())
    by_name: dict[str, Relation] = {rel.name: rel for rel in example.relations}
    assert by_name["contraindicated_in"].evidence == "asserted"
    assert by_name["temporal_context_qualifier"].evidence == "mined"
    assert all(rel.negated is False for rel in example.relations)


def test_relations_carry_their_biolink_slot_description() -> None:
    """descriptions become gliner2 relation_descriptions label prompts; the export ships none"""
    example: TrainingExample = run(export_record())
    for rel in example.relations:
        assert rel.description == ScriptUtils.predicate_description(rel.name)
    assert next(rel for rel in example.relations if rel.name == "contraindicated_in").description


def test_biolink_labels_pass_through_and_qualifier_labels_become_pascal_raw_tails() -> None:
    """temporal_interval_qualifier is not a biolink class; an invented class would train garbage,
    so it rides PascalCase like every other unmapped label"""
    labels: set[str] = {entity.label for entity in run(export_record()).entities}
    assert labels == {"Disease", "AnatomicalEntity", "TemporalIntervalQualifier"}
    assert ScriptUtils.is_biolink_category("Disease")
    assert not ScriptUtils.is_biolink_category("TemporalIntervalQualifier")


def test_a_mention_absent_from_the_text_drops() -> None:
    """entities must satisfy the same in-text contract as relations"""
    example: TrainingExample = run(
        export_record(entities=[{"label": "Disease", "mentions": ["liver dysfunction", "renal failure"], "description": None}])
    )
    assert [entity.mentions for entity in example.entities] == [["liver dysfunction"]]


def test_malformed_entities_drop_without_raising() -> None:
    """skip-don't-coerce: one bad sub-structure must not kill a 187k-row stream"""
    bad: list[Any] = [
        "x",
        {"label": 3, "mentions": ["Hepatic"]},
        {"label": "Disease", "mentions": "liver dysfunction"},
        {"label": " ", "mentions": ["Hepatic"]},
    ]
    assert run(export_record(entities=bad)).entities == []
    assert run(export_record(entities="nope")).entities == []


def test_malformed_relations_drop_without_raising() -> None:
    """every malformed shape the avro reader could surface drops only its own relation"""
    bad: list[Any] = [
        "x",
        {"name": 1, "fields": []},
        {"name": "treats", "fields": []},
        {"name": "treats", "fields": "head"},
        {"name": "treats", "fields": [{"name": "head", "value": 5}]},
        {"name": "treats", "fields": [{"name": "head", "value": "Fluconazole"}], "negated": "no"},
        {"name": "treats", "fields": [{"name": "head", "value": "Fluconazole"}], "evidence": 1},
        {"name": "treats", "fields": ["head"]},
    ]
    assert run(export_record(relations=bad)).relations == []
    assert run(export_record(relations=None)).relations == []


def test_classifications_ride_through_unchanged() -> None:
    """the indication-context task is the export's only classification signal"""
    (kept,) = run(export_record()).classifications
    assert (kept.task, kept.labels, kept.true_label, kept.multi_label, kept.prompt) == (
        "indication context classification",
        CONTEXT_LABELS,
        ["contraindication"],
        False,
        None,
    )


def test_a_classification_whose_true_label_is_outside_its_labels_drops() -> None:
    """a true label outside the choice set is not a supervisable example"""
    assert run(export_record(classifications=[classification(true_label="unknown")])).classifications == []


def test_malformed_classifications_drop_without_raising() -> None:
    bad: list[Any] = [
        "x",
        classification(task=""),
        classification(labels=[]),
        classification(labels=["a", 2]),
        classification(true_label=[]),
        classification(multi_label="no"),
    ]
    assert run(export_record(classifications=bad)).classifications == []
    assert run(export_record(classifications={"task": "x"})).classifications == []


def test_well_formed_label_descriptions_ride_through() -> None:
    described: dict[str, Any] = classification(label_descriptions=[{"key": "indication", "description": "treats"}, {"key": 1}])
    (kept,) = run(export_record(classifications=[described])).classifications
    assert kept.label_descriptions is not None
    assert [(entry.key, entry.description) for entry in kept.label_descriptions] == [("indication", "treats")]


def test_a_zero_span_row_keeps_its_classification() -> None:
    """8,713 measured rows carry only the classification: DAKP keeps them to train abstention,
    and the permitted-shapes contract ships them on the classifications shape"""
    example: TrainingExample = run(export_record(entities=[], relations=[]))
    assert example.populated() == frozenset({"classifications"})


def test_a_non_dict_record_produces_an_empty_example() -> None:
    assert run(["not", "a", "record"]) == TrainingExample(text="")


def test_a_blank_or_missing_text_produces_an_empty_example() -> None:
    assert run(export_record(text="   ")) == TrainingExample(text="")
    assert run(export_record(text=None)) == TrainingExample(text="")


def test_dispatch_routes_through_the_registry_and_keeps_declared_outputs() -> None:
    """the pipeline calls Script.dispatch by name; this is the exact production entry point"""
    outputs, example = Script.dispatch("DakpNerExportScript", (("entities", "classifications", "relations"), (export_record(),)))
    assert outputs == ("entities", "classifications", "relations")
    assert example.populated() == frozenset({"entities", "classifications", "relations"})
