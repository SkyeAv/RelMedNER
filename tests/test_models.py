from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from relmedner.enums import OutputShapes, ProcessingTypes
from relmedner.models import (
    ChoiceField,
    Classification,
    Dataset,
    Description,
    Entity,
    HuggingFaceJsonDataset,
    Relation,
    RelationField,
    ScriptTask,
    StrictBase,
    Structure,
    StructureField,
    TrainingExample,
)
from relmedner.pipeline import matches_declared_outputs
from relmedner.utils import ScriptUtils


def every_strict_subclass(base: type[StrictBase] = StrictBase) -> list[type[StrictBase]]:
    Subclasses: list[type[StrictBase]] = []
    for Subclass in base.__subclasses__():
        Subclasses.append(Subclass)
        Subclasses.extend(every_strict_subclass(Subclass))
    return Subclasses


@pytest.mark.parametrize("model", every_strict_subclass(), ids=lambda model: model.__name__)
def test_every_strict_model_generates_an_avro_schema(model: type[StrictBase]) -> None:
    Schema: dict[str, Any] = model.avro_schema_to_python()

    assert Schema["type"] == "record"
    assert Schema["name"] == model.__name__


def test_the_structure_value_union_carries_all_three_arms() -> None:
    Schema: dict[str, Any] = StructureField.avro_schema_to_python()
    Value: Any = next(field["type"] for field in Schema["fields"] if field["name"] == "value")
    Arms: list[Any] = [arm if isinstance(arm, str) else arm["type"] for arm in Value]

    assert Arms == ["string", "array", "record"]


def test_populated_reports_only_the_fields_that_carry_tasks() -> None:
    Example: TrainingExample = TrainingExample(
        text="Alice manages engineering team.",
        entities=[Entity(label="person", mentions=["Alice"])],
        relations=[
            Relation(
                name="manages",
                fields=[RelationField(name="head", value="Alice"), RelationField(name="tail", value="engineering team")],
            )
        ],
    )

    assert Example.populated() == frozenset({"entities", "relations"})


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        (("entities",), True),
        (("relations",), False),
        (("entities", "relations"), True),
        (("classifications",), False),
        (("entities", "classifications", "structures", "relations"), True),
    ],
)
def test_matches_declared_outputs_keeps_any_nonempty_subset_of_the_declared_shapes(outputs: tuple[str, ...], expected: bool) -> None:
    """permitted-shapes contract: one script emits different shapes per row family, so a row ships when
    it produced something and everything it produced was declared"""
    Example: TrainingExample = TrainingExample(text="Alice", entities=[Entity(label="person", mentions=["Alice"])])

    assert matches_declared_outputs((outputs, Example)) is expected


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        (("entities",), True),
        (("entities", "relations"), True),
        (("relations",), True),
        (("classifications",), False),
    ],
)
def test_matches_declared_outputs_keeps_extra_populated_shapes(outputs: tuple[str, ...], expected: bool) -> None:
    """subset semantics: a declared [entities] ingest keeps rows that also extracted relations"""
    Example: TrainingExample = TrainingExample(
        text="Aspirin treats headache",
        entities=[Entity(label="Drug", mentions=["Aspirin"])],
        relations=[Relation(name="treats", fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="headache")])],
    )

    assert matches_declared_outputs((outputs, Example)) is expected


def test_an_example_with_no_tasks_never_matches_a_declaration() -> None:
    assert matches_declared_outputs((("entities",), TrainingExample(text="nothing here"))) is False


def test_to_output_emits_descriptions_only_when_they_are_declared() -> None:
    Bare: dict[str, Any] = TrainingExample(text="Alice", entities=[Entity(label="person", mentions=["Alice"])]).to_output()
    Described: dict[str, Any] = TrainingExample(
        text="Alice", entities=[Entity(label="person", mentions=["Alice"], description="Names of people")]
    ).to_output()

    assert "entity_descriptions" not in Bare["output"]
    assert Described["output"]["entity_descriptions"] == {"person": "Names of people"}


def test_to_output_emits_relation_descriptions_only_when_they_are_declared() -> None:
    """relation_descriptions mirrors entity_descriptions: emitted only when a relation carries one"""
    Bare: TrainingExample = TrainingExample(
        text="Aspirin treats headache",
        relations=[Relation(name="treats", fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="headache")])],
    )
    Described: TrainingExample = TrainingExample(
        text="Aspirin treats headache",
        relations=[
            Relation(
                name="treats",
                fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="headache")],
                description="holds between an intervention and a condition it ameliorates",
            )
        ],
    )

    assert "relation_descriptions" not in Bare.to_output()["output"]
    assert Described.to_output()["output"]["relation_descriptions"] == {"treats": "holds between an intervention and a condition it ameliorates"}


def test_predicate_description_reads_biolink_slots_for_every_declared_predicate() -> None:
    """the biolink slot yaml covers all 23 gazetteer predicates; the lookup normalizes underscores"""
    from relmedner.gazetteer import PREDICATE_TRIGGERS

    missing: list[str] = [predicate for predicate in PREDICATE_TRIGGERS if ScriptUtils.predicate_description(predicate) is None]
    assert not missing, missing
    assert ScriptUtils.predicate_description("superclass_of") is not None
    assert ScriptUtils.predicate_description("not_a_biolink_predicate") is None


def test_choice_fields_round_trip_into_the_gliner_choice_shape() -> None:
    Example: TrainingExample = TrainingExample(
        text="Book a single room.",
        structures=[
            Structure(
                name="booking",
                fields=[StructureField(name="room_type", value=ChoiceField(value="single", choices=["single", "double"]))],
            )
        ],
    )

    assert Example.to_output()["output"]["json_structures"] == [{"booking": {"room_type": {"value": "single", "choices": ["single", "double"]}}}]


def test_multi_label_and_label_descriptions_survive_projection() -> None:
    Example: TrainingExample = TrainingExample(
        text="Great camera, poor battery.",
        classifications=[
            Classification(
                task="aspects",
                labels=["camera", "battery"],
                true_label=["camera", "battery"],
                multi_label=True,
                label_descriptions=[Description(key="camera", description="Photo quality")],
            )
        ],
    )
    Projected: dict[str, Any] = Example.to_output()["output"]["classifications"][0]

    assert Projected["multi_label"] is True
    assert Projected["true_label"] == ["camera", "battery"]
    assert Projected["label_descriptions"] == {"camera": "Photo quality"}


def test_relations_carry_provenance_defaults() -> None:
    Relation_: Relation = Relation(
        name="treats",
        fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="headache")],
    )

    assert Relation_.negated is False and Relation_.evidence == "asserted"


def test_relation_provenance_survives_avro_but_stays_out_of_the_gliner_projection() -> None:
    """gliner2's Relation(**fields) swallows extra keys (dropping head/tail) and validates
    every string value as a mention -- so evidence/negated ride in avro records only"""
    import io

    from fastavro import reader, writer

    Mined: Relation = Relation(
        name="treats",
        fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="headache")],
        negated=False,
        evidence="distant",
    )
    Example: TrainingExample = TrainingExample(text="Aspirin treats headache.", relations=[Mined])

    Buffer: io.BytesIO = io.BytesIO()
    writer(Buffer, Example.avro_schema_to_python(), [Example.asdict()])
    Buffer.seek(0)
    Restored: TrainingExample = TrainingExample(**next(iter(reader(Buffer))))

    assert Restored == Example and Restored.relations[0].evidence == "distant"
    assert Restored.to_output()["output"]["relations"] == [{"treats": {"head": "Aspirin", "tail": "headache"}}]


def test_huggingface_json_dataset_pins_the_positional_payload_contract() -> None:
    """registry.build_stream splats to_tuple positionally into HuggingFaceJsonDataStream.__init__, so the
    model field order minus source IS the stream constructor order"""
    Entry: HuggingFaceJsonDataset = HuggingFaceJsonDataset(
        task=ScriptTask(type=ProcessingTypes.SCRIPT, name="PubmedAbstractsScript", outputs=[OutputShapes.ENTITIES]),
        source="hf_json",
        dataset="knowledgator/PubMedAbstractsNER",
        file="train.json",
        split="train",
        match_on=None,
        columns_out=["tokenized_text", "ner"],
    )

    assert list(HuggingFaceJsonDataset.model_fields) == ["task", "source", "dataset", "file", "split", "match_on", "columns_out"]
    assert Entry.to_tuple() == (
        "hf_json",
        (
            ("script", "PubmedAbstractsScript", ("entities",)),
            "knowledgator/PubMedAbstractsNER",
            "train.json",
            "train",
            None,
            ("tokenized_text", "ner"),
        ),
    )


def test_the_dataset_discriminated_union_accepts_the_hf_json_variant() -> None:
    """the annotated Dataset union is what ingests.yaml validation dispatches on; a missing arm would
    reject the new source at yaml parse time"""
    Parsed: HuggingFaceJsonDataset = TypeAdapter(Dataset).validate_python(
        {
            "task": {"type": "script", "name": "PubmedAbstractsScript", "outputs": ["entities"]},
            "source": "hf_json",
            "dataset": "knowledgator/PubMedAbstractsNER",
            "file": "train.json",
            "split": "train",
            "match_on": None,
            "columns_out": ["tokenized_text", "ner"],
        }
    )

    assert isinstance(Parsed, HuggingFaceJsonDataset)


def test_huggingface_json_dataset_rejects_an_empty_file_name() -> None:
    """file rides into an hf://datasets/{dataset}/{file} URL; an empty name would only fail deep inside
    the hub client, so the model gate fails loudly instead"""
    with pytest.raises(ValidationError):
        HuggingFaceJsonDataset(
            task=ScriptTask(type=ProcessingTypes.SCRIPT, name="PubmedAbstractsScript", outputs=[OutputShapes.ENTITIES]),
            source="hf_json",
            dataset="knowledgator/PubMedAbstractsNER",
            file="",
            split="train",
            match_on=None,
            columns_out=["tokenized_text"],
        )
