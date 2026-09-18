from __future__ import annotations

import importlib.util
import io
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastavro import reader, writer

from relmedner.models import (
    ChoiceField,
    Classification,
    Description,
    Entity,
    Relation,
    RelationField,
    Structure,
    StructureField,
    TrainingExample,
)
from relmedner.scripts import GlinerBiomedScript
from relmedner.utils import ResolvedMention, ScriptUtils

GLINER_DATA_MODULE: str = "gliner2_training_data"


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


ENTITY_EXAMPLE: TrainingExample = TrainingExample(
    text="Dr. Sarah Johnson prescribed Metformin 500mg daily for diabetes.",
    entities=[
        Entity(label="person", mentions=["Dr. Sarah Johnson"], description="Names of people"),
        Entity(label="medication", mentions=["Metformin"]),
        Entity(label="condition", mentions=["diabetes"]),
    ],
)

CLASSIFICATION_EXAMPLE: TrainingExample = TrainingExample(
    text="This smartphone has an amazing camera but the battery life is poor.",
    classifications=[
        Classification(
            task="product_aspects",
            labels=["camera", "battery", "screen"],
            true_label=["camera", "battery"],
            multi_label=True,
            prompt="Which aspects are discussed?",
            label_descriptions=[Description(key="camera", description="Photo quality")],
        )
    ],
)

STRUCTURE_EXAMPLE: TrainingExample = TrainingExample(
    text="Book a single room at Grand Hotel for 2 nights with breakfast included.",
    structures=[
        Structure(
            name="booking",
            fields=[
                StructureField(name="hotel", value="Grand Hotel", description="Name of the hotel"),
                StructureField(name="nights", value="2"),
                StructureField(name="room_type", value=ChoiceField(value="single", choices=["single", "double", "suite"])),
            ],
        )
    ],
)

RELATION_EXAMPLE: TrainingExample = TrainingExample(
    text="Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne.",
    entities=[Entity(label="person", mentions=["Elon Musk"]), Entity(label="organization", mentions=["SpaceX"])],
    relations=[
        Relation(name="founded", fields=[RelationField(name="head", value="Elon Musk"), RelationField(name="tail", value="SpaceX")]),
        Relation(name="located_in", fields=[RelationField(name="head", value="SpaceX"), RelationField(name="tail", value="Hawthorne")]),
    ],
)

EXAMPLES: list[TrainingExample] = [ENTITY_EXAMPLE, CLASSIFICATION_EXAMPLE, STRUCTURE_EXAMPLE, RELATION_EXAMPLE]
EXAMPLE_IDS: list[str] = ["entities", "classifications", "structures", "relations"]


DERIVED_KEYS: frozenset[str] = frozenset({"record_metadata"})


@pytest.mark.parametrize("example", EXAMPLES, ids=EXAMPLE_IDS)
def test_our_output_round_trips_through_the_real_gliner_types(example: TrainingExample) -> None:
    """gliner2 re-emits everything we declare -- it only adds keys it derives itself"""

    GlinerData: ModuleType = load_gliner_data()
    Output: dict[str, Any] = example.to_output()
    Restored: dict[str, Any] = GlinerData.InputExample.from_dict(Output).to_dict()

    assert Restored["input"] == Output["input"]
    assert {key: value for key, value in Restored["output"].items() if key not in DERIVED_KEYS} == Output["output"]


def test_record_metadata_is_the_only_key_gliner_adds_to_our_output() -> None:
    """locks the derived-key allowance so a future gliner2 bump cannot widen it unnoticed"""

    GlinerData: ModuleType = load_gliner_data()
    Added: set[str] = set()
    for example in EXAMPLES:
        Output: dict[str, Any] = example.to_output()
        Added |= set(GlinerData.InputExample.from_dict(Output).to_dict()["output"]) - set(Output["output"])

    assert Added == {"record_metadata"}


def test_the_derived_record_metadata_anchors_on_our_first_declared_field() -> None:
    GlinerData: ModuleType = load_gliner_data()
    Restored: dict[str, Any] = GlinerData.InputExample.from_dict(STRUCTURE_EXAMPLE.to_output()).to_dict()

    assert Restored["output"]["record_metadata"] == {"booking": {"mode": "natural", "anchor": "hotel"}}


@pytest.mark.parametrize("example", EXAMPLES, ids=EXAMPLE_IDS)
def test_gliner_strict_validation_accepts_our_output(example: TrainingExample) -> None:
    GlinerData: ModuleType = load_gliner_data()
    Parsed: Any = GlinerData.InputExample.from_dict(example.to_output())

    assert Parsed.validate() == []


def test_a_dataset_of_every_example_validates_without_raising() -> None:
    GlinerData: ModuleType = load_gliner_data()
    Dataset: Any = GlinerData.TrainingDataset([GlinerData.InputExample.from_dict(example.to_output()) for example in EXAMPLES])
    Report: dict[str, Any] = Dataset.validate()

    assert Report["invalid"] == 0
    assert Report["valid"] == len(EXAMPLES)
    assert Dataset.validate_relation_consistency() == []


def test_an_example_with_a_mention_outside_the_text_is_rejected_by_gliner() -> None:
    GlinerData: ModuleType = load_gliner_data()
    Invalid: TrainingExample = TrainingExample(text="Alice manages the team.", entities=[Entity(label="person", mentions=["Bob"])])
    Parsed: Any = GlinerData.InputExample.from_dict(Invalid.to_output())

    assert Parsed.validate() != []


def test_gazetteer_shaped_script_relations_validate_through_real_gliner(monkeypatch: pytest.MonkeyPatch) -> None:
    """US-003's GlinerBiomedScript relation wiring must emit output the real gliner2 types accept"""

    def fake_resolve(spans: list[tuple[str, str]]) -> list[ResolvedMention]:
        return [
            ResolvedMention(mention="Aspirin", category="Drug", origin="fallback"),
            ResolvedMention(mention="migraine", category="Disease", origin="fallback"),
        ]

    monkeypatch.setattr(ScriptUtils, "resolve_mentions", staticmethod(fake_resolve))
    Tokens: list[str] = ["Aspirin", "is", "used", "to", "treat", "migraine"]
    Example: TrainingExample = GlinerBiomedScript().run((Tokens, [[0, 0, "Drug"], [5, 5, "Condition"]]))

    assert Example.relations == [
        Relation(name="treats", fields=[RelationField(name="head", value="Aspirin"), RelationField(name="tail", value="migraine")])
    ]
    GlinerData: ModuleType = load_gliner_data()
    Parsed: Any = GlinerData.InputExample.from_dict(Example.to_output())

    assert Parsed.validate() == []


@pytest.mark.parametrize("example", EXAMPLES, ids=EXAMPLE_IDS)
def test_examples_round_trip_through_avro(example: TrainingExample) -> None:
    Schema: dict[str, Any] = TrainingExample.avro_schema_to_python()
    Buffer: io.BytesIO = io.BytesIO()
    writer(Buffer, Schema, [example.asdict()])
    Buffer.seek(0)
    Restored: list[dict[str, Any]] = list(reader(Buffer))

    assert len(Restored) == 1
    assert TrainingExample(**Restored[0]) == example


def test_the_structure_union_survives_avro_for_every_arm() -> None:
    Example: TrainingExample = TrainingExample(
        text="iPhone 15 costs $999 in blue, black and white.",
        structures=[
            Structure(
                name="product",
                fields=[
                    StructureField(name="name", value="iPhone 15"),
                    StructureField(name="colors", value=["blue", "black", "white"]),
                    StructureField(name="tier", value=ChoiceField(value="pro", choices=["base", "pro"])),
                ],
            )
        ],
    )
    Schema: dict[str, Any] = TrainingExample.avro_schema_to_python()
    Buffer: io.BytesIO = io.BytesIO()
    writer(Buffer, Schema, [Example.asdict()])
    Buffer.seek(0)
    Restored: TrainingExample = TrainingExample(**next(iter(reader(Buffer))))

    assert Restored == Example
    assert [type(field.value) for field in Restored.structures[0].fields] == [str, list, ChoiceField]


def test_the_generated_schema_matches_its_snapshot() -> None:
    Schema: dict[str, Any] = TrainingExample.avro_schema_to_python()
    Fields: list[str] = [field["name"] for field in Schema["fields"]]

    assert Schema["type"] == "record"
    assert Schema["name"] == "TrainingExample"
    assert Schema["namespace"] == "relmedner.ingests"
    assert Fields == ["text", "entities", "classifications", "structures", "relations"]
