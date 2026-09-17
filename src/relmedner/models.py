from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from dataclasses_avroschema.pydantic import AvroBaseModel
from pydantic import ConfigDict, Field

from relmedner.constants import DEFAULT_OUTPUT, TEST_ROW_LIMIT
from relmedner.enums import OutputShapes, ProcessingTypes


class StrictBase(AvroBaseModel):
    model_config: ConfigDict = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    class Meta:
        namespace: str = "relmedner.ingests"

    @staticmethod
    def freeze(value: Any) -> Any:
        if isinstance(value, StrictBase):
            return value.to_tuple()
        if isinstance(value, list):
            return tuple(StrictBase.freeze(item) for item in value)
        return value

    def to_tuple(self: Self) -> tuple[Any, ...]:
        return tuple(self.freeze(getattr(self, name)) for name in type(self).model_fields)


class RunConfig(StrictBase):
    sample_limit: int | None = Field(None)
    output: str = Field(DEFAULT_OUTPUT)

    @classmethod
    def from_flags(cls, test_run: bool, output: str = DEFAULT_OUTPUT) -> Self:
        return cls(sample_limit=TEST_ROW_LIMIT if test_run else None, output=output)


class TaskBase(StrictBase):
    type: ProcessingTypes = Field(...)


class ScriptTask(TaskBase):
    type: Literal[ProcessingTypes.SCRIPT] = Field(...)
    name: str = Field(...)
    outputs: list[OutputShapes] = Field(..., min_length=1)


class BabelTask(TaskBase):
    """placeholder to get the annotated Task type to work"""

    type: Literal[ProcessingTypes.BABEL] = Field(...)


Task: Annotated = Annotated[
    ScriptTask | BabelTask,
    Field(discriminator="type"),
]


class DatasetBase(StrictBase):
    task: Task = Field(...)

    def to_tuple(self: Self) -> tuple[str, tuple[Any, ...]]:
        return (self.source, tuple(self.freeze(getattr(self, name)) for name in type(self).model_fields if name != "source"))


class MatchOn(StrictBase):
    column: str = Field(...)
    values: list[str] = Field(...)


class HuggingFaceDataset(DatasetBase):
    source: Literal["hf"] = Field(...)
    dataset: str = Field(...)
    subset: str | None = Field(None)
    split: str | None = Field(None)
    match_on: list[MatchOn] | None = Field(None)
    columns_out: list[str] = Field(...)


class LocalDataset(DatasetBase):
    """placeholder to get the annotated Dataset type to work"""

    source: Literal["local"] = Field(...)
    path: str = Field(...)


Dataset: Annotated = Annotated[
    HuggingFaceDataset | LocalDataset,
    Field(discriminator="source"),
]


class YamlIngests(StrictBase):
    datasets: list[Dataset] = Field(...)

    def generate_tuples(self: Self) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        return tuple(dataset.to_tuple() for dataset in self.datasets)


class Entity(StrictBase):
    label: str = Field(...)
    mentions: list[str] = Field(...)
    description: str | None = Field(None)


class Description(StrictBase):
    key: str = Field(...)
    description: str = Field(...)


class ChoiceField(StrictBase):
    value: str = Field(...)
    choices: list[str] = Field(...)

    def to_output(self: Self) -> dict[str, Any]:
        return {"value": self.value, "choices": self.choices}


class StructureField(StrictBase):
    name: str = Field(...)
    value: str | list[str] | ChoiceField = Field(...)
    description: str | None = Field(None)

    def to_value(self: Self) -> Any:
        return self.value.to_output() if isinstance(self.value, ChoiceField) else self.value


class Structure(StrictBase):
    name: str = Field(...)
    fields: list[StructureField] = Field(...)


class Classification(StrictBase):
    task: str = Field(...)
    labels: list[str] = Field(...)
    true_label: list[str] = Field(..., min_length=1)
    multi_label: bool = Field(False)
    prompt: str | None = Field(None)
    label_descriptions: list[Description] | None = Field(None)


class RelationField(StrictBase):
    name: str = Field(...)
    value: str = Field(...)


class Relation(StrictBase):
    name: str = Field(...)
    fields: list[RelationField] = Field(...)


def describe(descriptions: list[Description] | None) -> dict[str, str]:
    return {entry.key: entry.description for entry in descriptions or []}


class TrainingExample(StrictBase):
    text: str = Field(...)
    entities: list[Entity] = Field(default_factory=list)
    classifications: list[Classification] = Field(default_factory=list)
    structures: list[Structure] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    def populated(self: Self) -> frozenset[str]:
        return frozenset(shape for shape in OutputShapes if getattr(self, shape))

    def entities_out(self: Self) -> dict[str, Any]:
        described: list[Description] = [
            Description(key=entity.label, description=entity.description) for entity in self.entities if entity.description
        ]
        output: dict[str, Any] = {"entities": {entity.label: entity.mentions for entity in self.entities}}
        return output | ({"entity_descriptions": describe(described)} if described else {})

    def classifications_out(self: Self) -> dict[str, Any]:
        return {
            "classifications": [
                {"task": task.task, "labels": task.labels, "true_label": task.true_label}
                | ({"multi_label": True} if task.multi_label else {})
                | ({"prompt": task.prompt} if task.prompt else {})
                | ({"label_descriptions": describe(task.label_descriptions)} if task.label_descriptions else {})
                for task in self.classifications
            ]
        }

    def structures_out(self: Self) -> dict[str, Any]:
        described: dict[str, dict[str, str]] = {
            structure.name: describe([Description(key=field.name, description=field.description) for field in structure.fields if field.description])
            for structure in self.structures
            if any(field.description for field in structure.fields)
        }
        output: dict[str, Any] = {
            "json_structures": [{structure.name: {field.name: field.to_value() for field in structure.fields}} for structure in self.structures]
        }
        return output | ({"json_descriptions": described} if described else {})

    def relations_out(self: Self) -> dict[str, Any]:
        return {"relations": [{relation.name: {field.name: field.value for field in relation.fields}} for relation in self.relations]}

    def to_output(self: Self) -> dict[str, Any]:
        populated: frozenset[str] = self.populated()
        output: dict[str, Any] = {}
        for shape in OutputShapes:
            if shape in populated:
                output |= getattr(self, f"{shape}_out")()
        return {"input": self.text, "output": output}
