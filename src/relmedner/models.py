from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from relmedner.enums import ProcessingTypes


class StrictBase(BaseModel):
    model_config: ConfigDict = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    @staticmethod
    def freeze(value: Any) -> Any:
        if isinstance(value, StrictBase):
            return value.to_tuple()
        if isinstance(value, list):
            return tuple(StrictBase.freeze(item) for item in value)
        return value

    def to_tuple(self: Self) -> tuple[Any, ...]:
        return tuple(self.freeze(getattr(self, name)) for name in type(self).model_fields)


class DatasetBase(StrictBase):
    type: ProcessingTypes = Field(...)

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
