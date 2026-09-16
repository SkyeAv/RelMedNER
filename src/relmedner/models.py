from __future__ import annotations

from relmedner.enums import ProcessingTypes

from typing import Literal, Union, Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field


class StrictBase(BaseModel):
    model_config: ConfigDict = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)


class DatasetBase(StrictBase):
    type: ProcessingTypes = Field(...)


class MatchOn(StrictBase):
    column: str = Field(...)
    values: list[str] = Field(...)


class HuggingFaceDataset(DatasetBase):
    source: Literal["hf"] = Field(...)
    dataset: str = Field(...)
    subset: Optional[str] = Field(None)
    split: Optional[str] = Field(None)
    match_on: Optional[list[MatchOn]] = Field(None)
    columns_out: list[str] = Field(...)


class LocalDataset(DatasetBase):
    """placeholder to get the annotated Dataset type to work"""

    source: Literal["local"] = Field(...)
    path: str = Field(...)


Dataset: Annotated = Annotated[
    Union[HuggingFaceDataset, LocalDataset],
    Field(discriminator="source"),
]


class YamlIngests(StrictBase):
    datasets: list[Dataset] = Field(...)
