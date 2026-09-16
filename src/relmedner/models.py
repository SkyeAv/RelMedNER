from __future__ import annotations

from typing import Literal, Union, Annotated, Optional
from pathlib import Path

from pydantic import ConfigDict, Field
from dataclasses_avroschema.pydantic import AvroBaseModel


class StrictBase(AvroBaseModel):
    model_config: ConfigDict = ConfigDict(frozen=True, extra="forbid")

    class Meta:
        namespace: str = "relmedner.ingests"


class DatasetBase(StrictBase):
    type: Literal["vocab"] = Field(...)


class HuggingFaceDataset(DatasetBase):
    source: Literal["hf"] = Field(...)
    dataset: str = Field(...)
    subset: Optional[str] = Field(None)
    split: Optional[str] = Field(None)
    column: Optional[str] = Field(None)


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
