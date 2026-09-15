from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field
from dataclasses_avroschema.pydantic import AvroBaseModel


class StrictBase(AvroBaseModel):
    model_config: ConfigDict = ConfigDict(frozen=True, extra="forbid")

class YamlIngests(StrictBase):
    datasets: list[Dataset] = Field(...)

class Dataset(StrictBase):
    type: Literal["vocab"] = Field(...)

class HuggingFaceDataset(Dataset):
    source: Literal["hf"] = Field(...)
    dataset: str = Field(...)
    subset: str = Field(...)
    split: str = Field(...)
    column: str = Field(...)
