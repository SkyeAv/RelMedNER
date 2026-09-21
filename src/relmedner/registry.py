from __future__ import annotations

from typing import Any

from relmedner.hf_json import HuggingFaceJsonDataStream
from relmedner.huggingface import HuggingFaceDataStream
from relmedner.local import LocalAvroDataStream
from relmedner.streams import DataStream

SOURCE_REGISTRY: dict[str, type[DataStream]] = {
    HuggingFaceDataStream.SOURCE: HuggingFaceDataStream,
    LocalAvroDataStream.SOURCE: LocalAvroDataStream,
    HuggingFaceJsonDataStream.SOURCE: HuggingFaceJsonDataStream,
}


def build_stream(source: str, payload: tuple[Any, ...]) -> DataStream:
    return SOURCE_REGISTRY[source](*payload)
