from __future__ import annotations

from typing import Any

from relmedner.huggingface import HuggingFaceDataStream
from relmedner.streams import DataStream

SOURCE_REGISTRY: dict[str, type[DataStream]] = {
    HuggingFaceDataStream.SOURCE: HuggingFaceDataStream,
}


def build_stream(source: str, payload: tuple[Any, ...]) -> DataStream:
    return SOURCE_REGISTRY[source](*payload)
