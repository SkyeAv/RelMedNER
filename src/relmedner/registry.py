from __future__ import annotations

from typing import Any

from relmedner.hf_json import HuggingFaceJsonDataStream
from relmedner.huggingface import HuggingFaceDataStream
from relmedner.local import LocalAvroDataStream, LocalDelimitedDataStream
from relmedner.models import RowFilters
from relmedner.streams import DataStream

SOURCE_REGISTRY: dict[str, type[DataStream]] = {
    HuggingFaceDataStream.SOURCE: HuggingFaceDataStream,
    LocalAvroDataStream.SOURCE: LocalAvroDataStream,
    LocalDelimitedDataStream.SOURCE: LocalDelimitedDataStream,
    HuggingFaceJsonDataStream.SOURCE: HuggingFaceJsonDataStream,
}


def build_stream(source: str, payload: tuple[Any, ...], filters: RowFilters | None = None) -> DataStream:
    """filters rides keyword-only from the (source, payload, filters) stream_args envelope; the
    default keeps every existing positional 2-arg call valid"""
    return SOURCE_REGISTRY[source](*payload, filters=filters)
