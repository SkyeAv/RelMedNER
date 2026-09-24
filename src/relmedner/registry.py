from __future__ import annotations

from typing import Any

from relmedner.hf_json import HuggingFaceJsonDataStream
from relmedner.hf_parquet import HuggingFaceParquetDataStream
from relmedner.huggingface import HuggingFaceDataStream
from relmedner.local import LocalAvroDataStream, LocalDelimitedDataStream
from relmedner.models import RowFilters
from relmedner.streams import DataStream

SOURCE_REGISTRY: dict[str, type[DataStream]] = {
    HuggingFaceDataStream.SOURCE: HuggingFaceDataStream,
    LocalAvroDataStream.SOURCE: LocalAvroDataStream,
    LocalDelimitedDataStream.SOURCE: LocalDelimitedDataStream,
    HuggingFaceJsonDataStream.SOURCE: HuggingFaceJsonDataStream,
    HuggingFaceParquetDataStream.SOURCE: HuggingFaceParquetDataStream,
}


def build_stream(source: str, payload: tuple[Any, ...], filters: RowFilters | None = None, sample_rate: float = 1.0) -> DataStream:
    """filters and sample_rate ride keyword-only from the (source, payload, filters, sample_rate)
    stream_args envelope; the defaults keep every existing positional 2-arg call valid"""
    return SOURCE_REGISTRY[source](*payload, filters=filters, sample_rate=sample_rate)
