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


def build_stream(
    source: str,
    payload: tuple[Any, ...],
    filters: RowFilters | None = None,
    sample_rate: float = 1.0,
    read_shards: int = 1,
    shard_index: int = 0,
) -> DataStream:
    """filters and sample_rate ride keyword-only from the stream_args envelope; read_shards
    rides positionally as the envelope's fifth slot (to_stream_args splat callers) and
    shard_index as the sixth, so the defaults keep every existing call valid. read_shards
    over 1 raises here for sources whose DataStream does not shard its read (the base
    __init__ owns the guard), so a mis-declared source fails at build time, never mid-run"""
    return SOURCE_REGISTRY[source](*payload, filters=filters, sample_rate=sample_rate, read_shards=read_shards, shard_index=shard_index)
