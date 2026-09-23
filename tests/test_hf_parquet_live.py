from __future__ import annotations

import os

import pytest

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams real bigbio/gad rows from the hub's parquet conversion; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


@requires_live_hf
def test_a_live_parquet_conversion_row_streams_through_the_declared_shape() -> None:
    """the hf_parquet route must work against the REAL hub inventory and conversion branch, not
    just the patched unit path: the datasets-server listing must return shards for the declared
    (config, split), the parquet builder must accept their resolve URLs, and the first streamed
    row must decode into the declared columns_out projection with the measured value shapes
    (text str, labels a one-element list of "0"/"1"). Gated behind RELMEDNER_LIVE_HF=1 and
    wenceslaus-only: the download must never happen in the default offline suite.

    Everything that transitively imports `datasets` is imported INSIDE the gated body so plain
    `pytest --collect-only` stays offline-safe.
    """
    from relmedner.hf_parquet import HuggingFaceParquetDataStream, parquet_shard_urls

    shards: list[str] = parquet_shard_urls("bigbio/gad", "gad_blurb_bigbio_text", "train")
    assert len(shards) >= 1
    assert all(url.startswith("https://huggingface.co/datasets/bigbio/gad/") for url in shards)

    Stream: HuggingFaceParquetDataStream = HuggingFaceParquetDataStream(
        ("script", "GadBlurbScript", ("classifications",)),
        1.0,
        "bigbio/gad",
        "gad_blurb_bigbio_text",
        "train",
        None,
        ("text", "labels"),
    )
    Name: str
    Values: tuple[object, ...]
    Name, (_task, Values) = next(Stream.rows())

    assert Name == "bigbio/gad"
    text, labels = Values
    assert isinstance(text, str) and text
    assert isinstance(labels, list) and len(labels) == 1
    assert labels[0] in ("0", "1")
    assert Stream.stats.rows_in >= Stream.stats.rows_out == 1
