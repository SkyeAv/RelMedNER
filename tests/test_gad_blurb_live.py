from __future__ import annotations

import os
from typing import Any

import pytest

from relmedner.models import TrainingExample, YamlIngests
from relmedner.streams import DataStream
from relmedner.types import Script

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams a real bigbio/gad row through the parquet conversion branch; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


@requires_live_hf
def test_a_live_streamed_gad_row_dispatches_end_to_end() -> None:
    """the exact production path -- declared ingest tuple -> build_stream ->
    HuggingFaceParquetDataStream streaming the refs/convert/parquet branch -> Script.REGISTRY
    dispatch -- must turn one real hub row into a valid classification example. Hand-built rows
    cannot catch hub-side schema drift (the bigbio conversion could re-encode labels differently
    across revisions), so this runs against the real streamed first train row. Gated behind
    RELMEDNER_LIVE_HF=1 and wenceslaus-only.

    Everything that transitively imports `datasets` is imported INSIDE the gated body so plain
    `pytest --collect-only` stays offline-safe.
    """
    from relmedner.hf_parquet import HuggingFaceParquetDataStream
    from relmedner.ingests import YamlIngestsParser
    from relmedner.registry import build_stream

    # select on the declared script NAME: bigbio/gad carries three declared ingests (train,
    # validation, test) and the weight field moved the repo id to payload position 2
    Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
    source, payload = next(
        (dataset.source, dataset.to_tuple()[1]) for dataset in Ingests.datasets if getattr(dataset.task, "name", None) == "GadBlurbScript"
    )
    stream: DataStream = build_stream(source, payload)
    assert isinstance(stream, HuggingFaceParquetDataStream)

    Name: str
    Task: tuple[Any, ...]
    Values: tuple[Any, ...]
    Name, (Task, Values) = next(stream.rows())
    assert Name == "bigbio/gad"
    assert Task == ("script", "GadBlurbScript", ("classifications",))
    assert len(Values) == 2  # the declared columns_out projection: text, labels

    Outputs, Example = Script.dispatch("GadBlurbScript", (tuple(Task[2]), Values))
    assert Outputs == ("classifications",)
    assert isinstance(Example, TrainingExample)
    # subset semantics of matches_declared_outputs: the row must produce the declared shape
    assert Example.populated() & {"classifications"}
    assert Example.classifications[0].labels == ["associated", "not associated"]
    assert Example.classifications[0].true_label in (["associated"], ["not associated"])
    # the anonymized placeholder text ships as-is, so it must be present in the shipped text
    assert Example.text and "@GENE$" in Example.text
