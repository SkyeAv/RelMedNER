from __future__ import annotations

import os
from typing import Any

import pytest

from relmedner.models import TrainingExample, YamlIngests
from relmedner.streams import DataStream
from relmedner.types import Script

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams a real BioRED row from the hub; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


@requires_live_hf
def test_a_live_streamed_biored_row_dispatches_end_to_end() -> None:
    """the exact production path for the biored ingest -- declared tuple -> build_stream ->
    HuggingFaceDataStream streaming -> Script.REGISTRY dispatch -- must turn one real hub row
    into a valid training example. Hand-built fixtures cannot catch hub-side layout drift (the
    row shape here is the list-of-dicts bigbio_kb conversion, not the usual dict-of-parallel-
    lists), so this pins the real streamed first train row (document_id 10491763, measured:
    32 entities, 107 raw mention-pair relations collapsing to 3 concept-signature assertions).
    Gated behind RELMEDNER_LIVE_HF=1 and wenceslaus-only; every datasets-importing statement
    lives INSIDE the body so plain `pytest --collect-only` stays offline.
    """
    from relmedner.huggingface import HuggingFaceDataStream
    from relmedner.ingests import YamlIngestsParser
    from relmedner.pipeline import dispatch_row
    from relmedner.registry import build_stream

    Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
    source, payload = next(
        (dataset.source, dataset.to_tuple()[1])
        for dataset in Ingests.datasets
        if getattr(dataset.task, "name", None) == "BioredScript" and dataset.to_tuple()[1][4] == "train"
    )
    stream: DataStream = build_stream(source, payload)
    assert isinstance(stream, HuggingFaceDataStream)

    Name: str
    Task: tuple[Any, ...]
    Values: tuple[Any, ...]
    Name, (Task, Values) = next(stream.rows())
    assert Name == "wcole3/biored-parquet"
    assert Task == ("script", "BioredScript", ("entities", "relations"))
    assert len(Values) == 3  # the declared columns_out projection: passages, entities, relations

    Outputs, Example = Script.dispatch("BioredScript", (tuple(Task[2]), Values))
    assert Outputs == ("entities", "relations")
    assert isinstance(Example, TrainingExample)
    # the measured contract on this row: 107 raw mention pairs collapse to the 3 concept
    # signatures, first-occurrence order, asserted evidence
    assert [relation.name for relation in Example.relations] == [
        "associated_with",
        "positively_correlated_with",
        "associated_with",
    ]
    assert Example.populated() == frozenset({"entities", "relations"})
    for entity in Example.entities:
        for mention in entity.mentions:
            assert mention in Example.text, f"mention {mention!r} does not occur in the emitted text"

    # the pipeline's own dispatch wrapper must agree with the direct registry call above
    PipelineOutputs, PipelineExample = dispatch_row((Name, (Task, Values)), Ingests.weights_by_source())
    assert PipelineOutputs == Outputs
    assert PipelineExample == Example
