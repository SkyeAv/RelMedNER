from __future__ import annotations

import os
from typing import Any

import pytest

from relmedner.models import TrainingExample
from relmedner.streams import DataStream
from relmedner.types import Script

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams a real DocRED row from the hub; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


@requires_live_hf
def test_a_live_streamed_docred_row_dispatches_end_to_end() -> None:
    """the exact production path -- declared ingest tuple -> build_stream ->
    HuggingFaceJsonDataStream (non-streaming json builder over the raw .json.gz) -> registry
    dispatch -- must turn one real hub row into a valid training example. Hand-built rows cannot
    catch hub-side schema drift (the labels arrive keyed h/t/r, pos is end-exclusive, and test
    rows drop the labels key entirely), so this runs against the real cached train_annotated
    file. Gated behind RELMEDNER_LIVE_HF=1 and wenceslaus-only: the hub download must never
    happen in the default offline suite.

    Everything that transitively imports `datasets` (hf_json, registry, pipeline, ingests) is
    imported INSIDE the gated body so plain `pytest --collect-only` stays offline-safe.
    """
    from relmedner.hf_json import HuggingFaceJsonDataStream
    from relmedner.ingests import YamlIngestsParser
    from relmedner.pipeline import dispatch_args, dispatch_row
    from relmedner.registry import build_stream

    Ingests = YamlIngestsParser().parse_ingests()
    source, payload = next(
        (dataset.source, dataset.to_tuple()[1])
        for dataset in Ingests.datasets
        if getattr(dataset.task, "name", None) == "DocredScript" and getattr(dataset, "file", "").endswith("train_annotated.json.gz")
    )
    stream: DataStream = build_stream(source, payload)
    assert isinstance(stream, HuggingFaceJsonDataStream)

    Name: str
    Task: tuple[Any, ...]
    Values: tuple[Any, ...]
    Name, (Task, Values) = next(stream.rows())
    assert Name == "thunlp/docred"
    assert Task == ("script", "DocredScript", ("entities", "relations"))
    assert len(Values) == 3  # the declared columns_out projection: sents, vertexSet, labels

    Outputs, Example = Script.dispatch("DocredScript", (tuple(Task[2]), Values))
    assert Outputs == ("entities", "relations")
    assert isinstance(Example, TrainingExample)
    # subset semantics of matches_declared_outputs: the row must produce at least one declared shape
    assert Example.populated() & {"entities", "relations"}
    # every emitted mention must occur in the shipped text: the spans are dataset gold measured
    # end-exclusive, and a mention outside the text would be fabricated signal the gliner2
    # validator rejects
    for entity in Example.entities:
        for mention in entity.mentions:
            assert mention in Example.text, f"mention {mention!r} does not occur in the document"
    for relation in Example.relations:
        for field in relation.fields:
            assert field.value in Example.text, f"relation field {field.value!r} does not occur in the document"
    # train_annotated is ~99% label-bearing (3,027 of 3,053), so the first row must carry gold
    assert Example.relations, "first train_annotated row unexpectedly produced no relations"

    # the pipeline's own dispatch wrapper must agree with the direct registry call above
    PipelineOutputs, PipelineExample = dispatch_row((Name, (Task, Values)), *dispatch_args(Ingests))
    assert PipelineOutputs == Outputs
    assert PipelineExample == Example
