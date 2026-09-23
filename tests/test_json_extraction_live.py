from __future__ import annotations

import os
from itertools import islice
from typing import Any

import pytest

from relmedner.models import TrainingExample, YamlIngests
from relmedner.streams import DataStream
from relmedner.types import Script

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams real json-extraction rows from the hub; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


@requires_live_hf
def test_live_streamed_json_extraction_rows_dispatch_end_to_end() -> None:
    """US-002 live smoke: the exact production path -- declared ingest tuple -> build_stream ->
    HuggingFaceDataStream streaming -> Script.REGISTRY dispatch -- must turn real hub rows from
    each of the six source configs into valid training examples. Hand-built rows cannot catch
    hub-side drift (the `source` column keys the maps, and the first US-001 pass matched ZERO
    rows on dashed config names), so three rows per source config run end to end, each with the
    gliner2 invariants re-asserted: structures populated, every emitted mention surface and
    relation endpoint verbatim in the shipped text. Gated behind RELMEDNER_LIVE_HF=1 and
    wenceslaus-only: the hub stream must never happen in the default offline suite.

    Everything that transitively imports `datasets` (registry -> huggingface, pipeline) is
    imported INSIDE the gated body so plain `pytest --collect-only` stays offline-safe.
    """
    from relmedner.huggingface import HuggingFaceDataStream
    from relmedner.ingests import YamlIngestsParser
    from relmedner.pipeline import dispatch_row
    from relmedner.registry import build_stream

    Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
    Entries = [dataset for dataset in Ingests.datasets if getattr(dataset.task, "name", None) == "JsonExtractionScript"]
    assert len(Entries) == 6  # one per source config; the hub `all` config is deliberately undeclared

    Name: str
    Task: tuple[Any, ...]
    Values: tuple[Any, ...]
    for dataset in Entries:
        Source, Payload = dataset.to_tuple()
        stream: DataStream = build_stream(Source, Payload)
        assert isinstance(stream, HuggingFaceDataStream)
        Outputs: tuple[str, ...] = tuple(dataset.task.outputs)
        for Name, (Task, Values) in islice(stream.rows(), 3):
            assert Name == "agentlans/json-extraction"
            assert Task[1] == "JsonExtractionScript"
            assert len(Values) == 3  # the declared columns_out projection: text, json, source

            Dispatched: tuple[tuple[str, ...], TrainingExample] = Script.dispatch("JsonExtractionScript", (tuple(Task[2]), Values))
            RowOutputs, Example = Dispatched
            assert RowOutputs == tuple(Task[2])
            assert isinstance(Example, TrainingExample)
            # structures must stay populated on every real row (0 json parse failures measured
            # corpus-wide); the shape itself is a subset of the declared outputs
            assert Example.structures and Example.structures[0].fields
            assert Example.populated() & set(Outputs)
            # every emitted mention must occur in the shipped text: the entities guard drops
            # unlocatable surfaces at the leaf, and this re-asserts it against real hub values
            for entity in Example.entities:
                for mention in entity.mentions:
                    assert mention in Example.text, f"mention {mention!r} does not occur in the text"
            for relation in Example.relations:
                for field in relation.fields:
                    assert field.value in Example.text, f"relation endpoint {field.value!r} does not occur in the text"

            # the pipeline's own dispatch wrapper must agree with the direct registry call above
            PipelineOutputs, PipelineExample = dispatch_row((Name, (Task, Values)), Ingests.weights_by_source())
            assert PipelineOutputs == RowOutputs
            assert PipelineExample == Example
