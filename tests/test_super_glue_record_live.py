from __future__ import annotations

import os
from typing import Any

import pytest

from relmedner.models import TrainingExample
from relmedner.types import Script

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams a real ReCoRD row from the hub; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


@requires_live_hf
def test_a_live_streamed_record_row_dispatches_end_to_end() -> None:
    """US-003 live smoke: the exact production path -- declared ingest tuple -> build_stream ->
    HuggingFaceDataStream streaming -> Script.REGISTRY dispatch -- must turn one real hub row
    into a valid training example. Hand-built rows cannot catch hub-side schema drift (the
    ReCoRD span table shape changed across revisions before), so this runs against the real
    streamed first train row. Gated behind RELMEDNER_LIVE_HF=1 and wenceslaus-only: the hub
    download must never happen in the default offline suite.

    Everything that transitively imports `datasets` (registry -> huggingface, pipeline) is
    imported INSIDE the gated body so plain `pytest --collect-only` stays offline-safe.
    """
    from relmedner.huggingface import HuggingFaceDataStream
    from relmedner.ingests import YamlIngestsParser
    from relmedner.pipeline import dispatch_row
    from relmedner.registry import build_stream

    by_dataset: dict[str, tuple[object, ...]] = {payload[1]: (source, payload) for source, payload in YamlIngestsParser().generate_tuples()}
    source: object
    payload: tuple[Any, ...]
    source, payload = by_dataset["aps/super_glue"]
    stream = build_stream(source, payload)
    assert isinstance(stream, HuggingFaceDataStream)

    Name: str
    Task: tuple[Any, ...]
    Values: tuple[Any, ...]
    Name, (Task, Values) = next(stream.rows())
    assert Name == "aps/super_glue"
    assert Task == ("script", "SuperGlueRecordScript", ("entities", "classifications"))
    assert len(Values) == 5  # the declared columns_out projection: passage, query, entities, entity_spans, answers

    Outputs, Example = Script.dispatch("SuperGlueRecordScript", (tuple(Task[2]), Values))
    assert Outputs == ("entities", "classifications")
    assert isinstance(Example, TrainingExample)
    # subset semantics of matches_declared_outputs: the row must produce at least one declared shape
    assert Example.populated() & {"entities", "classifications"}
    # every emitted mention must occur in the shipped passage: the spans are dataset gold, and a
    # mention outside the text would be a fabricated training signal the gliner2 validator rejects
    for entity in Example.entities:
        for mention in entity.mentions:
            assert mention in Example.text, f"mention {mention!r} does not occur in the passage"
    # answers are candidate surfaces, so every gold fill must be an emittable label
    for classification in Example.classifications:
        assert set(classification.true_label) <= set(classification.labels)

    # the pipeline's own dispatch wrapper must agree with the direct registry call above
    PipelineOutputs, PipelineExample = dispatch_row((Name, (Task, Values)))
    assert PipelineOutputs == Outputs
    assert PipelineExample == Example
