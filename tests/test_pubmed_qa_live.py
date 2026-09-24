from __future__ import annotations

import os

import pytest

from relmedner.models import TrainingExample, YamlIngests
from relmedner.streams import DataStream
from relmedner.types import Script

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams a real bigbio/pubmed_qa row from the hub; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


@requires_live_hf
def test_a_live_streamed_pubmed_qa_labeled_row_dispatches_end_to_end() -> None:
    """live smoke over the first real row of the declared pubmed_qa_labeled_fold0_source
    parquet. Gated behind RELMEDNER_LIVE_HF=1 and wenceslaus-only: the hub download must never
    happen in the default offline suite.

    Everything that transitively imports `datasets` (registry -> huggingface, pipeline) is
    imported INSIDE the gated body so plain `pytest --collect-only` stays offline-safe.
    """
    from relmedner.ingests import YamlIngestsParser
    from relmedner.pipeline import dispatch_args, dispatch_row
    from relmedner.registry import build_stream

    Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
    source, payload = next(
        (dataset.source, dataset.to_tuple()[1])
        for dataset in Ingests.datasets
        if getattr(dataset, "file", None) == "pubmed_qa_labeled_fold0_source/train/0000.parquet"
    )
    stream: DataStream = build_stream(source, payload)

    Name: str
    Task: tuple[str, ...]
    Values: tuple[object, ...]
    Name, (Task, Values) = next(stream.rows())
    assert Name == "bigbio/pubmed_qa"
    assert Task == ("script", "PubmedQaScript", ("classifications",))

    Outputs, Example = Script.dispatch("PubmedQaScript", (tuple(Task[2]), Values))
    assert Outputs == ("classifications",)
    assert isinstance(Example, TrainingExample)
    assert Example.populated() == {"classifications"}
    assert len(Example.classifications) == 1
    classification = Example.classifications[0]
    assert classification.labels == ["no", "yes", "maybe"]
    assert classification.true_label == ["yes"]
    assert Example.text.startswith("Does ")

    # the pipeline's own dispatch wrapper must agree with the direct registry call above
    PipelineOutputs, PipelineExample = dispatch_row((Name, (Task, Values)), *dispatch_args(Ingests))
    assert PipelineOutputs == Outputs
    assert PipelineExample == Example
