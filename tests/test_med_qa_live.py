from __future__ import annotations

import os

import pytest

from relmedner.models import TrainingExample, YamlIngests
from relmedner.streams import DataStream
from relmedner.types import Script

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams a real GBaker/MedQA-USMLE-4-options row from the hub; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


@requires_live_hf
def test_a_live_streamed_med_qa_row_dispatches_end_to_end() -> None:
    """live smoke over the first real row of the declared phrases_no_exclude_train.jsonl.
    Gated behind RELMEDNER_LIVE_HF=1 and wenceslaus-only: the hub download must never happen in
    the default offline suite.

    Everything that transitively imports `datasets` (registry -> huggingface, pipeline) is
    imported INSIDE the gated body so plain `pytest --collect-only` stays offline-safe.
    """
    from relmedner.ingests import YamlIngestsParser
    from relmedner.pipeline import dispatch_args, dispatch_row
    from relmedner.registry import build_stream

    Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
    source, payload = next(
        (dataset.source, dataset.to_tuple()[1]) for dataset in Ingests.datasets if getattr(dataset, "file", None) == "phrases_no_exclude_train.jsonl"
    )
    stream: DataStream = build_stream(source, payload)

    Name: str
    Task: tuple[str, ...]
    Values: tuple[object, ...]
    Name, (Task, Values) = next(stream.rows())
    assert Name == "GBaker/MedQA-USMLE-4-options"
    assert Task == ("script", "MedQaScript", ("classifications",))
    # columns_out order: question, options, answer_idx. The row's `answer` column is undeclared,
    # so the label text can never reach the script.
    assert len(Values) == 3

    Outputs, Example = Script.dispatch("MedQaScript", (tuple(Task[2]), Values))
    assert Outputs == ("classifications",)
    assert isinstance(Example, TrainingExample)
    assert Example.populated() == {"classifications"}
    assert len(Example.classifications) == 1
    classification = Example.classifications[0]
    assert classification.task == "medical multiple-choice question answering"
    assert classification.labels == ["A", "B", "C", "D"]
    # measured row 0 of the declared split (laptop census 2026-09-24): answer_idx D,
    # options A Ampicillin / B Ceftriaxone / C Doxycycline / D Nitrofurantoin
    assert classification.true_label == ["D"]
    assert Example.text.startswith("A 23-year-old pregnant woman at 22 weeks gestation")
    assert "D. Nitrofurantoin" in Example.text

    # the pipeline's own dispatch wrapper must agree with the direct registry call above
    PipelineOutputs, PipelineExample = dispatch_row((Name, (Task, Values)), *dispatch_args(Ingests))
    assert PipelineOutputs == Outputs
    assert PipelineExample == Example
