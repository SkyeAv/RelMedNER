from __future__ import annotations

import os

import pytest

from relmedner.models import TrainingExample, YamlIngests
from relmedner.streams import DataStream
from relmedner.types import Script

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams a real ncbi/ncbi_disease row from the hub; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


def live_example(marker: str) -> TrainingExample:
    """the exact production path for one real row of one declared split: parse -> build_stream ->
    hf_parquet streaming -> Script.REGISTRY dispatch. The three ncbi/ncbi_disease entries share
    one task name, so selection pins the declared parquet `file` (the entry key discriminator)."""
    from relmedner.ingests import YamlIngestsParser
    from relmedner.pipeline import dispatch_args, dispatch_row
    from relmedner.registry import build_stream

    Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
    source, payload = next((dataset.source, dataset.to_tuple()[1]) for dataset in Ingests.datasets if getattr(dataset, "file", None) == marker)
    stream: DataStream = build_stream(source, payload)

    Name: str
    Task: tuple[str, ...]
    Values: tuple[object, ...]
    Name, (Task, Values) = next(stream.rows())
    assert Name == "ncbi/ncbi_disease"
    assert Task == ("script", "NcbiDiseaseScript", ("entities", "relations"))

    Outputs, Example = Script.dispatch("NcbiDiseaseScript", (tuple(Task[2]), Values))
    assert Outputs == ("entities", "relations")
    # the pipeline's own dispatch wrapper must agree with the direct registry call above
    PipelineOutputs, PipelineExample = dispatch_row((Name, (Task, Values)), *dispatch_args(Ingests))
    assert PipelineOutputs == Outputs
    assert PipelineExample == Example
    return Example


@requires_live_hf
def test_a_live_streamed_train_row_dispatches_end_to_end() -> None:
    """live smoke over the first real row of the train split. Hand-built rows cannot catch
    hub-side schema drift (the parquet conversion can silently reshape the int tag column), so
    this runs against the real streamed row. Gated behind RELMEDNER_LIVE_HF=1 and wenceslaus-
    only: the hub download must never happen in the default offline suite."""
    example: TrainingExample = live_example("ncbi_disease/train/0000.parquet")
    assert "entities" in example.populated()
    # the first real train row carries the gold APC span; the entities shape must fill
    surfaces: list[str] = [mention for entity in example.entities for mention in entity.mentions]
    assert surfaces, "the first real row must emit its gold disease mention"
    for surface in surfaces:
        assert surface in example.text, f"mention {surface!r} does not occur in the text"
    assert {entity.label for entity in example.entities} == {"Disease"}


@requires_live_hf
def test_a_live_streamed_test_row_dispatches_end_to_end() -> None:
    """the test split file streams through the same declared path (the entry key discriminator is
    the file, so this proves the third lock's source actually yields)"""
    example: TrainingExample = live_example("ncbi_disease/test/0000.parquet")
    assert example.text, "the first real test row must ship its rejoined text"
