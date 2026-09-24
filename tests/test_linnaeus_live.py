from __future__ import annotations

import os

import pytest

from relmedner.models import TrainingExample, YamlIngests
from relmedner.streams import DataStream
from relmedner.types import Script

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams a real bigbio/linnaeus row from the hub; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


@requires_live_hf
def test_a_live_streamed_linnaeus_row_dispatches_end_to_end() -> None:
    """live smoke over the first real row of the declared linnaeus_bigbio_kb parquet. Hand-built
    rows cannot catch hub-side schema drift (the parquet conversion can silently reshuffle the
    nested passages/entities columns), so this runs against the real streamed row. Gated behind
    RELMEDNER_LIVE_HF=1 and wenceslaus-only: the hub download must never happen in the default
    offline suite.

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
        if getattr(dataset, "file", None) == "linnaeus_bigbio_kb/train/0000.parquet"
    )
    stream: DataStream = build_stream(source, payload)

    Name: str
    Task: tuple[str, ...]
    Values: tuple[object, ...]
    Name, (Task, Values) = next(stream.rows())
    assert Name == "bigbio/linnaeus"
    assert Task == ("script", "LinnaeusScript", ("entities",))

    Outputs, Example = Script.dispatch("LinnaeusScript", (tuple(Task[2]), Values))
    assert Outputs == ("entities",)
    assert isinstance(Example, TrainingExample)
    assert Example.populated() == {"entities"}
    # the first real document (pmcA1805739) carries 47 gold species mentions; the entities shape
    # must fill, and every emitted surface must occur in the re-joined token text
    surfaces: list[str] = [mention for entity in Example.entities for mention in entity.mentions]
    assert surfaces, "the first real row must emit its gold species mentions"
    for surface in surfaces:
        assert surface in Example.text, f"mention {surface!r} does not occur in the text"
    assert {entity.label for entity in Example.entities} == {"OrganismTaxon"}

    # the pipeline's own dispatch wrapper must agree with the direct registry call above
    PipelineOutputs, PipelineExample = dispatch_row((Name, (Task, Values)), *dispatch_args(Ingests))
    assert PipelineOutputs == Outputs
    assert PipelineExample == Example
