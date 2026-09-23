from __future__ import annotations

import os
from typing import Any

import pytest

from relmedner.models import TrainingExample, YamlIngests
from relmedner.streams import DataStream
from relmedner.types import Script

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams real bigbio/chia rows from the hub; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


def live_example(marker: str) -> TrainingExample:
    """the exact production path for one real row of one declared chia subset: parse -> build_stream
    -> hf_parquet streaming -> Script.REGISTRY dispatch. The five chia entries share one task name,
    so selection pins the declared parquet `file` (the entry key discriminator)."""
    from relmedner.ingests import YamlIngestsParser
    from relmedner.pipeline import dispatch_args, dispatch_row
    from relmedner.registry import build_stream

    Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
    source, payload = next((dataset.source, dataset.to_tuple()[1]) for dataset in Ingests.datasets if getattr(dataset, "file", None) == marker)
    stream: DataStream = build_stream(source, payload)

    Name: str
    Task: tuple[Any, ...]
    Values: tuple[Any, ...]
    Name, (Task, Values) = next(stream.rows())
    assert Name == "bigbio/chia"
    assert Task == ("script", "ChiaScript", ("entities", "relations"))

    Outputs, Example = Script.dispatch("ChiaScript", (tuple(Task[2]), Values))
    assert Outputs == ("entities", "relations")
    # the pipeline's own dispatch wrapper must agree with the direct registry call above
    PipelineOutputs, PipelineExample = dispatch_row((Name, (Task, Values)), *dispatch_args(Ingests))
    assert PipelineOutputs == Outputs
    assert PipelineExample == Example
    return Example


def assert_mentions_in_text(example: TrainingExample) -> None:
    """every emitted mention and relation surface must occur in the emitted text: the spans are
    dataset gold, and a surface outside the text would be a fabricated training signal"""
    assert "entities" in example.populated() and "relations" in example.populated()
    surfaces: list[str] = [mention for entity in example.entities for mention in entity.mentions]
    surfaces += [field.value for relation in example.relations for field in relation.fields]
    for surface in surfaces:
        assert surface in example.text, f"surface {surface!r} does not occur in the text"


@requires_live_hf
def test_a_live_streamed_flat_row_dispatches_end_to_end() -> None:
    """live smoke over the first real row of a flat subset (str text column). Hand-built rows
    cannot catch hub-side schema drift, and the parquet conversion can silently reshuffle nested
    columns, so this runs against the real streamed row. Gated behind RELMEDNER_LIVE_HF=1 and
    wenceslaus-only: the hub download must never happen in the default offline suite. Every
    `datasets`-importing statement lives inside the gated body so `pytest --collect-only` stays
    offline-safe.
    """
    assert_mentions_in_text(live_example("chia_fixed_source/train/0000.parquet"))


@requires_live_hf
def test_a_live_streamed_bigbio_kb_row_dispatches_end_to_end() -> None:
    """the same live smoke over the passages-list subset: the two row shapes branch inside
    ChiaScript.run, and both must yield entities and relations from a real streamed row"""
    assert_mentions_in_text(live_example("chia_bigbio_kb/train/0000.parquet"))
