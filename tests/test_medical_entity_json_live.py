from __future__ import annotations

import os
from typing import Any

import pytest

from relmedner.models import TrainingExample, YamlIngests
from relmedner.streams import DataStream
from relmedner.types import Script

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams a real Medical-Entity-JSON-Extraction row from the hub; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


@requires_live_hf
def test_a_live_streamed_json_extraction_row_dispatches_end_to_end() -> None:
    """US-004 live smoke: the exact production path -- declared ingest tuple -> build_stream ->
    HuggingFaceDataStream streaming -> Script.REGISTRY dispatch -- must turn one real hub row
    into a valid training example. Hand-built fixtures cannot catch hub-side schema drift (the
    fenced JSON block could gain keys, nesting, or a closing fence in a revision), so this runs
    against the real streamed first test row. Gated behind RELMEDNER_LIVE_HF=1 and
    wenceslaus-only: the hub download must never happen in the default offline suite.

    Everything that transitively imports `datasets` (registry -> huggingface, pipeline) is
    imported INSIDE the gated body so plain `pytest --collect-only` stays offline-safe.
    """
    from relmedner.huggingface import HuggingFaceDataStream
    from relmedner.ingests import YamlIngestsParser
    from relmedner.pipeline import dispatch_row
    from relmedner.registry import build_stream

    Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
    source, payload = next(
        (dataset.source, dataset.to_tuple()[1]) for dataset in Ingests.datasets if getattr(dataset.task, "name", None) == "MedicalEntityJsonScript"
    )
    stream: DataStream = build_stream(source, payload)
    assert isinstance(stream, HuggingFaceDataStream)

    Name: str
    Task: tuple[Any, ...]
    Values: tuple[Any, ...]
    Name, (Task, Values) = next(stream.rows())
    assert Name == "Pennlaine/Medical-Entity-JSON-Extraction"
    assert Task == ("script", "MedicalEntityJsonScript", ("entities",))
    assert len(Values) == 1  # the declared columns_out projection: text

    Outputs, Example = Script.dispatch("MedicalEntityJsonScript", (tuple(Task[2]), Values))
    assert Outputs == ("entities",)
    assert isinstance(Example, TrainingExample)
    # the measured 100%-emit property over the full split holds for the streamed row too
    assert Example.entities, "a live row shipped with zero entities"
    # the gliner2 containment rule: every emitted mention is a substring of the emitted text
    for entity in Example.entities:
        for mention in entity.mentions:
            assert mention in Example.text
    # trust-gold stance, spot-checked on the live row: raw PascalCase attribute labels, no fullmap
    assert all(entity.label[:1].isupper() for entity in Example.entities)
    assert dispatch_row is not None  # production-path import kept honest
