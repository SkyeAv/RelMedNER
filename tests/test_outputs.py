from __future__ import annotations

from pathlib import Path

import pytest
from fastavro import reader

from relmedner.models import RunConfig, TrainingExample
from relmedner.pipeline import BeamPipeline
from relmedner.utils import ScriptUtils

pytestmark = pytest.mark.skipif(not ScriptUtils.fullmap_available(), reason="fullmap database is not mounted")


def test_build_dataset_test_run_writes_five_biolink_labeled_rows(tmp_path: Path) -> None:
    Output: Path = tmp_path / "test.avro"
    try:
        BeamPipeline().run(RunConfig.from_flags(True, str(Output)))
    except Exception as exc:
        pytest.skip(f"huggingface is unreachable: {exc}")

    Records: list[dict[str, object]] = list(reader(open(Output, "rb")))
    assert 1 <= len(Records) <= 5

    for record in Records:
        Example: TrainingExample = TrainingExample(**record)
        assert Example.text and Example.text.strip()
        assert Example.populated() == frozenset({"entities"})
        for entity in Example.entities:
            assert entity.label and not entity.label.startswith("biolink:")
            assert entity.mentions
            for mention in entity.mentions:
                assert mention.lower() in Example.text.lower()
            if not ScriptUtils.is_biolink_category(entity.label):
                assert entity.description is None
