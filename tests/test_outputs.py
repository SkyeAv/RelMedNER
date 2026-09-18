from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastavro import reader
from huggingface_hub.errors import OfflineModeIsEnabled
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from relmedner.models import RunConfig, TrainingExample
from relmedner.pipeline import BeamPipeline
from relmedner.utils import ScriptUtils

TRANSPORT_ERRORS = (OfflineModeIsEnabled, httpx.NetworkError, httpx.TimeoutException, RequestsConnectionError, RequestsTimeout)


def run_smoke_pipeline(output: Path) -> None:
    """skip only unavailable Hugging Face transport, preserving pipeline failures"""
    try:
        BeamPipeline().run(RunConfig.from_flags(True, str(output)))
    except TRANSPORT_ERRORS as exc:
        pytest.skip(f"huggingface is unreachable: {exc}")


def test_smoke_pipeline_skips_expected_transport_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """known Hugging Face transport loss remains an environmental smoke skip"""
    monkeypatch.setattr(BeamPipeline, "run", lambda self, config: (_ for _ in ()).throw(httpx.ConnectError("offline")))

    with pytest.raises(pytest.skip.Exception, match="huggingface is unreachable: offline"):
        run_smoke_pipeline(tmp_path / "test.avro")


def test_smoke_pipeline_propagates_unexpected_pipeline_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """pipeline regressions must fail smoke tests instead of becoming skips"""
    monkeypatch.setattr(BeamPipeline, "run", lambda self, config: (_ for _ in ()).throw(RuntimeError("pipeline bug")))

    with pytest.raises(RuntimeError, match="pipeline bug"):
        run_smoke_pipeline(tmp_path / "test.avro")


@pytest.mark.skipif(not ScriptUtils.fullmap_available(), reason="fullmap database is not mounted")
def test_build_dataset_test_run_writes_five_biolink_labeled_rows(tmp_path: Path) -> None:
    """live-data smoke run; the transport skip/propagation tests above stay un-gated without fullmap"""
    Output: Path = tmp_path / "test.avro"
    run_smoke_pipeline(Output)

    Records: list[dict[str, object]] = list(reader(open(Output, "rb")))
    assert 1 <= len(Records) <= 5

    for record in Records:
        Example: TrainingExample = TrainingExample(**record)
        assert Example.text and Example.text.strip()
        assert Example.populated() == frozenset({"entities", "relations"})
        # relation head/tail surfaces must occur in the record text, proving extraction never invents mentions
        for relation in Example.relations:
            Fields: dict[str, str] = {field.name: field.value for field in relation.fields}
            for side in ("head", "tail"):
                assert Fields[side].lower() in Example.text.lower()
        for entity in Example.entities:
            assert entity.label and not entity.label.startswith("biolink:")
            assert entity.mentions
            for mention in entity.mentions:
                assert mention.lower() in Example.text.lower()
            if not ScriptUtils.is_biolink_category(entity.label):
                assert entity.description is None
