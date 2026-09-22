from __future__ import annotations

import importlib.util
from pathlib import Path
from urllib.error import URLError

import httpx
import pytest
from fastavro import reader
from huggingface_hub.errors import OfflineModeIsEnabled
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from relmedner.models import RunConfig, TrainingExample
from relmedner.pipeline import BeamPipeline
from relmedner.types import Script
from relmedner.utils import ScriptUtils

TRANSPORT_ERRORS = (OfflineModeIsEnabled, httpx.NetworkError, httpx.TimeoutException, RequestsConnectionError, RequestsTimeout, URLError)


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
def test_build_dataset_test_run_writes_rows_matching_their_declared_shapes(tmp_path: Path) -> None:
    """live-data smoke run; the transport skip/propagation tests above stay un-gated without fullmap.
    Six declared datasets (pre-training script, pile-ner IOB script, sentence_rex relations script,
    curated-corpus fullmap mining, balanced curated-corpus fullmap mining, post-training multi-task
    script), up to five rows each."""
    Output: Path = tmp_path / "test.avro"
    run_smoke_pipeline(Output)

    Records: list[dict[str, object]] = list(reader(open(Output, "rb")))
    # six declared ingests, each sampling up to five rows; empty/malformed rows may shrink the count
    assert 1 <= len(Records) <= 6 * 5

    Declared: frozenset[str] = frozenset({"entities", "classifications", "structures", "relations"})
    for record in Records:
        Example: TrainingExample = TrainingExample(**record)
        assert Example.text and Example.text.strip()
        Populated: frozenset[str] = Example.populated()
        # subset contract, not exact equality; every shipped record carries at least one output shape
        # (the sentence_rex ingest emits relations only, so pinning "entities" here would fail it)
        assert Populated
        assert Populated <= Declared
        # relation head/tail surfaces must occur in the record text, proving extraction never invents mentions
        for relation in Example.relations:
            Fields: dict[str, str] = {field.name: field.value for field in relation.fields}
            assert set(Fields) == {"head", "tail"}  # gliner2 cannot carry a third field
            for side in ("head", "tail"):
                assert Fields[side].lower() in Example.text.lower()
            assert Fields["head"].lower() != Fields["tail"].lower()  # no mined self-loops
            assert relation.evidence in {"asserted", "distant", "sampled_negative"}
        for entity in Example.entities:
            assert entity.label and not entity.label.startswith("biolink:")
            assert entity.mentions
            for mention in entity.mentions:
                assert mention.lower() in Example.text.lower()
            if not ScriptUtils.is_biolink_category(entity.label):
                assert entity.description is None


def test_a_post_training_relation_row_survives_the_real_gliner_sanitizer(tmp_path: Path) -> None:
    """offline smoke for the post-training path: the sampled-negative encoding must stay gliner2-safe
    (verified live against the corpus in test_families; this locks the script surface itself)"""
    if importlib.util.find_spec("gliner2") is None:
        pytest.skip("gliner2 is not installed")

    Tokens: list[str] = [
        "Identify",
        "relations",
        "between",
        "entities",
        "based",
        "on",
        "provided",
        "source",
        "entity",
        "and",
        "relation",
        ":",
        "kynurenine",
        "pathway",
        "KP",
        "metabolites",
        "are",
        "associated",
        "with",
        "accelerated",
        "atherosclerosis",
        "in",
        "chronic",
        "kidney",
        "disease",
        "CKD",
        "patients",
        ".",
    ]
    _, Example = Script.dispatch(
        "GlinerBiomedPostScript",
        (
            ("relations",),
            (
                Tokens,
                [[12, 15, "accelerated atherosclerosis <> associated with"], [22, 26, "accelerated atherosclerosis <> occurs in"]],
                ["chronic kidney disease CKD patients <> associated with <> kynurenine pathway KP metabolites"],
            ),
        ),
    )
    Output: dict[str, object] = Example.to_output()

    assert Example.populated() == frozenset({"relations"})
    assert Output["output"] == {
        "relations": [
            {"associated_with": {"head": "accelerated atherosclerosis", "tail": "kynurenine pathway KP metabolites"}},
            {"occurs_in": {"head": "accelerated atherosclerosis", "tail": "chronic kidney disease CKD patients"}},
            {"not_associated_with": {"head": "chronic kidney disease CKD patients", "tail": "kynurenine pathway KP metabolites"}},
        ]
    }
    assert [relation.evidence for relation in Example.relations] == ["asserted", "asserted", "sampled_negative"]
    assert [relation.negated for relation in Example.relations] == [False, False, True]
