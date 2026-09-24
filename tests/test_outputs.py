from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Self
from urllib.error import URLError

import httpx
import pytest
from apache_beam.options.pipeline_options import PipelineOptions
from fastavro import reader
from huggingface_hub.errors import OfflineModeIsEnabled
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from relmedner.constants import OUTPUTS_MOUNT, TEST_ROW_LIMIT
from relmedner.enums import OutputShapes, ProcessingTypes
from relmedner.ingests import YamlIngestsParser
from relmedner.models import Entity, HuggingFaceDataset, RunConfig, ScriptTask, TrainingExample, YamlIngests
from relmedner.pipeline import BeamPipeline, output_path
from relmedner.streams import DataStream, StreamedRow, StreamStats
from relmedner.types import Script, ScriptValues
from relmedner.utils import ScriptUtils

TRANSPORT_ERRORS = (OfflineModeIsEnabled, httpx.NetworkError, httpx.TimeoutException, RequestsConnectionError, RequestsTimeout, URLError)

requires_live_hf: pytest.MarkDecorator = pytest.mark.skipif(
    os.environ.get("RELMEDNER_LIVE_HF") != "1",
    reason="streams every declared ingest from the hub; set RELMEDNER_LIVE_HF=1 (wenceslaus) to run",
)


def test_output_path_flips_to_the_worker_mount_only_for_flink() -> None:
    """options presence must not decide the output location: local prism runs carry worker-count
    options yet keep the caller's path; only a declared flink runner writes into /opt/outputs"""
    Config: RunConfig = RunConfig.from_flags(True, "test.avro")
    assert output_path(None, Config) == Path("test.avro")
    assert output_path(PipelineOptions(["--direct_num_workers=1"]), Config) == Path("test.avro")
    assert output_path(PipelineOptions(["--runner=FlinkRunner"]), Config) == Path(OUTPUTS_MOUNT) / Config.artifact_name()


def run_smoke_pipeline(output: Path) -> None:
    """skip only unavailable Hugging Face transport, preserving pipeline failures.
    One worker, one thread: twelve ingests streamed concurrently trip the unauthenticated
    Hugging Face rate limiter and push single bundles past the prism result deadline."""
    Options: PipelineOptions = PipelineOptions(
        [
            "--direct_num_workers=1",
            # prism's result stream inherits the 300s job_server_timeout default; a throttled
            # Hugging Face run (unauthenticated rate limiting) legitimately exceeds that
            "--job_server_timeout=1800",
        ]
    )
    try:
        BeamPipeline(options=Options).run(RunConfig.from_flags(True, str(output)))
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
@requires_live_hf
def test_build_dataset_test_run_writes_rows_matching_their_declared_shapes(tmp_path: Path) -> None:
    """live-data smoke run over EVERY declared ingest; the transport skip/propagation tests above
    stay un-gated without fullmap. Gated behind RELMEDNER_LIVE_HF=1 (the tests/test_*_live.py
    convention) in addition to the fullmap gate: hub-side rate limiting makes this test flaky in
    a default suite run, and the offline twin below carries the same output-shape contract
    through the same pipeline graph without the network.

    The record bound derives from the declaration rather than naming datasets, so adding a source
    cannot silently invalidate it: a hardcoded "five declared datasets" went stale the moment the
    CTKP local source landed and nothing noticed, because the assertion is an upper bound."""
    Output: Path = tmp_path / "test.avro"
    run_smoke_pipeline(Output)

    Records: list[dict[str, object]] = list(reader(open(Output, "rb")))
    Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
    # every declared ingest samples up to the test-run limit; empty/malformed rows shrink the count
    assert 1 <= len(Records) <= len(Ingests.datasets) * TEST_ROW_LIMIT

    Declared: frozenset[str] = frozenset({"entities", "classifications", "structures", "relations"})
    for record in Records:
        Example: TrainingExample = TrainingExample(**record)
        assert Example.text and Example.text.strip()
        # per-source mixing weight stamped from the ingest declaration; the corpus no longer declares
        # one uniform weight (bigbio/ehr_rel declares 0.5), so assert against the declared set
        assert Example.weight in set(Ingests.weights_by_source().values())
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


class SmokeScript(Script):
    """offline registry script for the mocked smoke: the minimal (text, label) -> entities
    contract, so the pipeline graph runs with no hub and no fullmap dependency"""

    NAME: ClassVar[str] = "SmokeScript"

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        text, label = values
        return TrainingExample(text=text or "", entities=[Entity(label=label or "", mentions=[text or ""])])


class SmokeDataStream(DataStream):
    """offline DataStream: rows() yields deterministic (text, label) values shaped for
    SmokeScript. Subclassing DataStream (not duck-typing) keeps stream()'s sample_rate and
    sample-limit wrapping, so the twin exercises the same streaming entry point as the hub"""

    SOURCE: ClassVar[str] = "smoke"

    def __init__(self: Self, task: tuple[Any, ...], weight: float, values: tuple[ScriptValues, ...], *, name: str) -> None:
        super().__init__(task, weight)
        self.name: str = name
        self.values: tuple[ScriptValues, ...] = values

    def rows(self: Self) -> Iterator[StreamedRow]:
        self.stats = StreamStats()
        for values in self.values:
            self.stats.rows_in += 1
            self.stats.rows_out += 1
            yield (self.name, (self.task, values))


def _smoke_ingests() -> YamlIngests:
    """two declared script sources with distinct weights: two sources prove the weight stamping
    and the fan-out carry real per-source values instead of one degenerate constant"""
    datasets = [
        HuggingFaceDataset(
            source="hf",
            task=ScriptTask(type=ProcessingTypes.SCRIPT, name="SmokeScript", outputs=[OutputShapes.ENTITIES]),
            weight=weight,
            dataset=name,
            subset=None,
            split=None,
            match_on=None,
            columns_out=["text", "label"],
        )
        for name, weight in (("smoke/alpha", 1.0), ("smoke/beta", 0.5))
    ]
    return YamlIngests(datasets=datasets)


def test_a_mocked_streamed_pipeline_writes_rows_matching_their_declared_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """offline twin of the fullmap+RELMEDNER_LIVE_HF-gated live smoke above: the SAME
    BeamPipeline graph (stream -> dispatch -> declared-output filter -> dedup -> avro shards)
    runs over monkeypatched stream sources, so the output-shape contract stays enforced on
    every default suite run (laptop included) instead of only where the hub is reachable.
    Row values carry distinct texts so near-dedup cannot collapse them, and the assertions
    mirror the live test one for one."""
    Output: Path = tmp_path / "test.avro"
    Ingests: YamlIngests = _smoke_ingests()
    monkeypatch.setattr(YamlIngestsParser, "parse_ingests", lambda self: Ingests)

    def fake_build_stream(
        source: str,
        payload: tuple[Any, ...],
        filters: Any = None,
        sample_rate: float = 1.0,
        read_shards: int = 1,
        shard_index: int = 0,
    ) -> SmokeDataStream:
        # build_stream's envelope is (source, payload, filters, sample_rate, read_shards,
        # shard_index) and the pipeline splats all six positionally, so the stand-in must accept
        # the two sharding slots too. The smoke sources declare no read_shards, so the twin
        # exercises the unsharded path: one envelope per source, index 0. Asserting that keeps a
        # future read_shards declaration on a smoke source from silently testing one shard of a
        # multi-shard read as if it were the whole corpus.
        assert (read_shards, shard_index) == (1, 0)
        # source is the SOURCE kind ("hf"), so the row key comes from the payload's dataset slot
        # (DatasetBase.row_key): the pipeline looks the mixing weight up by the stamped name.
        # Each source gets its own texts: identical rows across sources would be exact
        # duplicates and dedup (on by default) would rightly drop the second source's copies
        Texts: dict[str, tuple[ScriptValues, ...]] = {
            "smoke/alpha": (
                ("Aspirin treats headache.", "chemical"),
                ("Interleukin-2 signals through its receptor.", "gene"),
            ),
            "smoke/beta": (
                ("Metformin lowers blood glucose.", "chemical"),
                ("BRCA1 mutations predispose carriers to cancer.", "gene"),
            ),
        }
        return SmokeDataStream(payload[0], 1.0, Texts[payload[2]], name=payload[2])

    monkeypatch.setattr("relmedner.pipeline.build_stream", fake_build_stream)

    BeamPipeline().run(RunConfig.from_flags(True, str(Output)))

    Records: list[dict[str, object]] = list(reader(open(Output, "rb")))
    # every declared source samples up to the test-run limit; empty/malformed rows shrink the count
    assert 1 <= len(Records) <= len(Ingests.datasets) * TEST_ROW_LIMIT

    Declared: frozenset[str] = frozenset({"entities", "classifications", "structures", "relations"})
    for record in Records:
        Example: TrainingExample = TrainingExample(**record)
        assert Example.text and Example.text.strip()
        # per-source mixing weight stamped from the ingest declaration; the twin declares 1.0
        # and 0.5, and both values must appear across the merged records
        assert Example.weight in set(Ingests.weights_by_source().values())
        Populated: frozenset[str] = Example.populated()
        # subset contract, not exact equality; every shipped record carries at least one output shape
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
    # both declared sources reached the output: a source silently dropped by the graph must fail here
    Stamped: set[object] = {record["weight"] for record in Records}
    assert set(Ingests.weights_by_source().values()) <= Stamped


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


def test_streamed_rows_pass_a_reshuffle_before_dispatch() -> None:
    """each declared source is ONE Create element, so without a fusion break the source read and
    every dispatch/mine step run fused in one task (one core per dataset). The pipeline must
    place a Reshuffle between streaming and the task-type split; the Flink translator lowers it
    to rebalance(), which is what fans rows out over every slot"""
    import inspect

    from relmedner import pipeline

    source = inspect.getsource(pipeline.BeamPipeline.run)
    assert source.index('"stream declared data"') < source.index("beam.Reshuffle()") < source.index('"split by task type"')
