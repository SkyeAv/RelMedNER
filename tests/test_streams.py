from __future__ import annotations

from collections.abc import Iterator
from itertools import count
from typing import Any, ClassVar, Self

import pytest

from relmedner.huggingface import HuggingFaceDataStream
from relmedner.ingests import YamlIngestsParser
from relmedner.local import LocalAvroDataStream
from relmedner.models import RunConfig, YamlIngests
from relmedner.registry import SOURCE_REGISTRY, build_stream
from relmedner.streams import DataStream, StreamedRow


class CountingDataStream(DataStream):
    SOURCE: ClassVar[str] = "counting"

    def rows(self: Self) -> Iterator[StreamedRow]:
        for index in count():
            yield ("GlinerBiomedScript", (("entities",), (str(index),)))


class FiniteDataStream(DataStream):
    SOURCE: ClassVar[str] = "finite"

    def rows(self: Self) -> Iterator[StreamedRow]:
        for index in range(20):
            yield ("GlinerBiomedScript", (("entities",), (str(index),)))


def test_stream_truncates_an_unbounded_source() -> None:
    Streamed: list[StreamedRow] = list(CountingDataStream().stream(RunConfig(sample_limit=5)))
    assert Streamed == [("GlinerBiomedScript", (("entities",), (str(index),))) for index in range(5)]


def test_stream_without_a_limit_yields_every_row() -> None:
    Streamed: list[StreamedRow] = list(FiniteDataStream().stream(RunConfig()))
    assert Streamed == [("GlinerBiomedScript", (("entities",), (str(index),))) for index in range(20)]


def test_stream_limit_is_not_shared_between_calls() -> None:
    Stream: CountingDataStream = CountingDataStream()
    assert len(list(Stream.stream(RunConfig(sample_limit=5)))) == 5
    assert len(list(Stream.stream(RunConfig(sample_limit=5)))) == 5


def test_build_stream_constructs_every_declared_ingest() -> None:
    """every declared tuple must unpack into its stream ctor positionally, and the key the stream
    stamps its rows with must equal DatasetBase.row_key, because the pipeline looks the source's
    mixing weight up by that string (drift is a KeyError partway through a run, not a wrong number).

    Building ALL of them is what catches a source whose ctor was never updated for a new DatasetBase
    field: the local stream took (task, path) while the declared payload already carried
    (task, weight, path), so the CTKP ingest raised TypeError at build_stream while every hf-only
    test stayed green. Construction is offline-safe, since no source touches the network or the disk
    until rows() is called.
    """
    Ingests: YamlIngests = YamlIngestsParser().parse_ingests()
    Weights: dict[str, float] = Ingests.weights_by_source()

    Built: list[DataStream] = [build_stream(*dataset.to_tuple()) for dataset in Ingests.datasets]

    assert len(Built) == len(Ingests.datasets)
    # both source kinds stay covered, so this invariant cannot silently degrade to hf-only again
    assert {type(stream) for stream in Built} == {HuggingFaceDataStream, LocalAvroDataStream}
    for dataset, stream in zip(Ingests.datasets, Built, strict=True):
        assert stream.name == dataset.row_key, f"{type(dataset).__name__} row key drift"
        assert stream.weight == dataset.weight
        assert Weights[stream.name] == dataset.weight

    First = Built[0]
    assert isinstance(First, HuggingFaceDataStream)
    assert First.dataset == "anthonyyazdaniml/gliner-biomed-pre-training"
    assert First.split == "train"
    assert First.columns_out == ("tokenized_text", "ner")
    assert First.weight == 1.0


def test_build_stream_carries_the_declared_weight() -> None:
    """the weight rides the frozen payload tuple (DatasetBase field order) positionally into
    the stream ctor, so a nondefault declaration must land on the stream untouched"""
    Stream: DataStream = build_stream(
        "hf",
        (
            ("script", "GlinerBiomedScript", ("entities",)),
            0.25,
            "some/dataset",
            None,
            "train",
            None,
            ("text",),
        ),
    )

    assert isinstance(Stream, HuggingFaceDataStream)
    assert Stream.weight == 0.25


def test_rebuild_task_round_trips_every_declared_task_type() -> None:
    from relmedner.models import FullmapTask, ScriptTask
    from relmedner.streams import rebuild_task

    for Payload in YamlIngestsParser().generate_tuples():
        Task = rebuild_task(Payload[1][0])
        assert isinstance(Task, ScriptTask | FullmapTask)
        assert Task.to_tuple() == Payload[1][0]

    with pytest.raises(ValueError):
        rebuild_task(("teleport",))


def test_registry_keys_on_the_source_discriminator() -> None:
    assert SOURCE_REGISTRY["hf"] is HuggingFaceDataStream
    assert all(Source == Stream.SOURCE for Source, Stream in SOURCE_REGISTRY.items())


def test_apply_match_keeps_only_declared_values() -> None:
    Stream: HuggingFaceDataStream = HuggingFaceDataStream(
        task=("script", "GlinerBiomedScript", ("entities",)),
        weight=1.0,
        dataset="anthonyyazdaniml/gliner-biomed-pre-training",
        subset=None,
        split="train",
        match_on=(("domain", ("Healthcare",)),),
        columns_out=("tokenized_text",),
    )
    Kept: dict[str, Any] = {"domain": "Healthcare", "tokenized_text": ["a"]}
    Dropped: dict[str, Any] = {"domain": "Finance", "tokenized_text": ["b"]}

    assert Stream.apply_match(Kept) is True
    assert Stream.apply_match(Dropped) is False
