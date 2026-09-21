from __future__ import annotations

from collections.abc import Iterator
from itertools import count
from typing import Any, ClassVar, Self

import pytest

from relmedner.huggingface import HuggingFaceDataStream
from relmedner.ingests import YamlIngestsParser
from relmedner.models import RunConfig
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


def test_build_stream_constructs_from_declared_ingests() -> None:
    Source, Payload = YamlIngestsParser().generate_tuples()[0]
    Stream: DataStream = build_stream(Source, Payload)

    assert isinstance(Stream, HuggingFaceDataStream)
    assert Stream.dataset == "anthonyyazdaniml/gliner-biomed-pre-training"
    assert Stream.split == "train"
    assert Stream.columns_out == ("tokenized_text", "ner")


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
