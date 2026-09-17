from __future__ import annotations

from collections.abc import Iterator
from itertools import count
from typing import Any, ClassVar, Self

from relmedner.huggingface import HuggingFaceDataStream
from relmedner.ingests import YamlIngestsParser
from relmedner.models import RunConfig
from relmedner.registry import SOURCE_REGISTRY, build_stream
from relmedner.streams import DataStream, StreamedRow


class CountingDataStream(DataStream):
    SOURCE: ClassVar[str] = "counting"

    def rows(self: Self) -> Iterator[StreamedRow]:
        for index in count():
            yield ("NemotronPiiScript", (("entities",), (str(index),)))


class FiniteDataStream(DataStream):
    SOURCE: ClassVar[str] = "finite"

    def rows(self: Self) -> Iterator[StreamedRow]:
        for index in range(20):
            yield ("NemotronPiiScript", (("entities",), (str(index),)))


def test_stream_truncates_an_unbounded_source() -> None:
    Streamed: list[StreamedRow] = list(CountingDataStream().stream(RunConfig(sample_limit=5)))
    assert Streamed == [("NemotronPiiScript", (("entities",), (str(index),))) for index in range(5)]


def test_stream_without_a_limit_yields_every_row() -> None:
    Streamed: list[StreamedRow] = list(FiniteDataStream().stream(RunConfig()))
    assert Streamed == [("NemotronPiiScript", (("entities",), (str(index),))) for index in range(20)]


def test_stream_limit_is_not_shared_between_calls() -> None:
    Stream: CountingDataStream = CountingDataStream()
    assert len(list(Stream.stream(RunConfig(sample_limit=5)))) == 5
    assert len(list(Stream.stream(RunConfig(sample_limit=5)))) == 5


def test_build_stream_constructs_from_declared_ingests() -> None:
    Source, Payload = YamlIngestsParser().generate_tuples()[0]
    Stream: DataStream = build_stream(Source, Payload)

    assert isinstance(Stream, HuggingFaceDataStream)
    assert Stream.dataset == "nvidia/Nemotron-PII"
    assert Stream.split == "train"


def test_registry_keys_on_the_source_discriminator() -> None:
    assert SOURCE_REGISTRY["hf"] is HuggingFaceDataStream
    assert all(Source == Stream.SOURCE for Source, Stream in SOURCE_REGISTRY.items())


def test_apply_match_keeps_only_declared_values() -> None:
    Stream: HuggingFaceDataStream = HuggingFaceDataStream(
        task=("script", "NemotronPiiScript", ("entities",)),
        dataset="nvidia/Nemotron-PII",
        subset=None,
        split="train",
        match_on=(("domain", ("Healthcare",)),),
        columns_out=("text",),
    )
    Kept: dict[str, Any] = {"domain": "Healthcare", "text": "a"}
    Dropped: dict[str, Any] = {"domain": "Finance", "text": "b"}

    assert Stream.apply_match(Kept) is True
    assert Stream.apply_match(Dropped) is False
