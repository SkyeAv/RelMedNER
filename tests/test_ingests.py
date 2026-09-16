from __future__ import annotations

from typing import Any

from relmedner.ingests import YamlIngestsParser
from relmedner.models import LocalDataset, MatchOn


def test_generate_tuples_shape() -> None:
    IngestsParser: YamlIngestsParser = YamlIngestsParser()
    generated: tuple[tuple[str, tuple[Any, ...]], ...] = IngestsParser.generate_tuples()

    assert generated == (
        (
            "hf",
            (
                "script",
                "nvidia/Nemotron-PII",
                None,
                "train",
                (("domain", ("Healthcare", "Biotechnology")),),
                ("text", "spans"),
            ),
        ),
        (
            "hf",
            (
                "script",
                "nvidia/Nemotron-PII",
                None,
                "test",
                (("domain", ("Healthcare", "Biotechnology")),),
                ("text", "spans"),
            ),
        ),
    )


def test_match_on_tuples_without_an_override() -> None:
    Match: MatchOn = MatchOn(column="domain", values=["Healthcare", "Biotechnology"])
    assert Match.to_tuple() == ("domain", ("Healthcare", "Biotechnology"))


def test_local_dataset_keys_on_source_without_an_override() -> None:
    Local: LocalDataset = LocalDataset(type="babel", source="local", path="/some/path.jsonl")
    assert Local.to_tuple() == ("local", ("babel", "/some/path.jsonl"))
