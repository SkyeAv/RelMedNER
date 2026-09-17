from __future__ import annotations

from typing import Any

import pytest

from relmedner.ingests import YamlIngestsParser
from relmedner.models import LocalDataset, MatchOn, YamlIngests


def test_generate_tuples_shape() -> None:
    IngestsParser: YamlIngestsParser = YamlIngestsParser()
    generated: tuple[tuple[str, tuple[Any, ...]], ...] = IngestsParser.generate_tuples()

    assert generated == (
        (
            "hf",
            (
                ("script", "NemotronPiiScript", ("entities",)),
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
                ("script", "NemotronPiiScript", ("entities",)),
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
    Local: LocalDataset = LocalDataset(task={"type": "babel"}, source="local", path="/some/path.jsonl")
    assert Local.to_tuple() == ("local", (("babel",), "/some/path.jsonl"))


def test_script_task_carries_its_declared_output_shapes() -> None:
    ParsedIngests: YamlIngests = YamlIngestsParser().parse_ingests()
    assert all(dataset.task.outputs == ["entities"] for dataset in ParsedIngests.datasets)


def test_an_undeclared_script_fails_before_the_pipeline_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    IngestsParser: YamlIngestsParser = YamlIngestsParser()
    Declared: dict[str, Any] = IngestsParser.parse()
    Declared["datasets"][0]["task"]["name"] = "NoSuchScript"
    monkeypatch.setattr(IngestsParser, "parse", lambda: Declared)

    with pytest.raises(ValueError, match="undeclared script 'NoSuchScript'"):
        IngestsParser.parse_ingests()
