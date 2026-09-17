from __future__ import annotations

from typing import Any, Self

import relmedner.scripts  # noqa: F401 -- imported so every Script subclass self-registers
from relmedner.constants import INGESTS_YAML
from relmedner.models import ScriptTask, YamlIngests
from relmedner.parsers import YamlParser
from relmedner.types import Script


class YamlIngestsParser(YamlParser):
    def __init__(self: Self) -> None:
        super().__init__(INGESTS_YAML)

    def parse_ingests(self: Self) -> YamlIngests:
        serialized_yaml: Any = self.parse()
        ParsedIngests: YamlIngests = YamlIngests.model_validate(serialized_yaml)

        for dataset in ParsedIngests.datasets:
            if isinstance(dataset.task, ScriptTask) and dataset.task.name not in Script.REGISTRY:
                raise ValueError(f"undeclared script {dataset.task.name!r} -- declared scripts are {sorted(Script.REGISTRY)}")

        return ParsedIngests

    def generate_tuples(self: Self) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        ParsedIngests: YamlIngests = self.parse_ingests()
        return ParsedIngests.generate_tuples()
