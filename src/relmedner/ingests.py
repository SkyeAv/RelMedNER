from __future__ import annotations

from typing import Any, Self

from relmedner.constants import INGESTS_YAML
from relmedner.models import YamlIngests
from relmedner.parsers import YamlParser


class YamlIngestsParser(YamlParser):
    def __init__(self: Self) -> None:
        super().__init__(INGESTS_YAML)

    def parse_ingests(self: Self) -> YamlIngests:
        serialized_yaml: Any = self.parse()
        return YamlIngests.model_validate(serialized_yaml)

    def generate_tuples(self: Self) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        ParsedIngests: YamlIngests = self.parse_ingests()
        return ParsedIngests.generate_tuples()
