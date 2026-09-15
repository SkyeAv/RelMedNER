from __future__ import annotations

from relmedner.parsers import YamlParser
from relmedner.models import YamlIngests
from relmedner.constants import DATA

from importlib.resources.abc import Traversable
from typing import Self, Any, Union, Literal

import fastarvo


class YamlIngestsParser(YamlParser):
    def __init__(self: Self, yaml_p: str = "ingests.yaml", arvo_p: str = "ingests.arvo") -> None:
        super().__init__()
        self.yaml_p: Traversable = DATA / yaml_p
        self.arvo_p: Traversable = DATA / arvo_p

    def parse_ingests(self: Self) -> YamlIngests:
        serialized_yaml: Any = self.parse()
        return YamlIngests.model_validate(serialized_yaml)

    def write_ingests_arvo(self: Self) -> None:
        ParsedIngests: YamlIngests = self.parse_ingests()

        arvo_schema: Any = ParsedIngests.avro_schema_to_python()
        parsed_arvo_schema: Any = fastarvo.parsed_schema(arvo_schema)

        arvo_blob: Any = ParsedIngests.model_dump()

        with self.arvo_p.open("wb") as f:
            fastavro.writer(f, parsed_arvo_schema, arvo_blob)
