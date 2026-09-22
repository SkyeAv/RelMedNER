from __future__ import annotations

from enum import StrEnum


class ProcessingTypes(StrEnum):
    FULLMAP = "fullmap"
    SCRIPT = "script"


class OutputShapes(StrEnum):
    ENTITIES = "entities"
    CLASSIFICATIONS = "classifications"
    STRUCTURES = "structures"
    RELATIONS = "relations"


class DedupMode(StrEnum):
    OFF = "off"
    EXACT = "exact"
    NEAR = "near"
