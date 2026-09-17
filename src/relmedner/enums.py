from __future__ import annotations

from enum import StrEnum


class ProcessingTypes(StrEnum):
    BABEL = "babel"
    SCRIPT = "script"


class OutputShapes(StrEnum):
    ENTITIES = "entities"
    CLASSIFICATIONS = "classifications"
    STRUCTURES = "structures"
    RELATIONS = "relations"
