from __future__ import annotations

from enum import Enum


class ProcessingTypes(str, Enum):
    BABEL = "babel"
    SCRIPT = "script"
