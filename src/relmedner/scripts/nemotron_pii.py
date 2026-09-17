from __future__ import annotations

from typing import ClassVar, Self

from relmedner.models import TrainingExample
from relmedner.types import Script, ScriptValues


class NemotronPiiScript(Script):
    """placeholder to get the script dispatch machinery to work"""

    NAME: ClassVar[str] = "NemotronPiiScript"

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        text, _ = values
        return TrainingExample(text=text or "")
