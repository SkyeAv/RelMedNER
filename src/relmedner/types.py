from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self

from relmedner.models import TrainingExample

ScriptValues = tuple[str | None, ...]
ScriptPayload = tuple[tuple[str, ...], ScriptValues]
DispatchedExample = tuple[tuple[str, ...], TrainingExample]


class Script(ABC):
    NAME: ClassVar[str]
    REGISTRY: ClassVar[dict[str, Script]] = {}

    def __init_subclass__(cls: type[Script], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        Script.REGISTRY[cls.NAME] = cls()

    @classmethod
    def dispatch(cls, name: str, payload: ScriptPayload) -> DispatchedExample:
        outputs, values = payload
        return (outputs, cls.REGISTRY[name].run(values))

    @abstractmethod
    def run(self: Self, values: ScriptValues) -> TrainingExample:
        """turns one streamed row into one gliner2 training example"""
