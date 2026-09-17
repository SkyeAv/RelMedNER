from __future__ import annotations

from typing import ClassVar, Self

import pytest

from relmedner.models import Entity, TrainingExample
from relmedner.scripts import NemotronPiiScript
from relmedner.types import DispatchedExample, Script, ScriptValues


class StubScript(Script):
    NAME: ClassVar[str] = "StubScript"

    def run(self: Self, values: ScriptValues) -> TrainingExample:
        text, label = values
        return TrainingExample(text=text or "", entities=[Entity(label=label or "", mentions=[text or ""])])


def test_subclasses_self_register_on_import() -> None:
    assert Script.REGISTRY["NemotronPiiScript"] is not None
    assert isinstance(Script.REGISTRY["NemotronPiiScript"], NemotronPiiScript)
    assert isinstance(Script.REGISTRY["StubScript"], StubScript)


def test_the_registry_keys_on_the_declared_name() -> None:
    assert all(Name == type(Instance).NAME for Name, Instance in Script.REGISTRY.items())


def test_dispatch_routes_by_name_and_preserves_declared_outputs() -> None:
    Dispatched: DispatchedExample = Script.dispatch("StubScript", (("entities",), ("Alice", "person")))
    Outputs, Example = Dispatched

    assert Outputs == ("entities",)
    assert Example.text == "Alice"
    assert Example.entities == [Entity(label="person", mentions=["Alice"])]


def test_dispatch_on_an_unregistered_name_raises() -> None:
    with pytest.raises(KeyError):
        Script.dispatch("NoSuchScript", (("entities",), ("Alice", "person")))


def test_the_placeholder_script_emits_no_tasks_yet() -> None:
    _, Example = Script.dispatch("NemotronPiiScript", (("entities",), ("some text", "[]")))

    assert Example.text == "some text"
    assert Example.populated() == frozenset()
