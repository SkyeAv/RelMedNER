from __future__ import annotations

from typing import Self, Any
from importlib.resources.abc import Traversable

from yaml import load, CSafeLoader

class YamlParser:
    def __init__(self: Self, yaml_p: Traversable) -> None:
        self.yaml_p: Traversable = p

    def parse(self: Self) -> Any:
        with self.yaml_p.open("r") as f:
            return load(f, loader=CSafeLoader)
