from __future__ import annotations

from importlib.resources.abc import Traversable
from typing import Any, Self

from yaml import CSafeLoader, load


class YamlParser:
    def __init__(self: Self, yaml_p: Traversable) -> None:
        self.yaml_p: Traversable = yaml_p

    def parse(self: Self) -> Any:
        with self.yaml_p.open("r") as f:
            return load(f, Loader=CSafeLoader)
