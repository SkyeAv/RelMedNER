from __future__ import annotations

from importlib.resources.abc import Traversable
from typing import Any, Self

from yaml import CSafeLoader, load


class YamlParser:
    def __init__(self: Self, yaml_p: Traversable) -> None:
        self.yaml_p: Traversable = yaml_p

    def parse(self: Self) -> Any:
        # encoding is explicit because the packaged yaml carries non-ascii (em-dashes in
        # cluster.yaml comments) and a non-utf-8 locale -- LANG=en_US on the compute box resolves
        # to ISO-8859-1 -- silently mis-decodes them into control chars that CSafeLoader rejects
        with self.yaml_p.open("r", encoding="utf-8") as f:
            return load(f, Loader=CSafeLoader)
