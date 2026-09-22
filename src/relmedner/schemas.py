from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from relmedner.models import Cluster, YamlIngests


def ingests_schema() -> dict[str, Any]:
    """the single source of truth for the ingests.yaml JSON Schema. WHY a function and not a
    checked-in hand-edited file: the schema is generated from YamlIngests, so ANY models.py
    change (a new field, a new constraint) flows here automatically, and the drift guard in
    tests/test_schemas.py fails CI until `relmedner schema` regenerates the artifact. Aliased
    fields render under their YAML names (validation mode uses aliases by default), so the
    anchor host appears as `x-defaults`; validation shape only -- tuple packing
    (to_tuple/tuple_fields) is internal and deliberately NOT represented"""
    return TypeAdapter(YamlIngests).json_schema()


def cluster_schema() -> dict[str, Any]:
    """same single-source contract as ingests_schema(), for cluster.yaml -> Cluster"""
    return Cluster.model_json_schema()
