"""drift guard + content sanity for the checked-in JSON Schema artifacts (US-011).

The artifacts under schemas/ are GENERATED from relmedner.models via relmedner.schemas (the
single source of truth, imported by both the CLI and this test so they cannot diverge). The
drift guard regenerates and asserts byte-identity: ANY models.py change fails CI here until
`uv run relmedner schema` reruns. Sanity assertions are pure-dict checks (no jsonschema
library -- CI validates by regenerating; editors validate via the yaml-language-server pointer).
"""

import json
import pathlib

from relmedner.cli import APP
from relmedner.schemas import cluster_schema, ingests_schema

SCHEMAS: pathlib.Path = pathlib.Path("schemas")


# the CLI's exact rendering contract; the drift guard pins it so regeneration stays a no-op
def render(schema: dict) -> str:
    return json.dumps(schema, indent=2) + "\n"


def load_ingests() -> dict:
    return json.loads((SCHEMAS / "ingests.schema.json").read_text(encoding="utf-8"))


def load_cluster() -> dict:
    return json.loads((SCHEMAS / "cluster.schema.json").read_text(encoding="utf-8"))


def test_ingests_schema_is_byte_current() -> None:
    checked_in: str = (SCHEMAS / "ingests.schema.json").read_text(encoding="utf-8")
    assert checked_in == render(ingests_schema())


def test_cluster_schema_is_byte_current() -> None:
    checked_in: str = (SCHEMAS / "cluster.schema.json").read_text(encoding="utf-8")
    assert checked_in == render(cluster_schema())


def test_ingests_top_level_properties() -> None:
    schema: dict = load_ingests()
    # x-defaults proves aliases render under their YAML names; gazetteer is the US-010 overlay.
    # filters is NOT top-level: it is a DatasetBase (per-dataset) field, asserted in $defs below
    assert {"x-defaults", "gazetteer"} <= set(schema["properties"])


def test_ingests_filters_live_on_every_dataset_shape() -> None:
    # DatasetBase is not its own $defs entry: pydantic flattens the inherited fields into each
    # concrete dataset shape, so `filters` (US-008) must appear on EVERY source arm -- a new
    # arm that silently lacked it would accept a declared filter and never apply it
    defs: dict = load_ingests()["$defs"]
    for shape in ("HuggingFaceDataset", "HuggingFaceJsonDataset", "LocalAvroDataset", "LocalDelimitedDataset"):
        assert "filters" in defs[shape]["properties"]


def test_datasets_carries_source_discriminator() -> None:
    items: dict = load_ingests()["properties"]["datasets"]["items"]
    assert items["discriminator"] == {
        "propertyName": "source",
        "mapping": {
            "hf": "#/$defs/HuggingFaceDataset",
            "hf_json": "#/$defs/HuggingFaceJsonDataset",
            "local": "#/$defs/LocalAvroDataset",
            "local_delimited": "#/$defs/LocalDelimitedDataset",
        },
    }


def test_cluster_root_is_cluster_and_worker_slots_minimum_one() -> None:
    schema: dict = load_cluster()
    assert schema["title"] == "Cluster"
    # jobmanager: the head host (runs the jobmanager stack AND its own worker duty, see cluster.yaml)
    assert set(schema["properties"]) == {"ssh_user", "jobmanager", "workers"}
    assert "WorkerNode" in schema["$defs"]
    assert schema["$defs"]["WorkerNode"]["properties"]["slots"]["minimum"] == 1


def test_schema_command_parses_with_output_dir_alias() -> None:
    Command, Bound, _ = APP.parse_args(["schema", "-o", "elsewhere/"], exit_on_error=False)
    assert Command.__name__ == "schema_command"
    assert Bound.arguments.get("output_dir") == "elsewhere/"
