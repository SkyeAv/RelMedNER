from __future__ import annotations

import pytest
from apache_beam.options.pipeline_options import FlinkRunnerOptions, PipelineOptions, PortableOptions

from relmedner.clusters import YamlClusterParser
from relmedner.models import Cluster, WorkerNode
from relmedner.utils import package_version


@pytest.fixture(autouse=True)
def pinned_jobmanager(monkeypatch: pytest.MonkeyPatch) -> None:
    """jobmanager() probes the live lan route; keep these tests offline and deterministic"""
    monkeypatch.setattr(YamlClusterParser, "jobmanager", lambda self: "10.4.0.30")


def test_cluster_shape() -> None:
    ClusterSpec: Cluster = YamlClusterParser().parse_cluster()

    assert ClusterSpec.ssh_user == "sgoetz"
    assert tuple(worker.to_tuple() for worker in ClusterSpec.workers) == (
        ("local", 1, "8g"),
        ("10.2.9.11", 60, "110g"),
    )


def test_worker_node_tuple_without_an_override() -> None:
    Worker: WorkerNode = WorkerNode(host="10.2.9.19", slots=4, memory="6g")
    assert Worker.to_tuple() == ("10.2.9.19", 4, "6g")


def test_cluster_endpoints() -> None:
    Parser: YamlClusterParser = YamlClusterParser()

    assert Parser.flink_master() == "10.4.0.30:18081"
    assert Parser.worker_image() == f"localhost/relmedner-worker:{package_version()}"
    assert Parser.total_slots() == 61
    assert tuple(worker.host for worker in Parser.remotes()) == ("10.2.9.11",)
    assert Parser.local_worker().slots == 1


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValueError):
        WorkerNode.model_validate({"host": "h", "slots": 1, "memory": "1g", "arch": "amd64"})


def test_runner_options() -> None:
    Parser: YamlClusterParser = YamlClusterParser()
    Options: PipelineOptions = Parser.runner_options()

    assert Options.get_all_options(drop_default=True)["runner"] == "FlinkRunner"
    assert Options.get_all_options(drop_default=True)["streaming"] is True
    assert Options.get_all_options(drop_default=True)["no_wait_until_finish"] is True
    assert Options.get_all_options(drop_default=True).get("parallelism") is None
    assert Options.view_as(FlinkRunnerOptions).flink_master == Parser.flink_master()
    assert Options.view_as(PortableOptions).environment_config == Parser.worker_image()
    assert Options.view_as(PortableOptions).environment_type == "DOCKER"


def test_runner_options_use_explicit_parallelism() -> None:
    Options: PipelineOptions = YamlClusterParser().runner_options(2)

    assert Options.get_all_options(drop_default=True)["parallelism"] == 2
