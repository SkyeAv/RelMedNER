from __future__ import annotations

import pytest
from apache_beam.options.pipeline_options import FlinkRunnerOptions, PipelineOptions, PortableOptions

from relmedner.clusters import YamlClusterParser
from relmedner.models import Cluster, WorkerNode
from relmedner.utils import package_version


def test_cluster_shape() -> None:
    ClusterSpec: Cluster = YamlClusterParser().parse_cluster()

    assert ClusterSpec.ssh_user == "sgoetz"
    assert ClusterSpec.jobmanager == "10.2.9.11"
    assert tuple(worker.to_tuple() for worker in ClusterSpec.workers) == (
        ("10.2.9.11", 64, "112g", "/local_raid1/sgoetz/DBSTORE/FULLMAP/fullmap", "/local_raid1/sgoetz/DBSTORE/FULLMAP/outputs"),
        ("10.2.9.19", 16, "40g", "/ssd2/sgoetz/fullmap", "/ssd2/sgoetz/outputs"),
    )


def test_worker_node_tuple_without_an_override() -> None:
    Worker: WorkerNode = WorkerNode(host="10.2.9.19", slots=4, memory="6g", fullmap="/data/fullmap", outputs="/data/outputs")
    assert Worker.to_tuple() == ("10.2.9.19", 4, "6g", "/data/fullmap", "/data/outputs")


def test_cluster_endpoints() -> None:
    Parser: YamlClusterParser = YamlClusterParser()

    assert Parser.jobmanager() == "10.2.9.11"
    assert Parser.flink_master() == "10.2.9.11:18081"
    assert Parser.rest_url() == "http://10.2.9.11:18081"
    assert Parser.worker_image() == f"localhost/relmedner-worker:{package_version()}"
    assert Parser.total_slots() == 80
    # every worker — head included — gets a taskmanager + sdkworker stack
    assert tuple(worker.host for worker in Parser.remotes()) == ("10.2.9.11", "10.2.9.19")


def test_jobmanager_must_be_a_declared_worker() -> None:
    with pytest.raises(ValueError, match="not a declared worker"):
        Cluster.model_validate(
            {
                "ssh_user": "sgoetz",
                "jobmanager": "10.0.0.1",
                "workers": [{"host": "10.2.9.11", "slots": 1, "memory": "1g", "fullmap": "/f", "outputs": "/o"}],
            }
        )


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValueError):
        WorkerNode.model_validate({"host": "h", "slots": 1, "memory": "1g", "fullmap": "/f", "outputs": "/o", "arch": "amd64"})


def test_runner_options() -> None:
    Parser: YamlClusterParser = YamlClusterParser()
    Options: PipelineOptions = Parser.runner_options()

    assert Options.get_all_options(drop_default=True)["runner"] == "FlinkRunner"
    assert Options.get_all_options(drop_default=True)["streaming"] is True
    assert Options.get_all_options(drop_default=True)["no_wait_until_finish"] is True
    assert Options.get_all_options(drop_default=True).get("parallelism") is None
    assert Options.view_as(FlinkRunnerOptions).flink_master == Parser.flink_master()
    assert Options.view_as(PortableOptions).environment_config == "localhost:50000"
    assert Options.view_as(PortableOptions).environment_type == "EXTERNAL"


def test_runner_options_use_explicit_parallelism() -> None:
    Options: PipelineOptions = YamlClusterParser().runner_options(2)

    assert Options.get_all_options(drop_default=True)["parallelism"] == 2
