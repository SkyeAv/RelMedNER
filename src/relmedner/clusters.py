from __future__ import annotations

from socket import AF_INET, SOCK_DGRAM, socket
from typing import Self

from apache_beam.options.pipeline_options import PipelineOptions

from relmedner.constants import CLUSTER_YAML, FLINK_REST_PORT, FLINK_VERSION, LOCAL_HOST, WORKER_IMAGE_NAME
from relmedner.models import Cluster, WorkerNode
from relmedner.parsers import YamlParser
from relmedner.utils import package_version


class YamlClusterParser(YamlParser):
    def __init__(self: Self) -> None:
        super().__init__(CLUSTER_YAML)

    def parse_cluster(self: Self) -> Cluster:
        return Cluster.model_validate(self.parse())

    def remotes(self: Self) -> list[WorkerNode]:
        return [worker for worker in self.parse_cluster().workers if worker.host != LOCAL_HOST]

    def local_worker(self: Self) -> WorkerNode:
        return next(worker for worker in self.parse_cluster().workers if worker.host == LOCAL_HOST)

    def jobmanager(self: Self) -> str:
        """laptop address as remotes see it — re-resolved per run because wifi and vpn move it"""
        with socket(AF_INET, SOCK_DGRAM) as probe:
            probe.connect((self.remotes()[0].host, 1))
            return probe.getsockname()[0]

    def flink_master(self: Self) -> str:
        return f"{self.jobmanager()}:{FLINK_REST_PORT}"

    def rest_url(self: Self) -> str:
        return f"http://{self.flink_master()}"

    def worker_image(self: Self) -> str:
        return f"{WORKER_IMAGE_NAME}:{package_version()}"

    def runner_options(self: Self, parallelism: int | None = None) -> PipelineOptions:
        flags: list[str] = [
            "--runner=FlinkRunner",
            f"--flink_master={self.flink_master()}",
            f"--flink_version={FLINK_VERSION}",
            # streaming mode: batch jobs release empty result partitions early,
            # which races with slow remote consumers and kills the job
            "--streaming",
            "--no_wait_until_finish",
        ]
        if parallelism:
            flags.append(f"--parallelism={parallelism}")
        flags += [
            f"--environment_config={self.worker_image()}",
            "--environment_type=DOCKER",
            "--sdk_location=container",
        ]
        return PipelineOptions(flags)

    def total_slots(self: Self) -> int:
        return sum(worker.slots for worker in self.parse_cluster().workers)
