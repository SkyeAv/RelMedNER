from __future__ import annotations

from typing import Self

from apache_beam.options.pipeline_options import PipelineOptions

from relmedner.constants import (
    CLUSTER_YAML,
    FLINK_REST_PORT,
    FLINK_VERSION,
    WORKER_IMAGE_NAME,
    WORKER_POOL_PORT,
)
from relmedner.models import Cluster, WorkerNode
from relmedner.parsers import YamlParser
from relmedner.utils import package_version


class YamlClusterParser(YamlParser):
    def __init__(self: Self) -> None:
        super().__init__(CLUSTER_YAML)

    def parse_cluster(self: Self) -> Cluster:
        return Cluster.model_validate(self.parse())

    def remotes(self: Self) -> list[WorkerNode]:
        """every worker that runs a taskmanager + sdkworker — all of them, including the head

        compose stacks and image pushes always go through ssh (the head deploys to itself the same
        way), so there is no local/remote split anymore
        """
        return list(self.parse_cluster().workers)

    def jobmanager(self: Self) -> str:
        """head host from cluster.yaml — deploy, submit, and collection all run there"""
        return self.parse_cluster().jobmanager

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
            # EXTERNAL: user code runs in the per-node sdkworker pool (see the compose files), which
            # owns the fullmap volume mount; DOCKER environment cannot mount host dirs (beam #19240).
            # localhost resolves because both taskmanager and sdkworker use host networking.
            f"--environment_config=localhost:{WORKER_POOL_PORT}",
            "--environment_type=EXTERNAL",
            "--sdk_location=container",
        ]
        return PipelineOptions(flags)

    def total_slots(self: Self) -> int:
        return sum(worker.slots for worker in self.parse_cluster().workers)
