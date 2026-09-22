from __future__ import annotations

from typing import Annotated

import cyclopts

from relmedner.clusters import YamlClusterParser
from relmedner.collect import collect_outputs
from relmedner.constants import DEFAULT_OUTPUT
from relmedner.deploy import deploy_cluster, run_cmd
from relmedner.enums import DedupMode
from relmedner.ingests import YamlIngestsParser
from relmedner.models import RunConfig
from relmedner.monitor import fetch_jobs, job_ids, watch_jobs
from relmedner.pipeline import BeamPipeline

APP: cyclopts.App = cyclopts.App()


@APP.command(name="build-dataset")
def build_dataset(
    test_run: Annotated[bool, cyclopts.Parameter(alias="-t")] = False,
    output: Annotated[str, cyclopts.Parameter(alias="-o")] = DEFAULT_OUTPUT,
    direct: Annotated[bool, cyclopts.Parameter("--direct", alias="-d")] = False,
    dedup_mode: Annotated[DedupMode, cyclopts.Parameter("--dedup-mode")] = DedupMode.NEAR,
) -> None:
    Config: RunConfig = RunConfig.from_flags(test_run, output, dedup_mode)
    if direct:
        # cluster-free path: the DirectRunner keeps the caller's local output path and needs no
        # flink cluster, no vpn route, and no podman -- see PLAN.md "Cluster reachability without VPN"
        BeamPipeline().run(Config)
        return
    Parser: YamlClusterParser = YamlClusterParser()
    ClusterSpec = Parser.parse_cluster()
    # flink slots cannot outstrip the number of declared datasets; each dataset is one source bundle
    parallelism: int = min(Parser.total_slots(), len(YamlIngestsParser().generate_tuples()))
    rest_url: str = Parser.rest_url()
    # submission is detached (see runner_options), so the beam job server exits immediately;
    # snapshot the pre-existing jobs and then follow ours through the jobmanager REST api
    before: frozenset[str] = job_ids(fetch_jobs(rest_url))
    BeamPipeline(options=Parser.runner_options(parallelism)).run(Config)
    watch_jobs(rest_url, before)
    # the driver runs on the head host, whose worker entry is the collection source this process
    # can read directly; every other worker's shards arrive over ssh (gateway hop from off-LAN)
    collect_outputs(tuple(ClusterSpec.workers), ClusterSpec.ssh_user, Config.artifact_name(), output, Parser.jobmanager(), run_cmd)


@APP.command(name="deploy-cluster")
def deploy_cluster_command(
    teardown: Annotated[bool, cyclopts.Parameter("--teardown", alias=["-t"])] = False,
    dry_run: Annotated[bool, cyclopts.Parameter("--dry-run", alias=["-d"])] = False,
) -> None:
    deploy_cluster(teardown=teardown, dry_run=dry_run)
