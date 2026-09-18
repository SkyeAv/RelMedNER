from __future__ import annotations

from typing import Annotated

import cyclopts

from relmedner.clusters import YamlClusterParser
from relmedner.collect import collect_outputs
from relmedner.constants import DEFAULT_OUTPUT, LOCAL_HOST
from relmedner.deploy import deploy_cluster, run_cmd
from relmedner.ingests import YamlIngestsParser
from relmedner.models import RunConfig
from relmedner.monitor import fetch_jobs, job_ids, watch_jobs
from relmedner.pipeline import BeamPipeline

APP: cyclopts.App = cyclopts.App()


@APP.command(name="build-dataset")
def build_dataset(
    test_run: Annotated[bool, cyclopts.Parameter(alias="-t")] = False,
    output: Annotated[str, cyclopts.Parameter(alias="-o")] = DEFAULT_OUTPUT,
) -> None:
    Parser: YamlClusterParser = YamlClusterParser()
    ClusterSpec = Parser.parse_cluster()
    # flink slots cannot outstrip the number of declared datasets; each dataset is one source bundle
    parallelism: int = min(Parser.total_slots(), len(YamlIngestsParser().generate_tuples()))
    Config: RunConfig = RunConfig.from_flags(test_run, output)
    rest_url: str = Parser.rest_url()
    # submission is detached (see runner_options), so the beam job server exits immediately;
    # snapshot the pre-existing jobs and then follow ours through the jobmanager REST api
    before: frozenset[str] = job_ids(fetch_jobs(rest_url))
    BeamPipeline(options=Parser.runner_options(parallelism)).run(Config)
    watch_jobs(rest_url, before)
    collect_outputs(tuple(ClusterSpec.workers), ClusterSpec.ssh_user, Config.artifact_name(), output, LOCAL_HOST, run_cmd)


@APP.command(name="deploy-cluster")
def deploy_cluster_command(
    teardown: Annotated[bool, cyclopts.Parameter("--teardown", alias=["-t"])] = False,
    dry_run: Annotated[bool, cyclopts.Parameter("--dry-run", alias=["-d"])] = False,
) -> None:
    deploy_cluster(teardown=teardown, dry_run=dry_run)
