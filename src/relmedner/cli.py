from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

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
from relmedner.schemas import cluster_schema, ingests_schema
from relmedner.validate import PubMedClient, resolve_trust_settings, suggested_yaml, validate_sources, write_report

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
    # every slot does work: the pipeline's "fan rows out across workers" Reshuffle spreads streamed
    # rows over all operators, so parallelism is no longer capped at one source bundle per dataset
    parallelism: int = Parser.total_slots()
    rest_url: str = Parser.rest_url()
    # submission is detached (see runner_options), so the beam job server exits immediately;
    # snapshot the pre-existing jobs and then follow ours through the jobmanager REST api
    before: frozenset[str] = job_ids(fetch_jobs(rest_url))
    BeamPipeline(options=Parser.runner_options(parallelism)).run(Config)
    watch_jobs(rest_url, before)
    # the driver runs on the head host, whose worker entry is the collection source this process
    # can read directly; every other worker's shards arrive over ssh (gateway hop from off-LAN)
    collect_outputs(tuple(ClusterSpec.workers), ClusterSpec.ssh_user, Config.artifact_name(), output, Parser.jobmanager(), run_cmd)


@APP.command(name="validate-trust")
def validate_trust_command(
    sample_size: Annotated[int | None, cyclopts.Parameter("--sample-size", alias="-n")] = None,
    source: Annotated[str | None, cyclopts.Parameter("--source", alias="-s")] = None,
    report: Annotated[str | None, cyclopts.Parameter("--report", alias="-o")] = None,
    test_run: Annotated[bool, cyclopts.Parameter(alias="-t")] = False,
) -> None:
    """offline literature validation suggesting per-source trust values (docs/weighting.md).
    Samples records through the real dispatch path, queries PubMed E-utilities, writes the
    per-record JSONL report to --report and prints the suggested `trust:` yaml for
    hand-committing to ingests.yaml. Settings come from the optional top-level `x-trust:`
    section of ingests.yaml, overridden flag-by-flag; secrets (NCBI_API_KEY) come from env only"""
    Ingests = YamlIngestsParser().parse_ingests()
    Size, Report = resolve_trust_settings(sample_size, report, Ingests.x_trust)
    client = PubMedClient()
    Config: RunConfig = RunConfig.from_flags(test_run, Report, DedupMode.OFF)
    summaries, records = validate_sources(Ingests, Config, client, Size, source)
    write_report(Path(Report), summaries, records)
    print(suggested_yaml(summaries))


@APP.command(name="schema")
def schema_command(
    output_dir: Annotated[str, cyclopts.Parameter(alias="-o")] = "schemas/",
) -> None:
    # regenerate the checked-in JSON Schema artifacts from the models (single source of truth:
    # relmedner.schemas); tests/test_schemas.py is the drift guard that fails CI when a models.py
    # change lands without rerunning this command. Byte-stable rendering (indent=2 + trailing
    # newline) keeps regeneration a no-op on a fresh checkout
    Target: Path = Path(output_dir)
    Target.mkdir(parents=True, exist_ok=True)
    Artifacts: tuple[tuple[str, dict[str, Any]], ...] = (
        ("ingests.schema.json", ingests_schema()),
        ("cluster.schema.json", cluster_schema()),
    )
    for name, schema in Artifacts:
        (Target / name).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


@APP.command(name="deploy-cluster")
def deploy_cluster_command(
    teardown: Annotated[bool, cyclopts.Parameter("--teardown", alias=["-t"])] = False,
    dry_run: Annotated[bool, cyclopts.Parameter("--dry-run", alias=["-d"])] = False,
) -> None:
    deploy_cluster(teardown=teardown, dry_run=dry_run)
