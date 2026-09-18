from __future__ import annotations

from collections.abc import Callable, Iterable
from json import loads
from time import sleep as time_sleep
from urllib.request import urlopen

from relmedner.models import FlinkJob

# flink's own vocabulary — everything else (CREATED, RUNNING, RESTARTING, ...) is still in flight
TERMINAL_STATES: frozenset[str] = frozenset({"FINISHED", "CANCELED", "FAILED"})
FAILED_STATES: frozenset[str] = frozenset({"CANCELED", "FAILED"})
POLL_SECONDS: float = 10.0
EMPTY_POLL_LIMIT: int = 30  # five minutes; a detached submission must register before this

Fetcher = Callable[[str], tuple[FlinkJob, ...]]


def fetch_jobs(rest_url: str) -> tuple[FlinkJob, ...]:
    with urlopen(f"{rest_url}/jobs/overview", timeout=30) as response:  # noqa: S310 — fixed http scheme from cluster.yaml
        payload: dict = loads(response.read())
    return tuple(FlinkJob.model_validate(job) for job in payload.get("jobs", ()))


def job_ids(jobs: Iterable[FlinkJob]) -> frozenset[str]:
    return frozenset(job.jid for job in jobs)


def submitted_jobs(jobs: Iterable[FlinkJob], before: frozenset[str]) -> tuple[FlinkJob, ...]:
    return tuple(job for job in jobs if job.jid not in before)


def watch_jobs(
    rest_url: str,
    before: frozenset[str],
    fetch: Fetcher = fetch_jobs,
    sleep: Callable[[float], None] = time_sleep,
    log: Callable[[str], None] = print,
    poll_seconds: float = POLL_SECONDS,
    empty_poll_limit: int = EMPTY_POLL_LIMIT,
) -> None:
    """block until every job submitted by this run reaches a terminal flink state

    polls the jobmanager REST api rather than the beam job server, which is torn down as soon as
    the detached submission returns; raises SystemExit so a failed remote job fails the command
    """
    reported: dict[str, str] = {}
    empty_polls: int = 0
    while True:
        tracked: tuple[FlinkJob, ...] = submitted_jobs(fetch(rest_url), before)
        if not tracked:
            empty_polls += 1
            if empty_polls == 1:
                log(f"waiting for the job to register with {rest_url}")
            if empty_polls >= empty_poll_limit:
                raise SystemExit(f"job never registered with {rest_url}")
            sleep(poll_seconds)
            continue
        empty_polls = 0

        for job in tracked:
            if reported.get(job.jid) != job.state:
                log(f"{job.jid} {job.name}: {job.state}")
                reported[job.jid] = job.state

        if tracked and all(job.state in TERMINAL_STATES for job in tracked):
            failures: tuple[FlinkJob, ...] = tuple(job for job in tracked if job.state in FAILED_STATES)
            if failures:
                raise SystemExit("job failed: " + ", ".join(f"{job.jid} ({job.state})" for job in failures) + f" — see {rest_url}")
            return

        sleep(poll_seconds)
