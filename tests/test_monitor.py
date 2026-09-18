from __future__ import annotations

from collections.abc import Iterator

import pytest

from relmedner.models import FlinkJob
from relmedner.monitor import job_ids, submitted_jobs, watch_jobs

REST: str = "http://10.0.0.1:18081"


def job(jid: str, state: str) -> FlinkJob:
    return FlinkJob(jid=jid, name="BeamApp", state=state)


def fetcher(*rounds: tuple[FlinkJob, ...]):
    pending: Iterator[tuple[FlinkJob, ...]] = iter(rounds)

    def fetch(rest_url: str) -> tuple[FlinkJob, ...]:
        assert rest_url == REST
        return next(pending)

    return fetch


def test_job_ids_and_submitted_jobs_ignore_preexisting_jobs() -> None:
    before: frozenset[str] = job_ids((job("old", "RUNNING"),))

    assert before == frozenset({"old"})
    assert submitted_jobs((job("old", "RUNNING"), job("new", "RUNNING")), before) == (job("new", "RUNNING"),)


def test_watch_jobs_returns_when_the_submitted_job_finishes() -> None:
    logged: list[str] = []
    fetch = fetcher((job("new", "RUNNING"),), (job("new", "FINISHED"),))

    watch_jobs(REST, frozenset(), fetch=fetch, sleep=lambda _: None, log=logged.append, poll_seconds=0)

    assert [line.split(": ")[-1] for line in logged] == ["RUNNING", "FINISHED"]


def test_watch_jobs_raises_on_a_failed_job() -> None:
    fetch = fetcher((job("new", "RUNNING"),), (job("new", "FAILED"),))

    with pytest.raises(SystemExit, match="job failed: new"):
        watch_jobs(REST, frozenset(), fetch=fetch, sleep=lambda _: None, log=lambda _: None, poll_seconds=0)


def test_watch_jobs_waits_for_a_job_that_has_not_registered_yet() -> None:
    fetch = fetcher((), (job("new", "RUNNING"),), (job("new", "FINISHED"),))

    watch_jobs(REST, frozenset(), fetch=fetch, sleep=lambda _: None, log=lambda _: None, poll_seconds=0)


def test_watch_jobs_treats_cancellation_as_failure() -> None:
    fetch = fetcher((job("new", "CANCELED"),))

    with pytest.raises(SystemExit, match="CANCELED"):
        watch_jobs(REST, frozenset(), fetch=fetch, sleep=lambda _: None, log=lambda _: None, poll_seconds=0)


def test_watch_jobs_fails_loudly_when_submission_never_registers() -> None:
    fetch = fetcher((), (), ())

    with pytest.raises(SystemExit, match="job never registered"):
        watch_jobs(
            REST,
            frozenset(),
            fetch=fetch,
            sleep=lambda _: None,
            log=lambda _: None,
            poll_seconds=0,
            empty_poll_limit=3,
        )
