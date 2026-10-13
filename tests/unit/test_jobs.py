from __future__ import annotations

import random
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from idengine.config import JobSettings
from idengine.errors import NotFoundError, ValidationError
from idengine.jobs import JobQueue, backoff_delay
from idengine.observability.metrics import EngineMetrics
from idengine.storage import Database, JobRepository, JobStatus


@pytest.fixture
def queue(tmp_path: Path) -> Iterator[JobQueue]:
    db = Database(tmp_path / "jobs.sqlite3")
    db.migrate()
    settings = JobSettings(
        workers=2,
        max_attempts=3,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
        poll_interval_seconds=0.02,
    )
    q = JobQueue(JobRepository(db), settings, EngineMetrics(), rng=random.Random(1))
    yield q
    q.stop()
    db.close()


def test_successful_job_records_result(queue: JobQueue) -> None:
    queue.register("echo", lambda payload: {"echo": payload["value"]})
    job = queue.submit("echo", {"value": 42})
    assert job.status is JobStatus.QUEUED and job.max_attempts == 3
    done = queue.run_once()
    assert done is not None and done.status is JobStatus.SUCCEEDED
    assert done.result == {"echo": 42} and done.started_at and done.finished_at
    assert queue.run_once() is None
    assert queue.metrics.jobs.value({"type": "echo", "outcome": "succeeded"}) == 1


def test_transient_failures_are_retried_until_success(queue: JobQueue) -> None:
    calls: list[int] = []

    def flaky(_: dict[str, Any]) -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("temporary outage")
        return "ok"

    queue.register("flaky", flaky)
    job = queue.submit("flaky")
    assert queue.drain(timeout=5, include_delayed=True) == 3
    final = queue.repository.get(job.id)
    assert final.status is JobStatus.SUCCEEDED and final.attempts == 3
    assert queue.metrics.jobs.value({"type": "flaky", "outcome": "retried"}) == 2


def test_attempts_are_exhausted(queue: JobQueue) -> None:
    queue.register(
        "broken", lambda _: (_ for _ in ()).throw(RuntimeError("always")), max_attempts=2
    )
    job = queue.submit("broken")
    queue.drain(timeout=5, include_delayed=True)
    final = queue.repository.get(job.id)
    assert final.status is JobStatus.FAILED and final.attempts == 2
    assert final.error == "RuntimeError: always"


@pytest.mark.parametrize("error", [ValidationError("bad input"), NotFoundError("gone")])
def test_permanent_errors_are_not_retried(queue: JobQueue, error: Exception) -> None:
    def handler(_: dict[str, Any]) -> None:
        raise error

    queue.register("perm", handler)
    job = queue.submit("perm")
    queue.drain(timeout=5, include_delayed=True)
    final = queue.repository.get(job.id)
    assert final.status is JobStatus.FAILED and final.attempts == 1


def test_submission_validation(queue: JobQueue) -> None:
    with pytest.raises(ValidationError, match="unknown job type"):
        queue.submit("nope")
    queue.register("x", lambda _: None)
    with pytest.raises(ValidationError, match="priority"):
        queue.submit("x", priority=1000)
    assert queue.job_types == ["x"]


def test_orphaned_job_type_fails_cleanly(queue: JobQueue) -> None:
    job = queue.repository.enqueue("legacy", {}, max_attempts=1)
    final = queue.run_once()
    assert final is not None and final.id == job.id and final.status is JobStatus.FAILED
    assert "no handler" in (final.error or "")


def test_worker_threads_process_concurrently(queue: JobQueue) -> None:
    seen: set[str] = set()
    lock = threading.Lock()

    def work(payload: dict[str, Any]) -> None:
        time.sleep(0.01)
        with lock:
            seen.add(threading.current_thread().name)

    queue.register("work", work)
    jobs = [queue.submit("work", {"i": i}) for i in range(8)]
    queue.start()
    assert queue.running
    final = [queue.wait_for(j.id, timeout=10) for j in jobs]
    assert all(j.status is JobStatus.SUCCEEDED for j in final)
    assert seen <= {"idengine-worker-1", "idengine-worker-2"}
    queue.stop()
    assert not queue.running


def test_start_recovers_expired_leases(queue: JobQueue) -> None:
    queue.register("t", lambda _: "done")
    job = queue.submit("t")
    queue.repository.claim("crashed-worker", lease_seconds=0.0)
    time.sleep(0.01)
    queue.start(workers=1)
    assert queue.wait_for(job.id, timeout=10).status is JobStatus.SUCCEEDED


def test_backoff_delay_bounds() -> None:
    rng = random.Random(7)
    delays = [backoff_delay(a, base=1.0, maximum=8.0, rng=rng) for a in range(1, 7)]
    for attempt, delay in enumerate(delays, start=1):
        cap = min(8.0, 2 ** (attempt - 1))
        assert cap / 2 <= delay <= cap
    assert backoff_delay(1, base=0.0, maximum=10.0) == 0.0
