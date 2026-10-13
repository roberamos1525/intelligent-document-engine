"""Persistent background job queue with worker threads and retry backoff.

Jobs live in SQLite, so they survive restarts. Workers claim jobs with an
atomic UPDATE and hold a lease; on start-up, RUNNING jobs whose lease has
expired (their worker died) are requeued. Failures are retried with
exponential backoff and jitter until ``max_attempts`` is reached, except for
errors that can never succeed on retry (validation, not found).
"""

from __future__ import annotations

import random
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from idengine.config import JobSettings
from idengine.errors import (
    ConflictError,
    EngineError,
    NotFoundError,
    UnsupportedFileTypeError,
    ValidationError,
)
from idengine.observability.logging import get_logger, job_id_var
from idengine.observability.metrics import EngineMetrics
from idengine.storage.jobs import JobRepository
from idengine.storage.models import JobRecord, JobStatus

log = get_logger("jobs")

Handler = Callable[[dict[str, Any]], Any]

_PERMANENT_ERRORS: tuple[type[Exception], ...] = (
    ValidationError,
    UnsupportedFileTypeError,
    NotFoundError,
    ConflictError,
)


@dataclass(frozen=True, slots=True)
class _Registration:
    handler: Handler
    max_attempts: int | None


def backoff_delay(
    attempt: int, *, base: float, maximum: float, rng: random.Random | None = None
) -> float:
    """Exponential backoff with "equal jitter": half fixed, half random."""
    exp = min(maximum, base * (2 ** max(attempt - 1, 0)))
    jitter = (rng or random).random()
    return exp / 2 + jitter * exp / 2


class JobQueue:
    def __init__(
        self,
        repository: JobRepository,
        settings: JobSettings,
        metrics: EngineMetrics,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self.metrics = metrics
        self._rng = rng or random.Random()
        self._handlers: dict[str, _Registration] = {}
        self._wakeup = threading.Condition()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- registration / submission -------------------------------------------------
    def register(self, job_type: str, handler: Handler, *, max_attempts: int | None = None) -> None:
        self._handlers[job_type] = _Registration(handler, max_attempts)

    @property
    def job_types(self) -> list[str]:
        return sorted(self._handlers)

    def submit(
        self,
        job_type: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: int = 0,
        delay_seconds: float = 0.0,
    ) -> JobRecord:
        registration = self._handlers.get(job_type)
        if registration is None:
            raise ValidationError(
                f"unknown job type '{job_type}'", details={"known": self.job_types}
            )
        if not -100 <= priority <= 100:
            raise ValidationError("priority must be between -100 and 100")
        job = self.repository.enqueue(
            job_type,
            payload or {},
            max_attempts=registration.max_attempts or self.settings.max_attempts,
            priority=priority,
            delay_seconds=delay_seconds,
        )
        self.metrics.jobs.inc(labels={"type": job_type, "outcome": "submitted"})
        with self._wakeup:
            self._wakeup.notify()
        return job

    # -- execution -------------------------------------------------------------------
    def run_once(self, worker_id: str = "inline") -> JobRecord | None:
        """Claim and execute one runnable job; returns it (final state) or None."""
        job = self.repository.claim(worker_id, self.settings.lease_seconds)
        if job is None:
            return None
        token = job_id_var.set(job.id)
        started = time.perf_counter()
        try:
            registration = self._handlers.get(job.type)
            if registration is None:
                self.repository.fail(job.id, f"no handler for job type '{job.type}'", retry_in=None)
                self.metrics.jobs.inc(labels={"type": job.type, "outcome": "failed"})
                return self.repository.get(job.id)
            try:
                result = registration.handler(job.payload)
            except Exception as exc:  # handlers may raise anything; record and decide
                self._handle_failure(job, exc)
            else:
                self.repository.complete(job.id, result)
                self.metrics.jobs.inc(labels={"type": job.type, "outcome": "succeeded"})
                log.info("job succeeded", extra={"job_type": job.type, "attempt": job.attempts})
            finally:
                self.metrics.job_latency.observe(time.perf_counter() - started, {"type": job.type})
            return self.repository.get(job.id)
        finally:
            job_id_var.reset(token)

    def _handle_failure(self, job: JobRecord, exc: Exception) -> None:
        message = exc.message if isinstance(exc, EngineError) else f"{type(exc).__name__}: {exc}"
        permanent = isinstance(exc, _PERMANENT_ERRORS)
        exhausted = job.attempts >= job.max_attempts
        retry_in = None
        if not permanent and not exhausted:
            retry_in = backoff_delay(
                job.attempts,
                base=self.settings.backoff_base_seconds,
                maximum=self.settings.backoff_max_seconds,
                rng=self._rng,
            )
        status = self.repository.fail(job.id, message, retry_in=retry_in)
        outcome = "retried" if status is JobStatus.QUEUED else "failed"
        self.metrics.jobs.inc(labels={"type": job.type, "outcome": outcome})
        log.warning(
            "job failed",
            extra={
                "job_type": job.type,
                "attempt": job.attempts,
                "max_attempts": job.max_attempts,
                "retry_in": retry_in,
                "error": message,
                "trace": None if isinstance(exc, EngineError) else traceback.format_exc(limit=5),
            },
        )

    def drain(self, *, timeout: float = 60.0, include_delayed: bool = False) -> int:
        """Run jobs inline until none are runnable; returns how many ran.

        With ``include_delayed`` the call also waits for jobs that are
        scheduled for retry within the timeout window.
        """
        deadline = time.monotonic() + timeout
        ran = 0
        while time.monotonic() < deadline:
            if self.run_once() is not None:
                ran += 1
                continue
            if not include_delayed or self.repository.counts()[JobStatus.QUEUED.value] == 0:
                break
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return ran

    # -- worker threads ------------------------------------------------------------
    def _worker(self, worker_id: str) -> None:
        log.info("worker started", extra={"worker": worker_id})
        while not self._stop.is_set():
            try:
                job = self.run_once(worker_id)
            except Exception:  # never let a worker thread die (e.g. transient DB lock)
                log.exception("worker loop error", extra={"worker": worker_id})
                job = None
            if job is None:
                with self._wakeup:
                    self._wakeup.wait(self.settings.poll_interval_seconds)
        log.info("worker stopped", extra={"worker": worker_id})

    def start(self, workers: int | None = None) -> None:
        count = self.settings.workers if workers is None else workers
        if self._threads or count <= 0:
            return
        recovered = self.repository.recover_expired_leases()
        if recovered:
            log.warning("requeued jobs with expired leases", extra={"count": recovered})
        self._stop.clear()
        for i in range(count):
            thread = threading.Thread(
                target=self._worker,
                args=(f"worker-{i + 1}",),
                name=f"idengine-worker-{i + 1}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        with self._wakeup:
            self._wakeup.notify_all()
        for thread in self._threads:
            thread.join(timeout)
        self._threads.clear()

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def wait_for(self, job_id: str, *, timeout: float = 30.0, poll: float = 0.05) -> JobRecord:
        """Block until a job reaches a terminal state (or the timeout elapses)."""
        deadline = time.monotonic() + timeout
        while True:
            job = self.repository.get(job_id)
            if job.status.terminal or time.monotonic() >= deadline:
                return job
            time.sleep(poll)
