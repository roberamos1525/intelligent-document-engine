"""Persistent job table with atomic claim/lease semantics."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

from idengine.errors import ConflictError, NotFoundError, ValidationError
from idengine.storage.db import Database, new_id, utcnow
from idengine.storage.models import JobRecord, JobStatus


class JobRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def enqueue(
        self,
        job_type: str,
        payload: dict[str, Any],
        *,
        max_attempts: int,
        priority: int = 0,
        delay_seconds: float = 0.0,
    ) -> JobRecord:
        job_id = new_id("job")
        now = utcnow()
        self.db.execute(
            "INSERT INTO jobs (id, type, payload, status, priority, attempts, max_attempts,"
            " run_after, created_at, updated_at) VALUES (?,?,?,?,?,0,?,?,?,?)",
            (
                job_id,
                job_type,
                json.dumps(payload),
                JobStatus.QUEUED.value,
                priority,
                max_attempts,
                time.time() + delay_seconds,
                now,
                now,
            ),
        )
        return self.get(job_id)

    def get(self, job_id: str) -> JobRecord:
        row = self.db.query_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            raise NotFoundError(f"job {job_id} not found", details={"id": job_id})
        return JobRecord.from_row(row)

    def claim(self, worker_id: str, lease_seconds: float) -> JobRecord | None:
        """Atomically move the next runnable job to RUNNING and lease it to ``worker_id``."""
        now = time.time()
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE status = ? AND run_after <= ? "
                "ORDER BY priority DESC, run_after, created_at LIMIT 1",
                (JobStatus.QUEUED.value, now),
            ).fetchone()
            if row is None:
                return None
            stamp = utcnow()
            conn.execute(
                "UPDATE jobs SET status = ?, attempts = attempts + 1, locked_by = ?, "
                "lease_expires = ?, started_at = COALESCE(started_at, ?), updated_at = ? "
                "WHERE id = ?",
                (JobStatus.RUNNING.value, worker_id, now + lease_seconds, stamp, stamp, row["id"]),
            )
        return self.get(row["id"])

    def complete(self, job_id: str, result: Any) -> None:
        stamp = utcnow()
        self.db.execute(
            "UPDATE jobs SET status = ?, result = ?, error = NULL, locked_by = NULL, "
            "lease_expires = NULL, finished_at = ?, updated_at = ? WHERE id = ? AND status = ?",
            (
                JobStatus.SUCCEEDED.value,
                json.dumps(result, default=str),
                stamp,
                stamp,
                job_id,
                JobStatus.RUNNING.value,
            ),
        )

    def fail(self, job_id: str, error: str, *, retry_in: float | None) -> JobStatus:
        """Record a failure: requeue after ``retry_in`` seconds or mark FAILED."""
        stamp = utcnow()
        if retry_in is None:
            self.db.execute(
                "UPDATE jobs SET status = ?, error = ?, locked_by = NULL, lease_expires = NULL, "
                "finished_at = ?, updated_at = ? WHERE id = ?",
                (JobStatus.FAILED.value, error[:4000], stamp, stamp, job_id),
            )
            return JobStatus.FAILED
        self.db.execute(
            "UPDATE jobs SET status = ?, error = ?, locked_by = NULL, lease_expires = NULL, "
            "run_after = ?, updated_at = ? WHERE id = ?",
            (JobStatus.QUEUED.value, error[:4000], time.time() + retry_in, stamp, job_id),
        )
        return JobStatus.QUEUED

    def cancel(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        if job.status is not JobStatus.QUEUED:
            raise ConflictError(
                f"only queued jobs can be cancelled (job is {job.status.value})",
                details={"status": job.status.value},
            )
        stamp = utcnow()
        self.db.execute(
            "UPDATE jobs SET status = ?, finished_at = ?, updated_at = ? "
            "WHERE id = ? AND status = ?",
            (JobStatus.CANCELLED.value, stamp, stamp, job_id, JobStatus.QUEUED.value),
        )
        return self.get(job_id)

    def retry(self, job_id: str) -> JobRecord:
        """Manually requeue a FAILED or CANCELLED job with a fresh attempt budget."""
        job = self.get(job_id)
        if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
            raise ConflictError(
                f"only failed or cancelled jobs can be retried (job is {job.status.value})"
            )
        self.db.execute(
            "UPDATE jobs SET status = ?, attempts = 0, run_after = ?, finished_at = NULL, "
            "updated_at = ? WHERE id = ?",
            (JobStatus.QUEUED.value, time.time(), utcnow(), job_id),
        )
        return self.get(job_id)

    def recover_expired_leases(self) -> int:
        """Requeue RUNNING jobs whose worker died (lease expired)."""
        cursor = self.db.execute(
            "UPDATE jobs SET status = ?, locked_by = NULL, lease_expires = NULL, updated_at = ? "
            "WHERE status = ? AND lease_expires < ?",
            (JobStatus.QUEUED.value, utcnow(), JobStatus.RUNNING.value, time.time()),
        )
        return cursor.rowcount

    def list(
        self,
        *,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[JobRecord], int]:
        if status is not None and status not in {s.value for s in JobStatus}:
            raise ValidationError(f"unknown job status '{status}'")
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if job_type:
            clauses.append("type = ?")
            params.append(job_type)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        total = self.db.query_one(f"SELECT COUNT(*) AS n FROM jobs{where}", tuple(params))
        rows = self.db.query(
            f"SELECT * FROM jobs{where} ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        return [JobRecord.from_row(r) for r in rows], int(total["n"]) if total else 0

    def counts(self) -> dict[str, int]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        counts = {s.value: 0 for s in JobStatus}
        counts.update({r["status"]: r["n"] for r in rows})
        return counts

    def purge_finished(self, older_than_seconds: float) -> int:
        cutoff = (
            datetime.fromtimestamp(time.time() - older_than_seconds, UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        cursor = self.db.execute(
            "DELETE FROM jobs WHERE status IN (?, ?, ?) AND finished_at < ?",
            (JobStatus.SUCCEEDED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value, cutoff),
        )
        return cursor.rowcount
