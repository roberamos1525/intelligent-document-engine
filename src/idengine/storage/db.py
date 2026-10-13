"""SQLite access: per-thread connections, WAL mode and schema migrations."""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from idengine.errors import StorageError
from idengine.observability.logging import get_logger

log = get_logger("storage.db")

MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            text_hash TEXT,
            simhash INTEGER,
            sim_band0 INTEGER, sim_band1 INTEGER, sim_band2 INTEGER, sim_band3 INTEGER,
            status TEXT NOT NULL,
            title TEXT,
            authors TEXT NOT NULL DEFAULT '[]',
            document_date TEXT,
            language TEXT,
            category TEXT,
            category_confidence REAL,
            metadata TEXT NOT NULL DEFAULT '{}',
            tags TEXT NOT NULL DEFAULT '[]',
            near_duplicate_of TEXT,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_documents_text_hash ON documents(text_hash);
        CREATE INDEX idx_documents_status ON documents(status);
        CREATE INDEX idx_documents_category ON documents(category);
        CREATE INDEX idx_documents_band0 ON documents(sim_band0);
        CREATE INDEX idx_documents_band1 ON documents(sim_band1);
        CREATE INDEX idx_documents_band2 ON documents(sim_band2);
        CREATE INDEX idx_documents_band3 ON documents(sim_band3);

        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            heading_path TEXT NOT NULL DEFAULT '[]',
            page_start INTEGER,
            page_end INTEGER,
            UNIQUE(document_id, ordinal)
        );
        CREATE INDEX idx_chunks_document ON chunks(document_id);

        CREATE TABLE jobs (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            payload TEXT NOT NULL,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL,
            run_after REAL NOT NULL,
            locked_by TEXT,
            lease_expires REAL,
            result TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        );
        CREATE INDEX idx_jobs_ready ON jobs(status, run_after);
        """,
    ),
    (
        2,
        """
        CREATE TABLE summaries (
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            method TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (document_id, method)
        );
        """,
    ),
)


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class Database:
    """Thread-safe SQLite wrapper.

    Each thread gets its own connection (SQLite connections must not be shared
    across threads while in use). WAL mode lets API readers proceed while a
    worker thread writes.
    """

    def __init__(self, path: Path, *, busy_timeout_ms: int = 10_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._busy_timeout_ms = busy_timeout_ms
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._closed = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,  # explicit transactions only
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        if self._closed:
            raise StorageError("database is closed")
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
            with self._lock:
                self._connections.append(conn)
        return conn

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        """Run a block in a transaction; nested calls join the outer transaction."""
        conn = self.conn
        if conn.in_transaction:
            yield conn
            return
        conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def execute(
        self, sql: str, params: tuple[object, ...] | dict[str, object] = ()
    ) -> sqlite3.Cursor:
        try:
            return self.conn.execute(sql, params)
        except sqlite3.Error as exc:
            raise StorageError(f"database error: {exc}") from exc

    def query(
        self, sql: str, params: tuple[object, ...] | dict[str, object] = ()
    ) -> list[sqlite3.Row]:
        return self.execute(sql, params).fetchall()

    def query_one(
        self, sql: str, params: tuple[object, ...] | dict[str, object] = ()
    ) -> sqlite3.Row | None:
        return self.execute(sql, params).fetchone()

    def migrate(self) -> int:
        """Apply pending migrations; returns the resulting schema version."""
        with self.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
            for version, script in MIGRATIONS:
                if version in applied:
                    continue
                for statement in (s.strip() for s in script.split(";")):
                    if statement:
                        conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, utcnow()),
                )
                log.info("applied migration", extra={"version": version})
        return self.schema_version()

    def schema_version(self) -> int:
        row = self.query_one("SELECT MAX(version) AS v FROM schema_migrations")
        return int(row["v"]) if row and row["v"] is not None else 0

    def ping(self) -> float:
        """Round-trip a trivial query; returns latency in seconds."""
        start = time.perf_counter()
        self.query_one("SELECT 1")
        return time.perf_counter() - start

    def close(self) -> None:
        with self._lock:
            for conn in self._connections:
                with contextlib.suppress(sqlite3.Error):  # best effort on shutdown
                    conn.close()
            self._connections.clear()
            self._closed = True
