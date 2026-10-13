"""Persistent record types."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from idengine.ingest.hashing import simhash_from_signed


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


@dataclass(slots=True)
class DocumentRecord:
    id: str
    filename: str
    doc_type: str
    mime_type: str
    size_bytes: int
    content_hash: str
    text_hash: str | None
    simhash: int | None
    status: DocumentStatus
    title: str | None
    authors: list[str]
    document_date: str | None
    language: str | None
    category: str | None
    category_confidence: float | None
    metadata: dict[str, Any]
    tags: list[str]
    near_duplicate_of: str | None
    chunk_count: int
    error: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> DocumentRecord:
        return cls(
            id=row["id"],
            filename=row["filename"],
            doc_type=row["doc_type"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            content_hash=row["content_hash"],
            text_hash=row["text_hash"],
            simhash=simhash_from_signed(row["simhash"]) if row["simhash"] is not None else None,
            status=DocumentStatus(row["status"]),
            title=row["title"],
            authors=_loads(row["authors"], []),
            document_date=row["document_date"],
            language=row["language"],
            category=row["category"],
            category_confidence=row["category_confidence"],
            metadata=_loads(row["metadata"], {}),
            tags=_loads(row["tags"], []),
            near_duplicate_of=row["near_duplicate_of"],
            chunk_count=row["chunk_count"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def filter_metadata(self) -> dict[str, Any]:
        """Fields copied onto each chunk in the indexes for metadata filtering."""
        return {
            "document_id": self.id,
            "doc_type": self.doc_type,
            "language": self.language,
            "category": self.category,
            "document_date": self.document_date,
            "tags": list(self.tags),
            "title": self.title,
            "filename": self.filename,
        }


@dataclass(slots=True)
class ChunkRecord:
    id: str
    document_id: str
    ordinal: int
    text: str
    token_count: int
    heading_path: list[str]
    page_start: int | None
    page_end: int | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ChunkRecord:
        return cls(
            id=row["id"],
            document_id=row["document_id"],
            ordinal=row["ordinal"],
            text=row["text"],
            token_count=row["token_count"],
            heading_path=_loads(row["heading_path"], []),
            page_start=row["page_start"],
            page_end=row["page_end"],
        )

    @property
    def section(self) -> str | None:
        return self.heading_path[-1] if self.heading_path else None

    def index_text(self) -> str:
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n" + self.text


@dataclass(slots=True)
class JobRecord:
    id: str
    type: str
    payload: dict[str, Any]
    status: JobStatus
    priority: int
    attempts: int
    max_attempts: int
    run_after: float
    locked_by: str | None
    lease_expires: float | None
    result: Any
    error: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> JobRecord:
        return cls(
            id=row["id"],
            type=row["type"],
            payload=_loads(row["payload"], {}),
            status=JobStatus(row["status"]),
            priority=row["priority"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            run_after=row["run_after"],
            locked_by=row["locked_by"],
            lease_expires=row["lease_expires"],
            result=_loads(row["result"], None),
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )
