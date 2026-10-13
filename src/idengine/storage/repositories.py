"""Repositories for documents, chunks and summaries."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from idengine.errors import NotFoundError, ValidationError
from idengine.ingest.hashing import hamming_distance, simhash_to_signed
from idengine.storage.db import Database, new_id, utcnow
from idengine.storage.models import ChunkRecord, DocumentRecord, DocumentStatus

_SORTABLE = {"created_at", "updated_at", "title", "size_bytes", "document_date"}
_UPDATABLE = {
    "status",
    "title",
    "authors",
    "document_date",
    "language",
    "category",
    "category_confidence",
    "metadata",
    "tags",
    "near_duplicate_of",
    "chunk_count",
    "error",
    "text_hash",
    "simhash",
}
_JSON_FIELDS = {"authors", "metadata", "tags"}


def _bands(simhash: int) -> tuple[int, int, int, int]:
    """Split a 64-bit SimHash into four 16-bit bands for LSH candidate lookup.

    Two fingerprints within Hamming distance 3 must agree on at least one band
    (pigeonhole), so querying the four band columns finds all near duplicates.
    """
    return tuple((simhash >> (16 * i)) & 0xFFFF for i in range(4))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DocumentQuery:
    status: str | None = None
    category: str | None = None
    language: str | None = None
    doc_type: str | None = None
    tag: str | None = None
    title_contains: str | None = None
    sort: str = "created_at"
    descending: bool = True
    limit: int = 50
    offset: int = 0


class DocumentRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        *,
        filename: str,
        doc_type: str,
        mime_type: str,
        size_bytes: int,
        content_hash: str,
        tags: Sequence[str] = (),
    ) -> DocumentRecord:
        doc_id = new_id("doc")
        now = utcnow()
        self.db.execute(
            "INSERT INTO documents (id, filename, doc_type, mime_type, size_bytes, content_hash,"
            " status, tags, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                doc_id,
                filename,
                doc_type,
                mime_type,
                size_bytes,
                content_hash,
                DocumentStatus.PROCESSING.value,
                json.dumps(list(tags)),
                now,
                now,
            ),
        )
        return self.get(doc_id)

    def get(self, doc_id: str) -> DocumentRecord:
        row = self.db.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,))
        if row is None:
            raise NotFoundError(f"document {doc_id} not found", details={"id": doc_id})
        return DocumentRecord.from_row(row)

    def find(self, doc_id: str) -> DocumentRecord | None:
        row = self.db.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return DocumentRecord.from_row(row) if row else None

    def by_content_hash(self, content_hash: str) -> DocumentRecord | None:
        row = self.db.query_one("SELECT * FROM documents WHERE content_hash = ?", (content_hash,))
        return DocumentRecord.from_row(row) if row else None

    def by_text_hash(self, text_hash: str, *, exclude: str | None = None) -> DocumentRecord | None:
        row = self.db.query_one(
            "SELECT * FROM documents WHERE text_hash = ? AND id != ? AND status = 'ready' "
            "ORDER BY created_at LIMIT 1",
            (text_hash, exclude or ""),
        )
        return DocumentRecord.from_row(row) if row else None

    def near_duplicates(
        self, simhash: int, *, max_distance: int, exclude: str | None = None
    ) -> list[tuple[str, int]]:
        """Documents whose SimHash lies within ``max_distance`` bits, closest first."""
        bands = _bands(simhash)
        if max_distance > 3:
            rows = self.db.query(
                "SELECT id, simhash FROM documents WHERE simhash IS NOT NULL AND id != ?",
                (exclude or "",),
            )
        else:
            rows = self.db.query(
                "SELECT id, simhash FROM documents WHERE id != ? AND status = 'ready' AND "
                "(sim_band0 = ? OR sim_band1 = ? OR sim_band2 = ? OR sim_band3 = ?)",
                (exclude or "", *bands),
            )
        matches: list[tuple[str, int]] = []
        for row in rows:
            other = row["simhash"] % (1 << 64)
            distance = hamming_distance(simhash, other)
            if distance <= max_distance:
                matches.append((row["id"], distance))
        return sorted(matches, key=lambda item: item[1])

    def update(self, doc_id: str, **fields: Any) -> DocumentRecord:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValidationError(f"cannot update fields: {sorted(unknown)}")
        assignments: list[str] = []
        params: list[Any] = []
        for name, value in fields.items():
            if name in _JSON_FIELDS:
                value = json.dumps(value)
            elif name == "status" and isinstance(value, DocumentStatus):
                value = value.value
            if name == "simhash" and value is not None:
                for i, band in enumerate(_bands(value)):
                    assignments.append(f"sim_band{i} = ?")
                    params.append(band)
                value = simhash_to_signed(value)
            assignments.append(f"{name} = ?")
            params.append(value)
        assignments.append("updated_at = ?")
        params.append(utcnow())
        params.append(doc_id)
        cursor = self.db.execute(
            f"UPDATE documents SET {', '.join(assignments)} WHERE id = ?", tuple(params)
        )
        if cursor.rowcount == 0:
            raise NotFoundError(f"document {doc_id} not found", details={"id": doc_id})
        return self.get(doc_id)

    def delete(self, doc_id: str) -> None:
        cursor = self.db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        if cursor.rowcount == 0:
            raise NotFoundError(f"document {doc_id} not found", details={"id": doc_id})

    def _where(self, query: DocumentQuery) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column in ("status", "category", "language", "doc_type"):
            value = getattr(query, column)
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if query.tag:
            clauses.append("EXISTS (SELECT 1 FROM json_each(documents.tags) WHERE value = ?)")
            params.append(query.tag)
        if query.title_contains:
            clauses.append("LOWER(COALESCE(title, filename)) LIKE ? ESCAPE '\\'")
            escaped = query.title_contains.lower().replace("\\", "\\\\")
            escaped = escaped.replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def list(self, query: DocumentQuery) -> tuple[list[DocumentRecord], int]:
        if query.sort not in _SORTABLE:
            raise ValidationError(
                f"cannot sort by '{query.sort}'", details={"allowed": sorted(_SORTABLE)}
            )
        where, params = self._where(query)
        total_row = self.db.query_one(f"SELECT COUNT(*) AS n FROM documents{where}", tuple(params))
        order = "DESC" if query.descending else "ASC"
        rows = self.db.query(
            f"SELECT * FROM documents{where} ORDER BY {query.sort} {order}, id LIMIT ? OFFSET ?",
            (*params, query.limit, query.offset),
        )
        total = int(total_row["n"]) if total_row else 0
        return [DocumentRecord.from_row(r) for r in rows], total

    def all_ready(self) -> list[DocumentRecord]:
        rows = self.db.query("SELECT * FROM documents WHERE status = 'ready' ORDER BY created_at")
        return [DocumentRecord.from_row(r) for r in rows]

    def count(self, status: DocumentStatus | None = None) -> int:
        if status is None:
            row = self.db.query_one("SELECT COUNT(*) AS n FROM documents")
        else:
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM documents WHERE status = ?", (status.value,)
            )
        return int(row["n"]) if row else 0

    def category_counts(self) -> dict[str, int]:
        rows = self.db.query(
            "SELECT COALESCE(category, 'uncategorized') AS c, COUNT(*) AS n FROM documents "
            "WHERE status = 'ready' GROUP BY c ORDER BY c"
        )
        return {r["c"]: r["n"] for r in rows}


class ChunkRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def replace_for_document(
        self, document_id: str, chunks: Sequence[dict[str, Any]]
    ) -> list[ChunkRecord]:
        """Atomically replace all chunks of a document."""
        records: list[ChunkRecord] = []
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            for chunk in chunks:
                record = ChunkRecord(
                    id=new_id("chk"),
                    document_id=document_id,
                    ordinal=chunk["ordinal"],
                    text=chunk["text"],
                    token_count=chunk["token_count"],
                    heading_path=list(chunk.get("heading_path", [])),
                    page_start=chunk.get("page_start"),
                    page_end=chunk.get("page_end"),
                )
                conn.execute(
                    "INSERT INTO chunks (id, document_id, ordinal, text, token_count, heading_path,"
                    " page_start, page_end) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        record.id,
                        record.document_id,
                        record.ordinal,
                        record.text,
                        record.token_count,
                        json.dumps(record.heading_path),
                        record.page_start,
                        record.page_end,
                    ),
                )
                records.append(record)
        return records

    def for_document(self, document_id: str) -> list[ChunkRecord]:
        rows = self.db.query(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY ordinal", (document_id,)
        )
        return [ChunkRecord.from_row(r) for r in rows]

    def get_many(self, chunk_ids: Sequence[str]) -> dict[str, ChunkRecord]:
        found: dict[str, ChunkRecord] = {}
        ids = list(dict.fromkeys(chunk_ids))
        for start in range(0, len(ids), 500):  # stay below SQLite's variable limit
            batch = ids[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self.db.query(f"SELECT * FROM chunks WHERE id IN ({placeholders})", tuple(batch))
            for row in rows:
                found[row["id"]] = ChunkRecord.from_row(row)
        return found

    def iter_ready(self, batch_size: int = 500) -> Iterator[tuple[ChunkRecord, str]]:
        """All chunks of ready documents with their document id, in stable order."""
        last_rowid = 0
        while True:
            rows = self.db.query(
                "SELECT c.rowid AS rid, c.* FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE d.status = 'ready' AND c.rowid > ? ORDER BY c.rowid LIMIT ?",
                (last_rowid, batch_size),
            )
            if not rows:
                return
            for row in rows:
                yield ChunkRecord.from_row(row), row["document_id"]
            last_rowid = rows[-1]["rid"]

    def count(self) -> int:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.status = 'ready'"
        )
        return int(row["n"]) if row else 0


class SummaryRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, document_id: str, method: str) -> str | None:
        row = self.db.query_one(
            "SELECT summary FROM summaries WHERE document_id = ? AND method = ?",
            (document_id, method),
        )
        return row["summary"] if row else None

    def put(self, document_id: str, method: str, summary: str) -> None:
        self.db.execute(
            "INSERT INTO summaries(document_id, method, summary, created_at) VALUES (?,?,?,?) "
            "ON CONFLICT(document_id, method) DO UPDATE SET summary = excluded.summary, "
            "created_at = excluded.created_at",
            (document_id, method, summary, utcnow()),
        )

    def invalidate(self, document_id: str) -> None:
        self.db.execute("DELETE FROM summaries WHERE document_id = ?", (document_id,))
