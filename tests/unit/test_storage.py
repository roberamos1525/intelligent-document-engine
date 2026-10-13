from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from idengine.errors import ConflictError, NotFoundError, StorageError, ValidationError
from idengine.ingest.hashing import (
    content_hash,
    hamming_distance,
    simhash,
    simhash_from_signed,
    simhash_to_signed,
    text_hash,
)
from idengine.storage import (
    BlobStore,
    ChunkRepository,
    Database,
    DocumentQuery,
    DocumentRepository,
    DocumentStatus,
    JobRepository,
    JobStatus,
    SummaryRepository,
)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    yield database
    database.close()


def _doc(repo: DocumentRepository, name: str, **fields: object):  # type: ignore[no-untyped-def]
    record = repo.create(
        filename=name,
        doc_type="txt",
        mime_type="text/plain",
        size_bytes=10,
        content_hash=content_hash(name.encode()),
        tags=fields.pop("tags", []),  # type: ignore[arg-type]
    )
    return repo.update(record.id, **fields) if fields else record


class TestHashing:
    def test_simhash_detects_near_duplicates(self) -> None:
        base = " ".join(f"word{i}" for i in range(200))
        edited = base.replace("word50", "changed")
        other = " ".join(f"token{i}" for i in range(200))
        assert hamming_distance(simhash(base), simhash(edited)) <= 8
        assert hamming_distance(simhash(base), simhash(other)) > 16

    def test_signed_round_trip_and_text_hash(self) -> None:
        value = (1 << 64) - 5
        assert simhash_from_signed(simhash_to_signed(value)) == value
        assert simhash_to_signed(5) == 5
        assert text_hash("Hello,  World") == text_hash("hello world\n")
        assert len(content_hash(b"x")) == 64


class TestDatabase:
    def test_migrations_are_idempotent(self, db: Database) -> None:
        assert db.migrate() == 2
        assert db.migrate() == 2
        assert db.ping() >= 0

    def test_transaction_rolls_back(self, db: Database) -> None:
        repo = DocumentRepository(db)

        def failing_transaction() -> None:
            with db.transaction():
                _doc(repo, "a.txt")
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            failing_transaction()
        assert repo.count() == 0

    def test_closed_database_raises(self, tmp_path: Path) -> None:
        database = Database(tmp_path / "x.sqlite3")
        database.close()
        with pytest.raises(StorageError):
            database.query("SELECT 1")


class TestDocuments:
    def test_crud(self, db: Database) -> None:
        repo = DocumentRepository(db)
        record = _doc(repo, "a.txt", tags=["x"])
        assert record.status is DocumentStatus.PROCESSING
        updated = repo.update(
            record.id,
            status=DocumentStatus.READY,
            title="A",
            authors=["Ann"],
            metadata={"k": 1},
            simhash=(1 << 63) + 7,
        )
        assert (updated.title, updated.authors, updated.metadata) == ("A", ["Ann"], {"k": 1})
        assert updated.simhash == (1 << 63) + 7
        assert repo.by_content_hash(record.content_hash) == updated
        repo.delete(record.id)
        assert repo.find(record.id) is None
        with pytest.raises(NotFoundError):
            repo.get(record.id)
        with pytest.raises(NotFoundError):
            repo.delete(record.id)

    def test_update_rejects_unknown_fields(self, db: Database) -> None:
        repo = DocumentRepository(db)
        record = _doc(repo, "a.txt")
        with pytest.raises(ValidationError):
            repo.update(record.id, filename="hack")

    def test_listing_filters_sorting_and_pagination(self, db: Database) -> None:
        repo = DocumentRepository(db)
        _doc(repo, "a.txt", status=DocumentStatus.READY, category="finance", title="Alpha 100%")
        _doc(repo, "b.txt", status=DocumentStatus.READY, category="legal", title="Beta", tags=["x"])
        _doc(repo, "c.txt", status=DocumentStatus.FAILED, category="finance", title="Gamma")
        items, total = repo.list(DocumentQuery(category="finance"))
        assert total == 2 and {i.title for i in items} == {"Alpha 100%", "Gamma"}
        items, _ = repo.list(DocumentQuery(tag="x"))
        assert [i.title for i in items] == ["Beta"]
        items, _ = repo.list(DocumentQuery(title_contains="100%"))
        assert [i.title for i in items] == ["Alpha 100%"]
        items, total = repo.list(DocumentQuery(sort="title", descending=False, limit=2, offset=1))
        assert total == 3 and [i.title for i in items] == ["Beta", "Gamma"]
        with pytest.raises(ValidationError):
            repo.list(DocumentQuery(sort="id; DROP TABLE documents"))
        assert repo.count(DocumentStatus.READY) == 2
        assert repo.category_counts() == {"finance": 1, "legal": 1}

    def test_near_duplicate_lookup_uses_bands(self, db: Database) -> None:
        repo = DocumentRepository(db)
        base = 0x0123_4567_89AB_CDEF
        a = _doc(repo, "a.txt", status=DocumentStatus.READY, simhash=base)
        _doc(repo, "b.txt", status=DocumentStatus.READY, simhash=base ^ 0xFFFF_FFFF_FFFF_FFFF)
        matches = repo.near_duplicates(base ^ 0b101, max_distance=3)
        assert matches == [(a.id, 2)]
        assert repo.near_duplicates(base, max_distance=3, exclude=a.id) == []
        assert repo.near_duplicates(base ^ 0xFF, max_distance=10)[0] == (a.id, 8)

    def test_text_hash_lookup(self, db: Database) -> None:
        repo = DocumentRepository(db)
        a = _doc(repo, "a.txt", status=DocumentStatus.READY, text_hash="h1")
        assert repo.by_text_hash("h1") == a
        assert repo.by_text_hash("h1", exclude=a.id) is None


class TestChunksAndSummaries:
    def test_replace_and_iterate(self, db: Database) -> None:
        docs = DocumentRepository(db)
        chunks = ChunkRepository(db)
        doc = _doc(docs, "a.txt", status=DocumentStatus.READY)
        payload = [
            {"ordinal": i, "text": f"chunk {i}", "token_count": 2, "heading_path": ["H"]}
            for i in range(3)
        ]
        first = chunks.replace_for_document(doc.id, payload)
        second = chunks.replace_for_document(doc.id, payload[:2])
        assert len(chunks.for_document(doc.id)) == 2
        assert not {c.id for c in first} & set(chunks.get_many([c.id for c in first]))
        assert set(chunks.get_many([c.id for c in second])) == {c.id for c in second}
        assert [c.text for c, _ in chunks.iter_ready(batch_size=1)] == ["chunk 0", "chunk 1"]
        assert second[0].index_text() == "H\nchunk 0"
        assert chunks.count() == 2
        docs.delete(doc.id)  # cascades
        assert chunks.count() == 0

    def test_summaries_cache(self, db: Database) -> None:
        docs = DocumentRepository(db)
        doc = _doc(docs, "a.txt")
        summaries = SummaryRepository(db)
        assert summaries.get(doc.id, "textrank") is None
        summaries.put(doc.id, "textrank", "one")
        summaries.put(doc.id, "textrank", "two")
        assert summaries.get(doc.id, "textrank") == "two"
        summaries.invalidate(doc.id)
        assert summaries.get(doc.id, "textrank") is None


class TestBlobs:
    def test_round_trip_and_key_validation(self, tmp_path: Path) -> None:
        store = BlobStore(tmp_path / "blobs")
        digest = content_hash(b"payload")
        path = store.put(digest, b"payload")
        assert path.parent.parent.name == digest[:2]
        assert store.put(digest, b"payload") == path  # idempotent
        assert store.get(digest) == b"payload"
        assert store.exists(digest) and store.total_bytes() == 7
        assert store.delete(digest) and not store.delete(digest)
        with pytest.raises(NotFoundError):
            store.get(digest)
        with pytest.raises(StorageError):
            store.get("../../etc/passwd")


class TestJobs:
    def test_claim_complete_and_priority(self, db: Database) -> None:
        jobs = JobRepository(db)
        low = jobs.enqueue("t", {"n": 1}, max_attempts=3)
        high = jobs.enqueue("t", {"n": 2}, max_attempts=3, priority=5)
        delayed = jobs.enqueue("t", {}, max_attempts=3, delay_seconds=60)
        claimed = jobs.claim("w1", 30)
        assert claimed is not None and claimed.id == high.id
        assert claimed.status is JobStatus.RUNNING and claimed.attempts == 1
        jobs.complete(claimed.id, {"ok": True})
        assert jobs.get(high.id).result == {"ok": True}
        assert jobs.claim("w1", 30).id == low.id  # type: ignore[union-attr]
        assert jobs.claim("w1", 30) is None  # delayed job not yet runnable
        assert jobs.get(delayed.id).status is JobStatus.QUEUED

    def test_fail_retry_cancel(self, db: Database) -> None:
        jobs = JobRepository(db)
        job = jobs.enqueue("t", {}, max_attempts=2)
        jobs.claim("w", 30)
        assert jobs.fail(job.id, "transient", retry_in=0) is JobStatus.QUEUED
        jobs.claim("w", 30)
        assert jobs.fail(job.id, "fatal", retry_in=None) is JobStatus.FAILED
        with pytest.raises(ConflictError):
            jobs.cancel(job.id)
        retried = jobs.retry(job.id)
        assert retried.status is JobStatus.QUEUED and retried.attempts == 0
        cancelled = jobs.cancel(job.id)
        assert cancelled.status is JobStatus.CANCELLED and cancelled.status.terminal
        with pytest.raises(ConflictError):
            jobs.retry(jobs.enqueue("t", {}, max_attempts=1).id)

    def test_expired_leases_are_recovered(self, db: Database) -> None:
        jobs = JobRepository(db)
        job = jobs.enqueue("t", {}, max_attempts=3)
        jobs.claim("dead-worker", lease_seconds=0.01)
        time.sleep(0.03)
        assert jobs.recover_expired_leases() == 1
        assert jobs.get(job.id).status is JobStatus.QUEUED

    def test_listing_counts_and_purge(self, db: Database) -> None:
        jobs = JobRepository(db)
        a = jobs.enqueue("ingest", {}, max_attempts=1)
        jobs.enqueue("summarize", {}, max_attempts=1)
        jobs.claim("w", 30)
        jobs.complete(a.id, None)
        items, total = jobs.list(job_type="ingest")
        assert total == 1 and items[0].id == a.id
        assert jobs.counts()["succeeded"] == 1 and jobs.counts()["queued"] == 1
        with pytest.raises(ValidationError):
            jobs.list(status="bogus")
        assert jobs.purge_finished(older_than_seconds=3600) == 0
        assert jobs.purge_finished(older_than_seconds=-1) == 1
