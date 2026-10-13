"""Ingestion edge cases: reprocessing, failure recovery, concurrency and cache invalidation."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from builders import INVOICE_TEXT, POLICY_MARKDOWN, build_pdf
from conftest import make_settings
from idengine.engine import Engine
from idengine.errors import ExtractionError, NotFoundError, ValidationError
from idengine.ingest.hashing import content_hash
from idengine.storage import DocumentStatus, JobStatus


def test_reprocessing_replaces_chunks_and_index_entries(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with Engine(make_settings(data_dir, chunking={"max_tokens": 400})) as engine:
        outcome, _ = engine.ingest_bytes(POLICY_MARKDOWN.encode(), "policy.md")
        doc_id = outcome.document.id
        old_chunks = {c.id for c in engine.chunks.for_document(doc_id)}
        summary = engine.summarize_document(doc_id, max_sentences=1)
        assert summary.provider == "textrank"
    # Smaller chunks after a configuration change: reprocess the same document.
    with Engine(
        make_settings(data_dir, chunking={"max_tokens": 20, "overlap_tokens": 0})
    ) as engine:
        job = engine.jobs.submit("ingest", {"document_id": doc_id})
        engine.jobs.drain(timeout=10)
        assert engine.jobs.repository.get(job.id).status is JobStatus.SUCCEEDED
        new_chunks = {c.id for c in engine.chunks.for_document(doc_id)}
        assert len(new_chunks) > len(old_chunks) and not new_chunks & old_chunks
        assert len(engine.indexes.bm25) == len(new_chunks)
        assert len(engine.indexes.vectors) == len(new_chunks)
        # The cached summary was invalidated by reprocessing.
        assert engine.summarize_document(doc_id, max_sentences=1).provider == "textrank"


def test_failed_document_is_retried_on_reupload(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = engine.ingestion._process
    calls = {"n": 0}

    def flaky(record):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise ExtractionError("simulated parser crash")
        return original(record)

    monkeypatch.setattr(engine.ingestion, "_process", flaky)
    with pytest.raises(ExtractionError):
        engine.ingest_bytes(INVOICE_TEXT.encode(), "invoice.txt")
    failed = engine.documents.by_content_hash(content_hash(INVOICE_TEXT.encode()))
    assert failed is not None and failed.status is DocumentStatus.FAILED
    outcome, _ = engine.ingest_bytes(INVOICE_TEXT.encode(), "invoice.txt")
    assert outcome.duplicate and outcome.document.status is DocumentStatus.READY
    assert outcome.document.id == failed.id and outcome.document.error is None


def test_unexpected_exceptions_are_wrapped(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(record):  # type: ignore[no-untyped-def]
        raise KeyError("bug")

    monkeypatch.setattr(engine.ingestion, "_process", boom)
    with pytest.raises(ExtractionError, match="KeyError"):
        engine.ingest_bytes(b"some text here", "a.txt")
    failed = engine.documents.all_ready()
    assert failed == []
    assert engine.metrics.documents_ingested.value({"type": "txt", "outcome": "failed"}) == 1


def test_background_job_for_unprocessable_document_fails_without_retry(engine: Engine) -> None:
    outcome, job = engine.ingest_bytes(build_pdf([[]]), "scan.pdf", background=True)
    assert job is not None
    engine.jobs.drain(timeout=10, include_delayed=True)
    final = engine.jobs.repository.get(job.id)
    # Extraction errors are retried (they may be transient), then fail permanently.
    assert final.status is JobStatus.FAILED and final.attempts == final.max_attempts
    assert engine.documents.get(outcome.document.id).status is DocumentStatus.FAILED


def test_missing_document_job_is_permanent_failure(engine: Engine) -> None:
    job = engine.jobs.submit("ingest", {"document_id": "doc_" + "0" * 32})
    engine.jobs.drain(timeout=10, include_delayed=True)
    final = engine.jobs.repository.get(job.id)
    assert final.status is JobStatus.FAILED and final.attempts == 1


def test_concurrent_search_during_ingestion(engine: Engine) -> None:
    engine.ingest_bytes(INVOICE_TEXT.encode(), "invoice.txt")
    errors: list[BaseException] = []
    stop = threading.Event()

    def searcher() -> None:
        try:
            while not stop.is_set():
                response = engine.search("invoice payment", top_k=5)
                assert response.hits
                engine.ask("What is the total amount due?")
        except BaseException as exc:  # surfaced to the main thread below
            errors.append(exc)

    threads = [threading.Thread(target=searcher) for _ in range(3)]
    for thread in threads:
        thread.start()
    try:
        for i in range(8):
            text = f"Report {i}\n\nThe invoice backlog for region {i} was reviewed on day {i}."
            engine.ingest_bytes(text.encode(), f"report{i}.txt")
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=30)
    assert not errors
    assert engine.stats()["documents"] == 9


def test_document_update_edge_cases(engine: Engine) -> None:
    outcome, _ = engine.ingest_bytes(INVOICE_TEXT.encode(), "invoice.txt")
    doc_id = outcome.document.id
    assert engine.ingestion.update(doc_id) == engine.documents.get(doc_id)  # no-op
    with pytest.raises(ValidationError):
        engine.ingestion.update(doc_id, title=" ")
    with pytest.raises(ValidationError):
        engine.ingestion.update(doc_id, category="x" * 65)
    with pytest.raises(ValidationError):
        engine.ingestion.update(doc_id, tags=[f"t{i}" for i in range(25)])
    with pytest.raises(NotFoundError):
        engine.ingestion.update("doc_" + "1" * 32, title="x")
    updated = engine.ingestion.update(doc_id, tags=["A", "a", " b "])
    assert updated.tags == ["a", "b"]


def test_summaries_require_ready_documents(engine: Engine) -> None:
    outcome, _ = engine.ingest_bytes(POLICY_MARKDOWN.encode(), "policy.md", background=True)
    with pytest.raises(ValidationError, match="processing"):
        engine.summarize_document(outcome.document.id)
    engine.jobs.drain(timeout=10)
    assert engine.summarize_document(outcome.document.id).text


def test_disallowed_types_are_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data", ingest={"allowed_types": ["pdf"]})
    with Engine(settings) as engine:
        with pytest.raises(ValidationError, match="not enabled"):
            engine.ingest_bytes(INVOICE_TEXT.encode(), "invoice.txt")
        outcome, _ = engine.ingest_bytes(build_pdf([["Only PDFs are accepted here."]]), "a.pdf")
        assert outcome.document.status is DocumentStatus.READY


def test_auto_classification_can_be_disabled(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data", ingest={"auto_classify": False})
    with Engine(settings) as engine:
        engine.classifier.train(
            ["invoice payment due", "late invoice fee", "deploy cluster", "rollback release"],
            ["finance", "finance", "engineering", "engineering"],
            test_ratio=0.0,
        )
        outcome, _ = engine.ingest_bytes(INVOICE_TEXT.encode(), "invoice.txt")
        assert outcome.document.category is None
        prediction = engine.classify_document(outcome.document.id, apply=False)
        assert prediction.label == "finance"
        assert engine.documents.get(outcome.document.id).category is None
