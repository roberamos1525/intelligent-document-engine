from __future__ import annotations

import json
from pathlib import Path

import pytest

from builders import GUIDE_HTML, INVOICE_TEXT, POLICY_MARKDOWN, build_docx, build_pdf
from conftest import GOLDEN_DIR, make_settings
from idengine.engine import Engine, LabeledExample
from idengine.errors import (
    ExtractionError,
    FileTooLargeError,
    ModelNotTrainedError,
    NotFoundError,
    UnsupportedFileTypeError,
    ValidationError,
)
from idengine.storage import DocumentQuery, DocumentStatus, JobStatus


def ingest_samples(engine: Engine) -> dict[str, str]:
    samples = {
        "invoice.txt": INVOICE_TEXT.encode(),
        "policy.md": POLICY_MARKDOWN.encode(),
        "guide.html": GUIDE_HTML.encode(),
        "report.pdf": build_pdf(
            [
                ["QUARTERLY REPORT", "Revenue grew 12 percent in the third quarter."],
                ["Outlook", "Management expects stable demand next year."],
            ],
            title="Q3 Report",
        ),
        "contract.docx": build_docx(
            [
                ("title", "Supplier Contract"),
                ("h1", "Delivery"),
                ("p", "The supplier delivers within 10 business days of each order."),
            ],
            author="Legal Team",
        ),
    }
    ids = {}
    for name, data in samples.items():
        outcome, job = engine.ingest_bytes(data, name, tags=["sample"])
        assert job is None and not outcome.duplicate
        assert outcome.document.status is DocumentStatus.READY, outcome.document.error
        ids[name] = outcome.document.id
    return ids


def test_ingests_every_format_with_metadata(engine: Engine) -> None:
    ids = ingest_samples(engine)
    pdf = engine.documents.get(ids["report.pdf"])
    assert pdf.title == "Q3 Report" and pdf.doc_type == "pdf"
    assert pdf.metadata["page_count"] == 2
    docx = engine.documents.get(ids["contract.docx"])
    assert docx.authors == ["Legal Team"] and docx.chunk_count >= 1
    policy = engine.documents.get(ids["policy.md"])
    assert policy.authors == ["Jane Doe", "John Roe"] and policy.document_date == "2023-11-01"
    invoice = engine.documents.get(ids["invoice.txt"])
    assert invoice.document_date == "2024-03-05"
    assert "USD 12450.00" in invoice.metadata["entities"]["MONEY"]
    assert invoice.language == "en" and invoice.metadata["keywords"]
    chunks = engine.chunks.for_document(ids["report.pdf"])
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2


def test_exact_and_near_duplicates(engine: Engine) -> None:
    first, _ = engine.ingest_bytes(INVOICE_TEXT.encode(), "invoice.txt")
    again, _ = engine.ingest_bytes(INVOICE_TEXT.encode(), "copy.txt")
    assert again.duplicate and again.document.id == first.document.id
    # Same text, different bytes (extra whitespace) -> new document flagged as duplicate text.
    variant, _ = engine.ingest_bytes((INVOICE_TEXT + "\n\n").encode(), "variant.txt")
    assert not variant.duplicate
    assert variant.document.near_duplicate_of == first.document.id
    # A one-word edit in a longer document is caught by SimHash (Hamming distance <= 3).
    report = "\n\n".join(
        f"Section {i}. The regional office {i} reported {i * 7} new customers and {i + 3} "
        f"support tickets during review cycle {i}."
        for i in range(40)
    )
    original, _ = engine.ingest_bytes(report.encode(), "report.txt")
    edited = report.replace("regional office 17", "regional branch 17")
    near, _ = engine.ingest_bytes(edited.encode(), "edited.txt")
    assert near.document.near_duplicate_of == original.document.id


def test_search_modes_filters_and_grouping(engine: Engine) -> None:
    ids = ingest_samples(engine)
    for mode in ("lexical", "semantic", "hybrid"):
        response = engine.search("invoice payment late fee", mode=mode, top_k=3)
        assert response.hits[0].document_id == ids["invoice.txt"], mode
    hybrid = engine.search("autoscaler replicas", top_k=5)
    assert hybrid.hits[0].ranks.keys() <= {"vector", "bm25"}
    filtered = engine.search("policy", filters={"doc_type": "markdown"}, top_k=10)
    assert {h.document_id for h in filtered.hits} == {ids["policy.md"]}
    grouped = engine.search("remote work policy", group_by_document=True, top_k=10)
    assert len({h.document_id for h in grouped.hits}) == len(grouped.hits)
    with pytest.raises(ValidationError):
        engine.search("   ")
    with pytest.raises(ValidationError):
        engine.search("x", top_k=1000)
    with pytest.raises(ValidationError):
        engine.search("x", filters={"$bogus": 1})


def test_ask_with_citations_and_abstention(engine: Engine) -> None:
    ids = ingest_samples(engine)
    answer = engine.ask("What is the total amount due on the invoice?")
    assert answer.answerable and answer.short_answer == "$12,450.00"
    assert answer.citations[0].document_id == ids["invoice.txt"]
    assert answer.retrieval["context_passages"] >= 1
    unknown = engine.ask("What is the boiling point of mercury?")
    assert not unknown.answerable and unknown.citations == []
    scoped = engine.ask("What is the total amount due?", filters={"doc_type": "pdf"})
    assert all(c.document_id == ids["report.pdf"] for c in scoped.citations)
    with pytest.raises(ValidationError, match="top_k"):
        engine.ask("What is the total amount due?", top_k=0)


def test_summaries_are_cached_and_support_map_reduce(engine: Engine) -> None:
    ids = ingest_samples(engine)
    doc_id = ids["policy.md"]
    first = engine.summarize_document(doc_id, max_sentences=2)
    assert first.provider == "textrank" and first.text
    assert "Remote Work Policy" not in first.text  # headings are excluded
    cached = engine.summarize_document(doc_id, max_sentences=2)
    assert cached.provider == "cache" and cached.text == first.text
    mapreduce = engine.summarize_document(doc_id, method="mapreduce", max_sentences=2)
    assert mapreduce.provider == "extractive" and mapreduce.stages
    text_summary = engine.summarize_text(INVOICE_TEXT, max_sentences=1)
    assert len(text_summary.sentences) == 1
    with pytest.raises(ValidationError):
        engine.summarize_text("  ")


def test_classifier_training_and_auto_classification(engine: Engine) -> None:
    dataset = json.loads((GOLDEN_DIR / "dataset.json").read_text(encoding="utf-8"))
    examples = [
        LabeledExample(label=e["label"], text=e["text"]) for e in dataset["classification"]["train"]
    ]
    with pytest.raises(ModelNotTrainedError):
        engine.classify_text("anything")
    result = engine.train_classifier(examples, test_ratio=0.0)
    assert result.labels == ["engineering", "finance", "hr", "legal"]
    outcome, _ = engine.ingest_bytes(INVOICE_TEXT.encode(), "invoice.txt")
    assert outcome.document.category == "finance"
    assert outcome.document.metadata["category_source"] == "model"
    assert (
        engine.classify_text("Deploy containers and roll back failed releases.").label
        == "engineering"
    )
    # Training from stored documents works too.
    doc_example = LabeledExample(label="finance", document_id=outcome.document.id)
    engine.train_classifier([*examples, doc_example], test_ratio=0.0)
    with pytest.raises(ValidationError):
        engine.train_classifier([LabeledExample(label="x")])


def test_manual_category_is_preserved(engine: Engine) -> None:
    outcome, _ = engine.ingest_bytes(INVOICE_TEXT.encode(), "invoice.txt", category="contracts")
    record = outcome.document
    assert record.category == "contracts" and record.metadata["category_source"] == "manual"
    hits = engine.search("invoice", filters={"category": "contracts"}).hits
    assert hits and all(h.metadata["category"] == "contracts" for h in hits)


def test_update_and_delete_keep_indexes_in_sync(engine: Engine) -> None:
    ids = ingest_samples(engine)
    doc_id = ids["guide.html"]
    engine.ingestion.update(doc_id, tags=["ops", "k8s"], title="Ops Guide", category="engineering")
    hits = engine.search("helm rollback", filters={"tags": "k8s"}).hits
    assert hits and hits[0].title == "Ops Guide"
    with pytest.raises(ValidationError):
        engine.ingestion.update(doc_id, tags=["bad tag!"])
    engine.ingestion.delete(doc_id)
    assert all(h.document_id != doc_id for h in engine.search("helm rollback").hits)
    with pytest.raises(NotFoundError):
        engine.documents.get(doc_id)


def test_validation_errors_on_ingest(engine: Engine, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        engine.ingest_bytes(b"MZ\x00\x01", "virus.exe")
    with pytest.raises(ValidationError):
        engine.ingest_bytes(b"text", "a.txt", tags=["x" * 50])
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * (engine.settings.ingest.max_file_bytes + 1))
    with pytest.raises(FileTooLargeError):
        engine.ingest_path(big)
    with pytest.raises(ValidationError):
        engine.ingest_path(tmp_path / "missing.txt")


def test_extraction_failure_marks_document_failed(engine: Engine) -> None:
    with pytest.raises(ExtractionError):
        engine.ingest_bytes(build_pdf([[]]), "empty.pdf")
    failed, _ = engine.documents.list(DocumentQuery(status="failed"))
    assert len(failed) == 1 and "no extractable text" in (failed[0].error or "")


def test_background_ingestion_via_job_queue(engine: Engine) -> None:
    outcome, job = engine.ingest_bytes(POLICY_MARKDOWN.encode(), "policy.md", background=True)
    assert job is not None and outcome.document.status is DocumentStatus.PROCESSING
    assert engine.jobs.drain(timeout=10) == 1
    assert engine.jobs.repository.get(job.id).status is JobStatus.SUCCEEDED
    assert engine.documents.get(outcome.document.id).status is DocumentStatus.READY
    summary_job = engine.jobs.submit(
        "summarize", {"document_id": outcome.document.id, "max_sentences": 1}
    )
    engine.jobs.drain(timeout=10)
    assert engine.jobs.repository.get(summary_job.id).result["summary"]


def test_indexes_persist_across_restarts_and_recover_from_corruption(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data", index={"kind": "hnsw"})
    with Engine(settings) as first:
        ingest_samples(first)
        expected = [h.chunk_id for h in first.search("remote work equipment", top_k=3).hits]
    with Engine(settings) as second:
        assert [
            h.chunk_id for h in second.search("remote work equipment", top_k=3).hits
        ] == expected
        assert second.indexes.last_rebuild is None  # loaded from disk, no rebuild needed
    (settings.storage.index_dir / "vectors.json").write_text("{corrupt", encoding="utf-8")
    (settings.storage.index_dir / "bm25.json").unlink()
    with Engine(settings) as third:
        assert third.indexes.last_rebuild is not None
        assert [h.chunk_id for h in third.search("remote work equipment", top_k=3).hits] == expected


def test_lsa_embedder_requires_reindex(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "data", embeddings={"provider": "lsa", "dim": 8})
    with Engine(settings) as engine:
        ingest_samples(engine)
        assert not engine.indexes.vectors_available
        hybrid = engine.search("invoice payment")
        assert hybrid.warnings and hybrid.hits
        with pytest.raises(ModelNotTrainedError):
            engine.search("invoice", mode="semantic")
        stats = engine.reindex()
        assert stats["vectors_available"] and stats["chunks"] > 0
        assert engine.search("invoice payment", mode="semantic").hits
    with Engine(settings) as reopened:  # fitted model and vectors are persisted
        assert reopened.indexes.vectors_available
        assert reopened.search("invoice payment", mode="semantic").hits


def test_stats_and_health(engine: Engine) -> None:
    ingest_samples(engine)
    stats = engine.stats()
    assert stats["documents"] == 5 and stats["chunks"] >= 5
    assert stats["documents_by_status"]["ready"] == 5
    assert stats["indexes"]["vector_index"]["vectors"] == stats["chunks"]
    assert stats["storage"]["blob_bytes"] > 0
    health = engine.health()
    assert health["status"] == "ok" and health["checks"]["database"]["ok"]
