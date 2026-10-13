"""IndexManager and SearchService against real SQLite repositories."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from conftest import make_settings
from idengine.config import Settings
from idengine.errors import ValidationError
from idengine.observability.metrics import EngineMetrics
from idengine.search.indexes import IndexManager
from idengine.search.service import SearchHit, SearchMode, SearchService
from idengine.storage import ChunkRepository, Database, DocumentRepository, DocumentStatus

DOCS = {
    "billing.txt": [
        ("Invoices", "Invoices are payable within thirty days of receipt."),
        ("Invoices", "A late fee of two percent applies to overdue invoices."),
    ],
    "people.txt": [
        ("Leave", "Employees receive sixteen weeks of parental leave."),
        ("Remote", "Remote work requires the corporate VPN."),
    ],
}


class Harness:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings.storage.database_path)
        self.db.migrate()
        self.documents = DocumentRepository(self.db)
        self.chunks = ChunkRepository(self.db)
        self.metrics = EngineMetrics()
        self.indexes = IndexManager(settings, self.documents, self.chunks)
        self.indexes.open()
        self.search = SearchService(settings.search, self.indexes, self.chunks, self.metrics)

    def add(self, name: str, sections: list[tuple[str, str]], **fields: object) -> str:
        record = self.documents.create(
            filename=name,
            doc_type="txt",
            mime_type="text/plain",
            size_bytes=1,
            content_hash=name.encode().hex().ljust(64, "0")[:64],
        )
        record = self.documents.update(record.id, status=DocumentStatus.READY, title=name, **fields)
        chunks = self.chunks.replace_for_document(
            record.id,
            [
                {"ordinal": i, "text": text, "token_count": len(text.split()), "heading_path": [h]}
                for i, (h, text) in enumerate(sections)
            ],
        )
        self.indexes.index_document(record, chunks)
        return record.id

    def close(self) -> None:
        self.db.close()


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    h = Harness(make_settings(tmp_path / "data", index={"kind": "flat"}))
    for name, sections in DOCS.items():
        h.add(name, sections, category="finance" if "billing" in name else "hr")
    yield h
    h.close()


def test_modes_return_scores_from_the_right_source(harness: Harness) -> None:
    lexical = harness.search.search("late fee", mode="lexical", top_k=2)
    assert lexical.hits[0].bm25_score and lexical.hits[0].vector_score is None
    assert lexical.hits[0].ranks == {"bm25": 1}
    semantic = harness.search.search("late fee", mode="semantic", top_k=2)
    assert semantic.hits[0].vector_score and semantic.hits[0].bm25_score is None
    hybrid = harness.search.search("late fee", mode=SearchMode.HYBRID, top_k=2)
    assert hybrid.hits[0].text.startswith("A late fee")
    assert set(hybrid.hits[0].ranks) == {"vector", "bm25"}
    assert harness.metrics.search_latency.count({"mode": "hybrid"}) == 1


def test_filters_grouping_and_min_score(harness: Harness) -> None:
    hr_only = harness.search.search("leave invoices vpn", filters={"category": "hr"}, top_k=10)
    assert {h.metadata["category"] for h in hr_only.hits} == {"hr"}
    grouped = harness.search.search("invoices leave", group_by_document=True, top_k=10)
    assert len(grouped.hits) == len({h.document_id for h in grouped.hits}) == 2
    strict = harness.search.search("invoices", mode="lexical", min_score=1000.0)
    assert strict.hits == []


def test_query_validation_and_warnings(harness: Harness) -> None:
    with pytest.raises(ValidationError, match="empty"):
        harness.search.search("   ")
    with pytest.raises(ValidationError, match="top_k"):
        harness.search.search("x", top_k=0)
    with pytest.raises(ValidationError, match="characters"):
        harness.search.search("x" * 3000)
    stopwords = harness.search.search("the and of", mode="hybrid")
    assert any("no searchable terms" in w for w in stopwords.warnings)


def test_index_manager_flush_and_reload(harness: Harness) -> None:
    assert harness.indexes.stats()["dirty"]
    harness.indexes.flush()
    assert not harness.indexes.stats()["dirty"]
    harness.indexes.flush()  # no-op when clean
    reloaded = IndexManager(harness.settings, harness.documents, harness.chunks)
    reloaded.open()
    assert reloaded.last_rebuild is None  # files were consistent, no rebuild
    assert len(reloaded.bm25) == len(harness.indexes.bm25) == 4


def test_index_manager_detects_inconsistency(harness: Harness) -> None:
    harness.indexes.flush()
    extra = harness.add("late.txt", [("Notes", "Another late fee note.")])
    # The new chunk was indexed in memory but never flushed: files on disk are stale.
    reloaded = IndexManager(harness.settings, harness.documents, harness.chunks)
    reloaded.open()
    assert reloaded.last_rebuild is not None and reloaded.last_rebuild["chunks"] == 5
    assert any(m["document_id"] == extra for m in reloaded.bm25.metadata.values())


def test_metadata_updates_and_removal(harness: Harness) -> None:
    doc_id = next(r.id for r in harness.documents.all_ready() if r.filename == "people.txt")
    chunk_ids = [c.id for c in harness.chunks.for_document(doc_id)]
    harness.indexes.update_metadata(chunk_ids, {"tags": ["policy"]})
    tagged = harness.search.search("leave", filters={"tags": "policy"})
    assert {h.document_id for h in tagged.hits} == {doc_id}
    harness.indexes.remove_chunks(chunk_ids)
    assert harness.search.search("leave", filters={"tags": "policy"}).hits == []
    assert len(harness.indexes.vectors) == 2
    assert harness.indexes.vectors.stats()["tombstone_ratio"] == 0.5  # at threshold: kept
    billing = next(r.id for r in harness.documents.all_ready() if r.filename == "billing.txt")
    harness.indexes.remove_chunks([harness.chunks.for_document(billing)[0].id])
    stats = harness.indexes.vectors.stats()
    assert stats["tombstones"] == 0 and stats["vectors"] == 1  # compacted automatically


def test_rebuild_reports_statistics(harness: Harness) -> None:
    stats = harness.indexes.rebuild()
    assert stats["documents"] == 2 and stats["chunks"] == 4
    assert stats["embedder"]["provider"] == "hashing" and stats["vectors_available"]


def test_citation_labels() -> None:
    def make(**kw: object) -> SearchHit:
        base: dict[str, object] = {
            "chunk_id": "c",
            "document_id": "d",
            "title": "Report",
            "filename": "r.pdf",
            "text": "",
            "snippet": "",
            "highlights": [],
            "score": 0.0,
            "heading_path": [],
            "page_start": None,
            "page_end": None,
        }
        base.update(kw)
        return SearchHit(**base)  # type: ignore[arg-type]

    assert make().citation_label == "Report"
    assert make(page_start=3, page_end=3).citation_label == "Report, p. 3"
    assert make(page_start=3, page_end=5).citation_label == "Report, pp. 3-5"
    assert make(heading_path=["Report", "Costs"]).citation_label == "Report, Costs"
    assert make(title=None, heading_path=["A", "B"]).citation_label == "r.pdf, A > B"
