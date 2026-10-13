"""Owns the embedder, the vector index and the BM25 index, and keeps them in
sync with the SQLite chunk table.

SQLite is the source of truth: if index files are missing, corrupt, built
for another embedding space, or disagree with the database about how many
chunks exist, the indexes are rebuilt from the stored chunk texts.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from idengine.config import Settings
from idengine.embeddings import EmbeddingProvider, create_embedder
from idengine.errors import IndexCorruptedError
from idengine.observability.logging import get_logger
from idengine.search.bm25 import BM25Index
from idengine.storage.models import ChunkRecord, DocumentRecord
from idengine.storage.repositories import ChunkRepository, DocumentRepository
from idengine.text.tokenize import analyze
from idengine.vectorindex import VectorIndex, create_index

log = get_logger("search.indexes")

_EMBED_BATCH = 256
COMPACTION_THRESHOLD = 0.5


class IndexManager:
    def __init__(
        self,
        settings: Settings,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        *,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings
        self.documents = documents
        self.chunks = chunks
        self.directory = settings.storage.index_dir
        self.embedder = embedder or create_embedder(settings.embeddings)
        self.vectors: VectorIndex = create_index(
            settings.index, self.embedder.dim, seed=settings.embeddings.seed
        )
        self.bm25 = BM25Index(k1=settings.search.bm25_k1, b=settings.search.bm25_b)
        self._lock = threading.RLock()
        self._dirty = False
        self.last_rebuild: dict[str, Any] | None = None

    # -- lifecycle ---------------------------------------------------------------
    @property
    def vectors_available(self) -> bool:
        """False while a corpus-fitted embedder has not been fitted yet."""
        return self.embedder.is_fitted

    def open(self) -> None:
        """Load persisted indexes, rebuilding from SQLite when they are unusable."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.embedder.load(self.directory)
            try:
                vectors_ok = self.vectors_available and self.vectors.load(
                    self.directory, self.embedder.signature()
                )
                bm25_ok = self.bm25.load(self.directory / "bm25.json")
            except IndexCorruptedError as exc:
                log.warning("index files unusable; rebuilding", extra={"error": str(exc)})
                vectors_ok = bm25_ok = False
            expected = self.chunks.count()
            consistent = len(self.bm25) == expected and (
                not self.vectors_available or len(self.vectors) == expected
            )
            usable = bm25_ok and (vectors_ok or not self.vectors_available) and consistent
            if not usable and (expected or not bm25_ok):
                log.info(
                    "rebuilding indexes from database",
                    extra={"chunks": expected, "bm25": len(self.bm25)},
                )
                self.rebuild(refit=False)

    def flush(self) -> None:
        """Persist indexes if they changed since the last flush."""
        with self._lock:
            if not self._dirty:
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.vectors_available:
                self.vectors.save(self.directory, self.embedder.signature())
            self.bm25.save(self.directory / "bm25.json")
            self._dirty = False

    # -- mutations ---------------------------------------------------------------
    def index_document(self, document: DocumentRecord, chunks: Sequence[ChunkRecord]) -> None:
        meta = document.filter_metadata()
        with self._lock:
            ids = [c.id for c in chunks]
            texts = [c.index_text() for c in chunks]
            metadata = [{**meta, "chunk_ordinal": c.ordinal, "section": c.section} for c in chunks]
            for chunk_id, text, chunk_meta in zip(ids, texts, metadata, strict=True):
                self.bm25.add(chunk_id, analyze(text), chunk_meta)
            if self.vectors_available and ids:
                for start in range(0, len(ids), _EMBED_BATCH):
                    end = start + _EMBED_BATCH
                    vectors = self.embedder.embed(texts[start:end])
                    self.vectors.add(ids[start:end], vectors, metadata[start:end])
            self._dirty = True

    def remove_chunks(self, chunk_ids: Iterable[str]) -> None:
        ids = list(chunk_ids)
        with self._lock:
            self.bm25.remove(ids)
            self.vectors.remove(ids)
            # Deleted rows stay as tombstones (HNSW keeps routing through them); compact
            # once they dominate so memory and search cost do not grow without bound.
            if self.vectors.storage.tombstone_ratio() > COMPACTION_THRESHOLD:
                self.vectors.rebuild()
            self._dirty = True

    def update_metadata(self, chunk_ids: Iterable[str], changes: Mapping[str, Any]) -> None:
        ids = list(chunk_ids)
        with self._lock:
            self.bm25.update_metadata(ids, changes)
            self.vectors.update_metadata(ids, changes)
            self._dirty = True

    def rebuild(self, *, refit: bool = True) -> dict[str, Any]:
        """Recreate both indexes from the database (refitting the embedder if needed)."""
        started = time.perf_counter()
        with self._lock:
            docs = {d.id: d for d in self.documents.all_ready()}
            records = [c for c, _ in self.chunks.iter_ready()]
            if self.embedder.requires_fit and (refit or not self.embedder.is_fitted):
                texts = [c.index_text() for c in records]
                if len(texts) >= 2:
                    self.embedder.fit(texts)
                    self.embedder.save(self.directory)
            self.vectors = create_index(
                self.settings.index, self.embedder.dim, seed=self.settings.embeddings.seed
            )
            self.bm25 = BM25Index(k1=self.settings.search.bm25_k1, b=self.settings.search.bm25_b)
            by_doc: dict[str, list[ChunkRecord]] = {}
            for record in records:
                by_doc.setdefault(record.document_id, []).append(record)
            for doc_id, doc_chunks in by_doc.items():
                if doc_id in docs:
                    self.index_document(docs[doc_id], doc_chunks)
            self._dirty = True
            self.flush()
            self.last_rebuild = {
                "documents": len(by_doc),
                "chunks": len(records),
                "seconds": round(time.perf_counter() - started, 3),
                "embedder": self.embedder.signature(),
                "vectors_available": self.vectors_available,
            }
        log.info("indexes rebuilt", extra=self.last_rebuild)
        return self.last_rebuild

    def stats(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder.signature(),
            "vectors_available": self.vectors_available,
            "vector_index": self.vectors.stats(),
            "bm25": self.bm25.stats(),
            "dirty": self._dirty,
            "last_rebuild": self.last_rebuild,
        }
