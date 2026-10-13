"""Composition root: builds and owns every service of the document engine."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from idengine import __version__
from idengine.classify import ClassifierService, Prediction, TrainingResult
from idengine.config import Settings
from idengine.embeddings import EmbeddingProvider
from idengine.errors import FileTooLargeError, ModelNotTrainedError, ValidationError
from idengine.ingest.service import IngestionService, IngestOutcome
from idengine.jobs import JobQueue
from idengine.llm import LLMAdapter, create_llm
from idengine.observability.logging import get_logger
from idengine.observability.metrics import EngineMetrics
from idengine.rag import Answer, RagPipeline, Reranker
from idengine.search.indexes import IndexManager
from idengine.search.service import SearchMode, SearchResponse, SearchService
from idengine.storage import (
    BlobStore,
    ChunkRepository,
    Database,
    DocumentRepository,
    DocumentStatus,
    JobRecord,
    JobRepository,
    SummaryRepository,
)
from idengine.summarize import Summarizer, Summary, SummaryMethod

log = get_logger("engine")

MAX_INLINE_TEXT_CHARS = 200_000


@dataclass(frozen=True, slots=True)
class LabeledExample:
    label: str
    text: str | None = None
    document_id: str | None = None


class Engine:
    """High-level facade used by the HTTP API, the CLI and the evaluation runner."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLMAdapter | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings
        self.started_at = time.time()
        storage = settings.storage
        storage.data_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = EngineMetrics()
        self.db = Database(storage.database_path)
        self.db.migrate()
        self.blobs = BlobStore(storage.blob_dir)
        self.documents = DocumentRepository(self.db)
        self.chunks = ChunkRepository(self.db)
        self.summaries = SummaryRepository(self.db)

        self.indexes = IndexManager(settings, self.documents, self.chunks, embedder=embedder)
        self.indexes.open()
        self.classifier = ClassifierService(settings.classifier, storage.model_dir)
        self.classifier.load()

        self.ingestion = IngestionService(
            settings,
            self.documents,
            self.chunks,
            self.summaries,
            self.blobs,
            self.indexes,
            self.classifier,
            self.metrics,
        )
        self.search_service = SearchService(
            settings.search, self.indexes, self.chunks, self.metrics
        )
        self.llm = llm or create_llm(settings.llm)
        self.rag = RagPipeline(
            settings.rag,
            settings.llm,
            self.search_service,
            Reranker(self.indexes.embedder, mmr_lambda=settings.rag.mmr_lambda),
            self.llm,
            self.metrics,
        )
        self.summarizer = Summarizer(self.llm)
        self.jobs = JobQueue(JobRepository(self.db), settings.jobs, self.metrics)
        self._register_jobs()
        self._closed = False
        self._close_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------------
    def start_workers(self, workers: int | None = None) -> None:
        self.jobs.start(workers)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self.jobs.stop()
        self.indexes.flush()
        self.llm.close()
        self.db.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- jobs ------------------------------------------------------------------------
    def _register_jobs(self) -> None:
        self.jobs.register("ingest", self._job_ingest)
        self.jobs.register("summarize", self._job_summarize)
        self.jobs.register(
            "reindex",
            lambda payload: self.reindex(refit=bool(payload.get("refit", True))),
            max_attempts=1,
        )
        self.jobs.register("train_classifier", self._job_train, max_attempts=1)

    def _job_ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = self.ingestion.process(str(payload["document_id"]))
        return {
            "document_id": record.id,
            "chunks": record.chunk_count,
            "status": record.status.value,
        }

    def _job_summarize(self, payload: dict[str, Any]) -> dict[str, Any]:
        summary = self.summarize_document(
            str(payload["document_id"]),
            method=payload.get("method", SummaryMethod.TEXTRANK.value),
            max_sentences=int(payload.get("max_sentences", 5)),
        )
        return {
            "document_id": payload["document_id"],
            "method": summary.method.value,
            "summary": summary.text,
        }

    def _job_train(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = [LabeledExample(**item) for item in payload.get("examples", [])]
        result = self.train_classifier(
            examples,
            algorithm=payload.get("algorithm"),
            test_ratio=float(payload.get("test_ratio", 0.2)),
            folds=int(payload.get("folds", 0)),
        )
        return result.to_dict()

    # -- documents -----------------------------------------------------------------
    def ingest_bytes(
        self,
        data: bytes,
        filename: str | None,
        *,
        tags: Sequence[str] | None = None,
        category: str | None = None,
        background: bool = False,
    ) -> tuple[IngestOutcome, JobRecord | None]:
        """Ingest an upload synchronously, or register it and enqueue an ingest job."""
        if not background:
            return self.ingestion.ingest(data, filename, tags=tags, category=category), None
        outcome = self.ingestion.register(data, filename, tags=tags, category=category)
        if outcome.duplicate and outcome.document.status is DocumentStatus.READY:
            return outcome, None
        job = self.jobs.submit("ingest", {"document_id": outcome.document.id})
        return outcome, job

    def ingest_path(self, path: Path, **kwargs: Any) -> tuple[IngestOutcome, JobRecord | None]:
        if not path.is_file():
            raise ValidationError(f"not a file: {path}")
        size = path.stat().st_size
        if size > self.settings.ingest.max_file_bytes:
            # Check before reading so huge files are never loaded into memory.
            raise FileTooLargeError(
                f"file is {size} bytes; limit is {self.settings.ingest.max_file_bytes}"
            )
        return self.ingest_bytes(path.read_bytes(), path.name, **kwargs)

    # -- retrieval -------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        mode: SearchMode | str = SearchMode.HYBRID,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        group_by_document: bool = False,
    ) -> SearchResponse:
        return self.search_service.search(
            query, mode=mode, top_k=top_k, filters=filters, group_by_document=group_by_document
        )

    def ask(
        self,
        question: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        mode: SearchMode | str = SearchMode.HYBRID,
    ) -> Answer:
        return self.rag.ask(question, filters=filters, top_k=top_k, mode=mode)

    # -- summaries -------------------------------------------------------------------
    def summarize_document(
        self,
        document_id: str,
        *,
        method: SummaryMethod | str = SummaryMethod.TEXTRANK,
        max_sentences: int = 5,
        use_cache: bool = True,
    ) -> Summary:
        method = SummaryMethod(method)
        record = self.documents.get(document_id)
        if record.status is not DocumentStatus.READY:
            raise ValidationError(f"document is {record.status.value}, not ready")
        cache_key = f"{method.value}:{max_sentences}:{self.llm.name}"
        if use_cache:
            cached = self.summaries.get(document_id, cache_key)
            if cached is not None:
                return Summary(method, cached, [cached], "cache")
        if method is SummaryMethod.TEXTRANK:
            summary = self.summarizer.summarize(
                [self.ingestion.full_text(document_id, include_headings=False)],
                method=method,
                max_sentences=max_sentences,
            )
        else:
            summary = self.summarizer.summarize(
                self.ingestion.chunk_texts(document_id), method=method, max_sentences=max_sentences
            )
        if summary.text:
            self.summaries.put(document_id, cache_key, summary.text)
        return summary

    def summarize_text(
        self,
        text: str,
        *,
        method: SummaryMethod | str = SummaryMethod.TEXTRANK,
        max_sentences: int = 5,
    ) -> Summary:
        if not text.strip():
            raise ValidationError("text must not be empty")
        if len(text) > MAX_INLINE_TEXT_CHARS:
            raise ValidationError(f"text exceeds {MAX_INLINE_TEXT_CHARS} characters")
        return self.summarizer.summarize([text], method=method, max_sentences=max_sentences)

    # -- classification ----------------------------------------------------------------
    def train_classifier(
        self,
        examples: Sequence[LabeledExample],
        *,
        algorithm: str | None = None,
        test_ratio: float = 0.2,
        folds: int = 0,
    ) -> TrainingResult:
        texts: list[str] = []
        labels: list[str] = []
        for example in examples:
            if (example.text is None) == (example.document_id is None):
                raise ValidationError("each example needs exactly one of 'text' or 'document_id'")
            if example.document_id is not None:
                texts.append(self.ingestion.full_text(example.document_id))
            else:
                assert example.text is not None
                if len(example.text) > MAX_INLINE_TEXT_CHARS:
                    raise ValidationError("training text is too long")
                texts.append(example.text)
            labels.append(example.label)
        return self.classifier.train(
            texts, labels, algorithm=algorithm, test_ratio=test_ratio, folds=folds
        )

    def classify_text(self, text: str) -> Prediction:
        if not text.strip():
            raise ValidationError("text must not be empty")
        if len(text) > MAX_INLINE_TEXT_CHARS:
            raise ValidationError(f"text exceeds {MAX_INLINE_TEXT_CHARS} characters")
        return self.classifier.predict([text])[0]

    def classify_document(self, document_id: str, *, apply: bool = True) -> Prediction:
        """Predict a document's category and (optionally) store it unless set manually."""
        record = self.documents.get(document_id)
        if not self.classifier.is_trained:
            raise ModelNotTrainedError("no classifier has been trained yet")
        prediction = self.classifier.predict([self.ingestion.full_text(document_id)])[0]
        if apply and record.metadata.get("category_source") != "manual":
            self.documents.update(
                document_id,
                category=prediction.label,
                category_confidence=prediction.confidence,
                metadata={
                    **record.metadata,
                    "category_source": "model",
                    "category_probabilities": prediction.probabilities,
                },
            )
            self.indexes.update_metadata(
                [c.id for c in self.chunks.for_document(document_id)],
                {"category": prediction.label},
            )
            self.indexes.flush()
        return prediction

    # -- maintenance -----------------------------------------------------------------
    def reindex(self, *, refit: bool = True) -> dict[str, Any]:
        return self.indexes.rebuild(refit=refit)

    def stats(self) -> dict[str, Any]:
        documents = self.documents.count()
        chunks = self.chunks.count()
        self.metrics.documents_gauge.set(documents)
        self.metrics.chunks_gauge.set(chunks)
        job_counts = self.jobs.repository.counts()
        self.metrics.queue_depth.set(job_counts.get("queued", 0))
        return {
            "version": __version__,
            "documents": documents,
            "documents_by_status": {s.value: self.documents.count(s) for s in DocumentStatus},
            "categories": self.documents.category_counts(),
            "chunks": chunks,
            "jobs": job_counts,
            "indexes": self.indexes.stats(),
            "classifier": self.classifier.status(),
            "llm": {"provider": self.llm.name, "model": self.llm.model},
            "storage": {"blob_bytes": self.blobs.total_bytes()},
            "uptime_seconds": round(time.time() - self.started_at, 1),
        }

    def health(self) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        try:
            checks["database"] = {"ok": True, "latency_ms": round(self.db.ping() * 1000, 3)}
        except Exception as exc:  # report, never raise from a health check
            checks["database"] = {"ok": False, "error": str(exc)}
        checks["indexes"] = {
            "ok": True,
            "vectors_available": self.indexes.vectors_available,
            "chunks": len(self.indexes.bm25),
        }
        checks["workers"] = {
            "ok": self.jobs.running or self.settings.jobs.workers == 0,
            "running": self.jobs.running,
        }
        checks["llm"] = self.llm.health()
        status = "ok" if all(c.get("ok", False) for c in checks.values()) else "degraded"
        return {"status": status, "version": __version__, "checks": checks}
