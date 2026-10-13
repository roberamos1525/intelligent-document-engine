"""Ingestion pipeline: validate -> store -> extract -> analyse -> chunk -> index -> classify."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from idengine.analysis.metadata import MetadataExtractor
from idengine.classify.service import ClassifierService
from idengine.config import Settings
from idengine.errors import EngineError, ExtractionError, ValidationError
from idengine.ingest.chunker import Chunker
from idengine.ingest.detect import detect_type, sanitize_filename
from idengine.ingest.extractors import extract
from idengine.ingest.hashing import content_hash, simhash, text_hash
from idengine.observability.logging import get_logger
from idengine.observability.metrics import EngineMetrics
from idengine.search.indexes import IndexManager
from idengine.storage.blobs import BlobStore
from idengine.storage.models import DocumentRecord, DocumentStatus
from idengine.storage.repositories import ChunkRepository, DocumentRepository, SummaryRepository

log = get_logger("ingest")

MAX_TAGS = 20
MAX_TAG_CHARS = 40
MAX_CLASSIFY_CHARS = 50_000


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    document: DocumentRecord
    duplicate: bool


def validate_tags(tags: Sequence[str] | None) -> list[str]:
    cleaned: list[str] = []
    for tag in tags or ():
        tag = tag.strip().lower()
        if not tag:
            continue
        if len(tag) > MAX_TAG_CHARS or not all(ch.isalnum() or ch in "-_.:" for ch in tag):
            raise ValidationError(
                f"invalid tag '{tag[:50]}': use up to {MAX_TAG_CHARS} letters, digits, -_.:"
            )
        if tag not in cleaned:
            cleaned.append(tag)
    if len(cleaned) > MAX_TAGS:
        raise ValidationError(f"at most {MAX_TAGS} tags are allowed")
    return cleaned


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        summaries: SummaryRepository,
        blobs: BlobStore,
        indexes: IndexManager,
        classifier: ClassifierService,
        metrics: EngineMetrics,
        metadata_extractor: MetadataExtractor | None = None,
    ) -> None:
        self.settings = settings
        self.documents = documents
        self.chunks = chunks
        self.summaries = summaries
        self.blobs = blobs
        self.indexes = indexes
        self.classifier = classifier
        self.metrics = metrics
        self.metadata = metadata_extractor or MetadataExtractor()
        self.chunker = Chunker(settings.chunking)

    # -- registration ------------------------------------------------------------
    def register(
        self,
        data: bytes,
        filename: str | None,
        *,
        tags: Sequence[str] | None = None,
        category: str | None = None,
    ) -> IngestOutcome:
        """Validate and store the upload; returns the existing record for exact duplicates."""
        name = sanitize_filename(filename)
        detected = detect_type(
            data,
            name,
            max_bytes=self.settings.ingest.max_file_bytes,
            allowed=set(self.settings.ingest.allowed_types),
        )
        clean_tags = validate_tags(tags)
        if category is not None and not (0 < len(category.strip()) <= 64):
            raise ValidationError("category must be 1-64 characters")
        digest = content_hash(data)
        existing = self.documents.by_content_hash(digest)
        if existing is not None:
            self.metrics.documents_ingested.inc(
                labels={"type": existing.doc_type, "outcome": "duplicate"}
            )
            log.info("duplicate upload", extra={"document_id": existing.id, "file_name": name})
            return IngestOutcome(existing, duplicate=True)
        self.blobs.put(digest, data)
        record = self.documents.create(
            filename=name,
            doc_type=detected.doc_type.value,
            mime_type=detected.mime,
            size_bytes=len(data),
            content_hash=digest,
            tags=clean_tags,
        )
        if category:
            record = self.documents.update(
                record.id,
                category=category.strip(),
                category_confidence=1.0,
                metadata={"category_source": "manual"},
            )
        return IngestOutcome(record, duplicate=False)

    # -- processing ----------------------------------------------------------------
    def process(self, document_id: str) -> DocumentRecord:
        """Extract, analyse, chunk and index a registered document."""
        started = time.perf_counter()
        record = self.documents.get(document_id)
        try:
            result = self._process(record)
        except EngineError as exc:
            self._fail(record, exc.message)
            raise
        except Exception as exc:  # unexpected bug in an extractor or analyser
            self._fail(record, f"internal error: {type(exc).__name__}")
            log.exception("ingestion crashed", extra={"document_id": document_id})
            raise ExtractionError(f"processing failed: {type(exc).__name__}: {exc}") from exc
        self.metrics.ingest_latency.observe(time.perf_counter() - started)
        self.metrics.documents_ingested.inc(labels={"type": record.doc_type, "outcome": "ok"})
        return result

    def _fail(self, record: DocumentRecord, message: str) -> None:
        self.metrics.documents_ingested.inc(labels={"type": record.doc_type, "outcome": "failed"})
        self.documents.update(record.id, status=DocumentStatus.FAILED, error=message[:1000])
        log.warning("ingestion failed", extra={"document_id": record.id, "error": message})

    def _process(self, record: DocumentRecord) -> DocumentRecord:
        data = self.blobs.get(record.content_hash)
        detected = detect_type(data, record.filename)
        document = extract(data, detected, self.settings.ingest)
        full_text = document.text

        meta = self.metadata.extract(
            document,
            filename=record.filename,
            document_frequency=self.indexes.bm25.df,
            total_documents=len(self.indexes.bm25),
        )
        fingerprint = simhash(full_text)
        thash = text_hash(full_text)
        duplicate_of: str | None = None
        same_text = self.documents.by_text_hash(thash, exclude=record.id)
        if same_text is not None:
            duplicate_of = same_text.id
        else:
            near = self.documents.near_duplicates(
                fingerprint,
                max_distance=self.settings.ingest.near_duplicate_distance,
                exclude=record.id,
            )
            if near:
                duplicate_of = near[0][0]

        drafts = self.chunker.chunk(document)
        if not drafts:
            raise ExtractionError("document produced no chunks")

        # Re-processing: drop stale chunks from the indexes first.
        stale = [c.id for c in self.chunks.for_document(record.id)]
        if stale:
            self.indexes.remove_chunks(stale)
        chunk_records = self.chunks.replace_for_document(
            record.id,
            [
                {
                    "ordinal": d.ordinal,
                    "text": d.text,
                    "token_count": d.token_count,
                    "heading_path": list(d.heading_path),
                    "page_start": d.page_start,
                    "page_end": d.page_end,
                }
                for d in drafts
            ],
        )

        category, confidence = record.category, record.category_confidence
        metadata: dict[str, Any] = {**record.metadata, **meta.to_dict()}
        manual = record.metadata.get("category_source") == "manual"
        if not manual and self.settings.ingest.auto_classify:
            prediction = self.classifier.predict_one(full_text[:MAX_CLASSIFY_CHARS])
            if prediction is not None:
                category, confidence = prediction.label, prediction.confidence
                metadata["category_source"] = "model"
                metadata["category_probabilities"] = prediction.probabilities

        updated = self.documents.update(
            record.id,
            status=DocumentStatus.READY,
            title=meta.title,
            authors=meta.authors,
            document_date=meta.document_date,
            language=meta.language,
            category=category,
            category_confidence=confidence,
            metadata=metadata,
            text_hash=thash,
            simhash=fingerprint,
            near_duplicate_of=duplicate_of,
            chunk_count=len(chunk_records),
            error=None,
        )
        self.summaries.invalidate(record.id)
        self.indexes.index_document(updated, chunk_records)
        self.indexes.flush()
        self.metrics.chunks_indexed.inc(len(chunk_records))
        log.info(
            "document ingested",
            extra={
                "document_id": record.id,
                "doc_type": record.doc_type,
                "chunks": len(chunk_records),
                "language": meta.language,
                "near_duplicate_of": duplicate_of,
            },
        )
        return updated

    def ingest(
        self,
        data: bytes,
        filename: str | None,
        *,
        tags: Sequence[str] | None = None,
        category: str | None = None,
    ) -> IngestOutcome:
        """Synchronous register + process."""
        outcome = self.register(data, filename, tags=tags, category=category)
        if outcome.duplicate and outcome.document.status is DocumentStatus.READY:
            return outcome
        return IngestOutcome(self.process(outcome.document.id), duplicate=outcome.duplicate)

    # -- maintenance ---------------------------------------------------------------
    def update(
        self,
        document_id: str,
        *,
        title: str | None = None,
        tags: Sequence[str] | None = None,
        category: str | None = None,
    ) -> DocumentRecord:
        record = self.documents.get(document_id)
        fields: dict[str, Any] = {}
        index_changes: dict[str, Any] = {}
        if title is not None:
            title = title.strip()
            if not 0 < len(title) <= 300:
                raise ValidationError("title must be 1-300 characters")
            fields["title"] = index_changes["title"] = title
        if tags is not None:
            fields["tags"] = index_changes["tags"] = validate_tags(tags)
        if category is not None:
            category = category.strip()
            if not 0 < len(category) <= 64:
                raise ValidationError("category must be 1-64 characters")
            fields["category"] = index_changes["category"] = category
            fields["category_confidence"] = 1.0
            fields["metadata"] = {**record.metadata, "category_source": "manual"}
        if not fields:
            return record
        updated = self.documents.update(document_id, **fields)
        if index_changes:
            self.indexes.update_metadata(
                [c.id for c in self.chunks.for_document(document_id)], index_changes
            )
            self.indexes.flush()
        return updated

    def delete(self, document_id: str) -> None:
        record = self.documents.get(document_id)
        chunk_ids = [c.id for c in self.chunks.for_document(document_id)]
        self.indexes.remove_chunks(chunk_ids)
        self.documents.delete(document_id)  # chunks and summaries cascade
        self.blobs.delete(record.content_hash)
        self.indexes.flush()
        log.info("document deleted", extra={"document_id": document_id, "chunks": len(chunk_ids)})

    def chunk_texts(self, document_id: str) -> list[str]:
        """Chunk texts in order (map-reduce summarisation works chunk by chunk)."""
        return [c.text for c in self.chunks.for_document(document_id)]

    def full_text(self, document_id: str, *, include_headings: bool = True) -> str:
        """Re-extract the clean full text (chunks overlap, so they are not concatenated).

        ``include_headings=False`` returns body text only, which keeps headings
        out of extractive summaries.
        """
        record = self.documents.get(document_id)
        data = self.blobs.get(record.content_hash)
        document = extract(data, detect_type(data, record.filename), self.settings.ingest)
        if include_headings:
            return document.text
        return "\n\n".join(b.text for b in document.blocks if not b.is_heading)
