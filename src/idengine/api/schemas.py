"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from idengine.classify import Prediction, TrainingResult
from idengine.rag import Answer, Citation
from idengine.search.service import SearchHit, SearchResponse
from idengine.storage.models import ChunkRecord, DocumentRecord, JobRecord
from idengine.summarize import Summary

FilterSpec = dict[str, Any]


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- errors ----------------------------------------------------------------------------
class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None
    request_id: str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


# -- documents ---------------------------------------------------------------------------
class DocumentOut(BaseModel):
    id: str
    filename: str
    doc_type: str
    mime_type: str
    size_bytes: int
    content_hash: str
    status: str
    title: str | None
    authors: list[str]
    document_date: str | None
    language: str | None
    category: str | None
    category_confidence: float | None
    tags: list[str]
    near_duplicate_of: str | None
    chunk_count: int
    error: str | None
    created_at: str
    updated_at: str
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_record(cls, record: DocumentRecord, *, include_metadata: bool = True) -> DocumentOut:
        return cls(
            id=record.id,
            filename=record.filename,
            doc_type=record.doc_type,
            mime_type=record.mime_type,
            size_bytes=record.size_bytes,
            content_hash=record.content_hash,
            status=record.status.value,
            title=record.title,
            authors=record.authors,
            document_date=record.document_date,
            language=record.language,
            category=record.category,
            category_confidence=record.category_confidence,
            tags=record.tags,
            near_duplicate_of=record.near_duplicate_of,
            chunk_count=record.chunk_count,
            error=record.error,
            created_at=record.created_at,
            updated_at=record.updated_at,
            metadata=record.metadata if include_metadata else None,
        )


class JobOut(BaseModel):
    id: str
    type: str
    status: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int
    priority: int
    result: Any = None
    error: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None

    @classmethod
    def from_record(cls, job: JobRecord) -> JobOut:
        return cls(
            id=job.id,
            type=job.type,
            status=job.status.value,
            payload=job.payload,
            attempts=job.attempts,
            max_attempts=job.max_attempts,
            priority=job.priority,
            result=job.result,
            error=job.error,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )


class UploadResponse(BaseModel):
    document: DocumentOut
    duplicate: bool
    job: JobOut | None = None


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class DocumentPatch(_Request):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    tags: list[str] | None = Field(default=None, max_length=20)
    category: str | None = Field(default=None, min_length=1, max_length=64)


class ChunkOut(BaseModel):
    id: str
    ordinal: int
    text: str
    token_count: int
    heading_path: list[str]
    page_start: int | None
    page_end: int | None

    @classmethod
    def from_record(cls, chunk: ChunkRecord) -> ChunkOut:
        return cls(
            id=chunk.id,
            ordinal=chunk.ordinal,
            text=chunk.text,
            token_count=chunk.token_count,
            heading_path=chunk.heading_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )


# -- search / ask ----------------------------------------------------------------------
class SearchRequest(_Request):
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["semantic", "lexical", "hybrid"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=100)
    filters: FilterSpec | None = None
    group_by_document: bool = False


class SearchHitOut(BaseModel):
    chunk_id: str
    document_id: str
    title: str | None
    filename: str | None
    snippet: str
    highlights: list[tuple[int, int]]
    text: str | None
    score: float
    vector_score: float | None
    bm25_score: float | None
    ranks: dict[str, int]
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    metadata: dict[str, Any]

    @classmethod
    def from_hit(cls, hit: SearchHit, *, include_text: bool) -> SearchHitOut:
        return cls(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            title=hit.title,
            filename=hit.filename,
            snippet=hit.snippet,
            highlights=hit.highlights,
            text=hit.text if include_text else None,
            score=hit.score,
            vector_score=hit.vector_score,
            bm25_score=hit.bm25_score,
            ranks=hit.ranks,
            heading_path=hit.heading_path,
            page_start=hit.page_start,
            page_end=hit.page_end,
            metadata=hit.metadata,
        )


class SearchResponseOut(BaseModel):
    query: str
    mode: str
    took_ms: float
    hits: list[SearchHitOut]
    warnings: list[str]

    @classmethod
    def from_response(cls, response: SearchResponse, *, include_text: bool) -> SearchResponseOut:
        return cls(
            query=response.query,
            mode=response.mode.value,
            took_ms=response.took_ms,
            hits=[SearchHitOut.from_hit(h, include_text=include_text) for h in response.hits],
            warnings=response.warnings,
        )


class AskRequest(_Request):
    question: str = Field(min_length=3, max_length=2000)
    filters: FilterSpec | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["semantic", "lexical", "hybrid"] = "hybrid"


class CitationOut(BaseModel):
    number: int
    chunk_id: str
    document_id: str
    title: str | None
    source: str
    page_start: int | None
    page_end: int | None
    heading_path: list[str]
    snippet: str

    @classmethod
    def from_citation(cls, c: Citation) -> CitationOut:
        return cls(
            number=c.number,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            title=c.title,
            source=c.source,
            page_start=c.page_start,
            page_end=c.page_end,
            heading_path=c.heading_path,
            snippet=c.snippet,
        )


class AnswerOut(BaseModel):
    question: str
    answer: str
    answerable: bool
    short_answer: str | None
    citations: list[CitationOut]
    provider: str
    model: str
    took_ms: float
    retrieval: dict[str, Any]
    usage: dict[str, int]

    @classmethod
    def from_answer(cls, a: Answer) -> AnswerOut:
        return cls(
            question=a.question,
            answer=a.answer,
            answerable=a.answerable,
            short_answer=a.short_answer,
            citations=[CitationOut.from_citation(c) for c in a.citations],
            provider=a.provider,
            model=a.model,
            took_ms=a.took_ms,
            retrieval=a.retrieval,
            usage=a.usage,
        )


# -- summaries -----------------------------------------------------------------------------
class SummarizeDocumentRequest(_Request):
    method: Literal["textrank", "mapreduce"] = "textrank"
    max_sentences: int = Field(default=5, ge=1, le=50)
    background: bool = False


class SummarizeTextRequest(_Request):
    text: str = Field(min_length=1, max_length=200_000)
    method: Literal["textrank", "mapreduce"] = "textrank"
    max_sentences: int = Field(default=5, ge=1, le=50)


class SummaryOut(BaseModel):
    method: str
    summary: str
    sentences: list[str]
    provider: str
    stages: list[dict[str, Any]]

    @classmethod
    def from_summary(cls, s: Summary) -> SummaryOut:
        return cls(
            method=s.method.value,
            summary=s.text,
            sentences=s.sentences,
            provider=s.provider,
            stages=s.stages,
        )


# -- classification ---------------------------------------------------------------------------
class TrainingExample(_Request):
    label: str = Field(min_length=1, max_length=64)
    text: str | None = Field(default=None, max_length=200_000)
    document_id: str | None = Field(default=None, max_length=64)


class TrainRequest(_Request):
    examples: list[TrainingExample] = Field(min_length=4, max_length=50_000)
    algorithm: Literal["naive_bayes", "logistic"] | None = None
    test_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    folds: int = Field(default=0, ge=0, le=20)
    background: bool = False

    @field_validator("examples")
    @classmethod
    def _exactly_one_source(cls, value: list[TrainingExample]) -> list[TrainingExample]:
        for i, example in enumerate(value):
            if (example.text is None) == (example.document_id is None):
                raise ValueError(f"examples[{i}] needs exactly one of 'text' or 'document_id'")
        return value


class TrainingResultOut(BaseModel):
    model_id: str
    algorithm: str
    labels: list[str]
    examples: int
    holdout: dict[str, Any] | None
    cross_validation: dict[str, Any] | None
    seconds: float

    @classmethod
    def from_result(cls, r: TrainingResult) -> TrainingResultOut:
        return cls(**r.to_dict())


class ClassifyRequest(_Request):
    text: str = Field(min_length=1, max_length=200_000)


class PredictionOut(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]

    @classmethod
    def from_prediction(cls, p: Prediction) -> PredictionOut:
        return cls(label=p.label, confidence=p.confidence, probabilities=p.probabilities)


# -- jobs / admin --------------------------------------------------------------------------------
class JobList(BaseModel):
    items: list[JobOut]
    total: int
    limit: int
    offset: int


class ReindexRequest(_Request):
    refit: bool = True
    background: bool = True
