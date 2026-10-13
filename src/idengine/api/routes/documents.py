"""Document upload, CRUD, chunks, content download, summaries and classification."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, Path, Query, Response, UploadFile, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from idengine.api.deps import EngineDep, Reader, Writer
from idengine.api.schemas import (
    ChunkOut,
    DocumentList,
    DocumentOut,
    DocumentPatch,
    JobOut,
    PredictionOut,
    SummarizeDocumentRequest,
    SummaryOut,
    UploadResponse,
)
from idengine.errors import FileTooLargeError, ValidationError
from idengine.storage import DocumentQuery

router = APIRouter(prefix="/v1/documents", tags=["documents"])

DocumentId = Annotated[str, Path(pattern=r"^doc_[0-9a-f]{32}$", description="Document id")]
_READ_CHUNK = 1024 * 1024


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    """Read an upload without ever buffering more than ``limit + 1`` bytes."""
    parts: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise FileTooLargeError(
                f"file exceeds the {limit} byte limit", details={"limit": limit}
            )
        parts.append(chunk)
    return b"".join(parts)


def _split_tags(raw: str | None) -> list[str]:
    return [t for t in (raw or "").split(",") if t.strip()]


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"description": "Duplicate"}, 201: {"description": "Ingested synchronously"}},
)
async def upload_document(
    engine: EngineDep,
    _: Writer,
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT, Markdown or HTML")],
    tags: Annotated[str | None, Form(max_length=1000)] = None,
    category: Annotated[str | None, Form(max_length=64)] = None,
    wait: Annotated[bool, Query(description="Process synchronously before responding")] = False,
) -> JSONResponse:
    data = await _read_limited(file, engine.settings.ingest.max_file_bytes)
    outcome, job = await run_in_threadpool(
        engine.ingest_bytes,
        data,
        file.filename,
        tags=_split_tags(tags),
        category=category,
        background=not wait,
    )
    body = UploadResponse(
        document=DocumentOut.from_record(outcome.document),
        duplicate=outcome.duplicate,
        job=JobOut.from_record(job) if job else None,
    )
    if outcome.duplicate:
        code = status.HTTP_200_OK
    elif job is None:
        code = status.HTTP_201_CREATED
    else:
        code = status.HTTP_202_ACCEPTED
    return JSONResponse(body.model_dump(mode="json"), status_code=code)


@router.get("", response_model=DocumentList)
def list_documents(
    engine: EngineDep,
    _: Reader,
    status_: Annotated[
        Literal["processing", "ready", "failed"] | None, Query(alias="status")
    ] = None,
    category: Annotated[str | None, Query(max_length=64)] = None,
    language: Annotated[str | None, Query(max_length=8)] = None,
    doc_type: Annotated[Literal["pdf", "docx", "txt", "markdown", "html"] | None, Query()] = None,
    tag: Annotated[str | None, Query(max_length=40)] = None,
    q: Annotated[str | None, Query(max_length=200, description="Title substring")] = None,
    sort: Annotated[
        Literal["created_at", "updated_at", "title", "size_bytes", "document_date"], Query()
    ] = "created_at",
    order: Annotated[Literal["asc", "desc"], Query()] = "desc",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
) -> DocumentList:
    records, total = engine.documents.list(
        DocumentQuery(
            status=status_,
            category=category,
            language=language,
            doc_type=doc_type,
            tag=tag,
            title_contains=q,
            sort=sort,
            descending=order == "desc",
            limit=limit,
            offset=offset,
        )
    )
    return DocumentList(
        items=[DocumentOut.from_record(r, include_metadata=False) for r in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(engine: EngineDep, _: Reader, document_id: DocumentId) -> DocumentOut:
    return DocumentOut.from_record(engine.documents.get(document_id))


@router.patch("/{document_id}", response_model=DocumentOut)
def update_document(
    engine: EngineDep, _: Writer, document_id: DocumentId, patch: DocumentPatch
) -> DocumentOut:
    record = engine.ingestion.update(
        document_id, title=patch.title, tags=patch.tags, category=patch.category
    )
    return DocumentOut.from_record(record)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(engine: EngineDep, _: Writer, document_id: DocumentId) -> Response:
    engine.ingestion.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{document_id}/chunks", response_model=list[ChunkOut])
def document_chunks(engine: EngineDep, _: Reader, document_id: DocumentId) -> list[ChunkOut]:
    engine.documents.get(document_id)
    return [ChunkOut.from_record(c) for c in engine.chunks.for_document(document_id)]


@router.get("/{document_id}/content")
def document_content(engine: EngineDep, _: Reader, document_id: DocumentId) -> Response:
    record = engine.documents.get(document_id)
    data = engine.blobs.get(record.content_hash)
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in record.filename)
    return Response(
        content=data,
        media_type=record.mime_type,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/{document_id}/reprocess", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED
)
def reprocess_document(engine: EngineDep, _: Writer, document_id: DocumentId) -> JobOut:
    engine.documents.get(document_id)
    return JobOut.from_record(engine.jobs.submit("ingest", {"document_id": document_id}))


@router.post("/{document_id}/summarize", response_model=SummaryOut | JobOut)
def summarize_document(
    engine: EngineDep,
    _: Reader,
    document_id: DocumentId,
    request: SummarizeDocumentRequest,
) -> SummaryOut | JSONResponse:
    if request.background:
        engine.documents.get(document_id)
        job = engine.jobs.submit(
            "summarize",
            {
                "document_id": document_id,
                "method": request.method,
                "max_sentences": request.max_sentences,
            },
        )
        return JSONResponse(
            JobOut.from_record(job).model_dump(mode="json"), status_code=status.HTTP_202_ACCEPTED
        )
    summary = engine.summarize_document(
        document_id, method=request.method, max_sentences=request.max_sentences
    )
    return SummaryOut.from_summary(summary)


@router.post("/{document_id}/classify", response_model=PredictionOut)
def classify_document(
    engine: EngineDep,
    _: Writer,
    document_id: DocumentId,
    apply: Annotated[bool, Query(description="Store the predicted category")] = True,
) -> PredictionOut:
    record = engine.documents.get(document_id)
    if record.status.value != "ready":
        raise ValidationError(f"document is {record.status.value}, not ready")
    return PredictionOut.from_prediction(engine.classify_document(document_id, apply=apply))
