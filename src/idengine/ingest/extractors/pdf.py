"""PDF extractor using pypdf, preserving page numbers."""

from __future__ import annotations

import io
import logging
from datetime import datetime

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from idengine.errors import ExtractionError, ValidationError
from idengine.ingest.extractors.structure import blocks_from_plain_text
from idengine.ingest.model import Block, DocType, ExtractedDocument

# pypdf logs recoverable syntax problems at WARNING level; they are noise here.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def _pdf_date(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    return None


def extract_pdf(data: bytes, *, max_pages: int = 500) -> ExtractedDocument:
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            # Many PDFs are "encrypted" with an empty user password.
            try:
                if not reader.decrypt(""):
                    raise ExtractionError("PDF is password protected")
            except (NotImplementedError, PdfReadError) as exc:
                raise ExtractionError("PDF is password protected") from exc
        page_count = len(reader.pages)
    except PdfReadError as exc:
        raise ExtractionError(f"unreadable PDF: {exc}") from exc

    if page_count > max_pages:
        raise ValidationError(
            f"PDF has {page_count} pages; limit is {max_pages}",
            details={"pages": page_count, "limit": max_pages},
        )

    blocks: list[Block] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pypdf raises a variety of internal errors
            raise ExtractionError(f"failed to extract text from page {number}: {exc}") from exc
        blocks.extend(blocks_from_plain_text(text, page=number))

    if not blocks:
        raise ExtractionError(
            "PDF contains no extractable text (scanned documents need OCR, which is not supported)"
        )

    title: str | None = None
    authors: list[str] = []
    created: str | None = None
    properties: dict[str, object] = {}
    try:
        info = reader.metadata
    except PdfReadError:
        info = None
    if info is not None:
        title = (info.title or "").strip() or None
        if info.author:
            authors = [
                a.strip() for a in str(info.author).replace(";", ",").split(",") if a.strip()
            ]
        try:
            created = _pdf_date(info.creation_date)
        except (ValueError, TypeError):
            created = None
        if info.subject:
            properties["subject"] = str(info.subject)
        if info.producer:
            properties["producer"] = str(info.producer)
    if title is None:
        title = next((b.text for b in blocks if b.is_heading), None)
    return ExtractedDocument(
        doc_type=DocType.PDF,
        blocks=blocks,
        title=title,
        authors=authors,
        created=created,
        page_count=page_count,
        properties=properties,
    )
