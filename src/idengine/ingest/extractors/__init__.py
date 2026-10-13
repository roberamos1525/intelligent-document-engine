"""Format-specific text extractors and the dispatch function."""

from __future__ import annotations

from idengine.config import IngestSettings
from idengine.ingest.detect import DetectedType
from idengine.ingest.extractors.docx import extract_docx
from idengine.ingest.extractors.html import extract_html
from idengine.ingest.extractors.markdown import extract_markdown
from idengine.ingest.extractors.pdf import extract_pdf
from idengine.ingest.extractors.plaintext import extract_plaintext
from idengine.ingest.model import DocType, ExtractedDocument


def extract(data: bytes, detected: DetectedType, settings: IngestSettings) -> ExtractedDocument:
    """Dispatch to the extractor for the detected type, applying configured limits."""
    match detected.doc_type:
        case DocType.PDF:
            return extract_pdf(data, max_pages=settings.max_pdf_pages)
        case DocType.DOCX:
            return extract_docx(
                data,
                max_uncompressed=settings.max_docx_uncompressed_bytes,
                max_ratio=settings.max_docx_compression_ratio,
            )
        case DocType.HTML:
            return extract_html(data)
        case DocType.MARKDOWN:
            return extract_markdown(data)
        case DocType.TXT:
            return extract_plaintext(data)


__all__ = ["extract"]
