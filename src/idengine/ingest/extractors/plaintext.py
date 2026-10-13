"""Plain-text extractor."""

from __future__ import annotations

from idengine.errors import ExtractionError
from idengine.ingest.detect import decode_text
from idengine.ingest.extractors.structure import blocks_from_plain_text
from idengine.ingest.model import BlockKind, DocType, ExtractedDocument


def extract_plaintext(data: bytes) -> ExtractedDocument:
    text, encoding = decode_text(data)
    blocks = blocks_from_plain_text(text)
    if not blocks:
        raise ExtractionError("document contains no text")
    title: str | None = None
    first = blocks[0]
    if first.kind is BlockKind.HEADING or (
        first.kind is BlockKind.PARAGRAPH and len(first.text) <= 80 and len(blocks) > 1
    ):
        title = first.text
    return ExtractedDocument(
        doc_type=DocType.TXT,
        blocks=blocks,
        title=title,
        properties={"encoding": encoding},
    )
