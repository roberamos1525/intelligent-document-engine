"""DOCX extractor using python-docx, with zip-bomb protection."""

from __future__ import annotations

import io
import re
import zipfile

import docx
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from idengine.errors import ExtractionError, ValidationError
from idengine.ingest.model import Block, BlockKind, DocType, ExtractedDocument
from idengine.text.normalize import clean_text

_HEADING_STYLE_RE = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)


def check_archive_limits(data: bytes, *, max_uncompressed: int, max_ratio: float) -> None:
    """Reject ZIP containers whose declared expansion is suspicious (zip bombs)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise ExtractionError("corrupt DOCX container") from exc
    total = sum(info.file_size for info in infos)
    compressed = sum(info.compress_size for info in infos) or 1
    if total > max_uncompressed:
        raise ValidationError(
            f"DOCX expands to {total} bytes; limit is {max_uncompressed}",
            details={"uncompressed": total, "limit": max_uncompressed},
        )
    if total / compressed > max_ratio and total > 1024 * 1024:
        raise ValidationError(
            "DOCX compression ratio is suspiciously high",
            details={"ratio": round(total / compressed, 1), "limit": max_ratio},
        )


def _iter_body(document: DocxDocument) -> list[Paragraph | Table]:
    items: list[Paragraph | Table] = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            items.append(Paragraph(child, document))
        elif child.tag == qn("w:tbl"):
            items.append(Table(child, document))
    return items


def _paragraph_block(paragraph: Paragraph) -> Block | None:
    text = clean_text(paragraph.text)
    if not text:
        return None
    style = (paragraph.style.name if paragraph.style is not None else "") or ""
    match = _HEADING_STYLE_RE.match(style)
    if match:
        return Block(BlockKind.HEADING, text, level=int(match.group(1)))
    if style.lower() == "title":
        return Block(BlockKind.HEADING, text, level=1)
    if style.lower().startswith("list") or (
        paragraph._p.pPr is not None and (paragraph._p.pPr.numPr is not None)
    ):
        return Block(BlockKind.LIST_ITEM, text)
    return Block(BlockKind.PARAGRAPH, text)


def _table_block(table: Table) -> Block | None:
    rows: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            value = clean_text(cell.text).replace("\n", " ")
            if not cells or cells[-1] != value:  # merged cells repeat their text
                cells.append(value)
        if any(cells):
            rows.append(" | ".join(cells))
    return Block(BlockKind.TABLE, "\n".join(rows)) if rows else None


def extract_docx(
    data: bytes, *, max_uncompressed: int = 200 * 1024 * 1024, max_ratio: float = 150.0
) -> ExtractedDocument:
    check_archive_limits(data, max_uncompressed=max_uncompressed, max_ratio=max_ratio)
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # python-docx raises KeyError/ValueError/lxml errors
        raise ExtractionError(f"unreadable DOCX: {exc}") from exc

    blocks: list[Block] = []
    title: str | None = None
    for item in _iter_body(document):
        block = _paragraph_block(item) if isinstance(item, Paragraph) else _table_block(item)
        if block is None:
            continue
        if (
            title is None
            and isinstance(item, Paragraph)
            and item.style is not None
            and (item.style.name or "").lower() == "title"
        ):
            title = block.text
        blocks.append(block)
    if not blocks:
        raise ExtractionError("document contains no text")

    props = document.core_properties
    title = (props.title or "").strip() or title
    if title is None:
        title = next((b.text for b in blocks if b.is_heading), None)
    authors = [a.strip() for a in re.split(r",|;", props.author or "") if a.strip()]
    created = props.created.date().isoformat() if props.created else None
    properties = {
        key: value
        for key, value in {
            "subject": props.subject,
            "keywords": props.keywords,
            "category": props.category,
        }.items()
        if value
    }
    return ExtractedDocument(
        doc_type=DocType.DOCX,
        blocks=blocks,
        title=title,
        authors=authors,
        created=created,
        language_hint=(props.language or "").split("-")[0].lower() or None,
        properties=properties,
    )
