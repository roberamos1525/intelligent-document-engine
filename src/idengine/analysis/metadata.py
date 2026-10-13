"""Document-level metadata extraction combining native properties and text analysis."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from idengine.analysis.dates import find_dates, parse_date
from idengine.analysis.entities import EntityExtractor, summarize_entities
from idengine.analysis.keywords import extract_keywords
from idengine.analysis.language import detect_language
from idengine.ingest.model import BlockKind, ExtractedDocument
from idengine.text.tokenize import words

_AUTHOR_LINE_RE = re.compile(
    r"^(?:by|authors?|written by|prepared by)\s*[:\-]?\s+"
    r"(?P<names>[A-Z][\w.'\- ]+(?:(?:,|;|\band\b|&)\s*[A-Z][\w.'\- ]+)*)$",
    re.IGNORECASE,
)
_DATE_LINE_RE = re.compile(r"^(?:date|dated|effective date|published)\s*[:\-]", re.IGNORECASE)
_WORDS_PER_MINUTE = 230


@dataclass(slots=True)
class DocumentMetadata:
    title: str | None
    authors: list[str]
    document_date: str | None
    language: str
    language_confidence: float
    keywords: list[str]
    entities: dict[str, list[str]]
    dates: list[str]
    word_count: int
    reading_minutes: float
    heading_count: int
    page_count: int | None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_names(raw: str) -> list[str]:
    parts = re.split(r",|;|\band\b|&", raw)
    return [" ".join(p.split()) for p in parts if p.strip() and len(p.split()) <= 5]


def _authors_from_text(document: ExtractedDocument) -> list[str]:
    for block in document.blocks[:8]:
        line = block.text.split("\n", 1)[0].strip()
        match = _AUTHOR_LINE_RE.match(line)
        if match and len(line) <= 160:
            return _split_names(match.group("names"))
    return []


def _date_from_text(document: ExtractedDocument) -> str | None:
    """Prefer an explicitly labelled date line, then the first date near the top."""
    for block in document.blocks[:15]:
        if _DATE_LINE_RE.match(block.text):
            found = find_dates(block.text, limit=1)
            if found:
                return found[0].iso
    head = "\n".join(b.text for b in document.blocks[:6])
    found = find_dates(head, limit=1)
    return found[0].iso if found else None


def _fallback_title(document: ExtractedDocument, filename: str | None) -> str | None:
    if document.title:
        return document.title.strip()[:300]
    first = document.blocks[0] if document.blocks else None
    if first is not None and first.kind is not BlockKind.TABLE and len(first.text) <= 120:
        return first.text.strip()
    if filename:
        stem = filename.rsplit(".", 1)[0]
        return re.sub(r"[_\-]+", " ", stem).strip() or None
    return None


class MetadataExtractor:
    def __init__(self, entity_extractor: EntityExtractor | None = None) -> None:
        self.entities = entity_extractor or EntityExtractor()

    def extract(
        self,
        document: ExtractedDocument,
        *,
        filename: str | None = None,
        document_frequency: Callable[[str], int] | None = None,
        total_documents: int = 0,
        keyword_count: int = 10,
    ) -> DocumentMetadata:
        text = document.text
        language = detect_language(text, hint=document.language_hint)
        entity_list = self.entities.extract(text)
        dates = sorted({e.normalized for e in entity_list if e.label == "DATE"})

        document_date = parse_date(document.created) if document.created else None
        document_date = document_date or _date_from_text(document)
        authors = document.authors or _authors_from_text(document)
        word_count = len(words(text))
        keywords = (
            extract_keywords(
                text,
                top_k=keyword_count,
                document_frequency=document_frequency,
                total_documents=total_documents,
            )
            if language.code in ("en", "und")
            # RAKE/TF-IDF rely on English stopwords and stemming.
            else []
        )
        return DocumentMetadata(
            title=_fallback_title(document, filename),
            authors=authors,
            document_date=document_date,
            language=language.code,
            language_confidence=language.confidence,
            keywords=keywords,
            entities=summarize_entities(entity_list),
            dates=dates[:50],
            word_count=word_count,
            reading_minutes=round(word_count / _WORDS_PER_MINUTE, 1),
            heading_count=len(document.headings),
            page_count=document.page_count,
            extra={k: v for k, v in document.properties.items() if v and k != "front_matter"},
        )
