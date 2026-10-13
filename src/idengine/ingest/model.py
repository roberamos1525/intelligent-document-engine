"""Format-independent representation of an extracted document."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DocType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"

    @property
    def mime(self) -> str:
        return {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "txt": "text/plain",
            "markdown": "text/markdown",
            "html": "text/html",
        }[self.value]


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"


@dataclass(slots=True)
class Block:
    """A structural unit of text (heading, paragraph, list item, ...)."""

    kind: BlockKind
    text: str
    level: int = 0
    page: int | None = None

    @property
    def is_heading(self) -> bool:
        return self.kind is BlockKind.HEADING


@dataclass(slots=True)
class Section:
    """A run of blocks under one heading path, e.g. ("Contract", "Payment terms")."""

    heading_path: tuple[str, ...]
    blocks: list[Block] = field(default_factory=list)

    @property
    def title(self) -> str | None:
        return self.heading_path[-1] if self.heading_path else None


@dataclass(slots=True)
class ExtractedDocument:
    """Output of every extractor: ordered blocks plus native metadata."""

    doc_type: DocType
    blocks: list[Block]
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    created: str | None = None
    language_hint: str | None = None
    page_count: int | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text)

    @property
    def headings(self) -> list[Block]:
        return [b for b in self.blocks if b.is_heading]

    def sections(self) -> list[Section]:
        """Group blocks into sections using a heading-level stack."""
        sections: list[Section] = []
        stack: list[tuple[int, str]] = []
        current = Section(heading_path=())
        for block in self.blocks:
            if block.is_heading:
                if current.blocks or current.heading_path:
                    sections.append(current)
                level = max(block.level, 1)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, block.text))
                current = Section(heading_path=tuple(text for _, text in stack))
            else:
                current.blocks.append(block)
        if current.blocks or current.heading_path:
            sections.append(current)
        return [s for s in sections if s.blocks]
