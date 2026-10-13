"""HTML extractor built on the standard library ``html.parser``.

Boilerplate containers (scripts, styles, navigation, footers, forms) are
dropped; headings, paragraphs, list items, preformatted text and tables are
turned into structural blocks.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from idengine.errors import ExtractionError
from idengine.ingest.detect import decode_text
from idengine.ingest.model import Block, BlockKind, DocType, ExtractedDocument
from idengine.text.normalize import clean_text

_SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "template", "svg", "nav", "footer", "form", "iframe", "head"}
)
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "section", "article", "main", "header", "aside", "blockquote", "br",
        "dd", "dt", "figcaption", "address", "hr", "body", "html",
    }
)  # fmt: skip
_HEADINGS = {f"h{i}": i for i in range(1, 7)}
_VOID_TAGS = frozenset({"br", "hr", "img", "meta", "link", "input", "wbr", "col", "source"})
_DATE_META = ("date", "dc.date", "article:published_time", "dcterms.created", "pubdate")


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self.lang: str | None = None
        self._buffer: list[str] = []
        self._kind = BlockKind.PARAGRAPH
        self._level = 0
        self._skip_depth = 0
        self._in_title = False
        self._in_pre = False
        self._table_rows: list[list[str]] | None = None
        self._cell: list[str] | None = None

    # -- helpers -----------------------------------------------------------
    def _flush(self) -> None:
        raw = "".join(self._buffer)
        self._buffer = []
        text = raw.strip("\n") if self._kind is BlockKind.CODE else " ".join(raw.split())
        if text:
            self.blocks.append(Block(self._kind, text, level=self._level))
        self._kind = BlockKind.PARAGRAPH
        self._level = 0

    # -- parser callbacks ----------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {k.lower(): (v or "") for k, v in attrs}
        if tag == "html" and attributes.get("lang"):
            self.lang = attributes["lang"].split("-")[0].lower()
        if tag == "meta":
            name = (attributes.get("name") or attributes.get("property") or "").lower()
            if name and attributes.get("content"):
                self.meta[name] = attributes["content"].strip()
            return
        if tag == "title":
            self._in_title = True
            return
        if self._skip_depth or tag in _SKIP_TAGS:
            if tag not in _VOID_TAGS:
                self._skip_depth += 1
            return
        if tag == "table":
            self._flush()
            self._table_rows = []
        elif tag == "tr" and self._table_rows is not None:
            self._table_rows.append([])
        elif tag in ("td", "th") and self._table_rows is not None:
            self._cell = []
        elif tag in _HEADINGS:
            self._flush()
            self._kind = BlockKind.HEADING
            self._level = _HEADINGS[tag]
        elif tag == "li":
            self._flush()
            self._kind = BlockKind.LIST_ITEM
        elif tag == "pre":
            self._flush()
            self._kind = BlockKind.CODE
            self._in_pre = True
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if self._skip_depth:
            if tag not in _VOID_TAGS:
                self._skip_depth -= 1
            return
        if tag in ("td", "th") and self._cell is not None and self._table_rows:
            self._table_rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "table" and self._table_rows is not None:
            rows = [" | ".join(c for c in row) for row in self._table_rows if any(row)]
            if rows:
                self.blocks.append(Block(BlockKind.TABLE, "\n".join(rows)))
            self._table_rows = None
        elif tag == "pre":
            self._flush()
            self._in_pre = False
        elif tag in _HEADINGS or tag == "li" or tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._skip_depth:
            return
        if self._cell is not None:
            self._cell.append(data)
        elif self._table_rows is None:
            self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def extract_html(data: bytes) -> ExtractedDocument:
    raw, encoding = decode_text(data)
    parser = _DocumentParser()
    try:
        parser.feed(raw)
        parser.close()
    except (AssertionError, ValueError) as exc:  # pragma: no cover - parser is lenient
        raise ExtractionError(f"malformed HTML: {exc}") from exc

    blocks: list[Block] = []
    for block in parser.blocks:
        text = block.text if block.kind is BlockKind.CODE else clean_text(block.text)
        if text:
            block.text = text
            blocks.append(block)
    if not blocks:
        raise ExtractionError("document contains no text")

    meta = parser.meta
    title = " ".join("".join(parser.title_parts).split()) or meta.get("og:title")
    if not title:
        title = next((b.text for b in blocks if b.is_heading and b.level == 1), None)
    authors = [a.strip() for a in re.split(r",|;", meta.get("author", "")) if a.strip()]
    created = next((meta[k] for k in _DATE_META if meta.get(k)), None)
    return ExtractedDocument(
        doc_type=DocType.HTML,
        blocks=blocks,
        title=title or None,
        authors=authors,
        created=created,
        language_hint=parser.lang,
        properties={
            "encoding": encoding,
            "description": meta.get("description"),
            "keywords": meta.get("keywords"),
        },
    )
