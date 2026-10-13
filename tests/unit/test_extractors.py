from __future__ import annotations

import io
import zipfile

import pytest

from builders import GUIDE_HTML, INVOICE_TEXT, POLICY_MARKDOWN, build_docx, build_pdf
from idengine.config import IngestSettings
from idengine.errors import ExtractionError, ValidationError
from idengine.ingest.detect import detect_type
from idengine.ingest.extractors import extract
from idengine.ingest.extractors.docx import check_archive_limits, extract_docx
from idengine.ingest.extractors.html import extract_html
from idengine.ingest.extractors.markdown import extract_markdown, parse_front_matter, strip_inline
from idengine.ingest.extractors.pdf import extract_pdf
from idengine.ingest.extractors.plaintext import extract_plaintext
from idengine.ingest.extractors.structure import blocks_from_plain_text, heading_level
from idengine.ingest.model import BlockKind, DocType


def kinds(doc) -> list[tuple[str, str]]:  # type: ignore[no-untyped-def]
    return [(b.kind.value, b.text) for b in doc.blocks]


class TestPdf:
    def test_pages_headings_and_metadata(self) -> None:
        data = build_pdf(
            [
                ["1. Introduction", "This report covers the first quarter.", "Revenue grew."],
                ["2. Costs", "Costs were flat (as planned)."],
            ],
            title="Q1 Report",
            author="Ann Lee; Bob Kim",
            created="20240305120000Z",
        )
        doc = extract_pdf(data)
        assert doc.page_count == 2
        assert doc.title == "Q1 Report"
        assert doc.authors == ["Ann Lee", "Bob Kim"]
        assert doc.created == "2024-03-05"
        assert [b.page for b in doc.blocks] == [1, 1, 2, 2]
        assert doc.blocks[0].kind is BlockKind.HEADING
        assert doc.blocks[1].text == "This report covers the first quarter. Revenue grew."
        assert doc.sections()[1].heading_path == ("2. Costs",)

    def test_page_limit(self) -> None:
        data = build_pdf([["a"], ["b"], ["c"]])
        with pytest.raises(ValidationError, match="3 pages"):
            extract_pdf(data, max_pages=2)

    def test_pdf_without_text_is_an_extraction_error(self) -> None:
        with pytest.raises(ExtractionError, match="no extractable text"):
            extract_pdf(build_pdf([[]]))

    def test_corrupt_pdf(self) -> None:
        with pytest.raises(ExtractionError):
            extract_pdf(b"%PDF-1.4\nthis is not really a pdf")

    def test_title_falls_back_to_first_heading(self) -> None:
        doc = extract_pdf(build_pdf([["QUARTERLY SUMMARY", "Sales rose in every region."]]))
        assert doc.title == "QUARTERLY SUMMARY"


class TestDocx:
    def test_headings_lists_tables_and_properties(self) -> None:
        data = build_docx(
            [
                ("title", "Supplier Agreement"),
                ("h1", "Scope"),
                ("p", "The supplier delivers parts monthly."),
                ("bullet", "Delivery within 10 days"),
                ("h2", "Pricing"),
                ("p", "Prices are fixed for one year."),
            ],
            author="Legal Team",
            table=[["Item", "Price"], ["Bolt", "$1.00"]],
        )
        doc = extract_docx(data)
        assert doc.title == "Supplier Agreement"
        assert doc.authors == ["Legal Team"]
        assert ("heading", "Scope") in kinds(doc)
        assert ("list_item", "Delivery within 10 days") in kinds(doc)
        assert doc.blocks[-1].kind is BlockKind.TABLE
        assert doc.blocks[-1].text == "Item | Price\nBolt | $1.00"
        # "Title" and "Heading 1" share level 1, so Scope starts a new top-level path.
        assert [s.heading_path for s in doc.sections()] == [("Scope",), ("Scope", "Pricing")]

    def test_core_property_title_wins(self) -> None:
        doc = extract_docx(build_docx([("p", "Body text.")], title="From Properties"))
        assert doc.title == "From Properties"

    def test_zip_bomb_protection(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", "A" * (3 * 1024 * 1024))
        data = buffer.getvalue()
        with pytest.raises(ValidationError, match="compression ratio"):
            check_archive_limits(data, max_uncompressed=10**9, max_ratio=50)
        with pytest.raises(ValidationError, match="expands to"):
            check_archive_limits(data, max_uncompressed=1024 * 1024, max_ratio=10**6)

    def test_invalid_docx(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", "<broken")
        with pytest.raises(ExtractionError):
            extract_docx(buffer.getvalue())


class TestHtml:
    def test_structure_metadata_and_boilerplate_removal(self) -> None:
        doc = extract_html(GUIDE_HTML.encode())
        text = doc.text
        assert doc.title == "Kubernetes Deployment Guide"
        assert doc.authors == ["Platform Team"]
        assert doc.language_hint == "en"
        assert doc.properties["description"] == "How to deploy"
        assert "tracking" not in text and "Login" not in text and "Copyright" not in text
        assert ("heading", "Scaling") in kinds(doc)
        assert ("table", "Env | Replicas\nprod | 6") in kinds(doc)

    def test_entities_and_preformatted_text(self) -> None:
        html = (
            b"<html><body><p>Fish &amp; chips &lt;3</p><pre>  indented\n  code</pre></body></html>"
        )
        doc = extract_html(html)
        assert doc.blocks[0].text == "Fish & chips <3"
        assert doc.blocks[1].kind is BlockKind.CODE
        assert doc.blocks[1].text == "  indented\n  code"

    def test_empty_html(self) -> None:
        with pytest.raises(ExtractionError):
            extract_html(
                b"<html><head><title>x</title></head><body><script>1</script></body></html>"
            )


class TestMarkdown:
    def test_front_matter_headings_and_metadata(self) -> None:
        doc = extract_markdown(POLICY_MARKDOWN.encode())
        assert doc.title == "Remote Work Policy"
        assert doc.authors == ["Jane Doe", "John Roe"]
        assert doc.created == "2023-11-01"
        assert doc.sections()[0].heading_path == ("Remote Work Policy", "Eligibility")
        # Hard-wrapped paragraph lines are joined.
        assert any(b.text.startswith("Employees who have completed") for b in doc.blocks)

    def test_code_tables_lists_and_inline_markup(self) -> None:
        source = (
            "Intro with **bold**, *italic*, `code` and [a link](http://x).\n\n"
            "```python\nprint('hi')\n```\n\n"
            "| Name | Value |\n|---|---|\n| a | 1 |\n\n"
            "- [x] done item\n1. first\n\n"
            "Setext Title\n============\n\n> quoted text\n"
        )
        doc = extract_markdown(source.encode())
        result = kinds(doc)
        assert ("paragraph", "Intro with bold, italic, code and a link.") in result
        assert ("code", "print('hi')") in result
        assert ("table", "Name | Value\na | 1") in result
        assert ("list_item", "done item") in result
        assert ("list_item", "first") in result
        assert ("heading", "Setext Title") in result
        assert ("paragraph", "quoted text") in result

    def test_front_matter_parsing_variants(self) -> None:
        meta, body = parse_front_matter("---\ntitle: 'X'\nauthors:\n  - A\n  - B\n---\nbody")
        assert meta == {"title": "X", "authors": ["A", "B"]}
        assert body == "body"
        assert parse_front_matter("---\nunterminated: yes\n") == ({}, "---\nunterminated: yes\n")

    def test_strip_inline(self) -> None:
        assert strip_inline("![logo](x.png) ~~old~~ <b>new</b>") == "logo old new"


class TestPlaintextAndStructure:
    def test_plaintext_title_and_sections(self) -> None:
        doc = extract_plaintext(INVOICE_TEXT.encode())
        assert doc.title == "ACME WIDGETS INVOICE"
        headings = [b.text for b in doc.headings]
        assert headings == ["ACME WIDGETS INVOICE", "Payment Terms", "Line Items"]

    @pytest.mark.parametrize(
        ("line", "level"),
        [
            ("1. Introduction", 1),
            ("2.3 Scope of Work", 2),
            ("GENERAL TERMS", 1),
            ("Payment Terms", 2),
            ("This is a normal sentence.", None),
            ("Short line ending with colon:", None),
            ("x" * 120, None),
        ],
    )
    def test_heading_heuristics(self, line: str, level: int | None) -> None:
        assert heading_level(line, isolated=True) == level

    def test_setext_and_list_items(self) -> None:
        blocks = blocks_from_plain_text("Title\n=====\n\n- one\n- two\n\nBody text here.")
        assert [(b.kind.value, b.level) for b in blocks] == [
            ("heading", 1),
            ("list_item", 0),
            ("list_item", 0),
            ("paragraph", 0),
        ]

    def test_empty_text_fails(self) -> None:
        with pytest.raises(ExtractionError):
            extract_plaintext(b"   \n\n  ")


def test_dispatch_uses_detected_type() -> None:
    settings = IngestSettings()
    for data, name, expected in [
        (build_pdf([["Hello world"]]), "a.pdf", DocType.PDF),
        (build_docx([("p", "Hello world")]), "a.docx", DocType.DOCX),
        (GUIDE_HTML.encode(), "a.html", DocType.HTML),
        (POLICY_MARKDOWN.encode(), "a.md", DocType.MARKDOWN),
        (INVOICE_TEXT.encode(), "a.txt", DocType.TXT),
    ]:
        assert extract(data, detect_type(data, name), settings).doc_type is expected
