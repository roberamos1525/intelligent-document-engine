from __future__ import annotations

import io
import zipfile

import pytest

from builders import build_docx, build_pdf
from idengine.errors import FileTooLargeError, UnsupportedFileTypeError, ValidationError
from idengine.ingest.detect import decode_text, detect_type, sanitize_filename
from idengine.ingest.model import DocType


def _zip(names: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, "<x/>")
    return buffer.getvalue()


def test_detects_binary_formats_by_magic() -> None:
    assert detect_type(build_pdf([["hello"]])).doc_type is DocType.PDF
    assert detect_type(build_docx([("p", "hello")]), "a.docx").doc_type is DocType.DOCX
    # Declared extension is irrelevant when magic bytes identify the type.
    assert detect_type(build_pdf([["hi"]]), "notes.txt").doc_type is DocType.PDF


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"MZ\x90\x00binary", "executable"),
        (b"\x7fELF\x02\x01", "executable"),
        (b"#!/bin/sh\necho hi", "executable"),
        (b"\x89PNG\r\n\x1a\n\x00\x00", "PNG"),
        (b"\x1f\x8b\x08\x00", "gzip"),
        (b"\x00\x01\x02\x03binary\x00\x00", "binary"),
    ],
)
def test_rejects_dangerous_or_unknown_binaries(payload: bytes, message: str) -> None:
    with pytest.raises(UnsupportedFileTypeError, match=message):
        detect_type(payload, "file.bin")


def test_rejects_non_docx_zip_containers() -> None:
    with pytest.raises(UnsupportedFileTypeError, match="xlsx"):
        detect_type(_zip(["xl/workbook.xml"]), "sheet.xlsx")
    with pytest.raises(UnsupportedFileTypeError, match="zip"):
        detect_type(_zip(["readme.txt"]), "a.zip")


def test_extension_mismatch_for_binary_types() -> None:
    with pytest.raises(UnsupportedFileTypeError, match="does not match"):
        detect_type(b"just some text", "report.pdf")


def test_text_types_use_extension_then_sniffing() -> None:
    html = b"<!DOCTYPE html><html><body><p>x</p></body></html>"
    assert detect_type(html).doc_type is DocType.HTML
    assert detect_type(b"# Title\n\n## Section\ntext", None).doc_type is DocType.MARKDOWN
    assert detect_type(b"plain words", "a.md").doc_type is DocType.MARKDOWN
    assert detect_type(html, "page.txt").doc_type is DocType.TXT
    assert detect_type(b"plain words only").doc_type is DocType.TXT


def test_size_and_emptiness_limits() -> None:
    with pytest.raises(ValidationError, match="empty"):
        detect_type(b"")
    with pytest.raises(FileTooLargeError) as info:
        detect_type(b"x" * 11, max_bytes=10)
    assert info.value.details == {"size": 11, "limit": 10}


def test_allowed_types_are_enforced() -> None:
    with pytest.raises(UnsupportedFileTypeError, match="not enabled"):
        detect_type(b"hello", "a.txt", allowed={"pdf"})


def test_decode_text_handles_boms_and_legacy_encodings() -> None:
    assert decode_text("héllo".encode("utf-8-sig")) == ("héllo", "utf-8-sig")
    assert decode_text("héllo".encode("utf-16")) == ("héllo", "utf-16")
    text, encoding = decode_text("café".encode("cp1252"))
    assert (text, encoding) == ("café", "cp1252")
    assert detect_type("héllo".encode("utf-16"), "a.txt").encoding == "utf-16"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\x\\report.pdf", "report.pdf"),
        ('bad<>:"|?*name.txt', "badname.txt"),
        ("", "untitled"),
        (None, "untitled"),
        ("...", "untitled"),
    ],
)
def test_sanitize_filename(raw: str | None, expected: str) -> None:
    assert sanitize_filename(raw) == expected
