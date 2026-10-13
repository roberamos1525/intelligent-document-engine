"""File type detection from magic bytes, with extension cross-checks.

The declared filename and content type are treated as untrusted hints: the
actual bytes decide. Executables and archives that are not DOCX are rejected
outright, and a mismatch between a binary extension and the content (for
example ``report.pdf`` that is really a ZIP) is an error.
"""

from __future__ import annotations

import codecs
import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePath

from idengine.errors import FileTooLargeError, UnsupportedFileTypeError, ValidationError
from idengine.ingest.model import DocType

_EXECUTABLE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"MZ", "Windows executable"),
    (b"\x7fELF", "ELF executable"),
    (b"\xca\xfe\xba\xbe", "Mach-O/Java class"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable"),
    (b"#!", "script"),
)

_OTHER_BINARY_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"GIF87a", "GIF image"),
    (b"GIF89a", "GIF image"),
    (b"\x1f\x8b", "gzip archive"),
    (b"Rar!", "RAR archive"),
    (b"7z\xbc\xaf\x27\x1c", "7z archive"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "legacy OLE2 (doc/xls)"),
    (b"{\\rtf", "RTF document"),
)

_EXTENSIONS: dict[str, DocType] = {
    ".pdf": DocType.PDF,
    ".docx": DocType.DOCX,
    ".txt": DocType.TXT,
    ".text": DocType.TXT,
    ".log": DocType.TXT,
    ".csv": DocType.TXT,
    ".md": DocType.MARKDOWN,
    ".markdown": DocType.MARKDOWN,
    ".html": DocType.HTML,
    ".htm": DocType.HTML,
    ".xhtml": DocType.HTML,
}

_MAX_FILENAME_CHARS = 255


@dataclass(frozen=True, slots=True)
class DetectedType:
    doc_type: DocType
    encoding: str | None = None

    @property
    def mime(self) -> str:
        return self.doc_type.mime


def sanitize_filename(filename: str | None) -> str:
    """Strip directories and control characters from a client-supplied filename."""
    if not filename:
        return "untitled"
    name = PurePath(filename.replace("\\", "/")).name
    name = "".join(ch for ch in name if ch.isprintable() and ch not in '<>:"|?*')
    name = name.strip(" .")
    return name[:_MAX_FILENAME_CHARS] or "untitled"


def extension_type(filename: str | None) -> DocType | None:
    if not filename:
        return None
    return _EXTENSIONS.get(PurePath(filename).suffix.lower())


def decode_text(data: bytes) -> tuple[str, str]:
    """Decode text bytes, honouring BOMs; returns ``(text, encoding)``.

    Falls back from UTF-8 to cp1252 (a superset of Latin-1 for printable
    characters) when the data is not valid UTF-8.
    """
    for bom, encoding in (
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    ):
        if data.startswith(bom):
            return data.decode(encoding), encoding
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace"), "cp1252"


def _looks_binary(data: bytes) -> bool:
    sample = data[:8192]
    if sample.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return False
    if b"\x00" in sample:
        return True
    control = sum(1 for b in sample if b < 32 and b not in (9, 10, 12, 13))
    return control / max(len(sample), 1) > 0.05


def _sniff_html(text: str) -> bool:
    head = text[:2048].lstrip().lower()
    if head.startswith(("<!doctype html", "<html")):
        return True
    return "<body" in head or ("<head" in head and "</" in head)


def _sniff_markdown(text: str) -> bool:
    lines = text[:4096].splitlines()
    if lines and lines[0].strip() == "---":
        return True
    signals = sum(
        1
        for line in lines
        if line.startswith(("# ", "## ", "### ", "```", "> ")) or line.startswith("|")
    )
    return signals >= 2


def _zip_kind(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile as exc:
        raise UnsupportedFileTypeError("corrupt ZIP container") from exc
    if "word/document.xml" in names:
        return "docx"
    if "xl/workbook.xml" in names:
        return "xlsx"
    if "ppt/presentation.xml" in names:
        return "pptx"
    return "zip"


def detect_type(
    data: bytes,
    filename: str | None = None,
    *,
    max_bytes: int | None = None,
    allowed: set[str] | None = None,
) -> DetectedType:
    """Identify the document type of ``data`` or raise a typed validation error."""
    if not data:
        raise ValidationError("file is empty")
    if max_bytes is not None and len(data) > max_bytes:
        raise FileTooLargeError(
            f"file is {len(data)} bytes; limit is {max_bytes}",
            details={"size": len(data), "limit": max_bytes},
        )

    declared = extension_type(filename)
    detected = _detect(data, declared)
    if declared in (DocType.PDF, DocType.DOCX) and detected.doc_type is not declared:
        raise UnsupportedFileTypeError(
            f"content does not match extension: expected {declared.value}, "
            f"found {detected.doc_type.value}",
            details={"declared": declared.value, "detected": detected.doc_type.value},
        )
    if allowed is not None and detected.doc_type.value not in allowed:
        raise UnsupportedFileTypeError(
            f"document type '{detected.doc_type.value}' is not enabled",
            details={"detected": detected.doc_type.value, "allowed": sorted(allowed)},
        )
    return detected


def _detect(data: bytes, declared: DocType | None) -> DetectedType:
    for magic, label in _EXECUTABLE_MAGIC:
        if data.startswith(magic):
            raise UnsupportedFileTypeError(f"executable content rejected ({label})")
    if b"%PDF-" in data[:1024]:
        return DetectedType(DocType.PDF)
    if data.startswith(b"PK\x03\x04"):
        kind = _zip_kind(data)
        if kind == "docx":
            return DetectedType(DocType.DOCX)
        raise UnsupportedFileTypeError(f"unsupported ZIP-based format ({kind})")
    for magic, label in _OTHER_BINARY_MAGIC:
        if data.startswith(magic):
            raise UnsupportedFileTypeError(f"unsupported binary format ({label})")
    if _looks_binary(data):
        raise UnsupportedFileTypeError("unrecognised binary content")

    text, encoding = decode_text(data)
    if declared is DocType.HTML or (declared is None and _sniff_html(text)):
        return DetectedType(DocType.HTML, encoding)
    if declared is DocType.MARKDOWN or (declared is None and _sniff_markdown(text)):
        return DetectedType(DocType.MARKDOWN, encoding)
    return DetectedType(DocType.TXT, encoding)
