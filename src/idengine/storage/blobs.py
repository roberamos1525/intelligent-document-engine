"""Content-addressed blob store on the local filesystem."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from idengine.errors import NotFoundError, StorageError
from idengine.fsutil import replace_file

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class BlobStore:
    """Stores original uploads under ``<root>/<h[0:2]>/<h[2:4]>/<sha256>``.

    Keys are validated SHA-256 hex digests, so callers can never escape the
    root directory. Writes go to a temporary file first and are renamed into
    place, so a crash never leaves a truncated blob behind.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        if not _HASH_RE.match(digest):
            raise StorageError(f"invalid blob key: {digest!r}")
        return self.root / digest[:2] / digest[2:4] / digest

    def put(self, digest: str, data: bytes) -> Path:
        path = self._path(digest)
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            replace_file(tmp, path)
        except OSError as exc:
            Path(tmp).unlink(missing_ok=True)
            raise StorageError(f"failed to write blob: {exc}") from exc
        return path

    def get(self, digest: str) -> bytes:
        path = self._path(digest)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise NotFoundError(f"blob {digest} not found") from exc

    def exists(self, digest: str) -> bool:
        return self._path(digest).exists()

    def delete(self, digest: str) -> bool:
        path = self._path(digest)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
