"""Small filesystem helpers for crash-safe persistence."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from idengine.errors import IndexCorruptedError


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object; missing file yields ``{}``, corrupt file raises."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexCorruptedError(f"unreadable file {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise IndexCorruptedError(f"{path.name} does not contain a JSON object")
    return data


def write_json_atomic(path: Path, data: Mapping[str, Any], *, indent: int | None = None) -> None:
    """Write JSON to a temporary sibling, fsync it and rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=indent,
            separators=None if indent else (",", ":"),
            ensure_ascii=False,
        )
        handle.flush()
        os.fsync(handle.fileno())
    replace_file(tmp, path)


def replace_file(
    src: Path | str, dst: Path | str, *, attempts: int = 20, delay: float = 0.05
) -> None:
    """``os.replace`` that tolerates transient sharing violations.

    On Windows, antivirus scanners and search indexers briefly open freshly
    written files, making ``os.replace`` fail with ``PermissionError``
    (WinError 32/5). Retrying for a short time resolves it.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))
