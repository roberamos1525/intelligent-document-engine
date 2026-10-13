"""Shared fixtures."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from idengine.config import Settings, load_settings
from idengine.engine import Engine

ADMIN_KEY = "admin-key-123"
WRITER_KEY = "writer-key-123"
READER_KEY = "reader-key-123"
GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


def make_settings(data_dir: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "storage": {"data_dir": str(data_dir)},
        "jobs": {"workers": 0, "backoff_base_seconds": 0.0, "backoff_max_seconds": 0.0},
        "api": {
            "api_keys": [f"{ADMIN_KEY}:admin", f"{WRITER_KEY}:writer", f"{READER_KEY}:reader"],
            "rate_limit_per_minute": 0,
        },
    }
    for section, values in overrides.items():
        base.setdefault(section, {}).update(values)
    return load_settings(environ={}, overrides=base)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path / "data")


@pytest.fixture
def engine(settings: Settings) -> Iterator[Engine]:
    eng = Engine(settings)
    try:
        yield eng
    finally:
        eng.close()
