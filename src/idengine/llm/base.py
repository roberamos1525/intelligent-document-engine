"""LLM adapter interface and request/response types."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Task(StrEnum):
    ANSWER = "answer"
    SUMMARIZE = "summarize"


@dataclass(frozen=True, slots=True)
class Passage:
    """A numbered context passage; ``number`` is what the model cites as ``[n]``."""

    number: int
    text: str
    source: str
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class LLMRequest:
    task: Task
    instruction: str
    passages: list[Passage]
    max_output_tokens: int = 512
    temperature: float = 0.0
    max_sentences: int = 3


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    citations: list[int] = field(default_factory=list)
    short_answer: str | None = None
    answerable: bool = True
    usage: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


_CITATION_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


def parse_citations(text: str, valid: set[int]) -> list[int]:
    """Extract ``[n]`` / ``[n, m]`` markers that refer to supplied passages, in order."""
    found: list[int] = []
    for match in _CITATION_RE.finditer(text):
        for part in match.group(1).split(","):
            number = int(part.strip())
            if number in valid and number not in found:
                found.append(number)
    return found


class LLMAdapter(ABC):
    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Produce an answer or summary grounded in ``request.passages``."""

    def health(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "ok": True}

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""
