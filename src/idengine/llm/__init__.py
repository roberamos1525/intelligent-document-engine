"""LLM adapters: local extractive model plus optional HTTP backends."""

from __future__ import annotations

import httpx

from idengine.config import LLMSettings
from idengine.llm.base import LLMAdapter, LLMRequest, LLMResponse, Passage, Task
from idengine.llm.extractive import ExtractiveLLM
from idengine.llm.http import OllamaLLM, OpenAICompatibleLLM


def create_llm(
    settings: LLMSettings, *, transport: httpx.BaseTransport | None = None
) -> LLMAdapter:
    """Build the configured adapter (extractive unless explicitly overridden)."""
    if settings.provider == "extractive":
        return ExtractiveLLM()
    assert settings.base_url is not None  # enforced by settings validation
    common = {
        "base_url": settings.base_url,
        "model": settings.model,
        "timeout": settings.timeout_seconds,
        "max_retries": settings.max_retries,
        "transport": transport,
    }
    if settings.provider == "ollama":
        return OllamaLLM(**common)
    api_key = settings.api_key.get_secret_value() if settings.api_key else None
    return OpenAICompatibleLLM(api_key=api_key, **common)


__all__ = [
    "ExtractiveLLM",
    "LLMAdapter",
    "LLMRequest",
    "LLMResponse",
    "OllamaLLM",
    "OpenAICompatibleLLM",
    "Passage",
    "Task",
    "create_llm",
]
