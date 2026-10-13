"""Optional HTTP adapters: Ollama and OpenAI-compatible chat completion APIs.

Both are disabled by default (``llm.provider = "extractive"``). They are only
constructed when explicitly configured, use bounded timeouts and retry
transient failures (connection errors, 429 and 5xx) with exponential backoff.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

from idengine.errors import LLMError
from idengine.llm.base import LLMAdapter, LLMRequest, LLMResponse, parse_citations
from idengine.llm.prompts import NOT_FOUND_ANSWER, render_messages
from idengine.observability.logging import get_logger

log = get_logger("llm.http")

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class _HttpAdapter(LLMAdapter):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            headers={"Content-Type": "application/json", **(headers or {})},
            transport=transport,
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(path, json=payload)
            except httpx.TimeoutException:
                last_error = "request timed out"
            except httpx.TransportError as exc:
                last_error = f"connection failed: {exc}"
            else:
                if response.status_code < 400:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise LLMError(f"{self.name} returned invalid JSON") from exc
                    if not isinstance(data, dict):
                        raise LLMError(f"{self.name} returned an unexpected payload")
                    return data
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise LLMError(f"{self.name} request failed ({last_error})")
            if attempt < self.max_retries:
                delay = self.backoff_seconds * (2**attempt)
                log.warning(
                    "llm request failed; retrying",
                    extra={"provider": self.name, "attempt": attempt + 1, "error": last_error},
                )
                self._sleep(delay)
        raise LLMError(
            f"{self.name} request failed after {self.max_retries + 1} attempts ({last_error})"
        )

    def _to_response(self, text: str, request: LLMRequest, usage: dict[str, int]) -> LLMResponse:
        text = text.strip()
        valid = {p.number for p in request.passages}
        citations = parse_citations(text, valid)
        answerable = NOT_FOUND_ANSWER.lower().rstrip(".") not in text.lower()
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            citations=citations,
            answerable=answerable and bool(text),
            usage=usage,
        )

    def close(self) -> None:
        self._client.close()


class OllamaLLM(_HttpAdapter):
    """Adapter for a local Ollama server (``POST /api/chat``)."""

    name = "ollama"

    def generate(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": render_messages(request),
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        data = self._post("/api/chat", payload)
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMError("ollama response has no message content")
        usage = {
            "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
            "completion_tokens": int(data.get("eval_count", 0) or 0),
        }
        return self._to_response(message["content"], request, usage)

    def health(self) -> dict[str, Any]:
        try:
            response = self._client.get("/api/tags")
            ok = response.status_code == 200
        except httpx.HTTPError:
            ok = False
        return {"provider": self.name, "model": self.model, "ok": ok}


class OpenAICompatibleLLM(_HttpAdapter):
    """Adapter for any server implementing ``POST /v1/chat/completions``."""

    name = "openai"

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        super().__init__(headers=headers, **kwargs)
        self._path = (
            "/chat/completions" if self.base_url.endswith("/v1") else "/v1/chat/completions"
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": render_messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        data = self._post(self._path, payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("chat completion response has no message content") from exc
        if not isinstance(content, str):
            raise LLMError("chat completion content is not text")
        raw_usage = data.get("usage") or {}
        usage = {
            "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
        }
        return self._to_response(content, request, usage)
