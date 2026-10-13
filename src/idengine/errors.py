"""Typed error hierarchy shared by the library, CLI and HTTP API.

Every error carries a stable machine-readable ``code`` and the HTTP status it
maps to, so the API layer can render a consistent error envelope without
knowing about individual subsystems.
"""

from __future__ import annotations

from typing import Any


class EngineError(Exception):
    """Base class for all expected (handled) engine errors."""

    code: str = "engine_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class ConfigurationError(EngineError):
    code = "configuration_error"
    http_status = 500


class ValidationError(EngineError):
    code = "validation_error"
    http_status = 422


class UnsupportedFileTypeError(ValidationError):
    code = "unsupported_file_type"
    http_status = 415


class FileTooLargeError(ValidationError):
    code = "file_too_large"
    http_status = 413


class ExtractionError(EngineError):
    """The file looked valid but its content could not be extracted."""

    code = "extraction_failed"
    http_status = 422


class NotFoundError(EngineError):
    code = "not_found"
    http_status = 404


class ConflictError(EngineError):
    code = "conflict"
    http_status = 409


class AuthenticationError(EngineError):
    code = "unauthorized"
    http_status = 401


class AuthorizationError(EngineError):
    code = "forbidden"
    http_status = 403


class RateLimitError(EngineError):
    code = "rate_limited"
    http_status = 429

    def __init__(self, message: str, *, retry_after: float) -> None:
        super().__init__(message, details={"retry_after_seconds": round(retry_after, 3)})
        self.retry_after = retry_after


class ModelNotTrainedError(EngineError):
    code = "model_not_trained"
    http_status = 409


class LLMError(EngineError):
    """An LLM adapter failed (network error, bad response, timeout)."""

    code = "llm_error"
    http_status = 502


class StorageError(EngineError):
    code = "storage_error"
    http_status = 500


class IndexCorruptedError(StorageError):
    code = "index_corrupted"
