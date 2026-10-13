"""Exception handlers producing the consistent error envelope.

Every error response has the shape::

    {"error": {"code": "...", "message": "...", "details": {...}, "request_id": "..."}}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from idengine.errors import EngineError, RateLimitError
from idengine.observability.logging import get_logger, request_id_var

log = get_logger("api.errors")

_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "request_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
}


def envelope(
    status: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {"code": code, "message": message, "request_id": request_id_var.get()}
    if details:
        body["details"] = details
    return JSONResponse({"error": body}, status_code=status, headers=headers)


async def _engine_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, EngineError)
    headers = None
    if isinstance(exc, RateLimitError):
        headers = {"Retry-After": str(max(1, round(exc.retry_after)))}
    if exc.http_status >= 500:
        log.error("engine error", extra={"code": exc.code, "error": exc.message})
    return envelope(exc.http_status, exc.code, exc.message, exc.details, headers)


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    problems = [
        {
            "loc": [str(p) for p in err.get("loc", ())],
            "msg": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    return envelope(422, "validation_error", "request validation failed", {"errors": problems})


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _STATUS_CODES.get(exc.status_code, "http_error")
    return envelope(exc.status_code, code, str(exc.detail), headers=exc.headers)


async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error", exc_info=exc)
    return envelope(500, "internal_error", "an unexpected error occurred")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(EngineError, _engine_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled)
