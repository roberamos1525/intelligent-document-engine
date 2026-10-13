"""ASGI middleware: request ids, access logging, metrics and request size limits."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from idengine.observability.logging import get_logger, request_id_var
from idengine.observability.metrics import EngineMetrics

log = get_logger("api.access")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


def error_body(code: str, message: str, request_id: str | None, **details: Any) -> bytes:
    error: dict[str, Any] = {"code": code, "message": message, "request_id": request_id}
    if details:
        error["details"] = details
    return json.dumps({"error": error}).encode("utf-8")


class RequestContextMiddleware:
    """Assigns a request id, records latency/status metrics and logs each request."""

    def __init__(self, app: ASGIApp, metrics: EngineMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [
                    *message["headers"],
                    (b"x-request-id", request_id.encode("latin-1")),
                ]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - started
            route = scope.get("route")
            template = getattr(route, "path", None) or "unmatched"
            method = scope.get("method", "GET")
            self.metrics.http_requests.inc(
                labels={"method": method, "route": template, "status": str(status)}
            )
            self.metrics.http_latency.observe(elapsed, {"method": method, "route": template})
            # The auth dependency stores the principal in request.state (backed by scope["state"]);
            # only its non-reversible key fingerprint is logged.
            principal = (scope.get("state") or {}).get("principal")
            log.info(
                "request",
                extra={
                    "method": method,
                    "path": scope.get("path"),
                    "route": template,
                    "status": status,
                    "key_id": getattr(principal, "key_id", None),
                    "duration_ms": round(elapsed * 1000, 2),
                },
            )
            request_id_var.reset(token)


class BodySizeLimitMiddleware:
    """Rejects requests whose body exceeds ``max_bytes`` with a 413 envelope.

    Checks ``Content-Length`` up front and also counts streamed bytes, so
    chunked uploads without a length header cannot bypass the limit.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, send: Send) -> None:
        body = error_body(
            "request_too_large",
            f"request body exceeds {self.max_bytes} bytes",
            request_id_var.get(),
            limit=self.max_bytes,
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                too_large = int(declared) > self.max_bytes
            except ValueError:
                too_large = False
            if too_large:
                await self._reject(send)
                return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLargeError:
            if not response_started:
                await self._reject(send)


class _BodyTooLargeError(Exception):
    pass
