"""ASGI-level tests for the middleware (no FastAPI app involved)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.types import Message, Receive, Scope, Send

from idengine.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware, error_body
from idengine.observability.logging import request_id_var
from idengine.observability.metrics import EngineMetrics


class Route:
    path = "/items/{item_id}"


async def echo_app(scope: Scope, receive: Receive, send: Send) -> None:
    """Reads the whole body and echoes its length plus the active request id."""
    size = 0
    while True:
        message = await receive()
        size += len(message.get("body", b""))
        if not message.get("more_body"):
            break
    scope["route"] = Route()
    body = json.dumps({"size": size, "request_id": request_id_var.get()}).encode()
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": body})


def call(
    app: Any, *, body_chunks: list[bytes], headers: list[tuple[bytes, bytes]] | None = None
) -> tuple[int, dict[str, str], bytes]:
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/items/1",
        "headers": headers or [],
    }
    incoming = [
        {"type": "http.request", "body": chunk, "more_body": i < len(body_chunks) - 1}
        for i, chunk in enumerate(body_chunks)
    ]
    sent: list[Message] = []

    async def receive() -> Message:
        return incoming.pop(0) if incoming else {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    response_headers = {k.decode(): v.decode() for k, v in start.get("headers", [])}
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], response_headers, body


def test_body_limit_by_content_length() -> None:
    app = BodySizeLimitMiddleware(echo_app, max_bytes=10)
    status, _, body = call(app, body_chunks=[b"x" * 20], headers=[(b"content-length", b"20")])
    assert status == 413
    assert json.loads(body)["error"]["code"] == "request_too_large"


def test_body_limit_for_streamed_chunks_without_length() -> None:
    app = BodySizeLimitMiddleware(echo_app, max_bytes=10)
    status, _, body = call(app, body_chunks=[b"x" * 6, b"y" * 6])
    assert status == 413 and json.loads(body)["error"]["details"] == {"limit": 10}
    ok_status, _, ok_body = call(app, body_chunks=[b"x" * 4, b"y" * 4])
    assert ok_status == 200 and json.loads(ok_body)["size"] == 8


def test_invalid_content_length_falls_back_to_streaming_check() -> None:
    app = BodySizeLimitMiddleware(echo_app, max_bytes=10)
    status, _, _ = call(app, body_chunks=[b"abc"], headers=[(b"content-length", b"abc")])
    assert status == 200


def test_request_context_sets_ids_and_records_metrics() -> None:
    metrics = EngineMetrics()
    app = RequestContextMiddleware(echo_app, metrics)
    status, headers, body = call(app, body_chunks=[b"hi"], headers=[(b"x-request-id", b"abc-1")])
    assert status == 200
    assert headers["x-request-id"] == "abc-1"
    assert json.loads(body)["request_id"] == "abc-1"
    labels = {"method": "POST", "route": "/items/{item_id}", "status": "200"}
    assert metrics.http_requests.value(labels) == 1
    assert metrics.http_latency.count({"method": "POST", "route": "/items/{item_id}"}) == 1
    # Invalid ids are replaced by a generated one; the context var is reset afterwards.
    _, headers, _ = call(app, body_chunks=[b""], headers=[(b"x-request-id", b"no spaces allowed")])
    assert len(headers["x-request-id"]) == 32
    assert request_id_var.get() is None


def test_non_http_scopes_pass_through() -> None:
    seen: list[str] = []

    async def lifespan_app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    for middleware in (
        BodySizeLimitMiddleware(lifespan_app, max_bytes=1),
        RequestContextMiddleware(lifespan_app, EngineMetrics()),
    ):
        asyncio.run(middleware({"type": "lifespan"}, None, None))  # type: ignore[arg-type]
    assert seen == ["lifespan", "lifespan"]


def test_error_body_shape() -> None:
    payload = json.loads(error_body("x", "msg", "rid", limit=5))
    assert payload == {
        "error": {"code": "x", "message": "msg", "request_id": "rid", "details": {"limit": 5}}
    }
    assert "details" not in json.loads(error_body("x", "msg", None))["error"]
