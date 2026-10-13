from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from builders import GUIDE_HTML, INVOICE_TEXT, POLICY_MARKDOWN, build_pdf
from conftest import ADMIN_KEY, GOLDEN_DIR, READER_KEY, WRITER_KEY, make_settings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from idengine.api.app import create_app
from idengine.config import Settings
from idengine.engine import Engine
from idengine.errors import ConfigurationError

ADMIN = {"X-API-Key": ADMIN_KEY}
WRITER = {"Authorization": f"Bearer {WRITER_KEY}"}
READER = {"X-API-Key": READER_KEY}


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path / "data", jobs={"workers": 1, "poll_interval_seconds": 0.02})


@pytest.fixture
def client(api_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(api_settings)) as test_client:
        yield test_client


def upload(
    client: TestClient,
    name: str,
    data: bytes,
    *,
    headers: dict[str, str] = WRITER,
    wait: bool = True,
    **form: str,
) -> Any:
    return client.post(
        f"/v1/documents?wait={str(wait).lower()}",
        headers=headers,
        files={"file": (name, data, "application/octet-stream")},
        data=form,
    )


def error_code(response: Any) -> str:
    body = response.json()
    assert set(body) == {"error"} and body["error"]["request_id"]
    return str(body["error"]["code"])


class TestAuth:
    def test_health_is_public_everything_else_requires_a_key(self, client: TestClient) -> None:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "ok"
        response = client.get("/v1/documents")
        assert response.status_code == 401 and error_code(response) == "unauthorized"
        bad = client.get("/v1/documents", headers={"X-API-Key": "wrong-key-000"})
        assert bad.status_code == 401
        assert client.get("/metrics").status_code == 401

    def test_roles_are_enforced(self, client: TestClient) -> None:
        denied = upload(client, "a.txt", b"hello world", headers=READER)
        assert denied.status_code == 403 and error_code(denied) == "forbidden"
        assert denied.json()["error"]["details"] == {"role": "reader", "required": "writer"}
        train = client.post("/v1/classifier/train", headers=WRITER, json={"examples": []})
        assert train.status_code in (403, 422)
        reindex = client.post("/v1/admin/reindex", headers=WRITER, json={})
        assert reindex.status_code == 403

    def test_rate_limit(self, tmp_path: Path) -> None:
        settings = make_settings(
            tmp_path / "rl", api={"rate_limit_per_minute": 60, "rate_limit_burst": 2}
        )
        with TestClient(create_app(settings, start_workers=False)) as client:
            assert client.get("/v1/stats", headers=READER).status_code == 200
            assert client.get("/v1/stats", headers=READER).status_code == 200
            limited = client.get("/v1/stats", headers=READER)
            assert limited.status_code == 429 and error_code(limited) == "rate_limited"
            assert int(limited.headers["retry-after"]) >= 1
            # Buckets are per key.
            assert client.get("/v1/stats", headers=ADMIN).status_code == 200

    def test_missing_keys_refuse_to_start(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path / "x", api={"api_keys": []})
        with pytest.raises(ConfigurationError, match="IDE_API_KEYS"):
            create_app(settings)

    def test_auth_can_be_disabled_for_local_use(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path / "x", api={"auth_enabled": False, "api_keys": []})
        with TestClient(create_app(settings, start_workers=False)) as client:
            assert client.get("/v1/documents").status_code == 200


class TestDocuments:
    def test_sync_upload_duplicate_and_crud(self, client: TestClient) -> None:
        created = upload(client, "invoice.txt", INVOICE_TEXT.encode(), tags="finance, q1")
        assert created.status_code == 201
        document = created.json()["document"]
        assert document["status"] == "ready" and document["tags"] == ["finance", "q1"]
        assert document["metadata"]["entities"]["EMAIL"] == ["billing@acme-widgets.example"]
        doc_id = document["id"]

        duplicate = upload(client, "copy.txt", INVOICE_TEXT.encode())
        assert duplicate.status_code == 200 and duplicate.json()["duplicate"] is True

        listing = client.get("/v1/documents?tag=finance&sort=title&order=asc", headers=READER)
        assert listing.json()["total"] == 1 and listing.json()["items"][0]["metadata"] is None
        assert client.get(f"/v1/documents/{doc_id}", headers=READER).json()["title"]

        patched = client.patch(
            f"/v1/documents/{doc_id}",
            headers=WRITER,
            json={"title": "Renamed", "category": "billing"},
        )
        assert patched.status_code == 200 and patched.json()["category"] == "billing"
        chunks = client.get(f"/v1/documents/{doc_id}/chunks", headers=READER).json()
        assert chunks and chunks[0]["ordinal"] == 0

        content = client.get(f"/v1/documents/{doc_id}/content", headers=READER)
        assert content.content == INVOICE_TEXT.encode()
        assert content.headers["content-type"].startswith("text/plain")
        assert 'filename="invoice.txt"' in content.headers["content-disposition"]

        assert client.delete(f"/v1/documents/{doc_id}", headers=WRITER).status_code == 204
        missing = client.get(f"/v1/documents/{doc_id}", headers=READER)
        assert missing.status_code == 404 and error_code(missing) == "not_found"

    def test_async_upload_creates_job(self, client: TestClient) -> None:
        response = upload(client, "policy.md", POLICY_MARKDOWN.encode(), wait=False)
        assert response.status_code == 202
        job_id = response.json()["job"]["id"]
        engine: Engine = client.app.state.engine  # type: ignore[attr-defined]
        job = engine.jobs.wait_for(job_id, timeout=15)
        assert job.status.value == "succeeded"
        fetched = client.get(f"/v1/jobs/{job_id}", headers=READER).json()
        assert fetched["status"] == "succeeded" and fetched["result"]["chunks"] >= 1
        jobs = client.get("/v1/jobs?status=succeeded&type=ingest", headers=READER).json()
        assert jobs["total"] == 1
        reprocess = client.post(
            f"/v1/documents/{response.json()['document']['id']}/reprocess", headers=WRITER
        )
        assert reprocess.status_code == 202

    def test_upload_validation(self, client: TestClient) -> None:
        executable = upload(client, "setup.exe", b"MZ\x90\x00payload")
        assert executable.status_code == 415 and error_code(executable) == "unsupported_file_type"
        mismatch = upload(client, "fake.pdf", b"plain text pretending")
        assert mismatch.status_code == 415
        empty = upload(client, "empty.txt", b"")
        assert empty.status_code == 422
        bad_tags = upload(client, "a.txt", b"hello there", tags="ok,bad tag!")
        assert bad_tags.status_code == 422 and error_code(bad_tags) == "validation_error"
        no_file = client.post("/v1/documents", headers=WRITER)
        assert no_file.status_code == 422
        bad_id = client.get("/v1/documents/not-an-id", headers=READER)
        assert bad_id.status_code == 422
        bad_sort = client.get("/v1/documents?sort=password", headers=READER)
        assert bad_sort.status_code == 422

    def test_size_limits(self, tmp_path: Path) -> None:
        settings = make_settings(
            tmp_path / "s",
            ingest={"max_file_bytes": 2048},
            api={"max_request_bytes": 8192},
        )
        with TestClient(create_app(settings, start_workers=False)) as client:
            too_big_file = upload(client, "big.txt", b"a " * 1500)
            assert too_big_file.status_code == 413 and error_code(too_big_file) == "file_too_large"
            too_big_request = upload(client, "huge.txt", b"a " * 10_000)
            assert too_big_request.status_code == 413
            assert too_big_request.json()["error"]["code"] == "request_too_large"

    def test_failed_extraction_is_reported(self, client: TestClient) -> None:
        response = upload(client, "scan.pdf", build_pdf([[]]))
        assert response.status_code == 422 and error_code(response) == "extraction_failed"
        failed = client.get("/v1/documents?status=failed", headers=READER).json()
        assert failed["total"] == 1 and failed["items"][0]["error"]


class TestIntelligence:
    @pytest.fixture
    def loaded(self, client: TestClient) -> dict[str, str]:
        ids = {}
        for name, data in [
            ("invoice.txt", INVOICE_TEXT.encode()),
            ("policy.md", POLICY_MARKDOWN.encode()),
            ("guide.html", GUIDE_HTML.encode()),
        ]:
            ids[name] = upload(client, name, data).json()["document"]["id"]
        return ids

    def test_search(self, client: TestClient, loaded: dict[str, str]) -> None:
        response = client.post(
            "/v1/search?include_text=true",
            headers=READER,
            json={"query": "remote work equipment stipend", "top_k": 3, "mode": "hybrid"},
        )
        body = response.json()
        assert response.status_code == 200 and body["hits"][0]["document_id"] == loaded["policy.md"]
        hit = body["hits"][0]
        assert hit["text"] and hit["snippet"] and hit["highlights"]
        assert set(hit["ranks"]) <= {"vector", "bm25"}
        filtered = client.post(
            "/v1/search",
            headers=READER,
            json={
                "query": "deploy",
                "filters": {"doc_type": {"$in": ["html"]}},
                "group_by_document": True,
            },
        ).json()
        assert {h["document_id"] for h in filtered["hits"]} == {loaded["guide.html"]}
        assert filtered["hits"][0]["text"] is None
        bad = client.post("/v1/search", headers=READER, json={"query": "x", "filters": {"$x": 1}})
        assert bad.status_code == 422
        extra = client.post("/v1/search", headers=READER, json={"query": "x", "unknown": 1})
        assert extra.status_code == 422

    def test_ask(self, client: TestClient, loaded: dict[str, str]) -> None:
        response = client.post(
            "/v1/ask", headers=READER, json={"question": "What is the total amount due?"}
        )
        body = response.json()
        assert body["answerable"] and body["short_answer"] == "$12,450.00"
        assert body["citations"][0]["document_id"] == loaded["invoice.txt"]
        assert body["provider"] == "extractive" and body["retrieval"]["context_passages"] >= 1
        none = client.post(
            "/v1/ask", headers=READER, json={"question": "Who won the 1998 olympiad?"}
        )
        assert none.json()["answerable"] is False
        short = client.post("/v1/ask", headers=READER, json={"question": "?"})
        assert short.status_code == 422

    def test_summaries(self, client: TestClient, loaded: dict[str, str]) -> None:
        doc_id = loaded["policy.md"]
        sync = client.post(
            f"/v1/documents/{doc_id}/summarize", headers=READER, json={"max_sentences": 2}
        )
        assert sync.status_code == 200 and sync.json()["summary"]
        queued = client.post(
            f"/v1/documents/{doc_id}/summarize",
            headers=READER,
            json={"method": "mapreduce", "background": True},
        )
        assert queued.status_code == 202 and queued.json()["type"] == "summarize"
        text = client.post(
            "/v1/summarize", headers=READER, json={"text": INVOICE_TEXT, "max_sentences": 1}
        )
        assert text.status_code == 200 and len(text.json()["sentences"]) == 1

    def test_classifier_lifecycle(self, client: TestClient, loaded: dict[str, str]) -> None:
        assert client.get("/v1/classifier", headers=READER).json() == {"trained": False}
        untrained = client.post("/v1/classify", headers=READER, json={"text": "invoice"})
        assert untrained.status_code == 409 and error_code(untrained) == "model_not_trained"
        dataset = json.loads((GOLDEN_DIR / "dataset.json").read_text(encoding="utf-8"))
        examples = dataset["classification"]["train"]
        trained = client.post(
            "/v1/classifier/train",
            headers=ADMIN,
            json={"examples": examples, "algorithm": "logistic", "test_ratio": 0.25},
        )
        assert trained.status_code == 200 and trained.json()["algorithm"] == "logistic"
        prediction = client.post(
            "/v1/classify", headers=READER, json={"text": "Rollback the failed Kubernetes deploy."}
        ).json()
        assert prediction["label"] == "engineering"
        applied = client.post(f"/v1/documents/{loaded['invoice.txt']}/classify", headers=WRITER)
        assert applied.json()["label"] == "finance"
        doc = client.get(f"/v1/documents/{loaded['invoice.txt']}", headers=READER).json()
        assert doc["category"] == "finance"
        invalid = client.post(
            "/v1/classifier/train",
            headers=ADMIN,
            json={"examples": [{"label": "x"}] * 4},
        )
        assert invalid.status_code == 422
        background = client.post(
            "/v1/classifier/train", headers=ADMIN, json={"examples": examples, "background": True}
        )
        assert background.status_code == 202

    def test_jobs_admin_and_observability(self, client: TestClient, loaded: dict[str, str]) -> None:
        reindex = client.post("/v1/admin/reindex", headers=ADMIN, json={"background": False})
        assert reindex.status_code == 200 and reindex.json()["chunks"] >= 3
        queued = client.post("/v1/admin/reindex", headers=ADMIN, json={}).json()
        engine: Engine = client.app.state.engine  # type: ignore[attr-defined]
        assert engine.jobs.wait_for(queued["id"], timeout=15).status.value == "succeeded"
        cancel = client.post(f"/v1/jobs/{queued['id']}/cancel", headers=WRITER)
        assert cancel.status_code == 409 and error_code(cancel) == "conflict"
        retry = client.post(f"/v1/jobs/{queued['id']}/retry", headers=ADMIN)
        assert retry.status_code == 409
        missing = client.get("/v1/jobs/job_" + "0" * 32, headers=READER)
        assert missing.status_code == 404

        stats = client.get("/v1/stats", headers=READER).json()
        assert stats["documents"] == 3 and stats["llm"]["provider"] == "extractive"
        metrics = client.get("/metrics", headers=READER)
        assert metrics.status_code == 200
        assert (
            'idengine_http_requests_total{method="POST",route="/v1/admin/reindex",status="200"}'
            in metrics.text
        )
        assert "idengine_documents 3" in metrics.text

    def test_request_id_is_propagated(self, client: TestClient) -> None:
        response = client.get("/health", headers={"X-Request-ID": "trace-123"})
        assert response.headers["x-request-id"] == "trace-123"
        generated = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
        assert generated.headers["x-request-id"] != "bad id with spaces"
        not_found = client.get("/v1/nope", headers=READER)
        assert not_found.status_code == 404 and error_code(not_found) == "not_found"


def test_existing_engine_can_be_injected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "inj", api={"metrics_public": True})
    with Engine(settings) as engine:
        app = create_app(engine=engine, start_workers=False)
        with TestClient(app) as client:
            assert client.get("/metrics").status_code == 200  # public metrics
            assert client.get("/health").json()["checks"]["workers"]["ok"]
        # The injected engine is still usable after the app shut down.
        assert engine.stats()["documents"] == 0


def test_method_not_allowed_and_degraded_health(tmp_path: Path) -> None:
    settings = make_settings(tmp_path / "deg")
    with Engine(settings) as engine:
        app = create_app(engine=engine, start_workers=False)
        with TestClient(app) as client:
            wrong = client.put("/v1/search", headers=READER, json={})
            assert wrong.status_code == 405 and error_code(wrong) == "method_not_allowed"
            engine.db.close()  # simulate a broken database
            health = client.get("/health")
            assert health.status_code == 503
            body = health.json()
            assert body["status"] == "degraded" and not body["checks"]["database"]["ok"]
