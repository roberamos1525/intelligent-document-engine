"""Minimal HTTP client for a running API (``idengine serve``).

    IDE_API_KEYS=dev-admin-key:admin idengine serve --port 8000
    python examples/api_client.py --url http://127.0.0.1:8000 --key dev-admin-key FILE...

Uploads the given files (asynchronously, via the job queue), waits for the
ingest jobs to finish, then runs a search and asks a question.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx


def wait_for_job(client: httpx.Client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").raise_for_status().json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.25)
    raise TimeoutError(f"job {job_id} did not finish in {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--key", required=True, help="API key with the writer or admin role")
    parser.add_argument("--question", default="What is the total amount due?")
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, headers={"X-API-Key": args.key}, timeout=30) as client:
        for path in args.files:
            with path.open("rb") as handle:
                response = client.post("/v1/documents", files={"file": (path.name, handle)})
            if response.status_code >= 400:
                print(f"{path.name}: {response.json()['error']}", file=sys.stderr)
                continue
            body = response.json()
            if body["job"]:
                job = wait_for_job(client, body["job"]["id"])
                print(f"{path.name}: job {job['status']} ({job.get('result')})")
            else:
                print(f"{path.name}: duplicate={body['duplicate']} id={body['document']['id']}")

        hits = client.post("/v1/search", json={"query": args.question, "top_k": 3}).json()["hits"]
        print("\nTop hits:")
        for hit in hits:
            print(f"  {hit['score']:.4f}  {hit['title']}: {hit['snippet'][:90]}")

        answer = client.post("/v1/ask", json={"question": args.question}).json()
        print(f"\nQ: {answer['question']}\nA: {answer['answer']}")
        for citation in answer["citations"]:
            print(f"   [{citation['number']}] {citation['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
