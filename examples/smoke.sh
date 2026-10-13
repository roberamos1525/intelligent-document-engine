#!/usr/bin/env bash
# End-to-end smoke test against a running server: upload -> search -> ask.
# Usage: BASE=http://127.0.0.1:8000 KEY=dev-admin-key ./examples/smoke.sh
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
KEY="${KEY:?set KEY to an API key with the writer or admin role}"
FILE="${FILE:-tests/fixtures/golden/corpus/f_invoice.txt}"

echo "== health"
curl -fsS "$BASE/health"
echo

echo "== upload (synchronous)"
curl -fsS -X POST "$BASE/v1/documents?wait=true" \
  -H "X-API-Key: $KEY" \
  -F "file=@${FILE}" -F "tags=smoke"
echo

echo "== search"
curl -fsS -X POST "$BASE/v1/search" \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query": "late payment fee", "top_k": 3}'
echo

echo "== ask"
curl -fsS -X POST "$BASE/v1/ask" \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is the total amount due on invoice INV-7781?"}'
echo
