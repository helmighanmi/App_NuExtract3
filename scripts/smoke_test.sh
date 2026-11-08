#!/usr/bin/env bash
# End-to-end smoke test against a running stack.
#   ./scripts/smoke_test.sh [API_URL]
set -euo pipefail

API_URL="${1:-${NUEXTRACT_API_BASE_URL:-http://localhost:8080}}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -f data/sample_invoice.pdf ]]; then
  python scripts/make_sample_invoice.py
fi

echo "== Health =="
curl -fsS "${API_URL}/health" | python3 -m json.tool

echo "== Readiness =="
curl -fsS "${API_URL}/ready" | python3 -m json.tool

echo "== Templates =="
curl -fsS "${API_URL}/v1/templates" | python3 -c 'import json,sys; print([t["name"] for t in json.load(sys.stdin)])'

echo "== Extract invoice =="
curl -fsS -X POST "${API_URL}/v1/extract" \
  -F "file=@data/sample_invoice.pdf" \
  -F "template_name=invoice" \
  -F "thinking=false" \
  | python3 -m json.tool

echo "== Markdown =="
curl -fsS -X POST "${API_URL}/v1/markdown" \
  -F "file=@data/sample_invoice.pdf" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["markdown"][:500])'

echo "Smoke test passed."
