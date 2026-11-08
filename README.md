# NuExtract3 Studio

Document intelligence application built around **[numind/NuExtract3](https://huggingface.co/numind/NuExtract3)**:
a 4B vision-language model for **schema-guided extraction** and **document-to-Markdown**.
It handles digital PDFs, scans, photos, receipts, invoices, contracts and multi-page
documents through a single pipeline.

The project is intentionally structured like a production AI service rather than a
notebook: a stateless FastAPI orchestration layer, a separately deployed GPU
inference server (vLLM), typed configuration, a template registry, JSON repair and
output validation, tests, evaluation tooling and a Streamlit demonstrator.

## Features

- **Structured extraction** — PDF/image + JSON template → validated JSON.
- **Document-to-Markdown** — OCR/layout conversion for RAG preprocessing.
- **Template generation** — create a template from a natural-language description.
- **JSON Schema conversion** — reuse Pydantic / JSON Schema definitions.
- **Hybrid page routing** — native text for digital pages, vision for scanned pages.
- **Multi-page handling** — page-level chunking with recursive result merging.
- **Robust post-processing** — code-fence stripping, JSON repair, shape completion.
- **Three interfaces** — FastAPI (OpenAPI docs), CLI, Streamlit demo.
- **Docker & vLLM deployment** — CPU app image + GPU model image + Compose/K8s.
- **Evaluation harness** — tree-based field scoring on a JSONL golden set.

## Architecture

```mermaid
flowchart LR
    U[Streamlit UI] -->|HTTP| API[FastAPI orchestrator]
    C[CLI] -->|HTTP / in-process| API
    API --> ING[Ingestion<br/>PyMuPDF + Pillow]
    ING --> ROUTE{Page router}
    ROUTE -->|digital text| TXT[Text prompt]
    ROUTE -->|scan / image| IMG[Page images]
    TXT --> VLLM[vLLM<br/>numind/NuExtract3]
    IMG --> VLLM
    VLLM --> POST[JSON repair<br/>merge + validate]
    POST --> API
    API --> UI[(results)]
```

**Why vLLM as a separate service?** The model server owns the GPU, batching,
KV-cache and model lifecycle. The API stays stateless, cheap to scale and easy to
test. It can point to a local container, a Kubernetes deployment, or a remote
GPU endpoint without code changes.

## Project structure

```text
.
├── src/nuextract_app/
│   ├── api/                  # FastAPI app, routes, schemas, dependencies
│   │   ├── main.py           # app factory, middleware, error handlers
│   │   └── routes/           # health, extract, markdown, templates
│   ├── core/
│   │   ├── document.py       # PDF/image ingestion, rendering, scan detection
│   │   ├── templates.py      # registry, JSON-Schema conversion, validation
│   │   ├── client.py         # vLLM (async) + Transformers backends
│   │   ├── pipeline.py       # routing, chunking, merging, orchestration
│   │   └── postprocess.py    # JSON repair, recursive merge, shape completion
│   ├── config.py             # pydantic-settings (env driven)
│   ├── logging.py            # request-scoped structured logs
│   └── cli.py                # `nuextract` Typer CLI
├── apps/streamlit/           # demonstrator (pure API client)
├── configs/templates/        # invoice, receipt, contract, id_card, bank_statement
├── docker/                   # Dockerfile.app (CPU) + Dockerfile.vllm (GPU)
├── deploy/
│   ├── vllm/serve.sh         # production vLLM flags
│   └── kubernetes/           # Deployment/Service/HPA/PVC manifests
├── eval/                     # golden-set runner + tree metrics
├── tests/                    # unit + API tests (no GPU required)
├── scripts/                  # sample invoice, model download, smoke test
├── docs/                     # architecture, deployment, extraction notes
├── docker-compose.yml        # API + Streamlit (external vLLM)
└── docker-compose.gpu.yml    # adds vLLM (NVIDIA)
```

## Quickstart

### Prerequisites

- Python **3.11 / 3.12**, [uv](https://docs.astral.sh/uv/)
- Docker (optional)
- A Linux/NVIDIA GPU for vLLM (or a remote vLLM endpoint)
- ~10 GB of disk for model weights

> vLLM does **not** run on macOS/Apple Silicon Docker. On a Mac, use a remote GPU
> endpoint or the CPU `transformers` fallback (very slow, for smoke tests only).

### 1. Install

```bash
cp .env.example .env
uv sync
```

### 2. Start NuExtract3 with vLLM

Bare metal (Linux + CUDA):

```bash
./deploy/vllm/serve.sh
# or with MTP speculative decoding:
SPECULATIVE_CONFIG='{"method":"qwen3_next_mtp","num_speculative_tokens":2}' ./deploy/vllm/serve.sh
```

The server exposes an OpenAI-compatible API on `http://localhost:8000/v1`.

### 3. Run the app

```bash
make api     # FastAPI on http://localhost:8080  (docs at /docs)
make ui      # Streamlit on http://localhost:8501
```

CLI examples:

```bash
uv run nuextract health
uv run nuextract templates list
uv run nuextract extract receipt.pdf --template receipt --output result.json
uv run nuextract markdown scan.png --output scan.md
uv run nuextract templates generate "Key fields from a commercial lease"
```

### 4. Docker

CPU app + external vLLM:

```bash
docker compose up --build
# remote endpoint:
NUEXTRACT_DOCKER_MODEL_BASE_URL=https://my-vllm.example.com/v1 docker compose up --build
```

Full GPU stack (Linux + NVIDIA Container Toolkit):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Then open <http://localhost:8501> (demo) and <http://localhost:8080/docs> (API).

## API

| Endpoint | Description |
| --- | --- |
| `GET /health` | Liveness + model metadata |
| `GET /ready` | Checks the vLLM backend |
| `POST /v1/extract` | Structured extraction (multipart) |
| `POST /v1/extract/batch` | Up to 20 documents, same template |
| `POST /v1/markdown` | Document → Markdown |
| `GET /v1/templates` | Bundled template catalogue |
| `POST /v1/templates/generate` | Template from a description |
| `POST /v1/templates/convert` | JSON Schema → NuExtract template |

```bash
curl -X POST http://localhost:8080/v1/extract \
  -F "file=@data/sample_invoice.pdf" \
  -F "template_name=invoice" \
  -F "thinking=false"
```

Extraction options: `template_name` **or** inline `template` JSON, `instructions`,
`thinking`, `dpi`, `max_pages`, `force_vision`, `include_raw`.

## Templates

A template mirrors the output shape; leaves are types:

```json
{
  "invoice_number": "verbatim-string",
  "invoice_date": "date",
  "total": "number",
  "currency": ["USD", "EUR", "TND", "Other"],
  "line_items": [
    { "description": "verbatim-string", "quantity": "number", "unit_price": "number" }
  ]
}
```

Supported leaves include `verbatim-string`, `string`, `integer`, `number`,
`boolean`, `date`, `time`, `date-time`, `country`, `currency`, `email`, `phone`,
`url`, enums (`["a", "b"]`) and multi-enums (`[["a", "b"]]`). See
[docs/extraction.md](docs/extraction.md).

## Configuration

All settings are read from `.env` with the `NUEXTRACT_` prefix. Key variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NUEXTRACT_PROVIDER` | `vllm` | `vllm` or `transformers` |
| `NUEXTRACT_MODEL_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `NUEXTRACT_MODEL_NAME` | `numind/NuExtract3` | Served model name |
| `NUEXTRACT_MAX_PAGES` | `40` | Hard page cap per document |
| `NUEXTRACT_DPI` | `200` | PDF render resolution |
| `NUEXTRACT_PAGES_PER_REQUEST` | `8` | Pages per model call |
| `NUEXTRACT_MAX_UPLOAD_MB` | `50` | Upload limit |
| `NUEXTRACT_ENABLE_THINKING` | `false` | Default reasoning mode |

See [.env.example](.env.example) for the full list.

## Deployment

- [docs/deployment.md](docs/deployment.md) — GPU sizing, vLLM flags, quantization,
  Docker, Kubernetes, cloud options.
- [deploy/kubernetes](deploy/kubernetes) — production manifests.
- [docs/architecture.md](docs/architecture.md) — internals and design decisions.

## Evaluation

The repository includes a small tree-based evaluator (field alignment, string
similarity, exact match for scalars) so changes can be measured instead of
eyeballed:

```bash
make sample
make eval ARGS="--dataset eval/datasets/example.jsonl --template invoice"
```

Dataset format (JSONL): `{"id": "...", "file": "path.pdf", "template": "invoice", "ground_truth": {...}}`.

## Testing

```bash
make check      # ruff + mypy + pytest
./scripts/smoke_test.sh   # against a running stack
```

Tests do not require a GPU: the model backend is faked at the client boundary.

## Roadmap

- Persistent extraction history + human review UI.
- Redis cache keyed by document hash + template.
- Async job queue for large batches (Celery/Arq).
- Pydantic response models per template and strict validation.
- Prometheus/OpenTelemetry metrics (tokens, latency, failure rate).
- AuthN/AuthZ, PII redaction, audit logs.
- Confidence scoring and selective re-runs in thinking mode.
- Fine-tuning / LoRA experiments and a public benchmark report.

## License

Code: Apache-2.0. Model: `numind/NuExtract3` is released under Apache-2.0 by NuMind.
