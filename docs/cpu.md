# Running NuExtract3 on CPU

Yes — NuExtract3 runs without a GPU. The official
[`numind/NuExtract3-GGUF`](https://huggingface.co/numind/NuExtract3-GGUF) build
ships 2–8 bit quantizations plus a vision projector, and Ollama publishes the
same weights as `numind/nuextract3`.

CPU inference is **functional, not fast**. A single scanned page can take tens of
seconds to a few minutes; plan for that in timeouts and UX.

## Choose a backend

| Backend | Command / setting | Hardware | Notes |
| --- | --- | --- | --- |
| **Ollama** (recommended) | `NUEXTRACT_PROVIDER=ollama` | CPU, Metal, CUDA, ROCm | Easiest setup; quantized GGUF; automatically uses the best accelerator |
| **llama.cpp server** | `NUEXTRACT_PROVIDER=llamacpp` | CPU, CUDA, Metal | Full control over context, threads, batching |
| **Transformers** | `NUEXTRACT_PROVIDER=transformers` | CPU / CUDA / MPS | No extra service, but loads full-precision weights and is the slowest |
| **vLLM** | `NUEXTRACT_PROVIDER=vllm` | NVIDIA GPU (Linux) | Best throughput; **not available on CPU or macOS** |

All four speak the same internal interface; only configuration changes.

## Memory and disk

| Model | Disk | Approx. RAM (CPU) | Quality |
| --- | --- | --- | --- |
| `nuextract3:q4_k_m` (Q4_K_M) | 3.4 GB | ~5–6 GB | Recommended default |
| `nuextract3:q6_k` (Q6_K) | 4.1 GB | ~6–8 GB | Slightly better, slower |
| `nuextract3:bf16` | 9.3 GB | ~12–16 GB | Reference quality |
| Transformers, `dtype=float32` | ~16 GB | ~18–20 GB | Full precision, slowest |
| Transformers, `dtype=bfloat16` | ~8 GB | ~10–12 GB | Needs a CPU with BF16 support for decent speed |

Context length dominates memory. The Ollama default is 131k tokens; lower it in
the Modelfile or with `OLLAMA_CONTEXT_LENGTH` if you run out of RAM (for example
`OLLAMA_CONTEXT_LENGTH=16384`). The OpenAI-compatible endpoint only forwards
`num_ctx` on newer Ollama builds, so the server-wide variable is the reliable
option. In this project, `NUEXTRACT_OLLAMA_NUM_CTX` sends the per-request field
and `NUEXTRACT_PAGES_PER_REQUEST` is the practical lever: smaller chunks mean
shorter prompts and less KV cache.

## Option A — native Ollama (best on macOS)

Install [Ollama](https://ollama.com/download), then:

```bash
ollama pull nuextract3:q4_k_m
cp .env.example .env
```

Set in `.env`:

```dotenv
NUEXTRACT_PROVIDER=ollama
NUEXTRACT_OLLAMA_MODEL=nuextract3:q4_k_m
NUEXTRACT_REQUEST_TIMEOUT_S=900
```

Start the app:

```bash
uv sync
make api     # terminal 1 → http://localhost:8080
make ui      # terminal 2 → http://localhost:8501
```

Verify:

```bash
uv run nuextract health
# OK ollama backend at http://localhost:11434/v1 serving nuextract3:q4_k_m
```

On Apple Silicon, run Ollama natively so it can use Metal. On Intel Macs, Ollama
runs on CPU only. A Linux Ollama container started by Docker Desktop on macOS
**cannot** access the Metal GPU and always uses CPU.

## Option B — Docker CPU stack (Ollama in a container)

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

This starts three services:

1. `ollama` — model server with a persistent `ollama-models` volume.
2. `ollama-pull` — one-shot job that downloads `nuextract3:q4_k_m` (~3.4 GB).
3. `api` + `studio` — FastAPI on <http://localhost:8080>, Streamlit on
   <http://localhost:8501>.

The first start is slow because of the model download. Later starts reuse the
volume. Change the tag with `NUEXTRACT_OLLAMA_MODEL=nuextract3:q6_k`.

The API points at `http://ollama:11434`. The Ollama API is also published on
<http://localhost:11434> if you want to query it directly.

## Option C — self-contained CPU image (Transformers)

Use this if you cannot run a model server at all. It installs CPU-only PyTorch
inside the API image (the `cpu` extra, backed by `download.pytorch.org/whl/cpu`)
and loads the model in-process.

```bash
docker build -f docker/Dockerfile.cpu -t nuextract3-studio:cpu .
docker run --rm -p 8080:8080 \
  -e NUEXTRACT_PROVIDER=transformers \
  -e NUEXTRACT_DEVICE=cpu \
  -e NUEXTRACT_DTYPE=bfloat16 \
  -e NUEXTRACT_LOCAL_MAX_OUTPUT_TOKENS=2048 \
  -v hf-cache:/models/huggingface \
  nuextract3-studio:cpu
```

Or bare metal:

```bash
uv sync --extra cpu     # Linux/Windows CPU-only wheels
# Apple Silicon / CUDA: use `uv sync --extra local`
NUEXTRACT_PROVIDER=transformers \
NUEXTRACT_DEVICE=cpu \
NUEXTRACT_DTYPE=bfloat16 \
NUEXTRACT_CPU_THREADS=8 \
uv run nuextract health
```

Notes:

- Recent PyTorch releases ship macOS wheels for Apple Silicon only; Intel Macs
  should use Ollama instead of the in-process backend.
- The first run downloads ~9 GB of safetensors to `HF_HOME`.
- `dtype=auto` on CPU resolves to `float32` (safest, ~16 GB of weights).
  `bfloat16` halves memory but needs AVX-512 BF16 or newer for acceptable speed.
- `NUEXTRACT_CPU_THREADS` sets `torch.set_num_threads`; use the number of
  physical cores, not hyperthreads.
- This path is best for smoke tests and low-volume use.

## Option D — llama.cpp server (advanced)

Download the official GGUF files:

```bash
huggingface-cli download numind/NuExtract3-GGUF \
  NuExtract3-Q4_K_M.gguf mmproj-NuExtract3-BF16.gguf \
  --local-dir models/gguf
```

Serve them:

```bash
llama-server \
  --model models/gguf/NuExtract3-Q4_K_M.gguf \
  --mmproj models/gguf/mmproj-NuExtract3-BF16.gguf \
  --jinja \
  --ctx-size 16384 \
  --threads 8 \
  --host 0.0.0.0 --port 8000
```

Then:

```dotenv
NUEXTRACT_PROVIDER=llamacpp
NUEXTRACT_MODEL_BASE_URL=http://localhost:8000/v1
```

`llama-server` forwards `chat_template_kwargs`, which is how the template,
instructions and mode reach the model. Keep `--jinja` enabled (default in recent
builds). Very recent llama.cpp builds moved thinking control to
`--reasoning on|off`; disable thinking in this project unless you have verified
it on your build.

## Latency expectations

Numbers vary by CPU, quantization and document size. Treat these as rough orders
of magnitude for a 1-page document at 200 DPI:

| Setup | Time per page | Typical use |
| --- | --- | --- |
| Apple M-series, Ollama + Metal, Q4_K_M | 10–40 s | Local demo, interactive |
| Intel Mac (e.g. i7), Ollama CPU, Q4_K_M | 1–5 min | Batch/offline |
| Modern x86 CPU, Q4_K_M | 40–180 s | Batch/offline |
| Older CPU, Q4_K_M | 2–6 min | Not interactive |
| Transformers CPU, float32 | 5–20+ min | Smoke test only |

Practical tuning:

```dotenv
NUEXTRACT_PAGES_PER_REQUEST=1
NUEXTRACT_MAX_OUTPUT_TOKENS=2048
NUEXTRACT_DPI=150
NUEXTRACT_ENABLE_THINKING=false
NUEXTRACT_MAX_PAGES=10
```

Disable reasoning mode: on small quantized models it multiplies latency and, per
the model card, can hurt extraction quality by triggering repetition loops.

## API and UI behaviour

- `GET /health` reports the configured provider plus `torch_available`; it never
  imports PyTorch, so it stays cheap.
- `GET /ready` probes the backend and returns the resolved provider,
  device, dtype, endpoint and a note. Example:

```json
{
  "status": "ready",
  "backend": true,
  "provider": "ollama",
  "resolved_provider": "ollama",
  "device": "auto",
  "dtype": null,
  "endpoint": "http://localhost:11434/v1",
  "note": "Ollama picks CPU/Metal/CUDA automatically"
}
```

- `provider=auto` tries an OpenAI-compatible server first, then Ollama, then
  in-process Transformers. Set an explicit provider in production so startup
  behaviour is predictable.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `/ready` → 503 with `not_ready` | Check `ollama list`; make sure the configured tag is pulled |
| `model requires more system memory` | Use a smaller quant (`q4_k_m`) or lower `num_ctx` |
| Very slow first request | Model load; subsequent requests reuse it. Raise the timeout. |
| Empty/truncated JSON | Increase `NUEXTRACT_MAX_OUTPUT_TOKENS`, lower `pages_per_request` |
| `float16 is not supported on CPU` | The runtime automatically falls back to `float32` for the Transformers backend |
| Docker on macOS feels slow | Ollama in a Linux VM has no Metal access; install Ollama natively |
| llama.cpp ignores the template | Ensure `--jinja` is enabled and the GGUF is `numind/NuExtract3-GGUF` |

## Expected quality

Quantization is a trade-off. Q4_K_M preserves the schema and most text fields
well, but OCR-heavy scans, unusual layouts, and long tables degrade first. Keep a
small golden set and compare Q4_K_M, Q6_K and BF16 with the evaluation harness:

```bash
make eval ARGS="--dataset eval/datasets/example.jsonl --template invoice"
```

If quality matters more than hardware cost, use the full-precision checkpoint on
a GPU with vLLM, or the managed NuExtract platform.
