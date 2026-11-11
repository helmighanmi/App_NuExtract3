# Deployment guide

## GPU sizing

NuExtract3 is a ~4B-parameter model in BF16 (≈8 GB of weights). The real constraint
is the KV cache for long documents and multi-image prompts.

| GPU VRAM | Recommended `--max-model-len` | Notes |
| --- | --- | --- |
| 16 GB | 8k–16k | tight; prefer an FP8/AWQ checkpoint |
| 24 GB (A10/L4/4090) | 32k | good default for 8-page chunks |
| 48 GB (L40S/A6000) | 64k–131k | comfortable |
| 80 GB (H100/A100) | 131k | full context, higher concurrency |

Start at `--max-model-len 32768`; raise it only after measuring KV-cache pressure
in the vLLM logs. `--gpu-memory-utilization 0.90` leaves headroom for the display /
other processes; use `0.95` on dedicated nodes.

## vLLM

The production flags live in [`deploy/vllm/serve.sh`](../deploy/vllm/serve.sh):

```bash
vllm serve numind/NuExtract3 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 16, "video": 0}' \
  --chat-template-content-format openai \
  --generation-config vllm \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

Tunables (environment variables for the script):

| Variable | Default | Effect |
| --- | --- | --- |
| `MAX_MODEL_LEN` | `32768` | context window; dominates KV usage |
| `GPU_MEMORY_UTILIZATION` | `0.90` | fraction of VRAM vLLM may use |
| `MAX_IMAGES_PER_PROMPT` | `16` | must be ≥ `pages_per_request` |
| `MAX_NUM_SEQS` | vLLM default | concurrent sequences |
| `TENSOR_PARALLEL_SIZE` | unset | multi-GPU tensor parallelism |
| `SPECULATIVE_CONFIG` | unset | MTP speculative decoding |

### Multi Token Prediction (MTP)

The model card recommends speculative decoding with `qwen3_next_mtp`:

```bash
SPECULATIVE_CONFIG='{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  ./deploy/vllm/serve.sh
```

Enable it only if your vLLM build supports this method; otherwise the server fails
at startup. Benchmark acceptance rate before relying on it.

### Quantization

NuExtract3 has community FP8/AWQ/GPTQ checkpoints on the Hub. Pass the matching
`--quantization` flag (e.g. `awq`, `gptq`, `fp8`) and point `MODEL_ID` at the
quantized repository. Validate extraction quality on your own golden set before
switching — small quantized VLMs can lose OCR fidelity.

### API key

vLLM reads `VLLM_API_KEY` from the environment and then requires
`Authorization: Bearer <key>`. Set the same key in `NUEXTRACT_MODEL_API_KEY`.
Never rely on the default `EMPTY` key on a public network.

## Docker

The CPU and GPU concerns are split into separate images/overlays:

- `docker/Dockerfile.app` — FastAPI + Streamlit, no model, non-root, ~small.
- `docker/Dockerfile.cpu` — adds CPU-only PyTorch for in-process Transformers.
- `docker/Dockerfile.vllm` — `vllm/vllm-openai` + our serving entrypoint.

```bash
# App only, pointing at an external vLLM (host gateway by default)
docker compose up --build -d

# or a remote endpoint
NUEXTRACT_DOCKER_MODEL_BASE_URL=https://gpu-host.example.com/v1 docker compose up --build -d

# Full GPU stack on a Linux NVIDIA host
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d

# CPU stack: Ollama (GGUF) + API + Streamlit, no GPU required
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build -d
```

Pin the vLLM base image (`vllm/vllm-openai:vX.Y.Z`) in production; `latest` can
change model support and flags.

> Docker Desktop on macOS cannot run vLLM (no NVIDIA GPU passthrough). Use a
> remote endpoint, the CPU stack, or native Ollama. For the best Apple Silicon
> performance, run Ollama natively so it can use Metal.

## CPU deployment

vLLM does not run on CPU or macOS. Use one of the paths below; all of them are
supported by the same API and Streamlit UI.

| Path | Image / command | Best for |
| --- | --- | --- |
| Ollama (GGUF) | `docker-compose.cpu.yml` or native `ollama pull nuextract3:q4_k_m` | Recommended CPU path |
| llama.cpp server | `llama-server` + official GGUF + `mmproj` | Fine-grained control |
| Transformers | `docker/Dockerfile.cpu`, `NUEXTRACT_PROVIDER=transformers` | No extra service |

Quick start with Docker:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build -d
```

Quick start with native Ollama:

```bash
ollama pull nuextract3:q4_k_m
export NUEXTRACT_PROVIDER=ollama
make api
```

CPU sizing:

| Quantization | Disk | Approx. RAM | Notes |
| --- | --- | --- | --- |
| Q4_K_M | 3.4 GB | 5–6 GB | default; best speed/quality trade-off |
| Q6_K | 4.1 GB | 6–8 GB | slower, slightly better |
| BF16 | 9.3 GB | 12–16 GB | reference quality |
| Transformers float32 | ~16 GB | 18–20 GB | smoke tests only |

Expect tens of seconds to several minutes per scanned page depending on CPU,
quantization, page count and output length. Reduce `pages_per_request`,
`dpi`, `max_output_tokens` and keep thinking disabled for CPU workloads.

See [docs/cpu.md](cpu.md) for setup, benchmarks, tuning and troubleshooting.

## Kubernetes

Manifests in [`deploy/kubernetes`](../deploy/kubernetes):

- `vllm.yaml` — 1 GPU replica, PVC for weights, `/health` probes, 16 Gi shared
  memory (`/dev/shm`) for vLLM, `Recreate` strategy to avoid two pods loading the
  model onto one GPU.
- `app.yaml` — stateless API (2+ replicas, HPA) and Streamlit.

Before applying:

1. Build and push both images; update the `image:` fields.
2. Adjust `nodeSelector` for your GPU node labels.
3. Create the optional `hf-token` secret for gated/private models.
4. Set `NUEXTRACT_MODEL_API_KEY` from a secret if vLLM auth is enabled.
5. Add a `StorageClass` / PV size that fits the weights (~10 GB) plus headroom.

```bash
kubectl apply -k deploy/kubernetes
```

Scale the GPU deployment to zero when idle if your cloud supports it; cold starts
then include weight loading (tens of seconds to a few minutes).

## Cloud GPU options

The model image is portable. Common patterns:

- **RunPod / Lambda / Vast.ai / Paperspace** — run the GPU container, expose port
  8000, point `NUEXTRACT_MODEL_BASE_URL` at the public HTTPS URL.
- **Modal / Baseten / Replicate** — wrap the same vLLM command as a serverless
  container; configure scale-to-zero and request timeouts for long PDFs.
- **AWS SageMaker / GCP Vertex AI** — custom container inference with the vLLM
  image, health path `/health`.
- **Hugging Face Inference Endpoints** — vLLM-backed endpoint where available;
  otherwise deploy your own container.

For all of them: put TLS and authentication in front of both the API and vLLM.

## Production checklist

- [ ] Pin image and model revisions (avoid floating `latest`).
- [ ] Add authentication to the API (OIDC/API gateway).
- [ ] Rate limit by tenant and cap concurrent extractions.
- [ ] Add request timeouts that match the longest document, plus client retries.
- [ ] Persist results / audit logs if documents are business records.
- [ ] Redact PII where required; avoid logging file contents.
- [ ] Monitor p95 latency, tokens/document, parse-failure rate and GPU utilization.
- [ ] Benchmark on a representative golden set after every model/vLLM upgrade.
