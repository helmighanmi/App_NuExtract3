#!/usr/bin/env bash
# Serve NuExtract3 with vLLM. Works inside the vLLM image and on bare metal.
set -euo pipefail

MODEL_ID="${MODEL_ID:-numind/NuExtract3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

args=(
  serve "${MODEL_ID}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
  --chat-template-content-format openai
  --generation-config vllm
  --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES_PER_PROMPT:-16}, \"video\": 0}"
  --max-model-len "${MAX_MODEL_LEN:-32768}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
)

if [[ -n "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  args+=(--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}")
fi

# Multi Token Prediction (MTP) speculative decoding. Enable only if your vLLM
# build supports qwen3_next_mtp:
#   SPECULATIVE_CONFIG='{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
  args+=(--speculative-config "${SPECULATIVE_CONFIG}")
fi

if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
  args+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi

echo "Starting vLLM: ${MODEL_ID} on ${HOST}:${PORT} (max-model-len=${MAX_MODEL_LEN:-32768})"
exec vllm "${args[@]}"
