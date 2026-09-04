#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5 hybrid attention/Mamba server sharing one A100 80GB with GDINO.
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
PYTHON_BIN="${SGLANG_PYTHON:-/home/work/AGILAB/conda/envs/sglang/bin/python}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.87}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-4096}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-24}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path /home/work/AGILAB/mllm_reasoning/data/checkpoints/Qwen3.5-35B-A3B \
    --served-model-name Qwen/Qwen3.5-35B-A3B \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
    --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}" \
    --mamba-ssm-dtype bfloat16 \
    --attention-backend flashinfer \
    --sampling-backend flashinfer
