#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:?GPU_ID is required}"
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be a non-negative integer: ${GPU_ID}" >&2
    exit 2
fi

SGLANG_PYTHON="${SGLANG_PYTHON:-/home/work/AGILAB/conda/envs/sglang/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/work/AGILAB/mllm_reasoning/data/checkpoints/Qwen3.5-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
PORT="${PORT:-$((8000 + GPU_ID))}"
HOST="${HOST:-0.0.0.0}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.94}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-80}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-16}"
NUM_CONTINUOUS_DECODE_STEPS="${NUM_CONTINUOUS_DECODE_STEPS:-4}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
HF_HOME="${HF_HOME:-/data/.cache/huggingface}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "${SGLANG_PYTHON}" ]]; then
    echo "SGLang Python not found or not executable: ${SGLANG_PYTHON}" >&2
    exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model path not found: ${MODEL_PATH}" >&2
    exit 2
fi

command=(
    "${SGLANG_PYTHON}" -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --trust-remote-code
    --host "${HOST}"
    --port "${PORT}"
    --tp-size 1
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --context-length "${CONTEXT_LENGTH}"
    --reasoning-parser qwen3
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}"
    --attention-backend "${ATTENTION_BACKEND}"
    --sampling-backend "${SAMPLING_BACKEND}"
    --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}"
    --num-continuous-decode-steps "${NUM_CONTINUOUS_DECODE_STEPS}"
    --enable-mixed-chunk
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q HF_HOME=%q ' "${GPU_ID}" "${HF_HOME}"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HOME
exec "${command[@]}"
