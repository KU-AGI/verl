#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8080}"
PYTHON_BIN="${DETECTOR_PYTHON:-/home/work/AGILAB/conda/envs/sglang/bin/python}"
GDINO_MODEL_PATH="${GDINO_MODEL_PATH:-/home/work/AGILAB/mllm_reasoning/data/checkpoints/mm_grounding_dino_large_all}"

# Cap the detector allocator at 10% of the A100 (about 8GiB) so a request
# cannot consume the remaining memory reserved for Qwen3.5 runtime peaks.
CUDA_MEMORY_FRACTION="${CUDA_MEMORY_FRACTION:-0.10}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "${PYTHON_BIN}" recipe/image_rl/detector.py \
    --gdino_ckpt_path "${GDINO_MODEL_PATH}" \
    --cuda_memory_fraction "${CUDA_MEMORY_FRACTION}" \
    --empty_cache_after_request \
    --host 0.0.0.0 \
    --port "${PORT}"
