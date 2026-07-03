#!/usr/bin/env bash
set -euo pipefail

# Reference:
# https://verl.readthedocs.io/en/latest/advance/checkpoint.html

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <experiment_name> <step|global_step_step>"
    echo "Example: $0 0530_KT_v5_fine_grained_lr_1e_6_GAE_length_norm_gamma_0_6_sglang_v2 3500"
    exit 1
fi

PROJECT_DIR=$1
STEP=$2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/ckpts/mllm_reasoning}"

if [[ "${STEP}" == global_step_* ]]; then
    GLOBAL_STEP="${STEP}"
else
    GLOBAL_STEP="global_step_${STEP}"
fi

ACTOR_DIR="${CKPT_ROOT}/${PROJECT_DIR}/${GLOBAL_STEP}/actor"
TARGET_DIR="${CKPT_ROOT}/${PROJECT_DIR}/${GLOBAL_STEP}/hf_model"

if [ ! -d "${ACTOR_DIR}" ]; then
    echo "Actor checkpoint directory does not exist: ${ACTOR_DIR}" >&2
    exit 1
fi

echo "Merging Janus FSDP checkpoint"
echo "  actor:  ${ACTOR_DIR}"
echo "  target: ${TARGET_DIR}"

# FSDP
python -m verl.model_merger_janus merge \
    --backend fsdp \
    --trust-remote-code \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$TARGET_DIR"
