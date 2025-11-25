# !/bin/bash

# Reference:
# https://verl.readthedocs.io/en/latest/advance/checkpoint.html

PROJECT_DIR=$1
GLOBAL_STEP=$2

ACTOR_DIR="/data/verl/ckpts/mllm_reasoning/${PROJECT_DIR}/${GLOBAL_STEP}/actor"
TARGET_DIR="/data/verl/ckpts/mllm_reasoning/${PROJECT_DIR}/${GLOBAL_STEP}/hf_model"

# FSDP
python -m verl.model_merger_janus merge \
    --backend fsdp \
    --trust-remote-code \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$TARGET_DIR"