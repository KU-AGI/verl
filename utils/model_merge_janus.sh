# !/bin/bash

# Reference:
# https://verl.readthedocs.io/en/latest/advance/checkpoint.html

# FSDP
python -m verl.model_merger_janus merge \
    --backend fsdp \
    --trust-remote-code \
    --local_dir /data/verl/ckpts/mllm_reasoning/naive_qwen_vlm/global_step_300/actor \
    --target_dir /data/verl/ckpts/mllm_reasoning/naive_qwen_vlm/global_step_300/hf_model

python -m verl.model_merger_janus merge \
    --backend fsdp \
    --trust-remote-code \
    --local_dir /data/verl/ckpts/mllm_reasoning/naive_qwen_vlm/global_step_600/actor \
    --target_dir /data/verl/ckpts/mllm_reasoning/naive_qwen_vlm/global_step_600/hf_model

python -m verl.model_merger_janus merge \
    --backend fsdp \
    --trust-remote-code \
    --local_dir /data/verl/ckpts/mllm_reasoning/naive_qwen_vlm/global_step_870/actor \
    --target_dir /data/verl/ckpts/mllm_reasoning/naive_qwen_vlm/global_step_870/hf_model