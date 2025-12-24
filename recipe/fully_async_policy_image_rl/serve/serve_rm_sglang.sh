#!/bin/bash

# CUDA_VISIBLE_DEVICES=4,5 vllm serve --host 0.0.0.0 --port 8004 --config ./recipe/fully_async_policy_image_rl/serve/rm_config.yaml &

# CUDA_VISIBLE_DEVICES=6,7 vllm serve --host 0.0.0.0 --port 8006 --config ./recipe/fully_async_policy_image_rl/serve/rm_config.yaml

for i in 6 7; do
    port=$((8000 + i))
    CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
        --model-path /home/work/AGILAB/mllm_reasoning/data/checkpoints/Qwen3-VL-30B-A3B-Instruct \
        --served-model-name Qwen/Qwen3-VL-30B-A3B-Instruct \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port $port \
        --mem-fraction-static 0.9 \
        --max-running-requests 128 \
        --max-total-tokens 4096 \
        --attention-backend flashinfer \
        --sampling-backend flashinfer &
done
wait