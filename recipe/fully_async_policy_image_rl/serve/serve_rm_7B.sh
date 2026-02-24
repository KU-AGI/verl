#!/bin/bash

# CUDA_VISIBLE_DEVICES=4,5 vllm serve --host 0.0.0.0 --port 8004 --config ./recipe/fully_async_policy_image_rl/serve/rm_config.yaml &

# CUDA_VISIBLE_DEVICES=6,7 vllm serve --host 0.0.0.0 --port 8006 --config ./recipe/fully_async_policy_image_rl/serve/rm_config.yaml

for i in 7; do
    port=$((8000 + i))
    CUDA_VISIBLE_DEVICES=$i,$((i + 1)) vllm serve \
        --host 0.0.0.0 \
        --port $port \
        --config ./recipe/fully_async_policy_image_rl/serve/rm_config.yaml &
done
wait