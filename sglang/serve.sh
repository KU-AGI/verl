#!/bin/bash

for i in 7; do
    port=$((8000 + i))
    CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
        --model-path /nas2/mllm_reasoning/checkpoints/Qwen3.5-27B \
        --served-model-name Qwen/Qwen3.5-27B \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port $port \
        --mem-fraction-static 0.9 \
        --max-running-requests 256 \
        --max-total-tokens 32768 \
        --chunked-prefill-size 4096 \
        --attention-backend flashinfer \
        --sampling-backend flashinfer &
done
wait