#!/bin/bash

export HF_HOME=/data/.cache/huggingface

for i in 6; do
    port=$((8000 + i))
    CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
        --model-path Qwen/Qwen3.5-35B-A3B \
        --served-model-name Qwen/Qwen3.5-35B-A3B \
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