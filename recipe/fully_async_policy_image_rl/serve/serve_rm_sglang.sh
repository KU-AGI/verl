#!/bin/bash

export HF_HOME=/data/.cache/huggingface

for i in 6 7; do
    port=$((8000 + i))
    CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-VL-7B-Instruct \
        --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port $port \
        --mem-fraction-static 0.95 \
        --max-running-requests 1024 \
        --max-total-tokens 131072 \
        --chunked-prefill-size 8192 \
        --attention-backend flashinfer \
        --sampling-backend flashinfer &
done
wait