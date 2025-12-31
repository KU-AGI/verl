#!/bin/bash

for i in 6 7; do
    port=$((8000 + i))
    CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
        --model-path /home/work/AGILAB/mllm_reasoning/data/checkpoints/Qwen3-VL-30B-A3B-Instruct \
        --served-model-name Qwen/Qwen3-VL-30B-A3B-Instruct \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port $port \
        --mem-fraction-static 0.90 \
        --max-running-requests 512 \
        --max-total-tokens 65536 \
        --chunked-prefill-size 4096 \
        --attention-backend flashinfer \
        --sampling-backend flashinfer &
done
wait