#!/bin/bash

GDINO_MODEL_PATH=/home/work/AGILAB/mllm_reasoning/data/checkpoints/mm_grounding_dino_large_all

# CUDA_VISIBLE_DEVICES=4 python recipe/image_rl/detector.py \
#     --gdino_ckpt_path $GDINO_MODEL_PATH \
#     --host 0.0.0.0 \
#     --port 8084 &

# CUDA_VISIBLE_DEVICES=5 python recipe/image_rl/detector.py \
#     --gdino_ckpt_path $GDINO_MODEL_PATH \
#     --host 0.0.0.0 \
#     --port 8087

for i in 6 7; do
    port=$((8080 + i))
    CUDA_VISIBLE_DEVICES=$i python recipe/image_rl/detector.py \
        --gdino_ckpt_path $GDINO_MODEL_PATH \
        --host 0.0.0.0 \
        --port $port &
done
wait