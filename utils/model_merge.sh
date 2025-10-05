# !/bin/bash

# Reference:
# https://verl.readthedocs.io/en/latest/advance/checkpoint.html

# FSDP
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /data/verl/ckpts/DAPO/DAPO-ReactionReasoner-lora-merged-realtest/global_step_42/actor \
    --target_dir /data/verl/ckpts/DAPO/DAPO-ReactionReasoner-lora-merged-realtest/hf_model

# Megatron
python -m verl.model_merger merge \
    --backend megatron \
    --tie-word-embedding \
    --local_dir /data/verl/ckpts/verl-dapo/DAPO-ReactionReasoner-reflection-v4-fullft-all-n8-bsz32-clip-low-sampling-megatron-tp2-pp4-reward-only-answer/global_step_800/actor \
    --target_dir /data/verl/ckpts/verl-dapo/DAPO-ReactionReasoner-reflection-v4-fullft-all-n8-bsz32-clip-low-sampling-megatron-tp2-pp4-reward-only-answer/hf_model