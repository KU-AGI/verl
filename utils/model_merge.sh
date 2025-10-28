# !/bin/bash

# Reference:
# https://verl.readthedocs.io/en/latest/advance/checkpoint.html

# # FSDP
# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /data/verl/ckpts/DAPO/DAPO-ReactionReasoner-lora-merged-realtest/global_step_42/actor \
#     --target_dir /data/verl/ckpts/DAPO/DAPO-ReactionReasoner-lora-merged-realtest/hf_model

# Megatron
python -m verl.model_merger merge \
    --backend megatron \
    --local_dir /data/verl/ckpts/verl-dapo/step_reward/global_step_450/actor \
    --target_dir /data/verl/ckpts/verl-dapo/step_reward/global_step_450/actor/hf_model