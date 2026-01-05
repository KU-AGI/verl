# !/bin/bash

# Reference:
# https://verl.readthedocs.io/en/latest/advance/checkpoint.html

# # FSDP
# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /data/verl/ckpts/verl-dapo/refl_bonus_0.3_2node/global_step_10200/actor \
#     --target_dir /data/verl/ckpts/verl-dapo/refl_bonus_0.3_2node/global_step_10200/hf_model

# # Megatron
# python -m verl.model_merger merge \
#     --backend megatron \
#     --local_dir /data/verl/ckpts/verl-dapo/step_reward/global_step_450/actor \
#     --target_dir /data/verl/ckpts/verl-dapo/step_reward/global_step_450/actor/hf_model

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /data/verl/ckpts/verl-dapo/retro_smallset_steprwd_v12/global_step_360/actor \
    --target_dir /data/verl/ckpts/verl-dapo/retro_smallset_steprwd_v12/global_step_360/hf_model