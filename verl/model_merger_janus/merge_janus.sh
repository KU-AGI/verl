#!bin/bash

python -m verl.model_merger_janus merge \
    --backend fsdp \
    --local_dir /home/work/AGILAB/mllm_reasoning/verl/ckpts/mllm_reasoning/0407_KT_v5_fine_grained_replay_mean_std_filtering_cfg5_entropy_v2/global_step_100/actor \
    --target_dir /home/work/AGILAB/mllm_reasoning/verl/ckpts/mllm_reasoning/0407_KT_v5_fine_grained_replay_mean_std_filtering_cfg5_entropy_v2/global_step_100/hf