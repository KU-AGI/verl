python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /data/verl/ckpts/DAPO/DAPO-ReactionReasoner-lora-merged-realtest/global_step_42/actor \
    --target_dir /data/verl/ckpts/DAPO/DAPO-ReactionReasoner-lora-merged-realtest/hf_model