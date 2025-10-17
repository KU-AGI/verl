#!/usr/bin/env bash
set -xeuo pipefail

# Designate log path
LOG_DIR=${LOG_DIR:-"logs"}
mkdir -p "${LOG_DIR}"
SCRIPT_LOG="${LOG_DIR}/script_$(date +%Y%m%d_%H%M%S).log"

# Save log files
exec > >(tee -a "${SCRIPT_LOG}")
exec 2>&1

# export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3

GPUS=4 # `nvidia-smi -L | wc -l`
MODEL_PATH=deepseek-community/Janus-Pro-7B
RM_MODEL_PATH=OpenGVLab/InternVL3_5-38B
RUN_NAME=debug
PROJ_NAME=mllm_reasoning
SAVE_DIR=/data/verl/ckpts/$PROJ_NAME/$RUN_NAME

# export HYDRA_FULL_ERROR=1
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_CUMEM_ENABLE=0
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

# Parameters
train_prompt_bsz=4
val_prompt_bsz=4
n_resp_per_prompt=2
train_prompt_mini_bsz=4

max_prompt_length=$((1024 * 1)) # 1k
max_response_length=$((1024 * 1)) # 1k

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.image_rl.main_image_generation_rl \
    algorithm.adv_estimator=grpo \
    data.train_files="/data/mllm/data/train.parquet" \
    data.val_files="/data/mllm/data/val.parquet" \
    data.prompt_key=prompt \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=-0.00 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=100000000 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=image_unified \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.cfg_weight=5.0 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.feedback_system_prompt="You should give me a feedback on the image generation." \
    actor_rollout_ref.rollout.refine_system_prompt="You should refine the image generation." \
    actor_rollout_ref.rollout.saving=True \
    actor_rollout_ref.rollout.save_dir="/verl/output/rollout" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +algorithm.max_token_start=-1 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=8 \
    +trainer.start_step=0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=False \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$SAVE_DIR \
    reward_model.reward_manager=image_generation \
    custom_reward_function.path=recipe/image_rl/reward_function.py \
    custom_reward_function.name=compute_score \
    reward_model.reward_kwargs.img_saving.save_freq=1 \
    reward_model.reward_kwargs.img_saving.num=4 \
    reward_model.reward_kwargs.img_saving.path=/data/verl/$PROJ_NAME/$RUN_NAME/rollout \
    reward_model.reward_kwargs.img_saving.experiment_name=$RUN_NAME
