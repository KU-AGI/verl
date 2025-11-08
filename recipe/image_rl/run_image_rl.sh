#!/usr/bin/env bash
set -xeuo pipefail

# Designate log path
LOG_DIR=${LOG_DIR:-"logs"}
mkdir -p "${LOG_DIR}"
SCRIPT_LOG="${LOG_DIR}/script_$(date +%Y%m%d_%H%M%S).log"

# Save log files
exec > >(tee -a "${SCRIPT_LOG}")
exec 2>&1

export NCCL_IB_GID_INDEX=0
export NCCL_CUDA_DEVICE_MAX_CONNECTIONS=8
export CUDA_DEVICE_MAX_CONNECTIONS=8
export NCCL_P2P_LEVEL="NVL"
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_TIMEOUT=300000
export NCCL_IB_TIMEOUT=300

# export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export RM_MODEL_PATH="Qwen/Qwen3-VL-30B-A3B-Instruct" # OpenGVLab/InternVL3_5-38B

GPUS=4 # `nvidia-smi -L | wc -l`
MODEL_PATH=/data/mllm/ckpt/pretrained # /data/mllm/checkpoints/Janus-Pro-7B
TRAIN_FILES=/data/mllm/data/train.parquet
VAL_FILES=/data/mllm/data/val.parquet
RUN_NAME=naive_qwen_vlm
PROJ_NAME=mllm_reasoning
SAVE_DIR=/data/verl/ckpts/$PROJ_NAME/$RUN_NAME

# pip install attrdict timm
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

# Parameters
train_prompt_bsz=4
# val_prompt_bsz=8
n_resp_per_prompt=8
train_prompt_mini_bsz=4

# Perf
fsdp_size=4

max_prompt_length=$((1024 * 1)) # 1k
max_response_length=$((1024 * 2)) # 2k

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.image_rl.main_image_generation_rl \
    algorithm.adv_estimator=grpo_task_skip \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.shuffle=False \
    data.prompt_key=prompt \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.truncation='left' \
    data.custom_cls.path=recipe/image_rl/image_rl_dataset.py \
    data.custom_cls.name=ImageRLDataset \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=-0.00 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=image_unified \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.cfg_weight=5.0 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.image_token_num_per_image=576 \
    actor_rollout_ref.rollout.prompt_length=1000 \
    actor_rollout_ref.rollout.response_length=2500 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +algorithm.max_token_start=-1 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.filter_groups.enable=False \
    algorithm.filter_groups.max_num_gen_batches=8 \
    +trainer.start_step=0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=False \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.rollout_data_dir="$SAVE_DIR/rollout" \
    trainer.rollout_freq=20 \
    trainer.log_val_generations=20 \
    reward_model.reward_manager=image_generation \
    custom_reward_function.path=recipe/image_rl/reward_function.py \
    custom_reward_function.name=compute_score_batch \
    # reward_model.reward_kwargs.img_saving.num=8 \