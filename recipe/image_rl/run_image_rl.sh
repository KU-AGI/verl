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
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT_MS=1200000
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export RM_MODEL_PATH="Qwen/Qwen3-VL-30B-A3B-Instruct" # OpenGVLab/InternVL3_5-38B

# pip install attrdict timm
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/recipe/image_rl/runtime_env.yaml"}

GPUS=4 # `nvidia-smi -L | wc -l`
MODEL_PATH=/data/mllm/ckpt/step=016000.ckpt/hf_model # /data/mllm/checkpoints/Janus-Pro-7B
TRAIN_FILES=/data/mllm/data/train.parquet
VAL_FILES=/data/mllm/data/val.parquet
RUN_NAME=sglang_debug # grpo_default_reward_b8_n8_lr2e-6_multi_task_new_reward_kl0.04_temp1.2_sampling
PROJ_NAME=mllm_reasoning
SAVE_DIR=/data/verl/ckpts/$PROJ_NAME/$RUN_NAME

# Algorithm parameters
adv_estimator=grpo_task_skip

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.04

clip_ratio_low=0.2
clip_ratio_high=0.2

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10

norm_adv_by_std_in_grpo=False

# Response length parameters
max_prompt_length=1000
max_response_length=2800
# enable_overlong_buffer=False
# overlong_buffer_len=$((1024 * 4))
# overlong_penalty_factor=1.0

# Training parameters
loss_agg_mode="token-mean"

# Algorithm
temperature=1.2
# txt_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
# txt_top_p=1.0
# img_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
# img_top_p=1.0
val_temperature=1.0
val_txt_top_k=50
val_txt_top_p=1.0
val_img_top_k=4096
val_img_top_p=1.0

# Performance Related Parameter
use_dynamic_bsz=False
# actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
# infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=False
gen_tp=1
sp_size=1

# Fully async specific parameters
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-$GPUS}

save_freq=20
test_freq=20
total_epochs=1
rollout_freq=1
log_val_generations=20

# Parameters
train_prompt_bsz=$GPUS # 4
gen_prompt_bsz=$((train_prompt_bsz * 1))
n_resp_per_prompt=8
train_prompt_mini_bsz=$GPUS # 4

# Perf
fsdp_size=$GPUS # 4

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.image_rl.main_image_generation_rl \
    algorithm.adv_estimator=grpo_task_skip \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.shuffle=False \
    data.prompt_key=prompt \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.val_batch_size=${gen_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.custom_cls.path=recipe/image_rl/image_rl_dataset.py \
    data.custom_cls.name=ImageRLDataset \
    actor_rollout_ref.model.path=\"${MODEL_PATH}\" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=-0.00 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=['LlamaDecoderLayer'] \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=image_unified \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.cfg_weight=5.0 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.image_token_num_per_image=576 \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=['LlamaDecoderLayer'] \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.val_txt_top_k=${val_txt_top_k} \
    actor_rollout_ref.rollout.val_kwargs.val_txt_top_p=${val_txt_top_p} \
    actor_rollout_ref.rollout.val_kwargs.val_img_top_k=${val_img_top_k} \
    actor_rollout_ref.rollout.val_kwargs.val_img_top_p=${val_img_top_p} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=8 \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=False \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.resume_mode=auto \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.rollout_data_dir="$SAVE_DIR/rollout" \
    trainer.rollout_freq=${rollout_freq} \
    trainer.validation_data_dir="$SAVE_DIR/validation" \
    trainer.log_val_generations=${log_val_generations} \
    reward_model.reward_manager=image_generation \
    custom_reward_function.path=recipe/image_rl/reward_function.py \
    custom_reward_function.name=compute_score_batch \
    # reward_model.reward_kwargs.img_saving.num=8 \
