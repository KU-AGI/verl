#!/usr/bin/env bash
set -euo pipefail

# Designate log path
LOG_DIR=${LOG_DIR:-"logs"}
mkdir -p "${LOG_DIR}"
SCRIPT_LOG="${LOG_DIR}/script_$(date +%Y%m%d_%H%M%S).log"

# Save log files
exec > >(tee -a "${SCRIPT_LOG}")
exec 2>&1

project_name='mllm_reasoning'
exp_name='0105_rollout_freeze_overfit'

export NCCL_IB_GID_INDEX=0
export NCCL_CUDA_DEVICE_MAX_CONNECTIONS=8
export CUDA_DEVICE_MAX_CONNECTIONS=8
export NCCL_P2P_LEVEL="NVL"
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_TIMEOUT=300000
export NCCL_IB_TIMEOUT=300

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/recipe/fully_async_policy_image_rl/shell/runtime_env.yaml"}

# Paths
HOME="/data"
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=/data/mllm/ckpt/step=014000.ckpt/hf_model # /data/mllm/checkpoints/Janus-Pro-7B
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILES=/data/verl/ckpts/mllm_reasoning/0105_rollout_freeze_dump/train_packed_png
VAL_FILES=/data/mllm/data/train_subset.parquet

export RM_VLM_MODEL_PATH="Qwen/Qwen3-VL-30B-A3B-Instruct" # OpenGVLab/InternVL3_5-38B
export RM_LLM_MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507" # OpenGVLab/InternVL3_5-38B

rollout_name=image_unified
rollout_mode=sync

# Algorithm parameters
adv_estimator=grpo_task_skip

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.04
entropy_coeff=0.00

clip_ratio_low=0.2
clip_ratio_high=0.2

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10

norm_adv_by_std_in_grpo=True

# Response length parameters
max_prompt_length=1000
max_response_length=2800
enable_overlong_buffer=True # temporary
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

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
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
ref_offload=False
actor_offload=False
gen_tp=1
sp_size=1

# Fully async specific parameters
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# n_gpus_rollout=6
# n_gpus_training=2 # $((NGPUS_PER_NODE - n_gpus_rollout))
n_gpus_training=8

fsdp_size=8 # Must be divisible by (n_gpus_training*n_nodes) and (n_gpus_rollout*n_nodes)

# https://verl.readthedocs.io/en/latest/advance/fully_async.html#parameter-description
train_prompt_bsz=128
# gen_prompt_bsz=1 # streaming generation, set to 1
val_prompt_bsz=16
n_resp_per_prompt=8
train_prompt_mini_bsz=128
train_prompt_micro_bsz=128
log_prob_micro_batch_size_per_gpu=16

test_freq=10
save_freq=100
total_epochs=10
# total_training_steps=3000
rollout_freq=10
# log_val_generations=1

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python -m recipe.fully_async_policy_image_rl.main_image_generation_rl \
    --config-name="fully_async_ppo_trainer.yaml" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.shuffle=False \
    data.prompt_key=prompt \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.custom_cls.path=recipe/image_rl/image_rl_dataset_dump.py \
    data.custom_cls.name=ImageRLDataset \
    actor_rollout_ref.nccl_timeout=120000000 \
    actor_rollout_ref.model.path=\"${MODEL_PATH}\" \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_prompt_micro_bsz} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=['LlamaDecoderLayer'] \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.cfg_weight=5.0 \
    actor_rollout_ref.rollout.image_token_num_per_image=576 \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.val_txt_top_k=${val_txt_top_k} \
    actor_rollout_ref.rollout.val_kwargs.val_txt_top_p=${val_txt_top_p} \
    actor_rollout_ref.rollout.val_kwargs.val_img_top_k=${val_img_top_k} \
    actor_rollout_ref.rollout.val_kwargs.val_img_top_p=${val_img_top_p} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=${ref_offload} \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=['LlamaDecoderLayer'] \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=8 \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.balance_batch=True \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq="${save_freq}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.resume_mode=auto \
    trainer.default_local_dir=$CKPTS_DIR \
    trainer.rollout_data_dir="$CKPTS_DIR/rollout" \
    trainer.rollout_freq=${rollout_freq} \
    trainer.validation_data_dir="$CKPTS_DIR/validation" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.total_epochs="${total_epochs}" \
    trainer.test_freq="${test_freq}" \
    reward_model.reward_manager=image_generation \
    custom_reward_function.path=recipe/image_rl/reward_function_naive.py \
    custom_reward_function.name=compute_score_batch \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \