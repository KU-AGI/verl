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
exp_name='fully_async_kt'

# export NCCL_IB_GID_INDEX=0
# export NCCL_CUDA_DEVICE_MAX_CONNECTIONS=8
# export CUDA_DEVICE_MAX_CONNECTIONS=8
# export NCCL_P2P_LEVEL="NVL"
# export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_DISABLE=1
export GLOO_SOCKET_IFNAME="eth0"
export NCCL_SOCKET_TIMEOUT=300000
export NCCL_IB_TIMEOUT=300000
# export NCCL_IB_HCA=mlx5_0

# export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/recipe/fully_async_policy_image_rl/shell/runtime_env_kt.yaml"}

# Paths
HOME="/home/work/AGILAB"
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
# very important! please modify the max_position_embeddings in config.json to 32768 after downloading from huggingface
MODEL_PATH=/home/work/AGILAB/mllm_reasoning/data/experiments/ckpt/janus_sft/1204_v9_sft_constant_sch/version_0/step=012000.ckpt/hf_model # /data/mllm/checkpoints/Janus-Pro-7B
RM_MODEL_PATH=/home/work/AGILAB/mllm_reasoning/data/checkpoints/Qwen3-VL-30B-A3B-Instruct # OpenGVLab/InternVL3_5-38B
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILES=/home/work/AGILAB/mllm_reasoning/pimang62/data/train.parquet
VAL_FILES=/home/work/AGILAB/mllm_reasoning/pimang62/data/val.parquet

rollout_name=image_unified
rollout_mode=async

# Algorithm parameters
adv_estimator=grpo_task_skip

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

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
# NNODES=${NNODES:-1}
# NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# n_gpus_rollout=12
# n_gpus_training=8 # $((NGPUS_PER_NODE - n_gpus_rollout))

fsdp_size=4 # Must be divisible by (n_gpus_training*n_nodes) and (n_gpus_rollout*n_nodes)

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=8
rollouter_world_size=12
train_prompt_mini_bsz=128
train_prompt_micro_bsz=16
total_rollout_steps=$(((512*100)))
staleness_threshold=1.25
trigger_parameter_sync_step=2
require_batches=1
partial_rollout=False

test_freq=1
save_freq=$((test_freq * trigger_parameter_sync_step * 1))
total_epochs=1
rollout_freq=1
log_val_generations=20

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python -m recipe.fully_async_policy_image_rl.fully_async_main \
    --config-name="fully_async_ppo_trainer.yaml" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.shuffle=False \
    data.prompt_key=prompt \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.val_batch_size=${gen_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.custom_cls.path=recipe/image_rl/image_rl_dataset.py \
    data.custom_cls.name=ImageRLDataset \
    actor_rollout_ref.nccl_timeout=120000000 \
    actor_rollout_ref.model.path=\"${MODEL_PATH}\" \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
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
    actor_rollout_ref.actor.entropy_coeff=-0.00 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
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
    trainer.val_before_train=False \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq="${save_freq}" \
    trainer.total_epochs=${total_epochs} \
    trainer.resume_mode=auto \
    trainer.default_local_dir=$CKPTS_DIR \
    trainer.rollout_data_dir="$CKPTS_DIR/rollout" \
    trainer.rollout_freq=${rollout_freq} \
    trainer.validation_data_dir="$CKPTS_DIR/validation" \
    trainer.log_val_generations=${log_val_generations} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    rollout.nnodes=2 \
    rollout.n_gpus_per_node=6 \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.total_epochs=${total_epochs} \
    rollout.test_freq="${test_freq}" \
    async_training.rollouter_world_size="${rollouter_world_size}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_rollout_log_probs=False \
    reward_model.reward_manager=image_generation \
    custom_reward_function.path=recipe/image_rl/reward_function.py \
    custom_reward_function.name=compute_score_batch \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \