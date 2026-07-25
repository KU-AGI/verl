#!/usr/bin/env bash
set -euo pipefail

###############################################################################
#                              LOGGING SETUP
###############################################################################
LOG_DIR=${LOG_DIR:-"logs"}
mkdir -p "${LOG_DIR}"
SCRIPT_LOG="${LOG_DIR}/script_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${SCRIPT_LOG}")
exec 2>&1

###############################################################################
#                         EXPERIMENT CONFIGURATION
###############################################################################
project_name='mllm_reasoning'
exp_name="${EXP_NAME:-0718_KT_v5_1e_6_multi_step_neurips_sglang_fixed_v3}"
# exp_name="0423_debug"
task_ids='[1,2,3]'

###############################################################################
#                           ENVIRONMENT VARIABLES
###############################################################################
# NCCL Settings
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_DISABLE=1
export GLOO_SOCKET_IFNAME="eth0"
export NCCL_SOCKET_TIMEOUT=300000
export NCCL_IB_TIMEOUT=300000

# CUDA & Other Settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

###############################################################################
#                               PATH SETTINGS
###############################################################################
# Ray Configuration
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/recipe/fully_async_policy_image_rl/shell/runtime_env_kt_sglang.yaml"}
export PYTHONPATH="${WORKING_DIR}/sglang/python:${WORKING_DIR}:${PYTHONPATH:-}"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# Model & Checkpoint Paths
MLLM_ROOT="/home/work/AGILAB/mllm_reasoning"
RAY_DATA_HOME=${RAY_DATA_HOME:-"${MLLM_ROOT}/verl"}
MODEL_PATH="/home/work/AGILAB/mllm_reasoning/data/experiments/ckpt/janus_sft/1223_v10_sft_warmup_constant_long_prompt/version_1/step=014000.ckpt/hf_model"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# Dataset Paths
TRAIN_FILES='[/home/work/AGILAB/mllm_reasoning/pimang62/data/train_v5_wo_focusdiff_aug.parquet,/home/work/AGILAB/mllm_reasoning/pimang62/data/train_sft_data_v5.parquet,/home/work/AGILAB/mllm_reasoning/pimang62/data/train_ospo_v5.parquet]'
VAL_FILES='[/home/work/AGILAB/mllm_reasoning/pimang62/data/val_v5.parquet,/home/work/AGILAB/mllm_reasoning/pimang62/data/val_benchmark_v5.parquet,/home/work/AGILAB/mllm_reasoning/pimang62/data/longalign_conpair_v5.parquet]'

# Reward Model Paths
rm_vlm_model_path="Qwen/Qwen3.5-35B-A3B"
rm_llm_model_path="Qwen/Qwen3-30B-A3B-Instruct-2507"

###############################################################################
#                          DISTRIBUTED TRAINING
###############################################################################
# Node & GPU Configuration
# NNODES=${NNODES:-1}
# NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
ROLLOUT_NNODES=2
TRAIN_NNODES=1
n_gpus_rollout=5
n_gpus_training=8

# FSDP & Parallelism
fsdp_size=8  # Must be divisible by (n_gpus_training*n_nodes) and (n_gpus_rollout*n_nodes)
gen_tp=1
sp_size=1

# Offloading
ref_offload=False
actor_offload=False

###############################################################################
#                           ALGORITHM PARAMETERS
###############################################################################
# Core Algorithm
adv_estimator=grpo
rollout_name=janus_sglang
rollout_mode=async

###############################################################################
#                          MULTI-TURN ORCHESTRATION
###############################################################################
# The rollout runs `task1 -> (task2 -> task3) x max_turns` with early-termination
# when every sample's task2 emits "No need to feedback". Each turn's task2/task3
# consumes the rolling `current_*` image (task1 output on turn 0, previous turn's
# task3 regen thereafter). Orchestrator returns a per-task dict
# {1: task1_dp, 2: task2_dp, 3: task3_dp}; task2/task3 DPs hold concatenated
# per-turn rollouts, task1 DP holds unique rollouts.
max_turns=2

###############################################################################
#                              MDP REWARD
###############################################################################
# Return discount used when backing up phase1 step rewards:
#   G_h = r_h + mdp_gamma * G_{h+1}
mdp_reward_version=multi_step # gae | multi_step
mdp_gamma=0.6

# Initial image reward weight:
#   r_step1 = mdp_init_reward_weight * S_1
mdp_init_reward_weight=1.2


# Task2 reasoning reward shaping:
#   r_reason = (judge_score / 2 - 1) / 3
#
# Task2 segment advantage:
#   A_step{k} = GRPO(G_step{k}) + task2_local_adv_weight * GRPO(R_step{k})
# where R_step2/3/4 are raw stage judge rewards.
task2_local_adv_weight=0.0

# Task3 edit reward:
#   r_step5 = S_next - S_prev + mdp_edit_if_weight * edit_if * 1[S_next > S_prev] - mdp_edit_cost
mdp_edit_if_weight=0.1
mdp_edit_cost=0.05

# Legacy outcome formula hyperparameters, kept only for reports/debug fields.
# eta   : mean edit instruction-following weight
# beta  : process-quality weight
# lambda: per-extra-turn cost penalty
outcome_eta=0.1
outcome_beta=0.1
outcome_lambda=0.12

# KL Divergence
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# PPO Clipping
clip_ratio_low=0.2
clip_ratio_high=0.28
entropy_coeff=0.0

# Group Filtering
enable_filter_groups=True
filter_groups_metric=reward
norm_adv_by_std_in_grpo=True

# Adaptive Entropy Coefficient (per-task)
adaptive_entropy_coeff_enable=True
adaptive_entropy_coeff_task1_target_entropy=4.5
adaptive_entropy_coeff_task2_target_entropy=0.20
adaptive_entropy_coeff_task3_target_entropy=4.5

###############################################################################
#                          SEQUENCE LENGTH SETTINGS
###############################################################################
max_prompt_length=2048
task3_max_prompt_length="${TASK3_MAX_PROMPT_LENGTH:-3072}"
max_response_length=2800
task3_compact_logits="${TASK3_COMPACT_LOGITS:-True}"

# Overlong Buffer Configuration
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

###############################################################################
#                          SAMPLING PARAMETERS
###############################################################################
# Training Sampling
cfg_weight=2.0
temperature=1.0
txt_top_k=0   # 0 for no top_k filtering
txt_top_p=1.0
img_top_k=0   # 0 for no top_k filtering
img_top_p=1.0

# Validation Sampling
val_cfg_weight=5.0
val_temperature=1.0
val_txt_top_k=0
val_txt_top_p=1.0
val_img_top_k=0
val_img_top_p=1.0

###############################################################################
#                            BATCH SIZE SETTINGS
###############################################################################
# Prompt Batch Sizes
train_prompt_bsz=0            # not used in async mode
gen_prompt_bsz=1              # streaming generation, set to 1
n_resp_per_prompt="${N_RESP_PER_PROMPT:-16}"
rollout_prompt_size=1         # prompts per actor per batch (async mode)
val_rollout_prompt_size="${VAL_ROLLOUT_PROMPT_SIZE:-32}"

# Response & Micro Batch
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-8}" # prompt groups per global PPO mini-batch
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-16}" # flattened samples per rank
log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}"
require_batches="${REQUIRE_BATCHES:-2}"

# Dynamic Batching
use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))

# Loss Aggregation
loss_agg_mode="token-mean"

###############################################################################
#                          OPTIMIZER SETTINGS
###############################################################################
lr=1e-6
lr_scheduler_type=constant
lr_warmup_steps=10
weight_decay=0.01

###############################################################################
#                        ASYNC TRAINING PARAMETERS
###############################################################################
# https://verl.readthedocs.io/en/latest/advance/fully_async.html#parameter-description
total_rollout_steps="${TOTAL_ROLLOUT_STEPS:-1536000}"
staleness_threshold=2.0
trigger_parameter_sync_step=1
partial_rollout=False
use_rollout_log_probs=True
compute_prox_log_prob=False
max_regen_retries=3
reward_finalize_workers=2

# Replay Buffer
replay_buffer_enable=True
replay_buffer_max_version_gap=-1
replay_buffer_max_size_per_task=64
replay_buffer_max_use_count=-1
replay_buffer_filter_mode=max_and_std_constant      # "mean", "std", or "max", "max_and_std_constant"
replay_buffer_score_threshold_1=0.7
replay_buffer_score_threshold_2=1.0
replay_buffer_score_threshold_3=0.5 # 2점 만점
replay_buffer_score_std_threshold_1=0.1
replay_buffer_score_std_threshold_2=0.1
replay_buffer_score_std_threshold_3=0.1
replay_buffer_reward_history_size=100
replay_buffer_max_quantile=0.75
replay_buffer_std_quantile=0.50

###############################################################################
#                        ROLLOUT CORRECTION
###############################################################################
rollout_is=token          # null (disabled), "token", "sequence"
rollout_is_threshold=2.0
bypass_mode=false

###############################################################################
#                         TRAINING SCHEDULE
###############################################################################
total_epochs="${TOTAL_EPOCHS:-10}"
test_freq="${TEST_FREQ:-250}"
save_freq="${SAVE_FREQ:-500}"
rollout_freq="${ROLLOUT_FREQ:-250}"
trainer_logger="${TRAINER_LOGGER:-['console','wandb']}"
resume_mode="${RESUME_MODE:-auto}"
val_before_train="${VAL_BEFORE_TRAIN:-True}"
# total_training_steps=3000
# log_val_generations=20

ray job submit --address="${RAY_ADDRESS}" --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python -m recipe.fully_async_policy_image_rl.fully_async_main \
    --config-name="fully_async_ppo_trainer.yaml" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.shuffle=True \
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
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=${lr_scheduler_type} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.task3_compact_logits=${task3_compact_logits} \
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
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    +algorithm.mdp_reward_version=${mdp_reward_version} \
    +algorithm.mdp_gamma=${mdp_gamma} \
    +algorithm.mdp_init_reward_weight=${mdp_init_reward_weight} \
    +algorithm.task2_local_adv_weight=${task2_local_adv_weight} \
    +algorithm.mdp_edit_if_weight=${mdp_edit_if_weight} \
    +algorithm.mdp_edit_cost=${mdp_edit_cost} \
    +algorithm.outcome_eta=${outcome_eta} \
    +algorithm.outcome_beta=${outcome_beta} \
    +algorithm.outcome_lambda=${outcome_lambda} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=float32 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=['LlamaDecoderLayer'] \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.cfg_weight=${cfg_weight} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.txt_top_k=${txt_top_k} \
    actor_rollout_ref.rollout.txt_top_p=${txt_top_p} \
    actor_rollout_ref.rollout.img_top_k=${img_top_k} \
    actor_rollout_ref.rollout.img_top_p=${img_top_p} \
    actor_rollout_ref.rollout.image_token_num_per_image=576 \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.task3_prompt_length=${task3_max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend=pytorch \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chat_template=janus-pro \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_janus_image_cuda_graph=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.val_cfg_weight=${val_cfg_weight} \
    actor_rollout_ref.rollout.val_kwargs.val_temperature=${val_temperature} \
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
    trainer.critic_warmup=0 \
    trainer.logger="${trainer_logger}" \
    trainer.val_before_train=${val_before_train} \
    trainer.balance_batch=False \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq="${save_freq}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.resume_mode="${resume_mode}" \
    trainer.default_local_dir=$CKPTS_DIR \
    trainer.rollout_data_dir="$CKPTS_DIR/rollout" \
    trainer.rollout_freq=${rollout_freq} \
    trainer.validation_data_dir="$CKPTS_DIR/validation" \
    trainer.nnodes="${TRAIN_NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${ROLLOUT_NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    actor_rollout_ref.rollout.agent.num_workers=$((ROLLOUT_NNODES * n_gpus_rollout)) \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.total_epochs=${total_epochs} \
    trainer.test_freq="${test_freq}" \
    rollout.test_freq="${test_freq}" \
    async_training.rollout_prompt_size="${rollout_prompt_size}" \
    async_training.val_rollout_prompt_size="${val_rollout_prompt_size}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_rollout_log_probs=${use_rollout_log_probs} \
    async_training.compute_prox_log_prob=${compute_prox_log_prob} \
    async_training.max_regen_retries=${max_regen_retries} \
    async_training.reward_finalize_workers=${reward_finalize_workers} \
    async_training.replay_buffer.enable=${replay_buffer_enable} \
    async_training.replay_buffer.max_version_gap=${replay_buffer_max_version_gap} \
    async_training.replay_buffer.max_size_per_task=${replay_buffer_max_size_per_task} \
    async_training.replay_buffer.max_use_count=${replay_buffer_max_use_count} \
    async_training.replay_buffer.filter_mode=${replay_buffer_filter_mode} \
    +async_training.replay_buffer.score_thresholds.1=${replay_buffer_score_threshold_1} \
    +async_training.replay_buffer.score_thresholds.2=${replay_buffer_score_threshold_2} \
    +async_training.replay_buffer.score_thresholds.3=${replay_buffer_score_threshold_3} \
    +async_training.replay_buffer.score_std_thresholds.1=${replay_buffer_score_std_threshold_1} \
    +async_training.replay_buffer.score_std_thresholds.2=${replay_buffer_score_std_threshold_2} \
    +async_training.replay_buffer.score_std_thresholds.3=${replay_buffer_score_std_threshold_3} \
    async_training.replay_buffer.reward_history_size=${replay_buffer_reward_history_size} \
    async_training.replay_buffer.max_quantile=${replay_buffer_max_quantile} \
    async_training.replay_buffer.std_quantile=${replay_buffer_std_quantile} \
    reward_model.reward_manager=image_generation \
    custom_reward_function.path=recipe/image_rl/reward_function_fine_grained.py \
    custom_reward_function.name=compute_score_batch \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    +reward_model.reward_kwargs.rm_vlm_model_path="${rm_vlm_model_path}" \
    +reward_model.reward_kwargs.rm_llm_model_path="${rm_llm_model_path}" \
    +actor_rollout_ref.actor.multi_task.enable=True \
    +actor_rollout_ref.actor.multi_task.task_ids="${task_ids}" \
    +actor_rollout_ref.actor.multi_task.task_weights='[0.3,0.3,0.3]' \
    actor_rollout_ref.actor.adaptive_entropy_coeff.enable=${adaptive_entropy_coeff_enable} \
    actor_rollout_ref.actor.adaptive_entropy_coeff.task1.target_entropy=${adaptive_entropy_coeff_task1_target_entropy} \
    actor_rollout_ref.actor.adaptive_entropy_coeff.task2.target_entropy=${adaptive_entropy_coeff_task2_target_entropy} \
    actor_rollout_ref.actor.adaptive_entropy_coeff.task3.target_entropy=${adaptive_entropy_coeff_task3_target_entropy} \
    +actor_rollout_ref.rollout.max_turns=${max_turns}
