#!/usr/bin/env bash
set -xeuo pipefail

export WANDB_ENTITY="llm-reaction-reasoning"
export WANDB_PROJECT="verl-dapo"
export NCCL_DEBUG="WARN"

project_name='verl-dapo'
exp_name='reagent_steprwd_naivecredit_naivespl_GRPO_temp1.2'

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"/verl/recipe/fully_async_policy/shell/runtime_env.yaml"}
# Paths
HOME="/data"
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
# very important! please modify the max_position_embeddings in config.json to 32768 after downloading from huggingface
MODEL_PATH="/data/llm-reaction-reasoning/all_checkpoints/reflection_v4_fullft_all/best.ckpt/hf_model"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
DUMP_DIR=${DUMP_DIR:-"${RAY_DATA_HOME}/dumps/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/chem_dapo/syntheticreact_reagent_train.parquet"}
VAL_FILE=${VAL_FILE:-"${RAY_DATA_HOME}/data/chem_dapo/syntheticreact_val.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/chem_dapo/syntheticreact_test.parquet"}

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# Algorithm parameters
adv_estimator=grpo # stepwise_grpo, stepcumul_grpo, grpo
loss_mode=vanilla # steplevel, stepcumul, vanilla
norm_adv_by_std_in_grpo=True # False for Dr.GRPO, True for standard GRPO

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.000

clip_ratio_low=0.2
clip_ratio_high=0.2

enable_filter_groups=False
# filter_groups_metric=acc
filter_groups_metric=seq_final_reward
max_num_gen_batches=0

balance_task=False
use_response_mask_to_reflection_step=False

# Reward related parameters
use_content_reward=True
use_decision_reward=True
use_reflection_bonus=True
reflection_bonus_weight=0.3

# Response length parameters
max_prompt_length=500 # $((1024 * 2))
max_response_length=2100 # $((1024 * 8))
enable_overlong_buffer=False
overlong_buffer_len=0 # $((1024 * 4))
overlong_penalty_factor=1.0

# Training parameters
loss_agg_mode="seq-mean-token-sum" # "seq-mean-token-sum" "seq-mean-token-sum-norm"

# Algorithm
temperature=1.2
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_temperature=0.0
val_top_k=0.0
val_top_p=1.0
rollout_strategy="naive_sampling" # "naive_sampling" | "reflection_sampling"
strategy_ratio=0.0 # 1.0 means all use above rollout_strategy, 0.0 means all use naive_sampling

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
ref_offload=False
actor_offload=False
gen_tp=1
sp_size=1
fsdp_size=4 # Must be divisible by (n_gpus_training*n_nodes) and (n_gpus_rollout*n_nodes)

# Fully async specific parameters
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

n_gpus_rollout=4
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))
# (train_prompt_mini_bsz * require_batches * n_resp_per_prompt) % total_trainer_gpus == 0 must be satisfied
train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=8
train_prompt_mini_bsz=16
total_rollout_steps=$(((512*100000)))
test_freq=1
staleness_threshold=0.0
trigger_parameter_sync_step=100
require_batches=3
partial_rollout=False
save_freq=$((test_freq * trigger_parameter_sync_step * 6))


# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --working-dir "${WORKING_DIR}" \
#     -- python -m recipe.fully_async_policy.fully_async_main \
python -m recipe.fully_async_policy.fully_async_main \
    --config-name="fully_async_ppo_trainer.yaml" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.test_files="${TEST_FILE}" \
    data.task_extra_info_key=task \
    data.validation_shuffle=False \
    data.test_shuffle=False \
    data.prompt_key=prompt \
    data.truncation='left' \
    +data.balance_task=${balance_task} \
    +data.use_response_mask_to_reflection_step=${use_response_mask_to_reflection_step} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_nonanswered.enable=True \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.rollout_is_threshold=2.0 \
    algorithm.rollout_is=True \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    +reward_model.reward_kwargs.use_content_reward=${use_content_reward} \
    +reward_model.reward_kwargs.use_decision_reward=${use_decision_reward} \
    +reward_model.reward_kwargs.use_reflection_bonus=${use_reflection_bonus} \
    +reward_model.reward_kwargs.reflection_bonus_weight=${reflection_bonus_weight} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.validation_data_dir="${DUMP_DIR}/val" \
    trainer.test_data_dir="${DUMP_DIR}/test" \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    +rollout.strategy=${rollout_strategy} \
    +rollout.strategy_ratio=${strategy_ratio} \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    trainer.save_freq="${save_freq}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.total_epochs=10 \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_rollout_log_probs=True