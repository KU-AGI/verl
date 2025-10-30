rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

exp_name='fully_async_debug'
project_name="verl-dapo" # 'DAPO'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

enable_filter_groups=False
# filter_groups_metric=acc
filter_groups_metric=seq_final_reward
max_num_gen_batches=0

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=8
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*400)))
test_freq=30
staleness_threshold=0.0
trigger_parameter_sync_step=16
partial_rollout=False


# NNODES_TRAIN=${NNODES_TRAIN:-1}
# NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
# N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}


# Paths
HOME="/data" ##
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
# MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/ReactionReasoner_stage12_lora_adapter_merged"} ##
# MODEL_PATH="/data/llm-reaction-reasoning/all_checkpoints/answeronly_fullft_8b/best.ckpt/hf_model"
MODEL_PATH="/data/llm-reaction-reasoning/all_checkpoints/reflection_v4_fullft_all/best.ckpt/hf_model"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/chem_dapo/syntheticreact_train.parquet"}
VAL_FILE=${VAL_FILE:-"${RAY_DATA_HOME}/data/chem_dapo/syntheticreact_val.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/chem_dapo/syntheticreact_test.parquet"}



python -m recipe.fully_async_policy.fully_async_main \
	data.train_batch_size=${train_prompt_bsz} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    data.task_extra_info_key=task \
    data.validation_shuffle=False \
    data.test_shuffle=False \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=300 \
    data.max_response_length=1700 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=4 \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.partial_rollout="${partial_rollout}" \
    critic.strategy=fsdp2 \
    # critic.ppo_micro_batch_size_per_gpu=16 \
    # critic.model.path="${MODEL_PATH}" \
