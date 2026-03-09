#!/usr/bin/env bash
#SBATCH --job-name=rl-chemistry-agent-gpt-oss
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -xeuo pipefail

# ================= cluster topology =================
export GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${GPUS_PER_NODE:-8}}  # GPUs on this node
NNODES=${SLURM_JOB_NUM_NODES:-${NNODES:-1}}
export NNODES
export RAY_NUM_NODES=$NNODES

# Require at least 2 GPUs
TOTAL_GPUS=$((GPUS_PER_NODE * NNODES))
if [ "$TOTAL_GPUS" -lt 2 ]; then
  echo "Error: at least 2 GPUs are required, detected $TOTAL_GPUS." >&2
  exit 1
fi

echo "Using $NNODES nodes and $GPUS_PER_NODE GPUs per node..."

# ================= data/model/tool =================
HDFS_ROOT=/data/verl
DATA_ROOT=/data/verl

# Prefer local model if present, otherwise fall back to HF hub path
model_path=$DATA_ROOT/models/gpt-oss-20b-bf16
if [ ! -d "$model_path" ]; then
  model_path=lmsys/gpt-oss-20b-bf16
fi

# Use the default output directory produced by create_dataset.py
train_files=${train_files:-$DATA_ROOT/data/chemistry_agent/train_generation.parquet}
test_files=${test_files:-$DATA_ROOT/data/chemistry_agent/test_generation.parquet}

# Chemistry tools path
export CHEMISTRY_TOOLS_PATH=${CHEMISTRY_TOOLS_PATH:-/data/users/pimang62/chemistry_tool_agent_clean/chemagent/tools}
export TOOL_SERVER_BASE_URL=${TOOL_SERVER_BASE_URL:-http://localhost}

# Tool execution mode: "true" = HTTP tool servers (requires start_tool_servers.sh)
#                     "false" = run tools in-process (requires chemagent conda env)
export TOOLS_USE_HTTP=${TOOLS_USE_HTTP:-false}

# config
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
CONFIG_NAME=chemagent_trainer

# Agent config
agent_loop_config_path=recipe/langgraph_chemagent/config/chemagent.yaml

# Reward function
reward_fn_path=recipe/langgraph_chemagent/chemagent_reward.py
ANALYZE_MOLECULE_PORT=${ANALYZE_MOLECULE_PORT:-9000}
FUNC_GROUP_PORT=${FUNC_GROUP_PORT:-9008}
tool_weight=${tool_weight:-0.0}
turn_consistency_weight=${turn_consistency_weight:-0.1}
judge_url=${judge_url:-"http://localhost:8000/v1/chat/completions"}
judge_model=${judge_model:-"judge"}

# =================== wandb ===================
project_name=tool_integerated_rl
experiment_name=chemistry_agent_gpt_oss_20b_bf16

default_local_dir=$DATA_ROOT/ckpts/$project_name/$experiment_name

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Chemistry tasks may require more turns for complex reasoning
max_turns=12
max_prompt_length=16384
max_response_length=16384
actor_lr=1e-6

train_batch_size=128
ppo_mini_batch_size=8
n_resp_per_prompt=8
n_resp_per_prompt_val=1

# =================== logging ===================
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

# ================= performance =================
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5_0,mlx5_4,mlx5_5,mlx5_8   # Active 400Gbps NIC만 지정
export UCX_NET_DEVICES=mlx5_0:1,mlx5_4:1,mlx5_5:1,mlx5_8:1  # 위와 동일하게 매칭
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=6000

infer_tp=8  # sglang tensor parallel size (20B 모델에 맞게 조정) ### OOM
train_sp=8  # Ulysses sequence parallel size for actor
offload=true
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=2

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 1 ))

train_files="['$train_files']"
test_files="['$test_files']"

# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --working-dir "${WORKING_DIR}" \
#     --address "${RAY_ADDRESS}" \
#     --
python3 -m recipe.langgraph_chemagent.main_ppo \
    --config-path=./config \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=true \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.format=gpt-oss \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    actor_rollout_ref.rollout.agent.default_agent_loop=mol_generation \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=40960 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    reward.custom_reward_function.path=$reward_fn_path \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.analyze_molecule_port=$ANALYZE_MOLECULE_PORT \
    +reward.custom_reward_function.reward_kwargs.func_group_port=$FUNC_GROUP_PORT \
    +reward.custom_reward_function.reward_kwargs.use_http=$TOOLS_USE_HTTP \
    +reward.custom_reward_function.reward_kwargs.tool_weight=$tool_weight \
    +reward.custom_reward_function.reward_kwargs.turn_consistency_weight=$turn_consistency_weight \
    +reward.custom_reward_function.reward_kwargs.judge_url=$judge_url \
    +reward.custom_reward_function.reward_kwargs.judge_model=$judge_model \
    trainer.rollout_data_dir="$default_local_dir/rollout" \
    trainer.validation_data_dir="$default_local_dir/validation" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.val_before_train=true \
    trainer.log_val_generations=50 \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=-1 \
    trainer.default_local_dir="$default_local_dir" \
    trainer.test_freq=5 \
    trainer.total_epochs=1 "$@"
