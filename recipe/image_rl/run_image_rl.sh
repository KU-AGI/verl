set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS

GPUS=4 # `nvidia-smi -L | wc -l`
MODEL_PATH=deepseek-community/Janus-Pro-7B
RM_MODEL_PATH=OpenGVLab/InternVL3_5-38B
RUN_NAME=test
PROJ_NAME="verl_janus_test"
SAVE_DIR=/data/verl/ckpts/janus_rl/$PROJ_NAME/$RUN_NAME

export HYDRA_FULL_ERROR=1

# if [ "$RANK" -eq 0 ]; then
python3 -m recipe.image_rl.main_image_generation_rl \
    algorithm.adv_estimator=grpo \
    data.train_files="/data/mllm/data/train.parquet" \
    data.val_files="/data/mllm/data/val.parquet" \
    data.image_key=images \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=-0.00 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=100000000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=image_unified \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    +generation_mode=image \
    +feedback_system_prompt="You should give me a feedback on the image generation." \
    +refine_system_prompt="You should refine the image generation." \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=8 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$SAVE_DIR \
    reward_model.reward_manager=image_generation \
    custom_reward_function.path=recipe/image_rl/reward_function.py \
    custom_reward_function.name=compute_score \
    +reward_model.reward_kwargs.img_saving.save_freq=5 \
    +reward_model.reward_kwargs.img_saving.num=16 \
    +reward_model.reward_kwargs.img_saving.experiment_name=$RUN_NAME
