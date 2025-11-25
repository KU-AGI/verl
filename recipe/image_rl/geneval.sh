#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export WORKSPACE=/verl

PROJECT_DIR=grpo_default_reward_b8_n8_lr2e-6_multi_task_new_reward_kl0.04
GLOBAL_STEP=global_step_60

bash utils/model_merge_janus.sh $PROJECT_DIR $GLOBAL_STEP

bash bench/geneval/generate.sh $PROJECT_DIR $GLOBAL_STEP

cd $WORKSPACE/bench/geneval; bash evaluate.sh $PROJECT_DIR $GLOBAL_STEP
cd $WORKSPACE
