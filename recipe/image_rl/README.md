
# Core Code

## Load pretrained model 

- You need arguments for
    - `--base_model_path`: Janus baseline model path
    - `--finetuned_ckpt_dir`: SFT checkpoint path
    - `--save_path`: saving path
    ```python
    python -m utils.model_loader --base_model_path ... --finetuned_ckpt_dir ... --save_path
    ```

## Data Preprocess

- https://verl.readthedocs.io/en/latest/preparation/prepare_data.html

- For ours
    - "examples/data_process/mllm_dataset.py"
        - `prompt`: "recipe/image_rl/image_rl_dataset.py" 에서 처리
        - `reward_model`: "verl/workers/reward_manager/image_generation.py" reward_manager가 필요로 할 내용

## Data Loader

- Custom data loader
    ```bash
    data.custom_cls.path=recipe/image_rl/image_rl_dataset.py \
    data.custom_cls.name=ImageRLDataset \
    ```

## Rollouter

- Custom rollouter class
    - "verl/workers/rollout/image_unified_rollout.py"

## Reward function

- Custom reward function
    - call `get_custom_reward_fn` in "recipe/image_rl/reward.py"
    ```bash
    custom_reward_function.path=recipe/image_rl/reward_function.py \
    custom_reward_function.name=compute_score_batch \
    ```

- Reward manager
    - "verl/workers/reward_manager/image_generation.py"
        - dataset `data_source`에 정의된 이름("image_generation")으로 register

## Trainer worker
- Custom worker class
    - "recipe/image_rl/image_generation_worker.py"

- Custom algorithm function
    - "recipe/image_rl/core_algos.py"

- Custom metric logging
    - "recipe/image_rl/custom_metric_utils.py"

## Run examples

- Change the trainer module in "main_image_generation_rl.py" 

- GRPO
    - `bash recipe/image_rl/run_image_rl.sh`

- DAPO
    - `bash recipe/image_rl/dapo_run_image_rl.sh`

## Save pretrained model 

- FSDP & Megatron
    - `$1` for project name in saving dir
    - `$2` for target global step
    ```bash
    bash utils/model_merge_janus.sh $PROJECT_DIR $GLOBAL_STEP
    ```