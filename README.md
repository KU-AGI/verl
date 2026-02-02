# Environment Setup
```bash
make init-container
```


# Data Preprocessing
```bash
python examples/data_preprocess/chem_dataset.py --train
python examples/data_preprocess/chem_dataset.py --val
python examples/data_preprocess/chem_dataset.py --test
```


# Run Experiments
### 1. Run Round-trip vLLM Server
```bash
make start-roundtrip-servers
```


### 2. Run Experiments Scripts
```bash
# Prediction-Only (RL)
sh recipe/fully_async_policy/shell/pdonly_retro_fullset_v14_wo_steprwd_w_rndtrp.sh

# RetroReasoner (RL)
sh recipe/fully_async_policy/shell/retro_fullset_v14_wo_steprwd_w_rndtrp.sh

# RetroReasoner (RL, w/ R^{exact})
sh recipe/fully_async_policy/shell/retro_fullset_v14_wo_steprwd.sh
```

### 3. Convert FSDP/Megatron to HuggingFace Model
```
# Replace 'local_dir', 'target_dir'
sh utils/model_merge.sh
```



# Core Codes
### Dataset
`examples/data_preprocess/chem_dataset.py`

### Evaluation
(_validate, _test)
`verl/trainer/ppo/ray_trainer.py`

### GRPO Loss
`verl/trainer/ppo/core_algos.py`

### Trainer
`recipe/fully_async_policy/fully_async_trainer.py`

### Rollouter
`recipe/fully_async_policy/fully_async_rollouter.py`

### Manager
`recipe/fully_async_policy/fully_async_main.py`

### Reward Calculation
`verl/utils/reward_score/chem_dapo_stepwise.py`

