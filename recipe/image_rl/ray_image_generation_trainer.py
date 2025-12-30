# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Dict, Optional, Type

import numpy as np
import PIL
import ray
import torch
import wandb
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, RandomSampler, Sampler, SequentialSampler, Subset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from recipe.image_rl import core_algos
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from recipe.image_rl.core_algos import AdvantageEstimator, agg_loss
from recipe.image_rl.custom_metric_utils import ( # custom metric
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_group_reward_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import (
    RLHFDataset,
    collate_fn,
)
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.reward_manager.abstract import AbstractRewardManager

from recipe.image_rl.reward import compute_reward, compute_reward_async
from recipe.image_rl.tracking import ValidationGenerationsLogger
from recipe.image_rl.utils import FormattingEvaluator

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", task_id: int = 1):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch[f"task{task_id}_response_mask"]
    token_level_scores = data.batch[f"task{task_id}_token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch[f"task{task_id}_old_log_probs"], data.batch[f"task{task_id}_ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch[f"task{task_id}_token_level_rewards"] = token_level_rewards

    metrics = {f"actor/task{task_id}_reward_kl_penalty": current_kl, f"actor/task{task_id}_reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    task_id: int = 1
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    if adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch[f"task{task_id}_response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch[f"task{task_id}_token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch[f"task{task_id}_advantages"] = advantages
        data.batch[f"task{task_id}_returns"] = returns

    elif adv_estimator == AdvantageEstimator.GRPO_TASK_SKIP: # GRPO task3 masking if needed
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch[f"task{task_id}_response_mask"]

        # Call compute_grpo_task_skip_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_task_skip_outcome_advantage(
            token_level_rewards=data.batch[f"task{task_id}_token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch[f"task{task_id}_advantages"] = advantages
        data.batch[f"task{task_id}_returns"] = returns
    
    return data

class RayImageGenerationTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.reward_kwargs = config.reward_model.get("reward_kwargs", {})
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, uid, prompt_id, prompt, gen_imgs_pil_list, feedback_texts, regen_imgs_pil_list,
            gts_imgs, gts_tuples, gts_vqas, scores, reward_extra_infos_dict, dump_path
        ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        image_dir = os.path.join(dump_path, f"{self.global_steps}")
        os.makedirs(image_dir, exist_ok=True)

        n = len(prompt)
        base_data = {
            "prompt_id": prompt_id,
            "uid": uid,
            "prompt": prompt,
            "rollout_id": list(range(n)),
            **scores,
            "step": [self.global_steps] * n,
        }

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")
        
        # save_num = self.reward_kwargs.get("img_saving", {}).get("num", n)
        for i in range(n): # (min(n, save_num)):

            # Set uid
            id = uid[i]

            # Prompt id
            pid = prompt_id[i]

            with open(os.path.join(image_dir, f"text_{pid}_{id}_{i}.txt"), 'w', encoding='utf-8') as f:
                f.write(f"Sample {pid}'s {i}th {id}\n")
                f.write("=" * 40 + "\n")

                # Save input text
                f.write(f"Input Text: {prompt[i]}\n\n")

                # Save generated image (Task 1)
                if gen_imgs_pil_list is not None and len(gen_imgs_pil_list) > i:
                    save_path = os.path.join(image_dir, f"gen_img_{pid}_{id}_{i}.png")
                    PIL.Image.fromarray(gen_imgs_pil_list[i].astype(np.uint8)).save(save_path)
                    f.write(f"Generated Image: gen_img_{pid}_{id}_{i}.png\n")
                    if "task1_scores" in scores and len(scores["task1_scores"]) > i:
                        task1_score = scores["task1_scores"][i]
                        f.write(f"Task1 Score: {task1_score}\n")
                    f.write("\n")

                # Save feedback text (Task 2)
                if feedback_texts is not None and len(feedback_texts) > i:
                    f.write(f"Feedback Text:\n{feedback_texts[i]}\n")
                    if "task2_scores" in scores and len(scores["task2_scores"]) > i:
                        task2_score = scores["task2_scores"][i]
                        f.write(f"Task2 Score: {task2_score}\n")
                    f.write("\n")

                # Save regenerated image (Task 3)
                if regen_imgs_pil_list is not None and len(regen_imgs_pil_list) > i:
                    regen_path = os.path.join(image_dir, f"regen_img_{pid}_{id}_{i}.png")
                    PIL.Image.fromarray(regen_imgs_pil_list[i].astype(np.uint8)).save(regen_path)
                    f.write(f"Regenerated Image: regen_img_{pid}_{id}_{i}.png\n")
                    if "task3_scores" in scores and len(scores["task3_scores"]) > i:
                        task3_score = scores["task3_scores"][i]
                        f.write(f"Task3 Score: {task3_score}\n")
                    f.write("\n")

                # Save ground truth image
                if gts_imgs is not None and len(gts_imgs) > i and gts_imgs[i] is not None:
                    ground_truth_path = os.path.join(image_dir, f"ground_truth_{pid}_{id}.png")
                    if not os.path.exists(ground_truth_path):
                        PIL.Image.open(gts_imgs[i]).convert("RGB").save(ground_truth_path)
                        f.write(f"Ground Truth Image: ground_truth_{pid}_{id}.png\n\n")

                # Save ground truth tuples and VQA
                if gts_tuples is not None and len(gts_tuples) > i:
                    ground_truth_tuple = gts_tuples[i]
                    f.write(f"Ground Truth Tuple:\n{ground_truth_tuple}\n")
                if gts_vqas is not None and len(gts_vqas) > i:
                    ground_truth_vqa_question = gts_vqas[i]
                    f.write(f"Ground Truth VQA:\n{ground_truth_vqa_question}\n\n")

                for key, value in reward_extra_infos_dict.items():
                    if key in value and len(value[key]) > i:
                        f.write(f"Reward Extra Infos:\n{key}:\n{value[key][i]}\n\n")

                f.write("=" * 40 + "\n\n")
        
        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            prompt_id = batch.non_tensor_batch['prompt_id'].tolist()
            uid = batch.non_tensor_batch["uid"].tolist()
            prompt = batch.non_tensor_batch['prompt'].tolist()
            gen_imgs_pil_list = batch.non_tensor_batch['task1_gen_imgs_pil_list']
            feedback_texts = batch.non_tensor_batch['task2_feedback_texts'].tolist()
            regen_imgs_pil_list = batch.non_tensor_batch['task3_regen_imgs_pil_list']
            gts_imgs = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]
            gts_tuples = [item.non_tensor_batch.get("reward_model", {}).get("tuple", None) for item in batch]
            gts_vqas = [item.non_tensor_batch.get("reward_model", {}).get("vqa_question", None) for item in batch]

            scores = {}
            # Add all task scores
            for tid in [1, 2, 3]:
                key = f"task{tid}_token_level_scores"
                if key in batch.batch:
                    scores[f"task{tid}_scores"] = batch.batch[key].sum(-1).cpu().tolist()

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()

            dump_path = rollout_data_dir
            os.makedirs(dump_path, exist_ok=True)

            self._dump_generations(
                prompt_id=prompt_id,
                uid=uid,
                prompt=prompt,
                gen_imgs_pil_list=gen_imgs_pil_list,
                feedback_texts=feedback_texts,
                regen_imgs_pil_list=regen_imgs_pil_list,
                gts_imgs=gts_imgs,
                gts_tuples=gts_tuples,
                gts_vqas=gts_vqas,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=dump_path,
            )

    def _maybe_log_val_generations(self, 
            prompts, task1_gen_imgs, task2_feedback_text, task3_regen_imgs, task1_scores, task2_scores, task3_scores
        ):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(prompts, task1_gen_imgs, task2_feedback_text, task3_regen_imgs, task1_scores, task2_scores, task3_scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        # task reward tensors collection
        task_reward_tensors = {1: [], 2: [], 3: []}
        data_source_lst = []

        # Lists to collect samples for the table
        sample_prompts = []
        sample_task1_gen_imgs = []
        sample_task2_feedback_texts = []
        sample_task3_regen_imgs = []
        sample_task1_scores = []
        sample_task2_scores = []
        sample_task3_scores = []
        self.val_reward_fn.steps = self.global_steps

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                        interleave=True)

            # add uid to batch <- dummy tensor in batch
            test_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
            )

            test_gen_batch = self._get_gen_batch(test_batch)
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            print('validation generation end')

            prompts = test_output_gen_batch.non_tensor_batch['prompt'].tolist()
            sample_prompts.extend(prompts)

            task1_gen_imgs_pil_list = test_output_gen_batch.non_tensor_batch['task1_gen_imgs_pil_list']
            task1_gen_imgs_pil_list = [wandb.Image(gen_img, caption=prompts[i]) for i, gen_img in enumerate(task1_gen_imgs_pil_list)]
            sample_task1_gen_imgs.extend(task1_gen_imgs_pil_list)

            task2_feedback_texts = test_output_gen_batch.non_tensor_batch['task2_feedback_texts'].tolist()
            sample_task2_feedback_texts.extend(task2_feedback_texts)

            task3_regen_imgs_pil_list = test_output_gen_batch.non_tensor_batch['task3_regen_imgs_pil_list']
            task3_regen_imgs_pil_list = [wandb.Image(regen_img, caption=prompts[i]) for i, regen_img in enumerate(task3_regen_imgs_pil_list)]
            sample_task3_regen_imgs.extend(task3_regen_imgs_pil_list)

            if self.config.reward_model.enable and not self.config.reward_model.paired:
                reward_tensor = self.rm_wg.compute_rm_score(test_batch)
                test_batch = test_batch.union(reward_tensor)

            # evaluate using reward_function for each task
            for task_id in [1, 2, 3]:
                task_reward_dict = self.val_reward_fn(test_batch, eval=True, task_id=task_id, return_dict=True)
                task_reward_tensor = task_reward_dict[f"task{task_id}_reward_tensor"]
                task_reward_tensors[task_id].append(task_reward_tensor)

                # Compute per-sample scores
                per_sample_scores = task_reward_tensor.sum(-1).cpu().tolist()

                if task_id == 1:
                    sample_task1_scores.extend(per_sample_scores)
                elif task_id == 2:
                    sample_task2_scores.extend(per_sample_scores)
                else:
                    sample_task3_scores.extend(per_sample_scores)

            batch_size = len(prompts)
            data_source = test_batch.non_tensor_batch.get('data_source', ['unknown'] * batch_size)
            data_source_lst.extend(data_source)

        # Log validation generations
        self._maybe_log_val_generations(
            prompts=sample_prompts,
            task1_gen_imgs=sample_task1_gen_imgs,
            task2_feedback_text=sample_task2_feedback_texts,
            task3_regen_imgs=sample_task3_regen_imgs,
            task1_scores=sample_task1_scores,
            task2_scores=sample_task2_scores,
            task3_scores=sample_task3_scores
        )

        # Store validation samples for logging by Trainer
        validation_samples = list(zip(
            sample_prompts,
            sample_task1_gen_imgs,
            sample_task2_feedback_texts,
            sample_task3_regen_imgs,
            sample_task1_scores,
            sample_task2_scores,
            sample_task3_scores
        ))

        data_sources = np.array(data_source_lst)

        # task metrics computation
        metric_dict = {}
        for task_id in [1, 2, 3]:
            reward_tensor = torch.cat(task_reward_tensors[task_id], dim=0).sum(-1).cpu()

            # evaluate test_score based on data source
            data_source_reward = {}
            for i in range(reward_tensor.shape[0]):
                data_source = data_sources[i]
                if data_source not in data_source_reward:
                    data_source_reward[data_source] = []
                data_source_reward[data_source].append(reward_tensor[i].item())

            for data_source, rewards in data_source_reward.items():
                valid_rewards = [r for r in rewards if r > 0] # positive
                if valid_rewards:
                    metric_dict[f'val/task{task_id}_score/{data_source}'] = np.mean(valid_rewards)
                metric_dict[f'val/task{task_id}_positive_ratio/{data_source}'] = np.mean(np.array(rewards) > 0)

        # Include validation generation samples in the result for Trainer to log
        metric_dict['validation_samples'] = validation_samples

        return metric_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["dummy_tensor"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from recipe.image_rl.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch <- dummy tensor in batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        # Not support now async mode
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # Remove AdvantageEstimator.REMAX line

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = batch.batch["task1_attention_mask"].sum(dim=-1) + batch.batch["task1_response_mask"].sum(dim=-1) \
                                                        + batch.batch["task2_attention_mask"].sum(dim=-1) + batch.batch["task2_response_mask"].sum(dim=-1) \
                                                        + batch.batch["task3_attention_mask"].sum(dim=-1) + batch.batch["task3_response_mask"].sum(dim=-1)

                    # Process all tasks to prepare data for multi-task training
                    for task_id in [1, 2, 3]:

                        batch.batch["attention_mask"] = torch.cat([batch.batch[f"task{task_id}_attention_mask"], batch.batch[f"task{task_id}_response_mask"]], dim=1)

                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(data=batch, config=self.config, tokenizer=self.tokenizer, processor=self.processor)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn, eval=False, task_id=task_id)
                            batch.batch["task_id"] = torch.tensor([task_id for _ in range(len(batch))], dtype=int)

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch[f"task{task_id}_entropys"]
                            response_masks = batch.batch[f"task{task_id}_response_mask"]

                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {f"actor/task{task_id}_entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop(f"task{task_id}_entropys")
                            batch = batch.union(old_log_prob)

                            if f"task{task_id}_rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # Not use computing values in RLVR

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch[f"task{task_id}_token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty, task_id=task_id
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch[f"task{task_id}_token_level_rewards"] = batch.batch[f"task{task_id}_token_level_scores"]

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                                task_id=task_id
                            )

                        # Log rollout generations if enabled (per task)
                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if self.config.trainer.rollout_freq > 0 and (
                            self.global_steps % self.config.trainer.rollout_freq == 0 and rollout_data_dir
                        ):
                            self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                        # Remove task-specific attention_mask and task_id for next iteration
                        batch.pop(batch_keys=["attention_mask", "task_id"])

                    # update critic (after all tasks prepared)
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor with all tasks together
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # No task_id set here - actor will process all tasks based on config
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                for task_id in [1, 2, 3]:
                    batch.batch["task_id"] = torch.tensor([task_id for _ in range(len(batch))], dtype=int)
                    # collect metrics
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    metrics.update(compute_group_reward_metrics(batch=batch))
                    # Remove universal keys from batch
                    batch.pop(batch_keys=["task_id"])
                
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
