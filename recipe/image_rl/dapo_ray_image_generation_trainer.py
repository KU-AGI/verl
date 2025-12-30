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
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
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

class RayImageGenerationDAPOTrainer(RayPPOTrainer):
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
                f.write("=" * 40 + "\n\n")

                # Save input text
                f.write(f"Input Text: {prompt[i]}\n\n")

                # Save generated image (Task 1)
                if gen_imgs_pil_list is not None and len(gen_imgs_pil_list) > i:
                    save_path = os.path.join(image_dir, f"gen_img_{pid}_{id}_{i}.png")
                    PIL.Image.fromarray(gen_imgs_pil_list[i].astype(np.uint8)).save(save_path)
                    f.write(f"Generated Image: gen_img_{pid}_{id}_{i}.png\n\n")
                    if "task1_reward_response" in reward_extra_infos_dict and len(reward_extra_infos_dict["task1_reward_response"]) > i:
                        task1_reward_response = reward_extra_infos_dict["task1_reward_response"][i]
                        f.write(f"Task1 Reward Response:\n{task1_reward_response}\n\n")
                    if "task1_scores" in scores and len(scores["task1_scores"]) > i:
                        task1_score = scores["task1_scores"][i]
                        f.write(f"Task1 Score: {task1_score}\n\n")
                    f.write("-" * 40 + "\n\n")

                # Save feedback text (Task 2)
                if feedback_texts is not None and len(feedback_texts) > i:
                    f.write(f"Feedback Text:\n{feedback_texts[i]}\n\n")
                    if "task2_reward_response" in reward_extra_infos_dict and len(reward_extra_infos_dict["task2_reward_response"]) > i:
                        task2_reward_response = reward_extra_infos_dict["task2_reward_response"][i]
                        f.write(f"Task2 Reward Response:\n{task2_reward_response}\n\n")
                    if "task2_scores" in scores and len(scores["task2_scores"]) > i:
                        task2_score = scores["task2_scores"][i]
                        f.write(f"Task2 Score: {task2_score}\n\n")
                    f.write("-" * 40 + "\n\n")

                # Save regenerated image (Task 3)
                if regen_imgs_pil_list is not None and len(regen_imgs_pil_list) > i:
                    regen_path = os.path.join(image_dir, f"regen_img_{pid}_{id}_{i}.png")
                    PIL.Image.fromarray(regen_imgs_pil_list[i].astype(np.uint8)).save(regen_path)
                    f.write(f"Regenerated Image: regen_img_{pid}_{id}_{i}.png\n\n")
                    if "task3_reward_response" in reward_extra_infos_dict and len(reward_extra_infos_dict["task3_reward_response"]) > i:
                        task3_reward_response = reward_extra_infos_dict["task3_reward_response"][i]
                        f.write(f"Task3 Reward Response:\n{task3_reward_response}\n\n")
                    if "task3_scores" in scores and len(scores["task3_scores"]) > i:
                        task3_score = scores["task3_scores"][i]
                        f.write(f"Task3 Score: {task3_score}\n\n")
                    f.write("-" * 40 + "\n\n")

                # Save ground truth image
                if gts_imgs is not None and len(gts_imgs) > i and gts_imgs[i] is not None:
                    ground_truth_path = os.path.join(image_dir, f"ground_truth_{pid}_{id}.png")
                    if not os.path.exists(ground_truth_path):
                        PIL.Image.open(gts_imgs[i]).convert("RGB").save(ground_truth_path)
                        f.write(f"Ground Truth Image: ground_truth_{pid}_{id}.png\n\n")

                # Save ground truth tuples and VQA
                if gts_tuples is not None and len(gts_tuples) > i:
                    ground_truth_tuple = gts_tuples[i]
                    f.write(f"Ground Truth Tuple:\n{ground_truth_tuple}\n\n")
                if gts_vqas is not None and len(gts_vqas) > i:
                    ground_truth_vqa_question = gts_vqas[i]
                    f.write(f"Ground Truth VQA:\n{ground_truth_vqa_question}\n\n")

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

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict, task_id: int = 1):
        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch[f"task{task_id}_entropys"]
            response_masks = batch.batch[f"task{task_id}_response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {f"actor/task{task_id}_entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop(f"task{task_id}_entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
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

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            print('validation generation end')

            prompts = test_output_gen_batch.non_tensor_batch['prompt'].tolist()
            task1_gen_imgs_pil_list = test_output_gen_batch.non_tensor_batch['task1_gen_imgs_pil_list']
            task1_gen_imgs_pil_list = [wandb.Image(gen_img, caption=prompts[i]) for i, gen_img in enumerate(task1_gen_imgs_pil_list)]
            sample_outputs.extend(task1_gen_imgs_pil_list)

            test_batch = test_batch.union(test_output_gen_batch)
            
            if self.config.reward_model.enable and not self.config.reward_model.paired:
                reward_tensor = self.rm_wg.compute_rm_score(test_batch)
                test_batch = test_batch.union(reward_tensor)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch, eval=True, task_id=1, return_dict=False)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

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
        self.gen_steps = 0

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
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            
            batch = None
            num_prompt_in_batch = 0
            num_gen_batches = 0
            
            for batch_dict in self.train_dataloader:
                metrics = {}
                
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                gen_batch = self._get_gen_batch(new_batch)
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    accumulated_gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    # === 모든 Task를 순서대로 실행 (의존성 때문에 skip 불가) ===
                    for task_id in [1, 2, 3]:
                        task_gen_batch = accumulated_gen_batch[:]
                        task_gen_batch.batch["task_id"] = torch.tensor([task_id for _ in range(len(task_gen_batch))], dtype=int)

                        # Generate
                        with marked_timer("gen", timing_raw, "red"):
                            task_gen_batch_output = self.actor_rollout_wg.generate_sequences(task_gen_batch)
                            timing_raw.update(task_gen_batch_output.meta_info["timing"])
                            task_gen_batch_output.meta_info.pop("timing", None)

                        accumulated_gen_batch = accumulated_gen_batch.union(task_gen_batch_output)
                        new_batch.batch.pop("task_id", None)
                        new_batch = new_batch.union(task_gen_batch_output)

                        # KL metrics if needed
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw, task_id)

                        # Reward computation
                        with marked_timer("reward", timing_raw, "yellow"):
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(
                                    data=new_batch, config=self.config, tokenizer=self.tokenizer, processor=self.processor
                                )
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(
                                    new_batch, self.reward_fn, eval=False, task_id=task_id
                                )
                            new_batch.batch[f"task{task_id}_token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                new_batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                        # KL penalty
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty, task_id=task_id
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch[f"task{task_id}_token_level_rewards"] = new_batch.batch[f"task{task_id}_token_level_scores"]

                    # === 모든 Task 완료 후: Task별 metric 기반으로 filtering ===
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])
                        num_prompt_in_batch = len(batch) // self.config.actor_rollout_ref.rollout.n
                    else:
                        metric_name = self.config.algorithm.filter_groups.metric
                        
                        # 각 task별로 kept_uids 계산
                        kept_uids_per_task = []
                        for task_id in [1, 2, 3]:
                            prompt_uid2metric_vals = defaultdict(list)
                            metric_key = f"task{task_id}_{metric_name}"
                            
                            if metric_key not in new_batch.batch:
                                continue
                                
                            for uid, metric_val in zip(
                                new_batch.non_tensor_batch["uid"],
                                new_batch.batch[metric_key],
                                strict=True
                            ):
                                prompt_uid2metric_vals[uid].append(metric_val)

                            kept_uids = set([
                                uid for uid, vals in prompt_uid2metric_vals.items()
                                if np.std(vals) > 0 or len(vals) == 1
                            ])
                            kept_uids_per_task.append(kept_uids)
                        
                        # 선택: Intersection (모든 task에서 통과) vs Union (하나라도 통과)
                        # Option A: Intersection - 가장 엄격
                        if kept_uids_per_task:
                            final_kept_uids = set.intersection(*kept_uids_per_task)
                        else:
                            final_kept_uids = set(new_batch.non_tensor_batch["uid"])
                        
                        # # Option B: Union - 가장 관대
                        # if kept_uids_per_task:
                        #     final_kept_uids = set.union(*kept_uids_per_task)
                        # else:
                        #     final_kept_uids = set(new_batch.non_tensor_batch["uid"])
                        
                        # # Option C: 특정 task 기준 (예: task1만)
                        # filter_task_id = self.config.algorithm.filter_groups.get("filter_task_id", 1)
                        # final_kept_uids = kept_uids_per_task[filter_task_id - 1] if kept_uids_per_task else set(new_batch.non_tensor_batch["uid"])

                        num_prompt_in_batch += len(final_kept_uids)

                        kept_traj_idxs = [
                            idx for idx, uid in enumerate(new_batch.non_tensor_batch["uid"])
                            if uid in final_kept_uids
                        ]

                        if len(kept_traj_idxs) > 0:
                            new_batch = new_batch[kept_traj_idxs]
                            batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                    # Batch size 체크
                    prompt_bsz = self.config.data.train_batch_size
                    traj_bsz = prompt_bsz * self.config.actor_rollout_ref.rollout.n
                    
                    if num_prompt_in_batch < prompt_bsz:
                        print(f"Prompts collected: {num_prompt_in_batch} < {prompt_bsz}")
                        max_num_gen_batches = self.config.algorithm.filter_groups.get("max_num_gen_batches", 0)
                        if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                            print(f"{num_gen_batches=}. Keep generating...")
                            self.gen_steps += 1
                            continue
                        else:
                            raise ValueError(
                                f"{num_gen_batches=} >= {max_num_gen_batches=}. "
                                "Generated too many batches."
                            )

                    # === 충분한 샘플 수집됨: 학습 진행 ===
                    batch = batch[:traj_bsz]

                    # After filtering (or not), proceed with the rest of the training logic
                    # create attention_mask for global use (using task3 as it's the last one)
                    batch.batch["attention_mask"] = torch.cat([batch.batch["task1_attention_mask"], batch.batch["task1_response_mask"], batch.batch["task2_attention_mask"], batch.batch["task2_response_mask"], batch.batch["task3_attention_mask"], batch.batch["task3_response_mask"]], dim=1)

                    # === Updating ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    if not self.config.algorithm.use_kl_in_reward:
                        # Need to compute KL metrics for all tasks after filtering
                        for task_id in [1, 2, 3]:
                            batch.batch["task_id"] = torch.tensor([task_id for _ in range(len(batch))], dtype=int)
                            batch = self.compute_kl_related_metrics(batch, metrics, timing_raw, task_id)
                            batch.batch.pop("task_id", None)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    # Compute advantages and update for all tasks
                    for task_id in [1, 2, 3]:
                        with marked_timer("adv", timing_raw, "brown"):
                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                task_id=task_id
                            )

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = batch.batch["task1_attention_mask"].sum(dim=-1) + batch.batch["task1_response_mask"].sum(dim=-1) \
                                                        + batch.batch["task2_attention_mask"].sum(dim=-1) + batch.batch["task2_response_mask"].sum(dim=-1) \
                                                        + batch.batch["task3_attention_mask"].sum(dim=-1) + batch.batch["task3_response_mask"].sum(dim=-1)

                    # update critic (once for all tasks)
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor (once for all tasks)
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled (for all tasks)
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if self.config.trainer.rollout_freq > 0 and (
                        self.global_steps % self.config.trainer.rollout_freq == 0 and rollout_data_dir
                    ):
                        reward_extra_infos_dict = {f"task{task_id}_reward_response": batch.non_tensor_batch[f"task{task_id}_reward_response"] for task_id in [1, 2, 3]}
                        reward_extra_infos_dict.update({"task2_reward_align_response": batch.non_tensor_batch["task2_reward_align_response"]})
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    # Remove universal keys from batch
                    batch.pop(batch_keys=["attention_mask"])

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
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
                    timing_raw = defaultdict(float)  # clear timing

                    metrics["train/num_gen_batches"] = num_gen_batches
                    batch = None
                    num_prompt_in_batch = 0
                    num_gen_batches = 0

                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=self.global_steps)

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return

                    progress_bar.update(1)
                    self.global_steps += 1
                    self.gen_steps += 1
            # check if last step checkpint exists
            checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
            if not os.path.exists(checkpoint_dir):
                # save last step checkpoint
                timing_raw = defaultdict(float)
                with marked_timer("save_checkpoint", timing_raw, "green"):
                    self._save_checkpoint()
                metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
                logger.log(data=metrics, step=self.global_steps)