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
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Dict, Optional, Type

import numpy as np
import PIL
import PIL.Image
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

        self._rollout_dump_executor: Optional[ThreadPoolExecutor] = None
        self._rollout_dump_futures = []
        self._rollout_dump_workers = int(self.config.trainer.get("rollout_dump_workers", 1) or 0)
        self._rollout_dump_max_pending = int(
            self.config.trainer.get("rollout_dump_max_pending", max(2, 2 * max(self._rollout_dump_workers, 1))) or 0
        )

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _clone_batch_for_rollout_dump(self, batch: DataProto) -> DataProto:
        batch_size = len(batch)
        non_tensor_batch = {}
        for key, val in (getattr(batch, "non_tensor_batch", None) or {}).items():
            if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == batch_size:
                non_tensor_batch[key] = deepcopy(val)
            else:
                try:
                    val_len = len(val)
                except Exception:
                    val_len = "unknown"
                print(
                    f"[RolloutDump] skip non_tensor key with mismatched length: "
                    f"{key} len={val_len} batch_size={batch_size}"
                )
        return DataProto(
            batch=batch.batch.clone() if getattr(batch, "batch", None) is not None else None,
            non_tensor_batch=non_tensor_batch,
            meta_info=deepcopy(getattr(batch, "meta_info", None)),
        )

    def _reap_rollout_dump_futures(self, wait: bool = False):
        if not self._rollout_dump_futures:
            return
        pending = []
        for fut in self._rollout_dump_futures:
            if wait:
                fut.result()
                continue
            if fut.done():
                fut.result()
            else:
                pending.append(fut)
        self._rollout_dump_futures = pending

    def _shutdown_rollout_dump_executor(self, wait: bool = True):
        self._reap_rollout_dump_futures(wait=wait)
        if self._rollout_dump_executor is not None:
            self._rollout_dump_executor.shutdown(wait=wait)
            self._rollout_dump_executor = None

    def _submit_rollout_dump(
        self,
        batch: DataProto,
        reward_extra_infos_dict: dict,
        timing_raw: dict,
        rollout_data_dir: str,
    ):
        if self._rollout_dump_workers <= 0:
            self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
            return

        if self._rollout_dump_executor is None:
            self._rollout_dump_executor = ThreadPoolExecutor(
                max_workers=self._rollout_dump_workers,
                thread_name_prefix="rollout_dump",
            )

        self._reap_rollout_dump_futures(wait=False)
        if self._rollout_dump_max_pending > 0 and len(self._rollout_dump_futures) >= self._rollout_dump_max_pending:
            oldest = self._rollout_dump_futures.pop(0)
            oldest.result()

        dump_batch = self._clone_batch_for_rollout_dump(batch)
        dump_reward_extra = deepcopy(reward_extra_infos_dict)
        dump_timing = dict(timing_raw)
        fut = self._rollout_dump_executor.submit(
            self._log_rollout_data,
            dump_batch,
            dump_reward_extra,
            dump_timing,
            rollout_data_dir,
        )
        self._rollout_dump_futures.append(fut)

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
                        gts_imgs, summarizes, gts_tuples, gts_vqas, scores, reward_extra_infos_dict,
                        sample_versions=None, dump_path=".", turn_idxs=None, trajectory_ids=None, phases=None,
                        branch_ids=None):

        trainer_ver = getattr(self, "current_param_version", self.global_steps)
        step_dir = os.path.abspath(os.path.join(dump_path, str(trainer_ver)))
        os.makedirs(step_dir, exist_ok=True)

        n = len(prompt)

        # Resolve per-row metadata with safe defaults
        _turns = [int(turn_idxs[i]) if (turn_idxs is not None and i < len(turn_idxs)) else 0 for i in range(n)]
        _trajs = [int(trajectory_ids[i]) if (trajectory_ids is not None and i < len(trajectory_ids)) else i for i in range(n)]
        _vers = sample_versions if sample_versions is not None else [getattr(self, "current_param_version", self.global_steps)] * n
        _phases = [int(phases[i]) if (phases is not None and i < len(phases)) else 1 for i in range(n)]
        _branches = [int(branch_ids[i]) if (branch_ids is not None and i < len(branch_ids)) else -1 for i in range(n)]

        # Parse prompt_id → sample_key for each row
        sample_keys = []
        for i in range(n):
            pid_raw = prompt_id[i]
            pid_clean = pid_raw.replace("gen_img_", "").replace("text_", "")
            pid_core = pid_clean.split("_uid_")[0]
            parts = pid_core.split("_")
            source_parts, sample_id = [], "unknown"
            for p in parts:
                if p.isdigit() and len(p) >= 4:
                    sample_id = p
                    break
                if p not in ["uid", "chunk", "sample"]:
                    source_parts.append(p)
            source_rel_path = os.path.join(*source_parts) if source_parts else "unknown"
            sample_keys.append(os.path.join(source_rel_path, sample_id))

        # Group rows by (sample_key, source_ver, trajectory_id) → [(row_idx, turn_idx), ...]
        traj_groups: dict = defaultdict(list)
        for i in range(n):
            key = (sample_keys[i], _vers[i], _trajs[i])
            traj_groups[key].append((i, _turns[i]))

        # Cluster trajectory groups by sample for the cross-trajectory summary
        sample_to_traj_keys: dict = defaultdict(list)
        for key in traj_groups:
            sample_key, source_ver, _ = key
            sample_to_traj_keys[(sample_key, source_ver)].append(key)

        # Write per-trajectory/per-turn directory structure
        for (sample_key, source_ver), traj_keys in sample_to_traj_keys.items():
            sample_dir = os.path.join(step_dir, f"rollouter_source_v{source_ver}", sample_key)
            os.makedirs(sample_dir, exist_ok=True)

            summary_path = os.path.join(sample_dir, "comparison_summary.txt")
            summary_header = [
                f"📋 GRPO COMPARISON SUMMARY | STEP: {trainer_ver}",
                f"Sample Path: {sample_dir}",
                "T1/T2/T3 are token-level score sums; phase1 training dumps use MDP discounted returns.",
                "=" * 200,
                f"{'Traj':<5} | {'Turn':<4} | {'Branch':<7} | {'T1Ret':<5} | {'T2Ret':<5} | {'T3Ret':<5} | {'MDPΣ':<6} | {'Gen Path (Current)':<65} | {'Regen Path (Edited)'}",
                "-" * 200,
            ]
            summary_rows = []

            for traj_key in sorted(traj_keys, key=lambda k: k[2]):  # sort by traj_id
                _, _, traj_id = traj_key
                row_turns = sorted(traj_groups[traj_key], key=lambda x: x[1])  # sort by turn_idx

                traj_dir = os.path.join(sample_dir, f"traj_{traj_id}")
                os.makedirs(traj_dir, exist_ok=True)
                turn_branch_total_counts: dict[tuple[int, int], int] = defaultdict(int)
                for i, turn in row_turns:
                    turn_branch_total_counts[(int(turn), int(_branches[i]))] += 1
                turn_branch_seen_counts: dict[tuple[int, int], int] = defaultdict(int)

                # Per-trajectory turn summary
                traj_summary_lines = [
                    f"Trajectory {traj_id} | Sample: {sample_key} | Step: {trainer_ver}",
                    "T1/T2/T3 are token-level score sums; phase1 training dumps use MDP discounted returns.",
                    "-" * 80,
                    f"{'Turn':<5} | {'Branch':<7} | {'T1Ret':<5} | {'T2Ret':<5} | {'T3Ret':<5} | {'MDPΣ':<6}",
                    "-" * 80,
                ]

                for i, turn in row_turns:
                    branch_id = _branches[i]
                    turn_dir = os.path.join(traj_dir, f"turn_{turn}")
                    branch_dir = os.path.join(turn_dir, f"branch_{branch_id}" if branch_id >= 0 else "branch_shared")
                    pair = (int(turn), int(branch_id))
                    dup_idx = turn_branch_seen_counts[pair]
                    turn_branch_seen_counts[pair] += 1
                    if turn_branch_total_counts[pair] > 1:
                        row_dir = os.path.join(branch_dir, f"sample_{dup_idx}")
                    else:
                        row_dir = branch_dir
                    os.makedirs(row_dir, exist_ok=True)

                    paths = self._save_images(row_dir, i, gen_imgs_pil_list, regen_imgs_pil_list, gts_imgs)

                    t1 = self._get_safe_val(scores, 'task1_scores', i, None)
                    t2 = self._get_safe_val(scores, 'task2_scores', i, None)
                    t3 = self._get_safe_val(scores, 'task3_scores', i, None)
                    present_total_terms = [v for v in [t1, t2, t3] if v is not None and v > -100]
                    total = sum(present_total_terms) if present_total_terms else None

                    traj_summary_lines.append(
                        f"turn_{turn:<3} | {branch_id:<7} | {self._fmt_metric(t1, 2):<5} | "
                        f"{self._fmt_metric(t2, 2):<5} | {self._fmt_metric(t3, 2):<5} | "
                        f"{self._fmt_metric(total, 2):<6}"
                    )
                    summary_rows.append(
                        f"{traj_id:<5} | {turn:<4} | {branch_id:<7} | {self._fmt_metric(t1, 2):<5} | "
                        f"{self._fmt_metric(t2, 2):<5} | {self._fmt_metric(t3, 2):<5} | {self._fmt_metric(total, 2):<6} | "
                        f"{paths['gen']:<65} | {paths['regen']}"
                    )

                    item = {
                        'rollout_idx': turn,
                        'uid': uid[i],
                        'turn_idx': turn,
                        'trajectory_id': traj_id,
                        'phase': _phases[i],
                        'branch_id': branch_id,
                    }
                    item['sample_idx_within_branch'] = dup_idx
                    self._write_detailed_report(row_dir, i, item, paths, prompt, feedback_texts, scores,
                                               summarizes, gts_tuples, gts_vqas, reward_extra_infos_dict)

                traj_summary_path = os.path.join(traj_dir, "traj_summary.txt")
                with open(traj_summary_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(traj_summary_lines) + "\n")

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(summary_header + summary_rows) + "\n")

    # --- Helper Methods ---

    def _get_safe_val(self, data, key, idx, default="N/A"):
        """딕셔너리에서 안전하게 값을 가져오고 수치형이면 float 반환"""
        if key in data and (isinstance(data[key], list) or isinstance(data[key], np.ndarray)):
            if len(data[key]) > idx:
                val = data[key][idx]
                if val is None:
                    return default
                try:
                    if np.isscalar(val):
                        num = float(val)
                        if not np.isfinite(num):
                            return default
                except Exception:
                    pass
                if isinstance(default, float) or default is None:
                    try:
                        num = float(val)
                    except Exception:
                        return default
                    if not np.isfinite(num):
                        return default
                    return num
                return val
        return default

    def _fmt_metric(self, val, precision=2, na="N/A"):
        if val is None:
            return na
        try:
            num = float(val)
        except Exception:
            return na
        if not np.isfinite(num):
            return na
        return f"{num:.{precision}f}"

    def _get_safe_response(self, data, key, idx, default="N/A"):
        """딕셔너리에서 안전하게 값을 가져오고 response text 기록"""
        if key in data and (isinstance(data[key], list) or isinstance(data[key], np.ndarray)):
            if len(data[key]) > idx:
                response = data[key][idx]
                return response
        return default

    def _pixel_values_to_pil_list(self, pixel_values):
        if pixel_values is None:
            return None
        if isinstance(pixel_values, np.ndarray):
            pixel_values = torch.from_numpy(pixel_values)
        if not torch.is_tensor(pixel_values):
            return None
        imgs = pixel_values.detach().cpu().float()
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.ndim != 4:
            return None

        image_processor = getattr(getattr(self, "processor", None), "image_processor", None)
        image_mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        image_std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
        rescale_factor = getattr(image_processor, "rescale_factor", 1.0 / 255.0)

        mean_t = torch.tensor(image_mean, dtype=imgs.dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(image_std, dtype=imgs.dtype).view(1, -1, 1, 1)
        imgs = imgs * std_t + mean_t
        if rescale_factor not in (None, 0):
            imgs = imgs / float(rescale_factor)
        imgs = imgs.clamp(0, 255).round().to(torch.uint8)
        imgs = imgs.permute(0, 2, 3, 1).contiguous().numpy()
        return np.array([PIL.Image.fromarray(img) for img in imgs], dtype=object)

    def _save_images(self, rollout_dir, i, gen_imgs, regen_imgs, gts_imgs):
        """이미지 저장 후 절대 경로 딕셔너리 반환"""
        paths = {"gen": "N/A", "regen": "N/A", "gt": "N/A"}
        
        # Task 1 Gen Image
        if gen_imgs is not None and len(gen_imgs) > i:
            p = os.path.join(rollout_dir, "gen.png")
            # numpy array 대응
            img_data = gen_imgs[i]
            if isinstance(img_data, np.ndarray):
                PIL.Image.fromarray(img_data.astype(np.uint8)).save(p)
            else:
                img_data.save(p)
            paths["gen"] = os.path.abspath(p)

        # Task 3 Regen Image
        if regen_imgs is not None and len(regen_imgs) > i:
            p = os.path.join(rollout_dir, "regen.png")
            img_data = regen_imgs[i]
            if isinstance(img_data, np.ndarray):
                PIL.Image.fromarray(img_data.astype(np.uint8)).save(p)
            else:
                img_data.save(p)
            paths["regen"] = os.path.abspath(p)

        # Ground Truth Image
        if gts_imgs is not None and len(gts_imgs) > i and gts_imgs[i]:
            p = os.path.join(rollout_dir, "ground_truth.png")
            if not os.path.exists(p):
                try:
                    PIL.Image.open(gts_imgs[i]).convert("RGB").save(p)
                except: pass
            paths["gt"] = os.path.abspath(p)
            
        return paths

    def _write_detailed_report(self, rollout_dir, i, item, paths, prompt, feedback_texts, scores, 
                            summarizes, gts_tuples, gts_vqas, reward_extra_infos_dict):

        txt_path = os.path.join(rollout_dir, "txt.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            turn_part = f" | TURN: {item['turn_idx']}" if 'turn_idx' in item else ""
            traj_part = f" | TRAJ: {item['trajectory_id']}" if 'trajectory_id' in item else ""
            phase_part = f" | PHASE: {item['phase']}" if 'phase' in item else ""
            branch_part = f" | BRANCH: {item['branch_id']}" if 'branch_id' in item else ""
            f.write(f"📊 ROLLOUT REPORT | STEP: {self.global_steps} | ROLLOUT: {item['rollout_idx']}{traj_part}{turn_part}{phase_part}{branch_part}\n")
            f.write(f"Location: {os.path.abspath(rollout_dir)}\n")
            f.write("=" * 100 + "\n")
            f.write(f"📝 [PROMPT]\n{prompt[i]}\n")
            f.write("=" * 100 + "\n\n")

            # Task 1
            f.write(f"🖼️ [TASK 1] INITIAL GEN\n")
            f.write(f"  - Token Score Sum (MDP return in phase1 train): {self._get_safe_val(scores, 'task1_scores', i)}\n")
            f.write(f"  - Image Score S1: {self._get_safe_val(reward_extra_infos_dict, 'task1_image_score', i)}\n")
            f.write(f"  - MDP Step1 Reward r1: {self._get_safe_val(reward_extra_infos_dict, 'task1_mdp_reward', i)}\n")
            f.write(f"  - VQA Reward: {self._get_safe_val(reward_extra_infos_dict, 'task1_vqa_reward', i)}\n")
            f.write(f"  - Detector Reward (bonus): {self._get_safe_val(reward_extra_infos_dict, 'task1_detector_reward', i)}\n")
            f.write(f"  - Path: {paths['gen']}\n")
            f.write(f"  - VQA Response:\n{self._get_safe_response(reward_extra_infos_dict, 'task1_vqa_reward_response', i)}\n")
            det_details_1 = self._get_safe_response(reward_extra_infos_dict, 'task1_detector_details', i, default=[])
            if det_details_1:
                f.write(f"  - Detector Details:\n")
                for d in det_details_1:
                    f.write(f"    [{d.get('tuple_idx','?')}] {d.get('det_info',{})}\n")
                    f.write(f"         judge={d.get('det_judge')} | reason={d.get('det_reason','')}\n")
            f.write("\n")

            _report_task_ids = list(self.config.actor_rollout_ref.actor.multi_task.get("task_ids", [1]))

            # Log Task 2 if feedback_texts exist (even if task_id != 2)
            if feedback_texts and i < len(feedback_texts) and feedback_texts[i] is not None:
                step2_return = self._get_safe_val(scores, 'task2_step2_return_scores', i)
                if step2_return == "N/A":
                    step2_return = self._get_safe_val(reward_extra_infos_dict, 'task2_step2_return_score', i)
                step3_return = self._get_safe_val(scores, 'task2_step3_return_scores', i)
                if step3_return == "N/A":
                    step3_return = self._get_safe_val(reward_extra_infos_dict, 'task2_step3_return_score', i)
                step4_return = self._get_safe_val(scores, 'task2_step4_return_scores', i)
                if step4_return == "N/A":
                    step4_return = self._get_safe_val(reward_extra_infos_dict, 'task2_step4_return_score', i)
                f.write(f"💬 [TASK 2] FEEDBACK GENERATION\n")
                f.write(f"  - Token Score Sum (G_step2 + G_step3 + G_step4 in phase1 train): {self._get_safe_val(scores, 'task2_scores', i)}\n")
                f.write(f"  - MDP Step2 Return G_step2: {step2_return}\n")
                f.write(f"  - MDP Step3 Return G_step3: {step3_return}\n")
                f.write(f"  - MDP Step4 Return G_step4: {step4_return}\n")
                f.write(f"  - MDP Step2 Reward r_step2: {self._get_safe_val(reward_extra_infos_dict, 'task2_step2_reward', i)}\n")
                f.write(f"  - MDP Step3 Reward r_step3: {self._get_safe_val(reward_extra_infos_dict, 'task2_step3_reward', i)}\n")
                f.write(f"  - MDP Step4 Reward r_step4: {self._get_safe_val(reward_extra_infos_dict, 'task2_step4_reward', i)}\n")
                f.write(f"  - Format Reward (rule-based) (add score): {self._get_safe_val(reward_extra_infos_dict, 'task2_rule_based_format_reward', i)}\n")
                f.write(f"  - Decompose Reward (rule-based) (add score): {self._get_safe_val(reward_extra_infos_dict, 'task2_rule_based_decompose_reward', i)}\n")
                f.write(f"  - Internal Consistency OK (gating decompose): {self._get_safe_val(reward_extra_infos_dict, 'task2_internal_consistency_ok', i)}\n")
                f.write(f"  - Tuple Format OK (gating prompt→tuple / tuple→vqa): {self._get_safe_val(reward_extra_infos_dict, 'task2_tuple_format_ok', i)}\n")
                f.write(f"  - VQA Format OK (gating tuple→vqa / vqa→feedback): {self._get_safe_val(reward_extra_infos_dict, 'task2_vqa_format_ok', i)}\n")
                f.write(f"  - Feedback Format OK (rule-based) (gating vqa→feedback): {self._get_safe_val(reward_extra_infos_dict, 'task2_rule_based_feedback_format_ok', i)}\n")
                f.write(f"  - No Feedback Needed (rule-based) (gating vqa→feedback): {self._get_safe_val(reward_extra_infos_dict, 'task2_no_feedback_needed', i)}\n")
                f.write(f"  - Stage1 Prompt→Tuple Reward: {self._get_safe_val(reward_extra_infos_dict, 'task2_prompt_to_tuple_reward', i)}\n")
                f.write(f"  - Stage2 Tuple→VQA Reward: {self._get_safe_val(reward_extra_infos_dict, 'task2_tuple_to_vqa_reward', i)}\n")
                f.write(f"  - Stage3 VQA→Feedback Reward: {self._get_safe_val(reward_extra_infos_dict, 'task2_vqa_to_feedback_reward', i)}\n")
                f.write(f"  - Total VLM Reward: {self._get_safe_val(reward_extra_infos_dict, 'task2_vlm_reward', i)}\n")
                f.write(f"  - Model Feedback:\n{feedback_texts[i]}\n")
                f.write(f"  - Stage1 Response:\n{self._get_safe_response(reward_extra_infos_dict, 'task2_prompt_to_tuple_response', i)}\n")
                f.write(f"  - Stage2 Response:\n{self._get_safe_response(reward_extra_infos_dict, 'task2_tuple_to_vqa_response', i)}\n")
                f.write(f"  - Stage3 Response:\n{self._get_safe_response(reward_extra_infos_dict, 'task2_vqa_to_feedback_response', i)}\n")
                f.write("\n")

            # Log Task 3 if regen images exist (even if task_id != 3)
            if 3 in _report_task_ids:
                f.write(f"🔄 [TASK 3] RE-GENERATION\n")
                f.write(f"  - Token Score Sum (MDP Step5 return in phase1 train): {self._get_safe_val(scores, 'task3_scores', i)}\n")
                f.write(f"  - Previous Image Score S_t: {self._get_safe_val(reward_extra_infos_dict, 'task3_prev_image_score', i)}\n")
                f.write(f"  - Next Image Score S_t+1: {self._get_safe_val(reward_extra_infos_dict, 'task3_next_image_score', i)}\n")
                f.write(f"  - Image Score Gain: {self._get_safe_val(reward_extra_infos_dict, 'task3_image_score_gain', i)}\n")
                f.write(f"  - Edit IF Reward e_t: {self._get_safe_val(reward_extra_infos_dict, 'task3_edit_if_reward', i)}\n")
                f.write(f"  - MDP Step5 Reward r_step5: {self._get_safe_val(reward_extra_infos_dict, 'task3_step5_reward', i)}\n")
                f.write(f"  - VQA Reward: {self._get_safe_val(reward_extra_infos_dict, 'task3_vqa_reward', i)}\n")
                f.write(f"  - Edit Instruction Following Reward: {self._get_safe_val(reward_extra_infos_dict, 'task3_edit_reward', i)}\n")
                f.write(f"  - Detector Reward (bonus): {self._get_safe_val(reward_extra_infos_dict, 'task3_detector_reward', i)}\n")
                f.write(f"  - Path: {paths['regen']}\n")
                f.write(f"  - VQA Response:\n{self._get_safe_response(reward_extra_infos_dict, 'task3_vqa_reward_response', i)}\n")
                f.write(f"  - Edit Instruction Following Response:\n{self._get_safe_response(reward_extra_infos_dict, 'task3_edit_reward_response', i)}\n")
                det_details_3 = self._get_safe_response(reward_extra_infos_dict, 'task3_detector_details', i, default=[])
                if det_details_3:
                    f.write(f"  - Detector Details:\n")
                    for d in det_details_3:
                        f.write(f"    [{d.get('tuple_idx','?')}] {d.get('det_info',{})}\n")
                        f.write(f"         judge={d.get('det_judge')} | reason={d.get('det_reason','')}\n")
                f.write("\n")

            f.write(f"📚 [GROUND TRUTH REFERENCE]\n")
            f.write(f"  - GT Image:\n{paths['gt']}\n")
            f.write(f"  - Summary:\n{summarizes[i] if summarizes else 'N/A'}\n")
            f.write(f"  - Tuples:\n{gts_tuples[i] if gts_tuples else 'N/A'}\n")
            f.write(f"  - VQA Questions:\n{gts_vqas[i] if gts_vqas else 'N/A'}\n")
            f.write("=" * 100 + "\n")

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
            task_ids = list(self.config.actor_rollout_ref.actor.multi_task.get("task_ids", [1]))
            host_task_id = 0
            if "task_id" in batch.batch:
                try:
                    host_task_id = int(batch.batch["task_id"].view(-1)[0].item())
                except Exception:
                    host_task_id = 0
            # Prefer the current image seen by the policy at this row/turn.
            gen_imgs_pil_list = None
            def _maybe_get_tensor(key: str):
                return batch.batch[key] if key in batch.batch.keys() else None
            if host_task_id == 3:
                current_img_sources = [
                    _maybe_get_tensor("task3_input_imgs_pixel_values"),
                    _maybe_get_tensor("task3_task1_gen_imgs_pixel_values"),
                    _maybe_get_tensor("task1_gen_imgs_pixel_values"),
                    _maybe_get_tensor("current_imgs_pixel_values"),
                ]
            else:
                current_img_sources = [
                    _maybe_get_tensor("current_imgs_pixel_values"),
                    _maybe_get_tensor("task2_task1_gen_imgs_pixel_values"),
                    _maybe_get_tensor("task3_task1_gen_imgs_pixel_values"),
                    _maybe_get_tensor("task1_gen_imgs_pixel_values"),
                ]
            for src in current_img_sources:
                gen_imgs_pil_list = self._pixel_values_to_pil_list(src)
                if gen_imgs_pil_list is not None:
                    break
            if gen_imgs_pil_list is None:
                gen_imgs_pil_list = batch.non_tensor_batch.get('task1_gen_imgs_pil_list')
            if gen_imgs_pil_list is None:
                for host_tid in task_ids:
                    gen_imgs_pil_list = batch.non_tensor_batch.get(f'task{host_tid}_task1_gen_imgs_pil_list')
                    if gen_imgs_pil_list is not None:
                        break
            feedback_texts = None
            # task2 feedback: direct key for task1/2 batch; cross-task alias for task3 replay batch
            fb = batch.non_tensor_batch.get('task2_feedback_texts')
            if fb is None:
                for host_tid in task_ids:
                    fb = batch.non_tensor_batch.get(f'task{host_tid}_task2_feedback_texts')
                    if fb is not None:
                        break
            if fb is not None:
                feedback_texts = fb.tolist() if hasattr(fb, 'tolist') else list(fb)
            regen_imgs_pil_list = batch.non_tensor_batch.get('task3_regen_imgs_pil_list') if 3 in task_ids else None
            gts_imgs = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]
            summarizes = [item.non_tensor_batch.get("reward_model", {}).get("summary", None) for item in batch]
            gts_tuples = [item.non_tensor_batch.get("reward_model", {}).get("tuple", None) for item in batch]
            gts_vqas = [item.non_tensor_batch.get("reward_model", {}).get("vqa_question", None) for item in batch]

            sample_versions = batch.non_tensor_batch.get('param_version')
            if sample_versions is not None:
                sample_versions = sample_versions.tolist()
            else:
                sample_versions = [self.current_param_version] * len(batch)

            # Per-row turn_idx (multi-turn rollout). Stamped by the orchestrator
            # via `_stamp_turn_idx`; may live in `batch` (tensor) or
            # `non_tensor_batch` (post-replay-buffer passthrough). Default 0 if
            # missing (single-turn / legacy).
            turn_idxs = None
            if 'turn_idx' in batch.non_tensor_batch:
                ti = batch.non_tensor_batch['turn_idx']
                turn_idxs = ti.tolist() if hasattr(ti, 'tolist') else list(ti)
            elif 'turn_idx' in batch.batch:
                ti = batch.batch['turn_idx']
                turn_idxs = ti.detach().cpu().tolist() if hasattr(ti, 'detach') else list(ti)
            else:
                turn_idxs = [0] * len(batch)

            # Per-row trajectory_id. Stamped by the orchestrator via
            # `_stamp_trajectory_ids` so rows from the same trajectory
            # (possibly across turns / padded duplicates) share an id.
            trajectory_ids = None
            if 'trajectory_id' in batch.non_tensor_batch:
                ti = batch.non_tensor_batch['trajectory_id']
                trajectory_ids = ti.tolist() if hasattr(ti, 'tolist') else list(ti)
            else:
                trajectory_ids = [-1] * len(batch)

            phases = None
            if 'phase' in batch.non_tensor_batch:
                ph = batch.non_tensor_batch['phase']
                phases = ph.tolist() if hasattr(ph, 'tolist') else list(ph)
            else:
                phases = [1] * len(batch)

            branch_ids = None
            if 'branch_id' in batch.non_tensor_batch:
                bid = batch.non_tensor_batch['branch_id']
                branch_ids = bid.tolist() if hasattr(bid, 'tolist') else list(bid)
            else:
                branch_ids = [-1] * len(batch)

            scores = {}
            # Add all task scores; for replay task_batch use cross-task aliases as fallback
            for tid in [1, 2, 3]:
                key = f"task{tid}_token_level_scores"
                if key in batch.batch:
                    scores[f"task{tid}_scores"] = batch.batch[key].sum(-1).cpu().tolist()
                else:
                    for host_tid in task_ids:
                        cross_key = f"task{host_tid}_task{tid}_token_level_scores"
                        if cross_key in batch.batch:
                            scores[f"task{tid}_scores"] = batch.batch[cross_key].sum(-1).cpu().tolist()
                            break

            # Task2 is one text forward pass, but MDP training treats the
            # decompose / verify / feedback spans as three separate decisions.
            # Dump the per-decision return sums so `task2_scores` is not an
            # opaque aggregate.
            task2_score_tensor = None
            if "task2_token_level_scores" in batch.batch:
                task2_score_tensor = batch.batch["task2_token_level_scores"]
            else:
                for host_tid in task_ids:
                    cross_key = f"task{host_tid}_task2_token_level_scores"
                    if cross_key in batch.batch:
                        task2_score_tensor = batch.batch[cross_key]
                        break

            task2_segment_mask = None
            if "task2_segment_mask" in batch.batch:
                task2_segment_mask = batch.batch["task2_segment_mask"]
            else:
                for host_tid in task_ids:
                    cross_key = f"task{host_tid}_task2_segment_mask"
                    if cross_key in batch.batch:
                        task2_segment_mask = batch.batch[cross_key]
                        break

            if task2_score_tensor is not None and task2_segment_mask is not None:
                for seg, name in ((2, "step2"), (3, "step3"), (4, "step4")):
                    seg_mask = (task2_segment_mask == seg).to(task2_score_tensor.dtype)
                    scores[f"task2_{name}_return_scores"] = (
                        task2_score_tensor * seg_mask
                    ).sum(-1).cpu().tolist()

            reward_extra_infos_to_dump = dict(reward_extra_infos_dict)
            # Merge nested reward-extra dicts directly here so replay/fresh dump
            # rendering does not depend on trainer-side flattening alone.
            for key, nested in list(reward_extra_infos_dict.items()):
                if not (isinstance(key, str) and key.startswith("task") and key.endswith("_reward_extra_info")):
                    continue
                if not isinstance(nested, dict) or not nested:
                    continue
                for sub_key, sub_vals in nested.items():
                    reward_extra_infos_to_dump.setdefault(sub_key, sub_vals)

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
                summarizes=summarizes,
                gts_tuples=gts_tuples,
                gts_vqas=gts_vqas,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                sample_versions=sample_versions,
                turn_idxs=turn_idxs,
                trajectory_ids=trajectory_ids,
                phases=phases,
                branch_ids=branch_ids,
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
        task_ids = list(self.config.actor_rollout_ref.actor.multi_task.get("task_ids", [1]))
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

        val_data_dir = self.config.trainer.get("validation_data_dir", "validation")

        print(f"VALIDATION EPOCH: {len(self.val_dataloader)}")
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.meta_info["validate"] = True

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

            batch_scores = {}
            batch_reward_extra_infos = defaultdict(list)

            prompts = test_output_gen_batch.non_tensor_batch['prompt'].tolist()
            sample_prompts.extend(prompts)

            task1_gen_imgs_pil_list = test_output_gen_batch.non_tensor_batch['task1_gen_imgs_pil_list']
            task1_gen_imgs_pil_list = [wandb.Image(gen_img, caption=prompts[i]) for i, gen_img in enumerate(task1_gen_imgs_pil_list)]
            sample_task1_gen_imgs.extend(task1_gen_imgs_pil_list)

            if 2 in task_ids:
                task2_feedback_texts = test_output_gen_batch.non_tensor_batch.get('task2_feedback_texts', [None] * len(prompts))
                if hasattr(task2_feedback_texts, 'tolist'):
                    task2_feedback_texts = task2_feedback_texts.tolist()
                sample_task2_feedback_texts.extend(task2_feedback_texts)

            if 3 in task_ids:
                task3_regen_imgs_pil_list = test_output_gen_batch.non_tensor_batch.get('task3_regen_imgs_pil_list')
                if task3_regen_imgs_pil_list is not None:
                    task3_regen_imgs_pil_list = [wandb.Image(regen_img, caption=prompts[i]) for i, regen_img in enumerate(task3_regen_imgs_pil_list)]
                    sample_task3_regen_imgs.extend(task3_regen_imgs_pil_list)

            if self.config.reward_model.enable and not self.config.reward_model.paired:
                reward_tensor = self.rm_wg.compute_rm_score(test_output_gen_batch)
                test_output_gen_batch = test_output_gen_batch.union(reward_tensor)

            # evaluate using reward_function for each task
            for task_id in task_ids:
                task_reward_dict = self.val_reward_fn(test_output_gen_batch, eval=True, task_id=task_id, return_dict=True)
                task_reward_tensor = task_reward_dict[f"task{task_id}_reward_tensor"]
                task_reward_tensors[task_id].append(task_reward_tensor)

                # Compute per-sample scores
                per_sample_scores = task_reward_tensor.sum(-1).cpu().tolist()

                batch_scores[f"task{task_id}_scores"] = per_sample_scores
                extra_info = task_reward_dict.get(f"task{task_id}_reward_extra_info", {})
                for k, v in extra_info.items():
                    batch_reward_extra_infos[k].extend(v)

                if task_id == 1:
                    sample_task1_scores.extend(per_sample_scores)
                elif task_id == 2:
                    sample_task2_scores.extend(per_sample_scores)
                else:
                    sample_task3_scores.extend(per_sample_scores)

            val_feedback_texts = None
            if 2 in task_ids:
                val_feedback_texts = test_output_gen_batch.non_tensor_batch.get('task2_feedback_texts')
                if val_feedback_texts is not None and hasattr(val_feedback_texts, 'tolist'):
                    val_feedback_texts = val_feedback_texts.tolist()
            val_regen_imgs = test_output_gen_batch.non_tensor_batch.get('task3_regen_imgs_pil_list') if 3 in task_ids else None
            self._dump_generations(
                uid=test_output_gen_batch.non_tensor_batch["uid"].tolist(),
                prompt_id=test_output_gen_batch.non_tensor_batch["prompt_id"].tolist(),
                prompt=test_output_gen_batch.non_tensor_batch["prompt"].tolist(),
                gen_imgs_pil_list=test_output_gen_batch.non_tensor_batch.get('task1_gen_imgs_pil_list'),
                feedback_texts=val_feedback_texts,
                regen_imgs_pil_list=val_regen_imgs,
                gts_imgs=[item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_output_gen_batch],
                summarizes=[item.non_tensor_batch.get("reward_model", {}).get("summary", None) for item in test_output_gen_batch],
                gts_tuples=[item.non_tensor_batch.get("reward_model", {}).get("tuple", None) for item in test_output_gen_batch],
                gts_vqas=[item.non_tensor_batch.get("reward_model", {}).get("vqa_question", None) for item in test_output_gen_batch],
                scores=batch_scores,
                reward_extra_infos_dict=batch_reward_extra_infos,
                dump_path=val_data_dir
                )

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
        for task_id in task_ids:
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
        # reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["dummy_tensor"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) # - reward_model_keys
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
                    for task_id in [1]:

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
                            self._submit_rollout_dump(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

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
                            # gen_params (temperature, cfg_weight, txt_top_k, …) are already in
                            # batch.meta_info from image_unified_rollout.generate_sequences.
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
                    self._shutdown_rollout_dump_executor(wait=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        self._shutdown_rollout_dump_executor(wait=True)
