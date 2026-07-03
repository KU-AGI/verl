# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import os
import time
from datetime import datetime
from pprint import pprint
from typing import Any, Optional
import threading
import copy
import queue

import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from recipe.fully_async_policy_image_rl.detach_utils import (
    MetricsAggregator,
    ValidateMetrics,
    assemble_batch_from_rollout_samples,
    _concat_dataprotos_with_meta,
)
from recipe.image_rl.reward import load_reward_manager
from recipe.fully_async_policy_image_rl.message_queue import MessageQueueClient
from recipe.fully_async_policy_image_rl.ray_trainer import FullyAsyncRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
import torch
import numpy as np
from recipe.image_rl.custom_metric_utils import reduce_metrics # custom metric
from recipe.image_rl.reward import compute_reward, compute_reward_async
import asyncio
from verl.protocol import DataProto
from recipe.fully_async_policy_image_rl.replay_buffer import ReplayBuffer

def log_prob_metrics(metrics, task_ids):
    log_prob_info = {}

    for task_id in task_ids:
        task_pos_log_prob = f"actor/task{task_id}_pos_log_prob"
        task_neg_log_prob = f"actor/task{task_id}_neg_log_prob"
        task_pos_log_prob_cnt = f"actor/task{task_id}_pos_log_prob_cnt"
        task_neg_log_prob_cnt = f"actor/task{task_id}_neg_log_prob_cnt"

        if task_pos_log_prob not in metrics:
            continue

        pos_log_prob_sum = np.array(metrics.pop(task_pos_log_prob)).sum()
        pos_log_prob_cnt = np.array(metrics.pop(task_pos_log_prob_cnt)).sum()
        neg_log_prob_sum = np.array(metrics.pop(task_neg_log_prob)).sum()
        neg_log_prob_cnt = np.array(metrics.pop(task_neg_log_prob_cnt)).sum()

        log_prob_info[f"actor/task{task_id}_pos_log_prob_mean"] = pos_log_prob_sum / pos_log_prob_cnt if pos_log_prob_cnt > 0 else 0.0
        log_prob_info[f"actor/task{task_id}_neg_log_prob_mean"] = neg_log_prob_sum / neg_log_prob_cnt if neg_log_prob_cnt > 0 else 0.0

        log_prob_info[f"actor/task{task_id}_log_probs_diff"] = log_prob_info[f"actor/task{task_id}_pos_log_prob_mean"] - log_prob_info[f"actor/task{task_id}_neg_log_prob_mean"]

        total_cnt = pos_log_prob_cnt + neg_log_prob_cnt
        log_prob_info[f"actor/task{task_id}_total_cnt"] = total_cnt
        log_prob_info[f"actor/task{task_id}_pos_ratio"] = pos_log_prob_cnt / total_cnt if total_cnt > 0 else 0.0

    return log_prob_info

@ray.remote(num_cpus=10)
class FullyAsyncTrainer(FullyAsyncRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

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
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = load_reward_manager(
            config, tokenizer, processor, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.val_reward_fn = load_reward_manager(
            config, tokenizer, processor, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # Mirror RayImageGenerationTrainer's rollout-dump initialization.
        # FullyAsyncTrainer does not call the parent __init__, but it now uses
        # the same background dump submission helpers.
        self._rollout_dump_executor = None
        self._rollout_dump_futures = []
        self._rollout_dump_workers = int(self.config.trainer.get("rollout_dump_workers", 8) or 0)
        self._rollout_dump_max_pending = int(
            self.config.trainer.get("rollout_dump_max_pending", max(2, 2 * max(self._rollout_dump_workers, 1))) or 0
        )

        # ==================== fully async config ====================

        self.message_queue_client = None
        self.param_synchronizer = None
        self.async_rollout_mode = False
        self.async_rollout_manager = None

        # Statistics
        # we start from step 1
        self.global_steps = 1
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_samples_processed = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches * config.actor_rollout_ref.rollout.n
        self.compute_prox_log_prob = self.config.async_training.compute_prox_log_prob
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        self.batch_buffer = queue.Queue(maxsize=1)
        self.stop_prefetch = False
        self.prefetch_thread = None

        # Replay buffer
        replay_cfg = config.async_training.get("replay_buffer", {})
        self.use_replay_buffer = replay_cfg.get("enable", False)
        if self.use_replay_buffer:
            task_ids = list(config.actor_rollout_ref.actor.multi_task.get("task_ids", [1]))
            replay_unit = str(replay_cfg.get("unit", "task"))
            if replay_unit == "step":
                stage2_cfg = config.actor_rollout_ref.rollout.get("stage2", {})
                raw_stage2_steps = stage2_cfg.get("steps", [1, 2, 3, 4, 5])
                if isinstance(raw_stage2_steps, str):
                    stage2_steps = [int(s.strip()) for s in raw_stage2_steps.strip().strip("[]").split(",") if s.strip()]
                else:
                    stage2_steps = [int(s) for s in list(raw_stage2_steps)]
                buffer_ids = [ReplayBuffer.encode_lane_id(1, s) for s in [1, 2, 3, 4, 5]]
                if bool(stage2_cfg.get("enable", False)):
                    buffer_ids.extend(ReplayBuffer.encode_lane_id(2, s) for s in stage2_steps)
            else:
                buffer_ids = task_ids
            raw_thresholds = replay_cfg.get("score_thresholds", {})
            score_thresholds = {int(k): float(v) for k, v in raw_thresholds.items()}
            raw_std_thresholds = replay_cfg.get("score_std_thresholds", {})
            score_std_thresholds = {int(k): float(v) for k, v in raw_std_thresholds.items()}
            self.replay_buffer = ReplayBuffer(
                task_ids=buffer_ids,
                score_thresholds=score_thresholds,
                score_std_thresholds=score_std_thresholds,
                max_size_per_task=replay_cfg.get("max_size_per_task", -1),
                max_version_gap=replay_cfg.get("max_version_gap", -1),
                max_use_count=replay_cfg.get("max_use_count", -1),
                filter_mode=replay_cfg.get("filter_mode", "mean"),
                reward_history_size=replay_cfg.get("reward_history_size", 100),
                max_quantile=replay_cfg.get("max_quantile", 0.25),
                std_quantile=replay_cfg.get("std_quantile", 0.25),
            )
            # Feeder thread state
            self._feeder_stop = False
            self._feeder_terminated = False
            self._feeder_thread = None
            self._feeder_task_ids = buffer_ids
            self._physical_task_ids = task_ids
            print(
                f"[FullyAsyncTrainer] ReplayBuffer enabled: unit={replay_unit}, ids={buffer_ids}, "
                f"max_version_gap={self.replay_buffer.max_version_gap}, "
                f"max_size_per_task={self.replay_buffer.max_size_per_task}, "
                f"filter_mode={self.replay_buffer.filter_mode}, "
                f"score_thresholds={score_thresholds}, "
                f"reward_history_size={self.replay_buffer.reward_history_size}, "
                f"max_quantile={self.replay_buffer.max_quantile}, "
                f"std_quantile={self.replay_buffer.std_quantile}"
            )


    def init_workers(self):
        """Initialize trainer workers.

        The fully async trainer consumes rollout samples from MessageQueue and
        does not own rollout workers. Avoid the base init path that tries to
        create an async rollout manager with a missing rollout worker group.
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()

    @staticmethod
    def _step_id_to_task_id(step_id: int) -> int:
        return ReplayBuffer.physical_task_id_for_step(step_id)

    @staticmethod
    def _lane_id(stage_id: int, step_id: int) -> int:
        return ReplayBuffer.encode_lane_id(stage_id, step_id)

    def _replay_unit(self) -> str:
        replay_cfg = self.config.async_training.get("replay_buffer", {})
        return str(replay_cfg.get("unit", "task"))

    @staticmethod
    def _as_int_list(value, default: list[int]) -> list[int]:
        if value is None:
            return list(default)
        if isinstance(value, str):
            stripped = value.strip().strip("[]")
            if not stripped:
                return []
            return [int(x.strip()) for x in stripped.split(",") if x.strip()]
        return [int(s) for s in list(value)]

    def _configured_train_stages(self) -> list[int]:
        stages = self.config.async_training.get("train_stages", [1])
        return self._as_int_list(stages, [1])

    def _configured_stage2_steps(self) -> list[int]:
        stage2_cfg = self.config.actor_rollout_ref.rollout.get("stage2", {})
        steps = stage2_cfg.get("steps", [1, 2, 3, 4, 5])
        return self._as_int_list(steps, [1, 2, 3, 4, 5])

    def _all_replay_lanes(self) -> list[int]:
        if self._replay_unit() != "step":
            return list(self.config.actor_rollout_ref.actor.multi_task.get("task_ids", [1]))
        lanes = [self._lane_id(1, step_id) for step_id in [1, 2, 3, 4, 5]]
        stage2_cfg = self.config.actor_rollout_ref.rollout.get("stage2", {})
        if bool(stage2_cfg.get("enable", False)):
            lanes.extend(self._lane_id(2, step_id) for step_id in self._configured_stage2_steps())
        return lanes

    def _active_lane_ids(self) -> list[int]:
        if self._replay_unit() != "step":
            return list(self.config.actor_rollout_ref.actor.multi_task.get("task_ids", [1]))
        train_stages = set(self._configured_train_stages())
        lanes = []
        if 1 in train_stages:
            lanes.extend(self._lane_id(1, step_id) for step_id in [1, 2, 3, 4, 5])
        if 2 in train_stages:
            lanes.extend(self._lane_id(2, step_id) for step_id in self._configured_stage2_steps())
        return lanes

    def _unit_loss_weight(self, stage_id: int, step_id: int) -> float:
        actor_cfg = self.config.actor_rollout_ref.actor
        step_weights = list(actor_cfg.get("step_weights", []))
        if step_weights and 1 <= int(step_id) <= len(step_weights):
            step_w = float(step_weights[int(step_id) - 1])
        else:
            task_weights = list(actor_cfg.get("multi_task", {}).get("task_weights", [1.0, 1.0, 1.0]))
            task_id = self._step_id_to_task_id(step_id)
            step_w = float(task_weights[task_id - 1]) if task_id - 1 < len(task_weights) else 1.0
        stage_weights = actor_cfg.get("stage_loss_weights", {})
        stage_w = float(stage_weights.get(str(stage_id), stage_weights.get(int(stage_id), 1.0)))
        return step_w * stage_w

    @staticmethod
    def _terminal_reward_tensor(response_mask: torch.Tensor, rewards: list[float]) -> torch.Tensor:
        out = torch.zeros_like(response_mask, dtype=torch.float32)
        for i, reward in enumerate(rewards):
            valid = torch.where(response_mask[i] > 0)[0]
            if len(valid) > 0:
                out[i, valid[-1]] = float(reward)
        return out

    def _local_task2_reward_tensor(self, batch: DataProto, step_id: int, target_mask: torch.Tensor) -> torch.Tensor | None:
        extras = (getattr(batch, "meta_info", None) or {}).get("task2_reward_extra_info", {}) or {}
        local_reward_keys = {
            2: "task2_prompt_to_tuple_reward",
            3: "task2_tuple_to_vqa_reward",
            4: "task2_vqa_to_feedback_reward",
        }
        key = local_reward_keys.get(int(step_id))
        values = extras.get(key) if key is not None else None
        if values is None:
            return None
        if hasattr(values, "tolist"):
            values = values.tolist()
        rewards = [float(v if v is not None else 0.0) for v in list(values)[:len(batch)]]
        if len(rewards) < len(batch):
            rewards.extend([0.0] * (len(batch) - len(rewards)))
        return self._terminal_reward_tensor(target_mask.long(), rewards)

    def _make_step_training_batch(self, batch: DataProto, stage_id: int, step_id: int) -> DataProto | None:
        if batch is None or len(batch) == 0:
            return None
        task_id = self._step_id_to_task_id(step_id)
        score_key = f"task{task_id}_token_level_scores"
        if score_key not in batch.batch:
            return None
        out = self._clone_dataproto(batch)
        n = len(out)
        out.non_tensor_batch["stage_id"] = np.full(n, int(stage_id), dtype=np.int64)
        out.non_tensor_batch["step_id"] = np.full(n, int(step_id), dtype=np.int64)
        out.non_tensor_batch["physical_task_id"] = np.full(n, int(task_id), dtype=np.int64)
        out.non_tensor_batch["lane_id"] = np.full(n, self._lane_id(stage_id, step_id), dtype=np.int64)
        out.non_tensor_batch["unit_loss_weight"] = np.full(n, self._unit_loss_weight(stage_id, step_id), dtype=np.float32)
        out.batch["unit_loss_weights"] = torch.full((n,), self._unit_loss_weight(stage_id, step_id), dtype=torch.float32)
        out.batch[f"task{task_id}_unit_loss_weights"] = out.batch["unit_loss_weights"]

        if int(stage_id) == 2 and int(step_id) == 1:
            local_key = "task1_local_token_level_scores"
            if local_key in out.batch:
                out.batch["task1_token_level_scores"] = out.batch[local_key]
                out.meta_info["task1_token_level_scores"] = out.batch[local_key]

        if task_id == 2:
            if "task2_response_mask" not in out.batch:
                return None
            response_mask = out.batch["task2_response_mask"]
            if int(stage_id) == 2 and "task2_stage2_target_mask" in out.batch:
                target_mask = (out.batch["task2_stage2_target_mask"] > 0) & (response_mask > 0)
            elif "task2_segment_mask" in out.batch:
                target_mask = (out.batch["task2_segment_mask"] == int(step_id)) & (response_mask > 0)
            else:
                return None
            if target_mask.sum().item() == 0:
                return None
            target_mask = target_mask.long()
            out.batch["task2_loss_mask"] = target_mask
            out.batch["task2_segment_mask"] = target_mask * int(step_id)
            if int(stage_id) == 2:
                local_scores = self._local_task2_reward_tensor(out, step_id, target_mask)
                if local_scores is not None:
                    out.batch["task2_token_level_scores"] = local_scores
        return out

    def _build_step_batches_from_task_batch(self, task_id: int, batch: DataProto) -> list[tuple[int, int, int, DataProto]]:
        if batch is None:
            return []
        phase_arr = batch.non_tensor_batch.get("phase")
        if phase_arr is None:
            phases = np.ones(len(batch), dtype=np.int64)
        else:
            phases = np.asarray(phase_arr, dtype=np.int64)
        out = []
        stage_steps = {1: [1, 2, 3, 4, 5], 2: self._configured_stage2_steps()}
        for stage_id, steps in stage_steps.items():
            idx = np.where(phases == int(stage_id))[0].tolist()
            if not idx:
                continue
            phase_batch = self._slice_batch(batch, idx)
            for step_id in steps:
                if self._step_id_to_task_id(step_id) != int(task_id):
                    continue
                source_batch = phase_batch
                if int(stage_id) == 2 and "stage2_target_step" in phase_batch.non_tensor_batch:
                    target_steps = np.asarray(phase_batch.non_tensor_batch["stage2_target_step"], dtype=np.int64)
                    step_idx = np.where(target_steps == int(step_id))[0].tolist()
                    if not step_idx:
                        continue
                    source_batch = self._slice_batch(phase_batch, step_idx)
                step_batch = self._make_step_training_batch(source_batch, stage_id, step_id)
                if step_batch is not None:
                    out.append((self._lane_id(stage_id, step_id), stage_id, step_id, step_batch))

        stage2_cfg = self.config.actor_rollout_ref.rollout.get("stage2", {})
        if (
            int(task_id) == 1
            and bool(stage2_cfg.get("enable", False))
            and 1 in self._configured_stage2_steps()
        ):
            phase1_idx = np.where(phases == 1)[0].tolist()
            if phase1_idx:
                phase1_batch = self._slice_batch(batch, phase1_idx)
                step_batch = self._make_step_training_batch(phase1_batch, 2, 1)
                if step_batch is not None:
                    out.append((self._lane_id(2, 1), 2, 1, step_batch))
        return out

    @staticmethod
    def _slice_batch(batch: DataProto, idxs: list[int]) -> DataProto:
        from recipe.fully_async_policy_image_rl.detach_utils import _slice_dataproto_with_meta
        return _slice_dataproto_with_meta(batch, idxs)

    def _step_lane_quotas_for_step(
        self,
        step_id: int,
        lane_ids: list[int],
        allow_available_fallback: bool = False,
    ) -> dict[int, int]:
        step_lanes = [lid for lid in lane_ids if ReplayBuffer.decode_lane_id(lid)[1] == int(step_id)]
        if not step_lanes:
            return {}
        stage_ratios_cfg = self.config.async_training.get("stage_ratios", {})
        raw_weights = []
        for lane_id in step_lanes:
            stage_id, _step_id = ReplayBuffer.decode_lane_id(lane_id)
            raw_weights.append(float(stage_ratios_cfg.get(str(stage_id), stage_ratios_cfg.get(int(stage_id), 1.0))))
        total = sum(raw_weights) if sum(raw_weights) > 0 else float(len(raw_weights))
        quotas = {}
        remaining = self.required_samples
        for i, (lane_id, weight) in enumerate(zip(step_lanes, raw_weights)):
            if i == len(step_lanes) - 1:
                q = remaining
            else:
                q = max(1, int(round(self.required_samples * (weight / total))))
                q = min(q, remaining)
            quotas[lane_id] = q
            remaining -= q
        if not allow_available_fallback:
            return quotas

        sizes = {lane_id: self.replay_buffer.task_size(lane_id) for lane_id in step_lanes}
        if all(sizes.get(lane_id, 0) >= quota for lane_id, quota in quotas.items()):
            return quotas
        if sum(sizes.values()) < self.required_samples:
            return {}

        fallback_lanes = [lane_id for lane_id in step_lanes if sizes.get(lane_id, 0) > 0]
        if not fallback_lanes:
            return {}
        fallback_weights = [
            float(stage_ratios_cfg.get(str(ReplayBuffer.decode_lane_id(lane_id)[0]), stage_ratios_cfg.get(ReplayBuffer.decode_lane_id(lane_id)[0], 1.0)))
            for lane_id in fallback_lanes
        ]
        total_weight = sum(fallback_weights) if sum(fallback_weights) > 0 else float(len(fallback_lanes))
        fallback_quotas = {}
        remaining = self.required_samples
        for i, (lane_id, weight) in enumerate(zip(fallback_lanes, fallback_weights)):
            if i == len(fallback_lanes) - 1:
                q = remaining
            else:
                q = max(1, int(round(self.required_samples * (weight / total_weight))))
            q = min(q, sizes[lane_id], remaining)
            fallback_quotas[lane_id] = q
            remaining -= q

        while remaining > 0:
            candidates = [lane_id for lane_id in fallback_lanes if sizes[lane_id] > fallback_quotas.get(lane_id, 0)]
            if not candidates:
                return {}
            for lane_id in candidates:
                if remaining <= 0:
                    break
                capacity = sizes[lane_id] - fallback_quotas.get(lane_id, 0)
                add = min(capacity, remaining)
                fallback_quotas[lane_id] = fallback_quotas.get(lane_id, 0) + add
                remaining -= add

        return {lane_id: quota for lane_id, quota in fallback_quotas.items() if quota > 0}

    def _sample_step_batches(self, metrics: dict, timing_raw: dict) -> tuple[dict[int, DataProto], dict[int, dict]]:
        active_lanes = self._active_lane_ids()
        step_ids = sorted({ReplayBuffer.decode_lane_id(lid)[1] for lid in active_lanes})
        step_batches: dict[int, DataProto] = {}
        step_timings: dict[int, dict] = {}
        for step_id in step_ids:
            lane_batches = []
            step_timing = {}
            desired_quotas = self._step_lane_quotas_for_step(step_id, active_lanes)
            quotas = self._step_lane_quotas_for_step(step_id, active_lanes, allow_available_fallback=True)
            for lane_id, desired_quota in desired_quotas.items():
                actual_quota = quotas.get(lane_id, 0)
                if actual_quota < desired_quota:
                    stage_id, lane_step_id = ReplayBuffer.decode_lane_id(lane_id)
                    shortfall = desired_quota - actual_quota
                    metrics[f"replay/stage{stage_id}_step{lane_step_id}_quota_shortfall"] = shortfall
                    print(
                        f"[FullyAsyncTrainer] lane stage{stage_id}/step{lane_step_id}: "
                        f"quota fallback {actual_quota}/{desired_quota}"
                    )
            for lane_id, quota in quotas.items():
                with marked_timer("replay/sample_from_buffer", step_timing):
                    lane_batch = self.replay_buffer.sample_task(lane_id, quota, self.current_param_version)
                stage_id, lane_step_id = ReplayBuffer.decode_lane_id(lane_id)
                if lane_batch is None:
                    print(f"[FullyAsyncTrainer] lane stage{stage_id}/step{lane_step_id}: buffer empty after sample")
                    continue
                lane_batches.append(lane_batch)
                metrics[f"replay/stage{stage_id}_step{lane_step_id}_sampled"] = len(lane_batch)
                zero_stats = self.replay_buffer.zero_std_stats_per_task().get(lane_id, {})
                seen = int(zero_stats.get("seen_groups", 0))
                dropped = int(zero_stats.get("dropped_groups", 0))
                metrics[f"replay/stage{stage_id}_step{lane_step_id}_zero_std_seen_groups"] = seen
                metrics[f"replay/stage{stage_id}_step{lane_step_id}_zero_std_dropped_groups"] = dropped
                metrics[f"replay/stage{stage_id}_step{lane_step_id}_zero_std_dropped_rows"] = int(zero_stats.get("dropped_rows", 0))
                metrics[f"replay/stage{stage_id}_step{lane_step_id}_zero_std_drop_ratio"] = float(dropped / seen) if seen > 0 else 0.0
            if not lane_batches:
                continue
            step_batch = _concat_dataprotos_with_meta(lane_batches) if len(lane_batches) > 1 else lane_batches[0]
            if len(step_batch) > self.required_samples:
                step_batch = self._slice_batch(step_batch, list(range(self.required_samples)))
            elif len(step_batch) < self.required_samples:
                print(f"[FullyAsyncTrainer] step{step_id}: sampled {len(step_batch)}/{self.required_samples} rows")
            task_id = self._step_id_to_task_id(step_id)
            step_batch.batch["task_id"] = torch.tensor([task_id for _ in range(len(step_batch))], dtype=int)
            step_batches[step_id] = step_batch
            step_timings[step_id] = step_timing
        return step_batches, step_timings

    def _rename_task_batch_to_step(self, batch: DataProto, step_id: int) -> DataProto:
        task_id = self._step_id_to_task_id(step_id)
        task_prefix = f"task{task_id}_"
        step_prefix = f"step{step_id}_"
        tensors = {}
        non_tensors = {}
        meta_info = {}

        for key, value in batch.batch.items():
            if key == "task_id":
                continue
            if key.startswith(task_prefix):
                tensors[step_prefix + key[len(task_prefix):]] = value
        if f"task{task_id}_unit_loss_weights" not in batch.batch.keys() and "unit_loss_weights" in batch.batch.keys():
            tensors[f"step{step_id}_unit_loss_weights"] = batch.batch["unit_loss_weights"]
        if "stage_id" in batch.non_tensor_batch:
            tensors[f"step{step_id}_stage_ids"] = torch.as_tensor(
                np.asarray(batch.non_tensor_batch["stage_id"], dtype=np.int64), dtype=torch.long
            )
        if "lane_id" in batch.non_tensor_batch:
            tensors[f"step{step_id}_lane_ids"] = torch.as_tensor(
                np.asarray(batch.non_tensor_batch["lane_id"], dtype=np.int64), dtype=torch.long
            )

        for key, value in batch.non_tensor_batch.items():
            out_key = step_prefix + key[len(task_prefix):] if key.startswith(task_prefix) else f"step{step_id}_{key}"
            non_tensors[out_key] = value
        if "uid" in batch.non_tensor_batch and "uid" not in non_tensors:
            non_tensors["uid"] = batch.non_tensor_batch["uid"]

        for key, value in batch.meta_info.items():
            if key.startswith(task_prefix):
                meta_info[step_prefix + key[len(task_prefix):]] = value
            elif key in self._TRAIN_RPC_META_KEYS:
                meta_info[key] = value

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def _merge_step_batches(self, step_batches: dict[int, DataProto]) -> DataProto:
        tensors = {}
        non_tensors = {}
        meta_info = {}
        step_ids = sorted(step_batches.keys())
        for step_id in step_ids:
            step_view = self._rename_task_batch_to_step(step_batches[step_id], step_id)
            tensors.update(dict(step_view.batch.items()))
            non_tensors.update(dict(step_view.non_tensor_batch.items()))
            for key, value in step_view.meta_info.items():
                meta_info.setdefault(key, value)

        if "uid" not in non_tensors:
            for step_id in step_ids:
                uid_key = f"step{step_id}_uid"
                if uid_key in non_tensors:
                    non_tensors["uid"] = non_tensors[uid_key]
                    break

        total = None
        for step_id in step_ids:
            att_key = f"step{step_id}_attention_mask"
            resp_key = f"step{step_id}_response_mask"
            if att_key in tensors:
                t = tensors[att_key].sum(-1)
                if resp_key in tensors:
                    t = t + tensors[resp_key].sum(-1)
                total = t if total is None else total + t
        if total is not None:
            meta_info["global_token_num"] = total.tolist()
        meta_info["step_mode_step_ids"] = step_ids
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def _prepare_step_combined_batch(
        self,
        step_batches: dict[int, DataProto],
        metrics: dict,
        timing_raw: dict,
        step_timings: dict[int, dict],
        should_log_rollout: bool = False,
        rollout_data_dir: str | None = None,
    ) -> tuple[DataProto, dict[int, DataProto]]:
        processed_steps: dict[int, DataProto] = {}
        for step_id in sorted(step_batches.keys()):
            task_id = self._step_id_to_task_id(step_id)
            step_timing = step_timings.get(step_id, {})
            step_batch = step_batches[step_id]
            step_batch.batch["task_id"] = torch.tensor([task_id for _ in range(len(step_batch))], dtype=int)
            if self.use_rm:
                step_batch = self._process_batch_common(
                    step_batch,
                    metrics,
                    step_timing,
                    self.local_trigger_step if self.compute_prox_log_prob else None,
                    task_id,
                    skip_old_log_prob=True,
                    skip_ref_values_adv=True,
                )
            task_batches = {task_id: step_batch}
            task_batches = self._compute_old_log_probs_for_task_batches(
                task_batches,
                metrics,
                step_timing,
                self.local_trigger_step if self.compute_prox_log_prob else None,
            )
            task_batches = self._compute_ref_log_probs_for_task_batches(task_batches, step_timing)
            step_batch = self._process_batch_common(
                task_batches[task_id],
                metrics,
                step_timing,
                None,
                task_id,
                skip_old_log_prob=True,
                skip_reward=True,
                skip_ref_log_prob=True,
            )
            self._restore_reward_extra_infos_for_logging(step_batch, task_id)
            processed_steps[step_id] = step_batch

            if should_log_rollout and rollout_data_dir:
                task_reward_extra = {k: v for k, v in step_batch.meta_info.items()}
                task_rollout_dir = os.path.join(rollout_data_dir, f"step{step_id}")
                self._submit_rollout_dump(step_batch, task_reward_extra, step_timing, task_rollout_dir)

            use_counts = step_batch.non_tensor_batch.get("entry_use_count")
            if use_counts is not None:
                metrics[f"replay/step{step_id}_mean_use_count"] = float(np.mean(use_counts))
                metrics[f"replay/step{step_id}_min_use_count"] = float(np.min(use_counts))
                metrics[f"replay/step{step_id}_max_use_count"] = float(np.max(use_counts))

        return self._merge_step_batches(processed_steps), processed_steps

    # ------------------------------------------------------------------
    # Background buffer feeder (replay-buffer mode only)
    # ------------------------------------------------------------------

    def _start_buffer_feeder(self):
        """Start background thread that continuously drains queue into replay buffer."""
        self._feeder_stop = False
        self._feeder_terminated = False
        self._feeder_thread = threading.Thread(target=self._run_buffer_feeder, daemon=True)
        self._feeder_thread.start()
        print("[FullyAsyncTrainer] Buffer feeder thread started.")

    def _run_buffer_feeder(self):
        """Background: blocking drain queue -> assemble -> push to replay buffer."""
        balance_fn = self._balance_batch if self.config.trainer.balance_batch else None

        while not self._feeder_stop:
            try:
                result = self.message_queue_client.get_sample_sync()
                if result is None:
                    print("[BufferFeeder] Termination signal (result is None).")
                    self._feeder_terminated = True
                    break
                sample_data, queue_len = result
                if sample_data is None:
                    print("[BufferFeeder] Termination signal (sample_data is None).")
                    self._feeder_terminated = True
                    break

                deserialized = ray.cloudpickle.loads(sample_data)

                # Multi-turn: prefer per-task batches produced by the orchestrator
                # so task1 / task2 / task3 buffers each receive their own distinct
                # rollouts (task1 = unique rollouts, task2/task3 = per-turn).
                # Legacy: fall back to a single assembled batch shared across tasks.
                multi_turn_task_batches = getattr(deserialized, "task_batches", None)

                def _assemble_for_tid(tid: int) -> Optional[DataProto]:
                    if multi_turn_task_batches is not None:
                        task_dp = multi_turn_task_batches.get(tid)
                        if task_dp is None:
                            return None
                        # Reuse assemble_batch's meta_info aggregation + token_lens
                        # logic by temporarily swapping rs.full_batch to the per-task
                        # DP for this assembly call.
                        original_full = deserialized.full_batch
                        deserialized.full_batch = task_dp
                        try:
                            return assemble_batch_from_rollout_samples(
                                [deserialized], self.tokenizer, self.config, balance_fn
                            )
                        finally:
                            deserialized.full_batch = original_full
                    # Legacy path: same batch for all tasks.
                    return assemble_batch_from_rollout_samples(
                        [deserialized], self.tokenizer, self.config, balance_fn
                    )

                if self._replay_unit() == "step":
                    replay_cfg = self.config.async_training.get("replay_buffer", {})
                    stage2_cfg = self.config.actor_rollout_ref.rollout.get("stage2", {})
                    drop_stage2_zero_std = bool(stage2_cfg.get("drop_zero_std", replay_cfg.get("drop_zero_std", False)))
                    for tid in self._physical_task_ids:
                        batch = _assemble_for_tid(tid)
                        if batch is None:
                            continue
                        for lane_id, stage_id, step_id, step_batch in self._build_step_batches_from_task_batch(tid, batch):
                            before_zero_stats = self.replay_buffer.zero_std_stats_per_task().get(lane_id, {})
                            before_dropped = int(before_zero_stats.get("dropped_groups", 0))
                            stored = self.replay_buffer.push(
                                step_batch,
                                tid,
                                lane_id=lane_id,
                                stage_id=stage_id,
                                step_id=step_id,
                                drop_zero_std=(
                                    drop_stage2_zero_std
                                    and int(stage_id) == 2
                                    and int(step_id) != 1
                                ),
                            )
                            zero_stats = self.replay_buffer.zero_std_stats_per_task().get(lane_id, {})
                            dropped_delta = int(zero_stats.get("dropped_groups", 0)) - before_dropped
                            if stored > 0:
                                n = self.config.actor_rollout_ref.rollout.n
                                required_groups = max(1, self.required_samples // n)
                                seen = int(zero_stats.get("seen_groups", 0))
                                dropped = int(zero_stats.get("dropped_groups", 0))
                                drop_ratio = float(dropped / seen) if seen > 0 else 0.0
                                entry_count = self.replay_buffer.entries_per_task()[lane_id]
                                entry_cap = self.replay_buffer.max_size_per_task
                                entry_cap_text = f"{entry_count}/{entry_cap}" if entry_cap > 0 else f"{entry_count}/unbounded"
                                print(
                                    f"[BufferFeeder] stage{stage_id}/step{step_id}: pushed {stored} rows, "
                                    f"buffer={self.replay_buffer.task_size(lane_id)}/{self.required_samples} rows, "
                                    f"groups={entry_count}/{required_groups} groups, "
                                    f"entries={entry_cap_text} capacity, "
                                    f"zero_std_drop={dropped}/{seen} ({drop_ratio:.3f}), "
                                    f"mq_len={queue_len}"
                                )
                            elif dropped_delta > 0:
                                seen = int(zero_stats.get("seen_groups", 0))
                                dropped = int(zero_stats.get("dropped_groups", 0))
                                drop_ratio = float(dropped / seen) if seen > 0 else 0.0
                                entry_count = self.replay_buffer.entries_per_task()[lane_id]
                                entry_cap = self.replay_buffer.max_size_per_task
                                entry_cap_text = f"{entry_count}/{entry_cap}" if entry_cap > 0 else f"{entry_count}/unbounded"
                                print(
                                    f"[BufferFeeder] stage{stage_id}/step{step_id}: zero-std dropped {dropped_delta} groups, "
                                    f"entries={entry_cap_text} capacity, "
                                    f"zero_std_drop={dropped}/{seen} ({drop_ratio:.3f}), mq_len={queue_len}"
                                )
                else:
                    # Push to all task buffers (lock is inside replay_buffer)
                    for tid in self._feeder_task_ids:
                        batch = _assemble_for_tid(tid)
                        if batch is None:
                            continue
                        stored = self.replay_buffer.push(batch, tid)
                        if stored > 0:
                            n = self.config.actor_rollout_ref.rollout.n
                            required_groups = self.required_samples // n
                            print(
                                f"[BufferFeeder] task{tid}: pushed {stored} rows, "
                                f"buffer={self.replay_buffer.task_size(tid)}/{self.required_samples} rows, "
                                f"groups={self.replay_buffer.entries_per_task()[tid]}/{required_groups} groups, "
                                f"mq_len={queue_len}"
                            )
            except Exception as e:
                print(f"[BufferFeeder Error] {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

    def _wait_for_buffer(self, min_samples: int | None = None) -> bool:
        """Block until replay buffer has enough samples for all tasks.

        Returns False if feeder terminated before buffer is ready.
        """
        if min_samples is None:
            min_samples = self.required_samples

        while True:
            if self._replay_unit() == "step":
                active_lanes = self._active_lane_ids()
                step_ids = sorted({ReplayBuffer.decode_lane_id(lid)[1] for lid in active_lanes})
                ready = True
                for step_id in step_ids:
                    quotas = self._step_lane_quotas_for_step(step_id, active_lanes, allow_available_fallback=True)
                    if not quotas:
                        ready = False
                        break
                    for lane_id, quota in quotas.items():
                        if self.replay_buffer.task_size(lane_id) < quota:
                            ready = False
                            break
                    if not ready:
                        break
                if ready:
                    return True
                if self._feeder_terminated:
                    return ready
            else:
                min_buf = min(
                    self.replay_buffer.task_size(tid)
                    for tid in self._feeder_task_ids
                )
                if min_buf >= min_samples:
                    return True
                if self._feeder_terminated:
                    # Feeder stopped — check once more
                    min_buf = min(
                        self.replay_buffer.task_size(tid)
                        for tid in self._feeder_task_ids
                    )
                    return min_buf >= min_samples
            time.sleep(0.5)

    def _run_prefetch(self):
        print("[FullyAsyncTrainer] Prefetch thread started.")
        while not self.stop_prefetch:
            try:
                epoch, batch = self._get_samples_from_queue()

                if batch is None:
                    self.batch_buffer.put((None, None))
                    break

                # 수집된 128개 묶음을 버퍼에 투척 (버퍼가 차있으면 여기서 대기)
                self.batch_buffer.put((epoch, batch))
            except Exception as e:
                print(f"[Prefetch Error] {e}")
                time.sleep(1)

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    def set_parameter_synchronizer(self, param_synchronizer):
        """Set parameter synchronizer"""
        self.param_synchronizer = param_synchronizer

    def set_total_train_steps(self, total_train_steps):
        self.total_train_steps = total_train_steps
        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """Collect samples from the message queue (non-replay path only).

        Blocks until required_samples are collected. Drops stale samples
        exceeding staleness_threshold.
        """
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} rows from queue",
            flush=True,
        )

        staleness_threshold = self.config.async_training.get("staleness_threshold", 1)

        consumer_start = time.time()
        queue_samples = []
        current_rows = 0
        queue_len = 0
        dropped_rows = 0

        while current_rows < self.required_samples:
            result = self.message_queue_client.get_sample_sync()

            if result is None:
                print(f"[FullyAsyncTrainer] Termination signal. Collected {current_rows}/{self.required_samples} rows.")
                break

            sample_data, queue_len = result
            if sample_data is None:
                print("[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection.")
                break

            deserialized_sample = ray.cloudpickle.loads(sample_data)

            sample_ver = deserialized_sample.param_version
            version_gap = self.current_param_version - sample_ver

            # version_gap > 0이면 on_policy
            # default는 version_gap > staleness_threshold
            if version_gap > staleness_threshold:
                num_rows = len(deserialized_sample.full_batch)
                dropped_rows += num_rows
                continue

            num_rows_in_sample = len(deserialized_sample.full_batch)

            queue_samples.append(deserialized_sample)
            current_rows += num_rows_in_sample

            print(
                f"[FullyAsyncTrainer] Progress: {current_rows}/{self.required_samples} rows "
                f"({len(queue_samples)} groups collected). mq_len: {queue_len}"
            )

        consumer_end = time.time()

        if not queue_samples:
            print(f"[FullyAsyncTrainer] No rows collected")
            return None, None

        if current_rows < self.required_samples:
            print(f"[FullyAsyncTrainer] Not enough rows collected: {current_rows}/{self.required_samples}")
            return None, None

        total_wait_time = consumer_end - consumer_start

        print(
            f"[FullyAsyncTrainer] Collection completed: {current_rows} rows from {len(queue_samples)} groups. "
            f"Wait time: {total_wait_time:.2f}s. "
            f"Dropped sample: {dropped_rows}"
        )

        balance_fn = self._balance_batch if self.config.trainer.balance_batch else None
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        assembled_task_batches = self._assemble_task_batches_from_samples(queue_samples, balance_fn)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        batch.meta_info["fully_async/total_sample_count"] = current_rows
        batch.meta_info["fully_async/batch_size"] = current_rows

        return 0, batch, assembled_task_batches

    @staticmethod
    def _clone_dataproto(dp: DataProto) -> DataProto:
        return DataProto(
            batch=dp.batch.clone() if getattr(dp, "batch", None) is not None else None,
            non_tensor_batch=copy.deepcopy(getattr(dp, "non_tensor_batch", None)),
            meta_info=copy.deepcopy(getattr(dp, "meta_info", None)),
        )

    def _assemble_task_batches_from_samples(self, rollout_samples: list, balance_fn=None) -> dict[int, DataProto]:
        task_ids = list(self.config.actor_rollout_ref.actor.multi_task.get("task_ids", [1]))
        assembled: dict[int, DataProto] = {}

        for tid in task_ids:
            selected_samples = []
            originals = []
            for rs in rollout_samples:
                task_batches = getattr(rs, "task_batches", None)
                if task_batches is None:
                    continue
                task_dp = task_batches.get(tid)
                if task_dp is None:
                    continue
                originals.append((rs, rs.full_batch))
                rs.full_batch = task_dp
                selected_samples.append(rs)

            if not selected_samples:
                for rs, original_full in originals:
                    rs.full_batch = original_full
                continue

            try:
                assembled[tid] = assemble_batch_from_rollout_samples(
                    selected_samples, self.tokenizer, self.config, balance_fn
                )
            finally:
                for rs, original_full in originals:
                    rs.full_batch = original_full

        return assembled

    def _create_actor_rollout_classes(self):
        # create actor
        for role in [Role.Actor]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(Role.Actor)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    async def _init_async_rollout_manager(self):
        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from recipe.fully_async_policy_image_rl.agent_loop import FullyAsyncAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
        )
        num_servers = len(self.async_rollout_manager.server_handles)
        self.server_token_q = asyncio.Queue()
        for sid in range(num_servers):
            self.server_token_q.put_nowait(sid)

        self.server_applied_versions = [-1] * num_servers

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.param_synchronizer is None:
            raise ValueError("param_synchronizer client not set. Call set_parameter_synchronizer() first.")

        from recipe.image_rl.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.max_steps_duration = 0

        # get validate data before training
        self._log_validation_data()

        # Start background feeder if replay buffer is enabled
        if self.use_replay_buffer:
            self._start_buffer_feeder()
        while True:
            metrics = {}
            timing_raw = {}

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw, color="red"):
                    training_start_time = time.time()
                    task_ids = list(self.config.actor_rollout_ref.actor.multi_task.get("task_ids", [1]))

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    should_log_rollout = (
                        rollout_data_dir
                        and self.config.trainer.rollout_freq > 0
                        and self.global_steps % self.config.trainer.rollout_freq == 0
                    )

                    task_batches: dict[int, DataProto] = {}
                    task_timings: dict[int, dict] = {}
                    combined_batch: DataProto | None = None
                    processed_step_batches: dict[int, DataProto] = {}

                    if self.use_replay_buffer:
                        # ---- Replay path: sample from buffer (feeder pushes in background) ----
                        with marked_timer("replay/wait", timing_raw):
                            buffer_ready = self._wait_for_buffer()
                        if not buffer_ready:
                            print("[FullyAsyncTrainer] Buffer feeder terminated and buffer insufficient, stopping.")
                            break

                        if self._replay_unit() == "step":
                            step_batches, step_timings = self._sample_step_batches(metrics, timing_raw)
                            if not step_batches:
                                break
                            combined_batch, processed_step_batches = self._prepare_step_combined_batch(
                                step_batches,
                                metrics,
                                timing_raw,
                                step_timings,
                                should_log_rollout=should_log_rollout,
                                rollout_data_dir=rollout_data_dir,
                            )
                            task_batches = processed_step_batches
                            task_timings = step_timings
                        else:
                            for task_id in task_ids:
                                task_timing = {}
                                with marked_timer("replay/sample_from_buffer", task_timing):
                                    task_batch = self.replay_buffer.sample_task(
                                        task_id, self.required_samples, self.current_param_version
                                    )
                                if task_batch is None:
                                    print(f"[FullyAsyncTrainer] task{task_id}: buffer empty after sample, skipping")
                                    continue

                                buf_size = self.replay_buffer.task_size(task_id)
                                print(
                                    f"[FullyAsyncTrainer] Replay sample task{task_id}: "
                                    f"sampled={len(task_batch)}, buffer_remaining={buf_size}"
                                )
                                use_counts = task_batch.non_tensor_batch.get("entry_use_count")
                                if use_counts is not None:
                                    metrics[f"replay/task{task_id}_mean_use_count"] = float(np.mean(use_counts))
                                    metrics[f"replay/task{task_id}_min_use_count"] = float(np.min(use_counts))
                                    metrics[f"replay/task{task_id}_max_use_count"] = float(np.max(use_counts))

                                task_batch.batch["task_id"] = torch.tensor([task_id for _ in range(len(task_batch))], dtype=int)
                                if self.use_rm:
                                    task_batch = self._process_batch_common(
                                        task_batch,
                                        metrics,
                                        task_timing,
                                        self.local_trigger_step if self.compute_prox_log_prob else None,
                                        task_id,
                                        skip_old_log_prob=True,
                                        skip_ref_values_adv=True,
                                    )
                                task_batches[task_id] = task_batch
                                task_timings[task_id] = task_timing
                    else:
                        # ---- Non-replay path: collect from queue (original behavior) ----
                        epoch, batch, queued_task_batches = self._get_samples_from_queue()
                        if batch is None:
                            break

                        self._collect_metrics_from_samples(batch, metrics)

                        if hasattr(batch, 'meta_info') and 'reward' in batch.meta_info:
                            timing_raw['reward'] = batch.meta_info['reward']

                        for task_id in task_ids:
                            if queued_task_batches and task_id in queued_task_batches:
                                task_batch = self._clone_dataproto(queued_task_batches[task_id])
                            else:
                                task_batch = self._clone_dataproto(batch)

                            task_batch.batch["task_id"] = torch.tensor([task_id for _ in range(len(task_batch))], dtype=int)
                            if self.use_rm:
                                task_batch = self._process_batch_common(
                                    task_batch,
                                    metrics,
                                    timing_raw,
                                    self.local_trigger_step if self.compute_prox_log_prob else None,
                                    task_id,
                                    skip_old_log_prob=True,
                                    skip_ref_values_adv=True,
                                )
                            task_batches[task_id] = task_batch

                    if combined_batch is None:
                        if not task_batches:
                            break

                        task_batches = self._compute_old_log_probs_for_task_batches(
                            task_batches,
                            metrics,
                            timing_raw,
                            self.local_trigger_step if self.compute_prox_log_prob else None,
                        )
                        task_batches = self._compute_ref_log_probs_for_task_batches(task_batches, timing_raw)

                        for task_id in list(task_batches.keys()):
                            task_timing = task_timings.get(task_id, timing_raw)
                            task_batch = self._process_batch_common(
                                task_batches[task_id],
                                metrics,
                                task_timing,
                                None,
                                task_id,
                                skip_old_log_prob=True,
                                skip_reward=True,
                                skip_ref_log_prob=True,
                            )
                            self._restore_reward_extra_infos_for_logging(task_batch, task_id)
                            task_batches[task_id] = task_batch

                            if should_log_rollout:
                                task_reward_extra = {k: v for k, v in task_batch.meta_info.items()}
                                task_rollout_dir = os.path.join(rollout_data_dir, f"task{task_id}")
                                self._submit_rollout_dump(task_batch, task_reward_extra, task_timing, task_rollout_dir)

                        # Merge all task batches into one combined batch
                        combined_batch = self._merge_task_batches(task_batches)

                    # update critic (once with all tasks)
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(combined_batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # update actor (once with all tasks)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            combined_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_update_batch = self._select_actor_update_rpc_batch(combined_batch)
                            actor_output = self.actor_rollout_wg.update_actor(actor_update_batch)
                        log_prob_info = log_prob_metrics(actor_output.meta_info["metrics"], task_ids)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(log_prob_info)
                        metrics.update(actor_output_metrics)

                    # Post-training: evict low quality used groups from replay buffer
                    if self.use_replay_buffer:
                        with marked_timer("replay/evict", timing_raw):
                            evict_ids = self._active_lane_ids() if self._replay_unit() == "step" else task_ids
                            evict_infos = {}
                            for _tid in evict_ids:
                                kept, evicted, evict_info = self.replay_buffer.evict_after_use(_tid)
                                evict_infos[_tid] = evict_info
                                if self._replay_unit() == "step":
                                    _stage, _step = ReplayBuffer.decode_lane_id(_tid)
                                    pfx = f"replay/stage{_stage}_step{_step}"
                                else:
                                    pfx = f"replay/task{_tid}"
                                metrics[f"{pfx}_kept"] = kept
                                metrics[f"{pfx}_evicted"] = evicted
                                for k, v in evict_info.items():
                                    metrics[f"{pfx}_{k}"] = v
                            # Read size and entry count under a single lock
                            buf_stats = self.replay_buffer.stats_per_task()
                            for _tid in evict_ids:
                                buf_size, buf_entries = buf_stats.get(_tid, (0, 0))
                                evict_info = evict_infos.get(_tid, {})
                                if self._replay_unit() == "step":
                                    _stage, _step = ReplayBuffer.decode_lane_id(_tid)
                                    pfx = f"replay/stage{_stage}_step{_step}"
                                    label = f"stage{_stage}/step{_step}"
                                else:
                                    pfx = f"replay/task{_tid}"
                                    label = f"task{_tid}"
                                metrics[f"{pfx}_buffer_size"] = buf_size
                                metrics[f"{pfx}_buffer_entries"] = buf_entries
                                evict_info_str = ", ".join(f"{k}={v:.4f}" for k, v in evict_info.items()) if evict_info else ""
                                print(
                                    f"[ReplayBuffer] {label}: evict_after_use "
                                    f"kept={metrics[f'{pfx}_kept']}, "
                                    f"evicted={metrics[f'{pfx}_evicted']}, "
                                    f"buffer_size={buf_size}, entries={buf_entries}"
                                    + (f", {evict_info_str}" if evict_info_str else "")
                                )

                    batch = combined_batch
            # Collect step-level metrics (timing, throughput) after step timer completes
            # NOTE: _collect_task_metrics is called here (outside both step/gen timers) so that
            # timing_raw["step"] and timing_raw["gen"] are populated before compute_timing_metrics runs.
            if processed_step_batches:
                for step_id, step_batch in processed_step_batches.items():
                    task_id = self._step_id_to_task_id(step_id)
                    per_task_timing = {**timing_raw, **task_timings.get(step_id, {})} if task_timings else timing_raw
                    self._collect_task_metrics(step_batch, metrics, per_task_timing, task_ids=[task_id])
            else:
                for task_id in task_ids:
                    # Replay path: merge shared timing with per-task timing so that
                    # reward/old_log_prob/adv reflect each task's actual compute time.
                    per_task_timing = {**timing_raw, **task_timings.get(task_id, {})} if task_timings else timing_raw
                    self._collect_task_metrics(combined_batch, metrics, per_task_timing, task_ids=[task_id])
            self._collect_step_metrics(batch, 0, metrics, timing_raw)
            self.metrics_aggregator.add_step_metrics(
                metrics=metrics, sample_count=self.required_samples, timestamp=time.time()
            )
            # Trigger parameter synchronization after training step
            time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(
                f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
                f"local_trigger_step: {self.local_trigger_step} "
                f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
                f"{time_str}"
            )
            self._trigger_parameter_sync_after_step(global_steps=self.global_steps)
            self._log_validation_data()
            self._check_save_checkpoint(timing_raw)
            self.global_steps += 1

            training_end_time = time.time()
            print(f"[FullyAsyncTrainer] One Step Training Finish Time: {training_end_time - training_start_time}s")

        # final parameter sync and validate
        # 1. waiting remaining validate task
        ray.get(self.param_synchronizer.wait_last_valid.remote())
        self._log_validation_data()
        # 2. perform addtional parameter_sync and validate if trainer already updated
        if self.current_param_version % self.config.rollout.test_freq != 0 or self.local_trigger_step > 1:
            self._trigger_parameter_sync_after_step(validate=True, global_steps=self.global_steps)
            ray.get(self.param_synchronizer.wait_last_valid.remote())
            self._log_validation_data()
            
        self.progress_bar.close()

        # Stop feeder thread
        if self.use_replay_buffer and self._feeder_thread is not None:
            self._feeder_stop = True
            self._feeder_thread.join(timeout=5)
            print("[FullyAsyncTrainer] Buffer feeder thread stopped.")

        self._check_save_checkpoint(timing_raw)
        self._shutdown_rollout_dump_executor(wait=True)

    def _check_save_checkpoint(self, timing_raw):
        if self.current_param_version == self.last_ckpt_version:
            return
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. The current step number is a multiple of the save frequency.
        # 3. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (
            self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _save_checkpoint(self):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        if self.use_replay_buffer:
            replay_local_path = os.path.join(local_global_step_folder, "replay_buffer.pt")
            torch.save(self.replay_buffer.state_dict(), replay_local_path)
            print(f"[FullyAsyncTrainer] Saved replay buffer checkpoint to {replay_local_path}")
        ray.get(self.param_synchronizer.rollouter_save_checkpoint.remote(local_global_step_folder))
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[FullyAsyncTrainer] Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        if self.use_replay_buffer:
            replay_path = os.path.join(global_step_folder, "replay_buffer.pt")
            if os.path.exists(replay_path):
                replay_state = torch.load(replay_path, weights_only=False)
                self.replay_buffer.load_state_dict(replay_state)
                print(f"[FullyAsyncTrainer] Loaded replay buffer checkpoint from {replay_path}")
                self._log_replay_buffer_stats(prefix="[FullyAsyncTrainer] Loaded replay buffer stats")
            else:
                print(f"[FullyAsyncTrainer] No replay buffer checkpoint found at {replay_path}")
        return self.current_param_version

    def _log_replay_buffer_stats(self, prefix: str = "[FullyAsyncTrainer] Replay buffer stats"):
        if not self.use_replay_buffer:
            return

        stats = self.replay_buffer.stats_per_task()
        entries = self.replay_buffer.entries_per_task()
        max_entries = self.replay_buffer.max_size_per_task
        entry_cap = str(max_entries) if max_entries > 0 else "unlimited"

        parts = []
        for task_id in sorted(stats):
            rows, entry_count = stats[task_id]
            row_ratio = rows / self.required_samples if self.required_samples > 0 else 0.0
            parts.append(
                f"task{task_id}: rows={rows}/{self.required_samples} "
                f"({row_ratio:.1%}), entries={entry_count}/{entry_cap}"
            )

        total_rows = sum(rows for rows, _ in stats.values())
        total_entries = sum(entries.values())
        print(f"{prefix}: total_rows={total_rows}, total_entries={total_entries}; " + "; ".join(parts))

    def _merge_task_batches(self, task_batches: dict) -> DataProto:
        """Merge per-task DataProto objects into a single combined batch.

        For each task_id, only keys prefixed with f"task{task_id}_" are taken from
        that task's batch. This ensures that replay samples for task N use the correct
        input data (e.g. task2_input_ids from task2's batch, not task1's).
        The base batch (task_ids[0]) provides all non-task-specific keys and meta_info.
        """
        task_ids = sorted(task_batches.keys())
        combined = task_batches[task_ids[0]]

        for task_id in task_ids[1:]:
            other = task_batches[task_id]
            task_prefix = f"task{task_id}_"

            for key in list(other.batch.keys()):
                if key.startswith(task_prefix):
                    combined.batch[key] = other.batch[key]

            for key in list(other.non_tensor_batch.keys()):
                if key.startswith(task_prefix):
                    combined.non_tensor_batch[key] = other.non_tensor_batch[key]

            # Alias cross-task context keys from the task's own source batch so
            # replay samples do not accidentally use the base task's context.
            self._set_task_context_alias_from_source(combined, other, task_id)

            # Alias sample_param_version per task so _collect_task_metrics
            # uses the correct replay freshness versions for each task.
            if "sample_param_version" in other.non_tensor_batch:
                combined.non_tensor_batch[f"task{task_id}_sample_param_version"] = \
                    other.non_tensor_batch["sample_param_version"]

            # Alias uid per task so compute_group_reward_metrics uses the correct
            # uid groupings for advantage collapse metrics (replay path has different
            # uid groups per task; without this, task2/task3 collapse metrics use task1's uids).
            if "uid" in other.non_tensor_batch:
                combined.non_tensor_batch[f"task{task_id}_uid"] = other.non_tensor_batch["uid"]

            for k, v in other.meta_info.items():
                if k.startswith(task_prefix):
                    combined.meta_info[k] = v

        for task_id in task_ids:
            self._ensure_task_context_alias(combined, task_id)

        # Create per-task data_source aliases for no-replay path
        # In no-replay case all tasks share the same batch (same rows, same data_source per row).
        if "data_source" in combined.non_tensor_batch:
            for tid in task_ids:
                ds_key = f"task{tid}_data_source"
                if ds_key not in combined.non_tensor_batch:
                    combined.non_tensor_batch[ds_key] = combined.non_tensor_batch["data_source"]

        # Recompute after merging so throughput/MFU logging reflects the tasks
        # actually present in this combined training batch.
        total = None
        for tid in task_ids:
            att_key = f"task{tid}_attention_mask"
            resp_key = f"task{tid}_response_mask"
            if att_key in combined.batch.keys():
                t = combined.batch[att_key].sum(-1)
                if resp_key in combined.batch.keys():
                    t = t + combined.batch[resp_key].sum(-1)
                total = t if total is None else total + t
        if total is not None:
            combined.meta_info["global_token_num"] = total.tolist()

        return combined

    def _collect_task_metrics(self, batch, metrics, timing_raw, task_ids=None):
        """Collect per-task data metrics (scores, advantages, sample sources)."""
        from recipe.image_rl.custom_metric_utils import (
            compute_data_metrics, compute_timing_metrics, compute_group_reward_metrics,
        )
        if task_ids is None:
            task_ids = list(self.config.actor_rollout_ref.actor.multi_task.get("task_ids", [1]))
        available_keys = set(batch.batch.keys())
        for task_id in task_ids:
            if f"task{task_id}_advantages" not in available_keys:
                continue
            batch.batch["task_id"] = torch.tensor([task_id for _ in range(len(batch))], dtype=int)
            # Temporarily set task-specific data_source so per-source metrics are correct per task
            ds_key = f"task{task_id}_data_source"
            orig_ds = batch.non_tensor_batch.get("data_source")
            if ds_key in batch.non_tensor_batch:
                batch.non_tensor_batch["data_source"] = batch.non_tensor_batch[ds_key]
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_group_reward_metrics(batch=batch))
            if orig_ds is not None:
                batch.non_tensor_batch["data_source"] = orig_ds
            elif "data_source" in batch.non_tensor_batch:
                del batch.non_tensor_batch["data_source"]
            # Log sample_param_version stats and fresh sample count
            # Use per-task aliased key if available (replay case with different samples per task),
            # fall back to common key (no-replay case where all tasks share the same batch).
            spv_key = f"task{task_id}_sample_param_version"
            if spv_key not in batch.non_tensor_batch:
                spv_key = "sample_param_version"
            if spv_key in batch.non_tensor_batch:
                versions = batch.non_tensor_batch[spv_key].astype(float)
                gaps = self.current_param_version - versions

                metrics[f"replay/task{task_id}_version_min"] = float(np.min(versions))
                metrics[f"replay/task{task_id}_version_mean"] = float(np.mean(versions))
                metrics[f"replay/task{task_id}_version_max"] = float(np.max(versions))

                metrics[f"replay/task{task_id}_gap_min"] = float(np.min(gaps))
                metrics[f"replay/task{task_id}_gap_mean"] = float(np.mean(gaps))
                metrics[f"replay/task{task_id}_gap_max"] = float(np.max(gaps))
                metrics[f"replay/task{task_id}_stale_ratio"] = float(np.mean(gaps >= 1))
                metrics[f"replay/task{task_id}_fresh_ratio"] = float(np.mean(gaps == 0))

                fresh_count = int(np.sum(versions == self.current_param_version))
                metrics[f"replay/task{task_id}_fresh_count"] = fresh_count

            batch.pop(batch_keys=["task_id"])

    def _collect_step_metrics(self, batch, epoch, metrics, timing_raw):
        """Collect step-level metrics (timing, throughput). Call after step timer completes."""
        from recipe.image_rl.custom_metric_utils import compute_throughout_metrics
        steps_duration = timing_raw["step"]
        self.max_steps_duration = max(self.max_steps_duration, steps_duration)
        metrics.update({
            "training/global_step": self.global_steps,
            "training/epoch": epoch,
        })
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            samples_param_versions = batch.meta_info["rollout_param_versions"]
            stale_count = sum(1 for v in samples_param_versions if self.current_param_version - v >= 1)
            self.stale_samples_processed += stale_count
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_samples_processed": self.stale_samples_processed,
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async"):
                    metrics[key] = value

    def _trigger_parameter_sync_after_step(self, validate: bool = False, global_steps: int = None):
        """
        Trigger parameter synchronization after training step
        This ensures rollouter always uses the latest trained parameters
        """
        if self.local_trigger_step < self.trigger_parameter_sync_step and not validate:
            self.local_trigger_step += 1
            return

        print(f"[FullyAsyncTrainer] Hard syncing workers before sync v{self.current_param_version + 1}...")

        self.current_param_version += 1
        self.local_trigger_step = 1
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.progress_bar.update(1)
        self.metrics_aggregator.reset()
        timing_param_sync = {}
        with marked_timer("timing_s/wait_last_valid", timing_param_sync):
            ray.get(self.param_synchronizer.wait_last_valid.remote())
        with marked_timer("timing_s/param_sync", timing_param_sync):
            t0 = time.time()
            exported_weights = ray.get(
                self.param_synchronizer.export_weights_only.remote(self.current_param_version)
            )
            self.param_synchronizer.distribute_weights.remote(
                self.current_param_version,
                exported_weights,
                validate=validate,
                global_steps=global_steps
            )
            # self.param_synchronizer.sync_weights.remote(
            # self.current_param_version, validate=validate, global_steps=global_steps
            # )
        self.logger.log(data=timing_param_sync, step=self.current_param_version)

    def _log_validation_data(self):
        """
        Log validation data
        """
        val_data = self.message_queue_client.get_validate_sync()
        if not val_data:
            return

        val_metrics: ValidateMetrics = ray.cloudpickle.loads(val_data)

        # Use max of val_metrics.param_version and current_param_version to avoid wandb step ordering warnings
        # This ensures monotonically increasing steps while still logging all validation results
        log_step = max(val_metrics.param_version, self.current_param_version)

        if val_metrics.param_version < self.current_param_version:
            print(
                f"[FullyAsyncTrainer] Logging stale validation result from param_version {val_metrics.param_version} "
                f"at step {log_step} (current: {self.current_param_version})"
            )

        if val_metrics.metrics:
            # Extract validation generation samples if present (image RL specific)
            validation_samples = val_metrics.metrics.pop('validation_samples', None)

            self.logger.log(data=val_metrics.metrics, step=log_step)
            pprint(
                f"[FullyAsyncTrainer] parameter version: {val_metrics.param_version} "
                f"Validation metrics: {val_metrics.metrics}"
            )

            # Log validation generation samples to wandb (image RL specific)
            if validation_samples and "wandb" in self.logger.logger:
                from recipe.image_rl.tracking import ValidationGenerationsLogger
                import wandb

                val_gen_logger = ValidationGenerationsLogger(
                    project_name=self.config.trainer.project_name,
                    experiment_name=self.config.trainer.experiment_name,
                )
                val_gen_logger._log_generations_to_wandb(validation_samples, log_step, wandb)

        self.logger.log(data=val_metrics.timing_raw, step=log_step)
