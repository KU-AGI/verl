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
        self._rollout_dump_workers = int(self.config.trainer.get("rollout_dump_workers", 1) or 0)
        self._rollout_dump_max_pending = int(
            self.config.trainer.get("rollout_dump_max_pending", max(2, 2 * max(self._rollout_dump_workers, 1))) or 0
        )

        # ==================== fully async config ====================

        self.message_queue_client = None
        self.param_synchronizer = None

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
            raw_thresholds = replay_cfg.get("score_thresholds", {})
            score_thresholds = {int(k): float(v) for k, v in raw_thresholds.items()}
            raw_std_thresholds = replay_cfg.get("score_std_thresholds", {})
            score_std_thresholds = {int(k): float(v) for k, v in raw_std_thresholds.items()}
            self.replay_buffer = ReplayBuffer(
                task_ids=task_ids,
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
            self._feeder_task_ids = task_ids
            print(
                f"[FullyAsyncTrainer] ReplayBuffer enabled: "
                f"max_version_gap={self.replay_buffer.max_version_gap}, "
                f"max_size_per_task={self.replay_buffer.max_size_per_task}, "
                f"filter_mode={self.replay_buffer.filter_mode}, "
                f"score_thresholds={score_thresholds}, "
                f"reward_history_size={self.replay_buffer.reward_history_size}, "
                f"max_quantile={self.replay_buffer.max_quantile}, "
                f"std_quantile={self.replay_buffer.std_quantile}"
            )

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

                    if self.use_replay_buffer:
                        # ---- Replay path: sample from buffer (feeder pushes in background) ----
                        with marked_timer("replay/wait", timing_raw):
                            buffer_ready = self._wait_for_buffer()
                        if not buffer_ready:
                            print("[FullyAsyncTrainer] Buffer feeder terminated and buffer insufficient, stopping.")
                            break

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
                            task_batch = self._process_batch_common(
                                task_batch, metrics, task_timing, self.local_trigger_step if self.compute_prox_log_prob else None, task_id
                            )
                            self._restore_reward_extra_infos_for_logging(task_batch, task_id)
                            task_batches[task_id] = task_batch
                            task_timings[task_id] = task_timing

                            if should_log_rollout:
                                task_reward_extra = {k: v for k, v in task_batch.meta_info.items()}
                                task_rollout_dir = os.path.join(rollout_data_dir, f"task{task_id}")
                                self._submit_rollout_dump(task_batch, task_reward_extra, task_timing, task_rollout_dir)
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
                            task_batch = self._process_batch_common(
                                task_batch, metrics, timing_raw, self.local_trigger_step if self.compute_prox_log_prob else None, task_id
                            )
                            self._restore_reward_extra_infos_for_logging(task_batch, task_id)
                            task_batches[task_id] = task_batch

                            if should_log_rollout:
                                task_reward_extra = {k: v for k, v in task_batch.meta_info.items()}
                                task_rollout_dir = os.path.join(rollout_data_dir, f"task{task_id}")
                                self._submit_rollout_dump(task_batch, task_reward_extra, timing_raw, task_rollout_dir)

                    if not task_batches:
                        break

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
                            actor_output = self.actor_rollout_wg.update_actor(combined_batch)
                        log_prob_info = log_prob_metrics(actor_output.meta_info["metrics"], task_ids)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(log_prob_info)
                        metrics.update(actor_output_metrics)

                    # Post-training: evict low quality used groups from replay buffer
                    if self.use_replay_buffer:
                        with marked_timer("replay/evict", timing_raw):
                            for _tid in task_ids:
                                kept, evicted, evict_info = self.replay_buffer.evict_after_use(_tid)
                                metrics[f"replay/task{_tid}_kept"] = kept
                                metrics[f"replay/task{_tid}_evicted"] = evicted
                                for k, v in evict_info.items():
                                    metrics[f"replay/task{_tid}_{k}"] = v
                            # Read size and entry count under a single lock
                            buf_stats = self.replay_buffer.stats_per_task()
                            for _tid in task_ids:
                                buf_size, buf_entries = buf_stats.get(_tid, (0, 0))
                                metrics[f"replay/task{_tid}_buffer_size"] = buf_size
                                metrics[f"replay/task{_tid}_buffer_entries"] = buf_entries
                                evict_info_str = ", ".join(f"{k}={v:.4f}" for k, v in evict_info.items()) if evict_info else ""
                                print(
                                    f"[ReplayBuffer] task{_tid}: evict_after_use "
                                    f"kept={metrics[f'replay/task{_tid}_kept']}, "
                                    f"evicted={metrics[f'replay/task{_tid}_evicted']}, "
                                    f"buffer_size={buf_size}, entries={buf_entries}"
                                    + (f", {evict_info_str}" if evict_info_str else "")
                                )

                    batch = combined_batch
            # Collect step-level metrics (timing, throughput) after step timer completes
            # NOTE: _collect_task_metrics is called here (outside both step/gen timers) so that
            # timing_raw["step"] and timing_raw["gen"] are populated before compute_timing_metrics runs.
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
            else:
                print(f"[FullyAsyncTrainer] No replay buffer checkpoint found at {replay_path}")
        return self.current_param_version

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

            # Alias cross-task context keys so dp_actor can use the correct
            # INPUT context (= the image actually shown to the policy at
            # rollout time) for each task's replay samples. Under multi-turn,
            # task2/task3 consume `current_*` (rolling image stream), not
            # task1's original output. Fall back to `task1_*` for single-turn
            # batches that have no `current_*` stamped.
            if task_id == 2:
                src_key = (
                    "current_imgs_pixel_values"
                    if "current_imgs_pixel_values" in other.batch.keys()
                    else "task1_gen_imgs_pixel_values"
                )
                if src_key in other.batch.keys():
                    combined.batch["task2_task1_gen_imgs_pixel_values"] = other.batch[src_key]
            elif task_id == 3:
                src_key = (
                    "current_img_tokens"
                    if "current_img_tokens" in other.batch.keys()
                    else "task1_gen_img_tokens"
                )
                if src_key in other.batch.keys():
                    combined.batch["task3_task1_gen_img_tokens"] = other.batch[src_key]

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

        # Create per-task data_source aliases for no-replay path
        # In no-replay case all tasks share the same batch (same rows, same data_source per row).
        if "data_source" in combined.non_tensor_batch:
            for tid in task_ids:
                ds_key = f"task{tid}_data_source"
                if ds_key not in combined.non_tensor_batch:
                    combined.non_tensor_batch[ds_key] = combined.non_tensor_batch["data_source"]

        # Recompute global_token_num if missing (e.g. after replay buffer meta_info filtering)
        if "global_token_num" not in combined.meta_info:
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
            weights_ref = ray.get(
            self.param_synchronizer.export_weights_only.remote(self.current_param_version)
            )
            self.param_synchronizer.distribute_weights.remote(
                self.current_param_version,
                weights_ref,
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
