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
import asyncio
import os
import time
from pprint import pformat

import numpy as np
import ray
import torch
from ray import ObjectRef

from recipe.fully_async_policy_image_rl.detach_utils import (
    RolloutSample,
    ValidateMetrics,
    prepare_single_generation_data,
    merge_rollout_sample,
    expand_rollout_sample,
)
from recipe.fully_async_policy_image_rl.message_queue import MessageQueueClient
from recipe.fully_async_policy_image_rl.ray_trainer import FullyAsyncRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from recipe.image_rl.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType
from verl import DataProto
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger


@ray.remote(num_cpus=10, max_concurrency=100)
class FullyAsyncRollouter(FullyAsyncRayPPOTrainer):
    """
    Asynchronous sample generator, responsible for continuously generating training samples
    and putting them into MessageQueue
    Based on the mature implementation improvements of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        print("[FullyAsyncRollouter] Loading reward functions...")
        self.reward_fn = load_reward_manager(
            config, tokenizer, processor, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.val_reward_fn = load_reward_manager(
            config, tokenizer, processor, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger than 1"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False
        self.use_critic = False
        self.use_reference_policy = False
        self.use_rm = False

        print("[FullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        print(f"[FullyAsyncRollouter] Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== fully async config ====================

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.rollout_prompt_size = config.async_training.get("rollout_prompt_size", 1)
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        # queue size
        self.max_queue_size = None

        # Statistics
        self.current_param_version = 0
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        # we start from step 1
        self.global_steps = 1
        self.idle_start_time = None
        self.version_start_time = None

        self.active_sample_count = 0
        self.cancel_sample_count = 0

        # Concurrency control
        # Modified by self.pause() or self._should_pause_generation()
        self.paused = False
        self.running = True
        self.monitor_loop_trigger = True

        # Add dataloader lock
        self.dataloader_lock = asyncio.Lock()

        # Initialize async queues
        self.pending_queue = asyncio.Queue(maxsize=2048)
        self.active_tasks = set()
        self.cancel_queue = asyncio.Queue()
        self.reward_tasks = set()

    def _init_async_objects(self):
        # Initialize asyncio synchronization primitives.
        # We let asyncio.Condition create the Lock internally to ensure they share the same Event Loop.
        # This avoids 'ValueError: loop argument must agree with lock' which can occur in Ray environments
        # where the lock's captured loop (get_running_loop) differs from Condition's default loop check.
        # Explicitly passing the loop is deprecated/removed in Python 3.10+, so this reverse-initialization
        # is the most robust workaround.
        self.condition = asyncio.Condition()
        self.lock = self.condition._lock

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_max_required_samples(self):
        async with self.lock:
            self.max_required_samples = int(
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )

            self.max_concurrent_samples = len(self.async_rollout_manager.server_handles) * 16
            self.max_concurrent_samples = min(self.max_concurrent_samples, self.max_required_samples)
            self.max_queue_size = self.max_required_samples

            print(
                f"[FullyAsyncRollouter] required_samples : {self.required_samples} "
                f"max_required_samples: {self.max_required_samples} "
                f"max_queue_size: {self.max_queue_size} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps: {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
            )

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        return self.total_train_steps

    async def update_param_version(self, version: int, validate: bool = False, global_steps: int = 0):
        """Update current parameter version"""
        async with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            # every time param change, reset staleness_samples (sample-level)
            mq_size = await self.message_queue_client.get_queue_size()
            # active_sample_count: in-flight rollout sample 수
            # cancel_sample_count: cancel_queue 안의 sample 수
            self.staleness_samples = self.active_sample_count + self.cancel_sample_count + mq_size
            timing_raw = {}
            idle_ratio = None
            if self.idle_start_time is not None and self.version_start_time is not None:
                rollout_active_time = self.idle_start_time - self.version_start_time
                rollout_version_time = time.time() - self.version_start_time
                idle_ratio = 1 - rollout_active_time / rollout_version_time
                timing_raw["rollouter/active_time"] = rollout_active_time
                timing_raw["rollouter/version_time"] = rollout_version_time
                timing_raw["rollouter/idle_ratio"] = idle_ratio
                self.idle_start_time = None
            print(
                f"[FullyAsyncRollouter][Public][update_param_version] "
                f"Parameter version updated from {old_version} to {version} "
                f",reset staleness_samples to: {self.staleness_samples}"
                f",idle_ratio: {idle_ratio}"
            )
            # val_metrics = None
            # if (
            #     self.val_reward_fn is not None
            #     and self.config.rollout.test_freq > 0
            #     and self.current_param_version % self.config.rollout.test_freq == 0
            #     and self.current_param_version > 0  # don't test here in the initial parameter sync
            # ) or (validate and self.val_reward_fn is not None):
            #     with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            #         val_metrics: dict = self._validate()
            # data = ValidateMetrics(
            #     timing_raw=timing_raw, metrics=val_metrics, global_steps=global_steps, param_version=version
            # )
            # await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))

            self.version_start_time = time.time()

    async def save_checkpoint(self, local_global_step_folder: str):
        # WARNING!: Due to the asynchronous nature, there are some in-flight samples
        # (pending/cancel/result queue and message queue).
        # Therefore, directly saving the state of the dataloader will result in losing these
        # samples when resuming training.
        # TODO: Implement dataloader recovery without losing in-flight samples.
        from verl.utils.fs import local_mkdir_safe

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        async with self.dataloader_lock:
            dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        print(f"[FullyAsyncRollouter] Saved dataloader checkpoint to {dataloader_local_path}")

    def load_checkpoint(self):
        """Load checkpoint including dataloader state based on resume mode"""

        if self.config.trainer.resume_mode == "disable":
            print("[FullyAsyncRollouter] Resume mode is disabled, starting from scratch")
            return 0

        # Determine checkpoint folder path
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("[FullyAsyncRollouter] Load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        # Find and validate global_step_folder based on resume mode
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[FullyAsyncRollouter] Training from scratch (no checkpoint found)")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), (
                "[FullyAsyncRollouter] resume_from_path must be str type"
            )
            assert "global_step_" in self.config.trainer.resume_from_path, (
                "[FullyAsyncRollouter] resume_from_path must specify the global_steps"
            )
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            raise ValueError(f"[FullyAsyncRollouter] Unknown resume_mode: {self.config.trainer.resume_mode}")

        print(f"[FullyAsyncRollouter] Loading checkpoint from: {global_step_folder}")

        # Extract and set global step
        trainer_global_steps = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = (
            trainer_global_steps * self.required_samples * self.config.async_training.trigger_parameter_sync_step + 1
        )
        print(f"[FullyAsyncRollouter] Setting global_steps to {self.global_steps}")

        # Load dataloader state
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            print(f"[FullyAsyncRollouter] Loaded dataloader state from {dataloader_local_path}")
        else:
            print(
                f"[FullyAsyncRollouter] Warning: No dataloader state found at {dataloader_local_path}, "
                f"will start from scratch"
            )

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_async_objects()
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        await self._init_async_rollout_manager()

    def _create_actor_rollout_classes(self):
        # only create rollout
        for role in [Role.Rollout]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.rollout.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _init_async_rollout_manager(self):
        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from recipe.fully_async_policy_image_rl.agent_loop import FullyAsyncAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
        )

    async def queue_async_weight_update(self, version: int, param_synchronizer=None):
        """
        Queue async weight update on all server replicas and apply when idle.
        Called from ParameterSynchronizer.
        """
        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Queue weight update on all server replicas
        server_handles = self.async_rollout_manager.server_handles
        print(f"[FullyAsyncRollouter] Queueing weight update version {version} to {len(server_handles)} servers")

        # Queue on all servers in parallel (non-blocking)
        await asyncio.gather(
            *[server.queue_weight_update.remote(version) for server in server_handles]
        )

        # Start background task to apply weights when servers are idle
        asyncio.create_task(self._apply_weights_when_idle(version, param_synchronizer))

    async def _apply_weights_when_idle(self, version: int, param_synchronizer):
        """
        Wait for all servers to become idle, then trigger NCCL sync via ParameterSynchronizer.

        Optimization: Uses adaptive polling - checks more frequently when servers are becoming idle.

        This monitors generation servers and when all are idle, calls the ParameterSynchronizer
        to perform the actual NCCL broadcast with both actor_wg and rollout_wg.
        """
        print(f"[FullyAsyncRollouter] Starting background task to monitor idle state for version {version}")

        check_interval = 0.01  # Start with 10ms for fast detection
        idle_count = 0

        while True:
            await asyncio.sleep(check_interval)

            # Check if all servers are idle (no ongoing generations)
            server_handles = self.async_rollout_manager.server_handles

            try:
                statuses = await asyncio.gather(
                    *[server.get_weight_update_status.remote() for server in server_handles]
                )
            except Exception as e:
                print(f"[FullyAsyncRollouter] Error getting server status: {e}")
                break

            all_idle = all(status['ongoing_generations'] == 0 for status in statuses)
            any_pending = any(status['pending_weight_version'] == version for status in statuses)

            # Count how many servers are idle for logging
            num_idle = sum(1 for s in statuses if s['ongoing_generations'] == 0)

            if all_idle and any_pending:
                print(f"[FullyAsyncRollouter] All {len(server_handles)} servers idle, triggering NCCL sync for version {version}")

                # Trigger NCCL weight sync via ParameterSynchronizer
                # The synchronizer has access to both actor_wg and rollout_wg
                if param_synchronizer is not None:
                    try:
                        print(f"[FullyAsyncRollouter] Calling apply_nccl_weight_sync for version {version}")
                        # Fire the remote call but don't wait synchronously - use ray.get in thread pool
                        apply_ref = param_synchronizer.apply_nccl_weight_sync.remote(version)

                        # Run ray.get in thread pool to avoid blocking the event loop
                        await asyncio.to_thread(ray.get, apply_ref)
                        print(f"[FullyAsyncRollouter] Successfully completed NCCL weight sync for version {version}")
                    except Exception as e:
                        print(f"[FullyAsyncRollouter] Failed to apply weight sync: {e}")
                        import traceback
                        traceback.print_exc()

                # Clear pending versions on all servers (fire-and-forget to avoid deadlock)
                print(f"[FullyAsyncRollouter] Clearing pending weight updates for version {version}")
                # Don't await - just fire the remote calls
                for server in server_handles:
                    server.queue_weight_update.remote(-1)  # Fire-and-forget
                print(f"[FullyAsyncRollouter] Background task complete for version {version}")
                break
            else:
                # Adaptive polling: if getting close to all idle, check more frequently
                if num_idle >= len(server_handles) - 1:
                    check_interval = 0.01  # 10ms - very frequent
                    idle_count += 1
                    if idle_count % 100 == 0:  # Log every 100ms when almost ready
                        print(f"[FullyAsyncRollouter] Waiting for weight sync v{version}: {num_idle}/{len(server_handles)} servers idle")
                elif num_idle >= len(server_handles) // 2:
                    check_interval = 0.05  # 50ms - moderate
                else:
                    check_interval = 0.1  # 100ms - normal

            # If no server has pending update for this version anymore, stop waiting
            if not any_pending:
                print(f"[FullyAsyncRollouter] No pending updates for version {version}, stopping background task")
                break

    async def check_all_servers_idle(self) -> bool:
        """Check if all generation servers are currently idle."""
        if self.async_rollout_manager is None:
            return False

        server_handles = self.async_rollout_manager.server_handles
        statuses = await asyncio.gather(
            *[server.get_weight_update_status.remote() for server in server_handles]
        )
        return all(status['ongoing_generations'] == 0 for status in statuses)

    async def clear_pending_weight_updates(self):
        """Clear all pending weight update versions on generation servers."""
        if self.async_rollout_manager is None:
            return

        server_handles = self.async_rollout_manager.server_handles
        await asyncio.gather(
            *[server.queue_weight_update.remote(-1) for server in server_handles]  # -1 means clear
        )

    # Add samples to the pending_queue
    async def _feed_samples(self):
        continuous_iterator = self._create_continuous_iterator()

        full_batch_list = []
        last_epoch = 0

        print(f"[FullyAsyncRollouter][Feed] Start feeding. Chunk size: require_batches {self.require_batches} * rollout.n {self.config.actor_rollout_ref.rollout.n}")

        for epoch, batch_dict in continuous_iterator:
            # Prepare generation data
            full_batch = prepare_single_generation_data(
                batch_dict, self.global_steps, self.config.actor_rollout_ref.rollout.n
            )

            # Accumulate batches
            full_batch_list.append(full_batch)

            self.global_steps += 1
            last_epoch = epoch

            # Send accumulated batches when reaching target size
            if len(full_batch_list) >= self.rollout_prompt_size:
                await self._merge_and_send(full_batch_list, epoch, self.global_steps)
                full_batch_list = []

            # Check if maximum steps reached
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[FullyAsyncRollouter][Feed] "
                    f"Maximum count reached: {self.global_steps} >= {self.total_rollout_steps}"
                )
                break

        # Send remaining batches
        if full_batch_list:
            await self._merge_and_send(full_batch_list, last_epoch, self.global_steps)

        # End signal
        await self.pending_queue.put("DONE")
        print(f"[FullyAsyncRollouter][Feed] Completed. Total steps: {self.global_steps}")

    async def _merge_and_send(self, batch_list, epoch, current_steps):
        """Helper function to merge batch list and send to pending queue"""
        # Merge batches using DataProto.concat
        merged_batch = DataProto.concat(batch_list)
        batch_size = len(merged_batch)

        # Ensure unique sample_id to avoid conflicts
        sample_id = f"chunk_sample_{epoch}_{current_steps}"

        rollout_sample = RolloutSample(
            full_batch=merged_batch,
            agent_loop_output_list=[None] * batch_size,
            sample_id=sample_id,
            epoch=epoch,
            tool_calls=[],
            param_version=0,
            param_version_start=[],
            param_version_end=[],
            processing_times=[],
            rollout_status={},
        )

        await self.pending_queue.put(rollout_sample)
        # print(f"[FullyAsyncRollouter][Feed] Sent chunk batch size: {batch_size}")

    async def _processor_worker(self):
        """
        Streaming worker coroutines, a sample is submitted for processing without waiting for batches
        """
        print("[FullyAsyncRollouter][Processor] processor worker started")

        while True:
            # Check for pause condition
            if self.paused or await self._should_pause_generation():
                async with self.lock:
                    self.paused = True
                    self.idle_start_time = time.time()
                    while self.paused:
                        await self.condition.wait()
                # After resume, continue to next iteration
                continue

            sample_from_cancel_queue = False
            rollout_sample = None

            if not self.cancel_queue.empty():
                rollout_sample = await self.cancel_queue.get()
                sample_from_cancel_queue = True
                if rollout_sample != "DONE":
                    batch_size = len(rollout_sample.full_batch)
                    async with self.lock:
                        self.cancel_sample_count -= batch_size
            else:
                rollout_sample = await self.pending_queue.get()

            # Handle DONE signal
            if rollout_sample == "DONE":
                print(
                    "[FullyAsyncRollouter][Processor] Received end signal, "
                    "waiting for remaining tasks to complete..."
                )
                # Wait for rollout tasks
                while self.active_tasks:
                    async with self.lock:
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                        for task in done_tasks:
                            await task
                
                # Wait for reward tasks
                while self.reward_tasks:
                    async with self.lock:
                        if self.reward_tasks:
                            done_tasks, self.reward_tasks = await asyncio.wait(
                                self.reward_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                        for task in done_tasks:
                            await task
                break

            # Process actual rollout sample
            batch_size = len(rollout_sample.full_batch)

            # Update in-flight sample count
            async with self.lock:
                if not sample_from_cancel_queue:
                    # Add to staleness only when first dequeued from pending_queue
                    self.staleness_samples += batch_size
                # Increment active sample count for both queues
                self.active_sample_count += batch_size

            # Check concurrency limit
            while self.active_sample_count >= self.max_concurrent_samples:
                async with self.lock:
                    if self.active_tasks:
                        done_tasks, self.active_tasks = await asyncio.wait(
                            self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                    for task in done_tasks:
                        await task

            # Create and submit task
            async with self.lock:
                while self.paused:
                    await self.condition.wait()
                task = asyncio.create_task(
                    self._process_single_sample_streaming(rollout_sample),
                    name=rollout_sample.sample_id,
                )
                self.active_tasks.add(task)

            # Mark queue task as done
            if sample_from_cancel_queue:
                self.cancel_queue.task_done()
            else:
                self.pending_queue.task_done()

    async def _compute_single_task_reward(
        self,
        task_id: int,
        batch_result: DataProto,
        reward_results: dict,
    ):
        """Compute reward for a single task asynchronously"""
        try:
            from recipe.image_rl.reward import compute_reward_async

            # Launch async reward computation
            future = compute_reward_async.remote(
                data=batch_result,
                config=self.config,
                tokenizer=self.tokenizer,
                processor=self.processor,
                reward_fn=None,
                eval=False,
                task_id=task_id
            )

            # Wait for reward computation to complete
            reward_result = await asyncio.to_thread(ray.get, future)
            reward_tensor, reward_extra_infos_dict = reward_result

            # Store reward in reward_results
            reward_results[f"task{task_id}_token_level_scores"] = reward_tensor

            if "meta_info" not in reward_results:
                reward_results["meta_info"] = {}
            reward_results["meta_info"][f"task{task_id}_reward_extra_info"] = reward_extra_infos_dict

            print(f"[FullyAsyncRollout] reward_result: {reward_results.keys()}")

        except Exception as e:
            print(f"[Rollouter][RewardTask{task_id}] ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            current_task = asyncio.current_task()
            self.reward_tasks.discard(current_task)


    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample with task-level streaming reward computation"""
        try:
            batch_size = len(rollout_sample.full_batch)
            current_version = self.current_param_version

            # Set param_version in non_tensor_batch
            rollout_sample.full_batch.non_tensor_batch["param_version"] = np.full(
                (batch_size,), current_version, dtype=np.int64
            )

            # Track reward tasks for this sample
            sample_reward_tasks = []

            # Store reward results
            reward_results = {}

            # Define callback for task completion
            def on_task_complete(task_id: int, batch_result: DataProto):
                """Called when each task completes - launch reward computation"""
                # Launch async reward computation for this task
                reward_task = asyncio.create_task(
                    self._compute_single_task_reward(
                        task_id=task_id,
                        batch_result=batch_result,
                        reward_results=reward_results,
                    ),
                    name=f"reward_task{task_id}_{rollout_sample.sample_id}",
                )
                sample_reward_tasks.append(reward_task)

                # Add to global reward_tasks tracking without lock to avoid deadlock
                # The lock here was causing callbacks to wait for each other
                self.reward_tasks.add(reward_task)

            # Execute rollout with task-level callbacks
            # generate_sequences_with_callback will sequentially execute Task 1, 2, 3
            # and call on_task_complete after each task finishes
            result_batch = await self.async_rollout_manager.generate_sequences_with_callback(
                rollout_sample.full_batch,
                on_task_complete=on_task_complete,
            )

            await asyncio.sleep(0)
            print(f"[Rollouter][ProcessSample] generate_sequences_with_callback completed...")

            # IMPORTANT: Decrement active_sample_count immediately after generation completes
            # This allows next generation to start while rewards are being computed
            # async with self.lock:
            self.active_sample_count -= batch_size

            # Update the rollout_sample with the result batch
            rollout_sample.full_batch = result_batch
            rollout_sample.agent_loop_output_list = []

            # Check if cancelled
            is_cancel = False

            if is_cancel:
                # Cancel all pending reward tasks
                for task in sample_reward_tasks:
                    task.cancel()

                # async with self.lock:
                self.cancel_sample_count += batch_size
                await self.cancel_queue.put(rollout_sample)
            else:
                # Wait for all task rewards to complete
                results = await asyncio.gather(*sample_reward_tasks, return_exceptions=True)
                print(f"[Rollouter][ProcessSample] All {len(results)} reward tasks completed")

                # Check for exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"[Rollouter][ProcessSample] ERROR in reward task {i}: {result}")
                        import traceback
                        traceback.print_exception(type(result), result, result.__traceback__)

                # Integrate reward results into rollout_sample
                for key, value in reward_results.items():
                    if key != "meta_info":
                        rollout_sample.full_batch.batch[key] = value

                # Merge meta_info
                if "meta_info" in reward_results:
                    if not hasattr(rollout_sample.full_batch, 'meta_info'):
                        rollout_sample.full_batch.meta_info = {}
                    rollout_sample.full_batch.meta_info.update(reward_results["meta_info"])

                # Merge rollout sample
                rollout_sample.param_version = current_version
                rollout_sample.rollout_status = {
                    "param_version": current_version,
                    "staleness_samples": self.staleness_samples,
                    "total_generated_samples": self.total_generated_samples,
                }

                rollout_sample = merge_rollout_sample(
                    self.config, self.tokenizer, rollout_sample, self.processor
                )

                # Expand the merged rollout sample back to individual samples
                individual_samples = expand_rollout_sample(rollout_sample)

                # Send individual samples to message queue
                success_count = 0
                for individual_sample in individual_samples:
                    success = await self.message_queue_client.put_sample(
                        sample=ray.cloudpickle.dumps(individual_sample),
                        param_version=individual_sample.param_version,
                    )
                    if success:
                        success_count += 1
                        print(f"[FullyAsyncRollouter] Message queue statistic: {self.message_queue_client.get_statistics_sync()}")

                # async with self.lock:
                self.total_generated_samples += success_count
                self.staleness_samples -= batch_size
                self.dropped_stale_samples += (batch_size - success_count)

            self.processed_sample_count += 1
        except Exception as e:
            print(f"[Rollouter][ProcessSample] EXCEPTION during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            current_task = asyncio.current_task()
            # async with self.lock:
            self.active_tasks.discard(current_task)

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        # we start from step 1
        self.global_steps += 1

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        print(f"[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: {self.max_concurrent_samples}")

        # Start sample feed and processor coroutines (no consumer worker needed)
        self.feed_task = asyncio.create_task(self._feed_samples())
        self.processor_task = asyncio.create_task(self._processor_worker())

        try:
            # Wait for sample feed to complete
            await self.feed_task
            print("[FullyAsyncRollouter] Sample feed completed")

            # Wait for streaming to complete
            await self.processor_task
            print("[FullyAsyncRollouter] Streaming process completed")

        except Exception as e:
            print(f"[FullyAsyncRollouter] Streaming process exception:{e}")

        finally:
            if self.processor_task:
                self.processor_task.cancel()

            await asyncio.gather(self.processor_task, return_exceptions=True)

        # Send a finish signal
        await self.message_queue_client.put_sample(
            sample=None,
            param_version=self.current_param_version,
        )

        async with self.lock:
            self.running = False

    async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines
        """

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.running = True

        # Create the main asynchronous task
        generation_task = asyncio.create_task(self._streaming_generation_main())
        monitor_task = asyncio.create_task(self._async_monitor_loop())

        try:
            # Run build and monitoring tasks concurrently
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[FullyAsyncRollouter] Rollouter fit completed")

    async def _async_monitor_loop(self):
        """
        Async coroutine for monitoring:
        Function 1: Log information output
        Function 2: Trigger rollout recovery
        """
        last_stats_time = time.time()
        stats_interval = 60.0
        check_interval = 10.0

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)
            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[FullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            # Trigger rollout recovery
            # if self.monitor_loop_trigger and not self.paused:
            #     if not await self._should_pause_generation():
            #         async with self.lock:
            #             self.paused = False
            #             self.condition.notify_all()

            if self.paused:
                should_pause = await self._should_pause_generation()
                print(f"[MonitorLoop] paused=True, should_pause={should_pause}", flush=True)
                if not should_pause:
                    print(f"[MonitorLoop] Triggering resume!", flush=True)
                    self.paused = False
                    async with self.condition:
                        self.condition.notify_all()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        queue_stats = self.message_queue_client.get_statistics_sync()
        queue_size = queue_stats["queue_size"]

        if queue_size >= self.max_queue_size:
            if not self.paused:
                print(
                    f"[FullyAsyncRollouter][ShouldPause]  "
                    f"due to full queue: size={queue_size}, max={self.max_queue_size}"
                )
            return True

        if self.staleness_samples >= self.max_required_samples:
            if not self.paused:
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to "
                    f"staleness_samples {self.staleness_samples} >= max_required_samples {self.max_required_samples} "
                )
            return True

        return False

    async def pause(self):
        """pause rollout"""
        print("[FullyAsyncRollouter][Public][Pause]")

        # Set paused flag
        async with self.lock:
            self.paused = True
            self.monitor_loop_trigger = False

        # Cancel all rollout tasks if partial_rollout is enabled
        if self.config.async_training.partial_rollout:
            await self.async_rollout_manager.cancel()

        # Wait for active tasks to complete with throttled logging
        last_logged_count = None
        last_logged_time = 0.0
        log_interval = 10.0
        sleep_interval = 0.5

        while True:
            async with self.lock:
                remaining_samples = self.active_sample_count
                remaining_rewards = len(self.reward_tasks)
                remaining_tasks = len(self.active_tasks)

            if remaining_samples == 0 and remaining_rewards == 0:
                break

            now = time.time()
            if (
                remaining_samples != last_logged_count
                or (now - last_logged_time) >= log_interval
            ):
                print(
                    "[FullyAsyncRollouter][Pause] Waiting for "
                    f"{remaining_samples} active samples "
                    f"(task set size={remaining_tasks})..."
                )
                last_logged_count = remaining_samples
                last_logged_time = now

            await asyncio.sleep(sleep_interval)

        print("[FullyAsyncRollouter][Public][Pause] All active samples completed")

    async def resume(self, dependency_ref: ObjectRef = None):
        if dependency_ref is not None:
            ray.get(dependency_ref)
        print("[FullyAsyncRollouter][Public][Resume]")
        async with self.lock:
            self.paused = False
            self.monitor_loop_trigger = True
            self.condition.notify_all()

        async with self.condition:
            self.condition.notify_all()

            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.resume()

    async def get_statistics(self) -> dict:
        queue_stats = self.message_queue_client.get_statistics_sync()

        stats = {
            # monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/active_sample_count": self.active_sample_count,
            "monitor/cancel_sample_count": self.cancel_sample_count,
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            "monitor/queue/cancel_queue_size": self.cancel_queue.qsize(),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            # counting stats
            "count/current_param_version": self.current_param_version,
            "count/total_generated_samples": self.total_generated_samples,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            # static stats
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
        }

        return stats
