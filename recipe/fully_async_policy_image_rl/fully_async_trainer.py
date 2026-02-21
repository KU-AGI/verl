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
from typing import Any
import threading
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
from recipe.image_rl.custom_metric_utils import reduce_metrics # custom metric
from recipe.image_rl.reward import compute_reward, compute_reward_async
import asyncio

def log_prob_metrics(metrics):
    import numpy as np 

    log_prob_info = {}

    for task_id in [1]:
        task_pos_log_prob = f"actor/task{task_id}_pos_log_prob"
        task_neg_log_prob = f"actor/task{task_id}_neg_log_prob"
        task_pos_log_prob_cnt = f"actor/task{task_id}_pos_log_prob_cnt"
        task_neg_log_prob_cnt = f"actor/task{task_id}_neg_log_prob_cnt"

        pos_log_prob_sum = np.array(metrics.pop(task_pos_log_prob)).sum()
        pos_log_prob_cnt = np.array(metrics.pop(task_pos_log_prob_cnt)).sum()
        neg_log_prob_sum = np.array(metrics.pop(task_neg_log_prob)).sum()
        neg_log_prob_cnt = np.array(metrics.pop(task_neg_log_prob_cnt)).sum()

        log_prob_info[f"actor/task{task_id}_pos_log_prob_mean"] = pos_log_prob_sum / pos_log_prob_cnt if pos_log_prob_cnt > 0 else 0.0
        log_prob_info[f"actor/task{task_id}_neg_log_prob_mean"] = neg_log_prob_sum / neg_log_prob_cnt if neg_log_prob_cnt > 0 else 0.0

        log_prob_info[f"actor/task{task_id}_log_probs_diff"] = log_prob_info[f"actor/task{task_id}_pos_log_prob_mean"] - log_prob_info[f"actor/task{task_id}_neg_log_prob_mean"]
    
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
                f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                break

            deserialized_sample = ray.cloudpickle.loads(sample_data)

            sample_ver = deserialized_sample.param_version
            version_gap = self.current_param_version - sample_ver

            # version_gap > 0이면 on_plolicy
            # default는 verison_gap > staleness_threshold
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

        if not queue_samples or current_rows < self.required_samples:
            print(f"[FullyAsyncTrainer] Not enough rows collected: {current_rows}/{self.required_samples}")
            return None, None

        total_wait_time = consumer_end - consumer_start

        print(
            f"[FullyAsyncTrainer] Collection completed: {current_rows} rows from {len(queue_samples)} groups. "
            f"Wait time: {total_wait_time:.2f}s."
            f"Dropped sample: {dropped_rows}"
        )
        print(f"[DEBUG] Balance Batch: {self.config.trainer.balance_batch}")
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        batch.meta_info["fully_async/total_sample_count"] = current_rows
        batch.meta_info["fully_async/batch_size"] = current_rows
        
        return 0, batch

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

        #self.prefetch_thread = threading.Thread(target=self._run_prefetch, daemon=True)
        #self.prefetch_thread.start()
        self.max_steps_duration = 0

        # get validate data before training
        self._log_validation_data()

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            metrics = {}
            timing_raw = {}

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw, color="red"):
                    epoch, batch = self._get_samples_from_queue()
                    training_start_time = time.time()
                    if batch is None: break
                    self._collect_metrics_from_samples(batch, metrics)

                    # Collect timing info from rollouter (reward computation time)
                    if hasattr(batch, 'meta_info') and 'reward' in batch.meta_info:
                        timing_raw['reward'] = batch.meta_info['reward']

                    # async_training = self.config.get("async_training", None)
                    # use_mis = async_training and async_training.use_rollout_log_probs
                    # local_trigger = self.local_trigger_step if self.compute_prox_log_prob else None
                    # should_swap = use_mis and local_trigger is not None and local_trigger > 1
                    
                    # if use_mis:
                    #     if local_trigger == 1:
                    #         self.actor_rollout_wg.save_model_to_cpu(1)
                    #     elif should_swap:
                    #         self.actor_rollout_wg.save_model_to_cpu(local_trigger)
                    #         self.actor_rollout_wg.restore_model_from_cpu(1)

                    for task_id in [1]:
                        batch.batch["task_id"] = torch.tensor([task_id for _ in range(len(batch))], dtype=int)

                        batch = self._process_batch_common(
                            batch, metrics, timing_raw, self.local_trigger_step if self.compute_prox_log_prob else None, task_id
                        )

                    # if should_swap:
                    #     self.actor_rollout_wg.restore_model_from_cpu(local_trigger)
                    #     self.actor_rollout_wg.clear_cpu_model(local_trigger)
                
                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        log_prob_info = log_prob_metrics(actor_output.meta_info["metrics"])
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(log_prob_info)
                        metrics.update(actor_output_metrics)

                    reward_extra_infos_dict: dict[str, list] = {}
                    for task_id in [1]:
                        # Get pre-computed reward_extra_infos_dict from meta_info
                        reward_extra_infos_dict.update({k: v for k, v in batch.meta_info.items()})

                    # Log rollout generations if enabled (per task)
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if self.config.trainer.rollout_freq > 0 and (
                        self.global_steps % self.config.trainer.rollout_freq == 0 and rollout_data_dir
                    ):
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

            self._collect_metrics(batch, 0, metrics, timing_raw)
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
        #self.stop_prefetch = True

        self._check_save_checkpoint(timing_raw)

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
        return self.current_param_version

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