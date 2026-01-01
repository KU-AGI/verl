# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from copy import deepcopy
from pprint import pprint
from typing import Any, Dict, Optional, Type

import numpy as np
import ray
import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
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
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.torch_functional import masked_mean
from recipe.image_rl.ray_image_generation_trainer import RayImageGenerationTrainer
from recipe.image_rl import core_algos
import asyncio


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


class FullyAsyncRayPPOTrainer(RayImageGenerationTrainer):
    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        self._init_async_rollout_manager()

    def _init_resource_pools(self):
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

    def _create_worker_classes(self):
        self._create_actor_rollout_classes()
        self._create_critic_class()
        self._create_reference_policy_class()
        self._create_reward_model_class()

    def _create_actor_rollout_classes(self):
        raise NotImplementedError

    def _create_critic_class(self):
        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

    def _create_reference_policy_class(self):
        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
                # profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

    def _create_reward_model_class(self):
        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

    def _init_worker_groups(self):
        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
        self.all_wg = all_wg

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

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = self.all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

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

    async def _validate_async(self):

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        val_data_dir = self.config.trainer.get("validation_data_dir", "validation")

        # task reward tensors collection
        task_reward_tensors = {1: [], 2: [], 3: []}
        task_reward_extra_infos = {1: [], 2: [], 3: []}
        data_source_lst = []

        # Lists to collect samples for the table
        sample_prompts = []
        sample_task1_gen_imgs = []
        sample_task2_feedback_texts = []
        sample_task3_regen_imgs = []
        sample_task1_scores = []
        sample_task2_scores = []
        sample_task3_scores = []

        # For async mode, create a server token queue similar to rollouter
        if self.async_rollout_mode:
            async def async_validate_all():
                num_servers = len(self.async_rollout_manager.server_handles)
                val_server_token_q = asyncio.Queue()
                for sid in range(num_servers):
                    val_server_token_q.put_nowait(sid)

                # Producer-consumer queues
                prepared_batch_queue = asyncio.Queue(maxsize=num_servers * 2)  # Limited buffer
                reward_finalize_queue = asyncio.Queue()  # For reward computation
                result_queue = asyncio.Queue()

                async def _val_producer_worker():
                    """Producer: Prepare validation batches and put them in queue"""
                    # Get rollout_prompt_size (number of batches to merge)
                    val_rollout_prompt_size = self.config.async_training.get("val_rollout_prompt_size", 1)

                    batch_list = []
                    batch_idx_list = []

                    for val_batch_idx, test_data in enumerate(self.val_dataloader):
                        # Prepare batch (heavy computation done in producer)
                        test_batch = DataProto.from_single_dict(test_data)

                        # # repeat test batch
                        # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                        #                             interleave=True)

                        # add uid to batch
                        test_batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                        )

                        # Accumulate batches
                        batch_list.append(test_batch)
                        batch_idx_list.append(val_batch_idx)

                        # Send accumulated batches when reaching target size
                        if len(batch_list) >= val_rollout_prompt_size:
                            merged_batch = DataProto.concat(batch_list)
                            merged_gen_batch = self._get_gen_batch(merged_batch)
                            await prepared_batch_queue.put((merged_batch, merged_gen_batch, batch_idx_list))
                            batch_list = []
                            batch_idx_list = []

                    # Send remaining batches
                    if batch_list:
                        merged_batch = DataProto.concat(batch_list)
                        merged_gen_batch = self._get_gen_batch(merged_batch)
                        await prepared_batch_queue.put((merged_batch, merged_gen_batch, batch_idx_list))

                    # Send done signals (one per consumer)
                    for _ in range(num_servers):
                        await prepared_batch_queue.put(None)

                async def _val_reward_worker():
                    """Reward worker: Process reward computations and post-processing separately"""
                    while True:
                        item = await reward_finalize_queue.get()

                        if item is None:
                            reward_finalize_queue.task_done()
                            break

                        test_batch, test_output_gen_batch, reward_tensor_dict, reward_reward_extra_infos, reward_tasks, val_batch_idx = item

                        # Wait for all reward computations to finish
                        await asyncio.gather(*reward_tasks)

                        # Do post-processing here (instead of at the end)
                        prompts = test_output_gen_batch.non_tensor_batch['prompt'].tolist()

                        task1_gen_imgs_pil_list = test_output_gen_batch.non_tensor_batch['task1_gen_imgs_pil_list']
                        task1_gen_imgs_pil_list = [wandb.Image(gen_img, caption=prompts[i]) for i, gen_img in enumerate(task1_gen_imgs_pil_list)]

                        task2_feedback_texts = test_output_gen_batch.non_tensor_batch['task2_feedback_texts'].tolist()

                        task3_regen_imgs_pil_list = test_output_gen_batch.non_tensor_batch['task3_regen_imgs_pil_list']
                        task3_regen_imgs_pil_list = [wandb.Image(regen_img, caption=prompts[i]) for i, regen_img in enumerate(task3_regen_imgs_pil_list)]

                        # Collect batch results (keep as batch for efficient queue operation)
                        batch_size = len(prompts)
                        data_source = test_batch.non_tensor_batch.get('data_source', ['unknown'] * batch_size)

                        # Compute per-sample scores
                        batch_scores = {1: [], 2: [], 3: []}
                        for task_id in [1, 2, 3]:
                            task_reward_tensor = reward_tensor_dict[task_id]
                            # Sum over sequence dimension (-1) to get per-sample scores
                            per_sample_scores = task_reward_tensor.sum(-1).detach().cpu().tolist()
                            # Handle both single sample and batch cases
                            if not isinstance(per_sample_scores, list):
                                per_sample_scores = [per_sample_scores]
                            batch_scores[task_id] = per_sample_scores

                        # Put batch result into result queue (one queue operation per batch)
                        result_item = {
                            'prompts': prompts,
                            'task1_gen_imgs': task1_gen_imgs_pil_list,
                            'task2_feedback_texts': task2_feedback_texts,
                            'task3_regen_imgs': task3_regen_imgs_pil_list,
                            'reward_tensors': reward_tensor_dict,
                            'reward_extra_infos': reward_reward_extra_infos,
                            'scores': batch_scores,
                            'data_source': data_source,
                        }
                        await result_queue.put(result_item)

                        reward_finalize_queue.task_done()

                async def _val_consumer_worker():
                    """Consumer: Get prepared batch and run generation on server"""
                    # Get available server token ONCE at the start
                    server_index = await val_server_token_q.get()

                    try:
                        while True:
                            # Get prepared batch from queue (server already acquired)
                            item = await prepared_batch_queue.get()

                            if item is None:
                                # Done signal - mark as done and exit
                                prepared_batch_queue.task_done()
                                break

                            test_batch, test_gen_batch, val_batch_idx_list = item

                            # Generation with callback for reward computation
                            reward_tensor_dict = {}
                            reward_reward_extra_infos = {}
                            reward_tasks = []

                            def on_task_complete(task_id: int, batch_result: DataProto):
                                # Launch async reward computation task
                                async def compute_reward_async():
                                    task_reward_dict = self.val_reward_fn(batch_result, task_id=task_id, eval=True, return_dict=True)
                                    reward_tensor_dict[task_id] = task_reward_dict[f"task{task_id}_reward_tensor"]
                                    reward_reward_extra_infos[task_id] = task_reward_dict[f"task{task_id}_reward_extra_info"]

                                reward_task = asyncio.create_task(compute_reward_async())
                                reward_tasks.append(reward_task)

                            # Use generate_sequences_with_callback_on_server
                            test_output_gen_batch = await self.async_rollout_manager.generate_sequences_with_callback_on_server(
                                test_gen_batch,
                                server_index=server_index,
                                on_task_complete=on_task_complete
                            )
                            
                            val_server_token_q.put_nowait(server_index)

                            # Send to reward worker for finalization (don't wait here!)
                            await reward_finalize_queue.put((test_batch, test_output_gen_batch, reward_tensor_dict, reward_reward_extra_infos, reward_tasks, val_batch_idx_list))

                            # Yield control to allow reward worker to process
                            await asyncio.sleep(0)

                            # Mark task as done in prepared_batch_queue
                            prepared_batch_queue.task_done()

                            # Immediately continue to next batch (server stays busy!)

                    finally:
                        pass

                # Start producer
                producer_task = asyncio.create_task(_val_producer_worker())

                # Start reward worker
                async def _val_reward_worker_wrapper():
                    try:
                        await _val_reward_worker()
                    except Exception as e:
                        print(f'[VALIDATION ERROR] Reward worker failed with exception: {e}')
                        import traceback
                        traceback.print_exc()
                        raise

                reward_worker_task = asyncio.create_task(_val_reward_worker_wrapper())

                # Start consumers (one per server)
                consumer_tasks = [
                    asyncio.create_task(_val_consumer_worker())
                    for _ in range(num_servers)
                ]

                # Wait for producer to finish
                await producer_task

                # Wait for all consumers to finish
                await asyncio.gather(*consumer_tasks)

                # Wait for reward queue to be processed
                await reward_finalize_queue.join()

                # Send done signal to reward worker
                await reward_finalize_queue.put(None)
                await reward_worker_task

                # Collect all results from result queue (batch results)
                while not result_queue.empty():
                    result_item = await result_queue.get()

                    # Extend with batch data
                    sample_prompts.extend(result_item['prompts'])
                    sample_task1_gen_imgs.extend(result_item['task1_gen_imgs'])
                    sample_task2_feedback_texts.extend(result_item['task2_feedback_texts'])
                    sample_task3_regen_imgs.extend(result_item['task3_regen_imgs'])

                    # Collect reward tensors (keep as batch tensors)
                    for task_id in [1, 2, 3]:
                        task_reward_tensors[task_id].append(result_item['reward_tensors'][task_id])
                        task_reward_extra_infos[task_id].append(result_item['reward_extra_infos'][task_id])

                    # Extend with per-sample scores
                    sample_task1_scores.extend(result_item['scores'][1])
                    sample_task2_scores.extend(result_item['scores'][2])
                    sample_task3_scores.extend(result_item['scores'][3])

                    data_source_lst.extend(result_item['data_source'])

            # Run async validation
            await async_validate_all()
            print('Validation completed')

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
            if self.async_rollout_mode:
                import asyncio
                val_metrics = asyncio.run(self._validate_async())
            else:
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

                batch, gen_batch = self._prepare_generate_batch(batch_dict)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch = self._post_generate_batch(batch, gen_batch_output, metrics)
                    for task_id in [1, 2, 3]:
                        batch.batch["task_id"] = torch.tensor([task_id for _ in range(len(batch))], dtype=int)
                        batch = self._process_batch_common(batch, metrics, timing_raw, local_trigger_step=None, task_id=task_id)
                        # Remove universal keys from batch
                        batch.pop(batch_keys=["task_id"])
                    
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
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Reward is already in batch from rollouter
                    # Get reward_extra_infos_dict from meta_info if available
                    reward_extra_infos_dict = batch.meta_info.get(f"task{task_id}_reward_extra_info", {})

                    # if reward_extra_infos_dict: # remove reward extra info to non_tensor_batch for logging
                    #     batch.non_tensor_batch.update({k: np.array(v, dtype=object) for k, v in reward_extra_infos_dict.items()})

                    # Log rollout generations if enabled (per task)
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if self.config.trainer.rollout_freq > 0 and (
                        self.global_steps % self.config.trainer.rollout_freq == 0 and rollout_data_dir
                    ):
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                last_val_metrics = self._validate_metrics(is_last_step, last_val_metrics, metrics, timing_raw)
                self._check_save_checkpoint(is_last_step, timing_raw)

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

                self._collect_metrics(batch, epoch, metrics, timing_raw)
                self._post_batch_processing(batch)

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

    def _prepare_generate_batch(self, batch_dict):
        batch: DataProto = DataProto.from_single_dict(batch_dict)

        # add uid to batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        gen_batch = self._get_gen_batch(batch)

        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        return batch, gen_batch

    def _post_generate_batch(self, batch, gen_batch_output, metrics):
        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        # if "response_mask" not in batch.batch.keys():
        #     batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        # TODO: Decouple the DP balancing and mini-batching.

        # compute global_valid tokens
        batch.batch["attention_mask"] = batch.batch["task1_attention_mask"].sum(dim=-1) + batch.batch["task1_response_mask"].sum(dim=-1) \
                                      + batch.batch["task2_attention_mask"].sum(dim=-1) + batch.batch["task2_response_mask"].sum(dim=-1) \
                                      + batch.batch["task3_attention_mask"].sum(dim=-1) + batch.batch["task3_response_mask"].sum(dim=-1)
        
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        return batch

    def _process_batch_common(self, batch, metrics, timing_raw, local_trigger_step=None, task_id: int = 1):
        # Compute reward if using reward model worker (use_rm)
        # Otherwise reward is already computed in rollouter
        with marked_timer("reward", timing_raw, color="yellow"):
            if self.use_rm:
                # Compute reward using reward model worker
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                # Remove existing response_masks to allow overwriting with modified masks from reward computation
                for tid in [1, 2, 3]:
                    if f"task{tid}_response_mask" in batch.batch:
                        batch.batch.pop(f"task{tid}_response_mask")
                batch = batch.union(reward_tensor)

        with marked_timer("old_log_prob", timing_raw, color="blue"):

            def compute_old_log_prob(batch):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch[f"task{task_id}_entropys"]
                response_masks = batch.batch[f"task{task_id}_response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {f"actor/task{task_id}_entropy": entropy_agg.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop(f"task{task_id}_entropys")
                batch = batch.union(old_log_prob)
                if f"task{task_id}_rollout_log_probs" in batch.batch.keys(): ### Not use
                    # TODO: we may want to add diff of probs too.
                    from verl.utils.debug.metrics import calculate_debug_metrics

                    metrics.update(calculate_debug_metrics(batch))
                return batch

            async_training = self.config.get("async_training", None)
            if async_training and async_training.use_rollout_log_probs:  ### Not use
                # If local_triger_step == 1, load the training engine's parameters to the CPU
                #  and save a copy for subsequent MIS use.
                # If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
                # then restore the parameters of the current version.
                if local_trigger_step is not None:
                    batch = compute_old_log_prob(batch)
                else:
                    batch.batch[f"task{task_id}_old_log_probs"] = batch.batch[f"task{task_id}_rollout_log_probs"]
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                print("[FullyAsyncRayPPOTrainer] restore model from cpu done.")
            else:
                batch = compute_old_log_prob(batch)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # compute values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

        with marked_timer("adv", timing_raw, color="brown"):

            # compute rewards. apply_kl_penalty if available
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty, task_id=task_id
                )
                metrics.update(kl_metrics)
            else:
                batch.batch[f"task{task_id}_token_level_rewards"] = batch.batch[f"task{task_id}_token_level_scores"]

            # Compute rollout correction weights centrally (once per batch)
            # This corrects for off-policy issues (policy mismatch, model staleness, etc.)
            # Also computes off-policy diagnostic metrics (KL, PPL, etc.)
            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch)
                # IS and off-policy metrics already have rollout_corr/ prefix
                metrics.update(is_metrics)

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

        return batch

    def _validate_metrics(self, is_last_step, last_val_metrics, metrics, timing_raw):
        if (
            self.val_reward_fn is not None
            and self.config.trainer.test_freq > 0
            and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
        ):
            with marked_timer("testing", timing_raw, color="green"):
                if self.async_rollout_mode:
                    import asyncio
                    val_metrics: dict = asyncio.run(self._validate_async())
                else:
                    val_metrics: dict = self._validate()
                if is_last_step:
                    last_val_metrics = val_metrics
            metrics.update(val_metrics)
            return last_val_metrics

    def _check_save_checkpoint(self, is_last_step, timing_raw):
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

    def _collect_metrics(self, batch, epoch, metrics, timing_raw):
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

    def _post_batch_processing(self, batch: DataProto):
        # this is experimental and may be changed/removed in the future in favor of a general-purpose one
        if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
            self.train_dataloader.sampler.update(batch=batch)

        # this is experimental and may be changed/removed in the future
        # in favor of a general-purpose data buffer pool
        if hasattr(self.train_dataset, "on_batch_end"):
            # The dataset may be changed after each training batch
            self.train_dataset.on_batch_end(batch=batch)
