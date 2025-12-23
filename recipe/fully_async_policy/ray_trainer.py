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
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.rollout_skip import RolloutSkip

import re
from typing import List, Dict
from collections import Counter, defaultdict


class FullyAsyncRayPPOTrainer(RayPPOTrainer):
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

    def _init_async_rollout_manager(self):
        pass

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

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
        last_test_metrics = None
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
                    batch, reward_extra_infos_dict = self._process_batch_common(batch, metrics, timing_raw)
                    self._log_rollout(batch, reward_extra_infos_dict, timing_raw)

                last_val_metrics = self._validate_metrics(is_last_step, last_val_metrics, metrics, timing_raw)
                last_test_metrics = self._test_metrics(is_last_step, last_test_metrics, metrics, timing_raw)
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
                    pprint(f"Final test metrics: {last_test_metrics}")
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

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        # TODO: Decouple the DP balancing and mini-batching.
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        return batch

    def _process_batch_common(self, batch, metrics, timing_raw, local_trigger_step=None):
        with marked_timer("reward", timing_raw, color="yellow"):
            # compute reward model score
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)

            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                # Append 'None' padding for task-specific extra info
                tasks = batch.non_tensor_batch['task']
                reward_extra_infos_dict = self._expand_taskwise_lists(reward_extra_infos_dict, tasks)


        with marked_timer("old_log_prob", timing_raw, color="blue"):

            def compute_old_log_prob(batch):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)
                if "rollout_log_probs" in batch.batch.keys():
                    # TODO: we may want to add diff of probs too.
                    from verl.utils.debug.metrics import calculate_debug_metrics

                    metrics.update(calculate_debug_metrics(batch))
                return batch

            async_training = self.config.get("async_training", None)
            if async_training and async_training.use_rollout_log_probs:
                # If local_triger_step == 1, load the training engine's parameters to the CPU
                #  and save a copy for subsequent MIS use.
                # If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
                # then restore the parameters of the current version.
                if local_trigger_step == 1:
                    self.actor_rollout_wg.save_model_to_cpu(1)
                    batch = compute_old_log_prob(batch)
                elif local_trigger_step is not None:
                    self.actor_rollout_wg.save_model_to_cpu(local_trigger_step)
                    self.actor_rollout_wg.restore_model_from_cpu(1)
                    batch = compute_old_log_prob(batch)
                    self.actor_rollout_wg.restore_model_from_cpu(local_trigger_step)
                    self.actor_rollout_wg.clear_cpu_model(local_trigger_step)
                else:
                    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

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
            # we combine with rule-based rm
            reward_extra_infos_dict: dict[str, list]
            if self.config.reward_model.launch_reward_fn_async:
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            batch.batch["token_level_scores"] = reward_tensor

            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            # compute rewards. apply_kl_penalty if available
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # Compute rollout importance sampling weights centrally (once per batch)
            # This corrects for mismatch between rollout policy and training policy
            # Also computes mismatch metrics (KL, PPL, etc.)
            batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
            # IS and mismatch metrics already have mismatch/ prefix
            metrics.update(is_metrics)

            # compute advantages, executed on the driver process
            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            )  # GRPO adv normalization factor
            # breakpoint()
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )
        # breakpoint()
        # for uid in list(set(batch.non_tensor_batch['uid'])):
        #     uid_inds = np.where(batch.non_tensor_batch['uid'] == uid)[0]
        #     input_ids = batch.batch['input_ids'][uid_inds]
        #     advantages = batch.batch['advantages'][uid_inds]
        #     responses = batch.batch['responses'][uid_inds]
        #     token_level_rewards = batch.batch['token_level_rewards'][uid_inds]
        #     reagents_all = batch.non_tensor_batch['reagents'][uid_inds]
        #     step_rewards_all = batch.non_tensor_batch['step_rewards'][uid_inds]
        #     content_rewards_all = batch.non_tensor_batch['reagent/reward/content_reward'][uid_inds]
        #     decision_rewards_all = batch.non_tensor_batch['reagent/reward/decision_reward'][uid_inds]
        #     reflection_bonus_rewards_all = batch.non_tensor_batch['reagent/reward/reflection_bonus_reward'][uid_inds]
        #     answer_correct_all = batch.non_tensor_batch['reagent/reward/answer_correct'][uid_inds]
        #     valid_structure_all = batch.non_tensor_batch['reagent/reward/valid_structure'][uid_inds]

        #     for i in range(len(uid_inds)):
        #         print(f"=== uid: {uid} | sample {i} ===")
        #         print("responses: ", self.tokenizer.decode(responses[i], skip_special_tokens=True))
        #         print("reagents: ", ".".join(reagents_all[i]))
        #         print("step_rewards: ", step_rewards_all[i])
        #         print("content_rewards: ", content_rewards_all[i])
        #         print("decision_rewards: ", decision_rewards_all[i])
        #         print("reflection_bonus_rewards: ", reflection_bonus_rewards_all[i])
        #         print("answer_correct: ", answer_correct_all[i])
        #         print("valid_structure: ", valid_structure_all[i])
        #         print("advantages: ", advantages[i].tolist())
        #         # print("token_level_rewards: ", token_level_rewards[i].tolist())
        #         print("-"*50)

        #     print("="*100)

            # break




# uid = list(set(batch.non_tensor_batch['uid']))[18]
# uid_inds = np.where(batch.non_tensor_batch['uid'] == uid)[0]
# input_ids = batch.batch['input_ids'][uid_inds]
# responses = batch.batch['responses'][uid_inds]
# token_level_rewards = batch.batch['token_level_rewards'][uid_inds]
# reagents_all = batch.non_tensor_batch['reagents'][uid_inds]
# step_rewards_all = batch.non_tensor_batch['step_rewards'][uid_inds]
# content_rewards_all = batch.non_tensor_batch['reagent/reward/content_reward'][uid_inds]
# decision_rewards_all = batch.non_tensor_batch['reagent/reward/decision_reward'][uid_inds]
# reflection_bonus_rewards_all = batch.non_tensor_batch['reagent/reward/reflection_bonus_reward'][uid_inds]
# answer_correct_all = batch.non_tensor_batch['reagent/reward/answer_correct'][uid_inds]
# valid_structure_all = batch.non_tensor_batch['reagent/reward/valid_structure'][uid_inds]

# for i in range(len(uid_inds)):
#     print(f"=== uid: {uid} | sample {i} ===")
#     if "<REFLECTION>" in self.tokenizer.decode(responses[i], skip_special_tokens=True):
#         print("responses: ", self.tokenizer.decode(responses[i], skip_special_tokens=True))
#         print("valid", validate_structure(self.tokenizer.decode(responses[i], skip_special_tokens=True), "reagent"))
#         print("reagents: ", ".".join(reagents_all[i]))
#         print("step_rewards: ", step_rewards_all[i])
#         print("content_rewards: ", content_rewards_all[i])
#         print("decision_rewards: ", decision_rewards_all[i])
#         print("reflection_bonus_rewards: ", reflection_bonus_rewards_all[i])
#         print("answer_correct: ", answer_correct_all[i])
#         print("valid_structure: ", valid_structure_all[i])
#         # print("token_level_rewards: ", token_level_rewards[i].tolist())
#         print("-"*50)
# print("="*100)



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
        return batch, reward_extra_infos_dict

    def _log_rollout(self, batch, reward_extra_infos_dict, timing_raw):
        # Log rollout generations if enabled
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

                if "request_id" in batch.non_tensor_batch:
                    reward_extra_infos_dict.setdefault(
                        "request_id",
                        batch.non_tensor_batch["request_id"].tolist(),
                    )

                self._dump_generations(
                    inputs=inputs,
                    outputs=outputs,
                    gts=sample_gts,
                    scores=scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=rollout_data_dir,
                )

    def _validate_metrics(self, is_last_step, last_val_metrics, metrics, timing_raw):
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
            return last_val_metrics
    
    def _test_metrics(self, is_last_step, last_test_metrics, metrics, timing_raw):
        if (
            self.val_reward_fn is not None
            and self.config.trainer.test_freq > 0
            and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
        ):
            with marked_timer("testing", timing_raw, color="green"):
                test_metrics: dict = self._test()
                if is_last_step:
                    last_test_metrics = test_metrics
            metrics.update(test_metrics)
            return last_test_metrics

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
        # collect metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        # TODO: implement actual tflpo and theoretical tflpo
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        # ReactionReasoner-specific metrics
        class_counter = Counter(batch.non_tensor_batch['class_name'])
        for class_name, count in class_counter.items():
            metrics[f'reaction_class_count/{class_name}'] = count
        task_counter = Counter(batch.non_tensor_batch['task'])
        for task_name, count in task_counter.items():
            metrics[f'task_count/{task_name}'] = count

        tasks = list(set(batch.non_tensor_batch['task']))
        for task in tasks:
            metrics[f"group_metrics/{task}/zero_std_num"] = 0
        task_group_metrics = defaultdict(list)
        uids = list(set(batch.non_tensor_batch['uid']))

        for uid in uids:
            uid_inds = np.where(batch.non_tensor_batch['uid'] == uid)[0]
            task = batch.non_tensor_batch['task'][uid_inds[0]]
            scores = batch.non_tensor_batch['score'][uid_inds]
            all_responses = [self.tokenizer.decode(batch.batch['responses'][idx]).replace("<|endoftext|>", "") for idx in uid_inds]
            all_parsed = [self._parse_steps_with_reflections(resp) for resp in all_responses]
            accs = batch.non_tensor_batch['acc'][uid_inds]
            answer_exact_k = any(accs)
            answer_exact_all = all(accs)
            answer_all_same = all(a == accs[0] for a in accs)
            answer_not_all_same = not answer_all_same
            answer_not_exact_all = all(not a for a in accs)

            scores_all_same = all(s == scores[0] for s in scores)
            scores_not_all_same = not scores_all_same

            if task == "forward":
                accs_1 = batch.non_tensor_batch[f'{task}/correct_molecular_role'][uid_inds]
                accs_2 = batch.non_tensor_batch[f'{task}/correct_precursor_stat'][uid_inds]
                accs_3 = batch.non_tensor_batch[f'{task}/correct_reactant_funcgroup'][uid_inds]
                accs_4 = batch.non_tensor_batch[f'{task}/correct_template'][uid_inds]

                exact_k_1 = any(accs_1)
                exact_k_2 = any(accs_2)
                exact_k_3 = any(accs_3)
                exact_k_4 = any(accs_4)
                exact_all_1 = all(accs_1)
                exact_all_2 = all(accs_2)
                exact_all_3 = all(accs_3)
                exact_all_4 = all(accs_4)
                all_same_1 = all(a == accs_1[0] for a in accs_1)
                all_same_2 = all(a == accs_2[0] for a in accs_2)
                all_same_3 = all(a == accs_3[0] for a in accs_3)
                all_same_4 = all(a == accs_4[0] for a in accs_4)
                not_all_same_1 = not all_same_1
                not_all_same_2 = not all_same_2
                not_all_same_3 = not all_same_3
                not_all_same_4 = not all_same_4
                not_exact_all_1 = all(not a for a in accs_1)
                not_exact_all_2 = all(not a for a in accs_2)
                not_exact_all_3 = all(not a for a in accs_3)
                not_exact_all_4 = all(not a for a in accs_4)

                task_group_metrics[f"group_metrics/{task}/correct_molecular_role/exact_k"].append(exact_k_1)
                task_group_metrics[f"group_metrics/{task}/correct_molecular_role/exact_all"].append(exact_all_1)
                task_group_metrics[f"group_metrics/{task}/correct_molecular_role/all_same"].append(all_same_1)
                task_group_metrics[f"group_metrics/{task}/correct_molecular_role/not_all_same"].append(not_all_same_1)
                task_group_metrics[f"group_metrics/{task}/correct_molecular_role/not_exact_all"].append(not_exact_all_1)
                task_group_metrics[f"group_metrics/{task}/correct_precursor_stat/exact_k"].append(exact_k_2)
                task_group_metrics[f"group_metrics/{task}/correct_precursor_stat/exact_all"].append(exact_all_2)
                task_group_metrics[f"group_metrics/{task}/correct_precursor_stat/all_same"].append(all_same_2)
                task_group_metrics[f"group_metrics/{task}/correct_precursor_stat/not_all_same"].append(not_all_same_2)
                task_group_metrics[f"group_metrics/{task}/correct_precursor_stat/not_exact_all"].append(not_exact_all_2)
                task_group_metrics[f"group_metrics/{task}/correct_reactant_funcgroup/exact_k"].append(exact_k_3)
                task_group_metrics[f"group_metrics/{task}/correct_reactant_funcgroup/exact_all"].append(exact_all_3)
                task_group_metrics[f"group_metrics/{task}/correct_reactant_funcgroup/all_same"].append(all_same_3)
                task_group_metrics[f"group_metrics/{task}/correct_reactant_funcgroup/not_all_same"].append(not_all_same_3)
                task_group_metrics[f"group_metrics/{task}/correct_reactant_funcgroup/not_exact_all"].append(not_exact_all_3)
                task_group_metrics[f"group_metrics/{task}/correct_template/exact_k"].append(exact_k_4)
                task_group_metrics[f"group_metrics/{task}/correct_template/exact_all"].append(exact_all_4)
                task_group_metrics[f"group_metrics/{task}/correct_template/all_same"].append(all_same_4)
                task_group_metrics[f"group_metrics/{task}/correct_template/not_all_same"].append(not_all_same_4)
                task_group_metrics[f"group_metrics/{task}/correct_template/not_exact_all"].append(not_exact_all_4)
            elif task == "retro":
                accs_1 = batch.non_tensor_batch[f'{task}/correct_product_funcgroup'][uid_inds]
                accs_2 = batch.non_tensor_batch[f'{task}/correct_product_stat'][uid_inds]
                accs_3 = batch.non_tensor_batch[f'{task}/correct_bond_disconnect'][uid_inds]
                accs_4 = batch.non_tensor_batch[f'{task}/correct_synthon'][uid_inds]
                accs_5 = batch.non_tensor_batch[f'{task}/correct_synthetic_equivalent'][uid_inds]

                exact_k_1 = any(accs_1)
                exact_k_2 = any(accs_2)
                exact_k_3 = any(accs_3)
                exact_k_4 = any(accs_4)
                exact_k_5 = any(accs_5)
                exact_all_1 = all(accs_1)
                exact_all_2 = all(accs_2)
                exact_all_3 = all(accs_3)
                exact_all_4 = all(accs_4)
                exact_all_5 = all(accs_5)
                all_same_1 = all(a == accs_1[0] for a in accs_1)
                all_same_2 = all(a == accs_2[0] for a in accs_2)
                all_same_3 = all(a == accs_3[0] for a in accs_3)
                all_same_4 = all(a == accs_4[0] for a in accs_4)
                all_same_5 = all(a == accs_5[0] for a in accs_5)
                not_all_same_1 = not all_same_1
                not_all_same_2 = not all_same_2
                not_all_same_3 = not all_same_3
                not_all_same_4 = not all_same_4
                not_all_same_5 = not all_same_5
                not_exact_all_1 = all(not a for a in accs_1)
                not_exact_all_2 = all(not a for a in accs_2)
                not_exact_all_3 = all(not a for a in accs_3)
                not_exact_all_4 = all(not a for a in accs_4)
                not_exact_all_5 = all(not a for a in accs_5)
                task_group_metrics[f"group_metrics/{task}/correct_product_funcgroup/exact_k"].append(exact_k_1)
                task_group_metrics[f"group_metrics/{task}/correct_product_funcgroup/exact_all"].append(exact_all_1)
                task_group_metrics[f"group_metrics/{task}/correct_product_funcgroup/all_same"].append(all_same_1)
                task_group_metrics[f"group_metrics/{task}/correct_product_funcgroup/not_all_same"].append(not_all_same_1)
                task_group_metrics[f"group_metrics/{task}/correct_product_funcgroup/not_exact_all"].append(not_exact_all_1)
                task_group_metrics[f"group_metrics/{task}/correct_product_stat/exact_k"].append(exact_k_2)
                task_group_metrics[f"group_metrics/{task}/correct_product_stat/exact_all"].append(exact_all_2)
                task_group_metrics[f"group_metrics/{task}/correct_product_stat/all_same"].append(all_same_2)
                task_group_metrics[f"group_metrics/{task}/correct_product_stat/not_all_same"].append(not_all_same_2)
                task_group_metrics[f"group_metrics/{task}/correct_product_stat/not_exact_all"].append(not_exact_all_2)
                task_group_metrics[f"group_metrics/{task}/correct_bond_disconnect/exact_k"].append(exact_k_3)
                task_group_metrics[f"group_metrics/{task}/correct_bond_disconnect/exact_all"].append(exact_all_3)
                task_group_metrics[f"group_metrics/{task}/correct_bond_disconnect/all_same"].append(all_same_3)
                task_group_metrics[f"group_metrics/{task}/correct_bond_disconnect/not_all_same"].append(not_all_same_3)
                task_group_metrics[f"group_metrics/{task}/correct_bond_disconnect/not_exact_all"].append(not_exact_all_3)
                task_group_metrics[f"group_metrics/{task}/correct_synthon/exact_k"].append(exact_k_4)
                task_group_metrics[f"group_metrics/{task}/correct_synthon/exact_all"].append(exact_all_4)
                task_group_metrics[f"group_metrics/{task}/correct_synthon/all_same"].append(all_same_4)
                task_group_metrics[f"group_metrics/{task}/correct_synthon/not_all_same"].append(not_all_same_4)
                task_group_metrics[f"group_metrics/{task}/correct_synthon/not_exact_all"].append(not_exact_all_4)
                task_group_metrics[f"group_metrics/{task}/correct_synthetic_equivalent/exact_k"].append(exact_k_5)
                task_group_metrics[f"group_metrics/{task}/correct_synthetic_equivalent/exact_all"].append(exact_all_5)
                task_group_metrics[f"group_metrics/{task}/correct_synthetic_equivalent/all_same"].append(all_same_5)
                task_group_metrics[f"group_metrics/{task}/correct_synthetic_equivalent/not_all_same"].append(not_all_same_5)
                task_group_metrics[f"group_metrics/{task}/correct_synthetic_equivalent/not_exact_all"].append(not_exact_all_5)

            elif task == "reagent":
                step6_accs = batch.non_tensor_batch[f'{task}/step6/has_reagents'][uid_inds]
                step7_accs = batch.non_tensor_batch[f'{task}/step7/has_correct_reagent_number'][uid_inds]

                step6_exact_k = any(step6_accs)
                step7_exact_k = any(step7_accs)
                step6_exact_all = all(step6_accs)
                step7_exact_all = all(step7_accs)
                step6_all_same = all(a == step6_accs[0] for a in step6_accs)
                step7_all_same = all(a == step7_accs[0] for a in step7_accs)
                step6_not_all_same = not step6_all_same
                step7_not_all_same = not step7_all_same
                step6_not_exact_all = all(not a for a in step6_accs)
                step7_not_exact_all = all(not a for a in step7_accs)

                task_group_metrics[f"group_metrics/{task}/step6/exact_k"].append(step6_exact_k)
                task_group_metrics[f"group_metrics/{task}/step6/exact_all"].append(step6_exact_all)
                task_group_metrics[f"group_metrics/{task}/step6/all_same"].append(step6_all_same)
                task_group_metrics[f"group_metrics/{task}/step6/not_all_same"].append(step6_not_all_same)
                task_group_metrics[f"group_metrics/{task}/step6/not_exact_all"].append(step6_not_exact_all)
                task_group_metrics[f"group_metrics/{task}/step7/exact_k"].append(step7_exact_k)
                task_group_metrics[f"group_metrics/{task}/step7/exact_all"].append(step7_exact_all)
                task_group_metrics[f"group_metrics/{task}/step7/all_same"].append(step7_all_same)
                task_group_metrics[f"group_metrics/{task}/step7/not_all_same"].append(step7_not_all_same)
                task_group_metrics[f"group_metrics/{task}/step7/not_exact_all"].append(step7_not_exact_all)

                step6_reflection_ratio = 0
                for parsed in all_parsed:
                    if "step 6" in parsed.keys():
                        if len(parsed['step 6']['reflections']) > 0:
                            step6_reflection_ratio += 1 / len(all_parsed)
                task_group_metrics[f"group_metrics/{task}/step6/avg_reflection_count"].append(step6_reflection_ratio)
                step7_reflection_ratio = 0
                for parsed in all_parsed:
                    if "step 7" in parsed.keys():
                        if len(parsed['step 7']['reflections']) > 0:
                            step7_reflection_ratio += 1 / len(all_parsed)
                task_group_metrics[f"group_metrics/{task}/step7/avg_reflection_count"].append(step7_reflection_ratio)
            else:
                raise ValueError(f"Unknown task: {task}")

            # print("-" * 100)
            # print(f"scores : {scores.tolist()}")
            # print(f"scores (std): {scores.std().item()}")
            task_group_metrics[f"group_metrics/{task}/reward_mean"].append(scores.mean())
            task_group_metrics[f"group_metrics/{task}/reward_std"].append(scores.std())
            task_group_metrics[f"group_metrics/{task}/reward_all_same"].append(scores_all_same)
            task_group_metrics[f"group_metrics/{task}/reward_not_all_same"].append(scores_not_all_same)
            task_group_metrics[f"group_metrics/{task}/answer/exact_k"].append(answer_exact_k)
            task_group_metrics[f"group_metrics/{task}/answer/exact_all"].append(answer_exact_all)
            task_group_metrics[f"group_metrics/{task}/answer/all_same"].append(answer_all_same)
            task_group_metrics[f"group_metrics/{task}/answer/not_all_same"].append(answer_not_all_same)
            task_group_metrics[f"group_metrics/{task}/answer/not_exact_all"].append(answer_not_exact_all)
            # current_ids = batch.batch['input_ids'][uid_inds]
            if scores.std() == 0:
                metrics[f"group_metrics/{task}/zero_std_num"] += 1
        for k, v in task_group_metrics.items():
            metrics[k] = np.mean(v)

    def _post_batch_processing(self, batch: DataProto):
        # this is experimental and may be changed/removed in the future in favor of a general-purpose one
        if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
            self.train_dataloader.sampler.update(batch=batch)

        # this is experimental and may be changed/removed in the future
        # in favor of a general-purpose data buffer pool
        if hasattr(self.train_dataset, "on_batch_end"):
            # The dataset may be changed after each training batch
            self.train_dataset.on_batch_end(batch=batch)

    def _expand_taskwise_lists(self, reward_extra_infos_dict, tasks, task_names=('forward', 'retro', 'reagent')):
        """
        reward_extra_infos_dict의 task-특화(key가 'task/...') 리스트를
        전체 batch_size 길이로 확장하며, 해당 task 위치에만 값을 채우고
        나머지는 None으로 채운다. 그 외(예: 'score', 'acc', 'pred')는 그대로 둔다.
        """
        # tasks는 np.array나 list 모두 허용
        tasks_list = list(tasks)
        batch_size = len(tasks_list)
        
        # 각 task별로 나타나는 인덱스 수집
        task_to_indices = {t: [] for t in task_names}
        for i, t in enumerate(tasks_list):
            if t in task_to_indices:
                task_to_indices[t].append(i)

        out = defaultdict(list)
        for k, v in reward_extra_infos_dict.items():
            # 키에서 task 추출: 'task/...' 형태면 task는 첫 세그먼트
            head = k.split('/', 1)[0]
            if head in task_to_indices:
                idxs = task_to_indices[head]
                expected_len = len(idxs)

                # v가 리스트/배열이라고 가정. 길이 검증(안 맞으면 최대한 안전하게 잘라/패딩)
                seq = list(v)
                if len(seq) != expected_len:
                    # 길이가 다르면 오류보다 관용적으로 맞춰줌
                    if len(seq) > expected_len:
                        seq = seq[:expected_len]
                    else:
                        seq = seq + [None] * (expected_len - len(seq))

                filled = [None] * batch_size
                for j, pos in enumerate(idxs):
                    filled[pos] = seq[j]
                out[k] = filled
            else:
                # 일반 키: 길이가 이미 batch_size면 그대로, 아니면 필요시 패딩/자르기
                seq = list(v) if hasattr(v, '__len__') and not isinstance(v, (str, bytes)) else [v]
                if len(seq) == batch_size:
                    out[k] = seq
                else:
                    # 보수적으로 batch_size에 맞춰 자르거나 None 패딩
                    if len(seq) > batch_size:
                        out[k] = seq[:batch_size]
                    else:
                        out[k] = seq + [None] * (batch_size - len(seq))
        return out

    def _parse_steps_with_reflections(self, text: str) -> List[Dict]:
        """
        주어진 문자열을 Step 단위로 파싱하고,
        각 Step에 포함된 <REFLECTION> 블록을 추출한다.
        
        반환 형식:
        [
            {
                "step": int,
                "content": str,        # REFLECTION 제외 Step 본문
                "reflections": [str]   # REFLECTION 블록 내용 리스트
            },
            ...
        ]
        """
        # Step 헤더 매칭
        step_pattern = re.compile(r"(## Step (\d+))")
        matches = list(step_pattern.finditer(text))
        
        steps_data = {}
        
        for i, match in enumerate(matches):
            step_header = match.group(1)
            step_num = int(match.group(2))
            
            # Step 구간의 끝 위치 계산
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            step_body = text[start_pos:end_pos].strip()
            
            # REFLECTION 블록 추출
            reflection_pattern = re.compile(r"<REFLECTION>(.*?)</REFLECTION>", re.DOTALL)
            reflections = reflection_pattern.findall(step_body)
            
            # REFLECTION 블록 제거 후 순수 Step 본문
            cleaned_body = reflection_pattern.sub("", step_body).strip()
            
            steps_data[f'step {step_num}'] = {
                "step": step_num,
                "content": cleaned_body,
                "reflections": [r.strip() for r in reflections]
            }
        
        return steps_data