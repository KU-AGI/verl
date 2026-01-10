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
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from recipe.fully_async_policy_image_rl.fully_async_rollouter import FullyAsyncRollouter
from recipe.fully_async_policy_image_rl.fully_async_trainer import FullyAsyncTrainer
from recipe.fully_async_policy_image_rl.message_queue import MessageQueue, MessageQueueClient
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role
from verl.utils.fs import copy_to_local


def create_resource_pool_manager(config, roles: list) -> ResourcePoolManager:
    """
    Create resource pool manager

    Args:
        config: Configuration object
        roles: List of roles that need to create resource pools

    Returns:
        ResourcePoolManager: Resource pool manager
    """
    resource_pool_spec = {}
    mapping = {}

    # Check if we need separate reward model pool for vLLM
    use_vllm_reward = (
        config.reward_model.enable 
        and config.reward_model.strategy == "vllm"
        and Role.RewardModel in roles
    )

    # Actor/Critic resource pool (excluding RewardModel if using vLLM)
    trainer_roles = [Role.Actor, Role.Critic, Role.RefPolicy]
    if not use_vllm_reward:
        trainer_roles.append(Role.RewardModel)
    
    if any(role in roles for role in trainer_roles):
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"

        trainer_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        resource_pool_spec["trainer_pool"] = trainer_pool

        # Map training-related roles to the same resource pool
        for role in trainer_roles:
            if role in roles:
                mapping[role] = "trainer_pool"

    # Separate reward model pool for vLLM with TP > 1
    if use_vllm_reward:
        # Get tensor parallel size from config
        tp_size = config.reward_model.model.get("tensor_parallel_size", 4)
        
        # Create a pool where a SINGLE worker has access to ALL TP GPUs
        # Key insight: we use [tp_size] meaning 1 node with tp_size GPUs
        # and max_colocate_count=1 in the worker group
        reward_pool = [tp_size]  # Single "node" with tp_size GPUs
        resource_pool_spec["reward_pool"] = reward_pool
        mapping[Role.RewardModel] = "reward_pool"
        
        print(f"[ResourcePool] Created separate reward_pool for vLLM with TP={tp_size}")

    # Rollout resource pool
    if Role.Rollout in roles:
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"

        rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes
        resource_pool_spec["rollout_pool"] = rollout_pool
        mapping[Role.Rollout] = "rollout_pool"

    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


def create_role_worker_mapping(config):
    """
    Create mapping from roles to worker classes

    Args:
        config: Configuration object

    Returns:
        dict: Mapping from roles to worker classes
    """
    # Select worker class based on strategy
    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        # assert config.actor_rollout_ref.actor.strategy == config.critic.strategy # we don't need critic in async image rl
        from recipe.fully_async_policy_image_rl.fsdp_workers import (
            DetachActorWorker,
            DetachAsyncRolloutWorker,
        )
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.critic.strategy == "megatron"
        from recipe.fully_async_policy_image_rl.megatron_worker import DetachActorWorker, DetachAsyncRolloutWorker
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")

    role_worker_mapping = {
        Role.Actor: ray.remote(DetachActorWorker),
        Role.Rollout: ray.remote(DetachAsyncRolloutWorker),
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == "vllm":
            # Use the vLLM worker that supports TP > 1
            from recipe.image_rl.image_generation_reward_worker_vllm import ImageGenerationRewardModelWorker

        elif config.reward_model.strategy in ["fsdp", "fsdp2"]:
            from recipe.image_rl.image_generation_worker import ImageGenerationRewardModelWorker

        # TODO megatron support
        else:
            raise NotImplementedError(f"Unsupported reward model strategy: {config.reward_model.strategy}")

        role_worker_mapping[Role.RewardModel] = ray.remote(ImageGenerationRewardModelWorker)

    # Add reference policy (if KL loss or reward is required)
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)

    return role_worker_mapping, ray_worker_group_cls


def create_vllm_reward_worker_group(config, ray_worker_group_cls):
    """
    Create a special worker group for vLLM reward model with full GPU visibility.
    
    This creates a SINGLE worker that can see all tensor_parallel_size GPUs,
    rather than N workers each seeing 1 GPU.
    """
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool
    
    tp_size = config.reward_model.model.get("tensor_parallel_size", 4)
    
    print(f"[vLLM RewardModel] Creating worker with TP={tp_size}")
    
    # Import the worker class
    from recipe.image_rl.image_generation_reward_worker_vllm import ImageGenerationRewardModelWorker
    
    # Create resource pool with ALL TP GPUs for a single worker
    # Use process_on_nodes to specify which node(s) to use
    resource_pool = RayResourcePool(
        process_on_nodes=[0],  # Use first node
        num_gpus_per_node=tp_size,
        max_colocate_count=1,  # Single worker gets all GPUs
        name="reward_model_vllm_pool",
    )
    
    # Create class wrapper with config
    reward_cls = RayClassWithInitArgs(
        cls=ImageGenerationRewardModelWorker,
        config=config.reward_model,
    )
    
    # Create worker group
    worker_group = ray_worker_group_cls(
        resource_pool=resource_pool,
        ray_cls_with_init=reward_cls,
        name_prefix="reward_model",
    )
    
    return worker_group


@ray.remote(num_cpus=1)
class FullyAsyncTaskRunner:
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[ASYNC MAIN] Starting fully async PPO training...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        print(f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[ASYNC MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        print("[ASYNC MAIN] Creating FullyAsyncRollouter...")
        self._create_rollouter(config)

        print("[ASYNC MAIN] Creating FullyAsyncTrainer...")
        self._create_trainer(config)

        # sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        # max_queue_size
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(self.components["message_queue_client"]))
        ray.get(self.components["trainer"].set_message_queue_client.remote(self.components["message_queue_client"]))

        print("[ASYNC MAIN] Setting up parameter synchronization...")
        from recipe.fully_async_policy_image_rl.param_sync import ParameterSynchronizer

        param_synchronizer = ParameterSynchronizer.remote(
            config=config,
            trainer=self.components["trainer"],
            rollouter=self.components["rollouter"],
            mq=self.components["message_queue_client"],
        )
        # Set the handle so it can pass itself to rollouter
        ray.get(param_synchronizer.set_self_handle.remote(param_synchronizer))
        ray.get(self.components["trainer"].set_parameter_synchronizer.remote(param_synchronizer))

        # load checkpoint and sync parameter before doing anything
        val_before_train = config.trainer.get("val_before_train", True)
        # param_version resume from ckpt or default 0
        param_version = ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())
        ray.get(param_synchronizer.sync_weights.remote(version=param_version, validate=val_before_train))
        ray.get(param_synchronizer.wait_last_valid.remote())

        self.components["param_synchronizer"] = param_synchronizer
        print("[ASYNC MAIN] All components initialized successfully")

    def _create_rollouter(self, config) -> None:
        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping={Role.Rollout: self.components["role_worker_mapping"][Role.Rollout]},
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            reward_fn=None,
            val_reward_fn=None,
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")

    def _run_training_loop(self):
        self.running = True

        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                # Use ray.wait to monitor all futures and return when any one is completed.
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[ASYNC MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            self.components["message_queue_client"].clear_queue()
            print("[ASYNC MAIN] Training completed or interrupted")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")
    from time import time

    start_time = time()
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
