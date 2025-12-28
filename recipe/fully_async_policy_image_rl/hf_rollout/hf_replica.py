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
import logging

import ray
from ray.actor import ActorHandle

from verl.single_controller.ray import RayClassWithInitArgs
from verl.workers.config import RewardModelConfig
from recipe.image_rl.config import ImageGenerationHFModelConfig, ImageGenerationRolloutConfig
from verl.workers.rollout.replica import RolloutReplica
from recipe.fully_async_policy_image_rl.hf_rollout.hf_async_server import HuggingFaceAsyncServerForImageRollout

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class HuggingFaceReplica(RolloutReplica):
    """
    HuggingFace replica that works with ImageUnifiedRollout for fully async image generation.
    Provides async infrastructure around ImageUnifiedRollout for cancellation and partial rollout support.
    """

    def __init__(
        self,
        replica_rank: int,
        config: ImageGenerationRolloutConfig | RewardModelConfig,
        model_config: ImageGenerationHFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = HuggingFaceAsyncServerForImageRollout

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """
        For HuggingFace rollout with ImageUnifiedRollout, we don't need to create workers here.
        The workers are already created by the worker group in the main training script.
        This replica just wraps existing workers with async servers.

        This method returns None because worker creation is handled elsewhere.
        """
        # For HuggingFace/ImageUnifiedRollout, workers are created by the worker group
        # (e.g., DetachAsyncRolloutWorker in fsdp_workers.py)
        # We don't create them here - we just wrap them with async servers
        return None

    async def launch_servers(self):
        """
        Launch HuggingFace async servers that wrap existing ImageUnifiedRollout workers.
        The workers are already initialized by the worker group - we just wrap them with async servers.
        """
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # Get node_id of all workers
        worker_node_ids = await asyncio.gather(
            *[
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.workers
            ]
        )

        # For non-data parallel case, there's only one server whether it's single or multi nodes
        nnodes, gpus_per_node = self.nnodes, self.gpus_per_node
        if getattr(self.config, 'data_parallel_size', 1) == 1:
            nnodes = 1
            gpus_per_node = self.world_size

        # Create async server actor in each node with node affinity
        # Each server wraps a group of workers from the same node
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_node : (node_rank + 1) * gpus_per_node]
            node_id = worker_node_ids[node_rank * gpus_per_node]
            name = (
                f"hf_image_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"hf_image_server_reward_{self.replica_rank}_{node_rank}"
            )

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_node,
                nnodes=nnodes,
            )
            self.servers.append(server)

        # Initialize servers
        await asyncio.gather(*[server.init_model.remote() for server in self.servers])

        # Get server address from first server (for compatibility)
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = f"{server_address}:{server_port}"

    async def cancel(self):
        """Cancel all ongoing generations in each rollout server."""
        if self.servers:
            await asyncio.gather(*[server.cancel.remote() for server in self.servers])
            logger.info(f"Cancelled all servers in replica {self.replica_rank}")

    async def resume(self):
        """Resume generation in each rollout server."""
        if self.servers:
            await asyncio.gather(*[server.resume.remote() for server in self.servers])
            logger.info(f"Resumed all servers in replica {self.replica_rank}")

    async def sleep(self):
        """Put each rollout server to sleep."""
        if self.servers:
            await asyncio.gather(*[server.sleep.remote() for server in self.servers])
            logger.info(f"Put all servers to sleep in replica {self.replica_rank}")

    async def wake_up(self):
        """Wake up each rollout server."""
        if self.servers:
            await asyncio.gather(*[server.wake_up.remote() for server in self.servers])
            logger.info(f"Woke up all servers in replica {self.replica_rank}")

    async def reset_prefix_cache(self):
        """Reset cache in each rollout server."""
        if self.servers:
            await asyncio.gather(*[server.reset_prefix_cache.remote() for server in self.servers])
            logger.info(f"Reset cache in all servers in replica {self.replica_rank}")


# For backward compatibility with vLLM code
FullyAsyncvLLMReplica = HuggingFaceReplica

