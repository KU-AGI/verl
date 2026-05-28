import asyncio
import logging
import os

import ray

from verl.single_controller.ray import RayClassWithInitArgs
from verl.workers.config import RewardModelConfig
from verl.workers.rollout.replica import RolloutReplica
from recipe.image_rl.config import ImageGenerationHFModelConfig, ImageGenerationRolloutConfig
from recipe.fully_async_policy_image_rl.sglang_rollout.janus_sglang_server import JanusSGLangAsyncServer


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class JanusSGLangReplica(RolloutReplica):
    """Replica wrapper for the Janus-Pro SGLang rollout path.

    The fully async image pipeline already creates one Ray rollout worker per
    rollout GPU. For this SGLang path those workers are GPU holders only; the
    actual model is launched by JanusSGLangAsyncServer on the same node/GPU.
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
        self.server_class = JanusSGLangAsyncServer

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        return None

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        os.environ["CUDA_VISIBLE_DEVICES"],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_node_ids = [info[0] for info in worker_infos]
        worker_cuda_visible_devices = [info[1] for info in worker_infos]

        for node_rank in range(self.nnodes):
            workers = self.workers[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            )
            node_id = worker_node_ids[node_rank * self.gpus_per_node]
            name = (
                f"janus_sglang_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"janus_sglang_server_reward_{self.replica_rank}_{node_rank}"
            )
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={
                    "env_vars": {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "PYTHONPATH": f"/verl/sglang/python:{os.environ.get('PYTHONPATH', '')}",
                        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
                    }
                },
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = f"{server_address}:{server_port}"

    async def cancel(self):
        if self.servers:
            await asyncio.gather(*[server.cancel.remote() for server in self.servers])

    async def resume(self):
        if self.servers:
            await asyncio.gather(*[server.resume.remote() for server in self.servers])

    async def sleep(self):
        if self.servers:
            await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def wake_up(self):
        if self.servers:
            await asyncio.gather(*[server.wake_up.remote() for server in self.servers])

    async def reset_prefix_cache(self):
        if self.servers:
            await asyncio.gather(*[server.reset_prefix_cache.remote() for server in self.servers])
