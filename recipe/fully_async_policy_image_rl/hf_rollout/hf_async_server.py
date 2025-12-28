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
from typing import Any, Optional
import concurrent.futures

import ray
import torch
from ray.actor import ActorHandle

from verl.workers.config import HFModelConfig, RewardModelConfig, RolloutConfig
from recipe.image_rl.config import ImageGenerationHFModelConfig, ImageGenerationRolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@ray.remote(num_cpus=1)
class HuggingFaceAsyncServerForImageRollout:
    """
    HuggingFace async server designed to work with ImageUnifiedRollout.
    This server provides async/cancellation infrastructure around ImageUnifiedRollout.
    It handles DataProto inputs/outputs instead of token IDs like vLLM does.
    """

    def __init__(
        self,
        config: ImageGenerationRolloutConfig | RewardModelConfig,
        model_config: ImageGenerationHFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        self.config = config
        self.model_config = model_config
        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes

        # For cancellation support
        self.paused = False
        self.lock = asyncio.Lock()
        self.cancel_event: dict[str, asyncio.Event] = {}
        self.generation_tasks: dict[str, asyncio.Task] = {}
        self.generation_results: dict[str, Optional[DataProto]] = {}

        # Thread pool for running synchronous generate_sequences in background
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # Weight update queue (maxsize=1 to keep only latest version)
        self.pending_weight_version: Optional[int] = None
        self.weight_update_lock = asyncio.Lock()
        self.weight_update_worker_task: Optional[asyncio.Task] = None
        self.ongoing_generations = 0  # Track number of ongoing generations

        logger.info(f"Initializing HF async server on replica {self.replica_rank}, node {self.node_rank}")

    async def init_model(self):
        """Initialize the server - workers are already initialized"""
        # Worker info logging
        for i, worker in enumerate(self.workers):
            try:
                info = await worker.__ray_call__.remote(
                    lambda self: {
                        'class': type(self).__name__,
                        'has_rollout': hasattr(self, 'rollout'),
                        'rollout_type': type(self.rollout).__name__ if hasattr(self, 'rollout') else None,
                        'methods': [m for m in dir(self) if not m.startswith('_') and callable(getattr(self, m, None))]
                    }
                )
                logger.info(f"Worker {i} info: {info}")
            except Exception as e:
                logger.error(f"Failed to get worker {i} info: {e}")

        # Start background weight update worker
        self.weight_update_worker_task = asyncio.create_task(self._weight_update_worker())

        logger.info(f"HF async server initialized on replica {self.replica_rank}")

    async def _generate_step(
        self,
        prompt_data: DataProto,
        sampling_params: dict[str, Any],
        request_id: str,
    ) -> DataProto:
        """
        Call ImageUnifiedRollout.generate_sequences() on the worker.
        Since generate_sequences is a synchronous method, we run it in an executor.
        """
        try:
            # Track ongoing generation
            async with self.weight_update_lock:
                self.ongoing_generations += 1

            # Update DataProto meta_info with sampling params
            if hasattr(prompt_data, 'meta_info'):
                for key, value in sampling_params.items():
                    prompt_data.meta_info[key] = value

            # Call generate_sequences on ALL workers simultaneously
            # This is critical for NCCL collective operations (all-reduce, etc.)
            # All workers must participate in the same collective op at the same time
            result_refs = [worker.generate_sequences.remote(prompt_data) for worker in self.workers]

            # Wait for all workers to complete
            # Convert Ray ObjectRefs to asyncio-compatible futures and gather
            results = await asyncio.gather(
                *[asyncio.wrap_future(ref.future()) for ref in result_refs]
            )

            # Return result from worker[0] (rank 0 in this replica)
            # Other workers' results are identical due to synchronized execution
            return results[0]

        except Exception as e:
            logger.error(f"Generation failed for request {request_id}: {e}", exc_info=True)
            raise
        finally:
            # Decrement ongoing generation count
            async with self.weight_update_lock:
                self.ongoing_generations -= 1

    async def generate_for_partial(
        self,
        prompt_data: DataProto,
        sampling_params: dict[str, Any],
        request_id: str,
    ) -> tuple[Optional[DataProto], bool]:
        """
        Generate with cancellation support for ImageUnifiedRollout.
        Returns (result_dataproto, is_cancelled)
        """

        async with self.lock:
            if self.paused:
                # After cancel, all tasks will return directly
                logger.debug(f"Request {request_id} skipped - server is paused")
                return None, True

            self.cancel_event[request_id] = asyncio.Event()
            self.generation_results[request_id] = None

            cancel_handle = asyncio.create_task(self.cancel_event[request_id].wait())
            generation_handle = asyncio.create_task(
                self._generate_step(prompt_data, sampling_params, request_id)
            )
            self.generation_tasks[request_id] = generation_handle

        # Wait for either generation to complete or cancellation
        done, pend = await asyncio.wait(
            [generation_handle, cancel_handle],
            return_when=asyncio.FIRST_COMPLETED
        )

        result = None
        is_cancel = False

        # Process completed tasks
        for task in done:
            if task == generation_handle:
                try:
                    result = await task
                    logger.debug(f"Generation completed for request {request_id}")
                except Exception as e:
                    logger.error(f"Generation task failed for {request_id}: {e}")
                    result = None
            elif task == cancel_handle:
                is_cancel = True
                logger.info(f"Request {request_id} was cancelled")

        # Cancel pending tasks
        for task in pend:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Clean up
        async with self.lock:
            self.cancel_event.pop(request_id, None)
            self.generation_tasks.pop(request_id, None)
            self.generation_results.pop(request_id, None)

            # If generation was cancelled (didn't complete)
            if generation_handle not in done:
                is_cancel = True

        return result, is_cancel

    async def generate(
        self,
        prompt_data: DataProto,
        sampling_params: dict[str, Any],
        request_id: str,
    ) -> DataProto:
        """Standard generation interface (non-partial)"""
        result, is_cancel = await self.generate_for_partial(
            prompt_data, sampling_params, request_id
        )
        if is_cancel or result is None:
            raise RuntimeError(f"Generation cancelled or failed for request {request_id}")
        return result

    async def cancel(self):
        """Cancel all ongoing generations"""
        async with self.lock:
            self.paused = True
            # Signal all cancel events
            for request_id in list(self.cancel_event.keys()):
                self.cancel_event[request_id].set()

            # Cancel all generation tasks
            for request_id, task in list(self.generation_tasks.items()):
                if not task.done():
                    task.cancel()

        logger.info(f"Cancelled all generations on replica {self.replica_rank}, node {self.node_rank}")

    async def resume(self):
        """Resume generation after cancellation"""
        async with self.lock:
            self.paused = False
        logger.info(f"Resumed generation on replica {self.replica_rank}, node {self.node_rank}")

    async def wake_up(self):
        """Wake up the server"""
        logger.info(f"Woke up server on replica {self.replica_rank}, node {self.node_rank}")

    async def sleep(self):
        """Sleep the server"""
        logger.info(f"Put server to sleep on replica {self.replica_rank}, node {self.node_rank}")

    async def reset_prefix_cache(self):
        """Reset any prefix cache and clear CUDA memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Reset cache on replica {self.replica_rank}, node {self.node_rank}")

    def get_server_address(self):
        """Get server address (for compatibility)"""
        return ray.util.get_node_ip_address().strip("[]"), 8000

    def get_master_address(self):
        """Get master address (for compatibility)"""
        return ray.util.get_node_ip_address().strip("[]"), 8001

    async def get_weight_update_status(self) -> dict:
        """Get current weight update status for monitoring"""
        async with self.weight_update_lock:
            return {
                "replica_rank": self.replica_rank,
                "node_rank": self.node_rank,
                "pending_weight_version": self.pending_weight_version,
                "ongoing_generations": self.ongoing_generations,
                "has_pending_update": self.pending_weight_version is not None,
            }

    async def queue_weight_update(self, version: int):
        """
        Queue a weight update to be applied after ongoing generations complete.
        Only keeps the latest version - older pending updates are discarded.
        Special value: version=-1 clears the pending update.
        """
        async with self.weight_update_lock:
            if version == -1:
                # Clear pending update
                self.pending_weight_version = None
                logger.info(f"[Replica {self.replica_rank}] Cleared pending weight update")
            elif self.pending_weight_version is None or version > self.pending_weight_version:
                self.pending_weight_version = version
                logger.info(
                    f"[Replica {self.replica_rank}] Queued weight update to version {version}, "
                    f"ongoing_generations={self.ongoing_generations}"
                )
            else:
                logger.debug(
                    f"[Replica {self.replica_rank}] Skipped weight update to version {version}, "
                    f"already have pending version {self.pending_weight_version}"
                )

    async def _weight_update_worker(self):
        """
        Background worker that monitors for weight updates when no generation is ongoing.
        Instead of applying weights directly, this just monitors the idle state.
        The actual weight sync is triggered by FullyAsyncRollouter when it detects idle replicas.
        """
        logger.info(f"[Replica {self.replica_rank}] Weight update worker started (monitor mode)")

        while True:
            await asyncio.sleep(0.1)  # Monitor every 100ms

            # Just log status for monitoring - actual sync happens at rollouter level
            async with self.weight_update_lock:
                if self.pending_weight_version is not None and self.ongoing_generations == 0:
                    logger.debug(
                        f"[Replica {self.replica_rank}] Ready for weight update: "
                        f"pending_version={self.pending_weight_version}, ongoing={self.ongoing_generations}"
                    )