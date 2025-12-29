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
import os
from typing import Any, Optional
from tensordict import TensorDict

import torch
import hydra
import numpy as np
import ray
from omegaconf import DictConfig

from recipe.fully_async_policy_image_rl.hf_rollout.hf_replica import HuggingFaceReplica
from recipe.fully_async_policy_image_rl.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorkerBase,
    AsyncLLMServerManager,
    _agent_loop_registry,
    _DummyConfig,
    get_trajectory_info,
)

from verl.protocol import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.rollout_trace import rollout_trace_attr
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FullyAsyncLLMServerManager(AsyncLLMServerManager):
    """Extended server manager with multi-GPU distribution support."""
    
    def __init__(self, config: DictConfig, server_handles: list):
        super().__init__(config, list(server_handles))
        self._server_index = 0
        self.server_handles_by_index = list(server_handles)
        self._lock = asyncio.Lock()
    
    def get_num_servers(self) -> int:
        """Get the number of available servers."""
        return len(self.server_handles)
    
    async def _get_next_server_index(self) -> int:
        """Thread-safe round-robin server selection."""
        async with self._lock:
            idx = self._server_index
            self._server_index = (self._server_index + 1) % len(self.server_handles)
            return idx
    
    async def generate_for_partial(self, request_id, prompt_data, sampling_params, **kwargs_extra):
        """
        Generate from DataProto with partial rollout function.
        For ImageUnifiedRollout, we pass DataProto instead of just prompt_ids.
        Returns (result_dataproto, is_cancelled)
        """
        server = self._choose_server(request_id)
        result, is_cancel = await server.generate_for_partial.remote(
            request_id=request_id,
            prompt_data=prompt_data,
            sampling_params=sampling_params,
        )
        return result, is_cancel

    async def generate_for_partial_on_server(
        self, 
        server_index: int, 
        request_id: str, 
        prompt_data: DataProto, 
        sampling_params: dict
    ):
        """
        Generate on a specific server by index.
        Used for explicit multi-GPU distribution.
        """
        server = self.server_handles_by_index[server_index]
        result, is_cancel = await server.generate_for_partial.remote(
            request_id=request_id,
            prompt_data=prompt_data,
            sampling_params=sampling_params,
        )
        return result, is_cancel


class FullyAsyncAgentLoopOutput(AgentLoopOutput):
    """Agent loop output."""

    is_cancel: bool = False
    """Indicates whether the request was interrupted"""
    log_probs: list[float] = None
    """Response token log probs including LLM generated token, tool response token."""
    param_version_start: int = 0
    """Indicate start parameter version when this response is generated"""
    param_version_end: int = 0
    """Indicate end parameter version when this response is generated, used for partial rollout"""
    generation_data: Any = None
    """Generated data (DataProto) from ImageUnifiedRollout for image generation tasks"""


@ray.remote
class FullyAsyncAgentLoopWorker(AgentLoopWorkerBase):
    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        # Ensure agent loop classes are imported and registered
        from recipe.fully_async_policy_image_rl.agent_loop import PartialSingleTurnAgentLoop
        _ = PartialSingleTurnAgentLoop

        self.server_manager = FullyAsyncLLMServerManager(config, server_handles)
        super().__init__(config, server_handles, reward_router_address)

    async def generate_sequences_no_post(
        self, batch: DataProto, partial_output_list: Optional[list[AgentLoopOutput]]
    ) -> list[AgentLoopOutput]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[FullyAsyncAgentLoopOutput]: List of agent loop outputs, one per sample in the batch.
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        )

        if not partial_output_list:
            partial_output_list = [None] * len(batch)

        batch_size = len(batch)
        num_servers = self.server_manager.get_num_servers()

        # Check conditions for batch processing
        agent_names = batch.non_tensor_batch.get("agent_name", [])
        all_same_agent = len(set(agent_names)) == 1
        no_partial_outputs = all([out is None for out in partial_output_list])

        if all_same_agent and no_partial_outputs:
            # Multi-GPU batch processing path
            agent_name = agent_names[0]
            result = await self._partial_run_agent_loop_batch_distributed(
                sampling_params, trajectory_info, agent_name, batch, partial_output_list
            )
            return result
        else:
            # Per-sample processing path
            tasks = []
            for i in range(batch_size):
                kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
                kwargs["output"] = partial_output_list[i]
                tasks.append(
                    asyncio.create_task(self._partial_run_agent_loop(sampling_params, trajectory_info[i], **kwargs))
                )
            result = await asyncio.gather(*tasks)
            return result

    async def _partial_run_agent_loop_batch_distributed(
        self,
        sampling_params: dict[str, Any],
        trajectory_list: list[dict[str, Any]],
        agent_name: str,
        batch: DataProto,
        partial_output_list: list[AgentLoopOutput],
    ) -> list[AgentLoopOutput]:
        """
        Process batch distributed across multiple GPUs/servers.
        """
        from uuid import uuid4

        batch_size = len(batch)
        num_servers = self.server_manager.get_num_servers()
        
        # Calculate how to split the batch across servers
        samples_per_server = batch_size // num_servers
        remainder = batch_size % num_servers
        
        # Split batch into chunks
        chunks = []
        start_idx = 0
        
        for server_idx in range(num_servers):
            chunk_size = samples_per_server + (1 if server_idx < remainder else 0)

            if chunk_size == 0:
                continue

            end_idx = start_idx + chunk_size
            chunk_batch = batch[start_idx:end_idx]
            chunks.append((server_idx, chunk_batch, start_idx, end_idx))
            start_idx = end_idx
        
        # Process chunks in parallel
        async def process_chunk(server_idx: int, chunk_batch: DataProto, chunk_start: int, chunk_end: int):
            try:
                # request_id 생성도 try 블록 안으로 이동하여 안전하게 처리
                request_id = f"{uuid4().hex}_server{server_idx}"
                result_data, is_cancel = await self.server_manager.generate_for_partial_on_server(
                    server_index=server_idx,
                    request_id=request_id,
                    prompt_data=chunk_batch,
                    sampling_params=sampling_params,
                )
                return (server_idx, chunk_start, chunk_end, result_data, is_cancel, None)
            except BaseException as e: # Exception -> BaseException으로 변경하여 모든 에러(Cancelled 등) 포착
                logger.error(f"[Batch Distributed] Server {server_idx} failed inside process_chunk: {e}")
                return (server_idx, chunk_start, chunk_end, None, True, e)
        
        tasks = [
            process_chunk(server_idx, chunk_batch, chunk_start, chunk_end)
            for server_idx, chunk_batch, chunk_start, chunk_end in chunks
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Reconstruct outputs in original order
        outputs = [None] * batch_size
        param_version = batch.non_tensor_batch.get("param_version", [0] * batch_size)

        # 청크 정보를 찾기 위한 헬퍼 맵 (혹시 Exception으로 chunk 정보를 잃어버릴 경우 대비)
        # 하지만 gather 순서는 tasks 순서와 동일하므로 순서대로 매핑 가능
        
        for i, result in enumerate(results):
            # 원래 할당된 청크 정보 가져오기 (Exception 발생 시 복구용)
            server_idx_ref, _, chunk_start_ref, chunk_end_ref = chunks[i]
            chunk_size_ref = chunk_end_ref - chunk_start_ref

            if isinstance(result, BaseException):
                logger.error(f"[Batch Distributed] Task failed completely: {result}")
                # [FIX 1] 예외 발생 시에도 None 대신 Cancelled 객체로 채움
                for k in range(chunk_size_ref):
                    outputs[chunk_start_ref + k] = FullyAsyncAgentLoopOutput(
                        prompt_ids=[], response_ids=[], response_mask=[], num_turns=1,
                        metrics={}, is_cancel=True, log_probs=[],
                        param_version_start=0, param_version_end=0,
                    )
                continue

            server_idx, chunk_start, chunk_end, result_data, is_cancel, error = result
            chunk_size = chunk_end - chunk_start

            # Handle cancellation or error
            if is_cancel or result_data is None or error is not None:
                if error:
                    logger.error(f"[Batch Distributed] Error returned from worker: {error}")
                for k in range(chunk_size):
                    original_idx = chunk_start + k
                    outputs[original_idx] = FullyAsyncAgentLoopOutput(
                        prompt_ids=[], 
                        response_ids=[], 
                        response_mask=[], 
                        num_turns=1,
                        metrics={}, 
                        is_cancel=True, 
                        log_probs=[],
                        param_version_start=0, 
                        param_version_end=0,
                    )
                continue

            # Validate result type
            if not isinstance(result_data, DataProto):
                logger.error(f"[Batch Distributed] Expected DataProto from server {server_idx}, got {type(result_data)}")
                for k in range(chunk_size):
                    original_idx = chunk_start + k
                    outputs[original_idx] = FullyAsyncAgentLoopOutput(
                        prompt_ids=[], response_ids=[], response_mask=[], num_turns=1,
                        metrics={}, is_cancel=True, log_probs=[], 
                        param_version_start=0, param_version_end=0,
                    )
                continue

            # Validate batch size
            if len(result_data) != chunk_size:
                logger.error(f"[Batch Distributed] Size mismatch from server {server_idx}: {len(result_data)} vs {chunk_size}")
                for k in range(chunk_size):
                    original_idx = chunk_start + k
                    outputs[original_idx] = FullyAsyncAgentLoopOutput(
                        prompt_ids=[], response_ids=[], response_mask=[], num_turns=1,
                        metrics={}, is_cancel=True, log_probs=[],
                        param_version_start=0, param_version_end=0,
                    )
                continue

            # Process successful results
            for local_i in range(chunk_size):
                original_idx = chunk_start + local_i
                
                # Extract data from result_data for this sample
                sample_data = result_data[local_i]
                
                outputs[original_idx] = FullyAsyncAgentLoopOutput(
                    prompt_ids=[],
                    response_ids=[],
                    response_mask=[],
                    num_turns=1,
                    metrics={},
                    is_cancel=False,
                    log_probs=[],
                    param_version_start=param_version[original_idx],
                    param_version_end=param_version[original_idx],
                    generation_data=sample_data,
                )

        # Final Safety Check [FIX 2]
        for i, out in enumerate(outputs):
            if out is None:
                logger.error(f"[Batch Distributed] Missing output at index {i} - Filling with dummy cancel")
                outputs[i] = FullyAsyncAgentLoopOutput(
                    prompt_ids=[], response_ids=[], response_mask=[], num_turns=1,
                    metrics={}, is_cancel=True, log_probs=[],
                    param_version_start=0, param_version_end=0,
                )

        return outputs

    async def generate_sequences(self, prompts: DataProto, on_task_complete=None) -> DataProto:
        """Generate sequences and convert agent loop outputs to DataProto."""
        from recipe.fully_async_policy_image_rl.detach_utils import postprocess_agent_loop_outputs
        # Preserve task_id through processing
        task_id = prompts.batch.get("task_id", None)

        # Generate sequences using multi-GPU batch processing
        outputs_list = await self.generate_sequences_no_post(prompts, partial_output_list=None)

        # Convert agent loop outputs to DataProto
        result_proto = postprocess_agent_loop_outputs(
            rs_or_list=outputs_list,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor
        )

        # Re-attach task_id to result so manager knows which task was completed
        if task_id is not None:
            batch_size = len(result_proto)
            current_task_id = task_id.view(-1)[0].item()
            result_proto.batch["task_id"] = torch.full(
                (batch_size,), current_task_id,
                dtype=torch.int32, device=result_proto.batch.device
            )

        return result_proto

    async def generate_sequences_on_server(self, prompts: DataProto, server_index: int) -> DataProto:
        from recipe.fully_async_policy_image_rl.detach_utils import postprocess_agent_loop_outputs
        # sampling_params 구성은 기존 generate_sequences_no_post와 동일하게
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        if prompts.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # server_index로 “한 번만” 호출 (batch 전체)
        from uuid import uuid4
        request_id = f"{uuid4().hex}_server{server_index}"
        result_data, is_cancel = await self.server_manager.generate_for_partial_on_server(
            server_index=server_index,
            request_id=request_id,
            prompt_data=prompts,
            sampling_params=sampling_params,
        )

        # 결과를 FullyAsyncAgentLoopOutput 리스트로 포장 (기존 batch_distributed 성공 경로와 동일)
        batch_size = len(prompts)
        param_version = prompts.non_tensor_batch.get("param_version", [0] * batch_size)

        outputs_list = []
        if is_cancel or result_data is None:
            for _ in range(batch_size):
                outputs_list.append(FullyAsyncAgentLoopOutput(
                    prompt_ids=[], response_ids=[], response_mask=[], num_turns=1,
                    metrics={}, is_cancel=True, log_probs=[],
                    param_version_start=0, param_version_end=0,
                ))
        else:
            for i in range(batch_size):
                outputs_list.append(FullyAsyncAgentLoopOutput(
                    prompt_ids=[], response_ids=[], response_mask=[], num_turns=1,
                    metrics={}, is_cancel=False, log_probs=[],
                    param_version_start=param_version[i], param_version_end=param_version[i],
                    generation_data=result_data[i],
                ))

        # DataProto로 후처리 + task_id 복구 (기존 generate_sequences와 동일)
        result_proto = postprocess_agent_loop_outputs(
            rs_or_list=outputs_list,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor,
        )

        task_id = prompts.batch.get("task_id", None)
        if task_id is not None:
            current_task_id = task_id.view(-1)[0].item()
            result_proto.batch["task_id"] = torch.full(
                (len(result_proto),), current_task_id,
                dtype=torch.int32, device=result_proto.batch.device,
            )

        return result_proto


class FullyAsyncAgentLoopManager(AgentLoopManager):
    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        self.agent_loop_workers_class = FullyAsyncAgentLoopWorker
        self.rollout_replica_class = HuggingFaceReplica

        self.rm_wg = rm_wg
        self.rollout_replicas = None
        self.server_handles = None
        self.server_addresses = None
        self.agent_loop_workers = None

    @classmethod
    async def create(cls, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        instance = cls(config, worker_group, rm_wg)
        await instance._async_init()
        return instance

    async def _async_init(self):
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward import RewardModelManager

            self.reward_model_manager = RewardModelManager(self.config.reward_model, self.rm_wg)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        await self._initialize_llm_servers_async()
        self._init_agent_loop_workers()
        print("[DEBUG] agent_loop_workers =", len(self.agent_loop_workers), flush=True)

    async def _initialize_llm_servers_async(self):
        rollout_world_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group:
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    async def generate_single_sample_async(
        self,
        sample: DataProto,
        partial_output_list: Optional[list[AgentLoopOutput]],
    ) -> list[AgentLoopOutput]:
        """Asynchronously process a single sample"""
        worker = self._select_best_worker()

        batch_size = len(sample)

        # Ray ObjectRef 를 그냥 await 하는 정석 패턴
        output_ref = worker.generate_sequences_no_post.remote(sample, partial_output_list)
        result: list[AgentLoopOutput] = await output_ref

        print(f"[AgentLoop] received result batch_size={len(result)}")

        return result

    async def generate_sequences_with_callback(self, prompts: DataProto, on_task_complete=None) -> DataProto:
        """Generate sequences with task-level callbacks for async reward computation.

        This method overrides the parent class to use the async HF replica approach.
        It executes tasks sequentially (Task 1 -> Task 2 -> Task 3) and calls
        on_task_complete callback after each task finishes.

        Args:
            prompts (DataProto): Input batch.
            on_task_complete: Optional async callback(task_id, batch_result) called after each task.

        Returns:
            DataProto: Output batch with all tasks completed.
        """
        logger.info("[FullyAsyncAgentLoopManager] generate_sequences_with_callback started")

        # Determine which tasks to run
        task_id_tensor = prompts.batch.get("task_id", None)
        if task_id_tensor is not None:
            task_id = task_id_tensor.view(-1)[0].item()
            tasks_to_run = [task_id]
            logger.info(f"[FullyAsyncAgentLoopManager] Running single task: {task_id}")
        else:
            tasks_to_run = [1, 2, 3]  # Run all tasks sequentially
            logger.info("[FullyAsyncAgentLoopManager] Running all tasks sequentially: [1, 2, 3]")

        # Execute tasks with overlapping generation and reward computation
        accumulated_batch = prompts

        for current_task_id in tasks_to_run:
            logger.info(f"[FullyAsyncAgentLoopManager] Starting task {current_task_id}")

            # Set task_id in batch by creating new TensorDict (ensures proper tracking through chunk operations)
            task_id_tensor = torch.tensor([current_task_id] * len(accumulated_batch), dtype=torch.int32)

            if accumulated_batch.batch is not None:
                new_batch_dict = {k: v for k, v in accumulated_batch.batch.items()}
                new_batch_dict["task_id"] = task_id_tensor
                new_batch = TensorDict(new_batch_dict, batch_size=accumulated_batch.batch.batch_size)
            else:
                new_batch = TensorDict({"task_id": task_id_tensor}, batch_size=[len(accumulated_batch)])

            accumulated_batch.batch = new_batch

            # Execute the task using async worker
            worker = self._select_best_worker()
            output_ref = worker.generate_sequences.remote(accumulated_batch, on_task_complete=None)
            accumulated_batch = await output_ref
            logger.info(f"Task {current_task_id} generation completed")

            # Launch callback for reward computation in background (DON'T WAIT)
            if on_task_complete is not None:
                on_task_complete(current_task_id, accumulated_batch)

        logger.info("[FullyAsyncAgentLoopManager] All tasks completed")
        return accumulated_batch
        
    async def generate_sequences_with_callback_on_server(self, prompts: DataProto, server_index: int, on_task_complete=None) -> DataProto:

        task_id_tensor = prompts.batch.get("task_id", None)
        if task_id_tensor is not None:
            tasks_to_run = [task_id_tensor.view(-1)[0].item()]
        else:
            tasks_to_run = [1, 2, 3]

        accumulated_batch = prompts
        for current_task_id in tasks_to_run:
            task_id_tensor = torch.tensor([current_task_id] * len(accumulated_batch), dtype=torch.int32)

            if accumulated_batch.batch is not None:
                new_batch_dict = {k: v for k, v in accumulated_batch.batch.items()}
                new_batch_dict["task_id"] = task_id_tensor
                accumulated_batch.batch = TensorDict(new_batch_dict, batch_size=accumulated_batch.batch.batch_size)
            else:
                accumulated_batch.batch = TensorDict({"task_id": task_id_tensor}, batch_size=[len(accumulated_batch)])

            # ★ 여기만 다름: 지정된 server_index로
            worker = self._select_best_worker()
            output_ref = worker.generate_sequences_on_server.remote(accumulated_batch, server_index)
            accumulated_batch = await output_ref

            if on_task_complete is not None:
                on_task_complete(current_task_id, accumulated_batch)

        return accumulated_batch

    def _select_best_worker(self):
        """Select the best worker, simple round-robin load balancing"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        return worker

    def get_num_servers(self) -> int:
        """Get the number of available servers/GPUs."""
        return len(self.server_handles) if self.server_handles else 0

    async def cancel(self):
        worker_cancel_tasks = [worker.cancel_agent_loops.remote() for worker in self.agent_loop_workers]
        rollout_cancel_tasks = [replica.cancel() for replica in self.rollout_replicas]
        await asyncio.gather(*rollout_cancel_tasks, *worker_cancel_tasks)

    async def resume(self):
        rollout_resume_tasks = [replica.resume() for replica in self.rollout_replicas]
        worker_resume_tasks = [worker.resume_agent_loops.remote() for worker in self.agent_loop_workers]
        await asyncio.gather(*rollout_resume_tasks, *worker_resume_tasks)

    async def wake_up(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    async def sleep(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])

    async def clear_kv_cache(self):
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])