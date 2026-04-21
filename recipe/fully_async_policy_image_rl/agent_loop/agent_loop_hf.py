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


_NO_EDIT_MARKER = "No need to generate feedback."


def _is_edit_sample(feedback_text) -> bool:
    """An edit-worthy task2 feedback is any non-empty string that does NOT
    contain the explicit `No need to generate feedback.` sentence (the exact
    marker used by the task2 prompt template — see
    `recipe/image_rl/prompts_finegrained.py` / `reward_function_fine_grained.py`).
    Empty / non-string feedback is treated as no-edit.
    """
    if not isinstance(feedback_text, str) or not feedback_text:
        return False
    return _NO_EDIT_MARKER not in feedback_text


def build_edit_batch(batch: DataProto, group_size: Optional[int] = None) -> DataProto:
    """Build a fresh task3 input batch from task2 edit-worthy samples.

    Per-UID policy (user spec):
      1. Target `group_size` rows per UID (defaults to each UID's current row
         count; callers pass `rollout.n` to pad back to the original group
         size after per-sample termination filtering).
      2. Pick edit-worthy rows first, sorted by task2 reward desc.
      3. If fewer than `group_size` picks, pad with the highest-reward edit
         row of the same UID (= "duplicate best edit to fill the group").
      4. If a UID has zero edit rows, fall back to reward-ranked full group
         and pad with its top row if short.

    Requires `task2_token_level_scores` (written by the task2 reward callback)
    and `task2_feedback_texts` on the batch. Returns the batch unchanged if
    either is missing.
    """
    from recipe.fully_async_policy_image_rl.detach_utils import _slice_dataproto_with_meta
    from collections import OrderedDict

    if batch is None or getattr(batch, "batch", None) is None:
        return batch
    if "task2_token_level_scores" not in batch.batch:
        return batch

    feedback_texts = batch.non_tensor_batch.get("task2_feedback_texts", None)
    if feedback_texts is None:
        return batch

    total = len(batch)
    if total == 0:
        return batch

    uids = batch.non_tensor_batch.get("uid", None)
    if uids is None:
        uids = np.array(["_default"] * total, dtype=object)

    scores = (
        batch.batch["task2_token_level_scores"]
        .clamp(min=0)
        .sum(dim=-1)
        .cpu()
        .numpy()
    )

    uid_to_rows: "OrderedDict[Any, list[int]]" = OrderedDict()
    for i, u in enumerate(uids):
        uid_to_rows.setdefault(u, []).append(i)

    new_indices: list[int] = []
    for _uid, rows in uid_to_rows.items():
        target = int(group_size) if group_size is not None and group_size > 0 else len(rows)
        edit_rows = [i for i in rows if _is_edit_sample(feedback_texts[i])]
        edit_ranked = sorted(edit_rows, key=lambda i: -float(scores[i]))

        if edit_ranked:
            picks = list(edit_ranked)
            if len(picks) < target:
                # Cycle through reward-ranked edits: if deficit > len(edit_ranked),
                # loop again (A, B, C, A, B, C, ...) instead of flooding with top-1.
                deficit = target - len(picks)
                for i in range(deficit):
                    picks.append(edit_ranked[i % len(edit_ranked)])
            elif len(picks) > target:
                picks = picks[:target]
        else:
            ranked = sorted(rows, key=lambda i: -float(scores[i]))
            picks = ranked[:target]
            if len(picks) < target and ranked:
                deficit = target - len(picks)
                for i in range(deficit):
                    picks.append(ranked[i % len(ranked)])

        new_indices.extend(picks)

    return _slice_dataproto_with_meta(batch, new_indices)


def _softmax_sample_per_uid(
    uids: np.ndarray,
    scores: np.ndarray,
    k: int,
    row_subset: Optional[list[int]] = None,
    bias: str = "low",
    alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> list[int]:
    """Per-UID softmax sampling: draws `k` rows with replacement using
    p_i ∝ exp(sign * score_i / τ), τ = alpha * std. Falls back to uniform
    when std ≈ 0. `row_subset` restricts candidates per UID."""
    from collections import OrderedDict

    if rng is None:
        rng = np.random.default_rng()

    allowed: Optional[set] = set(row_subset) if row_subset is not None else None
    sign = -1.0 if bias == "low" else 1.0

    uid_to_rows: "OrderedDict[Any, list[int]]" = OrderedDict()
    for i, u in enumerate(uids):
        if allowed is not None and i not in allowed:
            continue
        uid_to_rows.setdefault(u, []).append(i)

    out: list[int] = []
    for _uid, rows in uid_to_rows.items():
        if not rows:
            continue
        r = np.array([float(scores[i]) for i in rows], dtype=np.float64)
        std = float(r.std())
        if std <= 1e-8:
            probs = np.full(len(rows), 1.0 / len(rows))
        else:
            tau = max(alpha * std, 1e-8)
            logits = sign * r / tau
            logits = logits - logits.max()
            p = np.exp(logits)
            s = p.sum()
            probs = (p / s) if (np.isfinite(s) and s > 0) else np.full(len(rows), 1.0 / len(rows))
        sampled = rng.choice(len(rows), size=k, replace=True, p=probs)
        out.extend(rows[j] for j in sampled)

    return out


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
            cfg_weight=config.cfg_weight,
            txt_top_k=config.txt_top_k,
            txt_top_p=config.txt_top_p,
            img_top_k=config.img_top_k,
            img_top_p=config.img_top_p,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["temperature"] = config.val_kwargs.val_temperature
            sampling_params["cfg_weight"] = config.val_kwargs.val_cfg_weight
            sampling_params["txt_top_k"] = config.val_kwargs.val_txt_top_k
            sampling_params["txt_top_p"] = config.val_kwargs.val_txt_top_p
            sampling_params["img_top_k"] = config.val_kwargs.val_img_top_k
            sampling_params["img_top_p"] = config.val_kwargs.val_img_top_p

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
            cfg_weight=config.cfg_weight,
            txt_top_k=config.txt_top_k,
            txt_top_p=config.txt_top_p,
            img_top_k=config.img_top_k,
            img_top_p=config.img_top_p,
        )
        if prompts.meta_info.get("validate", False):
            sampling_params["temperature"] = config.val_kwargs.val_temperature
            sampling_params["cfg_weight"] = config.val_kwargs.val_cfg_weight
            sampling_params["txt_top_k"] = config.val_kwargs.val_txt_top_k
            sampling_params["txt_top_p"] = config.val_kwargs.val_txt_top_p
            sampling_params["img_top_k"] = config.val_kwargs.val_img_top_k
            sampling_params["img_top_p"] = config.val_kwargs.val_img_top_p

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

    def _set_task_id_on_batch(self, batch: DataProto, current_task_id: int) -> DataProto:
        task_id_tensor = torch.tensor([current_task_id] * len(batch), dtype=torch.int32)
        if batch.batch is not None:
            new_batch_dict = {k: v for k, v in batch.batch.items()}
            new_batch_dict["task_id"] = task_id_tensor
            batch.batch = TensorDict(new_batch_dict, batch_size=batch.batch.batch_size)
        else:
            batch.batch = TensorDict({"task_id": task_id_tensor}, batch_size=[len(batch)])
        return batch

    def _stamp_turn_idx(self, batch: DataProto, turn_idx: int) -> DataProto:
        """Stamp a per-row turn_idx tensor so downstream knows which turn each
        row belongs to after concatenation across turns."""
        if batch is None or getattr(batch, "batch", None) is None:
            return batch
        turn_tensor = torch.full((len(batch),), turn_idx, dtype=torch.int32)
        new_batch_dict = {k: v for k, v in batch.batch.items()}
        new_batch_dict["turn_idx"] = turn_tensor
        batch.batch = TensorDict(new_batch_dict, batch_size=batch.batch.batch_size)
        return batch

    async def _await_reward_task(
        self,
        reward_task,
        batch: Optional[DataProto],
        task_id: int,
    ):
        """Await a reward task created by the rollouter callback and attach
        its `(reward_tensor, extras_dict)` result onto `batch` at this exact
        site — mutating through the stored Python reference avoids the
        cross-frame identity issues that broke in-place writes done inside
        the reward task itself.
        """
        if reward_task is None:
            return
        try:
            result = await reward_task
        except Exception as e:
            logger.warning(f"[AgentLoop] task{task_id} reward await failed: {e}")
            return
        if result is None or batch is None:
            return
        reward_tensor, reward_extras = result
        score_key = f"task{task_id}_token_level_scores"
        extras_key = f"task{task_id}_reward_extra_info"

        if getattr(batch, "meta_info", None) is None:
            batch.meta_info = {}
        batch.meta_info[score_key] = reward_tensor
        batch.meta_info[extras_key] = reward_extras

        if batch.batch is not None:
            try:
                batch.batch[score_key] = reward_tensor
            except Exception as e:
                logger.warning(
                    f"[AgentLoop] task{task_id} attach to batch.batch failed: {e}"
                )

    @staticmethod
    def _apply_no_edit_response_mask(batch: "DataProto") -> "DataProto":
        """For val: zero `response_mask` on rows whose task2 feedback said
        "No need to generate feedback." Token-level score masking (to -100)
        happens in `_val_finalize_worker` once reward tensors are attached.
        """
        if batch is None or getattr(batch, "batch", None) is None:
            return batch
        ntb = getattr(batch, "non_tensor_batch", None) or {}
        feedback_texts = ntb.get("task2_feedback_texts")
        if feedback_texts is None:
            return batch
        is_no_edit = np.array([not _is_edit_sample(f) for f in feedback_texts])
        idx = np.where(is_no_edit)[0]
        if len(idx) == 0:
            return batch
        if "response_mask" in batch.batch:
            batch.batch["response_mask"][idx] = 0
        return batch

    @staticmethod
    def _stamp_trajectory_ids(batch: DataProto) -> DataProto:
        """Stamp a per-row unique trajectory_id into non_tensor_batch so each
        rollout row can be tracked across turns (even after per-sample
        termination filtering). The ids are `np.arange(len(batch))`; they
        propagate through Ray round-trips and slicing unchanged.
        """
        if batch is None:
            return batch
        N = len(batch)
        if "trajectory_id" in batch.non_tensor_batch:
            return batch
        batch.non_tensor_batch["trajectory_id"] = np.arange(N, dtype=np.int64)
        return batch

    @staticmethod
    def _lookup_row_by_tid(batches: list[DataProto], tid: int, score_key: str):
        """Scan `batches` in reverse order and return (batch, row_idx) for the
        most-recent entry containing trajectory_id == tid AND `score_key` in
        its batch dict. Returns (None, None) if not found.
        """
        for b in reversed(batches):
            if b is None or "trajectory_id" not in b.non_tensor_batch:
                continue
            if score_key not in b.batch:
                continue
            arr = b.non_tensor_batch["trajectory_id"]
            hit = np.where(arr == tid)[0]
            if len(hit) > 0:
                return b, int(hit[0])
        return None, None

    def _compute_outcomes_with_avg(
        self,
        all_tids,
        task1_batch: Optional[DataProto],
        task3_batches: list[DataProto],
    ) -> dict:
        """Aggregate outcome reward per trajectory_id across all turns.

        A single trajectory may execute task3 at multiple turns
        (turn 0, turn 1, ..., until it says "No need to feedback" or the
        turn loop exits). Each of those task3 runs produces its own per-row
        reward tensor on the corresponding turn's batch. Within a single
        turn the same tid may also appear as multiple padding duplicates
        produced by `build_edit_batch`. We aggregate with a two-level mean
        — collapse padding duplicates within each turn first, then mean
        across turns — so each turn contributes equally regardless of how
        many padding duplicates it carried.

        Precedence:
          * If ≥ 1 task3 rewards exist for this tid → two-level mean
            (within-turn → across-turn), outcome_task_id = 3.
          * Else (trajectory terminated at turn 0's task2 before any task3)
            → task1 reward of this tid, outcome_task_id = 1.
          * Else → skipped (fallback zero-fill happens in
            `_attach_outcome_per_row`).
        """
        outcomes: dict = {}

        def _get_score_tensor(dp: DataProto, key: str):
            """Return the per-row score tensor from wherever it was stamped.
            Checks `batch.batch[key]` → `meta_info[key]` → `non_tensor_batch[key]`
            in order. `meta_info` is where the reward callback writes as a
            always-survives fallback (plain dict, no TensorDict rebuild risk).
            Returns None if no source has it.
            """
            if dp is None:
                return None
            b = getattr(dp, "batch", None)
            if b is not None and key in b:
                return b[key]
            meta = getattr(dp, "meta_info", None) or {}
            if key in meta:
                v = meta[key]
                if isinstance(v, torch.Tensor):
                    return v
                try:
                    return torch.as_tensor(np.asarray(v))
                except Exception:
                    return None
            ntb = getattr(dp, "non_tensor_batch", None) or {}
            if key in ntb:
                v = ntb[key]
                try:
                    return torch.as_tensor(np.asarray(v))
                except Exception:
                    return None
            return None

        # Pre-index task3 batches by tid → list of rows (not a single row),
        # because `build_edit_batch` may duplicate the same tid within a turn
        # to pad the group, and each duplicate gets its own independent task3
        # rollout with its own reward. Using a dict collapses duplicates to
        # the last one and silently drops the rest from the outcome average.
        t3_index: list[tuple["torch.Tensor", dict[int, list[int]]]] = []
        for t3_b in task3_batches:
            if t3_b is None:
                continue
            scores = _get_score_tensor(t3_b, "task3_token_level_scores")
            if scores is None:
                continue
            if "trajectory_id" not in t3_b.non_tensor_batch:
                continue
            arr = t3_b.non_tensor_batch["trajectory_id"]
            tid_to_rows: dict[int, list[int]] = {}
            for i in range(len(arr)):
                tid_to_rows.setdefault(int(arr[i]), []).append(i)
            t3_index.append((scores, tid_to_rows))

        # Task1 lookup index (single-shot since task1 only runs on turn 0).
        t1_index: Optional[dict] = None
        t1_scores_tensor = None
        if task1_batch is not None and "trajectory_id" in task1_batch.non_tensor_batch:
            t1_scores_tensor = _get_score_tensor(task1_batch, "task1_token_level_scores")
            if t1_scores_tensor is not None:
                arr = task1_batch.non_tensor_batch["trajectory_id"]
                t1_index = {int(arr[i]): i for i in range(len(arr))}

        for raw_tid in all_tids:
            tid = int(raw_tid)
            # Two-level mean: collapse padding duplicates within each turn first,
            # then average across turns. Keeps each turn's contribution equal
            # regardless of how many padding duplicates `build_edit_batch` inserted
            # for this tid — a flat mean would let a turn with more padding
            # dominate the outcome signal.
            per_turn_means: list = []
            for t3_scores, tid_to_rows in t3_index:
                rows = tid_to_rows.get(tid, ())
                if not rows:
                    continue
                if len(rows) == 1:
                    per_turn_means.append(t3_scores[rows[0]].float())
                else:
                    per_turn_means.append(
                        torch.stack([t3_scores[r] for r in rows]).float().mean(dim=0)
                    )

            if per_turn_means:
                if len(per_turn_means) == 1:
                    avg = per_turn_means[0].clone()
                else:
                    avg = torch.stack(per_turn_means).mean(dim=0)
                outcomes[tid] = (avg, 3)
                continue

            if t1_index is not None and t1_scores_tensor is not None:
                row = t1_index.get(tid)
                if row is not None:
                    outcomes[tid] = (t1_scores_tensor[row].clone(), 1)

        return outcomes

    def _attach_outcome_per_row(self, dp: DataProto, outcomes: dict) -> DataProto:
        """Look up each row's trajectory_id in `outcomes` and stamp
        `outcome_token_level_scores` + `outcome_task_id` accordingly.

        Fills zeros (and outcome_task_id=0) for rows whose trajectory_id is
        missing from the outcomes dict — should not happen in practice but
        guards against asymmetric data flow.
        """
        if dp is None or "trajectory_id" not in dp.non_tensor_batch:
            return dp

        tids = dp.non_tensor_batch["trajectory_id"]
        B = len(dp)

        # Reference shape / dtype from first entry in outcomes for fallback zeros.
        ref_shape = None
        ref_dtype = torch.float32
        for entry in outcomes.values():
            ref_shape = entry[0].shape
            ref_dtype = entry[0].dtype
            break

        scores_list = []
        tid_task_list = []
        missing = 0
        for i in range(B):
            tid = int(tids[i])
            entry = outcomes.get(tid)
            if entry is None:
                missing += 1
                if ref_shape is not None:
                    scores_list.append(torch.zeros(ref_shape, dtype=ref_dtype))
                else:
                    scores_list.append(torch.zeros((1,), dtype=ref_dtype))
                tid_task_list.append(0)
            else:
                scores_list.append(entry[0])
                tid_task_list.append(int(entry[1]))

        if missing > 0:
            logger.warning(
                f"[AgentLoop] {missing}/{B} rows have no outcome entry; zero-filled"
            )

        scores_tensor = torch.stack(scores_list)
        tid_tensor = torch.tensor(tid_task_list, dtype=torch.int32)

        new_batch_dict = {k: v for k, v in dp.batch.items()}
        new_batch_dict["outcome_token_level_scores"] = scores_tensor
        new_batch_dict["outcome_task_id"] = tid_tensor
        dp.batch = TensorDict(new_batch_dict, batch_size=dp.batch.batch_size)
        return dp

    @staticmethod
    def _keys_to_keep_for_task(all_keys, target_task_id: int) -> list[str]:
        """Return the subset of `all_keys` relevant for `task{target_task_id}_dp`.

        Keep: non-task-prefixed keys (common), task1_* (always needed as reference
        even for task2/task3 trainer compound-key logic), and task{target_task_id}_*.
        Drop: other tasks' prefixed keys.
        """
        drop_prefixes = {"task1_", "task2_", "task3_"}
        # task1_dp keeps only task1_* (as its own task); others keep task1_* too.
        if target_task_id == 1:
            drop_prefixes = {"task2_", "task3_"}
        elif target_task_id == 2:
            drop_prefixes = {"task3_"}
        elif target_task_id == 3:
            drop_prefixes = {"task2_"}
        return [
            k for k in all_keys
            if not any(k.startswith(p) for p in drop_prefixes)
        ]

    def _extract_task_view(self, batch: DataProto, target_task_id: int) -> DataProto:
        """Return a DataProto view of `batch` with only the keys relevant to
        `target_task_id` (plus common + outcome). Also rewrites `task_id` column
        to `target_task_id` so downstream knows which task this view represents.
        """
        tensor_keep = self._keys_to_keep_for_task(list(batch.batch.keys()), target_task_id)
        nontensor_keep = self._keys_to_keep_for_task(list(batch.non_tensor_batch.keys()), target_task_id)
        view = batch.select(
            batch_keys=tensor_keep,
            non_tensor_batch_keys=nontensor_keep,
            deepcopy=False,
        )
        return self._set_task_id_on_batch(view, target_task_id)

    @staticmethod
    def _safe_concat(dps: list[DataProto]) -> Optional[DataProto]:
        """Concat per-turn DPs with per-batch key intersection so variable key
        sets (e.g., early-terminated turn batches missing task3_*) still
        concat safely. Uses `detach_utils._concat_dataprotos_with_meta` for
        proper per-sample meta_info merging (list-extend when length matches
        the batch length, dict-recursive merge otherwise).
        """
        if not dps:
            return None
        if len(dps) == 1:
            return dps[0]
        from recipe.fully_async_policy_image_rl.detach_utils import _concat_dataprotos_with_meta
        tensor_common = set(dps[0].batch.keys())
        nontensor_common = set(dps[0].non_tensor_batch.keys())
        for dp in dps[1:]:
            tensor_common &= set(dp.batch.keys())
            nontensor_common &= set(dp.non_tensor_batch.keys())
        trimmed = [
            dp.select(
                batch_keys=list(tensor_common),
                non_tensor_batch_keys=list(nontensor_common),
                deepcopy=False,
            )
            for dp in dps
        ]
        return _concat_dataprotos_with_meta(trimmed)

    def _build_task_dict_per_sample(
        self,
        task1_batch: Optional[DataProto],
        task2_batches_per_turn: list[DataProto],
        task3_batches_per_turn: list[DataProto],
        outcomes: dict,
        has_task1_first_turn: bool,
    ) -> dict[int, Optional[DataProto]]:
        """Produce `{1: task1_dp, 2: task2_dp, 3: task3_dp}` with per-row
        outcome attribution.

        With per-sample termination, each turn contributes a potentially
        variable number of rows to task2_dp / task3_dp (samples that said
        `No need to feedback` at turn T terminate there; subsequent turns
        run on fewer samples). Every row is attributed to its original
        trajectory via `trajectory_id`; outcomes are looked up per-row.
        """
        out: dict[int, Optional[DataProto]] = {1: None, 2: None, 3: None}

        if has_task1_first_turn and task1_batch is not None:
            out[1] = self._extract_task_view(task1_batch, target_task_id=1)
            self._attach_outcome_per_row(out[1], outcomes)

        if task2_batches_per_turn:
            task2_views = [self._extract_task_view(b, 2) for b in task2_batches_per_turn]
            out[2] = self._safe_concat(task2_views)
            if out[2] is not None:
                self._attach_outcome_per_row(out[2], outcomes)

        if task3_batches_per_turn:
            task3_views = [self._extract_task_view(b, 3) for b in task3_batches_per_turn]
            out[3] = self._safe_concat(task3_views)
            if out[3] is not None:
                self._attach_outcome_per_row(out[3], outcomes)

        # NOTE: outcome-advantage (GRPO-normalized trajectory outcome broadcast
        # to the task's response_mask) is now computed by the trainer in
        # `_process_batch_common`, not here. Each task DP carries only the raw
        # `outcome_token_level_scores` + `outcome_task_id` per row.
        return out

    async def _regen_via_worker(
        self,
        input_batch: DataProto,
        task_id: int,
        turn_idx: int,
        server_index: Optional[int],
        on_task_complete,
    ) -> DataProto:
        """Run task_id generation and inline-await the reward so the score
        is attached before return."""
        input_batch = self._set_task_id_on_batch(input_batch, task_id)
        worker = self._select_best_worker()
        if server_index is None:
            output = await worker.generate_sequences.remote(input_batch, on_task_complete=None)
        else:
            output = await worker.generate_sequences_on_server.remote(input_batch, server_index)
        output = self._stamp_turn_idx(output, turn_idx)
        if on_task_complete is not None:
            rt = on_task_complete(task_id, output)
            await self._await_reward_task(rt, output, task_id=task_id)
        return output

    async def _run_task3_on_task2(
        self,
        task2_batch: DataProto,
        turn_idx: int,
        server_index: Optional[int],
        on_task_complete,
    ) -> Optional[DataProto]:
        """Run task3 on `task2_batch`'s edit-worthy subset so Step 2 regen
        rows earn their own r3. Without this, regen rows inherit the
        original tid's outcome and their task2 advantage gets cancelled
        under outcome_gamma > 0."""
        if task2_batch is None or len(task2_batch) == 0:
            return None

        from recipe.fully_async_policy_image_rl.detach_utils import _slice_dataproto_with_meta

        feedback_texts = task2_batch.non_tensor_batch.get("task2_feedback_texts", None)
        if feedback_texts is not None:
            active = [i for i in range(len(task2_batch)) if _is_edit_sample(feedback_texts[i])]
        else:
            active = list(range(len(task2_batch)))
        if not active:
            return None

        regen_task3_input = _slice_dataproto_with_meta(task2_batch, active)
        return await self._regen_via_worker(
            input_batch=regen_task3_input,
            task_id=3,
            turn_idx=turn_idx,
            server_index=server_index,
            on_task_complete=on_task_complete,
        )

    async def _step2_branching_regen(
        self,
        pre_task2_batch: DataProto,
        task2_batch: DataProto,
        turn_idx: int,
        server_index: Optional[int],
        on_task_complete,
        alpha: float = 1.0,
    ) -> Optional[DataProto]:
        """Step 2 Branching: per UID, softmax(-r2/τ) samples `rollout.n`
        row indices and re-runs task2 on their pre-task2 prefix."""
        if (
            task2_batch is None
            or getattr(task2_batch, "batch", None) is None
            or "task2_token_level_scores" not in task2_batch.batch
        ):
            return None
        total = len(task2_batch)
        if total == 0 or pre_task2_batch is None or len(pre_task2_batch) != total:
            return None

        uids = task2_batch.non_tensor_batch.get("uid", None)
        if uids is None:
            uids = np.array(["_default"] * total, dtype=object)

        r2 = (
            task2_batch.batch["task2_token_level_scores"]
            .clamp(min=0)
            .sum(dim=-1)
            .cpu()
            .numpy()
        )
        group_size = int(self.config.actor_rollout_ref.rollout.n)
        regen_indices = _softmax_sample_per_uid(
            uids=uids, scores=r2, k=group_size, bias="low", alpha=alpha
        )
        if not regen_indices:
            return None

        from recipe.fully_async_policy_image_rl.detach_utils import _slice_dataproto_with_meta

        regen_input = _slice_dataproto_with_meta(pre_task2_batch, regen_indices)
        return await self._regen_via_worker(
            input_batch=regen_input,
            task_id=2,
            turn_idx=turn_idx,
            server_index=server_index,
            on_task_complete=on_task_complete,
        )

    async def _step3_branching_regen(
        self,
        pre_task3_batch: DataProto,
        task3_batch: DataProto,
        turn_idx: int,
        server_index: Optional[int],
        on_task_complete,
        alpha: float = 1.0,
    ) -> Optional[DataProto]:
        """Step 3 Branching: per UID, filter to r2 >= median, then
        softmax(-r3/τ) samples `rollout.n` row indices and re-runs task3
        on their pre-task3 prefix. Requires r2 and r3 attached on the
        batch (caller inline-awaits both)."""
        if (
            task3_batch is None
            or getattr(task3_batch, "batch", None) is None
            or "task3_token_level_scores" not in task3_batch.batch
            or "task2_token_level_scores" not in task3_batch.batch
        ):
            return None
        total = len(task3_batch)
        if total == 0 or pre_task3_batch is None or len(pre_task3_batch) != total:
            return None

        uids = task3_batch.non_tensor_batch.get("uid", None)
        if uids is None:
            uids = np.array(["_default"] * total, dtype=object)

        r2 = (
            task3_batch.batch["task2_token_level_scores"]
            .clamp(min=0)
            .sum(dim=-1)
            .cpu()
            .numpy()
        )
        r3 = (
            task3_batch.batch["task3_token_level_scores"]
            .clamp(min=0)
            .sum(dim=-1)
            .cpu()
            .numpy()
        )

        from collections import OrderedDict
        uid_to_rows: "OrderedDict[Any, list[int]]" = OrderedDict()
        for i, u in enumerate(uids):
            uid_to_rows.setdefault(u, []).append(i)
        high_r2_subset: list[int] = []
        for _uid, rows in uid_to_rows.items():
            if not rows:
                continue
            r2_rows = np.array([r2[i] for i in rows])
            median_r2 = float(np.median(r2_rows))
            high_r2_subset.extend(i for i in rows if r2[i] >= median_r2)

        if not high_r2_subset:
            return None

        group_size = int(self.config.actor_rollout_ref.rollout.n)
        regen_indices = _softmax_sample_per_uid(
            uids=uids,
            scores=r3,
            k=group_size,
            row_subset=high_r2_subset,
            bias="low",
            alpha=alpha,
        )
        if not regen_indices:
            return None

        from recipe.fully_async_policy_image_rl.detach_utils import _slice_dataproto_with_meta

        regen_input = _slice_dataproto_with_meta(pre_task3_batch, regen_indices)
        return await self._regen_via_worker(
            input_batch=regen_input,
            task_id=3,
            turn_idx=turn_idx,
            server_index=server_index,
            on_task_complete=on_task_complete,
        )

    async def generate_sequences_with_callback(self, prompts: DataProto, on_task_complete=None) -> dict[int, Optional[DataProto]]:
        """Multi-turn sequence generation with per-task reward callbacks.

        Runs `task1 -> (task2 -> task3) x max_turns`. Each turn is an
        independent batch keyed on the rolling `current_imgs_pixel_values` /
        `current_img_tokens` image stream.

        Returns a `{1: task1_dp, 2: task2_dp, 3: task3_dp}` dict so that each
        task is flushed independently to training: task1_dp carries only
        unique task1 rollouts (turn 0), while task2_dp / task3_dp carry
        per-turn rollouts concatenated. Each row has `turn_idx`,
        `outcome_token_level_scores` (task3 reward if completed, task1 reward
        if early-terminated), and `outcome_task_id` stamped.

        Early-termination: if every sample in a turn's task2 output contains
        the `No need to feedback` marker, the turn loop stops; per-turn
        batches accumulated so far are returned in the dict.
        """
        logger.info("[FullyAsyncAgentLoopManager] generate_sequences_with_callback started")

        max_turns = int(getattr(self.config.actor_rollout_ref.rollout, "max_turns", 1))
        is_validate = bool(prompts.meta_info.get("validate", False))

        # Caller-pinned single task (no turn loop).
        task_id_tensor = prompts.batch.get("task_id", None)
        rollout_task_ids_in_meta = prompts.meta_info.get("rollout_task_ids", None)
        if task_id_tensor is not None and rollout_task_ids_in_meta is None:
            only_task_id = int(task_id_tensor.view(-1)[0].item())
            accumulated_batch = self._set_task_id_on_batch(prompts, only_task_id)
            worker = self._select_best_worker()
            accumulated_batch = await worker.generate_sequences.remote(accumulated_batch, on_task_complete=None)
            if on_task_complete is not None:
                rt = on_task_complete(only_task_id, accumulated_batch)
                await self._await_reward_task(rt, accumulated_batch, only_task_id)
            return {only_task_id: accumulated_batch}

        first_turn_task_ids = (
            list(rollout_task_ids_in_meta) if rollout_task_ids_in_meta is not None else [1, 2, 3]
        )
        has_task1_first_turn = 1 in first_turn_task_ids

        # Validation: run the configured task sequence once without turn loop
        # or edit-batch selection (keeps val metrics aligned with input rows).
        # Rows whose task2 emitted "No need to generate feedback" get their
        # task3 response_mask zeroed so downstream ignores them; the matching
        # -100 fill for task3_token_level_scores happens in
        # `_validate_async._val_finalize_worker` after scores land on batch.
        if is_validate:
            accumulated_batch = prompts
            for current_task_id in first_turn_task_ids:
                accumulated_batch = self._set_task_id_on_batch(accumulated_batch, current_task_id)
                worker = self._select_best_worker()
                accumulated_batch = await worker.generate_sequences.remote(accumulated_batch, on_task_complete=None)
                accumulated_batch = self._stamp_turn_idx(accumulated_batch, 0)
                if current_task_id == 3:
                    self._apply_no_edit_response_mask(accumulated_batch)
                if on_task_complete is not None:
                    on_task_complete(current_task_id, accumulated_batch)
            return accumulated_batch

        from recipe.fully_async_policy_image_rl.detach_utils import _slice_dataproto_with_meta

        # Stamp a unique id per row so we can track each trajectory across
        # per-sample termination / build_edit_batch row reshuffling.
        accumulated_batch = self._stamp_trajectory_ids(prompts)

        task1_batch: Optional[DataProto] = None
        task2_batches_per_turn: list[DataProto] = []
        task3_batches_per_turn: list[DataProto] = []
        outcomes: dict = {}

        for turn_idx in range(max_turns):
            if accumulated_batch is None or len(accumulated_batch) == 0:
                break

            if turn_idx == 0 and has_task1_first_turn:
                accumulated_batch = self._set_task_id_on_batch(accumulated_batch, 1)
                worker = self._select_best_worker()
                accumulated_batch = await worker.generate_sequences.remote(accumulated_batch, on_task_complete=None)
                accumulated_batch = self._stamp_turn_idx(accumulated_batch, turn_idx)
                if on_task_complete is not None:
                    t1_rt = on_task_complete(1, accumulated_batch)
                    # Await task1 reward so task1_token_level_scores is on the
                    # batch before we look it up for early-terminated samples.
                    await self._await_reward_task(t1_rt, accumulated_batch, task_id=1)
                task1_batch = accumulated_batch

            # Pre-task2 prefix snapshot for Step 2 regen (generate returns
            # a new DataProto, so this ref stays the pre-generation state).
            pre_task2_batch = accumulated_batch

            # task2 Phase 1
            accumulated_batch = self._set_task_id_on_batch(accumulated_batch, 2)
            worker = self._select_best_worker()
            accumulated_batch = await worker.generate_sequences.remote(accumulated_batch, on_task_complete=None)
            accumulated_batch = self._stamp_turn_idx(accumulated_batch, turn_idx)
            task2_reward_task = on_task_complete(2, accumulated_batch) if on_task_complete is not None else None
            await self._await_reward_task(task2_reward_task, accumulated_batch, task_id=2)

            # Step 2 Branching. Regen rows stay OUT of `accumulated_batch` —
            # concat'ing them here would let duplicates crowd unique high-r2
            # tids out of `build_edit_batch`'s top-r2 slice.
            from recipe.fully_async_policy_image_rl.detach_utils import _concat_dataprotos_with_meta
            task2_regen = await self._step2_branching_regen(
                pre_task2_batch=pre_task2_batch,
                task2_batch=accumulated_batch,
                turn_idx=turn_idx,
                server_index=None,
                on_task_complete=on_task_complete,
            )

            if task2_regen is not None:
                task2_batches_per_turn.append(
                    _concat_dataprotos_with_meta([accumulated_batch, task2_regen])
                )
            else:
                task2_batches_per_turn.append(accumulated_batch)

            # Run task3 on regen so each branched trajectory earns its own
            # r3 — otherwise regen rows inherit the original tid's outcome
            # and their task2 advantage gets cancelled by outcome_gamma.
            task2_regen_task3 = None
            if task2_regen is not None:
                task2_regen_task3 = await self._run_task3_on_task2(
                    task2_batch=task2_regen,
                    turn_idx=turn_idx,
                    server_index=None,
                    on_task_complete=on_task_complete,
                )

            # Per-sample termination: rows whose task2 feedback contains the
            # "No need to feedback" marker end their trajectory here.
            feedback_texts = accumulated_batch.non_tensor_batch.get("task2_feedback_texts", None)
            if feedback_texts is not None:
                is_no_edit = np.array([not _is_edit_sample(f) for f in feedback_texts])
            else:
                is_no_edit = np.zeros(len(accumulated_batch), dtype=bool)

            # Newly-terminated trajectories don't get outcomes here — they're
            # computed at end-of-loop by averaging across ALL task3 turns the
            # trajectory ran in (see `_compute_outcomes_with_avg`).
            active_indices = np.where(~is_no_edit)[0].tolist()
            if not active_indices:
                break
            accumulated_batch = _slice_dataproto_with_meta(accumulated_batch, active_indices)

            group_size = int(self.config.actor_rollout_ref.rollout.n)
            accumulated_batch = build_edit_batch(accumulated_batch, group_size=group_size)

            pre_task3_batch = accumulated_batch

            # task3 Phase 1
            accumulated_batch = self._set_task_id_on_batch(accumulated_batch, 3)
            worker = self._select_best_worker()
            accumulated_batch = await worker.generate_sequences.remote(accumulated_batch, on_task_complete=None)
            accumulated_batch = self._stamp_turn_idx(accumulated_batch, turn_idx)
            if on_task_complete is not None:
                t3_rt = on_task_complete(3, accumulated_batch)
                # Inline-await — Step 3 regen needs r3 on the batch.
                await self._await_reward_task(t3_rt, accumulated_batch, task_id=3)

            # Step 3 Branching
            task3_regen = await self._step3_branching_regen(
                pre_task3_batch=pre_task3_batch,
                task3_batch=accumulated_batch,
                turn_idx=turn_idx,
                server_index=None,
                on_task_complete=on_task_complete,
            )
            if task3_regen is not None:
                accumulated_batch = _concat_dataprotos_with_meta([accumulated_batch, task3_regen])

            # Fold Step 2 regen's task3 so its r3 enters outcomes[tid] and
            # turn+1 sees the branched trajectories.
            if task2_regen_task3 is not None:
                accumulated_batch = _concat_dataprotos_with_meta(
                    [accumulated_batch, task2_regen_task3]
                )

            task3_batches_per_turn.append(accumulated_batch)

        # Compute per-trajectory outcome by averaging task3 rewards across
        # every turn a trajectory ran task3 (falls back to task1 reward for
        # trajectories that terminated before any task3). Covers both
        # newly-terminated mid-loop trajectories and still-active ones.
        all_tids = None
        if task1_batch is not None and "trajectory_id" in task1_batch.non_tensor_batch:
            all_tids = task1_batch.non_tensor_batch["trajectory_id"]
        elif task2_batches_per_turn and "trajectory_id" in task2_batches_per_turn[0].non_tensor_batch:
            all_tids = task2_batches_per_turn[0].non_tensor_batch["trajectory_id"]
        if all_tids is not None:
            outcomes = self._compute_outcomes_with_avg(all_tids, task1_batch, task3_batches_per_turn)

        return self._build_task_dict_per_sample(
            task1_batch,
            task2_batches_per_turn,
            task3_batches_per_turn,
            outcomes,
            has_task1_first_turn,
        )

    async def generate_sequences_with_callback_on_server(self, prompts: DataProto, server_index: int, on_task_complete=None) -> dict[int, Optional[DataProto]]:
        """Server-pinned counterpart of `generate_sequences_with_callback`.
        Returns the same `{1, 2, 3}`-keyed dict."""
        max_turns = int(getattr(self.config.actor_rollout_ref.rollout, "max_turns", 1))
        is_validate = bool(prompts.meta_info.get("validate", False))

        rollout_task_ids_in_meta = prompts.meta_info.get("rollout_task_ids", None)
        task_id_tensor = prompts.batch.get("task_id", None)

        if task_id_tensor is not None and rollout_task_ids_in_meta is None:
            only_task_id = int(task_id_tensor.view(-1)[0].item())
            accumulated_batch = self._set_task_id_on_batch(prompts, only_task_id)
            worker = self._select_best_worker()
            accumulated_batch = await worker.generate_sequences_on_server.remote(accumulated_batch, server_index)
            if on_task_complete is not None:
                rt = on_task_complete(only_task_id, accumulated_batch)
                await self._await_reward_task(rt, accumulated_batch, only_task_id)
            return {only_task_id: accumulated_batch}

        first_turn_task_ids = (
            list(rollout_task_ids_in_meta) if rollout_task_ids_in_meta is not None else [1, 2, 3]
        )
        has_task1_first_turn = 1 in first_turn_task_ids

        # Validation: same single-pass semantics as the non-server variant.
        # See the sibling block above for the no-edit response_mask / -100
        # score handoff to `_val_finalize_worker`.
        if is_validate:
            accumulated_batch = prompts
            for current_task_id in first_turn_task_ids:
                accumulated_batch = self._set_task_id_on_batch(accumulated_batch, current_task_id)
                worker = self._select_best_worker()
                accumulated_batch = await worker.generate_sequences_on_server.remote(accumulated_batch, server_index)
                accumulated_batch = self._stamp_turn_idx(accumulated_batch, 0)
                if current_task_id == 3:
                    self._apply_no_edit_response_mask(accumulated_batch)
                if on_task_complete is not None:
                    on_task_complete(current_task_id, accumulated_batch)
            return accumulated_batch

        from recipe.fully_async_policy_image_rl.detach_utils import _slice_dataproto_with_meta

        accumulated_batch = self._stamp_trajectory_ids(prompts)

        task1_batch: Optional[DataProto] = None
        task2_batches_per_turn: list[DataProto] = []
        task3_batches_per_turn: list[DataProto] = []
        outcomes: dict = {}

        for turn_idx in range(max_turns):
            if accumulated_batch is None or len(accumulated_batch) == 0:
                break

            if turn_idx == 0 and has_task1_first_turn:
                accumulated_batch = self._set_task_id_on_batch(accumulated_batch, 1)
                worker = self._select_best_worker()
                accumulated_batch = await worker.generate_sequences_on_server.remote(accumulated_batch, server_index)
                accumulated_batch = self._stamp_turn_idx(accumulated_batch, turn_idx)
                if on_task_complete is not None:
                    t1_rt = on_task_complete(1, accumulated_batch)
                    await self._await_reward_task(t1_rt, accumulated_batch, task_id=1)
                task1_batch = accumulated_batch

            # Pre-task2 prefix snapshot for Step 2 regen.
            pre_task2_batch = accumulated_batch

            # task2 Phase 1
            accumulated_batch = self._set_task_id_on_batch(accumulated_batch, 2)
            worker = self._select_best_worker()
            accumulated_batch = await worker.generate_sequences_on_server.remote(accumulated_batch, server_index)
            accumulated_batch = self._stamp_turn_idx(accumulated_batch, turn_idx)
            task2_reward_task = on_task_complete(2, accumulated_batch) if on_task_complete is not None else None
            await self._await_reward_task(task2_reward_task, accumulated_batch, task_id=2)

            # Step 2 Branching. Regen rows stay OUT of `accumulated_batch`
            # so `build_edit_batch` below sees only the phase-1 pool.
            from recipe.fully_async_policy_image_rl.detach_utils import _concat_dataprotos_with_meta
            task2_regen = await self._step2_branching_regen(
                pre_task2_batch=pre_task2_batch,
                task2_batch=accumulated_batch,
                turn_idx=turn_idx,
                server_index=server_index,
                on_task_complete=on_task_complete,
            )

            if task2_regen is not None:
                task2_batches_per_turn.append(
                    _concat_dataprotos_with_meta([accumulated_batch, task2_regen])
                )
            else:
                task2_batches_per_turn.append(accumulated_batch)

            # Run task3 on regen so branched trajectories earn their own r3.
            task2_regen_task3 = None
            if task2_regen is not None:
                task2_regen_task3 = await self._run_task3_on_task2(
                    task2_batch=task2_regen,
                    turn_idx=turn_idx,
                    server_index=server_index,
                    on_task_complete=on_task_complete,
                )

            feedback_texts = accumulated_batch.non_tensor_batch.get("task2_feedback_texts", None)
            if feedback_texts is not None:
                is_no_edit = np.array([not _is_edit_sample(f) for f in feedback_texts])
            else:
                is_no_edit = np.zeros(len(accumulated_batch), dtype=bool)

            active_indices = np.where(~is_no_edit)[0].tolist()
            if not active_indices:
                break
            accumulated_batch = _slice_dataproto_with_meta(accumulated_batch, active_indices)

            group_size = int(self.config.actor_rollout_ref.rollout.n)
            accumulated_batch = build_edit_batch(accumulated_batch, group_size=group_size)

            pre_task3_batch = accumulated_batch

            # task3 Phase 1
            accumulated_batch = self._set_task_id_on_batch(accumulated_batch, 3)
            worker = self._select_best_worker()
            accumulated_batch = await worker.generate_sequences_on_server.remote(accumulated_batch, server_index)
            accumulated_batch = self._stamp_turn_idx(accumulated_batch, turn_idx)
            if on_task_complete is not None:
                t3_rt = on_task_complete(3, accumulated_batch)
                # Inline-await — Step 3 regen needs r3 on the batch.
                await self._await_reward_task(t3_rt, accumulated_batch, task_id=3)

            # Step 3 Branching.
            task3_regen = await self._step3_branching_regen(
                pre_task3_batch=pre_task3_batch,
                task3_batch=accumulated_batch,
                turn_idx=turn_idx,
                server_index=server_index,
                on_task_complete=on_task_complete,
            )
            if task3_regen is not None:
                accumulated_batch = _concat_dataprotos_with_meta([accumulated_batch, task3_regen])

            # Fold Step 2 regen's task3 so its r3 enters outcomes[tid].
            if task2_regen_task3 is not None:
                accumulated_batch = _concat_dataprotos_with_meta(
                    [accumulated_batch, task2_regen_task3]
                )

            task3_batches_per_turn.append(accumulated_batch)

        # Compute per-trajectory outcome by averaging task3 rewards across every
        # turn a trajectory ran task3 (falls back to task1 reward if the
        # trajectory never reached a task3). Covers both newly-terminated
        # mid-loop trajectories and still-active ones.
        all_tids = None
        if task1_batch is not None and "trajectory_id" in task1_batch.non_tensor_batch:
            all_tids = task1_batch.non_tensor_batch["trajectory_id"]
        elif task2_batches_per_turn and "trajectory_id" in task2_batches_per_turn[0].non_tensor_batch:
            all_tids = task2_batches_per_turn[0].non_tensor_batch["trajectory_id"]
        if all_tids is not None:
            outcomes = self._compute_outcomes_with_avg(all_tids, task1_batch, task3_batches_per_turn)

        return self._build_task_dict_per_sample(
            task1_batch,
            task2_batches_per_turn,
            task3_batches_per_turn,
            outcomes,
            has_task1_first_turn,
        )

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