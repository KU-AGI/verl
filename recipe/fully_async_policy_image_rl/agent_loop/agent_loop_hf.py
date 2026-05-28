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


def get_image_rollout_replica_class(rollout_name: str):
    if rollout_name in {"image_unified", "hf"}:
        return HuggingFaceReplica
    if rollout_name == "janus_sglang":
        from recipe.fully_async_policy_image_rl.sglang_rollout import JanusSGLangReplica

        return JanusSGLangReplica
    raise NotImplementedError(f"Unsupported fully-async image rollout backend: {rollout_name}")


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
      2. Pick edit-worthy rows uniformly at random.
      3. If fewer than `group_size` picks, pad by uniform sampling with
         replacement from the same edit-worthy pool.
      4. If a UID has zero edit rows, fall back to random sampling from the
         full group (also with replacement when short).

    Requires only `task2_feedback_texts` and `uid`. This keeps Phase1 free of
    reward-driven selection and avoids a mid-loop task2 reward dependency.
    """
    from recipe.fully_async_policy_image_rl.detach_utils import _slice_dataproto_with_meta
    from collections import OrderedDict

    if batch is None or getattr(batch, "batch", None) is None:
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

    rng = np.random.default_rng()

    uid_to_rows: "OrderedDict[Any, list[int]]" = OrderedDict()
    for i, u in enumerate(uids):
        uid_to_rows.setdefault(u, []).append(i)

    new_indices: list[int] = []
    for _uid, rows in uid_to_rows.items():
        target = int(group_size) if group_size is not None and group_size > 0 else len(rows)
        edit_rows = [i for i in rows if _is_edit_sample(feedback_texts[i])]
        if edit_rows:
            if len(edit_rows) >= target:
                picks = rng.choice(edit_rows, size=target, replace=False).tolist()
            else:
                picks = list(edit_rows)
                extra = rng.choice(edit_rows, size=target - len(edit_rows), replace=True).tolist()
                picks.extend(extra)
        else:
            if len(rows) >= target:
                picks = rng.choice(rows, size=target, replace=False).tolist()
            else:
                picks = list(rows)
                if rows:
                    extra = rng.choice(rows, size=target - len(rows), replace=True).tolist()
                    picks.extend(extra)

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
        self.rollout_replica_class = get_image_rollout_replica_class(config.actor_rollout_ref.rollout.name)

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
        self._attach_reward_result(batch, task_id, result)

    @staticmethod
    def _attach_reward_result(
        batch: Optional[DataProto],
        task_id: int,
        reward_result,
    ) -> None:
        if reward_result is None or batch is None:
            return
        reward_tensor, reward_extras = reward_result
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

    async def _await_and_attach_reward_entries(
        self,
        entries: list[tuple[int, DataProto, asyncio.Task]],
    ) -> None:
        if not entries:
            return
        results = await asyncio.gather(
            *[reward_task for _, _, reward_task in entries],
            return_exceptions=True,
        )
        for (task_id, batch_ref, _reward_task), result in zip(entries, results):
            if isinstance(result, Exception):
                logger.warning(f"[AgentLoop] task{task_id} reward await failed: {result}")
                continue
            self._attach_reward_result(batch_ref, task_id, result)

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
    def _ensure_branch_metadata(batch: DataProto) -> DataProto:
        """Ensure branch-tracking columns exist on `non_tensor_batch`."""
        if batch is None:
            return batch
        n = len(batch)
        if "branch_id" not in batch.non_tensor_batch:
            batch.non_tensor_batch["branch_id"] = np.full(n, -1, dtype=np.int64)
        if "parent_branch_id" not in batch.non_tensor_batch:
            batch.non_tensor_batch["parent_branch_id"] = np.full(n, -1, dtype=np.int64)
        return batch

    def _spawn_child_branch_ids(
        self,
        batch: DataProto,
        parent_ids: Optional[np.ndarray] = None,
    ) -> DataProto:
        """Assign fresh branch ids to every row in `batch`."""
        if batch is None or len(batch) == 0:
            return batch
        batch = self._ensure_branch_metadata(batch)
        n = len(batch)
        if parent_ids is None:
            parent_ids = np.asarray(batch.non_tensor_batch["branch_id"], dtype=np.int64)
        else:
            parent_ids = np.asarray(parent_ids, dtype=np.int64)
        next_id = int(getattr(self, "_branch_id_counter", 0))
        new_ids = np.arange(next_id, next_id + n, dtype=np.int64)
        self._branch_id_counter = next_id + n
        batch.non_tensor_batch["parent_branch_id"] = parent_ids
        batch.non_tensor_batch["branch_id"] = new_ids
        return batch

    @staticmethod
    def _build_row_map_by_trajectory(
        source_batch: Optional[DataProto],
        target_batch: Optional[DataProto],
    ) -> dict[int, int]:
        if source_batch is None or target_batch is None:
            return {}
        src_tids = (getattr(source_batch, "non_tensor_batch", None) or {}).get("trajectory_id")
        tgt_tids = (getattr(target_batch, "non_tensor_batch", None) or {}).get("trajectory_id")
        if src_tids is None or tgt_tids is None:
            n = min(len(source_batch), len(target_batch))
            return {i: i for i in range(n)}
        src_first: dict[int, int] = {}
        for i, tid in enumerate(src_tids):
            src_first.setdefault(int(tid), i)
        return {
            tgt_i: src_first[int(tid)]
            for tgt_i, tid in enumerate(tgt_tids)
            if int(tid) in src_first
        }

    @staticmethod
    def _build_alias_score_tensor(
        source_batch: Optional[DataProto],
        score_key: str,
        target_len: int,
        row_map: dict[int, int],
        fill_value: float = float("nan"),
    ) -> Optional[torch.Tensor]:
        if source_batch is None or getattr(source_batch, "batch", None) is None or score_key not in source_batch.batch:
            return None
        src = source_batch.batch[score_key]
        fill = torch.full(
            (target_len, *src.shape[1:]),
            fill_value,
            dtype=src.dtype,
            device=src.device,
        )
        for tgt_i, src_i in row_map.items():
            if 0 <= src_i < src.shape[0]:
                fill[tgt_i] = src[src_i]
        return fill

    @staticmethod
    def _build_alias_extra_dict(
        source_batch: Optional[DataProto],
        extras_key: str,
        target_len: int,
        row_map: dict[int, int],
    ) -> dict[str, list[Any]]:
        if source_batch is None:
            return {}
        meta = getattr(source_batch, "meta_info", None) or {}
        src_extras = meta.get(extras_key, {}) or {}
        if not isinstance(src_extras, dict) or not src_extras:
            return {}

        alias_extras: dict[str, list[Any]] = {}
        for sub_key, sub_vals in src_extras.items():
            if isinstance(sub_vals, np.ndarray):
                src_list = sub_vals.tolist()
            elif isinstance(sub_vals, list):
                src_list = list(sub_vals)
            else:
                continue
            fill = [None] * target_len
            for tgt_i, src_i in row_map.items():
                if 0 <= src_i < len(src_list):
                    fill[tgt_i] = src_list[src_i]
            alias_extras[sub_key] = fill
        return alias_extras

    @staticmethod
    def _build_alias_object_array(
        source_vals: Any,
        target_len: int,
        row_map: dict[int, int],
        default=None,
    ) -> np.ndarray:
        if source_vals is None:
            return np.array([default] * target_len, dtype=object)
        if isinstance(source_vals, np.ndarray):
            src_list = source_vals.tolist()
        else:
            src_list = list(source_vals)
        fill = np.array([default] * target_len, dtype=object)
        for tgt_i, src_i in row_map.items():
            if 0 <= src_i < len(src_list):
                fill[tgt_i] = src_list[src_i]
        return fill

    @staticmethod
    def _attach_logging_alias_from_source(
        target_batch: Optional[DataProto],
        alias_prefix: str,
        source_batch: Optional[DataProto],
        source_task_id: int,
        row_map: dict[int, int],
        include_feedback: bool = False,
        include_segment_mask: bool = False,
    ) -> Optional[DataProto]:
        if target_batch is None or source_batch is None:
            return target_batch
        if getattr(target_batch, "meta_info", None) is None:
            target_batch.meta_info = {}

        target_len = len(target_batch)
        score_key = f"task{source_task_id}_token_level_scores"
        alias_score = FullyAsyncAgentLoopManager._build_alias_score_tensor(
            source_batch, score_key, target_len, row_map
        )
        if alias_score is not None:
            target_batch.batch[f"{alias_prefix}_token_level_scores"] = alias_score

        if include_segment_mask and source_task_id == 2:
            alias_segment_mask = FullyAsyncAgentLoopManager._build_alias_score_tensor(
                source_batch,
                "task2_segment_mask",
                target_len,
                row_map,
                fill_value=0,
            )
            if alias_segment_mask is not None:
                target_batch.batch[f"{alias_prefix}_segment_mask"] = alias_segment_mask

        alias_extras = FullyAsyncAgentLoopManager._build_alias_extra_dict(
            source_batch,
            f"task{source_task_id}_reward_extra_info",
            target_len,
            row_map,
        )
        if alias_extras:
            target_batch.meta_info[f"{alias_prefix}_reward_extra_info"] = alias_extras

        if include_feedback:
            source_ntb = getattr(source_batch, "non_tensor_batch", None) or {}
            if "task2_feedback_texts" in source_ntb:
                target_batch.non_tensor_batch[f"{alias_prefix}_feedback_texts"] = (
                    FullyAsyncAgentLoopManager._build_alias_object_array(
                        source_ntb["task2_feedback_texts"],
                        target_len,
                        row_map,
                        default=None,
                    )
                )
        return target_batch

    def _propagate_logging_context(
        self,
        task1_batch: Optional[DataProto],
        task2_batches_per_turn: list[DataProto],
        task3_batches_per_turn: list[DataProto],
    ) -> None:
        for t2_b in task2_batches_per_turn:
            if t2_b is None:
                continue
            if task1_batch is not None:
                row_map = self._build_row_map_by_trajectory(task1_batch, t2_b)
                self._attach_logging_alias_from_source(
                    t2_b,
                    "task2_task1",
                    task1_batch,
                    source_task_id=1,
                    row_map=row_map,
                )

        for turn_idx, t3_b in enumerate(task3_batches_per_turn):
            if t3_b is None:
                continue
            if task1_batch is not None:
                row_map = self._build_row_map_by_trajectory(task1_batch, t3_b)
                self._attach_logging_alias_from_source(
                    t3_b,
                    "task3_task1",
                    task1_batch,
                    source_task_id=1,
                    row_map=row_map,
                )

            if turn_idx >= len(task2_batches_per_turn):
                continue
            t2_b = task2_batches_per_turn[turn_idx]
            if t2_b is None:
                continue
            src_row_arr = (getattr(t3_b, "non_tensor_batch", None) or {}).get("source_task2_row_idx")
            if src_row_arr is not None:
                row_map = {
                    tgt_i: int(src_i)
                    for tgt_i, src_i in enumerate(src_row_arr)
                    if 0 <= int(src_i) < len(t2_b)
                }
            else:
                row_map = self._build_row_map_by_trajectory(t2_b, t3_b)
            self._attach_logging_alias_from_source(
                t3_b,
                "task3_task2",
                t2_b,
                source_task_id=2,
                row_map=row_map,
                include_feedback=True,
                include_segment_mask=True,
            )

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

    @staticmethod
    def _get_outcome_extra_scalar(
        dp: "DataProto",
        task_id: int,
        field: str,
        rows: list,
        default: float = 0.0,
    ) -> float:
        """Read the mean of `field` from `meta_info[task{task_id}_reward_extra_info]`
        at the given `rows`. Falls back to `default` if anything is missing.

        The reward callback stores a dict-of-lists in meta_info under the key
        `task{task_id}_reward_extra_info`; each list has one entry per batch row.
        `_concat_dataprotos_with_meta` merges these list-valued sub-keys by
        list-extension, so the indices remain valid after concat.
        """
        if dp is None:
            return default
        meta = getattr(dp, "meta_info", None) or {}
        extras = meta.get(f"task{task_id}_reward_extra_info", {})
        if not extras:
            return default
        vals = extras.get(field)
        if vals is None:
            return default
        row_vals = []
        for r in rows:
            try:
                row_vals.append(float(vals[r]))
            except Exception:
                pass
        return float(np.mean(row_vals)) if row_vals else default

    @staticmethod
    def _read_reward_extra_value(extras: dict, field: str, row: int, default=None):
        vals = extras.get(field) if isinstance(extras, dict) else None
        if vals is None:
            return default
        try:
            return vals[row]
        except Exception:
            return default

    @staticmethod
    def _ensure_reward_extra_list(extras: dict, field: str, length: int) -> list:
        vals = extras.get(field)
        if isinstance(vals, np.ndarray):
            vals = vals.tolist()
        elif vals is None:
            vals = []
        elif not isinstance(vals, list):
            try:
                vals = list(vals)
            except Exception:
                vals = []
        if len(vals) < length:
            vals.extend([None] * (length - len(vals)))
        extras[field] = vals
        return vals

    def _attach_phase1_task3_mdp_step5_rewards(
        self,
        task1_batch: Optional[DataProto],
        pre_task3_batches_per_turn: list[DataProto],
        task3_batches_per_turn: list[DataProto],
    ) -> None:
        """Attach phase1 MDP Step5 rewards after task1/task3 rewards land.

        Step5 reward uses the edited image quality and edit-instruction
        following score:
            2 * (S_next + E_t) / (S_max + 1)
        where S_next = V_{t+1} + m_x D_{t+1}, S_max = 1 + m_x, and
        E_t is the edit instruction following score in [0, 1].

        Relative image-score gain is still logged for diagnostics, but it is
        no longer the Step5 training reward.
        S_prev is task1_image_score for root branches, or the parent branch's
        previous task3_image_score for edited branches.
        """
        if not task3_batches_per_turn:
            return

        algo_cfg = getattr(getattr(self, "config", None), "algorithm", None) or {}
        _get_hp = algo_cfg.get if hasattr(algo_cfg, "get") else lambda k, d: getattr(algo_cfg, k, d)
        eps = float(_get_hp("mdp_score_eps", 1e-6))

        task1_image_by_tid: dict[int, float] = {}
        task1_image_max_by_tid: dict[int, float] = {}
        if task1_batch is not None:
            t1_meta = getattr(task1_batch, "meta_info", None) or {}
            t1_extras = t1_meta.get("task1_reward_extra_info", {}) or {}
            tids = (getattr(task1_batch, "non_tensor_batch", None) or {}).get("trajectory_id")
            if tids is not None:
                for row, tid in enumerate(tids):
                    val = self._read_reward_extra_value(t1_extras, "task1_image_score", row, None)
                    if val is None:
                        val = self._read_reward_extra_value(t1_extras, "task1_align", row, None)
                    try:
                        task1_image_by_tid[int(tid)] = float(val)
                        task1_image_max_by_tid[int(tid)] = self._as_float(
                            self._read_reward_extra_value(t1_extras, "task1_image_score_max", row, 1.0),
                            1.0,
                        )
                    except Exception:
                        pass

        task3_image_by_branch: dict[int, float] = {}
        task3_image_max_by_branch: dict[int, float] = {}
        for t3_b in task3_batches_per_turn:
            if t3_b is None:
                continue
            t3_meta = getattr(t3_b, "meta_info", None) or {}
            t3_extras = t3_meta.get("task3_reward_extra_info", {}) or {}
            bids = (getattr(t3_b, "non_tensor_batch", None) or {}).get("branch_id")
            if bids is None:
                continue
            for row, bid in enumerate(bids):
                val = self._read_reward_extra_value(t3_extras, "task3_image_score", row, None)
                if val is None:
                    val = self._read_reward_extra_value(t3_extras, "task3_align", row, None)
                try:
                    task3_image_by_branch[int(bid)] = float(val)
                    task3_image_max_by_branch[int(bid)] = self._as_float(
                        self._read_reward_extra_value(t3_extras, "task3_image_score_max", row, 1.0),
                        1.0,
                    )
                except Exception:
                    pass

        for pre_b, t3_b in zip(pre_task3_batches_per_turn, task3_batches_per_turn):
            if t3_b is None:
                continue
            if getattr(t3_b, "meta_info", None) is None:
                t3_b.meta_info = {}
            t3_extras = t3_b.meta_info.setdefault("task3_reward_extra_info", {})
            if not isinstance(t3_extras, dict):
                continue

            n = len(t3_b)
            ntb = getattr(t3_b, "non_tensor_batch", None) or {}
            pre_ntb = getattr(pre_b, "non_tensor_batch", None) or {}
            tids = ntb.get("trajectory_id", pre_ntb.get("trajectory_id"))
            parents = ntb.get("parent_branch_id", pre_ntb.get("parent_branch_id"))
            if tids is None or parents is None:
                continue

            prev_vals = self._ensure_reward_extra_list(t3_extras, "task3_prev_image_score", n)
            next_vals = self._ensure_reward_extra_list(t3_extras, "task3_next_image_score", n)
            max_vals = self._ensure_reward_extra_list(t3_extras, "task3_image_score_max", n)
            gain_vals = self._ensure_reward_extra_list(t3_extras, "task3_image_score_gain", n)
            gain_score_vals = self._ensure_reward_extra_list(t3_extras, "task3_image_gain_score", n)
            gain_positive_vals = self._ensure_reward_extra_list(t3_extras, "task3_image_gain_positive_score", n)
            relative_gain_vals = self._ensure_reward_extra_list(t3_extras, "task3_relative_image_gain", n)
            edit_if_vals = self._ensure_reward_extra_list(t3_extras, "task3_step5_edit_if_score", n)
            step5_vals = self._ensure_reward_extra_list(t3_extras, "task3_step5_reward", n)

            for row in range(n):
                next_s = self._read_reward_extra_value(t3_extras, "task3_image_score", row, None)
                if next_s is None:
                    next_s = self._read_reward_extra_value(t3_extras, "task3_align", row, None)
                try:
                    tid = int(tids[row])
                    parent_bid = int(parents[row])
                except Exception:
                    continue

                prev_s = task3_image_by_branch.get(parent_bid) if parent_bid >= 0 else task1_image_by_tid.get(tid)
                s_max = task3_image_max_by_branch.get(parent_bid) if parent_bid >= 0 else task1_image_max_by_tid.get(tid)
                if s_max is None:
                    s_max = self._read_reward_extra_value(t3_extras, "task3_image_score_max", row, 1.0)
                try:
                    prev_s = float(prev_s)
                    next_s = float(next_s)
                    s_max = float(s_max)
                except Exception:
                    continue

                gain = next_s - prev_s
                gain_positive = 1.0 if gain > 0.0 else 0.0
                if gain >= 0.0:
                    denom = max(s_max - prev_s, eps)
                else:
                    denom = max(prev_s, eps)
                relative_gain = max(-1.0, min(1.0, gain / denom))

                edit_if = self._read_reward_extra_value(t3_extras, "task3_edit_if_reward", row, None)
                if edit_if is None:
                    edit_if = self._read_reward_extra_value(t3_extras, "task3_if", row, 0.0)
                try:
                    edit_if = max(0.0, min(1.0, float(edit_if)))
                except Exception:
                    edit_if = 0.0

                # S_max = 1 + m_x, so S_max + 1 = 2 + m_x.
                # No detector: r5 = V_{t+1} + E_t.
                # Detector active: r5 = 2/3 * (V_{t+1} + D_{t+1} + E_t).
                step5_reward = 2.0 * (next_s + edit_if) / max(s_max + 1.0, eps)
                step5_reward = max(0.0, min(2.0, step5_reward))
                prev_vals[row] = prev_s
                next_vals[row] = next_s
                max_vals[row] = s_max
                gain_vals[row] = gain
                gain_score_vals[row] = gain
                gain_positive_vals[row] = gain_positive
                relative_gain_vals[row] = relative_gain
                edit_if_vals[row] = edit_if
                step5_vals[row] = step5_reward

    @staticmethod
    def _event_return_key(dp: DataProto, row: int, event_name: str) -> tuple[int, int, str]:
        return (id(dp), int(row), event_name)

    @staticmethod
    def _mean_return(return_map: dict, dp: DataProto, row: int, event_name: str, default: float = 0.0) -> float:
        vals = return_map.get(FullyAsyncAgentLoopManager._event_return_key(dp, row, event_name))
        if not vals:
            return default
        return float(np.mean(vals))

    @staticmethod
    def _put_scalar_on_last_mask_token(score_tensor: torch.Tensor, mask_tensor: torch.Tensor, row: int, scalar: float) -> None:
        if row < 0 or row >= score_tensor.shape[0]:
            return
        valid = torch.nonzero(mask_tensor[row] > 0, as_tuple=False).flatten()
        if valid.numel() == 0:
            return
        score_tensor[row, int(valid[-1].item())] = float(scalar)

    @staticmethod
    def _put_scalar_on_last_segment_token(score_tensor: torch.Tensor, mask_tensor: torch.Tensor, row: int, segment_id: int, scalar: float) -> None:
        if row < 0 or row >= score_tensor.shape[0]:
            return
        valid = torch.nonzero(mask_tensor[row] == int(segment_id), as_tuple=False).flatten()
        if valid.numel() == 0:
            return
        score_tensor[row, int(valid[-1].item())] = float(scalar)

    @staticmethod
    def _as_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _attach_phase1_mdp_token_scores(
        self,
        all_tids,
        task1_batch: Optional[DataProto],
        task2_batches: list[DataProto],
        task3_batches: list[DataProto],
    ) -> None:
        """Attach phase1 token scores according to `algorithm.mdp_reward_version`.

        Default `gae` preserves the existing discounted-return backup. The
        `multi_step` variant writes immediate per-step rewards only.
        """
        algo_cfg = getattr(getattr(self, "config", None), "algorithm", None) or {}
        _get_hp = algo_cfg.get if hasattr(algo_cfg, "get") else lambda k, d: getattr(algo_cfg, k, d)
        version = str(_get_hp("mdp_reward_version", "gae")).lower()
        if version in {"multi_step", "multistep"}:
            self._attach_phase1_mdp_multistep_token_scores(
                task1_batch,
                task2_batches,
                task3_batches,
            )
            return
        if version in {"outcome", "outcome_average", "outcome_avg"}:
            self._attach_phase1_mdp_outcome_token_scores(
                all_tids,
                task1_batch,
                task2_batches,
                task3_batches,
            )
            return
        if version in {"final_image_outcome", "final_image", "image_outcome"}:
            self._attach_phase1_mdp_final_image_outcome_token_scores(
                all_tids,
                task1_batch,
                task2_batches,
                task3_batches,
            )
            return
        if version not in {"gae", "discounted", "discounted_return"}:
            logger.warning(
                "[AgentLoop] Unknown algorithm.mdp_reward_version=%s; falling back to gae",
                version,
            )
        self._attach_phase1_mdp_discounted_token_scores(
            all_tids,
            task1_batch,
            task2_batches,
            task3_batches,
        )

    def _attach_phase1_mdp_multistep_token_scores(
        self,
        task1_batch: Optional[DataProto],
        task2_batches: list[DataProto],
        task3_batches: list[DataProto],
    ) -> None:
        """Replace phase1 token scores with immediate multi-step rewards.

        Rewards are intentionally built from raw components, not `*_image_score`
        or `*_align`, because those fields include a 0.2 detector weight.
        """
        if task1_batch is not None and "task1_response_mask" in task1_batch.batch:
            mask = task1_batch.batch["task1_response_mask"]
            scores = torch.zeros_like(mask, dtype=torch.float32)
            if getattr(task1_batch, "meta_info", None) is None:
                task1_batch.meta_info = {}
            task1_extras = task1_batch.meta_info.setdefault("task1_reward_extra_info", {})
            multi_vals = self._ensure_reward_extra_list(task1_extras, "task1_multi_step_score", len(task1_batch))
            for row in range(len(task1_batch)):
                vqa = self._as_float(self._read_reward_extra_value(task1_extras, "task1_vqa_reward", row, 0.0))
                detector = self._as_float(self._read_reward_extra_value(task1_extras, "task1_detector_reward", row, 0.0))
                value = vqa + detector
                multi_vals[row] = value
                self._put_scalar_on_last_mask_token(scores, mask, row, value)
            task1_batch.batch["task1_token_level_scores"] = scores
            task1_batch.meta_info["task1_token_level_scores"] = scores

        for t2_b in task2_batches:
            if t2_b is None or "task2_response_mask" not in t2_b.batch:
                continue
            response_mask = t2_b.batch["task2_response_mask"]
            segment_mask = t2_b.batch.get("task2_segment_mask", response_mask)
            scores = torch.zeros_like(response_mask, dtype=torch.float32)
            if getattr(t2_b, "meta_info", None) is None:
                t2_b.meta_info = {}
            task2_extras = t2_b.meta_info.setdefault("task2_reward_extra_info", {})
            step2_vals = self._ensure_reward_extra_list(task2_extras, "task2_step2_multi_step_score", len(t2_b))
            step3_vals = self._ensure_reward_extra_list(task2_extras, "task2_step3_multi_step_score", len(t2_b))
            step4_vals = self._ensure_reward_extra_list(task2_extras, "task2_step4_multi_step_score", len(t2_b))
            for row in range(len(t2_b)):
                step2 = self._as_float(self._read_reward_extra_value(task2_extras, "task2_prompt_to_tuple_reward", row, 0.0))
                step3 = self._as_float(self._read_reward_extra_value(task2_extras, "task2_tuple_to_vqa_reward", row, 0.0))
                step4 = self._as_float(self._read_reward_extra_value(task2_extras, "task2_vqa_to_feedback_reward", row, 0.0))
                step2_vals[row] = step2
                step3_vals[row] = step3
                step4_vals[row] = step4
                self._put_scalar_on_last_segment_token(scores, segment_mask, row, 2, step2)
                self._put_scalar_on_last_segment_token(scores, segment_mask, row, 3, step3)
                self._put_scalar_on_last_segment_token(scores, segment_mask, row, 4, step4)
            t2_b.batch["task2_token_level_scores"] = scores
            t2_b.batch["task2_local_token_level_scores"] = scores.clone()
            t2_b.meta_info["task2_token_level_scores"] = scores
            t2_b.meta_info["task2_local_token_level_scores"] = scores.clone()

        for t3_b in task3_batches:
            if t3_b is None or "task3_response_mask" not in t3_b.batch:
                continue
            mask = t3_b.batch["task3_response_mask"]
            scores = torch.zeros_like(mask, dtype=torch.float32)
            if getattr(t3_b, "meta_info", None) is None:
                t3_b.meta_info = {}
            task3_extras = t3_b.meta_info.setdefault("task3_reward_extra_info", {})
            multi_vals = self._ensure_reward_extra_list(task3_extras, "task3_multi_step_score", len(t3_b))
            for row in range(len(t3_b)):
                vqa = self._as_float(self._read_reward_extra_value(task3_extras, "task3_vqa_reward", row, 0.0))
                detector = self._as_float(self._read_reward_extra_value(task3_extras, "task3_detector_reward", row, 0.0))
                edit_if = self._read_reward_extra_value(task3_extras, "task3_edit_if_reward", row, None)
                if edit_if is None:
                    edit_if = self._read_reward_extra_value(task3_extras, "task3_if", row, 0.0)
                value = vqa + detector + self._as_float(edit_if)
                multi_vals[row] = value
                self._put_scalar_on_last_mask_token(scores, mask, row, value)
            t3_b.batch["task3_token_level_scores"] = scores
            t3_b.meta_info["task3_token_level_scores"] = scores

    def _attach_phase1_mdp_outcome_token_scores(
        self,
        all_tids,
        task1_batch: Optional[DataProto],
        task2_batches: list[DataProto],
        task3_batches: list[DataProto],
    ) -> None:
        """Replace phase1 token scores with trajectory-average outcome rewards.

        For each terminal trajectory path, compute:
            R_out = mean(r_step1, r_step2, ..., r_stepH)
        and write the same R_out back to every step event in that path. Shared
        prefix rows that feed multiple terminal branches receive the mean over
        descendant branch outcomes, matching the discounted-return branch logic.
        """
        algo_cfg = getattr(getattr(self, "config", None), "algorithm", None) or {}
        _get_hp = algo_cfg.get if hasattr(algo_cfg, "get") else lambda k, d: getattr(algo_cfg, k, d)
        eps = float(_get_hp("mdp_score_eps", 1e-6))

        _ex = self._get_outcome_extra_scalar
        outcomes_by_event: dict[tuple[int, int, str], list[float]] = {}

        def _append_outcome(dp: DataProto, row: int, event_name: str, value: float) -> None:
            outcomes_by_event.setdefault(self._event_return_key(dp, row, event_name), []).append(float(value))

        # task1 lookup ---------------------------------------------------------
        t1_index: dict[int, int] = {}
        if task1_batch is not None and "trajectory_id" in task1_batch.non_tensor_batch:
            for i, tid in enumerate(task1_batch.non_tensor_batch["trajectory_id"]):
                t1_index.setdefault(int(tid), i)

        # task2/task3 rows grouped by shared prefix vs branch path -------------
        branch_parent: dict[int, int] = {}
        branch_tid: dict[int, int] = {}
        shared_t2_rows: dict[int, list[tuple]] = {}
        shared_t3_rows: dict[int, list[tuple]] = {}
        branch_t2_rows: dict[int, list[tuple]] = {}
        branch_t3_rows: dict[int, list[tuple]] = {}

        for turn_i, t2_b in enumerate(task2_batches):
            if t2_b is None or "trajectory_id" not in t2_b.non_tensor_batch:
                continue
            self._ensure_branch_metadata(t2_b)
            tids = np.asarray(t2_b.non_tensor_batch["trajectory_id"], dtype=np.int64)
            bids = np.asarray(t2_b.non_tensor_batch["branch_id"], dtype=np.int64)
            parents = np.asarray(t2_b.non_tensor_batch["parent_branch_id"], dtype=np.int64)
            for row in range(len(t2_b)):
                tid = int(tids[row])
                bid = int(bids[row])
                item = (turn_i, t2_b, row)
                if bid >= 0:
                    branch_parent.setdefault(bid, int(parents[row]))
                    branch_tid.setdefault(bid, tid)
                    branch_t2_rows.setdefault(bid, []).append(item)
                else:
                    shared_t2_rows.setdefault(tid, []).append(item)

        for turn_i, t3_b in enumerate(task3_batches):
            if t3_b is None or "trajectory_id" not in t3_b.non_tensor_batch:
                continue
            self._ensure_branch_metadata(t3_b)
            tids = np.asarray(t3_b.non_tensor_batch["trajectory_id"], dtype=np.int64)
            bids = np.asarray(t3_b.non_tensor_batch["branch_id"], dtype=np.int64)
            parents = np.asarray(t3_b.non_tensor_batch["parent_branch_id"], dtype=np.int64)
            for row in range(len(t3_b)):
                tid = int(tids[row])
                bid = int(bids[row])
                item = (turn_i, t3_b, row)
                if bid >= 0:
                    branch_parent.setdefault(bid, int(parents[row]))
                    branch_tid.setdefault(bid, tid)
                    branch_t3_rows.setdefault(bid, []).append(item)
                else:
                    shared_t3_rows.setdefault(tid, []).append(item)

        all_branch_ids = set(branch_parent.keys())
        branch_children: dict[int, list[int]] = {}
        for bid, parent in branch_parent.items():
            if parent >= 0:
                branch_children.setdefault(parent, []).append(bid)
        terminal_branch_ids = sorted(all_branch_ids - set(branch_children.keys()))

        def _dedupe_sorted(rows: list[tuple]) -> list[tuple]:
            rows = sorted(rows, key=lambda x: (x[0], x[2]))
            out = []
            seen = set()
            for turn_i, dp, row in rows:
                key = (turn_i, id(dp), row)
                if key in seen:
                    continue
                seen.add(key)
                out.append((turn_i, dp, row))
            return out

        def _branch_chain(bid: int) -> list[int]:
            chain = []
            cur = bid
            while cur >= 0:
                chain.append(cur)
                cur = int(branch_parent.get(cur, -1))
            return list(reversed(chain))

        def _collect_branch_rows(bid: int) -> tuple[list[tuple], list[tuple]]:
            tid = int(branch_tid[bid])
            t2_rows = list(shared_t2_rows.get(tid, []))
            t3_rows = list(shared_t3_rows.get(tid, []))
            for seg_bid in _branch_chain(bid):
                t2_rows.extend(branch_t2_rows.get(seg_bid, []))
                t3_rows.extend(branch_t3_rows.get(seg_bid, []))
            return _dedupe_sorted(t2_rows), _dedupe_sorted(t3_rows)

        def _build_events(tid: int, t2_rows: list[tuple], t3_rows: list[tuple]) -> list[tuple]:
            events: list[tuple[float, DataProto, int, str, float]] = []
            if task1_batch is not None and tid in t1_index:
                row = t1_index[tid]
                image_score = _ex(
                    task1_batch,
                    1,
                    "task1_image_score",
                    [row],
                    default=_ex(task1_batch, 1, "task1_mdp_reward", [row], 0.0),
                )
                image_score_max = _ex(task1_batch, 1, "task1_image_score_max", [row], 1.0)
                reward = 2.0 * float(image_score) / max(float(image_score_max), eps)
                events.append((0.0, task1_batch, row, "step1", reward))

            for turn_i, t2_b, row in t2_rows:
                base = 1.0 + 4.0 * float(turn_i)
                events.append((base + 0.0, t2_b, row, "step2", _ex(t2_b, 2, "task2_prompt_to_tuple_reward", [row], 0.0)))
                events.append((base + 1.0, t2_b, row, "step3", _ex(t2_b, 2, "task2_tuple_to_vqa_reward", [row], 0.0)))
                events.append((base + 2.0, t2_b, row, "step4", _ex(t2_b, 2, "task2_vqa_to_feedback_reward", [row], 0.0)))

            for turn_i, t3_b, row in t3_rows:
                base = 1.0 + 4.0 * float(turn_i)
                events.append((base + 3.0, t3_b, row, "step5", _ex(t3_b, 3, "task3_step5_reward", [row], 0.0)))

            return sorted(events, key=lambda x: x[0])

        def _accumulate_outcome(events: list[tuple]) -> None:
            if not events:
                return
            outcome = sum(float(reward) for _, _, _, _, reward in events) / max(float(len(events)), eps)
            for _, dp, row, event_name, _ in events:
                _append_outcome(dp, row, event_name, outcome)

        for bid in terminal_branch_ids:
            tid = int(branch_tid[bid])
            t2_rows, t3_rows = _collect_branch_rows(bid)
            _accumulate_outcome(_build_events(tid, t2_rows, t3_rows))

        tids_with_branch = {int(branch_tid[bid]) for bid in all_branch_ids}
        for raw_tid in all_tids if all_tids is not None else []:
            tid = int(raw_tid)
            if tid in tids_with_branch:
                continue
            t2_rows = _dedupe_sorted(shared_t2_rows.get(tid, []))
            t3_rows = _dedupe_sorted(shared_t3_rows.get(tid, []))
            _accumulate_outcome(_build_events(tid, t2_rows, t3_rows))

        # Write outcome rewards back to task token-level score tensors ---------
        if task1_batch is not None and "task1_response_mask" in task1_batch.batch:
            scores = torch.zeros_like(task1_batch.batch["task1_response_mask"], dtype=torch.float32)
            if getattr(task1_batch, "meta_info", None) is None:
                task1_batch.meta_info = {}
            task1_extras = task1_batch.meta_info.setdefault("task1_reward_extra_info", {})
            outcome_vals = self._ensure_reward_extra_list(task1_extras, "task1_outcome_score", len(task1_batch))
            for tid, row in t1_index.items():
                value = self._mean_return(outcomes_by_event, task1_batch, row, "step1", 0.0)
                outcome_vals[row] = value
                self._put_scalar_on_last_mask_token(scores, task1_batch.batch["task1_response_mask"], row, value)
            task1_batch.batch["task1_token_level_scores"] = scores
            task1_batch.meta_info["task1_token_level_scores"] = scores

        for t2_b in task2_batches:
            if t2_b is None or "task2_response_mask" not in t2_b.batch:
                continue
            response_mask = t2_b.batch["task2_response_mask"]
            segment_mask = t2_b.batch.get("task2_segment_mask", response_mask)
            scores = torch.zeros_like(response_mask, dtype=torch.float32)
            local_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            if getattr(t2_b, "meta_info", None) is None:
                t2_b.meta_info = {}
            task2_extras = t2_b.meta_info.setdefault("task2_reward_extra_info", {})
            step2_vals = self._ensure_reward_extra_list(task2_extras, "task2_step2_outcome_score", len(t2_b))
            step3_vals = self._ensure_reward_extra_list(task2_extras, "task2_step3_outcome_score", len(t2_b))
            step4_vals = self._ensure_reward_extra_list(task2_extras, "task2_step4_outcome_score", len(t2_b))
            for row in range(len(t2_b)):
                step2 = self._mean_return(outcomes_by_event, t2_b, row, "step2", 0.0)
                step3 = self._mean_return(outcomes_by_event, t2_b, row, "step3", 0.0)
                step4 = self._mean_return(outcomes_by_event, t2_b, row, "step4", 0.0)
                step2_vals[row] = step2
                step3_vals[row] = step3
                step4_vals[row] = step4
                self._put_scalar_on_last_segment_token(scores, segment_mask, row, 2, step2)
                self._put_scalar_on_last_segment_token(scores, segment_mask, row, 3, step3)
                self._put_scalar_on_last_segment_token(scores, segment_mask, row, 4, step4)
                self._put_scalar_on_last_segment_token(
                    local_scores,
                    segment_mask,
                    row,
                    2,
                    _ex(t2_b, 2, "task2_prompt_to_tuple_reward", [row], 0.0),
                )
                self._put_scalar_on_last_segment_token(
                    local_scores,
                    segment_mask,
                    row,
                    3,
                    _ex(t2_b, 2, "task2_tuple_to_vqa_reward", [row], 0.0),
                )
                self._put_scalar_on_last_segment_token(
                    local_scores,
                    segment_mask,
                    row,
                    4,
                    _ex(t2_b, 2, "task2_vqa_to_feedback_reward", [row], 0.0),
                )
            t2_b.batch["task2_token_level_scores"] = scores
            t2_b.batch["task2_local_token_level_scores"] = local_scores
            t2_b.meta_info["task2_token_level_scores"] = scores
            t2_b.meta_info["task2_local_token_level_scores"] = local_scores

        for t3_b in task3_batches:
            if t3_b is None or "task3_response_mask" not in t3_b.batch:
                continue
            mask = t3_b.batch["task3_response_mask"]
            scores = torch.zeros_like(mask, dtype=torch.float32)
            if getattr(t3_b, "meta_info", None) is None:
                t3_b.meta_info = {}
            task3_extras = t3_b.meta_info.setdefault("task3_reward_extra_info", {})
            outcome_vals = self._ensure_reward_extra_list(task3_extras, "task3_outcome_score", len(t3_b))
            for row in range(len(t3_b)):
                value = self._mean_return(outcomes_by_event, t3_b, row, "step5", 0.0)
                outcome_vals[row] = value
                self._put_scalar_on_last_mask_token(scores, mask, row, value)
            t3_b.batch["task3_token_level_scores"] = scores
            t3_b.meta_info["task3_token_level_scores"] = scores

    def _attach_phase1_mdp_final_image_outcome_token_scores(
        self,
        all_tids,
        task1_batch: Optional[DataProto],
        task2_batches: list[DataProto],
        task3_batches: list[DataProto],
    ) -> None:
        """Use the final generated image score as the whole-path outcome.

        The final image is the last valid image generated on a terminal path:
        task1's initial image if no edit image was generated, otherwise the
        latest task3 edited image. A no-edit task2 turn does not create a new
        image, so it inherits the previous image as the final image.
        """
        algo_cfg = getattr(getattr(self, "config", None), "algorithm", None) or {}
        _get_hp = algo_cfg.get if hasattr(algo_cfg, "get") else lambda k, d: getattr(algo_cfg, k, d)
        eps = float(_get_hp("mdp_score_eps", 1e-6))

        _ex = self._get_outcome_extra_scalar
        outcomes_by_event: dict[tuple[int, int, str], list[float]] = {}

        def _append_outcome(dp: DataProto, row: int, event_name: str, value: float) -> None:
            outcomes_by_event.setdefault(self._event_return_key(dp, row, event_name), []).append(float(value))

        def _normalized_image_score(dp: DataProto, task_id: int, row: int) -> float:
            score_field = f"task{task_id}_image_score"
            max_field = f"task{task_id}_image_score_max"
            align_field = f"task{task_id}_align"
            score = _ex(dp, task_id, score_field, [row], default=_ex(dp, task_id, align_field, [row], 0.0))
            s_max = _ex(dp, task_id, max_field, [row], 1.0)
            return max(0.0, min(2.0, 2.0 * float(score) / max(float(s_max), eps)))

        # task1 lookup ---------------------------------------------------------
        t1_index: dict[int, int] = {}
        if task1_batch is not None and "trajectory_id" in task1_batch.non_tensor_batch:
            for i, tid in enumerate(task1_batch.non_tensor_batch["trajectory_id"]):
                t1_index.setdefault(int(tid), i)

        # task2/task3 rows grouped by shared prefix vs branch path -------------
        branch_parent: dict[int, int] = {}
        branch_tid: dict[int, int] = {}
        shared_t2_rows: dict[int, list[tuple]] = {}
        shared_t3_rows: dict[int, list[tuple]] = {}
        branch_t2_rows: dict[int, list[tuple]] = {}
        branch_t3_rows: dict[int, list[tuple]] = {}

        for turn_i, t2_b in enumerate(task2_batches):
            if t2_b is None or "trajectory_id" not in t2_b.non_tensor_batch:
                continue
            self._ensure_branch_metadata(t2_b)
            tids = np.asarray(t2_b.non_tensor_batch["trajectory_id"], dtype=np.int64)
            bids = np.asarray(t2_b.non_tensor_batch["branch_id"], dtype=np.int64)
            parents = np.asarray(t2_b.non_tensor_batch["parent_branch_id"], dtype=np.int64)
            for row in range(len(t2_b)):
                tid = int(tids[row])
                bid = int(bids[row])
                item = (turn_i, t2_b, row)
                if bid >= 0:
                    branch_parent.setdefault(bid, int(parents[row]))
                    branch_tid.setdefault(bid, tid)
                    branch_t2_rows.setdefault(bid, []).append(item)
                else:
                    shared_t2_rows.setdefault(tid, []).append(item)

        for turn_i, t3_b in enumerate(task3_batches):
            if t3_b is None or "trajectory_id" not in t3_b.non_tensor_batch:
                continue
            self._ensure_branch_metadata(t3_b)
            tids = np.asarray(t3_b.non_tensor_batch["trajectory_id"], dtype=np.int64)
            bids = np.asarray(t3_b.non_tensor_batch["branch_id"], dtype=np.int64)
            parents = np.asarray(t3_b.non_tensor_batch["parent_branch_id"], dtype=np.int64)
            for row in range(len(t3_b)):
                tid = int(tids[row])
                bid = int(bids[row])
                item = (turn_i, t3_b, row)
                if bid >= 0:
                    branch_parent.setdefault(bid, int(parents[row]))
                    branch_tid.setdefault(bid, tid)
                    branch_t3_rows.setdefault(bid, []).append(item)
                else:
                    shared_t3_rows.setdefault(tid, []).append(item)

        all_branch_ids = set(branch_parent.keys())
        branch_children: dict[int, list[int]] = {}
        for bid, parent in branch_parent.items():
            if parent >= 0:
                branch_children.setdefault(parent, []).append(bid)
        terminal_branch_ids = sorted(all_branch_ids - set(branch_children.keys()))

        def _dedupe_sorted(rows: list[tuple]) -> list[tuple]:
            rows = sorted(rows, key=lambda x: (x[0], x[2]))
            out = []
            seen = set()
            for turn_i, dp, row in rows:
                key = (turn_i, id(dp), row)
                if key in seen:
                    continue
                seen.add(key)
                out.append((turn_i, dp, row))
            return out

        def _branch_chain(bid: int) -> list[int]:
            chain = []
            cur = bid
            while cur >= 0:
                chain.append(cur)
                cur = int(branch_parent.get(cur, -1))
            return list(reversed(chain))

        def _collect_branch_rows(bid: int) -> tuple[list[tuple], list[tuple]]:
            tid = int(branch_tid[bid])
            t2_rows = list(shared_t2_rows.get(tid, []))
            t3_rows = list(shared_t3_rows.get(tid, []))
            for seg_bid in _branch_chain(bid):
                t2_rows.extend(branch_t2_rows.get(seg_bid, []))
                t3_rows.extend(branch_t3_rows.get(seg_bid, []))
            return _dedupe_sorted(t2_rows), _dedupe_sorted(t3_rows)

        def _build_events(tid: int, t2_rows: list[tuple], t3_rows: list[tuple]) -> list[tuple]:
            events: list[tuple[float, DataProto, int, str, Optional[float]]] = []
            if task1_batch is not None and tid in t1_index:
                row = t1_index[tid]
                events.append((0.0, task1_batch, row, "step1", _normalized_image_score(task1_batch, 1, row)))

            for turn_i, t2_b, row in t2_rows:
                base = 1.0 + 4.0 * float(turn_i)
                events.append((base + 0.0, t2_b, row, "step2", None))
                events.append((base + 1.0, t2_b, row, "step3", None))
                events.append((base + 2.0, t2_b, row, "step4", None))

            for turn_i, t3_b, row in t3_rows:
                base = 1.0 + 4.0 * float(turn_i)
                events.append((base + 3.0, t3_b, row, "step5", _normalized_image_score(t3_b, 3, row)))

            return sorted(events, key=lambda x: x[0])

        def _accumulate_final_image_outcome(events: list[tuple]) -> None:
            if not events:
                return
            image_events = [(pos, reward) for pos, _, _, _, reward in events if reward is not None]
            if not image_events:
                return
            _, final_image_reward = max(image_events, key=lambda x: x[0])
            for _, dp, row, event_name, _ in events:
                _append_outcome(dp, row, event_name, float(final_image_reward))

        for bid in terminal_branch_ids:
            tid = int(branch_tid[bid])
            t2_rows, t3_rows = _collect_branch_rows(bid)
            _accumulate_final_image_outcome(_build_events(tid, t2_rows, t3_rows))

        tids_with_branch = {int(branch_tid[bid]) for bid in all_branch_ids}
        for raw_tid in all_tids if all_tids is not None else []:
            tid = int(raw_tid)
            if tid in tids_with_branch:
                continue
            t2_rows = _dedupe_sorted(shared_t2_rows.get(tid, []))
            t3_rows = _dedupe_sorted(shared_t3_rows.get(tid, []))
            _accumulate_final_image_outcome(_build_events(tid, t2_rows, t3_rows))

        # Write final-image outcome rewards back to task token-level scores ----
        if task1_batch is not None and "task1_response_mask" in task1_batch.batch:
            scores = torch.zeros_like(task1_batch.batch["task1_response_mask"], dtype=torch.float32)
            if getattr(task1_batch, "meta_info", None) is None:
                task1_batch.meta_info = {}
            task1_extras = task1_batch.meta_info.setdefault("task1_reward_extra_info", {})
            outcome_vals = self._ensure_reward_extra_list(
                task1_extras, "task1_final_image_outcome_score", len(task1_batch)
            )
            for tid, row in t1_index.items():
                value = self._mean_return(outcomes_by_event, task1_batch, row, "step1", 0.0)
                outcome_vals[row] = value
                self._put_scalar_on_last_mask_token(scores, task1_batch.batch["task1_response_mask"], row, value)
            task1_batch.batch["task1_token_level_scores"] = scores
            task1_batch.meta_info["task1_token_level_scores"] = scores

        for t2_b in task2_batches:
            if t2_b is None or "task2_response_mask" not in t2_b.batch:
                continue
            response_mask = t2_b.batch["task2_response_mask"]
            segment_mask = t2_b.batch.get("task2_segment_mask", response_mask)
            scores = torch.zeros_like(response_mask, dtype=torch.float32)
            local_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            if getattr(t2_b, "meta_info", None) is None:
                t2_b.meta_info = {}
            task2_extras = t2_b.meta_info.setdefault("task2_reward_extra_info", {})
            step2_vals = self._ensure_reward_extra_list(task2_extras, "task2_step2_final_image_outcome_score", len(t2_b))
            step3_vals = self._ensure_reward_extra_list(task2_extras, "task2_step3_final_image_outcome_score", len(t2_b))
            step4_vals = self._ensure_reward_extra_list(task2_extras, "task2_step4_final_image_outcome_score", len(t2_b))
            for row in range(len(t2_b)):
                step2 = self._mean_return(outcomes_by_event, t2_b, row, "step2", 0.0)
                step3 = self._mean_return(outcomes_by_event, t2_b, row, "step3", 0.0)
                step4 = self._mean_return(outcomes_by_event, t2_b, row, "step4", 0.0)
                step2_vals[row] = step2
                step3_vals[row] = step3
                step4_vals[row] = step4
                self._put_scalar_on_last_segment_token(scores, segment_mask, row, 2, step2)
                self._put_scalar_on_last_segment_token(scores, segment_mask, row, 3, step3)
                self._put_scalar_on_last_segment_token(scores, segment_mask, row, 4, step4)
                self._put_scalar_on_last_segment_token(
                    local_scores,
                    segment_mask,
                    row,
                    2,
                    _ex(t2_b, 2, "task2_prompt_to_tuple_reward", [row], 0.0),
                )
                self._put_scalar_on_last_segment_token(
                    local_scores,
                    segment_mask,
                    row,
                    3,
                    _ex(t2_b, 2, "task2_tuple_to_vqa_reward", [row], 0.0),
                )
                self._put_scalar_on_last_segment_token(
                    local_scores,
                    segment_mask,
                    row,
                    4,
                    _ex(t2_b, 2, "task2_vqa_to_feedback_reward", [row], 0.0),
                )
            t2_b.batch["task2_token_level_scores"] = scores
            t2_b.batch["task2_local_token_level_scores"] = local_scores
            t2_b.meta_info["task2_token_level_scores"] = scores
            t2_b.meta_info["task2_local_token_level_scores"] = local_scores

        for t3_b in task3_batches:
            if t3_b is None or "task3_response_mask" not in t3_b.batch:
                continue
            mask = t3_b.batch["task3_response_mask"]
            scores = torch.zeros_like(mask, dtype=torch.float32)
            if getattr(t3_b, "meta_info", None) is None:
                t3_b.meta_info = {}
            task3_extras = t3_b.meta_info.setdefault("task3_reward_extra_info", {})
            outcome_vals = self._ensure_reward_extra_list(
                task3_extras, "task3_final_image_outcome_score", len(t3_b)
            )
            for row in range(len(t3_b)):
                value = self._mean_return(outcomes_by_event, t3_b, row, "step5", 0.0)
                outcome_vals[row] = value
                self._put_scalar_on_last_mask_token(scores, mask, row, value)
            t3_b.batch["task3_token_level_scores"] = scores
            t3_b.meta_info["task3_token_level_scores"] = scores

    def _attach_phase1_mdp_discounted_token_scores(
        self,
        all_tids,
        task1_batch: Optional[DataProto],
        task2_batches: list[DataProto],
        task3_batches: list[DataProto],
    ) -> None:
        """Replace phase1 local token scores with discounted MDP returns.

        The token-score tensors remain task-shaped:
          * task1: one return on the final response token.
          * task2: step2/3/4 returns on the last token of segment masks 2/3/4.
          * task3: step5 return on the final response token.
        Shared-prefix rows that feed multiple terminal branches receive the
        mean return over descendant branches, mirroring existing outcome logic.
        """
        algo_cfg = getattr(getattr(self, "config", None), "algorithm", None) or {}
        _get_hp = algo_cfg.get if hasattr(algo_cfg, "get") else lambda k, d: getattr(algo_cfg, k, d)
        gamma = float(_get_hp("mdp_gamma", _get_hp("gamma", 1.0)))
        eps = float(_get_hp("mdp_score_eps", 1e-6))

        _ex = self._get_outcome_extra_scalar
        returns_by_event: dict[tuple[int, int, str], list[float]] = {}

        def _append_return(dp: DataProto, row: int, event_name: str, value: float) -> None:
            returns_by_event.setdefault(self._event_return_key(dp, row, event_name), []).append(float(value))

        # task1 lookup ---------------------------------------------------------
        t1_index: dict[int, int] = {}
        if task1_batch is not None and "trajectory_id" in task1_batch.non_tensor_batch:
            for i, tid in enumerate(task1_batch.non_tensor_batch["trajectory_id"]):
                t1_index.setdefault(int(tid), i)

        # task2/task3 rows grouped by shared prefix vs branch path -------------
        branch_parent: dict[int, int] = {}
        branch_tid: dict[int, int] = {}
        shared_t2_rows: dict[int, list[tuple]] = {}
        shared_t3_rows: dict[int, list[tuple]] = {}
        branch_t2_rows: dict[int, list[tuple]] = {}
        branch_t3_rows: dict[int, list[tuple]] = {}

        for turn_i, t2_b in enumerate(task2_batches):
            if t2_b is None or "trajectory_id" not in t2_b.non_tensor_batch:
                continue
            self._ensure_branch_metadata(t2_b)
            tids = np.asarray(t2_b.non_tensor_batch["trajectory_id"], dtype=np.int64)
            bids = np.asarray(t2_b.non_tensor_batch["branch_id"], dtype=np.int64)
            parents = np.asarray(t2_b.non_tensor_batch["parent_branch_id"], dtype=np.int64)
            for row in range(len(t2_b)):
                tid = int(tids[row])
                bid = int(bids[row])
                item = (turn_i, t2_b, row)
                if bid >= 0:
                    branch_parent.setdefault(bid, int(parents[row]))
                    branch_tid.setdefault(bid, tid)
                    branch_t2_rows.setdefault(bid, []).append(item)
                else:
                    shared_t2_rows.setdefault(tid, []).append(item)

        for turn_i, t3_b in enumerate(task3_batches):
            if t3_b is None or "trajectory_id" not in t3_b.non_tensor_batch:
                continue
            self._ensure_branch_metadata(t3_b)
            tids = np.asarray(t3_b.non_tensor_batch["trajectory_id"], dtype=np.int64)
            bids = np.asarray(t3_b.non_tensor_batch["branch_id"], dtype=np.int64)
            parents = np.asarray(t3_b.non_tensor_batch["parent_branch_id"], dtype=np.int64)
            for row in range(len(t3_b)):
                tid = int(tids[row])
                bid = int(bids[row])
                item = (turn_i, t3_b, row)
                if bid >= 0:
                    branch_parent.setdefault(bid, int(parents[row]))
                    branch_tid.setdefault(bid, tid)
                    branch_t3_rows.setdefault(bid, []).append(item)
                else:
                    shared_t3_rows.setdefault(tid, []).append(item)

        all_branch_ids = set(branch_parent.keys())
        branch_children: dict[int, list[int]] = {}
        for bid, parent in branch_parent.items():
            if parent >= 0:
                branch_children.setdefault(parent, []).append(bid)
        terminal_branch_ids = sorted(all_branch_ids - set(branch_children.keys()))

        def _dedupe_sorted(rows: list[tuple]) -> list[tuple]:
            rows = sorted(rows, key=lambda x: (x[0], x[2]))
            out = []
            seen = set()
            for turn_i, dp, row in rows:
                key = (turn_i, id(dp), row)
                if key in seen:
                    continue
                seen.add(key)
                out.append((turn_i, dp, row))
            return out

        def _branch_chain(bid: int) -> list[int]:
            chain = []
            cur = bid
            while cur >= 0:
                chain.append(cur)
                cur = int(branch_parent.get(cur, -1))
            return list(reversed(chain))

        def _collect_branch_rows(bid: int) -> tuple[list[tuple], list[tuple]]:
            tid = int(branch_tid[bid])
            t2_rows = list(shared_t2_rows.get(tid, []))
            t3_rows = list(shared_t3_rows.get(tid, []))
            for seg_bid in _branch_chain(bid):
                t2_rows.extend(branch_t2_rows.get(seg_bid, []))
                t3_rows.extend(branch_t3_rows.get(seg_bid, []))
            return _dedupe_sorted(t2_rows), _dedupe_sorted(t3_rows)

        def _build_events(tid: int, t2_rows: list[tuple], t3_rows: list[tuple]) -> list[tuple]:
            events: list[tuple[float, DataProto, int, str, float]] = []
            if task1_batch is not None and tid in t1_index:
                row = t1_index[tid]
                image_score = _ex(
                    task1_batch,
                    1,
                    "task1_image_score",
                    [row],
                    default=_ex(task1_batch, 1, "task1_mdp_reward", [row], 0.0),
                )
                image_score_max = _ex(task1_batch, 1, "task1_image_score_max", [row], 1.0)
                reward = 2.0 * float(image_score) / max(float(image_score_max), eps)
                events.append((0.0, task1_batch, row, "step1", reward))

            for turn_i, t2_b, row in t2_rows:
                base = 1.0 + 4.0 * float(turn_i)
                events.append((base + 0.0, t2_b, row, "step2", _ex(t2_b, 2, "task2_prompt_to_tuple_reward", [row], 0.0)))
                events.append((base + 1.0, t2_b, row, "step3", _ex(t2_b, 2, "task2_tuple_to_vqa_reward", [row], 0.0)))
                events.append((base + 2.0, t2_b, row, "step4", _ex(t2_b, 2, "task2_vqa_to_feedback_reward", [row], 0.0)))

            for turn_i, t3_b, row in t3_rows:
                base = 1.0 + 4.0 * float(turn_i)
                events.append((base + 3.0, t3_b, row, "step5", _ex(t3_b, 3, "task3_step5_reward", [row], 0.0)))

            return sorted(events, key=lambda x: x[0])

        def _accumulate_discounted_returns(events: list[tuple]) -> None:
            running_sum = 0.0
            running_weight = 0.0
            for _, dp, row, event_name, reward in reversed(events):
                running_sum = float(reward) + gamma * running_sum
                running_weight = 1.0 + gamma * running_weight
                _append_return(dp, row, event_name, running_sum / max(running_weight, eps))

        for bid in terminal_branch_ids:
            tid = int(branch_tid[bid])
            t2_rows, t3_rows = _collect_branch_rows(bid)
            _accumulate_discounted_returns(_build_events(tid, t2_rows, t3_rows))

        tids_with_branch = {int(branch_tid[bid]) for bid in all_branch_ids}
        for raw_tid in all_tids if all_tids is not None else []:
            tid = int(raw_tid)
            if tid in tids_with_branch:
                continue
            t2_rows = _dedupe_sorted(shared_t2_rows.get(tid, []))
            t3_rows = _dedupe_sorted(shared_t3_rows.get(tid, []))
            _accumulate_discounted_returns(_build_events(tid, t2_rows, t3_rows))

        # Write discounted returns back to task token-level score tensors -------
        if task1_batch is not None and "task1_response_mask" in task1_batch.batch:
            scores = torch.zeros_like(task1_batch.batch["task1_response_mask"], dtype=torch.float32)
            for tid, row in t1_index.items():
                value = self._mean_return(returns_by_event, task1_batch, row, "step1", 0.0)
                self._put_scalar_on_last_mask_token(scores, task1_batch.batch["task1_response_mask"], row, value)
            task1_batch.batch["task1_token_level_scores"] = scores
            if getattr(task1_batch, "meta_info", None) is None:
                task1_batch.meta_info = {}
            task1_batch.meta_info["task1_token_level_scores"] = scores

        for t2_b in task2_batches:
            if t2_b is None or "task2_response_mask" not in t2_b.batch:
                continue
            response_mask = t2_b.batch["task2_response_mask"]
            segment_mask = t2_b.batch.get("task2_segment_mask", response_mask)
            scores = torch.zeros_like(response_mask, dtype=torch.float32)
            local_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            if getattr(t2_b, "meta_info", None) is None:
                t2_b.meta_info = {}
            task2_extras = t2_b.meta_info.setdefault("task2_reward_extra_info", {})
            step2_return_vals = self._ensure_reward_extra_list(task2_extras, "task2_step2_return_score", len(t2_b))
            step3_return_vals = self._ensure_reward_extra_list(task2_extras, "task2_step3_return_score", len(t2_b))
            step4_return_vals = self._ensure_reward_extra_list(task2_extras, "task2_step4_return_score", len(t2_b))
            for row in range(len(t2_b)):
                step2_return = self._mean_return(returns_by_event, t2_b, row, "step2", 0.0)
                step3_return = self._mean_return(returns_by_event, t2_b, row, "step3", 0.0)
                step4_return = self._mean_return(returns_by_event, t2_b, row, "step4", 0.0)
                step2_return_vals[row] = step2_return
                step3_return_vals[row] = step3_return
                step4_return_vals[row] = step4_return
                self._put_scalar_on_last_segment_token(scores, segment_mask, row, 2, step2_return)
                self._put_scalar_on_last_segment_token(scores, segment_mask, row, 3, step3_return)
                self._put_scalar_on_last_segment_token(scores, segment_mask, row, 4, step4_return)
                self._put_scalar_on_last_segment_token(
                    local_scores,
                    segment_mask,
                    row,
                    2,
                    _ex(t2_b, 2, "task2_prompt_to_tuple_reward", [row], 0.0),
                )
                self._put_scalar_on_last_segment_token(
                    local_scores,
                    segment_mask,
                    row,
                    3,
                    _ex(t2_b, 2, "task2_tuple_to_vqa_reward", [row], 0.0),
                )
                self._put_scalar_on_last_segment_token(
                    local_scores,
                    segment_mask,
                    row,
                    4,
                    _ex(t2_b, 2, "task2_vqa_to_feedback_reward", [row], 0.0),
                )
            t2_b.batch["task2_token_level_scores"] = scores
            t2_b.batch["task2_local_token_level_scores"] = local_scores
            t2_b.meta_info["task2_token_level_scores"] = scores
            t2_b.meta_info["task2_local_token_level_scores"] = local_scores

        for t3_b in task3_batches:
            if t3_b is None or "task3_response_mask" not in t3_b.batch:
                continue
            mask = t3_b.batch["task3_response_mask"]
            scores = torch.zeros_like(mask, dtype=torch.float32)
            for row in range(len(t3_b)):
                value = self._mean_return(returns_by_event, t3_b, row, "step5", 0.0)
                self._put_scalar_on_last_mask_token(scores, mask, row, value)
            t3_b.batch["task3_token_level_scores"] = scores
            if getattr(t3_b, "meta_info", None) is None:
                t3_b.meta_info = {}
            t3_b.meta_info["task3_token_level_scores"] = scores

    @staticmethod
    def _inject_task2_decision_signal(
        target_batch: Optional[DataProto],
        reference_batch: Optional[DataProto],
    ) -> Optional[DataProto]:
        """Inject row-wise task2 decision targets into `target_batch.meta_info["extra_info"]`.

        Decision target:
          * Prefer previous-image `task3_vqa_reward` when available.
          * Otherwise fall back to `task1_vqa_reward` (turn 0 case).
        """
        if target_batch is None or reference_batch is None:
            return target_batch
        n = len(target_batch)
        if n == 0:
            return target_batch

        ref_meta = getattr(reference_batch, "meta_info", None) or {}
        task3_extras = ref_meta.get("task3_reward_extra_info", {}) or {}
        task1_extras = ref_meta.get("task1_reward_extra_info", {}) or {}
        task3_vals = task3_extras.get("task3_vqa_reward")
        task1_vals = task1_extras.get("task1_vqa_reward")

        decision_vals: list[float] = []
        decision_srcs: list[str] = []
        for i in range(n):
            use_task3 = (
                isinstance(task3_vals, list)
                and i < len(task3_vals)
                and task3_vals[i] is not None
            )
            use_task1 = (
                isinstance(task1_vals, list)
                and i < len(task1_vals)
                and task1_vals[i] is not None
            )
            if use_task3:
                decision_vals.append(float(task3_vals[i]))
                decision_srcs.append("task3")
            elif use_task1:
                decision_vals.append(float(task1_vals[i]))
                decision_srcs.append("task1")
            else:
                decision_vals.append(0.0)
                decision_srcs.append("missing")

        if getattr(target_batch, "meta_info", None) is None:
            target_batch.meta_info = {}
        extra_info = dict(target_batch.meta_info.get("extra_info", {}) or {})
        extra_info["task2_decision_vqa_reward"] = decision_vals
        extra_info["task2_decision_vqa_source"] = decision_srcs
        target_batch.meta_info["extra_info"] = extra_info
        return target_batch

    @staticmethod
    def _mark_task2_raw_stage_only(target_batch: Optional[DataProto]) -> Optional[DataProto]:
        if target_batch is None:
            return target_batch
        if getattr(target_batch, "meta_info", None) is None:
            target_batch.meta_info = {}
        extra_info = dict(target_batch.meta_info.get("extra_info", {}) or {})
        extra_info["task2_raw_stage_only"] = [True] * len(target_batch)
        target_batch.meta_info["extra_info"] = extra_info
        return target_batch

    @staticmethod
    def _build_terminal_reward_tensor(response_mask: torch.Tensor, rewards: list[float]) -> torch.Tensor:
        reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
        for i, reward in enumerate(rewards):
            valid_response_length = int(response_mask[i].sum().item())
            if valid_response_length <= 0:
                continue
            reward_tensor[i, valid_response_length - 1] = float(reward)
        return reward_tensor

    def _finalize_task2_reward_batch(self, batch: Optional[DataProto]) -> Optional[DataProto]:
        if batch is None or getattr(batch, "batch", None) is None:
            return batch
        raw_extras = (getattr(batch, "meta_info", None) or {}).get("task2_reward_extra_info_raw", {})
        if not raw_extras:
            return batch

        from recipe.image_rl.reward_function_fine_grained import finalize_task2_reward_extra_info

        extra_info = dict((getattr(batch, "meta_info", None) or {}).get("extra_info", {}) or {})
        decision_vals = extra_info.get("task2_decision_vqa_reward", [0.0] * len(batch))
        decision_srcs = extra_info.get("task2_decision_vqa_source", ["missing"] * len(batch))

        finalized_extras: dict[str, list[Any]] = {}
        rewards: list[float] = []
        for row in range(len(batch)):
            row_stage = {
                key: vals[row]
                for key, vals in raw_extras.items()
                if isinstance(vals, list) and row < len(vals)
            }
            decision_vqa = float(decision_vals[row]) if row < len(decision_vals) and decision_vals[row] is not None else 0.0
            decision_source = decision_srcs[row] if row < len(decision_srcs) and decision_srcs[row] is not None else "missing"
            vlm_reward, finalized_row = finalize_task2_reward_extra_info(
                row_stage,
                decision_vqa=decision_vqa,
                decision_source=decision_source,
            )
            rewards.append(float(finalized_row.get("task2_total_reward", vlm_reward)))
            for key, value in finalized_row.items():
                finalized_extras.setdefault(key, []).append(value)

        reward_tensor = self._build_terminal_reward_tensor(batch.batch["task2_response_mask"], rewards)
        if getattr(batch, "meta_info", None) is None:
            batch.meta_info = {}
        batch.meta_info["task2_token_level_scores"] = reward_tensor
        batch.meta_info["task2_reward_extra_info"] = finalized_extras
        batch.batch["task2_token_level_scores"] = reward_tensor
        return batch

    async def _await_and_finalize_task2_entries(
        self,
        entries: list[tuple[DataProto, asyncio.Task]],
        pre_task2_batches_per_turn: list[DataProto],
        task2_batches_per_turn: list[DataProto],
    ) -> None:
        if not entries:
            return
        results = await asyncio.gather(
            *[reward_task for _, reward_task in entries],
            return_exceptions=True,
        )
        for (batch_ref, _reward_task), result in zip(entries, results):
            if isinstance(result, Exception):
                logger.warning(f"[AgentLoop] task2 raw reward await failed: {result}")
                continue
            if result is None:
                continue
            _reward_tensor, reward_extras = result
            if getattr(batch_ref, "meta_info", None) is None:
                batch_ref.meta_info = {}
            batch_ref.meta_info["task2_reward_extra_info_raw"] = reward_extras

        for pre_b, t2_b in zip(pre_task2_batches_per_turn, task2_batches_per_turn):
            self._inject_task2_decision_signal(t2_b, pre_b)
            self._finalize_task2_reward_batch(t2_b)

    def _compute_outcomes_with_avg(
        self,
        all_tids,
        task1_batch: Optional[DataProto],
        task2_batches: list[DataProto],
        task3_batches: list[DataProto],
    ) -> tuple[dict, dict]:
        """Compute shared-prefix and branch-specific outcomes.

        Semantics:
          * branch_id == -1 rows (shared prefix) receive `tid_outcomes[tid]`
            = mean over descendant terminal branch outcomes.
          * branch_id >= 0 rows (post-branch path) receive
            `branch_outcomes[branch_id]` = mean over descendant terminal
            outcomes of that branch segment. For terminal branches, this is the
            branch's own final return.

        Terminal branches may end at:
          * final task2 row (no-edit)
          * final task3 row
        """
        # Hyperparameters -------------------------------------------------------
        algo_cfg = getattr(getattr(self, "config", None), "algorithm", None) or {}
        _get_hp = algo_cfg.get if hasattr(algo_cfg, "get") else lambda k, d: getattr(algo_cfg, k, d)
        eta     = float(_get_hp("outcome_eta",    0.4))
        beta    = float(_get_hp("outcome_beta",   0.1))
        lambda_ = float(_get_hp("outcome_lambda", 0.15))

        tid_outcomes: dict = {}
        branch_outcomes: dict = {}

        # --- helper: find score tensor wherever the reward callback stored it --
        def _get_score_tensor(dp: DataProto, key: str):
            """Return per-row score tensor: batch.batch → meta_info → non_tensor_batch."""
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

        # --- helper: scale ref_tensor so its sum equals `scalar` --------------
        def _make_outcome_tensor(ref: torch.Tensor, scalar: float) -> torch.Tensor:
            ref_f = ref.float()
            ref_sum = ref_f.sum()
            if ref_sum.abs() > 1e-8:
                return (ref_f * (scalar / ref_sum)).float()
            out = torch.zeros_like(ref_f)
            if out.numel() > 0:
                out[-1] = scalar
            return out

        # --- build per-turn indices --------------------------------------------
        # task3: list[(batch_ref, scores_tensor, tid→rows, branch→rows)]
        t3_index: list[tuple] = []
        for t3_b in task3_batches:
            if t3_b is None:
                continue
            self._ensure_branch_metadata(t3_b)
            scores = _get_score_tensor(t3_b, "task3_token_level_scores")
            if scores is None:
                continue
            if "trajectory_id" not in t3_b.non_tensor_batch:
                continue
            arr = t3_b.non_tensor_batch["trajectory_id"]
            bids = t3_b.non_tensor_batch["branch_id"]
            tid_to_rows: dict[int, list[int]] = {}
            bid_to_rows: dict[int, list[int]] = {}
            for i in range(len(arr)):
                tid_to_rows.setdefault(int(arr[i]), []).append(i)
                bid = int(bids[i])
                if bid >= 0:
                    bid_to_rows.setdefault(bid, []).append(i)
            t3_index.append((t3_b, scores, tid_to_rows, bid_to_rows))

        # task2: list[(batch_ref, scores_tensor, tid→rows, branch→rows)]
        t2_index: list[tuple] = []
        for t2_b in task2_batches:
            if t2_b is None:
                continue
            self._ensure_branch_metadata(t2_b)
            if "trajectory_id" not in t2_b.non_tensor_batch:
                continue
            scores = _get_score_tensor(t2_b, "task2_token_level_scores")
            arr = t2_b.non_tensor_batch["trajectory_id"]
            bids = t2_b.non_tensor_batch["branch_id"]
            tid_to_rows: dict[int, list[int]] = {}
            bid_to_rows: dict[int, list[int]] = {}
            for i in range(len(arr)):
                tid_to_rows.setdefault(int(arr[i]), []).append(i)
                bid = int(bids[i])
                if bid >= 0:
                    bid_to_rows.setdefault(bid, []).append(i)
            t2_index.append((t2_b, scores, tid_to_rows, bid_to_rows))

        # task1: single lookup (turn 0 only)
        t1_index: Optional[dict] = None
        t1_scores_tensor = None
        if task1_batch is not None and "trajectory_id" in task1_batch.non_tensor_batch:
            t1_scores_tensor = _get_score_tensor(task1_batch, "task1_token_level_scores")
            if t1_scores_tensor is not None:
                arr = task1_batch.non_tensor_batch["trajectory_id"]
                t1_index = {int(arr[i]): i for i in range(len(arr))}

        # --- per-trajectory / per-branch bookkeeping ---------------------------
        _ex = self._get_outcome_extra_scalar

        branch_parent: dict[int, int] = {}
        branch_tid: dict[int, int] = {}
        shared_t2_rows: dict[int, list[tuple]] = {}
        shared_t3_rows: dict[int, list[tuple]] = {}
        branch_t2_rows: dict[int, list[tuple]] = {}
        branch_t3_rows: dict[int, list[tuple]] = {}

        for turn_i, (t2_b, t2_scores, tid_to_rows, bid_to_rows) in enumerate(t2_index):
            tids = np.asarray(t2_b.non_tensor_batch["trajectory_id"], dtype=np.int64)
            bids = np.asarray(t2_b.non_tensor_batch["branch_id"], dtype=np.int64)
            parents = np.asarray(t2_b.non_tensor_batch["parent_branch_id"], dtype=np.int64)
            for row in range(len(t2_b)):
                tid = int(tids[row])
                bid = int(bids[row])
                if bid >= 0:
                    branch_parent.setdefault(bid, int(parents[row]))
                    branch_tid.setdefault(bid, tid)
                    branch_t2_rows.setdefault(bid, []).append((turn_i, t2_b, t2_scores, row))
                else:
                    shared_t2_rows.setdefault(tid, []).append((turn_i, t2_b, t2_scores, row))

        for turn_i, (t3_b, t3_scores, tid_to_rows, bid_to_rows) in enumerate(t3_index):
            tids = np.asarray(t3_b.non_tensor_batch["trajectory_id"], dtype=np.int64)
            bids = np.asarray(t3_b.non_tensor_batch["branch_id"], dtype=np.int64)
            parents = np.asarray(t3_b.non_tensor_batch["parent_branch_id"], dtype=np.int64)
            for row in range(len(t3_b)):
                tid = int(tids[row])
                bid = int(bids[row])
                if bid >= 0:
                    branch_parent.setdefault(bid, int(parents[row]))
                    branch_tid.setdefault(bid, tid)
                    branch_t3_rows.setdefault(bid, []).append((turn_i, t3_b, t3_scores, row))
                else:
                    shared_t3_rows.setdefault(tid, []).append((turn_i, t3_b, t3_scores, row))

        all_branch_ids = set(branch_parent.keys())
        branch_children: dict[int, list[int]] = {}
        for bid, parent in branch_parent.items():
            if parent >= 0:
                branch_children.setdefault(parent, []).append(bid)
        terminal_branch_ids = sorted(all_branch_ids - set(branch_children.keys()))

        def _dedupe_sorted(rows: list[tuple]) -> list[tuple]:
            rows = sorted(rows, key=lambda x: (x[0], x[3]))
            out = []
            seen = set()
            for turn_i, dp, scores, row in rows:
                key = (turn_i, row, id(dp))
                if key in seen:
                    continue
                seen.add(key)
                out.append((turn_i, dp, scores, row))
            return out

        def _branch_chain(bid: int) -> list[int]:
            chain = []
            cur = bid
            while cur >= 0:
                chain.append(cur)
                cur = int(branch_parent.get(cur, -1))
            return list(reversed(chain))

        def _collect_branch_rows(bid: int) -> tuple[list[tuple], list[tuple]]:
            tid = int(branch_tid[bid])
            t2_rows = list(shared_t2_rows.get(tid, []))
            t3_rows = list(shared_t3_rows.get(tid, []))
            for seg_bid in _branch_chain(bid):
                t2_rows.extend(branch_t2_rows.get(seg_bid, []))
                t3_rows.extend(branch_t3_rows.get(seg_bid, []))
            return _dedupe_sorted(t2_rows), _dedupe_sorted(t3_rows)

        terminal_branch_scalars: dict[int, float] = {}
        terminal_branch_entries: dict[int, tuple] = {}

        # Stats for logging
        log_A_T: list = []
        log_IF_bar: list = []
        log_Pbar: list = []
        log_T: list = []
        early_stop_count = 0

        for bid in terminal_branch_ids:
            tid = int(branch_tid[bid])
            A_1 = 0.0
            t1_ref: Optional[torch.Tensor] = None
            if t1_index is not None and t1_scores_tensor is not None:
                row1 = t1_index.get(tid)
                if row1 is not None:
                    A_1 = _ex(task1_batch, 1, "task1_align", [row1], 0.0)
                    t1_ref = t1_scores_tensor[row1].float()

            t2_rows, t3_rows = _collect_branch_rows(bid)
            P_vals: list[float] = []
            final_task2_ref: Optional[torch.Tensor] = None
            final_task2_turn = -1
            for turn_i, t2_b, t2_scores, row in t2_rows:
                P_vals.append(_ex(t2_b, 2, "task2_process", [row], 0.0))
                if t2_scores is not None and turn_i >= final_task2_turn:
                    final_task2_turn = turn_i
                    final_task2_ref = t2_scores[row].float()

            task3_events: list[tuple[int, float, float, torch.Tensor]] = []
            for turn_i, t3_b, t3_scores, row in t3_rows:
                task3_events.append(
                    (
                        turn_i,
                        _ex(t3_b, 3, "task3_align", [row], 0.0),
                        _ex(t3_b, 3, "task3_if", [row], 0.0),
                        t3_scores[row].float(),
                    )
                )
            task3_events.sort(key=lambda x: x[0])

            IF_vals: list[float] = []
            A_T = A_1
            final_task3_ref: Optional[torch.Tensor] = None
            for _, A_m, IF_m, ref_m in task3_events:
                IF_vals.append(float(IF_m))
                A_T = A_m
                final_task3_ref = ref_m

            IF_bar = float(np.mean(IF_vals)) if IF_vals else 0.0
            P_bar = float(np.mean(P_vals)) if P_vals else 0.0
            T = 1.0 + float(len(task3_events))
            outcome_scalar = A_T + eta * IF_bar + beta * P_bar - lambda_ * (T - 1.0)

            if final_task3_ref is not None:
                task_source = 3
            elif final_task2_ref is not None:
                task_source = 2
            elif t1_ref is not None:
                task_source = 1
            else:
                continue

            terminal_branch_scalars[bid] = outcome_scalar
            terminal_branch_entries[bid] = (
                outcome_scalar,
                task_source,
                {"A_T": A_T, "IF_bar": IF_bar, "P_bar": P_bar, "T": T},
            )

        def _mean_outcome_entries(entries: list[tuple], scalars: list[float]) -> tuple:
            mean_scalar = float(np.mean(scalars))
            return (
                mean_scalar,
                entries[0][1],
                {
                    "A_T": float(np.mean([e[2]["A_T"] for e in entries])),
                    "IF_bar": float(np.mean([e[2]["IF_bar"] for e in entries])),
                    "P_bar": float(np.mean([e[2]["P_bar"] for e in entries])),
                    "T": float(np.mean([e[2]["T"] for e in entries])),
                },
            )

        branch_outcome_scalars: dict[int, float] = {}

        def _compute_branch_outcome(bid: int) -> Optional[tuple]:
            if bid in branch_outcomes:
                return branch_outcomes[bid]
            children = sorted(branch_children.get(bid, []))
            if children:
                child_entries = []
                child_scalars = []
                for child in children:
                    entry = _compute_branch_outcome(child)
                    if entry is None:
                        continue
                    child_entries.append(entry)
                    child_scalars.append(branch_outcome_scalars[child])
                if not child_entries:
                    return None
                branch_outcomes[bid] = _mean_outcome_entries(child_entries, child_scalars)
                branch_outcome_scalars[bid] = float(np.mean(child_scalars))
                return branch_outcomes[bid]
            entry = terminal_branch_entries.get(bid)
            if entry is None:
                return None
            branch_outcomes[bid] = entry
            branch_outcome_scalars[bid] = terminal_branch_scalars[bid]
            return entry

        for bid in sorted(all_branch_ids):
            _compute_branch_outcome(bid)

        for raw_tid in all_tids:
            tid = int(raw_tid)
            root_branch_ids = sorted(
                [
                    bid for bid, bt in branch_tid.items()
                    if bt == tid and int(branch_parent.get(bid, -1)) < 0
                ]
            )
            if root_branch_ids:
                root_entries = []
                root_scalars = []
                for bid in root_branch_ids:
                    entry = _compute_branch_outcome(bid)
                    if entry is None:
                        continue
                    root_entries.append(entry)
                    root_scalars.append(branch_outcome_scalars[bid])
                if root_entries:
                    tid_outcomes[tid] = _mean_outcome_entries(root_entries, root_scalars)
            else:
                A_1 = 0.0
                t1_ref: Optional[torch.Tensor] = None
                if t1_index is not None and t1_scores_tensor is not None:
                    row1 = t1_index.get(tid)
                    if row1 is not None:
                        A_1 = _ex(task1_batch, 1, "task1_align", [row1], 0.0)
                        t1_ref = t1_scores_tensor[row1].float()

                shared_t2 = sorted(shared_t2_rows.get(tid, []), key=lambda x: (x[0], x[3]))
                shared_P = [
                    _ex(t2_b, 2, "task2_process", [row], 0.0)
                    for _, t2_b, _, row in shared_t2
                ]
                final_task2_ref: Optional[torch.Tensor] = None
                final_task2_feedback = None
                final_task2_turn = -1
                for turn_i, t2_b, t2_scores, row in shared_t2:
                    if t2_scores is not None and turn_i >= final_task2_turn:
                        final_task2_turn = turn_i
                        final_task2_ref = t2_scores[row].float()
                        feedback_texts = t2_b.non_tensor_batch.get("task2_feedback_texts", None)
                        if feedback_texts is not None and row < len(feedback_texts):
                            final_task2_feedback = feedback_texts[row]
                shared_t3 = sorted(
                    [
                        (
                            turn_i,
                            _ex(t3_b, 3, "task3_align", [row], 0.0),
                            _ex(t3_b, 3, "task3_if", [row], 0.0),
                            t3_scores[row].float(),
                        )
                        for turn_i, t3_b, t3_scores, row in shared_t3_rows.get(tid, [])
                    ],
                    key=lambda x: x[0],
                )

                IF_vals: list[float] = []
                A_T = A_1
                task_source = 1
                for _, A_m, IF_m, _ref_m in shared_t3:
                    IF_vals.append(float(IF_m))
                    A_T = A_m
                    task_source = 3
                IF_bar = float(np.mean(IF_vals)) if IF_vals else 0.0
                P_bar = float(np.mean(shared_P)) if shared_P else 0.0
                if not shared_t3 and final_task2_ref is not None and final_task2_feedback is not None and not _is_edit_sample(final_task2_feedback):
                    task_source = 2
                # T counts image states / generations:
                #   initial image (task1) + number of task3 edit generations.
                # This makes the turn penalty align with "how many times did we
                # actually regenerate an image?" rather than how many task2
                # reasoning turns we executed.
                T = 1.0 + float(len(shared_t3))
                if t1_ref is None and final_task2_ref is None and not shared_t3:
                    continue
                outcome_scalar = A_T + eta * IF_bar + beta * P_bar - lambda_ * (T - 1.0)
                tid_outcomes[tid] = (
                    outcome_scalar,
                    task_source,
                    {"A_T": A_T, "IF_bar": IF_bar, "P_bar": P_bar, "T": T},
                )

            comps = tid_outcomes[tid][2]
            log_A_T.append(comps["A_T"])
            log_IF_bar.append(comps["IF_bar"])
            log_Pbar.append(comps["P_bar"])
            log_T.append(comps["T"])
            if comps["T"] <= 1.0:
                early_stop_count += 1

        # Log aggregate stats
        if log_A_T:
            n = len(log_A_T)
            logger.info(
                "[OutcomeReward] n=%d | A_T=%.3f | IF_bar=%.3f | Pbar=%.3f | "
                "T=%.2f | early_stop=%.1f%% | branches=%d",
                n,
                float(np.mean(log_A_T)),
                float(np.mean(log_IF_bar)),
                float(np.mean(log_Pbar)),
                float(np.mean(log_T)),
                100.0 * early_stop_count / n,
                len(branch_outcomes),
            )

        return tid_outcomes, branch_outcomes

    def _attach_outcome_per_row(
        self,
        dp: DataProto,
        tid_outcomes: dict,
        branch_outcomes: Optional[dict] = None,
    ) -> DataProto:
        """Stamp `outcome_token_level_scores` + `outcome_task_id` per row.

        Lookup priority:
          1. If `branch_outcomes` is provided and branch_id >= 0, use
             branch_outcomes[branch_id]  (post-branch rows).
          2. Otherwise use tid_outcomes[trajectory_id]  (shared-prefix rows).

        Fills zeros for rows with no matching entry.
        """
        if dp is None or "trajectory_id" not in dp.non_tensor_batch:
            return dp

        self._ensure_branch_metadata(dp)
        tids    = dp.non_tensor_batch["trajectory_id"]
        bid_arr = dp.non_tensor_batch.get("branch_id", None)
        B = len(dp)

        # Outcome should be attached using the current task row's response
        # shape, matching local reward tensors (last valid token only).
        task_id = 0
        if "task_id" in dp.batch:
            task_id = int(dp.batch["task_id"].view(-1)[0].item())
        response_mask_key = f"task{task_id}_response_mask" if task_id else None
        score_key = f"task{task_id}_token_level_scores" if task_id else None

        scalar_rewards: list[float] = []
        tid_task_list = []
        comp_A_T:   list[float] = []
        comp_IF_bar: list[float] = []
        comp_P_bar: list[float] = []
        comp_T:     list[float] = []
        missing = 0
        for i in range(B):
            # Try branch-level lookup first for any post-branch row.
            entry = None
            if bid_arr is not None and branch_outcomes is not None:
                bid = int(bid_arr[i])
                if bid >= 0:
                    entry = branch_outcomes.get(bid)
            # Fall back to tid-level shared-prefix outcome.
            if entry is None:
                tid = int(tids[i])
                entry = tid_outcomes.get(tid)
            if entry is None:
                missing += 1
                scalar_rewards.append(0.0)
                tid_task_list.append(0)
                comp_A_T.append(0.0)
                comp_IF_bar.append(0.0)
                comp_P_bar.append(0.0)
                comp_T.append(1.0)
            else:
                outcome_value = entry[0]
                if torch.is_tensor(outcome_value):
                    outcome_scalar = float(outcome_value.float().sum().item())
                else:
                    outcome_scalar = float(outcome_value)
                scalar_rewards.append(outcome_scalar)
                tid_task_list.append(int(entry[1]))
                comps = entry[2] if len(entry) > 2 else {}
                comp_A_T.append(float(comps.get("A_T",   0.0)))
                comp_IF_bar.append(float(comps.get("IF_bar", 0.0)))
                comp_P_bar.append(float(comps.get("P_bar", 0.0)))
                comp_T.append(float(comps.get("T",     1.0)))

        if missing > 0:
            logger.warning(
                f"[AgentLoop] {missing}/{B} rows have no outcome entry; zero-filled"
            )

        if response_mask_key is not None and response_mask_key in dp.batch:
            scores_tensor = self._build_terminal_reward_tensor(
                dp.batch[response_mask_key],
                scalar_rewards,
            )
        else:
            # Legacy fallback if response_mask is absent: reuse current task's
            # score tensor shape and place the scalar on the last token.
            if score_key is not None and score_key in dp.batch:
                scores_tensor = torch.zeros_like(dp.batch[score_key], dtype=torch.float32)
            else:
                scores_tensor = torch.zeros((B, 1), dtype=torch.float32)
            if B > 0:
                scores_tensor[:, -1] = torch.tensor(scalar_rewards, dtype=torch.float32)
        tid_tensor = torch.tensor(tid_task_list, dtype=torch.int32)

        new_batch_dict = {k: v for k, v in dp.batch.items()}
        new_batch_dict["outcome_token_level_scores"] = scores_tensor
        new_batch_dict["outcome_task_id"] = tid_tensor
        dp.batch = TensorDict(new_batch_dict, batch_size=dp.batch.batch_size)

        # Store per-row outcome components in non_tensor_batch for trainer metrics
        dp.non_tensor_batch["outcome_A_T"]   = np.array(comp_A_T,   dtype=np.float32)
        dp.non_tensor_batch["outcome_IF_bar"] = np.array(comp_IF_bar, dtype=np.float32)
        dp.non_tensor_batch["outcome_P_bar"] = np.array(comp_P_bar, dtype=np.float32)
        dp.non_tensor_batch["outcome_T"]     = np.array(comp_T,     dtype=np.float32)
        return dp

    @staticmethod
    def _keys_to_keep_for_task(all_keys, target_task_id: int) -> list[str]:
        """Return the subset of `all_keys` relevant for `task{target_task_id}_dp`.

        Keep: non-task-prefixed keys (common), task1_* (always needed as reference
        even for task2/task3 trainer compound-key logic), and task{target_task_id}_*.
        Drop: other tasks' prefixed keys.
        For task3: additionally keep task2_feedback_texts so that per-turn
        logging shows the feedback paired with T3 regen. Do not keep
        task2_token_level_scores here: those are task2 local/context scores,
        not task3-row MDP returns, and they make rollout summaries look like
        task2 return was attached to the task3 action.
        """
        drop_prefixes = {"task1_", "task2_", "task3_"}
        # task1_dp keeps only task1_* (as its own task); others keep task1_* too.
        if target_task_id == 1:
            drop_prefixes = {"task2_", "task3_"}
        elif target_task_id == 2:
            drop_prefixes = {"task3_"}
        elif target_task_id == 3:
            # Keep T2 feedback text alongside T3 for per-turn logging.
            # All other task2_* tensor keys (input_ids, log_probs, etc.) are dropped
            # to avoid contaminating the training batch.
            _TASK2_LOG_KEYS = {"task2_feedback_texts"}
            return [
                k for k in all_keys
                if k in _TASK2_LOG_KEYS or not k.startswith("task2_")
            ]
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
        outcomes: tuple[dict, dict],
        has_task1_first_turn: bool,
    ) -> dict[int, Optional[DataProto]]:
        """Produce `{1: task1_dp, 2: task2_dp, 3: task3_dp}` with per-row
        outcome attribution.

        `outcomes` is a (tid_outcomes, branch_outcomes) 2-tuple returned by
        `_compute_outcomes_with_avg`.  task1/task2 rows look up by trajectory_id
        (tid_outcomes); task3 rows use branch_id for final-turn rows and fall
        back to trajectory_id for non-final turns.
        """
        tid_outcomes, branch_outcomes = outcomes
        out: dict[int, Optional[DataProto]] = {1: None, 2: None, 3: None}

        if has_task1_first_turn and task1_batch is not None:
            out[1] = self._extract_task_view(task1_batch, target_task_id=1)
            self._attach_outcome_per_row(out[1], tid_outcomes)
            out[1] = self._stamp_phase_metadata(out[1], 1)

        if task2_batches_per_turn:
            task2_views = [self._extract_task_view(b, 2) for b in task2_batches_per_turn]
            out[2] = self._safe_concat(task2_views)
            if out[2] is not None:
                self._attach_outcome_per_row(out[2], tid_outcomes, branch_outcomes)
                out[2] = self._stamp_phase_metadata(out[2], 1)

        if task3_batches_per_turn:
            task3_views = [self._extract_task_view(b, 3) for b in task3_batches_per_turn]
            out[3] = self._safe_concat(task3_views)
            if out[3] is not None:
                # Pass branch_outcomes so final-turn rows get per-branch R_out.
                self._attach_outcome_per_row(out[3], tid_outcomes, branch_outcomes)
                out[3] = self._stamp_phase_metadata(out[3], 1)

        # NOTE: outcome-advantage (GRPO-normalized trajectory outcome broadcast
        # to the task's response_mask) is now computed by the trainer in
        # `_process_batch_common`, not here. Each task DP carries only the raw
        # `outcome_token_level_scores` + `outcome_task_id` per row.
        return out

    @staticmethod
    def _stamp_phase_metadata(dp: Optional[DataProto], phase_id: int = 1) -> Optional[DataProto]:
        if dp is None:
            return None
        dp.non_tensor_batch["phase"] = np.full(len(dp), phase_id, dtype=np.int64)
        return dp

    @staticmethod
    def _stamp_zero_outcome_metadata(dp: Optional[DataProto], task_id: int) -> Optional[DataProto]:
        """Attach zeroed outcome fields so phase2 rows can share the same
        task-batch schema while contributing no outcome signal."""
        if dp is None or getattr(dp, "batch", None) is None:
            return dp

        score_key = f"task{task_id}_token_level_scores"
        response_mask_key = f"task{task_id}_response_mask"
        if score_key in dp.batch:
            zero_scores = torch.zeros_like(dp.batch[score_key], dtype=torch.float32)
        elif response_mask_key in dp.batch:
            zero_scores = torch.zeros_like(dp.batch[response_mask_key], dtype=torch.float32)
        else:
            zero_scores = torch.zeros((len(dp), 1), dtype=torch.float32)

        dp.batch["outcome_token_level_scores"] = zero_scores
        dp.batch["outcome_task_id"] = torch.zeros((len(dp),), dtype=torch.int32)
        dp.non_tensor_batch["outcome_A_T"] = np.zeros(len(dp), dtype=np.float32)
        dp.non_tensor_batch["outcome_IF_bar"] = np.zeros(len(dp), dtype=np.float32)
        dp.non_tensor_batch["outcome_P_bar"] = np.zeros(len(dp), dtype=np.float32)
        dp.non_tensor_batch["outcome_T"] = np.zeros(len(dp), dtype=np.float32)
        return dp

    @staticmethod
    def _ensure_grpo_group_id(dp: Optional[DataProto]) -> Optional[DataProto]:
        if dp is None or "grpo_group_id" in dp.non_tensor_batch or "uid" not in dp.non_tensor_batch:
            return dp
        uids = np.asarray(dp.non_tensor_batch["uid"], dtype=object)
        turn_idx_arr = None
        if getattr(dp, "batch", None) is not None and "turn_idx" in dp.batch:
            turn_idx_arr = dp.batch["turn_idx"].detach().cpu().numpy()
        elif "turn_idx" in dp.non_tensor_batch:
            turn_idx_arr = np.asarray(dp.non_tensor_batch["turn_idx"])
        if turn_idx_arr is None:
            dp.non_tensor_batch["grpo_group_id"] = np.asarray(uids, dtype=object)
        else:
            dp.non_tensor_batch["grpo_group_id"] = np.array(
                [f"{u}_t{int(t)}" for u, t in zip(uids, turn_idx_arr)],
                dtype=object,
            )
        return dp

    @staticmethod
    def _merge_phase_task_dp(phase1_dp: Optional[DataProto], phase2_dp: Optional[DataProto]) -> Optional[DataProto]:
        if phase1_dp is None:
            return phase2_dp
        if phase2_dp is None:
            return phase1_dp
        phase1_dp = FullyAsyncAgentLoopManager._ensure_grpo_group_id(phase1_dp)
        phase2_dp = FullyAsyncAgentLoopManager._ensure_grpo_group_id(phase2_dp)

        def _fill_missing_non_tensor(target: DataProto, reference: DataProto):
            for key, ref_val in reference.non_tensor_batch.items():
                if key in target.non_tensor_batch:
                    continue
                ref_arr = np.asarray(ref_val)
                if ref_arr.dtype == object:
                    fill = np.array([None] * len(target), dtype=object)
                elif np.issubdtype(ref_arr.dtype, np.integer):
                    fill = np.full(len(target), -1, dtype=ref_arr.dtype)
                else:
                    fill = np.zeros(len(target), dtype=ref_arr.dtype)
                target.non_tensor_batch[key] = fill

        _fill_missing_non_tensor(phase1_dp, phase2_dp)
        _fill_missing_non_tensor(phase2_dp, phase1_dp)
        return DataProto.concat([phase1_dp, phase2_dp])

    def _get_prev_align_from_prefix(self, prefix_batch: Optional[DataProto], row: int) -> float:
        if prefix_batch is None:
            return 0.0
        prev_t3 = self._get_outcome_extra_scalar(prefix_batch, 3, "task3_align", [row], default=-1.0)
        if prev_t3 >= 0.0:
            return prev_t3
        return self._get_outcome_extra_scalar(prefix_batch, 1, "task1_align", [row], default=0.0)

    def _build_phase2_candidates(
        self,
        pre_task2_batches_per_turn: list[DataProto],
        task2_batches_per_turn: list[DataProto],
        pre_task3_batches_per_turn: list[DataProto],
        task3_batches_per_turn: list[DataProto],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Build a global top-K list of post-hoc one-step replay targets."""
        candidates: list[dict[str, Any]] = []
        eps = 1e-6

        for turn_idx, (pre_b, t2_b) in enumerate(zip(pre_task2_batches_per_turn, task2_batches_per_turn)):
            if pre_b is None or t2_b is None or len(pre_b) != len(t2_b):
                continue
            feedback_texts = t2_b.non_tensor_batch.get("task2_feedback_texts", None)
            uids = t2_b.non_tensor_batch.get("uid", None)
            tids = t2_b.non_tensor_batch.get("trajectory_id", None)
            bids = t2_b.non_tensor_batch.get("branch_id", None)
            for row in range(len(t2_b)):
                task2_process = self._get_outcome_extra_scalar(t2_b, 2, "task2_process", [row], default=0.0)
                decision_vqa = self._get_outcome_extra_scalar(
                    t2_b, 2, "task2_decision_vqa_reward", [row], default=0.0
                )
                feedback = feedback_texts[row] if feedback_texts is not None and row < len(feedback_texts) else None
                model_no_edit = not _is_edit_sample(feedback)
                target_no_edit = bool(decision_vqa >= 1.0 - eps)
                wrong_decision = target_no_edit != model_no_edit
                local_reward = 0.0
                if "task2_token_level_scores" in t2_b.batch:
                    local_reward = float(
                        t2_b.batch["task2_token_level_scores"][row].clamp(min=0).sum().item()
                    )
                candidates.append(
                    {
                        "task_id": 2,
                        "turn_idx": int(turn_idx),
                        "priority": float((1.0 - task2_process) + (0.5 if wrong_decision else 0.0)),
                        "local_reward": float(local_reward),
                        "pre_batch": pre_b,
                        "row_idx": int(row),
                        "uid": str(uids[row]) if uids is not None else f"uid_{row}",
                        "trajectory_id": int(tids[row]) if tids is not None else -1,
                        "branch_id": int(bids[row]) if bids is not None else -1,
                    }
                )

        for turn_idx, (pre_b, t3_b) in enumerate(zip(pre_task3_batches_per_turn, task3_batches_per_turn)):
            if pre_b is None or t3_b is None or len(pre_b) != len(t3_b):
                continue
            uids = t3_b.non_tensor_batch.get("uid", None)
            tids = t3_b.non_tensor_batch.get("trajectory_id", None)
            bids = t3_b.non_tensor_batch.get("branch_id", None)
            for row in range(len(t3_b)):
                task3_if = self._get_outcome_extra_scalar(t3_b, 3, "task3_if", [row], default=0.0)
                task3_align = self._get_outcome_extra_scalar(t3_b, 3, "task3_align", [row], default=0.0)
                prev_align = self._get_prev_align_from_prefix(pre_b, row)
                align_gain = max(0.0, task3_align - prev_align)
                local_reward = 0.0
                if "task3_token_level_scores" in t3_b.batch:
                    local_reward = float(
                        t3_b.batch["task3_token_level_scores"][row].clamp(min=0).sum().item()
                    )
                candidates.append(
                    {
                        "task_id": 3,
                        "turn_idx": int(turn_idx),
                        "priority": float(1.0 - 0.5 * (task3_if + align_gain)),
                        "local_reward": float(local_reward),
                        "pre_batch": pre_b,
                        "row_idx": int(row),
                        "uid": str(uids[row]) if uids is not None else f"uid_{row}",
                        "trajectory_id": int(tids[row]) if tids is not None else -1,
                        "branch_id": int(bids[row]) if bids is not None else -1,
                    }
                )

        if not candidates or top_k <= 0:
            return []

        candidates.sort(
            key=lambda c: (
                -float(c["priority"]),
                -int(c["turn_idx"]),
                0 if int(c["task_id"]) == 2 else 1,
                float(c["local_reward"]),
            )
        )
        return candidates[:top_k]

    async def _run_phase2_from_phase1_context(
        self,
        pre_task2_batches_per_turn: list[DataProto],
        task2_batches_per_turn: list[DataProto],
        pre_task3_batches_per_turn: list[DataProto],
        task3_batches_per_turn: list[DataProto],
        server_index: Optional[int],
        on_task_complete,
    ) -> dict[int, Optional[DataProto]]:
        """Post-hoc one-step replay from saved Phase1 prefixes.

        Selected candidates are replayed exactly once from their saved prefix,
        expanded to `rollout.n` samples, and trained with local reward only.
        """
        algo_cfg = getattr(getattr(self, "config", None), "algorithm", None) or {}
        _get_hp = algo_cfg.get if hasattr(algo_cfg, "get") else lambda k, d: getattr(algo_cfg, k, d)
        top_k = int(_get_hp("phase2_topk", 0))
        if top_k <= 0:
            return {2: None, 3: None}

        candidates = self._build_phase2_candidates(
            pre_task2_batches_per_turn,
            task2_batches_per_turn,
            pre_task3_batches_per_turn,
            task3_batches_per_turn,
            top_k=top_k,
        )
        if not candidates:
            return {2: None, 3: None}

        from recipe.fully_async_policy_image_rl.detach_utils import _slice_dataproto_with_meta

        group_size = int(self.config.actor_rollout_ref.rollout.n)
        phase2_views: dict[int, list[DataProto]] = {2: [], 3: []}

        for replay_idx, cand in enumerate(candidates):
            prefix = _slice_dataproto_with_meta(cand["pre_batch"], [cand["row_idx"]] * group_size)
            prefix = self._ensure_branch_metadata(prefix)

            replay_uid = f"{cand['uid']}__p2_t{cand['task_id']}_turn{cand['turn_idx']}_k{replay_idx}"
            prefix.non_tensor_batch["uid"] = np.array([replay_uid] * len(prefix), dtype=object)
            prefix.non_tensor_batch["grpo_group_id"] = np.array([replay_uid] * len(prefix), dtype=object)
            prefix.non_tensor_batch["phase2_source_uid"] = np.array([cand["uid"]] * len(prefix), dtype=object)
            prefix.non_tensor_batch["phase2_source_task_id"] = np.full(len(prefix), cand["task_id"], dtype=np.int64)
            prefix.non_tensor_batch["phase2_source_turn_idx"] = np.full(len(prefix), cand["turn_idx"], dtype=np.int64)
            prefix.non_tensor_batch["phase2_source_trajectory_id"] = np.full(
                len(prefix), cand["trajectory_id"], dtype=np.int64
            )
            prefix.non_tensor_batch["phase2_source_branch_id"] = np.full(
                len(prefix), cand["branch_id"], dtype=np.int64
            )

            output = await self._regen_via_worker(
                input_batch=prefix,
                task_id=int(cand["task_id"]),
                turn_idx=int(cand["turn_idx"]),
                server_index=server_index,
                on_task_complete=on_task_complete,
            )
            output = self._extract_task_view(output, int(cand["task_id"]))
            output = self._stamp_phase_metadata(output, 2)
            output = self._stamp_zero_outcome_metadata(output, int(cand["task_id"]))
            phase2_views[int(cand["task_id"])].append(output)

        return {
            2: self._safe_concat(phase2_views[2]) if phase2_views[2] else None,
            3: self._safe_concat(phase2_views[3]) if phase2_views[3] else None,
        }

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
        if task_id == 2:
            output = self._inject_task2_decision_signal(output, input_batch)
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
        rows earn their own MDP edit return instead of inheriting only the
        original branch prefix signal."""
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
        per-turn rollouts concatenated. Each row has `turn_idx`; phase1 local
        token scores are overwritten with MDP discounted returns before the
        dict is returned.

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
            accumulated_batch = self._stamp_phase_metadata(accumulated_batch, 1)
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
            accumulated_batch = self._stamp_phase_metadata(accumulated_batch, 1)
            return accumulated_batch

        from recipe.fully_async_policy_image_rl.detach_utils import _slice_dataproto_with_meta

        # Stamp a unique id per row so we can track each trajectory across
        # per-sample termination / build_edit_batch row reshuffling.
        accumulated_batch = self._stamp_trajectory_ids(prompts)
        accumulated_batch = self._ensure_branch_metadata(accumulated_batch)
        self._branch_id_counter = 0

        task1_batch: Optional[DataProto] = None
        task2_batches_per_turn: list[DataProto] = []
        task3_batches_per_turn: list[DataProto] = []
        pre_task2_batches_per_turn: list[DataProto] = []
        pre_task3_batches_per_turn: list[DataProto] = []
        pending_task1_rewards: list[tuple[int, DataProto, asyncio.Task]] = []
        pending_task3_rewards: list[tuple[int, DataProto, asyncio.Task]] = []
        pending_task2_raw_rewards: list[tuple[DataProto, asyncio.Task]] = []
        outcomes: tuple[dict, dict] = ({}, {})

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
                    if t1_rt is not None:
                        pending_task1_rewards.append((1, accumulated_batch, t1_rt))
                task1_batch = accumulated_batch

            # Pre-task2 prefix snapshot for task2 decision-target lookup.
            pre_task2_batch = accumulated_batch
            pre_task2_batches_per_turn.append(pre_task2_batch)

            # task2 Phase 1
            accumulated_batch = self._set_task_id_on_batch(accumulated_batch, 2)
            worker = self._select_best_worker()
            accumulated_batch = await worker.generate_sequences.remote(accumulated_batch, on_task_complete=None)
            accumulated_batch = self._stamp_turn_idx(accumulated_batch, turn_idx)
            accumulated_batch = self._mark_task2_raw_stage_only(accumulated_batch)
            accumulated_batch.non_tensor_batch["source_task2_row_idx"] = np.arange(
                len(accumulated_batch), dtype=np.int64
            )
            task2_reward_task = on_task_complete(2, accumulated_batch) if on_task_complete is not None else None
            if task2_reward_task is not None:
                pending_task2_raw_rewards.append((accumulated_batch, task2_reward_task))
            task2_batches_per_turn.append(accumulated_batch)

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
            accumulated_batch = self._ensure_branch_metadata(accumulated_batch)

            group_size = int(self.config.actor_rollout_ref.rollout.n)
            accumulated_batch = build_edit_batch(accumulated_batch, group_size=group_size)
            parent_ids = np.full(len(accumulated_batch), -1, dtype=np.int64)
            if "branch_id" in accumulated_batch.non_tensor_batch:
                parent_ids = np.asarray(accumulated_batch.non_tensor_batch["branch_id"], dtype=np.int64)
            accumulated_batch = self._spawn_child_branch_ids(accumulated_batch, parent_ids=parent_ids)

            pre_task3_batch = accumulated_batch
            pre_task3_batches_per_turn.append(pre_task3_batch)

            # task3 Phase 1
            accumulated_batch = self._set_task_id_on_batch(accumulated_batch, 3)
            worker = self._select_best_worker()
            accumulated_batch = await worker.generate_sequences.remote(accumulated_batch, on_task_complete=None)
            accumulated_batch = self._stamp_turn_idx(accumulated_batch, turn_idx)
            if on_task_complete is not None:
                t3_rt = on_task_complete(3, accumulated_batch)
                if t3_rt is not None:
                    pending_task3_rewards.append((3, accumulated_batch, t3_rt))

            task3_batches_per_turn.append(accumulated_batch)

        await self._await_and_attach_reward_entries(pending_task1_rewards)
        await self._await_and_attach_reward_entries(pending_task3_rewards)
        self._attach_phase1_task3_mdp_step5_rewards(
            task1_batch,
            pre_task3_batches_per_turn,
            task3_batches_per_turn,
        )
        await self._await_and_finalize_task2_entries(
            pending_task2_raw_rewards,
            pre_task2_batches_per_turn,
            task2_batches_per_turn,
        )
        # Compute Phase1 trajectory outcomes over the surviving build_edit_batch
        # branch tree. Terminal leaves are final task2(no-edit) rows or final
        # task3 rows; shared-prefix rows receive descendant means and post-
        # duplication rows receive their own/descendant branch outcomes.
        all_tids = None
        if task1_batch is not None and "trajectory_id" in task1_batch.non_tensor_batch:
            all_tids = task1_batch.non_tensor_batch["trajectory_id"]
        elif task2_batches_per_turn and "trajectory_id" in task2_batches_per_turn[0].non_tensor_batch:
            all_tids = task2_batches_per_turn[0].non_tensor_batch["trajectory_id"]
        if all_tids is not None:
            outcomes = self._compute_outcomes_with_avg(
                all_tids, task1_batch, task2_batches_per_turn, task3_batches_per_turn
            )
            self._attach_phase1_mdp_token_scores(
                all_tids,
                task1_batch,
                task2_batches_per_turn,
                task3_batches_per_turn,
            )
        self._propagate_logging_context(
            task1_batch,
            task2_batches_per_turn,
            task3_batches_per_turn,
        )

        phase1_task_dict = self._build_task_dict_per_sample(
            task1_batch,
            task2_batches_per_turn,
            task3_batches_per_turn,
            outcomes,
            has_task1_first_turn,
        )
        phase2_task_dict = await self._run_phase2_from_phase1_context(
            pre_task2_batches_per_turn,
            task2_batches_per_turn,
            pre_task3_batches_per_turn,
            task3_batches_per_turn,
            server_index=None,
            on_task_complete=on_task_complete,
        )
        for tid in (2, 3):
            phase1_task_dict[tid] = self._merge_phase_task_dp(
                phase1_task_dict.get(tid),
                phase2_task_dict.get(tid),
            )
        return phase1_task_dict

    async def _finalize_phase1_task_dict(
        self,
        context: dict,
    ) -> dict[int, Optional[DataProto]]:
        task1_batch = context["task1_batch"]
        task2_batches_per_turn = context["task2_batches_per_turn"]
        task3_batches_per_turn = context["task3_batches_per_turn"]
        pre_task2_batches_per_turn = context["pre_task2_batches_per_turn"]
        pre_task3_batches_per_turn = context["pre_task3_batches_per_turn"]
        pending_task1_rewards = context["pending_task1_rewards"]
        pending_task3_rewards = context["pending_task3_rewards"]
        pending_task2_raw_rewards = context["pending_task2_raw_rewards"]
        has_task1_first_turn = context["has_task1_first_turn"]

        await self._await_and_attach_reward_entries(pending_task1_rewards)
        await self._await_and_attach_reward_entries(pending_task3_rewards)
        self._attach_phase1_task3_mdp_step5_rewards(
            task1_batch,
            pre_task3_batches_per_turn,
            task3_batches_per_turn,
        )
        await self._await_and_finalize_task2_entries(
            pending_task2_raw_rewards,
            pre_task2_batches_per_turn,
            task2_batches_per_turn,
        )

        outcomes: tuple[dict, dict] = ({}, {})
        all_tids = None
        if task1_batch is not None and "trajectory_id" in task1_batch.non_tensor_batch:
            all_tids = task1_batch.non_tensor_batch["trajectory_id"]
        elif task2_batches_per_turn and "trajectory_id" in task2_batches_per_turn[0].non_tensor_batch:
            all_tids = task2_batches_per_turn[0].non_tensor_batch["trajectory_id"]
        if all_tids is not None:
            outcomes = self._compute_outcomes_with_avg(
                all_tids, task1_batch, task2_batches_per_turn, task3_batches_per_turn
            )
            self._attach_phase1_mdp_token_scores(
                all_tids,
                task1_batch,
                task2_batches_per_turn,
                task3_batches_per_turn,
            )
        self._propagate_logging_context(
            task1_batch,
            task2_batches_per_turn,
            task3_batches_per_turn,
        )

        return self._build_task_dict_per_sample(
            task1_batch,
            task2_batches_per_turn,
            task3_batches_per_turn,
            outcomes,
            has_task1_first_turn,
        )

    async def generate_sequences_with_callback_on_server(
        self,
        prompts: DataProto,
        server_index: int,
        on_task_complete=None,
        defer_reward_finalize: bool = False,
    ) -> Any:
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
            accumulated_batch = self._stamp_phase_metadata(accumulated_batch, 1)
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
            accumulated_batch = self._stamp_phase_metadata(accumulated_batch, 1)
            return accumulated_batch

        from recipe.fully_async_policy_image_rl.detach_utils import _slice_dataproto_with_meta

        accumulated_batch = self._stamp_trajectory_ids(prompts)
        accumulated_batch = self._ensure_branch_metadata(accumulated_batch)
        self._branch_id_counter = 0

        task1_batch: Optional[DataProto] = None
        task2_batches_per_turn: list[DataProto] = []
        task3_batches_per_turn: list[DataProto] = []
        pre_task2_batches_per_turn: list[DataProto] = []
        pre_task3_batches_per_turn: list[DataProto] = []
        pending_task1_rewards: list[tuple[int, DataProto, asyncio.Task]] = []
        pending_task3_rewards: list[tuple[int, DataProto, asyncio.Task]] = []
        pending_task2_raw_rewards: list[tuple[DataProto, asyncio.Task]] = []
        outcomes: tuple[dict, dict] = ({}, {})

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
                    if t1_rt is not None:
                        pending_task1_rewards.append((1, accumulated_batch, t1_rt))
                task1_batch = accumulated_batch

            # Pre-task2 prefix snapshot for task2 decision-target lookup.
            pre_task2_batch = accumulated_batch
            pre_task2_batches_per_turn.append(pre_task2_batch)

            # task2 Phase 1
            accumulated_batch = self._set_task_id_on_batch(accumulated_batch, 2)
            worker = self._select_best_worker()
            accumulated_batch = await worker.generate_sequences_on_server.remote(accumulated_batch, server_index)
            accumulated_batch = self._stamp_turn_idx(accumulated_batch, turn_idx)
            accumulated_batch = self._mark_task2_raw_stage_only(accumulated_batch)
            accumulated_batch.non_tensor_batch["source_task2_row_idx"] = np.arange(
                len(accumulated_batch), dtype=np.int64
            )
            task2_reward_task = on_task_complete(2, accumulated_batch) if on_task_complete is not None else None
            if task2_reward_task is not None:
                pending_task2_raw_rewards.append((accumulated_batch, task2_reward_task))
            task2_batches_per_turn.append(accumulated_batch)

            feedback_texts = accumulated_batch.non_tensor_batch.get("task2_feedback_texts", None)
            if feedback_texts is not None:
                is_no_edit = np.array([not _is_edit_sample(f) for f in feedback_texts])
            else:
                is_no_edit = np.zeros(len(accumulated_batch), dtype=bool)

            active_indices = np.where(~is_no_edit)[0].tolist()
            if not active_indices:
                break
            accumulated_batch = _slice_dataproto_with_meta(accumulated_batch, active_indices)
            accumulated_batch = self._ensure_branch_metadata(accumulated_batch)

            group_size = int(self.config.actor_rollout_ref.rollout.n)
            accumulated_batch = build_edit_batch(accumulated_batch, group_size=group_size)
            parent_ids = np.full(len(accumulated_batch), -1, dtype=np.int64)
            if "branch_id" in accumulated_batch.non_tensor_batch:
                parent_ids = np.asarray(accumulated_batch.non_tensor_batch["branch_id"], dtype=np.int64)
            accumulated_batch = self._spawn_child_branch_ids(accumulated_batch, parent_ids=parent_ids)

            pre_task3_batch = accumulated_batch
            pre_task3_batches_per_turn.append(pre_task3_batch)

            # task3 Phase 1
            accumulated_batch = self._set_task_id_on_batch(accumulated_batch, 3)
            worker = self._select_best_worker()
            accumulated_batch = await worker.generate_sequences_on_server.remote(accumulated_batch, server_index)
            accumulated_batch = self._stamp_turn_idx(accumulated_batch, turn_idx)
            if on_task_complete is not None:
                t3_rt = on_task_complete(3, accumulated_batch)
                if t3_rt is not None:
                    pending_task3_rewards.append((3, accumulated_batch, t3_rt))

            task3_batches_per_turn.append(accumulated_batch)

        algo_cfg = getattr(getattr(self, "config", None), "algorithm", None) or {}
        _get_hp = algo_cfg.get if hasattr(algo_cfg, "get") else lambda k, d: getattr(algo_cfg, k, d)
        phase2_topk = int(_get_hp("phase2_topk", 0))
        if defer_reward_finalize and phase2_topk <= 0:
            deferred_context = {
                "task1_batch": task1_batch,
                "task2_batches_per_turn": task2_batches_per_turn,
                "task3_batches_per_turn": task3_batches_per_turn,
                "pre_task2_batches_per_turn": pre_task2_batches_per_turn,
                "pre_task3_batches_per_turn": pre_task3_batches_per_turn,
                "pending_task1_rewards": pending_task1_rewards,
                "pending_task3_rewards": pending_task3_rewards,
                "pending_task2_raw_rewards": pending_task2_raw_rewards,
                "has_task1_first_turn": has_task1_first_turn,
            }
            return {}, deferred_context

        await self._await_and_attach_reward_entries(pending_task1_rewards)
        await self._await_and_attach_reward_entries(pending_task3_rewards)
        self._attach_phase1_task3_mdp_step5_rewards(
            task1_batch,
            pre_task3_batches_per_turn,
            task3_batches_per_turn,
        )
        await self._await_and_finalize_task2_entries(
            pending_task2_raw_rewards,
            pre_task2_batches_per_turn,
            task2_batches_per_turn,
        )
        # Compute Phase1 trajectory outcomes over the surviving build_edit_batch
        # branch tree. Terminal leaves are final task2(no-edit) rows or final
        # task3 rows; shared-prefix rows receive descendant means and post-
        # duplication rows receive their own/descendant branch outcomes.
        all_tids = None
        if task1_batch is not None and "trajectory_id" in task1_batch.non_tensor_batch:
            all_tids = task1_batch.non_tensor_batch["trajectory_id"]
        elif task2_batches_per_turn and "trajectory_id" in task2_batches_per_turn[0].non_tensor_batch:
            all_tids = task2_batches_per_turn[0].non_tensor_batch["trajectory_id"]
        if all_tids is not None:
            outcomes = self._compute_outcomes_with_avg(
                all_tids, task1_batch, task2_batches_per_turn, task3_batches_per_turn
            )
            self._attach_phase1_mdp_token_scores(
                all_tids,
                task1_batch,
                task2_batches_per_turn,
                task3_batches_per_turn,
            )
        self._propagate_logging_context(
            task1_batch,
            task2_batches_per_turn,
            task3_batches_per_turn,
        )

        phase1_task_dict = self._build_task_dict_per_sample(
            task1_batch,
            task2_batches_per_turn,
            task3_batches_per_turn,
            outcomes,
            has_task1_first_turn,
        )
        phase2_task_dict = await self._run_phase2_from_phase1_context(
            pre_task2_batches_per_turn,
            task2_batches_per_turn,
            pre_task3_batches_per_turn,
            task3_batches_per_turn,
            server_index=server_index,
            on_task_complete=on_task_complete,
        )
        for tid in (2, 3):
            phase1_task_dict[tid] = self._merge_phase_task_dp(
                phase1_task_dict.get(tid),
                phase2_task_dict.get(tid),
            )
        return phase1_task_dict

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
