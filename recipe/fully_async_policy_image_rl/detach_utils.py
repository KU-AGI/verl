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
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union
import re
import json

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from recipe.fully_async_policy_image_rl.agent_loop.agent_loop import AgentLoopOutput
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.utils.model import compute_position_id_with_mask
import PIL
from recipe.image_rl.utils import FormattingEvaluatorV2

import PIL.Image

def safe_json_loads(text):
    try:
        # Markdown 블록 제거 및 순수 JSON 추출
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except:
        return None

def postprocess_agent_loop_outputs(rs_or_list: Union["RolloutSample", list], tokenizer, config, processor) -> DataProto:
    """Optimized postprocessing of AgentLoopOutput list into DataProto with dynamic task_id handling"""
    if isinstance(rs_or_list, list):
        inputs = rs_or_list
    else:
        inputs = rs_or_list.agent_loop_output_list
    if not inputs:
        return DataProto.concat([])
    
    batch_list = []
    IMAGE_KEY_PATTERNS = frozenset(('image', 'pixel', 'img', 'pil', 'gen_img_tokens', 'regen_img_tokens'))
    
    def is_image_key(key: str, _patterns=IMAGE_KEY_PATTERNS) -> bool:
        key_lower = key.lower()
        return any(p in key_lower for p in _patterns)

    for out in inputs:
        generation_data: DataProto = out.generation_data
        batch = generation_data.batch
        
        # --- 1) task_id 기반 누적 Attention Mask 계산 ---
        t_id_val = batch.get("task_id")
        # task_id가 텐서든 스칼라든 안전하게 정수로 변환
        current_task_id = int(t_id_val.reshape(-1)[0].item()) if t_id_val is not None else 1
        
        cumulative_sum = torch.tensor(0, dtype=torch.long, device=batch.device if hasattr(batch, 'device') else 'cpu')
        
        for i in range(1, current_task_id + 1):
            for mask_type in ["attention_mask", "response_mask"]:
                k = f"task{i}_{mask_type}"
                if k in batch:
                    # 각 마스크의 유효 토큰(1의 개수)을 더함
                    cumulative_sum += batch[k].sum().long()

        # 결과는 항상 [1] 차원의 텐서로 유지
        attention_mask = cumulative_sum.view(1)

        # --- 2) 텐서 처리 ---
        fixed_tensors: dict[str, torch.Tensor] = {"attention_mask": attention_mask}
        
        for name, t in batch.items():
            if name == "dummy_tensor" or not isinstance(t, torch.Tensor):
                continue
            
            dim = t.dim()
            is_img = is_image_key(name)
            
            if dim == 0:
                fixed_tensors[name] = t.unsqueeze(0)
            elif dim == 1:
                fixed_tensors[name] = t.unsqueeze(0)
            elif dim == 2:
                # [Seq, Hidden] -> [1, Seq, Hidden]
                fixed_tensors[name] = t.unsqueeze(0) 
            elif dim == 3:
                # [C, H, W] -> [1, C, H, W]
                fixed_tensors[name] = t.unsqueeze(0)
            else:
                fixed_tensors[name] = t[:1]

        # --- 3) Non-tensor 처리 ---
        non_tensor_batch_dict: dict[str, np.ndarray] = {
            "__num_turns__": np.array([out.num_turns], dtype=np.int32)
        }

        if generation_data.non_tensor_batch:
            for k, v in generation_data.non_tensor_batch.items():
                if isinstance(v, np.ndarray) and v.dtype == object and len(v) > 0:
                    if isinstance(v.flat[0], PIL.Image.Image):
                        non_tensor_batch_dict[k] = v[:1] if v.shape[0] > 1 else v
                        continue

                v = np.asarray(v)
                ndim = v.ndim
                is_img = is_image_key(k)
                
                if ndim == 0:
                    non_tensor_batch_dict[k] = v[None]
                elif ndim == 1:
                    non_tensor_batch_dict[k] = v[np.newaxis, ...] if is_img else (v[:1] if v.shape[0] != 1 else v)
                else:
                    non_tensor_batch_dict[k] = v[np.newaxis, ...] if is_img else v[:1]

        # --- 4) Metrics 추출 및 모든 필드를 meta_info에 추가 ---
        metrics = out.metrics.model_dump() if hasattr(out.metrics, "model_dump") else (
            out.metrics if isinstance(out.metrics, dict) else {}
        )

        # Extract all relevant fields from AgentLoopOutput to meta_info
        meta_info = {
            "metrics": metrics,
            "param_version_start": getattr(out, 'param_version_start', 0),
            "param_version_end": getattr(out, 'param_version_end', 0),
            "is_cancel": getattr(out, 'is_cancel', False),
            "num_turns": getattr(out, 'num_turns', 1),
        }

        batch_list.append(DataProto.from_dict(
            tensors=fixed_tensors,
            non_tensors=non_tensor_batch_dict,
            meta_info=meta_info,
        ))

    return DataProto.concat(batch_list)

@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: Any

    # AgentLoopOutput from generation
    agent_loop_output_list: list[AgentLoopOutput]

    # Metadata
    sample_id: str
    epoch: int

    # Processing metadata
    processing_times: list[float]
    tool_calls: list[float]
    param_version: int
    param_version_start: list[int]
    param_version_end: list[int]
    rollout_status: dict[str, Any]


@dataclass
class ValidateMetrics:
    """Metrics for validation"""

    timing_raw: dict[str, Any]
    metrics: Optional[dict[str, Any]] = None
    global_steps: Optional[int] = None
    param_version: Optional[int] = None


def prepare_single_generation_data(batch_dict, global_steps, rollout_n) -> DataProto:
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    full_batch = DataProto.from_single_dict(batch_dict)
    # reward_model_keys = set({"data_source", "reward_model", "extra_info"}) & full_batch.non_tensor_batch.keys()

    # pop those keys for generation
    batch_keys_to_pop = ["dummy_tensor"]
    non_tensor_batch_keys_to_pop = set(full_batch.non_tensor_batch.keys()) # - reward_model_keys

    batch = full_batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
    )

    # Setting agent - partial_single_turn_agent, that supports partial
    batch.non_tensor_batch["agent_name"] = np.array(["partial_single_turn_agent"] * len(batch), dtype=object)

    # Add global step count to generated data
    batch = batch.repeat(repeat_times=rollout_n, interleave=True)
    return batch


def process_rollout_log_probs(data_proto: DataProto, rollout_log_probs: list[list[float]]) -> torch.Tensor:
    """
    Process rollout_log_probs according to the mask in DataProto
    mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]

    Args:
        data_proto: A DataProto object containing batch information
        rollout_log_probs: A two-dimensional list, each sublist containing the log_probs of a sample

    Returns:
        torch.Tensor: The processed log_probs tensor, with shape: [bsz, response_length]
    """
    batch = data_proto.batch
    response_mask = batch["response_mask"]
    rollout_log_probs_tensor = torch.zeros(response_mask.shape, dtype=torch.float32) - 1

    for i, log_probs_seq in enumerate(rollout_log_probs):
        # Get the effective length of the current sample (the number of positions with 1 in the mask)
        valid_length = response_mask[i].sum().item()

        # Ensure that the length of log_probs_seq does not exceed the valid length
        actual_length = min(len(log_probs_seq), valid_length)

        # Fill log_probs into the corresponding position
        if actual_length > 0:
            rollout_log_probs_tensor[i, :actual_length] = torch.tensor(log_probs_seq[:actual_length])

    rollout_log_probs_tensor = rollout_log_probs_tensor.to(torch.float32)
    return rollout_log_probs_tensor


def merge_rollout_sample(config, tokenizer, rs: RolloutSample, processor):
    """
    Supplement and refine the RolloutSample object.
    """
    batch_size = len(rs.full_batch)
    
    # Step 1: Set processing_times from meta_info
    rs.processing_times = rs.full_batch.meta_info.get("metrics", {}).get(
        "generate_sequences", [0.0] * batch_size
    )
    
    # Step 2: Extract param_version_start and param_version_end from meta_info
    rs.param_version_start = [rs.full_batch.meta_info.get("param_version_start", 0)] * batch_size
    rs.param_version_end = [rs.full_batch.meta_info.get("param_version_end", 0)] * batch_size
    
    # Step 3: Clear agent_loop_output_list
    rs.agent_loop_output_list = []

    # Step 4: Calculate ramained reward
    # Extract task 1 VQA judge
    vqa_judges: list = rs.full_batch.meta_info.get("task1_vqa_reward", [None] * batch_size)

    # Extract task 2 predicted judge
    formatting_evaluator = FormattingEvaluatorV2()
    feedback_texts = rs.full_batch.non_tensor_batch.get('task2_feedback_texts', [None] * batch_size)

    predicted_judges: list = []
    for feedback_text in feedback_texts:
        par1, part2, part3, part4 = formatting_evaluator._split_text_into_parts(feedback_text.strip())
        predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback = par1, part2, part3, part4
        predict_decomposed_ans = formatting_evaluator._extract_answer_paragraphs(predicted_answer)
        predicted_judge = formatting_evaluator.check_all_answers_positive(predict_decomposed_ans)
        predicted_judges.append(predicted_judge)

    task2_token_level_scores = rs.full_batch.batch["task2_token_level_scores"]
    task2_response_mask = rs.full_batch.batch["task2_response_mask"]

    extra_info = rs.full_batch.meta_info
    extra_info["task2_judge_alignment_reward"] = [-100] * batch_size 
    extra_info["task2_judge_alignment_reward_response"] = [None] * batch_size

    task2_feedback_reward_score: list = [-100] * batch_size
    for i, (vqa_judge, predicted_judge) in enumerate(zip(vqa_judges, predicted_judges)):

        if vqa_judge is None or predicted_judge is None: # Invalid case
            task2_feedback_reward_score[i] = 0.0
            extra_info["task2_judge_alignment_reward"][i] = -100
            extra_info["task2_feedback_reward"][i] = -100
            extra_info["task2_judge_alignment_reward_response"][i] = "Judge value is None."
            extra_info["task2_feedback_reward_response"][i] = "Judge value is None."
            continue

        vqa_judge = (vqa_judge == 1)

        if vqa_judge and predicted_judge:  # VLM yes & Policy yes
            task2_feedback_reward_score[i] = 1.0
            extra_info["task2_judge_alignment_reward"][i] = 1.0
            extra_info["task2_feedback_reward"][i] = 1.0
            extra_info["task2_judge_alignment_reward_response"][i] = "No need to get feedback reward. Both VQA alignment and predicted answer judge are positive(+)."
            extra_info["task2_feedback_reward_response"][i] = "No need to get feedback response. Both VQA alignment and predicted answer judge are positive(+)."
            
        elif not vqa_judge and not predicted_judge:  # VLM no & Policy no
            # Get reward from API
            # feedback_response = await feedback_task
            feedback_reward = 0.0
            
            task2_feedback_reward_response = extra_info["task2_feedback_reward_response"][i]
            if task2_feedback_reward_response is not None:
                try:
                    feedback_success = safe_json_loads(task2_feedback_reward_response)
                    if feedback_success and feedback_success.get("answer", "").lower() == "yes":
                        feedback_reward = 1.0
                except:
                    feedback_reward = 0.0
            
            task2_feedback_reward_score[i] = feedback_reward
            extra_info["task2_judge_alignment_reward"][i] = 1.0
            extra_info["task2_judge_alignment_reward_response"][i] = "Both VQA alignment and predicted answer judge are negative(-). Proceed to get feedback reward."
            extra_info["task2_feedback_reward"][i] = feedback_reward
            extra_info["task2_feedback_reward_response"][i] = task2_feedback_reward_response if task2_feedback_reward_response is not None else str(task2_feedback_reward_response)
            
        else:  # VLM yes & Policy no / VLM no & Policy yes (Mismatch)
            task2_feedback_reward_score[i] = 0.0
            extra_info["task2_judge_alignment_reward"][i] = 0.0
            extra_info["task2_feedback_reward"][i] = 0.0
            extra_info["task2_judge_alignment_reward_response"][i] = "Fail due to mismatch between VQA alignment and predicted answer judge."
            extra_info["task2_feedback_reward_response"][i] = "Fail to get feedback reward due to mismatch between VQA alignment and predicted answer judge."
        
        # task2_token_level_scores update
        valid_response_length = task2_response_mask[i].sum().int()
        task2_token_level_scores[i, valid_response_length - 1] += task2_feedback_reward_score[i]
    
    rs.full_batch.batch["task2_token_level_scores"] = task2_token_level_scores
    rs.full_batch.meta_info.update(extra_info)

    # Step 5: Mask invalid rewards (-100) for each task
    for task_id in [1, 2, 3]:  # task1, task2, task3
        scores_key = f"task{task_id}_token_level_scores"
        response_mask_key = f"task{task_id}_response_mask"

        if scores_key in rs.full_batch.batch and response_mask_key in rs.full_batch.batch:
            token_level_scores = rs.full_batch.batch[scores_key]  # shape: [batch_size, seq_len]
            response_mask = rs.full_batch.batch[response_mask_key]  # shape: [batch_size, seq_len]

            # Check if any token in each instance has a score of -100
            has_invalid = (token_level_scores == -100).any(dim=1)  # shape: [batch_size]

            # For instances with invalid scores, set response_mask to all zeros
            for i in range(batch_size):
                if has_invalid[i]:
                    rs.full_batch.batch[response_mask_key][i] = torch.zeros_like(response_mask[i])

    # Step 6: Filtering logic
    if not config.algorithm.filter_groups.enable:
        return rs
    else:
        metric_name = config.algorithm.filter_groups.metric

        # Compute per-task metrics
        for task_id in [1, 2, 3]:
            task_scores = rs.full_batch.batch[f"task{task_id}_token_level_scores"]
            rs.full_batch.non_tensor_batch[f"task{task_id}_{metric_name}"] = (
                torch.where(task_scores >= 0, task_scores, torch.zeros_like(task_scores))
                .sum(dim=-1).cpu().numpy()
            )
    
        # Collect metric values per uid for each task
        task_prompt_uid2metric_vals = {1: defaultdict(list), 2: defaultdict(list), 3: defaultdict(list)}
        for task_id in [1, 2, 3]:
            task_metric_key = f"task{task_id}_{metric_name}"
            for uid, metric_val in zip(
                rs.full_batch.non_tensor_batch["uid"],
                rs.full_batch.non_tensor_batch[task_metric_key],
                strict=True
            ):
                task_prompt_uid2metric_vals[task_id][uid].append(metric_val)
        
        # Compute std per uid for each task
        task_prompt_uid2metric_std = {1: {}, 2: {}, 3: {}}
        for task_id in [1, 2, 3]:
            for prompt_uid, metric_vals in task_prompt_uid2metric_vals[task_id].items():
                task_prompt_uid2metric_std[task_id][prompt_uid] = np.std(metric_vals)
        
        # Keep uid only if ALL tasks have std > 0 (or single trajectory)
        all_uids = set(rs.full_batch.non_tensor_batch["uid"])
        kept_prompt_uids = set()
        
        for uid in all_uids:
            # Skip filtering if only one trajectory (no variance possible)
            n_trajs = len(task_prompt_uid2metric_vals[1][uid])
            if n_trajs == 1:
                kept_prompt_uids.add(uid)
                continue
            
            # Require std > 0 for ALL tasks to keep this prompt
            all_tasks_have_variance = all(
                task_prompt_uid2metric_std[task_id].get(uid, 0) > 0
                for task_id in [1, 2, 3]
            )
            if all_tasks_have_variance:
                kept_prompt_uids.add(uid)
        
        # Apply filtering
        kept_traj_idxs = [
            idx for idx, uid in enumerate(rs.full_batch.non_tensor_batch["uid"])
            if uid in kept_prompt_uids
        ]
        rs.full_batch = rs.full_batch[kept_traj_idxs]

        # Apply detail reward logging
        kept_meta_info = {}
        for meta_key, meta_value in rs.full_batch.meta_info.items():
            if isinstance(meta_value, list):
                kept_meta_info[meta_key] = [meta_value[idx] for idx in kept_traj_idxs]
            elif meta_key == "metrics":
                kept_meta_info[meta_key] = {}
                for sub_metric_name, sub_metric_value in rs.full_batch.meta_info["metrics"].items():
                    kept_meta_info[meta_key][sub_metric_name] = [sub_metric_value[idx] for idx in kept_traj_idxs]
            else:
                kept_meta_info[meta_key] = meta_value

        rs.full_batch.meta_info = kept_meta_info

    return rs


def expand_rollout_sample(rs: RolloutSample) -> list[RolloutSample]:
    uids = rs.full_batch.non_tensor_batch['uid']
    original_batch_size = len(uids)
    
    uid_to_indices = {}
    for i, uid in enumerate(uids):
        if uid not in uid_to_indices:
            uid_to_indices[uid] = []
        uid_to_indices[uid].append(i)

    def deep_slice(data, indices, batch_size):
        if isinstance(data, dict):
            return {k: deep_slice(v, indices, batch_size) for k, v in data.items()}
        elif isinstance(data, list) and len(data) == batch_size:
            return [data[i] for i in indices]
        return data

    individual_samples = []

    for uid, indices in uid_to_indices.items():
        group_batch = rs.full_batch[indices]
        
        if hasattr(group_batch, 'meta_info') and group_batch.meta_info:
            group_batch.meta_info = deep_slice(group_batch.meta_info, indices, original_batch_size)

        group_sample_id = f"{rs.sample_id}_{uid}"
        individual_sample = RolloutSample(
            full_batch=group_batch,
            agent_loop_output_list=[], 
            sample_id=group_sample_id,
            epoch=rs.epoch,
            tool_calls=rs.tool_calls,
            param_version=rs.param_version,
            param_version_start=[rs.param_version_start[i] for i in indices] if isinstance(rs.param_version_start, list) else rs.param_version_start,
            param_version_end=[rs.param_version_end[i] for i in indices] if isinstance(rs.param_version_end, list) else rs.param_version_end,
            processing_times=[rs.processing_times[i] for i in indices] if rs.processing_times else [0.0],
            rollout_status=rs.rollout_status,
        )
        individual_samples.append(individual_sample)

    return individual_samples


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None
) -> DataProto:
    """Optimized batch assembly"""
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided")

    n_groups = len(rollout_samples)
    total_rows = sum(len(rs.full_batch) for rs in rollout_samples)
    print(f"[BatchUtils] Assembling batch from {n_groups} groups (Total {total_rows} samples)")

    # Pre-allocate lists
    rollout_samples_batch = [rs.full_batch for rs in rollout_samples]
    final_batch = DataProto.concat(rollout_samples_batch)
    
    processing_times = [t for rs in rollout_samples for t in rs.processing_times]

    all_param_versions = []
    for rs in rollout_samples:
        num_in_group = len(rs.full_batch)
        all_param_versions.extend([rs.param_version] * num_in_group)
    
    # Prefix rollout_status once
    rollout_status = {f"fully_async/{k}": v for k, v in rollout_samples[0].rollout_status.items()}

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Attention mask processing
    if "attention_mask" in final_batch.batch:
        mask = final_batch.batch["attention_mask"]
        if mask.dim() <= 1:
            token_lens = mask.sum().view(1) if mask.dim() == 1 else mask.view(1)
        else:
            token_lens = mask.view(mask.shape[0], -1).sum(dim=-1)
        final_batch.meta_info["global_token_num"] = token_lens.cpu().tolist()

    # Collect param versions efficiently
    param_versions = [rs.param_version for rs in rollout_samples]
    param_version_diff = [
        abs(end - start)
        for rs in rollout_samples
        for start, end in zip(rs.param_version_start, rs.param_version_end)
    ]
    
    # Compute stats using numpy for efficiency
    pt_arr = np.array(processing_times)
    num_diff0 = param_version_diff.count(0)
    n_diff = len(param_version_diff)

    final_batch.meta_info.update({
        "rollout_param_versions": all_param_versions,
        "param_version_diversity": len(set(all_param_versions)),
        "trajectory_param_versions": [v for rs in rollout_samples for v in rs.param_version_end],
        "fully_async/processing_time/avg": pt_arr.mean(),
        "fully_async/processing_time/max": pt_arr.max(),
        "fully_async/processing_time/min": pt_arr.min(),
        "fully_async/processing_time/tp50": np.percentile(pt_arr, 50),
        "fully_async/processing_time/tp95": np.percentile(pt_arr, 95),
        "fully_async/processing_time/tp99": np.percentile(pt_arr, 99),
        "fully_async/partial/total_partial_num": n_diff - num_diff0,
        "fully_async/partial/partial_ratio": (n_diff - num_diff0) / n_diff,
        "fully_async/partial/max_partial_span": max(param_version_diff),
        **rollout_status,
    })

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")
    return final_batch


class MetricsAggregator:
    """Metrics aggregator, used to combine metrics from multiple training steps"""

    def __init__(self, total_gpus: int):
        # Store all values ​​for each metric
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        # Store the number of samples at each step for weighted averaging
        self.sample_counts: list[int] = []
        # Store the timestamp of each step for time-related calculations
        self.timestamps: list[float] = []
        # Step Count
        self.step_count = 0
        # total num gpus used
        self.total_gpus = total_gpus

        # Metric aggregation rule configuration
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, dict[str, list[str]]]:
        """Initialize metrics aggregation rules"""
        return {
            # Time-Based metrics, can add metrics here
            "time_sum": ["perf/time_per_step"],
            "last": [
                "fully_async/count/total_generated_samples",
                "fully_async/count/stale_samples_processed",
                "fully_async/count/stale_trajectory_processed",
                "fully_async/count/current_param_version",
                "fully_async/count/dropped_stale_samples",
                "training/global_step",  # TODO change name to: total_step
            ],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int, timestamp: float = None):
        """Adding a single-step metrics"""
        if timestamp is None:
            timestamp = time.time()

        self.sample_counts.append(sample_count)
        self.timestamps.append(timestamp)
        self.step_count += 1

        # Store all metrics values
        for key, value in metrics.items():
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
            elif isinstance(value, torch.Tensor):
                self.metric_values[key].append(float(value.item()))

    def _get_aggregation_type(self, metric_name: str) -> str:
        """Determine the aggregation type based on the metric name"""
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if any(keyword in metric_lower for keyword in ["timing_s/"]):
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["mean", "avg", "average"]):
            return "avg"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg"]):
            return "weighted_avg"

        return "avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        """Aggregating a single metric"""
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)

        if agg_type == "last":
            return values[-1]

        elif agg_type == "weighted_avg":
            # Weighted average
            if len(values) != len(self.sample_counts):
                # If the lengths do not match, use a simple average
                return sum(values) / len(values)

            total_samples = sum(self.sample_counts)
            if total_samples == 0:
                return sum(values) / len(values)

            weighted_sum = sum(v * c for v, c in zip(values, self.sample_counts, strict=False))
            return weighted_sum / total_samples

        elif agg_type == "sum" or agg_type == "time_sum":
            return sum(values)

        elif agg_type == "avg":
            return sum(values) / len(values)

        elif agg_type == "max":
            return max(values)

        elif agg_type == "min":
            return min(values)

        else:
            # Default average
            return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """aggregated metrics"""
        t = time.time()
        if self.step_count == 0:
            return {}

        aggregated = {}

        # Aggregate all metrics
        for metric_name, values in self.metric_values.items():
            aggregated[metric_name] = self._aggregate_single_metric(metric_name, values)

        # Aggregate special metrics
        aggregated = self._special_metrics_aggergate(aggregated)

        print(f"aggregated metrics done. cost {time.time() - t}")

        return aggregated

    def _special_metrics_aggergate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """calculate special metrics"""

        # global_seqlen/minmax_diff
        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        # perf/throughput
        REQUIRED_PERF_KEYS = {"perf/throughput", "perf/total_num_tokens", "perf/time_per_step"}
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        # trainer/idle_ratio
        if "timing_s/gen" in aggregated.keys() and "timing_s/step" in aggregated.keys():
            aggregated["trainer/idle_ratio"] = aggregated["timing_s/gen"] / aggregated["timing_s/step"]

        return aggregated

    def reset(self):
        """Reset Aggregator"""
        self.metric_values.clear()
        self.sample_counts.clear()
        self.timestamps.clear()
        self.step_count = 0

    def get_current_stats(self) -> dict[str, Any]:
        """Get statistics about the current aggregation state (for debugging)"""
        return {
            "step_count": self.step_count,
            "metric_count": len(self.metric_values),
            "total_samples": sum(self.sample_counts),
            "metric_names": list(self.metric_values.keys()),
        }
