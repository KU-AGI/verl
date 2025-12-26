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

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from recipe.fully_async_policy_image_rl.agent_loop.agent_loop import AgentLoopOutput
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.utils.model import compute_position_id_with_mask
import PIL


import PIL.Image

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
    
    # Step 1: Add uid
    rs.full_batch.non_tensor_batch["uid"] = np.array(
        [f"uid_{rs.sample_id}"] * batch_size, dtype=object
    )
    
    # Step 2: Set processing_times from meta_info
    rs.processing_times = rs.full_batch.meta_info.get("metrics", {}).get(
        "generate_sequences", [0.0] * batch_size
    )
    
    # Step 3: Extract param_version_start and param_version_end from meta_info
    rs.param_version_start = [rs.full_batch.meta_info.get("param_version_start", 0)] * batch_size
    rs.param_version_end = [rs.full_batch.meta_info.get("param_version_end", 0)] * batch_size
    
    # Step 4: Clear agent_loop_output_list
    rs.agent_loop_output_list = []
    
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

    return rs


def expand_rollout_sample(rs: RolloutSample) -> list[RolloutSample]:
    """
    Expand a merged RolloutSample (containing multiple samples in batch)
    back into individual RolloutSamples.

    This reverses the merge operation and creates individual samples
    that can be sent through the original put_sample/get_sample pattern.
    """
    batch_size = len(rs.full_batch)
    individual_samples = []

    for i in range(batch_size):
        # Extract single item from batch using DataProto slicing
        single_batch = rs.full_batch[i:i+1]

        # Create individual RolloutSample
        individual_sample = RolloutSample(
            full_batch=single_batch,
            agent_loop_output_list=[],  # Already cleared in merge_rollout_sample
            sample_id=f"{rs.sample_id}_idx{i}",
            epoch=rs.epoch,
            tool_calls=rs.tool_calls,
            param_version=rs.param_version,
            param_version_start=[rs.param_version_start[i]] if rs.param_version_start else [0],
            param_version_end=[rs.param_version_end[i]] if rs.param_version_end else [0],
            processing_times=[rs.processing_times[i]] if rs.processing_times else [0.0],
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

    n_samples = len(rollout_samples)
    print(f"[BatchUtils] Assembling batch from {n_samples} RolloutSample objects")

    # Pre-allocate lists
    rollout_samples_batch = [rs.full_batch for rs in rollout_samples]
    processing_times = [t for rs in rollout_samples for t in rs.processing_times]
    
    # Prefix rollout_status once
    rollout_status = {f"fully_async/{k}": v for k, v in rollout_samples[0].rollout_status.items()}

    final_batch = DataProto.concat(rollout_samples_batch)

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
        "rollout_param_versions": param_versions,
        "param_version_diversity": len(set(param_versions)),
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
