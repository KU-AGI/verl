# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    task_id = batch.batch["task_id"].view(-1)[0].item()

    prompt_mask = batch.batch[f"task{task_id}_attention_mask"]
    response_mask = batch.batch[f"task{task_id}_response_mask"]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    task_id = batch.batch["task_id"].view(-1)[0].item()

    sequence_score = batch.batch[f"task{task_id}_token_level_scores"].sum(-1)
    sequence_reward = batch.batch[f"task{task_id}_token_level_rewards"].sum(-1)

    advantages = batch.batch[f"task{task_id}_advantages"]
    returns = batch.batch[f"task{task_id}_returns"]

    prompt_mask = batch.batch[f"task{task_id}_attention_mask"].bool()
    response_mask = batch.batch[f"task{task_id}_response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)
    max_response_length = response_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    # Handle case when all samples are aborted (empty tensors)
    if non_aborted_sequence_score.numel() > 0:
        score_mean = torch.mean(non_aborted_sequence_score).detach().item()
        score_max = torch.max(non_aborted_sequence_score).detach().item()
        score_min = torch.min(non_aborted_sequence_score).detach().item()
    else:
        score_mean = 0.0
        score_max = 0.0
        score_min = 0.0

    if non_aborted_sequence_reward.numel() > 0:
        reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
        reward_max = torch.max(non_aborted_sequence_reward).detach().item()
        reward_min = torch.min(non_aborted_sequence_reward).detach().item()
    else:
        reward_mean = 0.0
        reward_max = 0.0
        reward_min = 0.0

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    # Handle empty tensors for advantages and returns
    if valid_adv.numel() > 0:
        adv_mean = torch.mean(valid_adv).detach().item()
        adv_max = torch.max(valid_adv).detach().item()
        adv_min = torch.min(valid_adv).detach().item()
    else:
        adv_mean = 0.0
        adv_max = 0.0
        adv_min = 0.0

    if valid_returns.numel() > 0:
        returns_mean = torch.mean(valid_returns).detach().item()
        returns_max = torch.max(valid_returns).detach().item()
        returns_min = torch.min(valid_returns).detach().item()
    else:
        returns_mean = 0.0
        returns_max = 0.0
        returns_min = 0.0

    if use_critic:
        values = batch.batch[f"task{task_id}_values"]
        valid_values = torch.masked_select(values, response_mask)
        if valid_values.numel() > 0 and valid_returns.numel() > 0:
            values_mean = torch.mean(valid_values).detach().item()
            values_max = torch.max(valid_values).detach().item()
            values_min = torch.min(valid_values).detach().item()
            return_diff_var = torch.var(valid_returns - valid_values)
            return_var = torch.var(valid_returns)
            vf_explained_var = (1.0 - return_diff_var / (return_var + 1e-5)).detach().item()
        else:
            values_mean = 0.0
            values_max = 0.0
            values_min = 0.0
            vf_explained_var = 0.0

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        non_aborted_response_length_mean = 0.0
        non_aborted_response_length_max = 0.0
        non_aborted_response_length_min = 0.0
        non_aborted_response_length_clip_ratio = 0.0
        # raise ValueError("All samples are aborted, this should not happen.")

    metrics = {
        # score
        f"critic/task{task_id}/score/mean": score_mean,
        f"critic/task{task_id}/score/max": score_max,
        f"critic/task{task_id}/score/min": score_min,
        # reward
        f"critic/task{task_id}/rewards/mean": reward_mean,
        f"critic/task{task_id}/rewards/max": reward_max,
        f"critic/task{task_id}/rewards/min": reward_min,
        # adv
        f"critic/task{task_id}/advantages/mean": adv_mean,
        f"critic/task{task_id}/advantages/max": adv_max,
        f"critic/task{task_id}/advantages/min": adv_min,
        # returns
        f"critic/task{task_id}/returns/mean": returns_mean,
        f"critic/task{task_id}/returns/max": returns_max,
        f"critic/task{task_id}/returns/min": returns_min,
        **(
            {
                # values
                f"critic/task{task_id}/values/mean": values_mean,
                f"critic/task{task_id}/values/max": values_max,
                f"critic/task{task_id}/values/min": values_min,
                # vf explained var
                f"critic/task{task_id}/vf_explained_var": vf_explained_var,
            }
            if use_critic
            else {}
        ),
        # response length
        f"response_length/task{task_id}/mean": torch.mean(response_length).detach().item(),
        f"response_length/task{task_id}/max": torch.max(response_length).detach().item(),
        f"response_length/task{task_id}/min": torch.min(response_length).detach().item(),
        f"response_length/task{task_id}/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        f"response_length_non_aborted/task{task_id}/mean": non_aborted_response_length_mean,
        f"response_length_non_aborted/task{task_id}/max": non_aborted_response_length_max,
        f"response_length_non_aborted/task{task_id}/min": non_aborted_response_length_min,
        f"response_length_non_aborted/task{task_id}/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        f"response/task{task_id}/aborted_ratio": aborted_ratio,
        # prompt length
        f"prompt_length/task{task_id}/mean": torch.mean(prompt_length).detach().item(),
        f"prompt_length/task{task_id}/max": torch.max(prompt_length).detach().item(),
        f"prompt_length/task{task_id}/min": torch.min(prompt_length).detach().item(),
        f"prompt_length/task{task_id}/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation: not use
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    if "data_source" in batch.non_tensor_batch:
        data_sources = batch.non_tensor_batch["data_source"]
        # 1. Compute score & reward statistics by data_source
        data_src2scores = defaultdict(list)
        data_src2rewards = defaultdict(list)
        
        for idx, data_source in enumerate(data_sources):
            if non_aborted_mask[idx]:  # Only non-aborted samples
                data_src2scores[data_source].append(sequence_score[idx].item())
                data_src2rewards[data_source].append(sequence_reward[idx].item())
        for data_source in data_src2scores.keys():
            scores = data_src2scores[data_source]
            rewards = data_src2rewards[data_source]
            
            if len(scores) > 0:
                metrics[f"critic/task{task_id}/score/{data_source}/mean"] = np.mean(scores)
                metrics[f"critic/task{task_id}/score/{data_source}/max"] = np.max(scores)
                metrics[f"critic/task{task_id}/score/{data_source}/min"] = np.min(scores)
                
                metrics[f"critic/task{task_id}/rewards/{data_source}/mean"] = np.mean(rewards)
                metrics[f"critic/task{task_id}/rewards/{data_source}/max"] = np.max(rewards)
                metrics[f"critic/task{task_id}/rewards/{data_source}/min"] = np.min(rewards)
    
    # 2. Compute statistics for all _reward & _score metrics in reward_extra_info (overall)
    for metric_name, metric_values in batch.meta_info.items():
        if metric_name.endswith("_reward") or metric_name.endswith("_score"):
            if len(metric_values) > 0:
                metrics[f"critic/{metric_name}/mean"] = np.mean(metric_values)
                metrics[f"critic/{metric_name}/max"] = np.max(metric_values)
                metrics[f"critic/{metric_name}/min"] = np.min(metric_values)
    
    # 3. Compute statistics for _reward & _score metrics by data_source
    if "data_source" in batch.non_tensor_batch:
        data_sources = batch.non_tensor_batch["data_source"]
        
        for metric_name, metric_values in batch.meta_info.items():
            if metric_name.endswith("_reward") or metric_name.endswith("_score"):
                # Group by data_source
                data_src2metric_vals = defaultdict(list)
                for idx, data_source in enumerate(data_sources):
                    if non_aborted_mask[idx] and idx < len(metric_values):
                        data_src2metric_vals[data_source].append(metric_values[idx])
                
                # Compute statistics for each data_source
                for data_source, vals in data_src2metric_vals.items():
                    if len(vals) > 0:
                        metrics[f"critic/{metric_name}/{data_source}/mean"] = np.mean(vals)
                        metrics[f"critic/{metric_name}/{data_source}/max"] = np.max(vals)
                        metrics[f"critic/{metric_name}/{data_source}/min"] = np.min(vals)
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    task_id = batch.batch["task_id"].view(-1)[0].item()

    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/task{task_id}/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/task{task_id}/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
            if num_tokens_of_section[name] > 0
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def compute_group_reward_metrics(batch: DataProto) -> dict[str, Any]:
    """
    Computes reward-related metrics from a batch of data.

    This function calculates statistics about the rewards assigned to responses
    in the batch, including mean, max, min, and the ratio of positive rewards.

    Args:
        batch: A DataProto object containing batch data with token-level rewards.

    Returns:
        A dictionary containing:
            - group_rewards/task{task_id}/mean: Mean of sequence rewards
            - group_rewards/task{task_id}/max: Max of sequence rewards
            - group_rewards/task{task_id}/min: Min of sequence rewards
            - group_rewards/task{task_id}/positive_ratio: Ratio of sequences with positive rewards
    """
    task_id = batch.batch["task_id"].view(-1)[0].item()
    task_reward_extra_info = batch.meta_info.get(f"task{task_id}_reward_extra_info", {})

    task_reward_section = {}
    for name, value in task_reward_extra_info.items():
        if name.endswith("_reward") or name.endswith("_score"):
            task_reward_section[f"group_rewards/{name}/mean"] = np.mean(value)
            task_reward_section[f"group_rewards/{name}/max"] = np.max(value)
            task_reward_section[f"group_rewards/{name}/min"] = np.min(value)
            task_reward_section[f"group_rewards/{name}/positive_ratio"] = np.mean(np.array(value) > 0)

    # Advantage collapse - computed per group (same uid = same prompt)
    sequence_score = batch.batch[f"task{task_id}_token_level_scores"].sum(-1).cpu().numpy()
    uids = batch.non_tensor_batch.get("uid", None)

    cv_threshold = 0.01  # under 1%: regard as collapse

    if uids is not None and len(uids) > 0:
        # Group scores by uid
        uid2scores = defaultdict(list)
        for uid, score in zip(uids, sequence_score):
            uid2scores[uid].append(float(score))

        group_all_same, group_collapse, group_stds, group_cvs = [], [], [], []
        for scores in uid2scores.values():
            if len(scores) < 2:
                continue
            scores_arr = np.array(scores)
            std = np.std(scores_arr)
            mean = np.mean(scores_arr)
            cv = std / (abs(mean) + 1e-8)
            group_all_same.append(float(len(np.unique(scores_arr)) == 1))
            group_collapse.append(float(cv < cv_threshold))
            group_stds.append(std)
            group_cvs.append(cv)

        if len(group_all_same) > 0:
            all_same_ratio = float(np.mean(group_all_same))
            collapse_ratio = float(np.mean(group_collapse))
            score_std = float(np.mean(group_stds))
            score_cv = float(np.mean(group_cvs))
        else:
            all_same_ratio = collapse_ratio = score_std = score_cv = 0.0
    else:
        # Fallback: no uid info, skip collapse metrics
        all_same_ratio = collapse_ratio = score_std = score_cv = 0.0

    task_reward_section.update({
        f"group_rewards/task{task_id}/advantage_collapse/all_same_ratio": all_same_ratio,
        f"group_rewards/task{task_id}/advantage_collapse/collapse_ratio": collapse_ratio,
        f"group_rewards/task{task_id}/advantage_collapse/score_std": score_std,
        f"group_rewards/task{task_id}/advantage_collapse/score_cv": score_cv,
    })

    return task_reward_section


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)
        - "advantage_collapse/all_same_ratio": Ratio of prompts where all values are identical
        - "advantage_collapse/collapse_ratio": Ratio of prompts with CV < 1%

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                # Filter out -100 and None values (invalid/masked values)
                valid_indices = [i for i, val in enumerate(var_vals) if val != -100 and val is not None]
                if len(valid_indices) == 0:
                    continue  # Skip if all values are invalid
                
                filtered_var_vals = [var_vals[i] for i in valid_indices]

                metric = {}
                n_resps = len(filtered_var_vals)
                metric[f"mean@{n_resps}"] = np.mean(filtered_var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(filtered_var_vals)

                    # Collapse detection
                    unique_vals = np.unique(filtered_var_vals)
                    all_same = len(unique_vals) == 1
                    metric["advantage_collapse/all_same_ratio"] = float(all_same)
                    
                    val_std = np.std(filtered_var_vals)
                    val_mean = np.mean(filtered_var_vals)
                    cv = val_std / (abs(val_mean) + 1e-8)
                    
                    cv_threshold = 0.01 # under 1%: regard as collapse
                    metric["advantage_collapse/collapse_ratio"] = float(cv < cv_threshold)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=filtered_var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            # Also filter pred values based on valid_indices
                            filtered_pred_vals = [var2vals["pred"][i] for i in valid_indices]
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(filtered_var_vals, filtered_pred_vals, strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)
                    data_src2var2metric2uid_vals["all"][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val
