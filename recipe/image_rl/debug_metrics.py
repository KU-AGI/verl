# Copyright 2025 Individual Contributor: TomQunChaoA
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

import logging

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def calculate_debug_metrics(data: DataProto, task_id: int = 1) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
        task_id: int
            task identifier (1, 2, or 3)
    Returns:
        dict: metrics with task-specific keys
    """

    # Get task-specific tensors
    rollout_key = f"task{task_id}_rollout_log_probs"
    old_key = f"task{task_id}_old_log_probs"
    response_mask_key = f"task{task_id}_response_mask"
    attention_mask_key = f"task{task_id}_attention_mask"
    responses_key = f"task{task_id}_responses"
    
    # Check if required keys exist
    if rollout_key not in data.batch or old_key not in data.batch:
        logger.warning(f"Missing log_probs for task{task_id}, skipping debug metrics")
        return {f"training/task{task_id}_rollout_probs_diff_valid": 0}
    
    rollout_old_log_probs = data.batch[rollout_key]
    actor_old_log_probs = data.batch[old_key]
    
    # Get mask (priority: response_mask > attention_mask > all ones)
    if response_mask_key in data.batch:
        logger.debug(f"task{task_id}: response mask found, use it to mask log probs")
        log_prob_mask = data.batch[response_mask_key]
    elif attention_mask_key in data.batch:
        log_prob_mask = data.batch[attention_mask_key]
    else:
        logger.warning(f"task{task_id}: no mask info found, use all log probs")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    
    # Get response length
    if responses_key in data.batch:
        responses = data.batch[responses_key]
        response_length = responses.size(1)
        response_mask = log_prob_mask[:, -response_length:]
    else:
        # Fallback: use full mask
        response_mask = log_prob_mask
    
    # Calculate metrics
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()
    
    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    
    return {
        f"training/task{task_id}_rollout_probs_diff_valid": 1,
        f"training/task{task_id}_rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        f"training/task{task_id}_rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        f"training/task{task_id}_rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        f"training/task{task_id}_rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }
