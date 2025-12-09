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

from verl.workers.reward_manager import register
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch

import os
import PIL
import datetime
from typing import Any, List
from collections import defaultdict
import contextlib
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import numpy as np
from recipe.image_rl.utils import FormattingEvaluator


@register("image_generation")
class ImageGenerationRewardManager:
    """The reward manager for batch processing."""

    def __init__(
        self, 
        tokenizer, 
        processor,
        num_examine, 
        compute_score=None, 
        reward_fn_key="data_source", 
        eval=False,
        **reward_kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data: DataProto, task_id: int) -> List[dict]:
        """Verify and compute scores for batch."""
        # Get lists from non_tensor_batch
        prompt = data.non_tensor_batch.get('prompt', [])
        gen_imgs_pil_list = data.non_tensor_batch.get('task1_gen_imgs_pil_list', [])
        feedback_texts = data.non_tensor_batch.get('task2_feedback_texts', [])
        regen_imgs_pil_list = data.non_tensor_batch.get('task3_regen_imgs_pil_list', [])
        ground_truth = data.non_tensor_batch.get('reward_model', {})
        
        print(f"[VERIFY] Processing batch of {len(prompt)} samples")
        
        # Prepare batch data
        prompts = prompt
        gen_imgs = [gen_imgs_pil_list[i] if i < len(gen_imgs_pil_list) else None for i in range(len(data))]
        feedback_texts_padded = [feedback_texts[i] if i < len(feedback_texts) else "" for i in range(len(data))]
        regen_imgs = [regen_imgs_pil_list[i] if i < len(regen_imgs_pil_list) else None for i in range(len(data))]
        ground_truth_imgs = [ground_truth[i]["ground_truth"] if i < len(ground_truth) else None for i in range(len(data))]
        feedback_tuples = [ground_truth[i]["tuple"] if i < len(ground_truth) else None for i in range(len(data))]
        vqa_questions = [ground_truth[i]["vqa_question"] if i < len(ground_truth) else None for i in range(len(data))]
        extra_infos = [data.non_tensor_batch.get("extra_info", {})] * len(data)
        task_ids = [task_id] * len(data)
        
        # Call batch processing function
        scores = self.compute_score(
            prompts, gen_imgs, feedback_texts_padded, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids
        )

        return scores

    def __call__(self, data: DataProto, task_id: int = 1, eval: bool = False, return_dict: bool = True):
        """Main reward computation with batch processing."""
        print(f"[REWARD] Computing rewards for batch_size={len(data)}")
        response_mask = data.batch[f"task{task_id}_response_mask"]

        # Initialize reward tensor
        reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        prompt = data.non_tensor_batch.get('prompt', [])

        # Get data sources
        data_sources = data.non_tensor_batch.get(self.reward_fn_key, ["unknown"] * len(data))
        
        # Compute scores
        scores = self.verify(data, task_id)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            valid_response_length = response_mask[i].sum()
            score_dict = scores[i]
            reward = score_dict.get("score", 0.0)
            # if reward != -100:
            #     reward = reward * 10 # scale up
            
            # Update extra info
            if "reward_extra_info" in score_dict:
                for key, value in score_dict["reward_extra_info"].items():
                    reward_extra_info[key].append(value)

            rewards.append(reward)
            reward_tensor[i, valid_response_length - 1] = reward

            if reward == -100:
                data.batch[f"task{task_id}_response_mask"][i] = torch.zeros_like(response_mask[i], dtype=torch.float32)

            # Print examination samples
            data_source = data_sources[i] if i < len(data_sources) else "unknown"
            if already_printed.get(data_source, 0) < self.num_examine:
                prompt_text = prompt[i]
                ground_truth = data[i].non_tensor_batch.get("reward_model", {}).get("ground_truth", "N/A")
                
                print(f"\n[EXAMINE {i}]")
                print(f"Data Source: {data_source}")
                print(f"Prompt: {prompt_text}")
                print(f"Ground Truth: {ground_truth}") # path
                print(f"Score: {score_dict}")
                print("-" * 80)
                
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        # Store accuracy
        data.batch[f"task{task_id}_acc"] = torch.tensor(rewards, dtype=torch.float32)

        # Compute mean excluding -100 rewards
        valid_rewards = [reward for reward in rewards if reward != -100]
        mean_reward = sum(valid_rewards) / len(valid_rewards) if len(valid_rewards) > 0 else 0.0
        print(f"[REWARD] Computed {len(rewards)} rewards, valid={len(valid_rewards)}, mean={mean_reward:.4f}")

        if return_dict:
            return {f"task{task_id}_reward_tensor": reward_tensor, f"task{task_id}_reward_extra_info": reward_extra_info}
        else:
            return reward_tensor