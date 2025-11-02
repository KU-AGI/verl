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
        self.save_freq = reward_kwargs.get("img_saving", {}).get("save_freq", 0)
        self.save_num = reward_kwargs.get("img_saving", {}).get("num", 0)
        
        # Create base save path once
        root_path: str = reward_kwargs.get("img_saving", {}).get("path", "")
        exp_name: str = reward_kwargs.get("img_saving", {}).get("experiment_name", "")
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_base = os.path.join(root_path, f"{exp_name}_{time_stamp}")
        
        # Initialize step counters
        self.steps = 0
        self.eval_steps = 0

    def save_img(self, data: DataProto, eval: bool = False):
        """Save images from meta_info lists."""
        prompt_id = data.non_tensor_batch.get('prompt_id', [])
        prompt = data.non_tensor_batch.get('prompt', [])
        gen_imgs_pil_list = data.non_tensor_batch.get('task1_gen_imgs_pil_list', [])
        feedback_texts = data.non_tensor_batch.get('task2_feedback_texts', [])
        regen_imgs_pil_list = data.non_tensor_batch.get('task3_regen_imgs_pil_list', [])
        ground_truth = data.non_tensor_batch.get('reward_model', {})

        # Create proper save path without repetition
        mode = "eval" if eval else "train"
        current_step = self.eval_steps if eval else self.steps
        step_dir = os.path.join(self.save_base, mode, str(current_step))
        os.makedirs(step_dir, exist_ok=True)
        
        print(f"[SAVE] Saving {min(len(gen_imgs_pil_list), self.save_num)} images to {step_dir}")
        
        with open(os.path.join(step_dir, "texts.txt"), 'w', encoding='utf-8') as f:
            f.write("Input Texts and Generated Content\n")
            f.write("=" * 80 + "\n\n")
            
            for i in range(min(len(prompt), len(gen_imgs_pil_list), len(feedback_texts), len(regen_imgs_pil_list), len(ground_truth), self.save_num)):
                current_prompt_id = prompt_id[i] if i < len(prompt_id) else i

                f.write(f"Sample {i}\n")
                f.write("=" * 40 + "\n")
                
                # Save input text
                if i < len(prompt):
                    f.write(f"Input Text: {prompt[i]}\n")
                
                # Save generated image
                if i < len(gen_imgs_pil_list):
                    save_path = os.path.join(step_dir, f"gen_img_{current_prompt_id}_{i}.png")
                    PIL.Image.fromarray(gen_imgs_pil_list[i].astype(np.uint8)).save(save_path)
                    f.write(f"Generated Image:\nimg_{current_prompt_id}_{i}.png\n\n")
                
                # Save feedback text
                if i < len(feedback_texts):
                    f.write(f"Feedback of {current_prompt_id}:\n{feedback_texts[i]}\n\n")
                
                # Save regen image
                if i < len(regen_imgs_pil_list):
                    regen_path = os.path.join(step_dir, f"regen_img_{current_prompt_id}_{i}.png")
                    PIL.Image.fromarray(regen_imgs_pil_list[i].astype(np.uint8)).save(regen_path)
                    f.write(f"Regenerated Image:\nregen_img_{current_prompt_id}_{i}.png\n\n")
                
                # Save RM text if available
                if i < len(ground_truth):
                    ground_truth_path = os.path.join(step_dir, f"ground_truth_{current_prompt_id}_{i}.png")
                    PIL.Image.open(ground_truth[i]["ground_truth"]).convert("RGB").save(ground_truth_path)
                    f.write(f"Ground Truth:\nground_truth_{current_prompt_id}_{i}.png\n\n")
                
                f.write("\n" + "=" * 40 + "\n\n")
        
        print(f"[SAVE] Saved {min(len(prompt), len(gen_imgs_pil_list), len(feedback_texts), len(regen_imgs_pil_list), len(ground_truth), self.save_num)} samples to {step_dir}")

    def verify(self, data: DataProto, task_id: int) -> List[dict]:
        """Verify and compute scores for batch."""
        # Get lists from non_tensor_batch
        prompt = data.non_tensor_batch.get('prompt', [])
        gen_imgs_pil_list = data.non_tensor_batch.get('task1_gen_imgs_pil_list', [])
        feedback_texts = data.non_tensor_batch.get('task2_feedback_texts', [])
        regen_imgs_pil_list = data.non_tensor_batch.get('task3_regen_imgs_pil_list', [])
        ground_truth = data.non_tensor_batch.get('reward_model', {})
        
        print(f"[VERIFY] Processing batch of {len(prompt)} samples")
        
        # Use batch processing
        print(f"[VERIFY] Using batch processing with {len(data)} samples")
        
        # Prepare batch data
        prompts = prompt
        gen_imgs = [gen_imgs_pil_list[i] if i < len(gen_imgs_pil_list) else None for i in range(len(data))]
        feedback_texts_padded = [feedback_texts[i] if i < len(feedback_texts) else "" for i in range(len(data))]
        regen_imgs = [regen_imgs_pil_list[i] if i < len(regen_imgs_pil_list) else None for i in range(len(data))]
        ground_truths = [ground_truth[i]["ground_truth"] if i < len(ground_truth) else None for i in range(len(data))]
        extra_infos = [data.non_tensor_batch.get("extra_info", {})] * len(data)
        task_ids = [task_id] * len(data)
        
        # Call batch processing function
        scores = self.compute_score(
            prompts, gen_imgs, feedback_texts_padded, regen_imgs, ground_truths, extra_infos, task_ids
        )

        return scores

    def __call__(self, data: DataProto, task_id: int = 1, eval: bool = False, return_dict: bool = True):
        """Main reward computation with batch processing."""
        
        # Save generated images periodically
        if self.save_freq > 0:
            current_step = self.eval_steps if eval else self.steps
            if current_step % self.save_freq == 0:
                self.save_img(data, eval)
                print(f"[SAVE] Saved images for {'eval' if eval else 'train'} step {current_step}")

        # Update step counters
        if eval:
            self.eval_steps += 1
        else:
            self.steps += 1

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
            valid_response_length = response_mask.sum()
            score_dict = scores[i]
            reward = score_dict.get("score", 0.0)
            
            # Update extra info
            if "reward_extra_info" in score_dict:
                for key, value in score_dict["reward_extra_info"].items():
                    reward_extra_info[key].append(value)
            
            rewards.append(reward)
            reward_tensor[i, :valid_response_length] = reward

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
        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32)
                
        print(f"[REWARD] Computed {len(rewards)} rewards, mean={sum(rewards)/len(rewards):.4f}")

        if return_dict:
            return {f"task{task_id}_reward_tensor": reward_tensor, f"task{task_id}_reward_extra_info": reward_extra_info}
        else:
            return reward_tensor