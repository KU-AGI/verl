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
from torchvision.transforms.functional import to_pil_image

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
        self.steps = 0
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.save_freq = reward_kwargs.get("img_saving", {}).get("save_freq", 0)
        self.save_num = reward_kwargs.get("img_saving", {}).get("num", 0)
        self.save_path = reward_kwargs.get("img_saving", {}).get("path", "")
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_path = os.path.join(
            self.save_path, 
            f"{reward_kwargs.get('img_saving', {}).get('experiment_name', '')}_{time_stamp}"
        )

    def save_img(self, data: DataProto):
        """Save images from meta_info lists."""
        prompt_id = data.non_tensor_batch.get('prompt_id', [])
        prompt = data.non_tensor_batch.get('prompt', [])
        gen_imgs_pil_list = data.non_tensor_batch.get('task1_gen_imgs_pil_list', [])
        feedback_texts = data.non_tensor_batch.get('task2_feedback_texts', [])
        regen_imgs_pil_list = data.non_tensor_batch.get('task3_regen_imgs_pil_list', [])
        ground_truth = data.non_tensor_batch.get('reward_model', {})

        step_dir = os.path.join(self.save_path, str(self.steps))
        os.makedirs(step_dir, exist_ok=True)
        
        print(f"[SAVE] Saving {min(len(gen_imgs_pil_list), self.save_num)} images to {step_dir}")
        
        with open(os.path.join(step_dir, "texts.txt"), 'w', encoding='utf-8') as f:
            f.write("Input Texts and Generated Content\n")
            f.write("=" * 80 + "\n\n")
            
            for i in range(min(len(prompt), len(gen_imgs_pil_list), len(feedback_texts), len(regen_imgs_pil_list), len(ground_truth), self.save_num)):
                prompt_id = prompt_id[i]

                f.write(f"Sample {i}\n")
                f.write("=" * 40 + "\n")
                
                # Save input text
                if i < len(prompt):
                    f.write(f"Input Text: {prompt[i]}\n")
                
                # Save generated image
                if i < len(gen_imgs_pil_list):
                    save_path = os.path.join(step_dir, f"img_{prompt_id}_{i}.png")
                    PIL.Image.fromarray(gen_imgs_pil_list[i].astype(np.uint8)).save(save_path)
                    f.write(f"Generated Image:\nimg_{prompt_id}_{i}.png\n\n")
                
                # Save feedback text
                if i < len(feedback_texts):
                    f.write(f"Feedback of {prompt_id}:\n{feedback_texts[i]}\n\n")
                
                # Save regen image
                if i < len(regen_imgs_pil_list):
                    regen_path = os.path.join(step_dir, f"regen_img_{prompt_id}_{i}.png")
                    PIL.Image.fromarray(regen_imgs_pil_list[i].astype(np.uint8)).save(regen_path)
                    f.write(f"Regenerated Image:\nregen_img_{prompt_id}_{i}.png\n\n")
                
                # Save RM text if available
                if i < len(ground_truth):
                    ground_truth_path = os.path.join(step_dir, f"ground_truth_{prompt_id}_{i}.png")
                    PIL.Image.open(ground_truth[i]["ground_truth"]).convert("RGB").save(ground_truth_path)
                    f.write(f"Ground Truth:\nground_truth_{prompt_id}_{i}.png\n\n")
                
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
        
        # Compute scores for all samples
        scores = []
        for i in range(len(data)):
            score_result = self.compute_score(
                prompt=prompt[i],
                gen_img=gen_imgs_pil_list[i] if i < len(gen_imgs_pil_list) else None,
                feedback_text=feedback_texts[i] if i < len(feedback_texts) else "",
                regen_img=regen_imgs_pil_list[i] if i < len(regen_imgs_pil_list) else None,
                ground_truth=ground_truth[i]["ground_truth"],
                extra_info=data.non_tensor_batch.get("extra_info", {}),
                task_id=task_id,
                **self.reward_kwargs,
            )
            scores.append(score_result)
        
        return scores

    def __call__(self, data: DataProto, task_id: int = 1, eval: bool = False, return_dict: bool = True):
        """Main reward computation with batch processing."""
        
        # Save generated images periodically
        if self.save_freq > 0 and self.steps % self.save_freq == 0:
            self.save_path = os.path.join(self.save_path, "eval" if eval else "train")
            os.makedirs(self.save_path, exist_ok=True)
            self.save_img(data)
            print(f"[SAVE] Saving images to {self.save_path}")

        if not eval:
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
            reward_tensor[i, valid_response_length - 1] = reward

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