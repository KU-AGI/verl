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

@register("image_generation")
class ImageGenerationRewardManager:
    """The reward manager for batch processing."""

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score=None, 
        reward_fn_key="data_source", 
        **reward_kwargs
    ) -> None:
        self.tokenizer = tokenizer
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
        # Get lists from meta_info
        if not hasattr(data, "meta_info"):
            print("[SAVE] No meta_info found, skipping save")
            return
            
        gen_img_list = data.meta_info.get('gen_img_list', [])
        refined_img_list = data.meta_info.get('refined_gen_img_list', [])
        feedback_texts = data.meta_info.get('feedback_texts', [])
        
        if not gen_img_list:
            print("[SAVE] No images in gen_img_list, skipping save")
            return
        
        step_dir = os.path.join(self.save_path, str(self.steps))
        os.makedirs(step_dir, exist_ok=True)
        
        print(f"[SAVE] Saving {min(len(gen_img_list), self.save_num)} images to {step_dir}")
        
        with open(os.path.join(step_dir, "texts.txt"), 'w', encoding='utf-8') as f:
            f.write("Prompts and Generated Content\n")
            f.write("=" * 80 + "\n\n")
            
            for i in range(min(len(gen_img_list), self.save_num)):
                f.write(f"Sample {i}\n")
                f.write("=" * 40 + "\n")
                
                # Save generated image
                if i < len(gen_img_list):
                    gen_img = gen_img_list[i]
                    if isinstance(gen_img, PIL.Image.Image):
                        save_path = os.path.join(step_dir, f"img_{i}.png")
                        gen_img.save(save_path)
                        f.write(f"Generated Image: img_{i}.png\n")
                
                # Save prompt
                if i < len(data.batch['input_ids']):
                    prompt = data.batch['input_ids'][i]
                    prompt_text = self.processor.decode(prompt, skip_special_tokens=True)
                    f.write(f"Prompt: {prompt_text}\n\n")
                
                # Save feedback text
                if i < len(feedback_texts):
                    f.write(f"Feedback: {feedback_texts[i]}\n\n")
                
                # Save refined image
                if i < len(refined_img_list):
                    refined_img = refined_img_list[i]
                    if isinstance(refined_img, PIL.Image.Image):
                        refined_path = os.path.join(step_dir, f"refined_img_{i}.png")
                        refined_img.save(refined_path)
                        f.write(f"Refined Image: refined_img_{i}.png\n")
                
                # Save RM text if available
                if 'rm_text' in data.non_tensor_batch and i < len(data.non_tensor_batch['rm_text']):
                    rm_text = data.non_tensor_batch['rm_text'][i]
                    f.write(f"RM Text: {rm_text}\n")
                
                # Save text tokens if available
                if 'text_tokens' in data.batch and i < len(data.batch['text_tokens']):
                    text_tokens = data.batch['text_tokens'][i]
                    decoded = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
                    f.write(f"Text Tokens: {decoded}\n")
                
                f.write("\n" + "=" * 40 + "\n\n")
        
        print(f"[SAVE] Saved {min(len(gen_img_list), self.save_num)} samples to {step_dir}")

    def verify(self, data: DataProto) -> List[dict]:
        """Verify and compute scores for batch."""
        # Get lists from meta_info
        gen_img_list = data.meta_info.get('gen_img_list', [])
        refined_img_list = data.meta_info.get('refined_gen_img_list', [])
        feedback_texts = data.meta_info.get('feedback_texts', [])
        
        batch_size = len(gen_img_list)
        print(f"[VERIFY] Processing batch of {batch_size} samples")
        
        # Prepare batch data for scoring
        prompts = []
        for i in range(batch_size):
            prompt_ids = data.batch["prompts"][i]
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            prompts.append(prompt_text)
        
        # Get ground truths and extras
        ground_truths = []
        extras = []
        for i in range(batch_size):
            gt = data[i].non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            ground_truths.append(gt)
            
            extra = data[i].non_tensor_batch.get("extra_info", {})
            extras.append(extra)
        
        # Compute scores for all samples
        scores = []
        for i in range(batch_size):
            score_result = self.compute_score(
                prompt=prompts[i],
                gen_img=gen_img_list[i] if i < len(gen_img_list) else None,
                feedback_text=feedback_texts[i] if i < len(feedback_texts) else "",
                refined_gen_img=refined_img_list[i] if i < len(refined_img_list) else None,
                ground_truth=ground_truths[i],
                extra_info=extras[i],
                **self.reward_kwargs,
            )
            scores.append(score_result)
        
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False, eval: bool = False):
        """Main reward computation with batch processing."""
        
        # Save generated images periodically
        if self.save_freq > 0 and self.steps % self.save_freq == 0:
            self.save_img(data)
        
        if not eval:
            self.steps += 1

        # If rm_scores already computed, return directly
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        # Get batch size from meta_info
        gen_img_list = data.meta_info.get('gen_img_list', [])
        batch_size = len(gen_img_list)
        
        if batch_size == 0:
            print("[REWARD] Warning: No images in gen_img_list")
            batch_size = data.batch["prompts"].shape[0]
        
        print(f"[REWARD] Computing rewards for batch_size={batch_size}")
        
        # Initialize reward tensor
        device = data.batch["prompts"].device
        reward_tensor = torch.zeros(batch_size, dtype=torch.float32, device=device)
        reward_extra_info = defaultdict(list)
        
        # Get data sources
        data_sources = data.non_tensor_batch.get(self.reward_fn_key, ["unknown"] * batch_size)
        
        # Compute scores
        scores = self.verify(data)
        rewards = []
        already_printed = {}

        for i in range(batch_size):
            score_dict = scores[i]
            reward = score_dict.get("score", 0.0)
            
            # Update extra info
            if "reward_extra_info" in score_dict:
                for key, value in score_dict["reward_extra_info"].items():
                    reward_extra_info[key].append(value)
            
            rewards.append(reward)
            reward_tensor[i] = reward

            # Print examination samples
            data_source = data_sources[i] if i < len(data_sources) else "unknown"
            if already_printed.get(data_source, 0) < self.num_examine:
                prompt_text = self.tokenizer.decode(
                    data.batch["prompts"][i], 
                    skip_special_tokens=True
                )
                ground_truth = data[i].non_tensor_batch.get("reward_model", {}).get("ground_truth", "N/A")
                
                print(f"\n[EXAMINE {i}]")
                print(f"Data Source: {data_source}")
                print(f"Prompt: {prompt_text}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Score: {score_dict}")
                print("-" * 80)
                
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        # Store accuracy
        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        print(f"[REWARD] Computed {len(rewards)} rewards, mean={sum(rewards)/len(rewards):.4f}")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor