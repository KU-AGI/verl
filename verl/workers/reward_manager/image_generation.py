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
from typing import Any
from collections import defaultdict

@register("image_generation")
class ImageGenerationRewardManager:
    """The reward manager."""

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score=None, 
        reward_fn_key="data_source", 
        **reward_kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.steps = 0
        self.reward_fn_key = reward_fn_key
        self.save_freq = reward_kwargs.get("img_saving", {}).get("save_freq", 0)
        self.save_num = reward_kwargs.get("img_saving", {}).get("num", 0)
        self.save_path = reward_kwargs.get("img_saving", {}).get("path", "")
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_path = os.path.join(self.save_path, f"{reward_kwargs.get('img_saving', {}).get('experiment_name', '')}_{time_stamp}")
        
    def save_img(self, data: DataProto):
        gen_img = data.batch['gen_img']
        feedback_text = data.batch['feedback_text']
        refined_gen_img = data.batch['refined_gen_img']
        gen_img = gen_img.to('cpu').numpy() if isinstance(gen_img, torch.Tensor) else gen_img
        refined_gen_img = refined_gen_img.to('cpu').numpy() if isinstance(refined_gen_img, torch.Tensor) else refined_gen_img
        feedback_text = feedback_text.to('cpu').numpy() if isinstance(feedback_text, torch.Tensor) else feedback_text
        step_dir = os.path.join(self.save_path, str(self.steps))
        os.makedirs(step_dir, exist_ok=True)
        with open(os.path.join(step_dir, "texts.txt"), 'a') as f:
            f.write("Prompts:\n")
            for i in range(min(len(gen_img), self.save_num)):
                f.write("="*40 + "\n")
                f.write("Image:\n")
                save_path = os.path.join(step_dir, "img_{}.jpg".format(i))
                PIL.Image.fromarray(gen_img[i]).save(save_path)
                prompt = data.batch['prompts'][i]
                f.write(f'{self.tokenizer.decode(prompt, skip_special_tokens=True)}\n\n')
 
            if feedback_text is not None:
                f.write("="*40 + "\n")
                f.write("Feedback Text:\n")
                for i in range(min(len(feedback_text), self.save_num)):
                    f.write(f'{feedback_text[i]}\n\n')
  
            if refined_gen_img is not None:
                f.write("="*40 + "\n")
                f.write("Refined Image:\n")
                for i in range(min(len(refined_gen_img), self.save_num)):
                    save_refined_path = os.path.join(step_dir, "refined_img_{}.jpg".format(i))
                    PIL.Image.fromarray(refined_gen_img[i]).save(save_refined_path)
                    f.write(f'{feedback_text[i]}\n\n')

            if 'rm_text' in data.non_tensor_batch:
                f.write("="*40 + "\n")
                f.write("RM Text:\n")
                for i in range(min(len(gen_img), self.save_num)):
                    rm_text = data.non_tensor_batch['rm_text'][i]
                    f.write(f'{rm_text}\n\n')
            
            if 'text_tokens' in data.batch:
                f.write("="*40 + "\n")
                f.write("Text Tokens:\n")
                for i in range(min(len(gen_img), self.save_num)):
                    text_tokens = data.batch['text_tokens'][i]
                    f.write(f'{self.tokenizer.decode(text_tokens, skip_special_tokens=True)}\n\n')

    def verify(self, data):
        prompt_ids = data.batch["prompts"]

        prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

        gen_img = data.batch["gen_img"]
        feedback_text = data.batch["feedback_text"]
        refined_gen_img = data.batch["refined_gen_img"]

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])

        scores = self.compute_score(
            prompt=prompt,
            gen_img=gen_img,
            feedback_text=feedback_text,
            refined_gen_img=refined_gen_img,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )

        return scores

    def __call__(self, data: DataProto, return_dict: bool = False, eval: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # save generated images
        if self.steps % self.save_freq == 0:
            self.save_img(data)
        if not eval:
            self.steps += 1
            
        print("Images saved to 'generated_samples' folder")

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["gen_img"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompt"]
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed: dict[str, Any] = {}

        for i in range(len(data)):
            reward = scores[i]["score"]
            reward_extra_info.update(scores[i]["reward_extra_info"])

            rewards.append(reward)

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                prompt_str = self.tokenizer.decode(data.batch["prompt"][i]["content"], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"]["ground_truth"]
                print("[prompt]", prompt_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
