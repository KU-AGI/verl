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

import inspect
from typing import Any, List

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import _default_compute_score


@register("image_generation")
class ImageGenerationRewardLoopManager(RewardLoopManagerBase):
    """The reward loop manager for image generation tasks."""

    def __init__(
        self,
        config,
        tokenizer,
        processor=None,
        compute_score=None,
        reward_fn_key="data_source",
        **reward_kwargs
    ) -> None:
        super().__init__(config, tokenizer)
        self.processor = processor
        self.compute_score = compute_score or _default_compute_score
        self.is_async_compute_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

        # Get task_id from config if available
        self.task_id = config.get("task_id", 1)

    async def run_single(self, data: DataProto) -> dict:
        """Process a single data item and compute reward.

        Args:
            data: DataProto containing a single item

        Returns:
            dict with keys:
                - reward_score: float reward value
                - reward_extra_info: dict of additional reward information
        """
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        # Get task_id from data if available, otherwise use default
        task_id = data_item.non_tensor_batch.get("task_id", self.task_id)

        # Extract data from non_tensor_batch
        prompt = data_item.non_tensor_batch.get('prompt', '')
        gen_imgs_pil_list = data_item.non_tensor_batch.get('task1_gen_imgs_pil_list', [])
        feedback_texts = data_item.non_tensor_batch.get('task2_feedback_texts', [])
        regen_imgs_pil_list = data_item.non_tensor_batch.get('task3_regen_imgs_pil_list', [])
        reward_model_data = data_item.non_tensor_batch.get('reward_model', {})

        # Prepare single item data
        gen_img = gen_imgs_pil_list[0] if len(gen_imgs_pil_list) > 0 else None
        feedback_text = feedback_texts[0] if len(feedback_texts) > 0 else ""
        regen_img = regen_imgs_pil_list[0] if len(regen_imgs_pil_list) > 0 else None
        ground_truth_img = reward_model_data.get("ground_truth", None)
        feedback_tuple = reward_model_data.get("tuple", None)
        vqa_question = reward_model_data.get("vqa_question", None)
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        # Call compute_score function (async or sync)
        if self.is_async_compute_score:
            score_result = await self.compute_score(
                [prompt],
                [gen_img],
                [feedback_text],
                [regen_img],
                [ground_truth_img],
                [feedback_tuple],
                [vqa_question],
                [extra_info],
                [task_id]
            )
        else:
            score_result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    [prompt],
                    [gen_img],
                    [feedback_text],
                    [regen_img],
                    [ground_truth_img],
                    [feedback_tuple],
                    [vqa_question],
                    [extra_info],
                    [task_id]
                )
            )

        # Extract score from result (compute_score returns a list)
        score_dict = score_result[0] if isinstance(score_result, list) else score_result
        reward = score_dict.get("score", 0.0)

        # Prepare reward_extra_info
        reward_extra_info = {}
        if "reward_extra_info" in score_dict:
            reward_extra_info.update(score_dict["reward_extra_info"])

        # Add any additional fields from score_dict
        for key, value in score_dict.items():
            if key not in ["score", "reward_extra_info"]:
                reward_extra_info[key] = value

        return {
            "reward_score": reward,
            "reward_extra_info": reward_extra_info
        }
