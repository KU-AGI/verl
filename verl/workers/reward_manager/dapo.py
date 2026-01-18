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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from openai import OpenAI


@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        use_roundtrip_reward=False,
        use_content_reward=False,
        use_decision_reward=False,
        use_reflection_bonus=False,
        reflection_bonus_weight=0.0,
        roundtrip_cache=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.use_roundtrip_reward = use_roundtrip_reward
        self.use_content_reward = use_content_reward
        self.use_decision_reward = use_decision_reward
        self.use_reflection_bonus = use_reflection_bonus
        self.reflection_bonus_weight = reflection_bonus_weight
        
        # self.roundtrip_client = roundtrip_client
        self.roundtrip_cache = roundtrip_cache

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""
        if "rm_scores" in data.batch.keys():
            data.batch.pop("rm_scores") # There is a rm_scores for some reason in the batch. reward model 사용 안하므로 제거
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        self.roundtrip_client = OpenAI(base_url="http://localhost:8007/v1", api_key="EMPTY")
        print("Using DAPORewardManager")
        print(f"client: {self.roundtrip_client}")
        print(f"cache: {self.roundtrip_cache}")

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str_special = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            task = extra_info['task']
            reward_extra_info['task'].append(task)


            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})

            extra_info["rollout_reward_scores"] = rollout_reward_scores

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str_special,
                ground_truth=ground_truth,
                extra_info=extra_info,
                use_roundtrip_reward=self.use_roundtrip_reward,
                use_content_reward=self.use_content_reward,
                use_decision_reward=self.use_decision_reward,
                use_reflection_bonus=self.use_reflection_bonus,
                reflection_bonus_weight=self.reflection_bonus_weight,
                roundtrip_client=self.roundtrip_client,
                roundtrip_cache=self.roundtrip_cache
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score
            reward_tensor[i, valid_response_length - 1] = reward

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            task = data_item.non_tensor_batch['extra_info']['task']

            ## TODO: stepwise reward assignment
            """
            reasoning_start_token_ids = [151667]
            reasoning_end_token_ids = [151668]
            step1_token_ids = [565, 14822, 220, 16, 25, 45451, 30106, 12783, 2303] # ## Step 1: Understanding molecular roels\n
            step2_token_ids = [565, 14822, 220, 17, 25, 18320, 315, 1887, 15629, 5203, 2303] # ## Step 2: Analysis of main functional groups\n
            step3_token_ids = [565, 14822, 220, 18, 25, 66841, 3811, 19639, 2303] # ## Step 3: Reaction template prediction\n
            MOLECULAR_ROLE_start_token_ids = [151673]
            MOLECULAR_ROLE_end_token_ids = [151674]
            PRECURSOR_STAT_start_token_ids = [151677]
            PRECURSOR_STAT_end_token_ids = [151678]
            REACTANT_FUNCGROUP_start_token_ids = [151669]
            REACTANT_FUNCGROUP_end_token_ids = [151670]
            TEMPLATE_start_token_ids = [151683]
            TEMPLATE_end_token_ids = [151684]
            answer_start_token_ids = [27, 11692, 39351, 397]
            answer_end_token_ids = [522, 11692, 39351, 29]
            turn_end_token_ids = [151645]
            """

            # input_ids = data_item.batch["input_ids"][prompt_length:]
            # step4_token_ids = [565, 14822, 220, 19] # "## Step 4"
            # step5_token_ids = [565, 14822, 220, 20] # "## Step 5"
            # step6_token_ids = [565, 14822, 220, 21] # "## Step 6"
            # step7_token_ids = [565, 14822, 220, 22] # "## Step 7"
            # think_end_token_ids = [151668] # "</think>"
            # last_token_ids = [522, 11692, 39351, 29, 151645] # "</ANSWER><|im_end|>"
            # last_index =self._find_subsequence_start(input_ids, last_token_ids)
            # if last_index is None:
            #     last_index = valid_response_length -1
            # else:
            #     last_index += len(last_token_ids)
            # step_start_indices = {
            #     "step4": self._find_subsequence_start(input_ids, step4_token_ids),
            #     "step5": self._find_subsequence_start(input_ids, step5_token_ids),
            #     "step6": self._find_subsequence_start(input_ids, step6_token_ids),
            #     "step7": self._find_subsequence_start(input_ids, step7_token_ids),
            #     "think_end": self._find_subsequence_start(input_ids, think_end_token_ids),
            #     "last": last_index,
            # }
            # try:
            #     if task == "forward":
            #         step4_last_idx = step_start_indices["step5"] - 1
            #         step5_last_idx = step_start_indices["step6"] - 1
            #         step6_last_idx = step_start_indices["think_end"] - 1
            #         answer_last_idx = step_start_indices["last"] - 1
            #         reward_tensor[i, step4_last_idx] = reward_extra_info['step_rewards'][i]['step4']
            #         reward_tensor[i, step5_last_idx] = reward_extra_info['step_rewards'][i]['step5']
            #         reward_tensor[i, step6_last_idx] = reward_extra_info['step_rewards'][i]['step6']
            #         reward_tensor[i, answer_last_idx] = reward_extra_info['step_rewards'][i]['answer']
            #         step_last_indices = {
            #             "step4": step4_last_idx,
            #             "step5": step5_last_idx,
            #             "step6": step6_last_idx,
            #             "answer": answer_last_idx
            #         }
            #     elif task == "retro":
            #         step5_last_idx = step_start_indices["step6"] - 1
            #         step6_last_idx = step_start_indices["step7"] - 1
            #         step7_last_idx = step_start_indices["think_end"] - 1
            #         answer_last_idx = step_start_indices["last"] - 1
            #         reward_tensor[i, step5_last_idx] = reward_extra_info['step_rewards'][i]['step5']
            #         reward_tensor[i, step6_last_idx] = reward_extra_info['step_rewards'][i]['step6']
            #         reward_tensor[i, step7_last_idx] = reward_extra_info['step_rewards'][i]['step7']
            #         reward_tensor[i, answer_last_idx] = reward_extra_info['step_rewards'][i]['answer']
            #         step_last_indices = {
            #             "step5": step5_last_idx,
            #             "step6": step6_last_idx,
            #             "step7": step7_last_idx,
            #             "answer": answer_last_idx
            #         }
            #     elif task == "reagent":
            #         step6_last_idx = step_start_indices["step7"] - 1
            #         step7_last_idx = step_start_indices["think_end"] - 1
            #         answer_last_idx = step_start_indices["last"] - 1
            #         reward_tensor[i, step6_last_idx] = reward_extra_info['step_rewards'][i]['step6']
            #         reward_tensor[i, step7_last_idx] = reward_extra_info['step_rewards'][i]['step7']
            #         reward_tensor[i, answer_last_idx] = reward_extra_info['step_rewards'][i]['answer']
            #         step_last_indices = {
            #             "step6": step6_last_idx,
            #             "step7": step7_last_idx,
            #             "answer": answer_last_idx
            #         }
            # except Exception as e:
            #     if task == "forward":
            #         step_last_indices = {
            #             "step4": None,
            #             "step5": None,
            #             "step6": None,
            #             "answer": None
            #         }
            #     elif task == "retro":
            #         step_last_indices = {
            #             "step5": None,
            #             "step6": None,
            #             "step7": None,
            #             "answer": None
            #         }
            #     elif task == "reagent":
            #         step_last_indices = {
            #             "step6": None,
            #             "step7": None,
            #             "answer": None
            #         }
            #     reward_tensor[i, valid_response_length - 1] = reward
            
            # reward_extra_info["step_last_indices"].append(step_last_indices)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # print("[prompt]", prompt_str)
                # print("[response]", response_str)
                # print("[ground_truth]", ground_truth)
                # if isinstance(result, dict):
                #     for key, value in result.items():
                #         print(f"[{key}]", value)
                # else:
                #     print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _find_subsequence_start(self, input_ids, step_ids):
        if not isinstance(step_ids, torch.Tensor):
            step_ids = torch.tensor(step_ids, dtype=input_ids.dtype)

        # unfold로 sliding windows 생성
        windows = input_ids.unfold(0, step_ids.numel(), 1)  # (N, len(B))
        matches = (windows == step_ids).all(dim=1)
        
        idx = torch.where(matches)[0]
        return int(idx[0].item()) if idx.numel() > 0 else None