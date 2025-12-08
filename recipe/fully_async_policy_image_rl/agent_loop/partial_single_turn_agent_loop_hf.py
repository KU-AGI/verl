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
import logging
import os
from typing import Any, Optional
from uuid import uuid4
import numpy as np

from recipe.fully_async_policy_image_rl.agent_loop.agent_loop_hf import AgentLoopOutput, FullyAsyncAgentLoopOutput
from recipe.fully_async_policy_image_rl.agent_loop.agent_loop import AgentLoopBase, register
from verl.utils.profiler import simple_timer
from verl import DataProto
import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("partial_single_turn_agent")
class PartialSingleTurnAgentLoop(AgentLoopBase):
    """
    Agent loop for image generation tasks using ImageUnifiedRollout.
    Instead of working with token IDs like text models, this works with DataProto
    containing multimodal data (prompts, images, etc.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Run the agent loop for image generation.
        For ImageUnifiedRollout, we prepare a DataProto instead of prompt_ids.

        Args:
            sampling_params: Sampling parameters for generation
            **kwargs: Should contain:
                - prompt: Text prompt for image generation
                - output: Optional previous output for resuming cancelled generation
                - param_version: Current parameter version
                - uid: Unique identifier for this sample

        Returns:
            FullyAsyncAgentLoopOutput containing the generation results
        """
        output: Optional[FullyAsyncAgentLoopOutput] = kwargs.get("output", None)
        prompt = kwargs.get("prompt", "")
        param_version = kwargs.get("param_version", 0)
        uid = kwargs.get("uid", str(uuid4()))

        metrics = {}
        request_id = uuid4().hex

        param_version_start = param_version
        param_version_end = param_version

        # For ImageUnifiedRollout, we need to prepare a DataProto with the prompt
        if not output or not output.is_cancel:
            # Create a new DataProto for this sample
            prompt_data = DataProto.from_dict(
                tensors={"dummy_tensor": torch.zeros(1)},
                non_tensors={
                    "prompt": np.array([prompt], dtype=object),
                    "uid": np.array([uid], dtype=object),
                },
                meta_info={
                    "validate": kwargs.get("validate", False),
                    "param_version": param_version,
                }
            )

            # If this is a resumed sample, we should include previous generation data
            if output and not output.is_cancel:
                # Sample already completed, return as-is
                return output
        else:
            # Resume cancelled generation
            # For now, we treat cancellation as needing to restart
            # (ImageUnifiedRollout doesn't support partial resume yet)
            prompt_data = DataProto.from_dict(
                tensors={"dummy_tensor": torch.zeros(1)},
                non_tensors={
                    "prompt": np.array([prompt], dtype=object),
                    "uid": np.array([uid], dtype=object),
                },
                meta_info={
                    "validate": kwargs.get("validate", False),
                    "param_version": param_version,
                }
            )
            metrics["generate_sequences"] = output.metrics.get("generate_sequences", 0.0)
            param_version_start = output.param_version_start

        # Call ImageUnifiedRollout via the server manager
        with simple_timer("generate_sequences", metrics):
            result_data, is_cancel = await self.server_manager.generate_for_partial(
                request_id=request_id,
                prompt_data=prompt_data,
                sampling_params=sampling_params,
            )

        # Process the result from ImageUnifiedRollout
        if is_cancel or result_data is None:
            # Generation was cancelled, return partial result
            return FullyAsyncAgentLoopOutput(
                prompt_ids=[],  # Empty for image generation
                response_ids=[],  # Empty for image generation
                response_mask=[],
                num_turns=1,
                metrics=metrics,
                is_cancel=True,
                log_probs=[],
                param_version_start=param_version_start,
                param_version_end=param_version,
            )

        return FullyAsyncAgentLoopOutput(
            prompt_ids=[],  # Not used for image generation
            response_ids=[],  # Not used for image generation
            response_mask=[1],  # Single token representing the full generation
            num_turns=1,
            metrics=metrics,
            is_cancel=False,
            log_probs=[],  # ImageUnifiedRollout doesn't compute log probs the same way
            param_version_start=param_version_start,
            param_version_end=param_version,
            # Store the actual result data for later processing
            generation_data=result_data,  # This will be a DataProto with images/text
        )
