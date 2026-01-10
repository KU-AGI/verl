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

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from .base import BaseRollout
from verl.single_controller.base.decorator import register, Dispatch

from transformers import GenerationConfig
import numpy as np
from typing import Union, List, Any, Optional, Dict, Tuple
from transformers.processing_utils import ProcessorMixin
import base64
from io import BytesIO
import PIL.Image
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask
import verl.utils.torch_functional as verl_F
import os
import json
from pathlib import Path
from torch.distributed.device_mesh import DeviceMesh
from verl.workers.config import HFModelConfig, RolloutConfig
from recipe.image_rl.config import ImageGenerationHFModelConfig
from recipe.image_rl.utils import FormattingEvaluatorV2
import asyncio
import logging
import time

__all__ = ['ImageUnifiedRollout']

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

EDIT_TEMPLATE="""You are a strict image editing assistant.
Your task is to revise a *failed* generated image according to the user's instruction and the original generation intent.

INPUT FORMAT:
1. The source image is located between {image_start_tag} and {image_end_tag}.
2. The original text-to-image generation prompt will be provided after the keyword 'INPUT_PROMPT:'
3. Step-by-step feedback will be provided after the keyword 'FEEDBACK:'.
   - The feedback will be a sequence of instructions, each starting with 'Step X:' (e.g., 'Step 1:', 'Step 2:', ...).
   - You MUST follow ALL steps in order and produce a final image that satisfies the entire sequence, not just an intermediate step.

CRITICAL RULES:
1. You MUST Look at the image between {image_start_tag} and {image_end_tag} as the ground truth.
2. Preserve the background, objects, and style from the input image unless explicitly asked to change them.
3. Do NOT generate a completely new image from scratch.
4. **You MUST strictly maintain the spatial layout and composition of the source image.**
5. You MUST also reference the INPUT_PROMPT as the original intended content of the image, but the visible source image remains the primary ground truth.
6. When applying FEEDBACK, carefully execute each step one by one while keeping previous changes consistent, and ensure the final result reflects all steps combined."""

class _HFModelWrapper:
    """
    Simple wrapper to provide vLLM-compatible interface for HuggingFace models.
    This allows ImageUnifiedRollout to be compatible with code expecting vLLM's inference_engine structure.
    """
    def __init__(self, module: nn.Module):
        self.worker = self._WorkerWrapper(module)

    class _WorkerWrapper:
        def __init__(self, module: nn.Module):
            self.model_runner = self._ModelRunnerWrapper(module)

        class _ModelRunnerWrapper:
            def __init__(self, module: nn.Module):
                self.model = self._ModelWithLoadWeights(module)

            class _ModelWithLoadWeights:
                def __init__(self, module: nn.Module):
                    self._module = module
                    self._weight_map = None
                    self._init_weight_map()
                
                def _init_weight_map(self):
                    actual_module = self._module
                    if hasattr(actual_module, "_fsdp_wrapped_module"):
                        actual_module = actual_module._fsdp_wrapped_module
                    
                    # 가중치 순서 보장을 위해 state_dict 키 정렬
                    current_sd = actual_module.state_dict()
                    sorted_keys = sorted(current_sd.keys())
                    
                    param_dict = dict(actual_module.named_parameters())
                    buffer_dict = dict(actual_module.named_buffers())
                    
                    self._weight_map = []
                    pointer = 0
                    
                    for name in sorted_keys:
                        target = param_dict.get(name)
                        if target is None:
                            target = buffer_dict.get(name)
                        
                        if target is None:
                            if name in current_sd and isinstance(current_sd[name], torch.Tensor):
                                pointer += current_sd[name].numel()
                            continue
                        
                        numel = target.numel()
                        self._weight_map.append({
                            'target': target,
                            'start': pointer,
                            'end': pointer + numel
                        })
                        pointer += numel
                        
                    self._total_numel = pointer
                    print(f"[ImageUnifiedRollout] Weight map initialized: {len(self._weight_map)} tensors, Total numel: {self._total_numel}")

                @torch.no_grad()
                def load_weights(self, flat_tensor: torch.Tensor, **kwargs):
                    if flat_tensor.numel() != self._total_numel:
                        raise ValueError(f"Size mismatch! Expected {self._total_numel}, got {flat_tensor.numel()}")

                    actual_module = self._module
                    if hasattr(actual_module, "_fsdp_wrapped_module"):
                        actual_module = actual_module._fsdp_wrapped_module
                    
                    device = next(actual_module.parameters()).device
                    gpu_flat_tensor = flat_tensor.to(device, non_blocking=True)
                    
                    for weight_info in self._weight_map:
                        target = weight_info['target']
                        source_slice = gpu_flat_tensor[weight_info['start']:weight_info['end']]
                        
                        target.copy_(source_slice.view_as(target))
                    
                    return {"loaded_elements": self._total_numel, "status": "success"}
                
                @torch.no_grad()
                def load_weights_from_path(self, file_path: str):

                    t0 = time.time()
                    try:
                        flat_tensor = torch.load(
                        file_path, 
                        map_location='cpu', 
                        mmap=True, 
                        weights_only=True
                    )

                    except Exception as e:
                        print(f"[ImageUnifiedRollout] Error loading binary weights from {file_path}: {e}")
                        raise e

                    result = self.load_weights(flat_tensor)
                    
                    dt = time.time() - t0
                    print(f"[ImageUnifiedRollout] Weights loaded from binary file {file_path} in {dt:.3f}s")
                    return result

class ImageUnifiedRollout(BaseRollout):
    def __init__(
        self,
        module: nn.Module,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = model_config.tokenizer
        # self.processor = model_config.processor
        self.module = module
        self.config = config
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dist.get_rank() == 0:
            print(f"[ImageUnifiedRollout] Using device: {self.device}")

        # Create inference_engine with vLLM-compatible interface for weight syncing
        self.inference_engine = _HFModelWrapper(module)

        self.generation_config = None

        # self.cfg_weight = getattr(config, "cfg_weight", 5.0)
        self.temperature = getattr(config, "temperature", 1.0)
        # self.txt_top_k = getattr(config, "txt_top_k", 50)
        # self.txt_top_p = getattr(config, "txt_top_p", 1.0)
        # self.img_top_k = getattr(config, "img_top_k", 4096)
        # self.img_top_p = getattr(config, "img_top_p", 1.0)
        self.top_k = getattr(config, "top_k", 50)
        self.top_p = getattr(config, "top_p", 1.0)

        self.prompt_length = getattr(config, "prompt_length", 1024)
        self.response_length = getattr(config, "response_length", 1024)

        # self.feedback_system_prompt = getattr(config, "feedback_system_prompt", "")
        # self.regen_system_prompt = getattr(config, "regen_system_prompt", "")
        self.regen_system_prompt = EDIT_TEMPLATE
        self.formatter = FormattingEvaluatorV2()

        self.image_token_num_per_image = getattr(config, "image_token_num_per_image", 576)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_node_id(self):
        import ray
        return ray.get_runtime_context().get_node_id()

    # [추가] 서버로부터 파일 경로를 받아 로드를 수행하는 진입점
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def apply_rollout_weights_from_path(self, version: int, file_path: str):
        return self.inference_engine.worker.model_runner.model.load_weights_from_path(file_path)

    def _pad_tensor_left(self, tensor: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
        """Left padding for input tensors"""
        if tensor.dim() == 2:  # [batch_size, seq_len]
            current_length = tensor.size(1)
            if current_length >= target_length:
                return tensor[:, -target_length:]
            
            pad_length = target_length - current_length
            padding = torch.full((tensor.size(0), pad_length), pad_value, 
                               dtype=tensor.dtype, device=tensor.device)
            return torch.cat([padding, tensor], dim=1)
        elif tensor.dim() == 3:  # [batch_size, seq_len, hidden_dim] - embeddings always use zero padding
            current_length = tensor.size(1)
            if current_length >= target_length:
                return tensor[:, :target_length, :]
            
            pad_length = target_length - current_length
            padding = torch.zeros((tensor.size(0), pad_length, tensor.size(2)), 
                                dtype=tensor.dtype, device=tensor.device)
            return torch.cat([padding, tensor], dim=1)
        else:
            return tensor

    def _pad_tensor_right(self, tensor: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
        """Right padding for output tensors"""
        if tensor.dim() == 2:  # [batch_size, seq_len]
            current_length = tensor.size(1)
            if current_length >= target_length:
                return tensor[:, :target_length]
            
            pad_length = target_length - current_length
            padding = torch.full((tensor.size(0), pad_length), pad_value, 
                               dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=1)
        elif tensor.dim() == 3:  # [batch_size, seq_len, hidden_dim] - embeddings always use zero padding
            current_length = tensor.size(1)
            if current_length >= target_length:
                return tensor[:, :target_length, :]
            
            pad_length = target_length - current_length
            padding = torch.zeros((tensor.size(0), pad_length, tensor.size(2)), 
                                dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=1)
        else:
            return tensor

    def _apply_padding_to_dataproto(self, data_proto: DataProto) -> DataProto:
        """Apply appropriate padding to all tensors in DataProto"""

        # Skip Task 1 - Image Generation (outputs) tokens are always same length (576)

        # Task 2 - Text Generation (inputs)
        input_tensors_task2 = [
            "task2_input_ids", "task2_attention_mask", "task2_position_ids"
        ]
        for key in input_tensors_task2:
            if key in data_proto.batch:
                if ("mask" in key) or ("position" in key):
                    pad_value = 0  # attention masks and embeddings use 0 padding
                else:
                    pad_value = self.tokenizer.pad_token_id
                data_proto.batch[key] = self._pad_tensor_left(
                    data_proto.batch[key], self.prompt_length, pad_value
                )

        # Task 2 - Text Generation (outputs)
        output_tensors_task2 = [
            "task2_feedback_ids", "task2_response_mask", "task2_response_position_ids"
        ]
        for key in output_tensors_task2:
            if key in data_proto.batch:
                if ("mask" in key) or ("position" in key):
                    pad_value = 0  # attention masks and embeddings use 0 padding
                else:
                    pad_value = self.tokenizer.eos_token_id
                data_proto.batch[key] = self._pad_tensor_right(
                    data_proto.batch[key], self.response_length, pad_value
                )

        data_proto.batch["prompts"] = data_proto.batch["task2_input_ids"]
        data_proto.batch["responses"] = data_proto.batch["task2_feedback_ids"]
        data_proto.batch["input_ids"] = torch.cat([data_proto.batch["task2_input_ids"], data_proto.batch["task2_feedback_ids"]], dim=1)
        data_proto.batch["attention_mask"] = torch.cat([data_proto.batch["task2_attention_mask"], data_proto.batch["task2_response_mask"]], dim=1)
        data_proto.batch["position_ids"] = torch.cat([data_proto.batch["task2_position_ids"], data_proto.batch["task2_response_position_ids"]], dim=1)

        return data_proto

    async def resume(self, tags: list[str]):
        return None

    async def release(self):
        torch.cuda.empty_cache()
        return None

    async def update_weights(self, weights, **kwargs):
        stats = self.inference_engine.worker.model_runner.model.load_weights(weights, **kwargs)
        return stats

    def set_generation_config(self, prompt: DataProto):
        is_validate = prompt.meta_info.get("validate", False)

        if is_validate:
            # Validation mode: use val_kwargs
            # self.cfg_weight = getattr(self.config.val_kwargs, 'val_cfg_weight', 5.0)
            self.temperature = getattr(self.config.val_kwargs, 'temperature', 1.0)
            # self.txt_top_k = getattr(self.config.val_kwargs, 'val_txt_top_k', 50)
            # self.txt_top_p = getattr(self.config.val_kwargs, 'val_txt_top_p', 1.0)
            # self.img_top_k = getattr(self.config.val_kwargs, 'val_img_top_k', 4096)
            # self.img_top_p = getattr(self.config.val_kwargs, 'val_img_top_p', 1.0)
            self.top_k = getattr(self.config.val_kwargs, 'top_k', 50)
            self.top_p = getattr(self.config.val_kwargs, 'top_p', 1.0)
        else:
            # Training mode: use config values (already set in __init__)
            # self.cfg_weight = getattr(self.config, "cfg_weight", 5.0)
            self.temperature = getattr(self.config, "temperature", 1.0)
            # self.txt_top_k = getattr(self.config, "txt_top_k", 50)
            # self.txt_top_p = getattr(self.config, "txt_top_p", 1.0)
            # self.img_top_k = getattr(self.config, "img_top_k", 4096)
            # self.img_top_p = getattr(self.config, "img_top_p", 1.0)
            self.top_k = getattr(self.config, "top_k", 50)
            self.top_p = getattr(self.config, "top_p", 0.7)

        # # Additional settings
        # self.image_start_tag = self.processor.image_start_tag
        # self.image_end_tag = self.processor.image_end_tag
        # self.image_tag = self.processor.image_tag

    def get_sft_format(self, prompt):
        sft_format = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return sft_format

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.module.eval()

        batch_size = prompts.batch.batch_size[0]

        self.set_generation_config(prompts[0])

        task_funcs = {
            2: self._generate_minibatch_text_generation,
        }

        # Check for task_id in batch
        task_id_tensor = prompts.batch.get("task_id", None)
        if task_id_tensor is None:
            selected_funcs = task_funcs.values()
        else:
            task_id = task_id_tensor.view(-1)[0].item()
            selected_funcs = [task_funcs.get(task_id)]

        for i, func in enumerate(selected_funcs):
            if func is not None:
                prompts = func(prompts)

        # Apply padding to all tensors before moving to CPU
        prompts = self._apply_padding_to_dataproto(prompts)
        _batch = prompts.pop(batch_keys=[key for key in prompts.batch.keys() if "embeds" not in key])

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train()
        return DataProto(batch=_batch.batch, non_tensor_batch=prompts.non_tensor_batch, meta_info=prompts.meta_info)

    def _generate_minibatch_text_generation(self, data_proto: DataProto) -> DataProto:
        batch_size = data_proto.batch.batch_size[0]
        if dist.get_rank() == 0:
            print(f"[TEXT_GEN] Input batch_size: {batch_size}")

        # Prepare messages for all images
        input_format = []
        for prompt in data_proto.non_tensor_batch['prompt']:
            # _prompt = self.get_sft_format(prompt) # Already formatted!!!
            input_format.append(prompt)
        
        inputs = self.tokenizer(
            input_format,
            padding_side='left',
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        inputs = inputs.to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill_(attention_mask == 0, 0)

        # For generating feedback texts
        data_proto.batch["task2_input_ids"] = input_ids
        data_proto.batch["task2_attention_mask"] = attention_mask
        data_proto.batch["task2_position_ids"] = position_ids

        feedback_texts = self.generate_text(inputs, position_ids)
        
        data_proto.non_tensor_batch["task2_feedback_texts"] = np.array(feedback_texts, dtype=object)
        outputs = self.tokenizer(
            feedback_texts, 
            padding=True, 
            padding_side='right', 
            return_tensors="pt", 
            add_special_tokens=False
        )
        data_proto.batch["task2_feedback_ids"] = outputs["input_ids"].to(self.device)
        data_proto.batch["task2_response_mask"] = outputs["attention_mask"].to(self.device)
        
        response_length = data_proto.batch["task2_feedback_ids"].size(1)
        batch_size = data_proto.batch["task2_feedback_ids"].size(0)
        
        # delta_position_id: [1, 2, 3, ..., response_length]
        delta_position_id = torch.arange(1, response_length + 1, device=self.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        
        # input의 마지막 position에서 이어서 계산
        response_position_ids = data_proto.batch["task2_position_ids"][:, -1:] + delta_position_id
        
        # mask가 0인 부분은 0으로 설정
        response_position_ids = response_position_ids.masked_fill_(
            data_proto.batch["task2_response_mask"] == 0, 0
        )
        
        data_proto.batch["task2_response_position_ids"] = response_position_ids
        if dist.get_rank() == 0:
            print(f"[TEXT_GEN] Completed feedback generation")

        torch.cuda.empty_cache()
        return data_proto

    @torch.no_grad()
    def generate_text(self, inputs: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        input_length = inputs['input_ids'].shape[1]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output_ids = self.module.generate(
                **inputs,
                max_new_tokens=self.response_length,
                use_cache=True,
                do_sample=True,
                temperature=self.temperature,
                top_k=int(self.top_k),
                top_p=float(self.top_p),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[:, input_length:]
        answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return answer
