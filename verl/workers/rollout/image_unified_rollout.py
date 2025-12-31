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
from recipe.image_rl.utils import FormattingEvaluator
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
        self.processor = model_config.processor
        self.module = module
        self.config = config
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dist.get_rank() == 0:
            print(f"[ImageUnifiedRollout] Using device: {self.device}")

        # Create inference_engine with vLLM-compatible interface for weight syncing
        self.inference_engine = _HFModelWrapper(module)

        self.generation_config = None

        self.cfg_weight = getattr(config, "cfg_weight", 5.0)
        self.temperature = getattr(config, "temperature", 1.0)
        self.txt_top_k = getattr(config, "txt_top_k", 50)
        self.txt_top_p = getattr(config, "txt_top_p", 1.0)
        self.img_top_k = getattr(config, "img_top_k", 4096)
        self.img_top_p = getattr(config, "img_top_p", 1.0)

        self.img_size = 384
        self.patch_size = 16

        self.prompt_length = getattr(config, "prompt_length", 1024)
        self.response_length = getattr(config, "response_length", 1024)

        self.feedback_system_prompt = getattr(config, "feedback_system_prompt", "")
        # self.regen_system_prompt = getattr(config, "regen_system_prompt", "")
        self.regen_system_prompt = EDIT_TEMPLATE
        self.formatter = FormattingEvaluator()

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
        
        # Task 1 - Image Generation (inputs)
        input_tensors_task1 = [
            "task1_input_ids", "task1_attention_mask"
        ]
        for key in input_tensors_task1:
            if key in data_proto.batch:
                if "mask" in key:
                    pad_value = 0  # attention masks and embeddings use 0 padding
                else:
                    pad_value = self.processor.pad_id
                data_proto.batch[key] = self._pad_tensor_left(
                    data_proto.batch[key], self.prompt_length, pad_value
                )
        
        # Skip Task 1 - Image Generation (outputs) tokens are always same length (576)

        # Task 2 - Text Generation (inputs)
        input_tensors_task2 = [
            "task2_input_ids", "task2_attention_mask"
        ]
        for key in input_tensors_task2:
            if key in data_proto.batch:
                if "mask" in key:
                    pad_value = 0  # attention masks and embeddings use 0 padding
                else:
                    pad_value = self.processor.pad_id
                data_proto.batch[key] = self._pad_tensor_left(
                    data_proto.batch[key], self.prompt_length, pad_value
                )

        # Task 2 - Text Generation (outputs)
        output_tensors_task2 = [
            "task2_feedback_ids", "task2_response_mask"
        ]
        for key in output_tensors_task2:
            if key in data_proto.batch:
                if "mask" in key:
                    pad_value = 0  # attention masks and embeddings use 0 padding
                else:
                    pad_value = self.processor.tokenizer.eos_token_id
                data_proto.batch[key] = self._pad_tensor_right(
                    data_proto.batch[key], self.response_length, pad_value
                )

        # Task 3 - Regen Image Generation (inputs)
        input_tensors_task3 = [
            "task3_input_ids", "task3_attention_mask"
        ]
        for key in input_tensors_task3:
            if key in data_proto.batch:
                if "mask" in key:
                    pad_value = 0  # attention masks and embeddings use 0 padding
                else:
                    pad_value = self.processor.pad_id
                data_proto.batch[key] = self._pad_tensor_left(
                    data_proto.batch[key], self.prompt_length, pad_value
                )

        # Skip Task 3 - Regen Image Generation (outputs) tokens are always same length (576)

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
        do_sample = prompt.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompt.meta_info.get("validate", False)

        temperature = prompt.meta_info.get("temperature", self.config.temperature)
        response_length = prompt.meta_info.get("response_length", self.config.response_length)
        top_p = prompt.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompt.meta_info.get("top_k", self.config.get("top_k", 0)))
        
        if not do_sample:
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,  # Set to 1 for for-loop processing
            }
        elif is_validate:
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # Set to 1 for for-loop processing
            }
        else:
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": 1,  # Set to 1 for for-loop processing
            }

        # Additional settings
        self.image_start_tag = self.processor.image_start_tag
        self.image_end_tag = self.processor.image_end_tag
        self.image_tag = self.processor.image_tag

    def get_sft_format(self, prompt, system_prompt=""):
        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=[{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}],
            sft_format=self.processor.sft_format,
            system_prompt=system_prompt.format(image_start_tag=self.image_start_tag, image_end_tag=self.image_end_tag) if system_prompt else "",
        )
        sft_format = sft_format + self.image_start_tag
        return sft_format

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.module.eval()

        batch_size = prompts.batch.batch_size[0]

        self.set_generation_config(prompts[0])

        input_format = [self.get_sft_format(prompt) for prompt in prompts.non_tensor_batch["prompt"]]

        self.processor.tokenizer.pad_token_id = self.processor.pad_id

        inputs = self.processor.tokenizer(
            input_format,
            padding=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        prompts.batch["task1_input_ids"] = input_ids
        prompts.batch["task1_attention_mask"] = attention_mask

        task_funcs = {
            1: self._generate_minibatch_image_generation,
            2: self._generate_minibatch_text_generation,
            3: self._generate_minibatch_regen_image_generation,
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

    @torch.no_grad()
    def _generate_minibatch_image_generation(self, data_proto: DataProto) -> DataProto:
        batch_size = data_proto.batch.batch_size[0]
        
        input_ids = data_proto.batch["task1_input_ids"]
        attention_mask = data_proto.batch["task1_attention_mask"]

        # embedding
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_embeds = self.module.language_model.get_input_embeddings()(input_ids)
            input_embeds = input_embeds.to(dtype=torch.bfloat16)

        # For computing logits: input (especially input text embedding)
        data_proto.batch["task1_input_embeds"] = input_embeds

        gen_final_cfg_embeds, gen_final_cfg_attention_mask = self._prepare_cfg_embeds(data_proto)
        generated_tokens = self.generate_img(gen_final_cfg_embeds, gen_final_cfg_attention_mask)
        # For reproducing generated images
        data_proto.batch["task1_gen_img_tokens"] = generated_tokens

        decoded_images = self._decode_image_tokens(generated_tokens)
        gen_imgs_pil_list = [PIL.Image.fromarray(img_array) for img_array in decoded_images]
        data_proto.non_tensor_batch["task1_gen_imgs_pil_list"] = np.array(gen_imgs_pil_list, dtype=object)
        gen_imgs_tensor = self.processor.image_processor(gen_imgs_pil_list).pixel_values
        # For generating feedback texts and computing logits: output
        data_proto.batch["task1_gen_imgs_pixel_values"] = gen_imgs_tensor.cpu()
        gen_imgs_pixel_values = gen_imgs_tensor.to(self.device, dtype=torch.bfloat16)

        # Postprocessing output embeds
        B, C, H, W = gen_imgs_pixel_values.shape

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, _, all_image_ids = self.module.gen_vision_model.encode(gen_imgs_pixel_values)
    
            image_ids = all_image_ids[2]
            image_ids = image_ids.view(B, -1)

            image_embeds = self.module.gen_aligner(self.module.gen_embed(image_ids))

        data_proto.batch["task1_response_mask"] = torch.ones((B, image_embeds.size(1)), dtype=torch.long, device=image_embeds.device)
        if dist.get_rank() == 0:
            print(f"[IMG_GEN] Created DataProto with batch_size: {batch_size}")

        torch.cuda.empty_cache()
        return data_proto

    def expand_image_placeholders(self, input_ids_tensor, gen_imgs_pixel_values):
        image_id = self.processor.image_id
        pad_id = self.processor.pad_id
        k = self.image_token_num_per_image

        processed_sequences = []
        shifted_output_start_indices = []
        shifted_all_image_start_indices = []
        images_to_batch = []

        tmp_output_start = []
        tmp_img_starts = []
        lengths = []

        for input_ids, images in zip(input_ids_tensor, gen_imgs_pixel_values):
            mask = (input_ids == image_id)
            num_images = int(mask.sum().item())

            output_start = int(len(input_ids) + num_images * (k - 1))
            tmp_output_start.append(output_start)

            counts = torch.where(
                mask,
                torch.full_like(input_ids, k),
                torch.ones_like(input_ids),
            )

            expanded_seq = input_ids.repeat_interleave(counts)

            starts = counts.cumsum(0) - counts
            img_starts = starts[mask].tolist()
            tmp_img_starts.append(img_starts)

            processed_sequences.append(expanded_seq)
            lengths.append(expanded_seq.numel())
            images_to_batch.append(images)

        # 2) left padding + shift 보정
        max_len = max(lengths)
        B = len(processed_sequences)

        batched_total_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=self.device)

        for i, seq in enumerate(processed_sequences):
            length = seq.numel()
            shift = max_len - length
            batched_total_ids[i, shift:] = seq

            shifted_output_start_indices.append(tmp_output_start[i] + shift)
            shifted_all_image_start_indices.append([p + shift for p in tmp_img_starts[i]])

        batched_attention_mask = (batched_total_ids != pad_id).to(dtype=torch.long)
        images_to_batch = torch.cat(images_to_batch, dim=0)

        return batched_total_ids, batched_attention_mask, shifted_output_start_indices, shifted_all_image_start_indices, images_to_batch

    def merge_text_and_image_embeds(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor, all_image_start_indices: List[List[int]]):

        batch_size = text_embeds.size(0)
        num_img = len(all_image_start_indices[0])
        reshape_image_embeds = image_embeds.view(-1, num_img, self.image_token_num_per_image, image_embeds.size(-1))

        assert text_embeds.size(0) == reshape_image_embeds.size(0) == len(all_image_start_indices)

        merged_embeds = text_embeds.clone()

        for i in range(batch_size):
            for j in range(num_img):
                start_idx = all_image_start_indices[i][j]
                end_idx = start_idx + self.image_token_num_per_image
                merged_embeds[i, start_idx:end_idx] = reshape_image_embeds[i, j]
    
        return merged_embeds

    def _generate_minibatch_text_generation(self, data_proto: DataProto) -> DataProto:
        batch_size = data_proto.batch.batch_size[0]
        if dist.get_rank() == 0:
            print(f"[TEXT_GEN] Input batch_size: {batch_size}")
        
        # Get images from batch
        gen_imgs_pixel_values = data_proto.batch.get('task1_gen_imgs_pixel_values', [])
        if len(gen_imgs_pixel_values) == 0:
            raise ValueError("No images found in batch['task1_gen_imgs_pixel_values']")
        
        # Process all images in batch
        if dist.get_rank() == 0:
            print(f"[TEXT_GEN] Processing feedback for {len(gen_imgs_pixel_values)} images in batch")

        # Prepare messages for all images
        input_format = []
        for prompt in data_proto.non_tensor_batch['prompt']:
            _prompt = self.get_sft_format(prompt)
            input_format.append(_prompt + self.image_tag + self.image_end_tag + "\nFirst, Decompose input prompt\n")

        self.processor.tokenizer.pad_token_id = self.processor.pad_id
        
        inputs = self.processor.tokenizer(
            input_format,
            padding=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        batched_total_ids, batched_attention_mask, output_start_indices, all_image_start_indices, images_to_batch = self.expand_image_placeholders(input_ids, gen_imgs_pixel_values)

        # embedding
        gen_imgs_pixel_values = gen_imgs_pixel_values.to(self.device, dtype=torch.bfloat16)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            text_embeds = self.module.language_model.get_input_embeddings()(batched_total_ids)
            image_embeds = self.module.aligner(self.module.vision_model(gen_imgs_pixel_values))
            text_embeds = text_embeds.to(dtype=torch.bfloat16)
            image_embeds = image_embeds.to(dtype=torch.bfloat16)

        # merge text and image embeds
        merged_embeds = self.merge_text_and_image_embeds(text_embeds, image_embeds, all_image_start_indices)

        # For generating feedback texts
        data_proto.batch["task2_input_ids"] = batched_total_ids
        data_proto.batch["task2_attention_mask"] = batched_attention_mask
        data_proto.batch["task2_input_embeds"] = merged_embeds

        feedback_texts = self.generate_text(merged_embeds, batched_attention_mask)

        # For computing logits: output
        data_proto.non_tensor_batch["task2_feedback_texts"] = np.array(feedback_texts, dtype=object)
        self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id # Right padding w/ eos_token_id
        outputs = self.processor.tokenizer(feedback_texts, padding=True, padding_side='right', return_tensors="pt")
        data_proto.batch["task2_feedback_ids"] = outputs["input_ids"]

        feedback_ids = data_proto.batch["task2_feedback_ids"]
        feedback_ids = feedback_ids.to(self.device)

        # Postprocessing output embeds
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            text_embeds = self.module.language_model.get_input_embeddings()(feedback_ids)
            text_embeds = text_embeds.to(dtype=torch.bfloat16)

        data_proto.batch["task2_response_mask"] = outputs["attention_mask"]
        if dist.get_rank() == 0:
            print(f"[TEXT_GEN] Completed feedback generation")

        torch.cuda.empty_cache()
        return data_proto

    def _generate_minibatch_regen_image_generation(self, data_proto: DataProto) -> DataProto:
        batch_size = data_proto.batch.batch_size[0]

        if dist.get_rank() == 0:
            print(f"[REGEN] Input batch_size: {batch_size}")

        # Get data from batch
        gen_imgs_pixel_values = data_proto.batch.get('task1_gen_imgs_pixel_values', [])
        feedback_texts = data_proto.non_tensor_batch.get('task2_feedback_texts', [])
        
        if len(gen_imgs_pixel_values) == 0:
            raise ValueError("No images found in batch['task1_gen_imgs_pixel_values']")
        if len(feedback_texts) == 0:
            raise ValueError("No feedback texts found in non_tensor_batch['task2_feedback_texts']")

        if dist.get_rank() == 0:
            print(f"[REGEN] Loaded {len(gen_imgs_pixel_values)} images and {len(feedback_texts)} feedback_texts from non_tensor_batch")

        # Process all images in batch
        if dist.get_rank() == 0:
            print(f"[REGEN] Processing regen for {batch_size} images in batch")

        # Parse feedback texts
        prompts = [prompt for prompt in data_proto.non_tensor_batch['prompt']]
        feedback_texts = [self.formatter._split_text_into_parts(feedback)[-1] for feedback in data_proto.non_tensor_batch['task2_feedback_texts']]

        # Prepare messages for all images
        input_format = []
        for (prompt, feedback) in zip(prompts, feedback_texts): # data_proto.non_tensor_batch['prompt'] 들어가야함
            _prefix =(
                f"{self.image_start_tag}{self.image_tag}{self.image_end_tag}\n"
                "Please edit the image as instructed.\n"
                "The FEEDBACK will be given as multiple steps (Step 1, Step 2, ...). "
                "You MUST apply all steps in order and produce a final image reflecting all changes.\n"
                "INPUT_PROMPT: {prompt}\n"
                "FEEDBACK: \n{feedback}")
            if feedback is None:
                prefix = self.get_sft_format(_prefix.format(prompt=prompt, feedback="No need to generate feedback."), system_prompt=self.regen_system_prompt)
            else:
                prefix = self.get_sft_format(_prefix.format(prompt=prompt, feedback=feedback), system_prompt=self.regen_system_prompt)
            input_format.append(prefix) # Add image placeholder at the end

        self.processor.tokenizer.pad_token_id = self.processor.pad_id

        inputs = self.processor.tokenizer(
            input_format,
            padding=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        batched_total_ids, batched_attention_mask, output_start_indices, all_image_start_indices, images_to_batch = self.expand_image_placeholders(input_ids, gen_imgs_pixel_values)

        B, C, H, W = gen_imgs_pixel_values.shape

        gen_imgs_pixel_values = gen_imgs_pixel_values.to(self.device, dtype=torch.bfloat16)

        # embedding
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, _, all_image_ids = self.module.gen_vision_model.encode(gen_imgs_pixel_values)
    
            image_ids = all_image_ids[2]
            image_ids = image_ids.view(B, -1)

            image_embeds = self.module.gen_aligner(self.module.gen_embed(image_ids))
            text_embeds = self.module.language_model.get_input_embeddings()(batched_total_ids)
            image_embeds = image_embeds.to(dtype=torch.bfloat16)
            text_embeds = text_embeds.to(dtype=torch.bfloat16)

        # merge text and image embeds
        merged_embeds = self.merge_text_and_image_embeds(text_embeds, image_embeds, all_image_start_indices)

        # For generating regen images
        data_proto.batch["task3_input_ids"] = batched_total_ids
        data_proto.batch["task3_attention_mask"] = batched_attention_mask
        data_proto.batch["task3_input_embeds"] = merged_embeds

        regen_final_cfg_embeds, regen_final_cfg_attention_mask = self._prepare_regen_cfg_embeds(data_proto)
        regenerated_tokens = self.generate_img(regen_final_cfg_embeds, regen_final_cfg_attention_mask)
        # For reproducing regen images
        data_proto.batch["task3_regen_img_tokens"] = regenerated_tokens

        regen_decoded_images = self._decode_image_tokens(regenerated_tokens)
        regen_imgs_pil_list = [PIL.Image.fromarray(img_array) for img_array in regen_decoded_images]
        data_proto.non_tensor_batch["task3_regen_imgs_pil_list"] = np.array(regen_imgs_pil_list, dtype=object)
        regen_imgs_tensor = self.processor.image_processor(regen_imgs_pil_list).pixel_values
        # For generating feedback texts and computing logits: output
        data_proto.batch["task3_regen_imgs_pixel_values"] = regen_imgs_tensor.cpu()
        regen_imgs_pixel_values = regen_imgs_tensor.to(self.device, dtype=torch.bfloat16)

        # Postprocessing output embeds
        B, C, H, W = regen_imgs_pixel_values.shape

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, _, all_image_ids = self.module.gen_vision_model.encode(regen_imgs_pixel_values)
    
            image_ids = all_image_ids[2]
            image_ids = image_ids.view(B, -1)

            image_embeds = self.module.gen_aligner(self.module.gen_embed(image_ids))

        data_proto.batch["task3_response_mask"] = torch.ones((B, image_embeds.size(1)), dtype=torch.long, device=image_embeds.device)
        if dist.get_rank() == 0:
            print(f"[IMG_GEN] Created DataProto with batch_size: {batch_size}")

        torch.cuda.empty_cache()
        return data_proto

    def _prepare_cfg_embeds(self, data_proto: DataProto) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = data_proto.batch['task1_input_ids']
        attention_mask = data_proto.batch['task1_attention_mask']
        input_embeds = data_proto.batch['task1_input_embeds']

        cond_embeds = input_embeds
        uncond_embeds = input_embeds.clone()

        pad_id = torch.tensor(self.processor.pad_id, device=input_ids.device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pad_embed = self.module.language_model.get_input_embeddings()(pad_id).unsqueeze(0)
            pad_embed = pad_embed.to(dtype=torch.bfloat16)

        start_marker = torch.tensor([100601], device=input_ids.device) # <|User|>
        end_marker = torch.tensor([100602], device=input_ids.device) # <|Assistant|>

        def find_sequence(inp_id, start_marker):
            len_needle = start_marker.shape[0]
            for i in range(inp_id.shape[0] - len_needle + 1):
                if torch.equal(inp_id[i:i+len_needle], start_marker):
                    return i # Return the starting index of the match
            return -1 # Return -1 if not found

        for i, row in enumerate(input_ids):
            start_pos = find_sequence(row, start_marker)
            end_pos = find_sequence(row, end_marker)
            
            if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
                content_start_index = start_pos
                content_end_index = end_pos + 2

                if content_start_index < content_end_index:
                    uncond_embeds[i, content_start_index:content_end_index] = pad_embed
        
        batch_size, seq_len, embed_dim = input_embeds.shape
        
        gen_final_cfg_embeds = pad_embed.expand(batch_size * 2, seq_len, embed_dim).clone()
        gen_final_cfg_attention_mask = torch.zeros((batch_size * 2, seq_len), dtype=torch.long, device=attention_mask.device)
        
        gen_final_cfg_embeds[0::2] = cond_embeds
        gen_final_cfg_embeds[1::2] = uncond_embeds
        
        gen_final_cfg_attention_mask[0::2] = attention_mask
        gen_final_cfg_attention_mask[1::2] = attention_mask 
            
        return gen_final_cfg_embeds, gen_final_cfg_attention_mask

    def _decode_image_tokens(self, generated_tokens: torch.Tensor) -> np.ndarray:
        batch_size = generated_tokens.shape[0]
        
        shape = [batch_size, 8, self.img_size // self.patch_size, self.img_size // self.patch_size]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            dec = self.module.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.long), shape=shape)
        
        dec = dec.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return dec

    def _sample_from_logits(self, logits: torch.Tensor, is_text: bool, generator: torch.Generator = None) -> torch.Tensor:

        if is_text:
            top_k = self.txt_top_k
            top_p = self.txt_top_p
        else:
            top_k = self.img_top_k
            top_p = self.img_top_p

        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        
        probs = torch.softmax(logits / self.temperature, dim=-1)
        
        if top_p:
            probs_sort, probs_idx = torch.sort(probs,
                                            dim=-1,
                                            descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p 
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # normalize
            next_token = torch.multinomial(probs_sort, num_samples=1, generator=generator)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            
        return next_token 

    @torch.no_grad()
    def generate_img(self, inputs_embeds: torch.Tensor, attention_masks: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        batch_size = inputs_embeds.shape[0] // 2
        generated_tokens = torch.zeros((batch_size, self.image_token_num_per_image), dtype=torch.int, device=self.device)

        attention_mask = attention_masks

        position_ids = attention_mask.long().cumsum(1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        past_len = attention_mask.sum(dim=1, keepdim=True).long()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            past_key_values = None
            for i in range(self.image_token_num_per_image):
                outputs = self.module.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                hidden_states = outputs.last_hidden_state
                past_key_values = outputs.past_key_values

                logits = self.module.gen_head(hidden_states[:, -1, :])
                
                # CFG
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                
                logits = logit_uncond + self.cfg_weight * (logit_cond-logit_uncond)
                next_token = self._sample_from_logits(logits, is_text=False, generator=generator)
                generated_tokens[:, i] = next_token.squeeze(-1)

                next_token_pair = next_token.repeat(1, 2).view(-1)
                inputs_embeds = self.module.prepare_gen_img_embeds(next_token_pair).unsqueeze(1)
                inputs_embeds = inputs_embeds.to(dtype=torch.bfloat16)

                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=self.device)], dim=1)

                position_ids = past_len.clone()
                past_len += 1

        return generated_tokens

    @torch.no_grad()
    def generate_text(self, inputs_embeds: torch.Tensor, attention_masks: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.module.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_masks,
                max_new_tokens=self.response_length,
                use_cache=True,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
            )

        answer = self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return answer

    def _prepare_regen_cfg_embeds(self, data_proto: DataProto) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = data_proto.batch['task3_input_ids']
        attention_mask = data_proto.batch['task3_attention_mask']
        input_embeds = data_proto.batch['task3_input_embeds']

        cond_embeds = input_embeds
        uncond_embeds = input_embeds.clone()

        pad_id = torch.tensor(self.processor.pad_id, device=input_ids.device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pad_embed = self.module.language_model.get_input_embeddings()(pad_id).unsqueeze(0)
            pad_embed = pad_embed.to(dtype=torch.bfloat16)

        start_marker = torch.tensor([100593, 185], device=input_ids.device) # <end_of_image>\n
        end_marker = torch.tensor([100602], device=input_ids.device) # <|Assistant|>

        def find_sequence(inp_id, start_marker):
            len_needle = start_marker.shape[0]
            for i in range(inp_id.shape[0] - len_needle + 1):
                if torch.equal(inp_id[i:i+len_needle], start_marker):
                    return i # Return the starting index of the match
            return -1 # Return -1 if not found

        for i, row in enumerate(input_ids):
            start_pos = find_sequence(row, start_marker)
            end_pos = find_sequence(row, end_marker)
            
            if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
                content_start_index = start_pos + 1
                content_end_index = end_pos + 2

                if content_start_index < content_end_index:
                    uncond_embeds[i, content_start_index:content_end_index] = pad_embed
        
        batch_size, seq_len, embed_dim = input_embeds.shape
        
        regen_final_cfg_embeds = pad_embed.expand(batch_size * 2, seq_len, embed_dim).clone()
        regen_final_cfg_attention_mask = torch.zeros((batch_size * 2, seq_len), dtype=torch.long, device=attention_mask.device)
        
        regen_final_cfg_embeds[0::2] = cond_embeds
        regen_final_cfg_embeds[1::2] = uncond_embeds
        
        regen_final_cfg_attention_mask[0::2] = attention_mask
        regen_final_cfg_attention_mask[1::2] = attention_mask 
            
        return regen_final_cfg_embeds, regen_final_cfg_attention_mask