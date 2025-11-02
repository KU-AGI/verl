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

import contextlib
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from .base import BaseRollout

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

__all__ = ['ImageUnifiedRollout']

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

        self.generation_config = None

        self.cfg_weight = getattr(config, "cfg_weight", 5.0)
        self.temperature = getattr(config, "temperature", 1.0)
        self.txt_top_k = getattr(config, "txt_top_k", None)
        self.txt_top_p = getattr(config, "txt_top_p", None)
        self.img_top_k = getattr(config, "img_top_k", None)
        self.img_top_p = getattr(config, "img_top_p", None)

        self.img_size = 384
        self.patch_size = 16

        self.prompt_length = getattr(config, "prompt_length", 1024)
        self.response_length = getattr(config, "response_length", 1024)

        self.feedback_system_prompt = getattr(config, "feedback_system_prompt", "")
        self.regen_system_prompt = getattr(config, "regen_system_prompt", "")
        self.formatter = FormattingEvaluator()

        self.image_token_num_per_image = getattr(config, "image_token_num_per_image", 576)
        self.max_reflect_len = getattr(config, "max_reflect_len", 1024)

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
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        return None

    async def update_weights(self, weights, **kwargs):
        for _ in weights:
            pass
        return None

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

        self.generation_config = GenerationConfig(**kwargs)

        boi_token_id = self.processor.tokenizer.convert_tokens_to_ids('<begin_of_image>')
        eoi_token_id = self.processor.tokenizer.convert_tokens_to_ids('<end_of_image>')
        
        if boi_token_id is not None:
            if not hasattr(self.generation_config, 'generation_kwargs'):
                self.generation_config.generation_kwargs = {}
            self.generation_config.generation_kwargs['boi_token_id'] = boi_token_id
            
        if eoi_token_id is not None:
            self.generation_config.generation_kwargs['eoi_token_id'] = eoi_token_id
            
        if self.generation_config.pad_token_id is None:
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                self.generation_config.pad_token_id = pad_token_id
            else:
                eos_token_id = self.processor.tokenizer.eos_token_id
                if eos_token_id is not None:
                    self.generation_config.pad_token_id = eos_token_id
    
        # Additional settings
        self.image_start_tag = self.processor.image_start_tag
        self.image_end_tag = self.processor.image_end_tag
        self.image_tag = self.processor.image_tag

    def get_sft_format(self, prompt):
        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=[{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}],
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        sft_format = sft_format + self.image_start_tag
        return sft_format

    @torch.inference_mode()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.module.eval()

        batch_size = prompts.batch.batch_size[0]

        self.set_generation_config(prompts[0])

        input_format = [self.get_sft_format(prompt) for prompt in prompts.non_tensor_batch["prompt"]]

        inputs = self.processor.tokenizer(
            input_format,
            padding=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        prompts.batch["task1_input_ids"] = input_ids
        prompts.batch["task1_attention_mask"] = attention_mask

        prompts = self._generate_minibatch_image_generation(prompts)
        prompts = self._generate_minibatch_text_generation(prompts)
        prompts = self._generate_minibatch_regen_image_generation(prompts)

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
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)

        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                input_embeds = self.module.language_model.get_input_embeddings()(input_ids)

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
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)

        B, C, H, W = gen_imgs_pixel_values.shape

        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, _, all_image_ids = self.module.gen_vision_model.encode(gen_imgs_pixel_values)
        
                image_ids = all_image_ids[2]
                image_ids = image_ids.view(B, -1)

                image_embeds = self.module.gen_aligner(self.module.gen_embed(image_ids))

        data_proto.batch["task1_response_mask"] = torch.ones((B, image_embeds.size(1)), dtype=torch.long, device=image_embeds.device)
        print(f"[IMG_GEN] Created DataProto with batch_size: {batch_size}")

        return data_proto

    def expand_image_placeholders(self, input_ids_tensor, gen_imgs_pixel_values):
        processed_sequences = []
        all_image_start_indices = []
        output_start_indices = []
        images_to_batch = []

        for input_ids, images in zip(input_ids_tensor, gen_imgs_pixel_values):

            # Find positions of image placeholders
            img_positions = (input_ids == self.processor.image_id).nonzero(as_tuple=True)[0]
            num_images_in_input = len(img_positions)

            # output_start_index
            output_start_idx = len(input_ids) + num_images_in_input * (self.image_token_num_per_image - 1)
            output_start_indices.append(output_start_idx)

            # expand sequence
            mask = (input_ids == self.processor.image_id)
            counts = torch.where(mask, self.image_token_num_per_image, 1)
            expanded_len = counts.sum().item()

            expanded_seq = torch.empty(expanded_len, dtype=torch.long, device=self.device)
            all_image_positions = []

            pos = 0
            for i, token in enumerate(input_ids):
                if token == self.processor.image_id:
                    all_image_positions.append(pos)
                    expanded_seq[pos:pos + self.image_token_num_per_image] = self.processor.image_id
                    pos += self.image_token_num_per_image
                else:
                    expanded_seq[pos] = token
                    pos += 1

            processed_sequences.append(expanded_seq)
            all_image_start_indices.append(all_image_positions)
            images_to_batch.append(images)

        # padding
        max_len = max(seq.size(0) for seq in processed_sequences)
        B = len(processed_sequences)

        batched_total_ids = torch.full((B, max_len), self.processor.pad_id, dtype=torch.long, device=self.device)
        batched_attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=self.device)

        for i, seq in enumerate(processed_sequences):
            seq_len = seq.size(0)
            batched_total_ids[i, :seq_len] = seq
            batched_attention_mask[i, :seq_len] = 1

        images_to_batch = torch.cat(images_to_batch, dim=0)

        return batched_total_ids, batched_attention_mask, output_start_indices, all_image_start_indices, images_to_batch

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
        print(f"[TEXT_GEN] Input batch_size: {batch_size}")
        
        # Get images from meta_info
        gen_imgs_pixel_values = data_proto.batch.get('task1_gen_imgs_pixel_values', [])
        if len(gen_imgs_pixel_values) == 0:
            raise ValueError("No images found in meta_info['task1_gen_imgs_pixel_values']")
        
        # Process all images in batch
        print(f"[TEXT_GEN] Processing feedback for {len(gen_imgs_pixel_values)} images in batch")

        # Prepare messages for all images
        input_format = []
        for prompt in data_proto.non_tensor_batch['prompt']:
            last_prompt = prompt.replace("<|User|>: ", "").replace("\n\n<|Assistant|>:<begin_of_image>", "")
            input_format.append(prompt + self.image_tag + self.image_end_tag + "\nFirst, Decompose input prompt: " + f"'{last_prompt}'" + '\n')

        inputs = self.processor.tokenizer(
            input_format,
            padding=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        batched_total_ids, batched_attention_mask, output_start_indices, all_image_start_indices, images_to_batch = self.expand_image_placeholders(input_ids, gen_imgs_pixel_values)

        # embedding
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)

        gen_imgs_pixel_values = gen_imgs_pixel_values.to(self.device, dtype=torch.bfloat16)

        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                text_embeds = self.module.language_model.get_input_embeddings()(batched_total_ids)
                image_embeds = self.module.aligner(self.module.vision_model(gen_imgs_pixel_values))

        # merge text and image embeds
        merged_embeds = self.merge_text_and_image_embeds(text_embeds, image_embeds, all_image_start_indices)
        # For computing logits: input

        batch_size, _, embed_dim = merged_embeds.shape
        max_input_len = max(output_start_indices)
        input_ids = torch.full((batch_size, max_input_len), self.processor.pad_id, dtype=torch.long, device=merged_embeds.device)
        input_embeds = torch.zeros((batch_size, max_input_len, embed_dim), dtype=merged_embeds.dtype, device=merged_embeds.device)
        input_attention_mask = torch.zeros((batch_size, max_input_len), dtype=torch.long, device=merged_embeds.device)
        
        # Left-padding 후의 이미지 시작 인덱스를 저장할 리스트
        new_all_image_start_indices = []

        for i in range(batch_size):
            input_len = output_start_indices[i]
            pad_len = max_input_len - input_len
            
            prompt_embeds = merged_embeds[i, :input_len]
            prompt_attn_mask = batched_attention_mask[i, :input_len]           

            input_ids[i, pad_len:] = batched_total_ids[i, :input_len]
            input_embeds[i, pad_len:] = prompt_embeds
            input_attention_mask[i, pad_len:] = prompt_attn_mask
            
            # 패딩 길이를 더해 새로운 이미지 시작 인덱스 계산
            original_indices = all_image_start_indices[i]
            new_indices = [idx + pad_len for idx in original_indices]
            new_all_image_start_indices.append(new_indices)

        # For generating feedback texts
        data_proto.batch["task2_input_ids"] = input_ids
        data_proto.batch["task2_attention_mask"] = input_attention_mask
        data_proto.batch["task2_input_embeds"] = input_embeds

        feedback_texts = self.generate_text(input_embeds, input_attention_mask)

        # For computing logits: output
        data_proto.non_tensor_batch["task2_feedback_texts"] = np.array(feedback_texts, dtype=object)
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token # Right padding w/ eos_token_id
        outputs = self.processor.tokenizer(feedback_texts, padding=True, padding_side='right', return_tensors="pt")
        data_proto.batch["task2_feedback_ids"] = outputs["input_ids"]

        feedback_ids = data_proto.batch["task2_feedback_ids"]
        feedback_ids = feedback_ids.to(self.device)

        # Postprocessing output embeds
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)

        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                text_embeds = self.module.language_model.get_input_embeddings()(feedback_ids)

        data_proto.batch["task2_response_mask"] = outputs["attention_mask"]
        print(f"[TEXT_GEN] Completed feedback generation")

        return data_proto

    def _generate_minibatch_regen_image_generation(self, data_proto: DataProto) -> DataProto:
        batch_size = data_proto.batch.batch_size[0]

        print(f"[REGEN] Input batch_size: {batch_size}")

        # Get data from meta_info
        gen_imgs_pixel_values = data_proto.batch.get('task1_gen_imgs_pixel_values', [])
        feedback_texts = data_proto.non_tensor_batch.get('task2_feedback_texts', [])
        
        if len(gen_imgs_pixel_values) == 0:
            raise ValueError("No images found in meta_info['task1_gen_imgs_pixel_values']")
        
        print(f"[REGEN] Loaded {len(gen_imgs_pixel_values)} images and {len(feedback_texts)} task2_feedback_texts from meta_info")

        # Process all images in batch
        print(f"[REGEN] Processing regen for {batch_size} images in batch")

        # Parse feedback texts
        feedback_texts = [self.formatter._split_text_into_parts(feedback)[-1] for feedback in data_proto.non_tensor_batch['task2_feedback_texts']]

        # Prepare messages for all images
        input_format = []
        for feedback in feedback_texts:
            _prefix = self.image_start_tag + self.image_tag + self.image_end_tag + "\n"
            if feedback is None:
                prefix = self.get_sft_format(_prefix + "No need to generate feedback.")
            else:
                prefix = self.get_sft_format(_prefix + feedback)
            input_format.append(prefix) # Add image placeholder at the end            

        inputs = self.processor.tokenizer(
            input_format,
            padding=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        data_proto.batch["task3_input_ids"] = input_ids
        data_proto.batch["task3_attention_mask"] = attention_mask

        # embedding
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)

        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                input_embeds = self.module.language_model.get_input_embeddings()(input_ids)

        # For computing logits: input (especially input text embedding)
        data_proto.batch["task3_input_embeds"] = input_embeds

        regen_final_cfg_embeds, regen_final_cfg_attention_mask = self._prepare_regen_cfg_embeds(data_proto)
        regenerated_tokens = self.generate_img(regen_final_cfg_embeds, regen_final_cfg_attention_mask)
        # For reproducing generated images
        data_proto.batch["task3_regen_img_tokens"] = regenerated_tokens

        regen_decoded_images = self._decode_image_tokens(regenerated_tokens)
        regen_imgs_pil_list = [PIL.Image.fromarray(img_array) for img_array in regen_decoded_images]
        data_proto.non_tensor_batch["task3_regen_imgs_pil_list"] = np.array(regen_imgs_pil_list, dtype=object)
        regen_imgs_tensor = self.processor.image_processor(regen_imgs_pil_list).pixel_values
        # For generating feedback texts and computing logits: output
        data_proto.batch["task3_regen_imgs_pixel_values"] = regen_imgs_tensor.cpu()
        regen_imgs_pixel_values = regen_imgs_tensor.to(self.device, dtype=torch.bfloat16)

        # Postprocessing output embeds
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)

        B, C, H, W = gen_imgs_pixel_values.shape

        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, _, all_image_ids = self.module.gen_vision_model.encode(regen_imgs_pixel_values)
        
                image_ids = all_image_ids[2]
                image_ids = image_ids.view(B, -1)

                image_embeds = self.module.gen_aligner(self.module.gen_embed(image_ids))

        data_proto.batch["task3_response_mask"] = torch.ones((B, image_embeds.size(1)), dtype=torch.long, device=image_embeds.device)
        print(f"[IMG_GEN] Created DataProto with batch_size: {batch_size}")

        return data_proto

    def _prepare_cfg_embeds(self, data_proto: DataProto) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = data_proto.batch['task1_input_ids']
        attention_mask = data_proto.batch['task1_attention_mask']
        input_embeds = data_proto.batch['task1_input_embeds']

        cond_embeds = input_embeds
        uncond_embeds = input_embeds.clone()

        pad_id = torch.tensor(self.processor.pad_id, device=input_ids.device)

        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)
        
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pad_embed = self.module.language_model.get_input_embeddings()(pad_id).unsqueeze(0)

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

        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)
        
        with param_ctx:
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

    @torch.inference_mode()
    def generate_img(self, inputs_embeds: torch.Tensor, attention_masks: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        batch_size = inputs_embeds.shape[0] // 2
        generated_tokens = torch.zeros((batch_size, self.image_token_num_per_image), dtype=torch.int, device=self.device)

        attention_mask = attention_masks

        position_ids = attention_mask.long().cumsum(1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        past_len = attention_mask.sum(dim=1, keepdim=True).long()

        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)
        
        with param_ctx:
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

                    attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=self.device)], dim=1)

                    position_ids = past_len.clone()
                    past_len += 1

        return generated_tokens

    @torch.inference_mode()
    def generate_text(self, inputs_embeds: torch.Tensor, attention_masks: torch.Tensor) -> torch.Tensor:
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)
        
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.module.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_masks,
                    max_new_tokens=self.max_reflect_len,
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

        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)
        
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pad_embed = self.module.language_model.get_input_embeddings()(pad_id).unsqueeze(0)

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