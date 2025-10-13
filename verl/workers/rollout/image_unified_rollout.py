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
from typing import Union, List, Any, Optional
from transformers.processing_utils import ProcessorMixin
import base64
from io import BytesIO
import PIL.Image
from verl.utils.torch_functional import get_response_mask
import os
import json
from pathlib import Path

__all__ = ['ImageUnifiedRollout']

class ImageUnifiedRollout:
    def __init__(self, module: nn.Module, config): 
        self.config = config
        self.module = module
        from transformers import JanusProcessor
        processor = JanusProcessor.from_pretrained("deepseek-community/Janus-Pro-7B")
        self.processor = processor
        self.generation_mode = getattr(config, "generation_mode", None)
        self.feedback_system_prompt = getattr(config, "feedback_system_prompt", "")
        self.refine_system_prompt = getattr(config, "refine_system_prompt", "")
        self.generation_config = None
        
        # Setup saving directory
        self.save_dir = getattr(config, "save_dir", "./output/rollout_generations")
        self.save_enabled = getattr(config, "saving", True)
        self.generation_counter = 0
        
        if self.save_enabled:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            print(f"[ROLLOUT] Saving generations to: {self.save_dir}")

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

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get('micro_batch_size', batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        
        self.set_generation_config(batch_prompts[0])
        
        # Use num_return_sequences instead of looping
        n = self.config.get('n', 1)
        print(f"[GENERATE_SEQ] batch_size={batch_size}, n={n}, will generate {batch_size * n} total sequences")
        print(f"[GENERATE_SEQ] Using num_return_sequences={n} for parallel generation")
        
        all_outputs = []
        for p in batch_prompts:
            # Generate all n sequences in one pass
            output = self._generate_minibatch_image_generation(p)
            output = self._generate_minibatch_text_generation(output)
            output = self._generate_minibatch_refine_image_generation(output)
            
            # Save generated content
            if self.save_enabled:
                self._save_generation(output, p)
            
            all_outputs.append(output)
        
        output = DataProto.concat(all_outputs)
        print(f"[GENERATE_SEQ] Final output batch_size: {output.batch.batch_size}")
        return output
    
    def set_generation_config(self, prompts: DataProto):
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))
        
        # Get n from config for num_return_sequences
        n = self.config.get('n', 1)

        if not do_sample:
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
            }
        elif is_validate:
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,
            }
        else:
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": 1,  # Use n for parallel generation
            }

        self.generation_config = GenerationConfig(**kwargs)
        
        if hasattr(self.module, 'config') and getattr(self.module.config, 'model_type', None) == 'janus':
            if hasattr(self.processor, 'tokenizer'):
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

    def extract_prompts(self, input_ids: torch.Tensor, pad_token_id: int, eos_token_id: int,tokenizer) -> List[str]:
        ids = input_ids

        ids = ids[ids != pad_token_id]

        if eos_token_id in ids:
            eos_idx = (ids == eos_token_id).nonzero(as_tuple=True)[0][0]
            ids = ids[:eos_idx]

        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        return text

    @torch.no_grad()
    def _generate_minibatch_image_generation(self, prompts: DataProto) -> DataProto:
        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        eos_token_id = prompts.meta_info['eos_token_id']
        pad_token_id = prompts.meta_info['pad_token_id']

        batch_size = input_ids.size(0)
        prompt_length = input_ids.size(1)
        n = self.generation_config.num_return_sequences
        
        print(f"[IMG_GEN] Input batch_size: {batch_size}, prompt_length: {prompt_length}, n={n}")
        print(f"[IMG_GEN] Will generate {batch_size * n} total sequences")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        self.module.eval()
        param_ctx = contextlib.nullcontext()
        
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)
        
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_mode="image",
                    generation_config=self.generation_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                    max_new_tokens=512)

        sequences = output.sequences
        generated_batch_size = sequences.size(0)
        
        print(f"[IMG_GEN] Generated sequences shape: {sequences.shape}")
        print(f"[IMG_GEN] generated_batch_size: {generated_batch_size} (expected: {batch_size * n})")

        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)
        
        # Batch decode all images at once
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                decoded_images = self.module.decode_image_tokens(sequences)
        
        print(f"[IMG_GEN] Decoded images shape: {decoded_images.shape}, dtype: {decoded_images.dtype}")
        
        # Batch postprocess all images
        result = self.processor.postprocess(
            list(decoded_images.float()), 
            return_tensors="PIL.Image.Image"
        )
        
        # Extract pixel_values from BatchFeature
        if hasattr(result, 'data') and 'pixel_values' in result.data:
            gen_img_list = result.data['pixel_values']
        elif isinstance(result, dict) and 'pixel_values' in result:
            gen_img_list = result['pixel_values']
        else:
            gen_img_list = [result] if not isinstance(result, list) else result
        
        print(f"[IMG_GEN] Processed {len(gen_img_list)} images")
        print(f"[IMG_GEN] Tensors: input_ids {input_ids.shape}, attention_mask {attention_mask.shape}, position_ids {position_ids.shape}")

        batch = TensorDict(
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            },
            batch_size=batch_size)
        
        # Create DataProto and store images in meta_info
        data_proto = DataProto(batch=batch, meta_info=prompts.meta_info)
        data_proto.meta_info['gen_img_list'] = gen_img_list
    
        print(f"[IMG_GEN] Created DataProto with batch_size: {batch.batch_size}")
        print(f"[IMG_GEN] Stored {len(gen_img_list)} images in meta_info")

        return data_proto

    def convert_gen_img_to_base64(self, gen_img: PIL.Image.Image) -> Optional[str]:
        """Convert gen_img to base64 data URL."""
        if not isinstance(gen_img, PIL.Image.Image):
            raise TypeError(f"Unsupported image type: {type(gen_img)}")

        buffer = BytesIO()
        gen_img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{img_base64}"

    def _generate_minibatch_text_generation(self, data_proto: DataProto) -> DataProto:
        batch_size = data_proto.batch.batch_size[0]
        
        print(f"[TEXT_GEN] Input batch_size: {batch_size}")
        print(f"[TEXT_GEN] Starting feedback generation for {batch_size} images...")
        
        # Get images from meta_info
        gen_img_list = data_proto.meta_info.get('gen_img_list', [])
        if not gen_img_list:
            raise ValueError("No images found in meta_info['gen_img_list']")
        
        # Prepare all messages in batch
        messages_batch = []
        for i, gen_img in enumerate(gen_img_list):
            img_url = self.convert_gen_img_to_base64(gen_img)
            message = [
                {
                    "role": "user", "content": [
                        {"type": "image", "url": img_url},
                        {"type": "text", "text": self.feedback_system_prompt}
                    ]
                }
            ]
            messages_batch.append(message)
        
        print(f"[TEXT_GEN] Prepared {len(messages_batch)} messages for batch processing")
        
        # Batch process all messages
        feedback_texts = self._batch_generate_feedback(messages_batch)
        
        print(f"[TEXT_GEN] All feedback texts generated: {len(feedback_texts)} total")
        
        # Store in meta_info
        data_proto.meta_info["feedback_texts"] = feedback_texts
        print(f"[TEXT_GEN] Stored feedback_texts in meta_info (count={len(feedback_texts)})")
        
        print(f"[TEXT_GEN] Text generation complete!")
        return data_proto
    
    def _batch_generate_feedback(self, messages_batch: List) -> List[str]:
        """Generate feedback for multiple images in batch."""
        print(f"[BATCH_FEEDBACK] Processing {len(messages_batch)} messages")
        
        # Process all messages to get input tensors
        all_input_ids = []
        all_attention_masks = []
        
        for message in messages_batch:
            inputs = self.processor.apply_chat_template(
                message, add_generation_prompt=True, generation_mode="text", 
                tokenize=True, return_dict=True, return_tensors="pt"
            )

            if hasattr(inputs, 'input_ids'):
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
            elif isinstance(inputs, dict):
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
            else:
                raise TypeError(f"Unexpected type from apply_chat_template: {type(inputs)}")
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
        
        # Pad to same length and stack
        max_length = max(ids.shape[1] for ids in all_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        
        padded_input_ids = []
        padded_attention_masks = []
        
        for input_ids, attention_mask in zip(all_input_ids, all_attention_masks):
            pad_length = max_length - input_ids.shape[1]
            if pad_length > 0:
                # Pad on the left
                input_ids = torch.cat([
                    torch.full((1, pad_length), pad_token_id, dtype=input_ids.dtype, device=input_ids.device),
                    input_ids
                ], dim=1)
                attention_mask = torch.cat([
                    torch.zeros((1, pad_length), dtype=attention_mask.dtype, device=attention_mask.device),
                    attention_mask
                ], dim=1)
            
            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)
        
        # Stack into batch
        batched_input_ids = torch.cat(padded_input_ids, dim=0).to(self.module.device)
        batched_attention_mask = torch.cat(padded_attention_masks, dim=0).to(self.module.device)
        
        print(f"[BATCH_FEEDBACK] Batched input shapes - input_ids: {batched_input_ids.shape}, "
            f"attention_mask: {batched_attention_mask.shape}")
        
        # # Create a text-specific generation config with num_return_sequences=1
        # text_gen_config = GenerationConfig(
        #     do_sample=self.generation_config.do_sample,
        #     num_beams=1,
        #     top_p=self.generation_config.top_p,
        #     top_k=self.generation_config.top_k,
        #     temperature=self.generation_config.temperature,
        #     num_return_sequences=1,  # Always 1 for text generation
        #     pad_token_id=self.generation_config.pad_token_id,
        #     eos_token_id=self.generation_config.eos_token_id,
        # )
        
        # Generate in batch
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)
        
        model_kwargs = {
            "attention_mask": batched_attention_mask,
            "use_cache": True,
        }
        
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.module.generate(
                    batched_input_ids,
                    generation_mode="text",
                    generation_config=self.generation_config, # text_gen_config,  # Use text-specific config
                    return_dict_in_generate=True,
                    **model_kwargs
                )
        
        # Handle both dict and tensor returns
        if isinstance(output, dict) or hasattr(output, 'sequences'):
            sequences = output.sequences if hasattr(output, 'sequences') else output['sequences']
        else:
            sequences = output
        
        # Decode all at once
        feedback_texts = []
        for seq in sequences:
            decoded_text = self.processor.decode(seq, skip_special_tokens=True)
            feedback_texts.append(decoded_text)
        
        print(f"[BATCH_FEEDBACK] Generated {len(feedback_texts)} feedback texts")
        return feedback_texts

    def build_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        pos = (torch.cumsum(attention_mask, dim=1) - 1).clamp(min=0)
        return pos * attention_mask

    def _generate_minibatch_refine_image_generation(self, data_proto: DataProto) -> DataProto:
        batch_size = data_proto.batch.batch_size[0]
        print(f"[REFINE] Input batch_size: {batch_size}")

        # Get data from meta_info
        gen_img_list = data_proto.meta_info.get('gen_img_list', [])
        feedback_texts = data_proto.meta_info.get('feedback_texts', [])
        
        if not gen_img_list:
            raise ValueError("No images found in meta_info['gen_img_list']")
        
        print(f"[REFINE] Loaded {len(gen_img_list)} images and {len(feedback_texts)} feedback_texts from meta_info")

        # Prepare all messages for batch processing
        messages_batch = []
        for i, gen_img in enumerate(gen_img_list):
            fb_txt = feedback_texts[i] if i < len(feedback_texts) else ""
            img_url = self.convert_gen_img_to_base64(gen_img)
            
            message = [
                {
                    "role": "system", "content": [
                        {"type": "text", "text": self.refine_system_prompt}
                    ]
                },
                {
                    "role": "user", "content": [
                        {"type": "image", "url": img_url},
                        {"type": "text", "text": fb_txt}
                    ]
                }
            ]
            messages_batch.append(message)
        
        print(f"[REFINE] Prepared {len(messages_batch)} messages for batch refinement")
        
        # Batch process all refinements
        refined_images = self._batch_refine_images(messages_batch)
        
        print(f"[REFINE] Generated {len(refined_images)} refined images")

        # Store refined images in meta_info
        data_proto.meta_info['refined_gen_img_list'] = refined_images

        torch.cuda.empty_cache()
        print(f"[REFINE] Completed refinement")
        return data_proto
    
    def _batch_refine_images(self, messages_batch: List) -> List[PIL.Image.Image]:
        """Refine multiple images in batch."""
        print(f"[BATCH_REFINE] Processing {len(messages_batch)} messages")
        
        # Process all messages to get input tensors
        all_input_ids = []
        all_attention_masks = []
        
        for message in messages_batch:
            inputs = self.processor.apply_chat_template(
                message, add_generation_prompt=True, generation_mode="image", 
                tokenize=True, return_dict=True, return_tensors="pt"
            )
            print(f"[BATCH_REFINE] Input ids: {inputs.input_ids}, attention_mask: {inputs.attention_mask}")
            
            if hasattr(inputs, 'input_ids'):
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
            elif isinstance(inputs, dict):
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
            else:
                raise TypeError(f"Unexpected type from apply_chat_template: {type(inputs)}")
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
        
        # Find max length
        max_length = max(ids.shape[1] for ids in all_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        
        padded_input_ids = []
        padded_attention_masks = []
        original_lengths = []
        
        for input_ids, attention_mask in zip(all_input_ids, all_attention_masks):
            original_length = input_ids.shape[1]
            original_lengths.append(original_length)
            pad_length = max_length - original_length
            
            if pad_length > 0:
                # Pad on the left
                input_ids = torch.cat([
                    torch.full((1, pad_length), pad_token_id, dtype=input_ids.dtype, device=input_ids.device),
                    input_ids
                ], dim=1)
                attention_mask = torch.cat([
                    torch.zeros((1, pad_length), dtype=attention_mask.dtype, device=attention_mask.device),
                    attention_mask
                ], dim=1)
            
            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)
        
        # Stack into batch
        batched_input_ids = torch.cat(padded_input_ids, dim=0).to(self.module.device)
        batched_attention_mask = torch.cat(padded_attention_masks, dim=0).to(self.module.device)
        
        print(f"[BATCH_REFINE] Batched input shapes - input_ids: {batched_input_ids.shape}, "
            f"attention_mask: {batched_attention_mask.shape}")
        print(f"[BATCH_REFINE] Original lengths: {original_lengths}")

        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)

        # Simple generation call - let model handle everything internally
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=batched_input_ids,
                    attention_mask=batched_attention_mask,
                    generation_mode="image",
                    generation_config=self.generation_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                    max_new_tokens=512)

        # Handle both dict and tensor returns
        if isinstance(output, dict) or hasattr(output, 'sequences'):
            sequences = output.sequences if hasattr(output, 'sequences') else output['sequences']
        else:
            sequences = output

        # Batch decode all images
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)
        
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                decoded_images = self.module.decode_image_tokens(sequences)
        
        print(f"[BATCH_REFINE] Decoded images shape: {decoded_images.shape}, dtype: {decoded_images.dtype}")
        
        # Batch postprocess all images
        result = self.processor.postprocess(
            list(decoded_images.float()), 
            return_tensors="PIL.Image.Image"
        )
        
        # Extract pixel_values from BatchFeature
        if hasattr(result, 'data') and 'pixel_values' in result.data:
            refined_img_list = result.data['pixel_values']
        elif isinstance(result, dict) and 'pixel_values' in result:
            refined_img_list = result['pixel_values']
        else:
            refined_img_list = [result] if not isinstance(result, list) else result
        
        print(f"[BATCH_REFINE] Processed {len(refined_img_list)} refined images")
        
        return refined_img_list

    def _save_generation(self, output: DataProto, prompts: DataProto):
            """Save generated images and metadata."""
            try:
                batch_size = output.batch.batch_size[0]
                
                # Get lists from meta_info
                gen_img_list = output.meta_info.get('gen_img_list', [])
                refined_img_list = output.meta_info.get('refined_gen_img_list', [])
                feedback_texts = output.meta_info.get('feedback_texts', [])
                
                print(f"[SAVE] batch_size: {batch_size}")
                print(f"[SAVE] gen_img_list length: {len(gen_img_list)}")
                print(f"[SAVE] refined_img_list length: {len(refined_img_list)}")
                print(f"[SAVE] feedback_texts length: {len(feedback_texts)}")
                
                if not gen_img_list:
                    print(f"[SAVE] No images in gen_img_list, skipping save")
                    return

                for i in range(batch_size):
                    gen_id = self.generation_counter
                    self.generation_counter += 1

                    gen_dir = Path(self.save_dir) / f"gen_{gen_id:06d}"
                    gen_dir.mkdir(parents=True, exist_ok=True)

                    # Get items for this index
                    gen_img = gen_img_list[i] if i < len(gen_img_list) else None
                    refined_img = refined_img_list[i] if i < len(refined_img_list) else None
                    feedback_text = feedback_texts[i] if i < len(feedback_texts) else ""
                    
                    # Get input_ids - handle the case where we have more outputs than original prompts
                    # due to num_return_sequences
                    original_batch_size = prompts.batch["input_ids"].shape[0]
                    prompt_idx = i % original_batch_size  # Map back to original prompt
                    input_ids = prompts.batch["input_ids"][prompt_idx]

                    # Decode input prompt
                    try:
                        input_text = self.processor.decode(input_ids.tolist(), skip_special_tokens=True)
                    except:
                        input_text = "Failed to decode input"
                    
                    # Save original generated image
                    if gen_img is not None and isinstance(gen_img, PIL.Image.Image):
                        gen_img.save(gen_dir / "generated_image.png")
                        print(f"[SAVE] Saved generated image to {gen_dir / 'generated_image.png'}")
                    
                    # Save refined image
                    if refined_img is not None and isinstance(refined_img, PIL.Image.Image):
                        refined_img.save(gen_dir / "refined_image.png")
                        print(f"[SAVE] Saved refined image to {gen_dir / 'refined_image.png'}")
                    
                    # Save metadata
                    metadata = {
                        'generation_id': gen_id,
                        'input_prompt': input_text,
                        'feedback_text': feedback_text,
                        'feedback_system_prompt': self.feedback_system_prompt,
                        'refine_system_prompt': self.refine_system_prompt,
                        'generation_config': {
                            'do_sample': bool(self.generation_config.do_sample) if self.generation_config.do_sample is not None else None,
                            'temperature': float(self.generation_config.temperature) if self.generation_config.temperature is not None else None,
                            'top_p': float(self.generation_config.top_p) if self.generation_config.top_p is not None else None,
                            'top_k': int(self.generation_config.top_k) if self.generation_config.top_k is not None else None,
                            'num_return_sequences': int(self.generation_config.num_return_sequences) if self.generation_config.num_return_sequences is not None else None,
                        }
                    }
                    
                    with open(gen_dir / "metadata.json", 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
                    print(f"[SAVE] Saved metadata to {gen_dir / 'metadata.json'}")
                    
                    # Save text files for easy viewing
                    with open(gen_dir / "input_prompt.txt", 'w', encoding='utf-8') as f:
                        f.write(input_text)
                    
                    with open(gen_dir / "feedback.txt", 'w', encoding='utf-8') as f:
                        f.write(feedback_text)
                    
            except Exception as e:
                print(f"[SAVE] Error saving generation: {e}")
                import traceback
                traceback.print_exc()