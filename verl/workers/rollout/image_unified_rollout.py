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
import verl.utils.torch_functional as verl_F
import os
import json
from pathlib import Path

from .image_rl_logit_process import CFGEmbeddingLogitsProcessor

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
        
        # Setup saving directory - force to use /verl directory
        self.save_dir = getattr(config, "save_dir", "/verl/output/rollout")
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

        all_outputs = []
        for p in batch_prompts:
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

    @torch.no_grad()
    def _generate_minibatch_image_generation(self, prompts: DataProto) -> DataProto:
        input_ids = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        eos_token_id = prompts.meta_info['eos_token_id']
        pad_token_id = prompts.meta_info['pad_token_id']

        batch_size = input_ids.size(0)
        prompt_length = input_ids.size(1)
        n = self.config.get('n', 1)
        
        print(f"[IMG_GEN] Input batch_size: {batch_size}, prompt_length: {prompt_length}, n={n}")
        print(f"[IMG_GEN] Will generate {batch_size * n} total sequences")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        self.module.eval()
        
        # Generate n images in batch
        print(f"[IMG_GEN] Generating {n} images in batch")
        
        # Replicate input tensors n times for batch generation
        replicated_input_ids = input_ids.repeat(n, 1)
        replicated_attention_mask = attention_mask.repeat(n, 1)
        
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)
        
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                cfg_processor = CFGEmbeddingLogitsProcessor(
                    task=1,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    cfg_weight=5.0,
                    model=self.module
                )

                output = self.module.generate(
                    input_ids=replicated_input_ids,
                    attention_mask=replicated_attention_mask,
                    generation_mode="image",
                    generation_config=self.generation_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                    max_new_tokens=512)
                    # logits_processor=[cfg_processor])

        sequences = output.sequences
        print(f"[IMG_GEN] Generated sequences shape: {sequences.shape}")

        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)
        
        # Decode images in batch
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                decoded_images = self.module.decode_image_tokens(sequences)
        
        print(f"[IMG_GEN] Decoded images shape: {decoded_images.shape}, dtype: {decoded_images.dtype}")
        
        # Postprocess images in batch
        result = self.processor.postprocess(
            list(decoded_images.float()), 
            return_tensors="PIL.Image.Image"
        )
        
        # Extract pixel_values from BatchFeature
        all_gen_imgs = result['pixel_values']
        print(f"[IMG_GEN] Generated {len(all_gen_imgs)} images in batch")

        # Store prompt length
        self.prompt_length = prompt_length

        start_marker = torch.tensor([100601, 25]).to(self.module.device) # <|User|>:
        end_marker = torch.tensor([100602, 25]).to(self.module.device) # <|Assistant|>:

        # Search in the first sequence (input_ids[0])
        output_start_idx = self.find_sequence(input_ids[0], start_marker)
        output_end_idx = self.find_sequence(input_ids[0], end_marker)

        if output_start_idx != -1:
            output_start_idx += len(start_marker)

        input_text = self.processor.decode(input_ids[0][output_start_idx:output_end_idx])
        print(f"[IMG_GEN] Input text: {input_text}")

        # Replicate position_ids to match the number of generated images
        replicated_position_ids = position_ids.repeat_interleave(n, dim=0)
        
        print(f"[IMG_GEN] Replicated input_ids shape: {replicated_input_ids.shape}")
        
        batch = TensorDict(
            {
                'input_ids': replicated_input_ids,
                'attention_mask': replicated_attention_mask,
                'position_ids': replicated_position_ids,
            },
            batch_size=batch_size * n) # batch_size is the number of generated images

        # Create DataProto and store images in meta_info
        data_proto = DataProto(batch=batch, meta_info=prompts.meta_info)
        data_proto.meta_info['gen_img_list'] = all_gen_imgs
        data_proto.meta_info['input_text'] = [input_text] * n

        print(f"[IMG_GEN] Created DataProto with batch_size: {batch_size * n}")
        print(f"[IMG_GEN] Stored {len(all_gen_imgs)} images in meta_info")

        return data_proto

    def _generate_minibatch_text_generation(self, data_proto: DataProto) -> DataProto:
        batch_size = data_proto.batch.batch_size[0]
        n = self.config.get('n', 1)

        print(f"[TEXT_GEN] Input batch_size: {batch_size}, n={n}")
        print(f"[TEXT_GEN] Starting feedback generation for {n} images in batch")
        
        # Get images from meta_info
        gen_img_list = data_proto.meta_info.get('gen_img_list', [])
        if not gen_img_list:
            raise ValueError("No images found in meta_info['gen_img_list']")
        
        # Process all images in batch
        print(f"[TEXT_GEN] Processing feedback for {n} images in batch")
        
        # Prepare messages for all images
        messages_batch = []
        for i in range(n):
            gen_img = gen_img_list[i]
            message = [
                {
                    "role": "user", "content": [
                        {"type": "image", "image": gen_img},
                        {"type": "text", "text": self.feedback_system_prompt}
                    ]
                }
            ]
            messages_batch.append(message)
        
        # Generate feedback for all images in batch
        all_feedback_texts = self._batch_generate_feedback(messages_batch)
        
        print(f"[TEXT_GEN] All feedback texts generated: {len(all_feedback_texts)} total")
        
        # Store in meta_info
        data_proto.meta_info["feedback_texts"] = all_feedback_texts
        print(f"[TEXT_GEN] Stored feedback_texts in meta_info (count={len(all_feedback_texts)})")
        
        print(f"[TEXT_GEN] Text generation complete!")
        return data_proto
    
    def _batch_generate_feedback(self, messages_batch: List) -> List[str]:
        """Generate feedback for multiple images in batch."""
        print(f"[BATCH_FEEDBACK] Processing {len(messages_batch)} messages")
       
        # Create a text-specific generation config with num_return_sequences=1
        text_gen_config = GenerationConfig(
            do_sample=self.generation_config.do_sample,
            num_beams=1,
            top_p=self.generation_config.top_p,
            top_k=self.generation_config.top_k,
            temperature=self.generation_config.temperature,
            num_return_sequences=1,  # Always 1 for text generation
            pad_token_id=self.generation_config.pad_token_id,
            eos_token_id=self.generation_config.eos_token_id,
        )
         
        # Process all messages to get input tensors
        all_input_ids = []
        all_attention_masks = []
        original_lengths = []
        
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
            original_lengths.append(input_ids.shape[1])

        # Find max length for proper padding
        max_length = max(tensor.shape[1] for tensor in all_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id

        padded_input_ids = []
        padded_attention_masks = []
        
        for i, (input_ids, attention_mask) in enumerate(zip(all_input_ids, all_attention_masks)):
            # Use verl_F.postprocess_data for consistent padding
            padded_input_ids_sample, padded_attention_mask_sample = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                pad_token_id=pad_token_id,
                left_pad=True,
                truncation="error"
            )
            
            padded_input_ids.append(padded_input_ids_sample)
            padded_attention_masks.append(padded_attention_mask_sample)
        
        # Stack into batch
        batched_input_ids = torch.cat(padded_input_ids, dim=0).to(self.module.device)
        batched_attention_mask = torch.cat(padded_attention_masks, dim=0).to(self.module.device)
        
        print(f"[BATCH_FEEDBACK] Batched input shapes - input_ids: {batched_input_ids.shape}, "
            f"attention_mask: {batched_attention_mask.shape}")
        
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
                    generation_config=text_gen_config,
                    return_dict_in_generate=True,
                    **model_kwargs
                )
        
        # Handle output properly
        if isinstance(output, dict) or hasattr(output, 'sequences'):
            sequences = output.sequences if hasattr(output, 'sequences') else output['sequences']
        else:
            sequences = output
        
        print(f"[DEBUG] Generated sequences shape: {sequences.shape}")

        # Decode responses correctly
        feedback_texts = []
        for i, (seq, orig_len) in enumerate(zip(sequences, original_lengths)):
            # Decode only the newly generated part
            if len(seq) > orig_len:
                new_tokens = seq[orig_len:]
                decoded_text = self.processor.decode(new_tokens, skip_special_tokens=True)
            else:
                decoded_text = ""
            
            feedback_texts.append(decoded_text)

        return feedback_texts

    def _generate_minibatch_refine_image_generation(self, data_proto: DataProto) -> DataProto:
        batch_size = data_proto.batch.batch_size[0]
        n = self.config.get('n', 1)

        print(f"[REFINE] Input batch_size: {batch_size}, n={n}")
        print(f"[REFINE] Starting image refinement for {n} images in batch")

        # Get data from meta_info
        gen_img_list = data_proto.meta_info.get('gen_img_list', [])
        feedback_texts = data_proto.meta_info.get('feedback_texts', [])
        
        if not gen_img_list:
            raise ValueError("No images found in meta_info['gen_img_list']")
        
        print(f"[REFINE] Loaded {len(gen_img_list)} images and {len(feedback_texts)} feedback_texts from meta_info")

        # Process all images in batch
        print(f"[REFINE] Processing refine for {n} images in batch")
        
        # Prepare messages for all images
        messages_batch = []
        for i in range(n):
            gen_img = gen_img_list[i]
            fb_txt = feedback_texts[i]
            
            message = [
                {
                    "role": "user", "content": [
                        {"type": "image", "image": gen_img},
                        {"type": "text", "text": fb_txt}
                    ]
                }
            ]
            messages_batch.append(message)
        
        # Refine all images in batch
        all_refined_images = self._batch_refine_images(messages_batch)
        
        print(f"[REFINE] Generated {len(all_refined_images)} refined images")

        # Store refined images in meta_info
        data_proto.meta_info['refined_gen_img_list'] = all_refined_images

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
            prompt = self.processor.apply_chat_template(
                message, add_generation_prompt=True, generation_mode="image", 
                tokenize=False
            )
            inputs = self.processor(text=prompt, generation_mode="image", return_tensors="pt")
            
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

        # Use verl_F.postprocess_data for consistent padding
        max_length = max(tensor.shape[1] for tensor in all_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        
        padded_input_ids = []
        padded_attention_masks = []
        
        for i, (input_ids, attention_mask) in enumerate(zip(all_input_ids, all_attention_masks)):
            # Use verl_F.postprocess_data for consistent padding
            padded_input_ids_sample, padded_attention_mask_sample = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                pad_token_id=pad_token_id,
                left_pad=True,
                truncation="error"
            )
            
            padded_input_ids.append(padded_input_ids_sample)
            padded_attention_masks.append(padded_attention_mask_sample)
        
        # Stack into batch
        batched_input_ids = torch.cat(padded_input_ids, dim=0).to(self.module.device)
        batched_attention_mask = torch.cat(padded_attention_masks, dim=0).to(self.module.device)
        
        # Debug output
        print(f"[DEBUG] Refine batched attention mask shape: {batched_attention_mask.shape}")
        
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=True)

        # Simple generation call - let model handle everything internally
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                cfg_processor = CFGEmbeddingLogitsProcessor(
                    task=3,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    cfg_weight=3.0,
                    model=self.module
                )

                output = self.module.generate(
                    input_ids=batched_input_ids,
                    attention_mask=batched_attention_mask,
                    generation_mode="image",
                    generation_config=self.generation_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                    max_new_tokens=512)
                    # logits_processor=[cfg_processor])

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
        refined_img_list = result['pixel_values']
        
        print(f"[BATCH_REFINE] Processed {len(refined_img_list)} refined images")
        
        return refined_img_list

    def find_sequence(self, tensor: torch.Tensor, sequence: torch.Tensor) -> int:
        """
        Find the starting position of a sequence within a tensor.
        
        Args:
            tensor (`torch.Tensor`): Input tensor to search in
            sequence (`torch.Tensor`): Sequence to find
            
        Returns:
            `int`: Starting position of the sequence, or -1 if not found
        """
        if len(sequence) > len(tensor):
            return -1
            
        len_needle = sequence.shape[0]
        for i in range(tensor.shape[0] - len_needle + 1):
            if torch.equal(tensor[i:i+len_needle], sequence):
                return i
        return -1

    def _save_generation(self, output: DataProto, prompts: DataProto):
        """Save generated images and metadata."""
        if not self.save_enabled:
            return
            
        batch_size = output.batch.batch_size[0]
        n = self.config.get('n', 1)

        # Get lists from meta_info
        input_texts = output.meta_info.get('input_text', [])
        gen_img_list = output.meta_info.get('gen_img_list', [])
        refined_img_list = output.meta_info.get('refined_gen_img_list', [])
        feedback_texts = output.meta_info.get('feedback_texts', [])
        
        if not gen_img_list:
            return

        for i in range(n):
            gen_id = self.generation_counter
            self.generation_counter += 1

            gen_dir = Path(self.save_dir) / f"gen_{gen_id:06d}"
            gen_dir.mkdir(parents=True, exist_ok=True)

            # Get items for this index
            input_text = input_texts[i] if i < len(input_texts) else ""
            gen_img = gen_img_list[i] if i < len(gen_img_list) else None
            refined_img = refined_img_list[i] if i < len(refined_img_list) else None
            feedback_text = feedback_texts[i] if i < len(feedback_texts) else ""
            
            # Save original generated image
            if gen_img is not None and isinstance(gen_img, PIL.Image.Image):
                gen_img.save(gen_dir / "generated_image.png")
            
            # Save refined image
            if refined_img is not None and isinstance(refined_img, PIL.Image.Image):
                refined_img.save(gen_dir / "refined_image.png")
            
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
            
            # Save text files for easy viewing
            with open(gen_dir / "input_prompt.txt", 'w', encoding='utf-8') as f:
                f.write(input_text)
            
            with open(gen_dir / "feedback.txt", 'w', encoding='utf-8') as f:
                f.write(feedback_text)
        
        print(f"[SAVE] Saved {n} generations to {self.save_dir}")
