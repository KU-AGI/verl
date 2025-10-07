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
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
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
import PIL
from verl.utils.torch_functional import get_response_mask

__all__ = ['ImageUnifiedRollout']

class ImageUnifiedRollout:
    def __init__(self, module: nn.Module, config, model_config): 
        self.config = config
        self.module = module
        from transformers import JanusProcessor
        processor = JanusProcessor.from_pretrained(model_config.local_path)
        self.processor = processor
        self.generation_mode = getattr(config, "generation_mode", None)
        self.feedback_system_prompt = getattr(config, "feedback_system_prompt", "")
        self.refine_system_prompt = getattr(config, "refine_system_prompt", "")
        self.generation_config = None

    async def resume(self, tags: list[str]):
        # No-op for now; this rollout does not manage an external engine.
        return None

    async def release(self):
        # Best-effort release of CUDA cache
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        return None

    async def update_weights(self, weights, **kwargs):
        # Placeholder hook to consume the generator to avoid backpressure when not used.
        # A concrete implementation should map incoming tensors to self.module parameters.
        for _ in weights:
            pass
        return None

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get('micro_batch_size', batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        
        # set generation config
        self.set_generation_config(batch_prompts[0])
        
        output = [self._generate_minibatch_image_generation(p) for p in batch_prompts] # task 1
        output = [self._generate_minibatch_text_generation(o) for o in output] # task 2
        output = [self._generate_minibatch_refine_image_generation(o) for o in output] # task 3
        output = DataProto.concat(output)
        return output
    
    def set_generation_config(self, prompts: DataProto):
        # make sampling args can be overridden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                # already repeat in ray_trainer
                # https://github.com/volcengine/verl/blob/2fdfbdcba6f2e076f64bc47922d8fe6cf7dc7da5/verl/trainer/ppo/ray_trainer.py#L1117
                "num_return_sequences": 1,
            }

        # make config according to generate mode
        self.generation_config = GenerationConfig(**kwargs)
        
        # Add boi_token_id for Janus model
        if hasattr(self.module, 'config') and getattr(self.module.config, 'model_type', None) == 'janus':
            # Get boi_token_id and eoi_token_id from tokenizer
            if hasattr(self.processor, 'tokenizer'):
                boi_token_id = self.processor.tokenizer.convert_tokens_to_ids('<begin_of_image>')
                eoi_token_id = self.processor.tokenizer.convert_tokens_to_ids('<end_of_image>')
                
                if boi_token_id is not None:
                    # Add boi_token_id to generation_config as generation_kwargs
                    if not hasattr(self.generation_config, 'generation_kwargs'):
                        self.generation_config.generation_kwargs = {}
                    self.generation_config.generation_kwargs['boi_token_id'] = boi_token_id
                    
                if eoi_token_id is not None:
                    self.generation_config.generation_kwargs['eoi_token_id'] = eoi_token_id
                    
                # Set pad_token_id if not set
                if self.generation_config.pad_token_id is None:
                    pad_token_id = self.processor.tokenizer.pad_token_id
                    if pad_token_id is not None:
                        self.generation_config.pad_token_id = pad_token_id
                    else:
                        # Use eos_token_id as fallback
                        eos_token_id = self.processor.tokenizer.eos_token_id
                        if eos_token_id is not None:
                            self.generation_config.pad_token_id = eos_token_id
                            
                print(f"DEBUG: boi_token_id: {boi_token_id}, eoi_token_id: {eoi_token_id}")
                print(f"DEBUG: generation_kwargs: {self.generation_config.generation_kwargs}")

    @torch.no_grad()
    def _generate_minibatch_image_generation(self, prompts: DataProto) -> TensorDict:
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']  # left-padded attention_mask
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        pad_token_id = prompts.meta_info['pad_token_id']

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()
        param_ctx = contextlib.nullcontext()
        
        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    generation_mode="image",
                    generation_config=self.generation_config,
                    output_scores=False,  # this is potentially very large
                    return_dict_in_generate=True,
                    use_cache=True,
                    max_new_tokens=512)  # Add max_new_tokens for image generation

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences
        seq_img_mask = output.seq_img_mask

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
            delta_seq_img_mask = torch.zeros(size=(batch_size, delta_length), device=seq.device, dtype=seq_img_mask.dtype)
            seq_img_mask = torch.cat((seq_img_mask, delta_seq_img_mask), dim=1)
        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = self.generation_config.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        decoded_image = self.module.decode_image_tokens(output)
        gen_img = self.processor.postprocess(list(decoded_image.float()), return_tensors="PIL.Image.Image")

        batch = TensorDict(
            {
                'prompts': prompt,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'gen_img': gen_img,
            },
            batch_size=batch_size)

        return batch

    def convert_gen_img_to_base64(self, gen_img: PIL.Image.Image) -> Optional[str]:
        """Convert gen_img to base64 data URL."""
        if not isinstance(gen_img, PIL.Image.Image):
            raise TypeError(f"Unsupported image type: {type(gen_img)}")

        # Convert PIL.Image → base64 data URL
        buffer = BytesIO()
        gen_img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{img_base64}"

    def parse_feedback_text(self, decoded_text: str) -> str:
        pass

    def _generate_minibatch_text_generation(self, batch: TensorDict) -> TensorDict:
        gen_img = batch['gen_img']
        img_url = self.convert_gen_img_to_base64(gen_img)
        
        message = [
            {
                "role": "system", "content": [
                    {"type": "text", "text": self.feedback_system_prompt}
                ]
            },
            {
                "role": "user", "content": [
                    {"type": "image", "url": img_url},
                    # {"type": "text", "text": self.feedback_user_prompt}
                ]
            }
        ]

        input_ids = self.processor.apply_chat_template(
            message, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.module.device, dtype=torch.bfloat16)

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=input_ids,
                    attention_mask=input_ids.attention_mask,
                    position_ids=self.build_position_ids(input_ids.attention_mask),
                    generation_mode="text",
                    generation_config=self.generation_config,
                    output_scores=False,  # this is potentially very large
                    return_dict_in_generate=True,
                    use_cache=True,
                )

        decoded_text = self.processor.decode(output.sequences[0], skip_special_tokens=True)

        # parsing the feedback text
        # feedback_text = self.parse_feedback_text(decoded_text)

        batch = batch.update(
            {
                'feedback_text': decoded_text,
            }
        )

        return batch

    def build_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        pos = (torch.cumsum(attention_mask, dim=1) - 1).clamp(min=0)
        return pos * attention_mask

    def _generate_minibatch_refine_image_generation(self, batch: TensorDict) -> DataProto:
        feedback_text = batch['feedback_text']
        gen_img = batch['gen_img']
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
                    {"type": "text", "text": feedback_text}
                ]
            }
        ]

        input_ids = self.processor.apply_chat_template(
            message, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.module.device, dtype=torch.bfloat16)

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=input_ids,
                    attention_mask=input_ids.attention_mask,
                    generation_mode="image",
                    generation_config=self.generation_config,
                    output_scores=False,  # this is potentially very large
                    return_dict_in_generate=True,
                    use_cache=True,
                )

        decoded_image = self.module.decode_image_tokens(output)
        refined_gen_img = self.processor.postprocess(list(decoded_image.float()), return_tensors="PIL.Image.Image")

        batch = batch.update(
            {
                'refined_gen_img': refined_gen_img,
            }
        )

        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        return DataProto(batch=batch)