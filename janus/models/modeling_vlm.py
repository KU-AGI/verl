# # Copyright (c) 2023-2024 DeepSeek.
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy of
# # this software and associated documentation files (the "Software"), to deal in
# # the Software without restriction, including without limitation the rights to
# # use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# # the Software, and to permit persons to whom the Software is furnished to do so,
# # subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# # FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# # COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# # IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# # CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# import torch
# from attrdict import AttrDict
# from einops import rearrange
# from typing import List, Tuple, Dict, Any
# from transformers import (
#     AutoConfig,
#     AutoModelForCausalLM,
#     LlamaConfig,
#     LlamaForCausalLM,
#     PreTrainedModel,
# )
# from transformers.configuration_utils import PretrainedConfig

# from janus.models.clip_encoder import CLIPVisionTower
# from janus.models.projector import MlpProjector


# class vision_head(torch.nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.output_mlp_projector = torch.nn.Linear(
#             params.n_embed, params.image_token_embed
#         )
#         self.vision_activation = torch.nn.GELU()
#         self.vision_head = torch.nn.Linear(
#             params.image_token_embed, params.image_token_size
#         )

#     def forward(self, x):
#         x = self.output_mlp_projector(x)
#         x = self.vision_activation(x)
#         x = self.vision_head(x)
#         return x


# def model_name_to_cls(cls_name):
#     if "MlpProjector" in cls_name:
#         cls = MlpProjector

#     elif "CLIPVisionTower" in cls_name:
#         cls = CLIPVisionTower

#     elif "VQ" in cls_name:
#         from janus.models.vq_model import VQ_models

#         cls = VQ_models[cls_name]
#     elif "vision_head" in cls_name:
#         cls = vision_head
#     else:
#         raise ValueError(f"class_name {cls_name} is invalid.")

#     return cls


# class VisionConfig(PretrainedConfig):
#     model_type = "vision"
#     cls: str = ""
#     params: AttrDict = {}

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#         self.cls = kwargs.get("cls", "")
#         if not isinstance(self.cls, str):
#             self.cls = self.cls.__name__

#         self.params = AttrDict(kwargs.get("params", {}))


# class AlignerConfig(PretrainedConfig):
#     model_type = "aligner"
#     cls: str = ""
#     params: AttrDict = {}

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#         self.cls = kwargs.get("cls", "")
#         if not isinstance(self.cls, str):
#             self.cls = self.cls.__name__

#         self.params = AttrDict(kwargs.get("params", {}))


# class GenVisionConfig(PretrainedConfig):
#     model_type = "gen_vision"
#     cls: str = ""
#     params: AttrDict = {}

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#         self.cls = kwargs.get("cls", "")
#         if not isinstance(self.cls, str):
#             self.cls = self.cls.__name__

#         self.params = AttrDict(kwargs.get("params", {}))


# class GenAlignerConfig(PretrainedConfig):
#     model_type = "gen_aligner"
#     cls: str = ""
#     params: AttrDict = {}

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#         self.cls = kwargs.get("cls", "")
#         if not isinstance(self.cls, str):
#             self.cls = self.cls.__name__

#         self.params = AttrDict(kwargs.get("params", {}))


# class GenHeadConfig(PretrainedConfig):
#     model_type = "gen_head"
#     cls: str = ""
#     params: AttrDict = {}

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#         self.cls = kwargs.get("cls", "")
#         if not isinstance(self.cls, str):
#             self.cls = self.cls.__name__

#         self.params = AttrDict(kwargs.get("params", {}))


# class MultiModalityConfig(PretrainedConfig):
#     model_type = "multi_modality"
#     vision_config: VisionConfig
#     aligner_config: AlignerConfig

#     gen_vision_config: GenVisionConfig
#     gen_aligner_config: GenAlignerConfig
#     gen_head_config: GenHeadConfig

#     language_config: LlamaConfig

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         vision_config = kwargs.get("vision_config", {})
#         self.vision_config = VisionConfig(**vision_config)

#         aligner_config = kwargs.get("aligner_config", {})
#         self.aligner_config = AlignerConfig(**aligner_config)

#         gen_vision_config = kwargs.get("gen_vision_config", {})
#         self.gen_vision_config = GenVisionConfig(**gen_vision_config)

#         gen_aligner_config = kwargs.get("gen_aligner_config", {})
#         self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

#         gen_head_config = kwargs.get("gen_head_config", {})
#         self.gen_head_config = GenHeadConfig(**gen_head_config)

#         language_config = kwargs.get("language_config", {})
#         if isinstance(language_config, LlamaConfig):
#             self.language_config = language_config
#         else:
#             self.language_config = LlamaConfig(**language_config)


# class MultiModalityPreTrainedModel(PreTrainedModel):
#     config_class = MultiModalityConfig
#     base_model_prefix = "multi_modality"
#     _no_split_modules = []
#     _skip_keys_device_placement = "past_key_values"


# class MultiModalityCausalLM(MultiModalityPreTrainedModel):
#     def __init__(self, config: MultiModalityConfig):
#         super().__init__(config)

#         vision_config = config.vision_config
#         vision_cls = model_name_to_cls(vision_config.cls)
#         self.vision_model = vision_cls(**vision_config.params)

#         aligner_config = config.aligner_config
#         aligner_cls = model_name_to_cls(aligner_config.cls)
#         self.aligner = aligner_cls(aligner_config.params)

#         gen_vision_config = config.gen_vision_config
#         gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
#         self.gen_vision_model = gen_vision_cls()

#         gen_aligner_config = config.gen_aligner_config
#         gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
#         self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

#         gen_head_config = config.gen_head_config
#         gen_head_cls = model_name_to_cls(gen_head_config.cls)
#         self.gen_head = gen_head_cls(gen_head_config.params)

#         self.gen_embed = torch.nn.Embedding(
#             gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
#         )

#         language_config = config.language_config
#         self.language_model = LlamaForCausalLM(language_config)

#     def prepare_inputs_embeds(
#         self,
#         input_ids: torch.LongTensor,
#         pixel_values: torch.FloatTensor,
#         images_seq_mask: torch.LongTensor,
#         images_emb_mask: torch.LongTensor,
#         **kwargs,
#     ):
#         """

#         Args:
#             input_ids (torch.LongTensor): [b, T]
#             pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
#             images_seq_mask (torch.BoolTensor): [b, T]
#             images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

#             assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

#         Returns:
#             input_embeds (torch.Tensor): [b, T, D]
#         """

#         bs, n = pixel_values.shape[0:2]
#         images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
#         # [b x n, T2, D]
#         images_embeds = self.aligner(self.vision_model(images))

#         # [b x n, T2, D] -> [b, n x T2, D]
#         images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
#         # [b, n, T2] -> [b, n x T2]
#         images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

#         # [b, T, D]
#         input_ids[input_ids < 0] = 0  # ignore the image embeddings
#         inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

#         # replace with the image embeddings
#         inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

#         return inputs_embeds

#     def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
#         return self.gen_aligner(self.gen_embed(image_ids))

#     def set_processor(self, processor):
#         """Set processor for image token handling"""
#         self.processor = processor

#     def _merge_text_and_image_embeds(
#         self,
#         text_embeds: torch.Tensor,
#         image_embeds: torch.Tensor,
#         all_image_start_indices: List[List[int]],
#     ) -> torch.Tensor:
#         """Merge text and image embeddings at specified positions (no in-place ops)."""
#         B, T, D = text_embeds.shape
#         num_img = len(all_image_start_indices[0])
#         num_img_tokens = self.processor.num_image_tokens

#         # image_embeds: [B * num_img * num_img_tokens, D] 형태라고 가정하고 있었음
#         # 위에서 view 로 맞춰놨던 걸 그대로 반영
#         image_embeds = image_embeds.view(B, num_img, num_img_tokens, D)  # [B, num_img, num_img_tokens, D]

#         merged_list = []

#         for b in range(B):
#             pieces = []
#             cursor = 0

#             for img_idx in range(num_img):
#                 start = all_image_start_indices[b][img_idx]
#                 end = start + num_img_tokens

#                 # 텍스트 부분
#                 if start > cursor:
#                     pieces.append(text_embeds[b, cursor:start])

#                 # 이미지 토큰 부분
#                 pieces.append(image_embeds[b, img_idx])

#                 cursor = end

#             # 뒤에 남은 텍스트
#             if cursor < T:
#                 pieces.append(text_embeds[b, cursor:])

#             merged_seq = torch.cat(pieces, dim=0)  # (L_b, D)
#             merged_list.append(merged_seq)

#         # 각 시퀀스를 최대 길이에 맞춰 pad
#         max_len = max(seq.size(0) for seq in merged_list)
#         out = text_embeds.new_zeros((B, max_len, D))

#         for b, seq in enumerate(merged_list):
#             out[b, : seq.size(0)] = seq

#         return out

#     def _remove_padding_and_concat_with_embeds(
#         self,
#         input_embeds: torch.Tensor,
#         attention_mask: torch.Tensor,
#         output_embeds: torch.Tensor,
#         output_mask: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
#         """Remove padding and concatenate embeddings (no in-place ops)."""
#         device = input_embeds.device
#         dtype = input_embeds.dtype

#         B, _, D = input_embeds.size()
#         sequences = []
#         masks = []
#         output_starts: List[int] = []
#         max_total_len = 0

#         for i in range(B):
#             # 1) 왼쪽 padding 제거 (input)
#             input_valid_mask = attention_mask[i] == 1
#             if input_valid_mask.any():
#                 first_valid = input_valid_mask.nonzero(as_tuple=False)[0].item()
#                 valid_input_embeds = input_embeds[i, first_valid:]        # (L_in, D)
#                 valid_input_mask = attention_mask[i, first_valid:]        # (L_in,)
#             else:
#                 valid_input_embeds = input_embeds.new_zeros((0, D))
#                 valid_input_mask = attention_mask.new_zeros((0,))

#             # 2) 오른쪽 padding 제거 (output)
#             output_valid_mask = output_mask[i] == 1
#             if output_valid_mask.any():
#                 last_valid = output_valid_mask.nonzero(as_tuple=False)[-1].item()
#                 valid_output_embeds = output_embeds[i, : last_valid + 1]  # (L_out, D)
#                 valid_output_mask = output_mask[i, : last_valid + 1]      # (L_out,)
#             else:
#                 valid_output_embeds = output_embeds.new_zeros((0, D))
#                 valid_output_mask = output_mask.new_zeros((0,))

#             # 3) concat
#             concat_embeds = torch.cat([valid_input_embeds, valid_output_embeds], dim=0)  # (L_i, D)
#             concat_mask = torch.cat([valid_input_mask, valid_output_mask], dim=0)        # (L_i,)

#             sequences.append(concat_embeds)
#             masks.append(concat_mask)
#             output_starts.append(valid_input_embeds.size(0))
#             max_total_len = max(max_total_len, concat_embeds.size(0))

#         if max_total_len == 0:
#             # 극단적으로 다 비어 있는 경우
#             concat_embeds_padded = input_embeds.new_zeros((B, 1, D))
#             concat_mask_padded = attention_mask.new_zeros((B, 1))
#             position_ids = torch.zeros((B, 1), dtype=torch.long, device=device)
#             return concat_embeds_padded, concat_mask_padded, position_ids, output_starts

#         # 4) pad 도 in-place 없이 cat 로만 만들기
#         padded_embeds = []
#         padded_masks = []

#         for embeds, mask in zip(sequences, masks):
#             L = embeds.size(0)
#             pad_len = max_total_len - L
#             if pad_len > 0:
#                 pad_embeds = embeds.new_zeros((pad_len, D))
#                 pad_mask = mask.new_zeros((pad_len,))
#                 embeds = torch.cat([embeds, pad_embeds], dim=0)   # (max_total_len, D)
#                 mask = torch.cat([mask, pad_mask], dim=0)         # (max_total_len,)
#             padded_embeds.append(embeds)
#             padded_masks.append(mask)

#         concat_embeds_padded = torch.stack(padded_embeds, dim=0)  # (B, max_total_len, D)
#         concat_mask_padded = torch.stack(padded_masks, dim=0)     # (B, max_total_len)

#         position_ids = (concat_mask_padded.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)
#         position_ids = position_ids.masked_fill(concat_mask_padded == 0, 0)

#         return concat_embeds_padded, concat_mask_padded, position_ids, output_starts

#     def _process_task1_data(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
#         """Process Task 1 (ReasonGen: Text + Image Generation) data with CFG logic applied"""
#         # 1. 원본 데이터 로드
#         task1_input_ids = batch["task1_input_ids"].long() # [B, L_in]
#         task1_attention_mask = batch["task1_attention_mask"] # [B, L_in]

#         # 2. Conditional Embeddings 생성
#         task1_input_embeds = self.language_model.get_input_embeddings()(task1_input_ids) # [B, L_in, D]

#         # 3. Unconditional Embeddings 생성 (Masking 로직 적용)
#         uncond_input_embeds = task1_input_embeds.clone()
#         pad_id = torch.tensor(self.processor.pad_id, device=task1_input_ids.device)

#         with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#             pad_embed = self.language_model.get_input_embeddings()(pad_id).squeeze() # [D]

#         start_marker = torch.tensor([100601], device=task1_input_ids.device) # <|User|>
#         end_marker = torch.tensor([100602], device=task1_input_ids.device)   # <|Assistant|>

#         def find_sequence(inp_id, marker):
#             len_needle = marker.shape[0]
#             for i in range(inp_id.shape[0] - len_needle + 1):
#                 if torch.equal(inp_id[i:i+len_needle], marker):
#                     return i
#             return -1

#         for i, row in enumerate(task1_input_ids):
#             start_pos = find_sequence(row, start_marker)
#             end_pos = find_sequence(row, end_marker)

#             if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
#                 # User 프롬프트 구간을 pad_embed로 치환 (Uncond)
#                 uncond_input_embeds[i, start_pos : end_pos + 2] = pad_embed

#         # 4. Batch Interleaving (B -> B*2)
#         batch_size, seq_len, embed_dim = task1_input_embeds.shape

#         final_input_embeds = torch.zeros((batch_size * 2, seq_len, embed_dim),
#                                         dtype=task1_input_embeds.dtype, device=task1_input_embeds.device)
#         final_input_embeds[0::2] = task1_input_embeds
#         final_input_embeds[1::2] = uncond_input_embeds

#         final_attention_mask = task1_attention_mask.repeat_interleave(2, dim=0) # [B*2, L_in]

#         # 5. Output (Text + Image) 데이터 처리
#         combined_output_ids = batch["task1_gen_img_tokens"].long()  # [B, L_out]
#         seq_img_mask = batch["task1_seq_img_mask"]  # [B, L_out] - True for image codebook tokens

#         # Vectorized embedding (gradient-safe: no in-place on zero tensor)
#         # Text embedding: mask out image positions to avoid OOB in text vocab
#         text_safe_ids = combined_output_ids.clone()
#         text_safe_ids[seq_img_mask] = 0
#         text_output_embeds = self.language_model.get_input_embeddings()(text_safe_ids)  # [B, L_out, D]

#         # Image embedding: mask out text positions to avoid OOB in codebook vocab
#         img_safe_ids = combined_output_ids.clone()
#         img_safe_ids[~seq_img_mask] = 0
#         img_output_embeds = self.gen_aligner(self.gen_embed(img_safe_ids))  # [B, L_out, D]

#         # Blend: image positions → img_output_embeds, text positions → text_output_embeds
#         combined_output_embeds = torch.where(
#             seq_img_mask.unsqueeze(-1), img_output_embeds, text_output_embeds
#         )  # [B, L_out, D]

#         # Interleave Output Embeds (B -> B*2)
#         final_output_embeds = combined_output_embeds.repeat_interleave(2, dim=0) # [B*2, L_out, D]

#         # Response mask 확장
#         task1_response_mask = batch["task1_response_mask"] # [B, L_out]
#         final_response_mask = task1_response_mask.repeat_interleave(2, dim=0) # [B*2, L_out]

#         # Store seq_img_mask for forward pass
#         self._seq_img_mask = seq_img_mask.repeat_interleave(2, dim=0) # [B*2, L_out]

#         # 6. Padding 제거 및 Concatenate (최종 시퀀스 구성)
#         concat_embeds, concat_mask, position_ids, output_starts = self._remove_padding_and_concat_with_embeds(
#             final_input_embeds, final_attention_mask, final_output_embeds, final_response_mask
#         )

#         return concat_embeds, concat_mask, position_ids, output_starts

#     def _process_task2_data(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
#         """Process Task 2 (Feedback Generation) data"""
#         # Input processing
#         task2_input_ids = batch["task2_input_ids"]
#         task2_attention_mask = batch["task2_attention_mask"]
#         gen_imgs_pixel_values = batch["task1_gen_imgs_pixel_values"]

#         task2_text_embeds = self.language_model.get_input_embeddings()(task2_input_ids)
#         # Ensure vision_model encode is done with torch.no_grad() since it's frozen
#         with torch.no_grad():
#             vision_features = self.vision_model(gen_imgs_pixel_values)
#         task2_image_embeds = self.aligner(vision_features)

#         # Find image token positions
#         pos_list = []
#         invalid_mask = []
#         for i, ids in enumerate(task2_input_ids):
#             pos_tensor = (ids == self.processor.image_id).nonzero(as_tuple=False)
#             if pos_tensor.numel() == 0:
#                 # No image placeholder found, mark this sample as invalid
#                 print(f"[WARNING] No image placeholder token (id={self.processor.image_id}) found in task2_input_ids[{i}]. Excluding from training.")
#                 pos_list.append([0])  # Use dummy position
#                 invalid_mask.append(i)
#             else:
#                 pos = pos_tensor[0].item()
#                 pos_list.append([pos])
#         task2_image_start_indices = pos_list

#         # Zero out response_mask for invalid samples to exclude from loss calculation
#         if len(invalid_mask) > 0 and "task2_response_mask" in batch:
#             for idx in invalid_mask:
#                 batch["task2_response_mask"][idx] = 0

#         task2_merged_embeds = self._merge_text_and_image_embeds(
#             task2_text_embeds, task2_image_embeds, task2_image_start_indices
#         )

#         # Output processing
#         feedback_ids = batch["task2_feedback_ids"]
#         task2_feedback_embeds = self.language_model.get_input_embeddings()(feedback_ids)
#         task2_response_mask = batch["task2_response_mask"]

#         # Remove padding and concatenate
#         concat_embeds, concat_mask, position_ids, output_starts = self._remove_padding_and_concat_with_embeds(
#             task2_merged_embeds, task2_attention_mask, task2_feedback_embeds, task2_response_mask
#         )

#         return concat_embeds, concat_mask, position_ids, output_starts

#     def _process_task3_data(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
#         """Process Task 3 (Regen Image Generation) data with CFG logic applied"""
#         # 1. 원본 데이터 로드 (정수형 캐스팅 필수)
#         task3_input_ids = batch["task3_input_ids"].long()
#         task3_attention_mask = batch["task3_attention_mask"]
#         task3_image_ids = batch["task1_gen_img_tokens"].long() # 이전 단계에서 생성된 이미지 토큰

#         # 2. 텍스트 임베딩 및 CFG 마스킹 (Uncond 생성)
#         task3_text_embeds = self.language_model.get_input_embeddings()(task3_input_ids)
#         uncond_text_embeds = task3_text_embeds.clone()

#         pad_id = torch.tensor(self.processor.pad_id, device=task3_input_ids.device)
#         with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#             pad_embed = self.language_model.get_input_embeddings()(pad_id).squeeze() # [D]

#         # Task 3용 마커: <end_of_image>\n 부터 <|Assistant|> 직전까지 마스킹
#         start_marker = torch.tensor([100593, 185], device=task3_input_ids.device) 
#         end_marker = torch.tensor([100602], device=task3_input_ids.device)

#         def find_sequence(inp_id, marker):
#             len_needle = marker.shape[0]
#             for i in range(inp_id.shape[0] - len_needle + 1):
#                 if torch.equal(inp_id[i:i+len_needle], marker):
#                     return i
#             return -1

#         for i, row in enumerate(task3_input_ids):
#             start_pos = find_sequence(row, start_marker)
#             end_pos = find_sequence(row, end_marker)
#             if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
#                 # 피드백 텍스트 구간을 마스킹
#                 uncond_text_embeds[i, start_pos + 1 : end_pos + 2] = pad_embed

#         # 3. 이미지 임베딩 준비
#         task3_image_embeds = self.gen_aligner(self.gen_embed(task3_image_ids))

#         # 4. 이미지 토큰 위치 찾기 (Cond/Uncond 공통 사용)
#         pos_list = []
#         invalid_mask = []
#         for i, ids in enumerate(task3_input_ids):
#             pos_tensor = (ids == self.processor.image_id).nonzero(as_tuple=False)
#             if pos_tensor.numel() == 0:
#                 invalid_mask.append(i)
#                 pos_list.append([0])
#             else:
#                 pos = pos_tensor[0].item()
#                 pos_list.append([pos])
        
#         # 5. Merge (Cond/Uncond 각각 수행)
#         cond_merged_embeds = self._merge_text_and_image_embeds(
#             task3_text_embeds, task3_image_embeds, pos_list
#         )
#         uncond_merged_embeds = self._merge_text_and_image_embeds(
#             uncond_text_embeds, task3_image_embeds, pos_list
#         )

#         # 6. Batch Interleaving (B -> B*2)
#         batch_size, seq_len, embed_dim = cond_merged_embeds.shape
#         final_merged_embeds = torch.zeros((batch_size * 2, seq_len, embed_dim), 
#                                          dtype=cond_merged_embeds.dtype, device=cond_merged_embeds.device)
#         final_merged_embeds[0::2] = cond_merged_embeds
#         final_merged_embeds[1::2] = uncond_merged_embeds
        
#         final_attention_mask = task3_attention_mask.repeat_interleave(2, dim=0)

#         # 7. Output (Regenerated Image) 데이터 처리 및 2배 확장
#         task3_regen_image_ids = batch["task3_regen_img_tokens"].long()
#         task3_regen_img_embeds = self.gen_aligner(self.gen_embed(task3_regen_image_ids))
#         final_regen_img_embeds = task3_regen_img_embeds.repeat_interleave(2, dim=0)

#         # 8. Response Mask 처리
#         task3_response_mask = batch["task3_response_mask"].clone()
#         if len(invalid_mask) > 0:
#             for idx in invalid_mask:
#                 task3_response_mask[idx] = 0
#         final_response_mask = task3_response_mask.repeat_interleave(2, dim=0)

#         # 9. 최종 시퀀스 구성
#         concat_embeds, concat_mask, position_ids, output_starts = self._remove_padding_and_concat_with_embeds(
#             final_merged_embeds, final_attention_mask, final_regen_img_embeds, final_response_mask
#         )

#         return concat_embeds, concat_mask, position_ids, output_starts

#     def forward(
#         self,
#         task_id: int = 1,
#         batch: Dict[str, Any] = None,
#         inputs_embeds: torch.Tensor = None,
#         attention_mask: torch.Tensor = None,
#         position_ids: torch.Tensor = None,
#         output_starts: List[int] = None,
#         **kwargs,
#     ):
#         """
#         Unified forward pass for multi-task model.

#         Args:
#             task_id: Task identifier (1: Image Generation, 2: Feedback Generation, 3: Regen Image Generation)
#             batch: Dict containing all task data (if provided, will process data internally)
#             inputs_embeds: Pre-processed input embeddings (optional, if batch is not provided)
#             attention_mask: Attention mask (optional, if batch is not provided)
#             position_ids: Position indices (optional, if batch is not provided)
#             output_starts: Output start positions (optional, if batch is not provided)

#         Returns:
#             Model output with logits (output_starts stored in self._output_starts)
#         """
#         from transformers.modeling_outputs import CausalLMOutputWithPast
        
#         self.guidance_scale = 1.0

#         # If batch is provided, process data internally
#         if batch is not None:
#             if task_id == 1:
#                 inputs_embeds, attention_mask, position_ids, output_starts = self._process_task1_data(batch)
#                 # Store output_starts in instance variable for retrieval
#                 self._output_starts = output_starts[0::2] if output_starts is not None else None
#             elif task_id == 2:
#                 inputs_embeds, attention_mask, position_ids, output_starts = self._process_task2_data(batch)
#                 # Store output_starts in instance variable for retrieval
#                 self._output_starts = output_starts
#             elif task_id == 3:
#                 inputs_embeds, attention_mask, position_ids, output_starts = self._process_task3_data(batch)
#                 # Store output_starts in instance variable for retrieval
#                 self._output_starts = output_starts[0::2] if output_starts is not None else None
#             else:
#                 raise ValueError(f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")

#         # Forward pass based on task_id
#         if task_id == 1:
#             # Task 1: ReasonGen (Text + Image Generation)
#             # Returns FULL sequence logits [B, total_len, vocab_size]
#             output = self.language_model.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#             )
#             hidden_states = output.last_hidden_state  # [B*2, total_len, D]

#             seq_img_mask = getattr(self, '_seq_img_mask', None)

#             if seq_img_mask is not None:
#                 vocab_size = self.language_model.lm_head.out_features
#                 img_vocab_size = self.gen_head.vision_head.out_features

#                 # Step 1: lm_head on ALL positions (default text logits)
#                 logits = self.language_model.lm_head(hidden_states).clone()  # [B*2, total_len, V]

#                 # Step 2: Overwrite image-predicting positions with gen_head
#                 # Causal LM: hidden[i, j] predicts token[j+1]
#                 # output_token[k] is at position output_starts[i]+k
#                 # hidden[i, output_starts[i]-1+k] predicts output_token[k]
#                 # If seq_img_mask[i, k]==True → use gen_head
#                 img_batch_indices = []
#                 img_seq_indices = []

#                 for i in range(hidden_states.size(0)):
#                     start = output_starts[i]
#                     img_pos_in_output = seq_img_mask[i].nonzero(as_tuple=True)[0]
#                     if len(img_pos_in_output) > 0:
#                         full_pos = start - 1 + img_pos_in_output
#                         valid = full_pos < hidden_states.size(1)
#                         if valid.any():
#                             full_pos = full_pos[valid]
#                             img_batch_indices.append(
#                                 torch.full((full_pos.size(0),), i, device=hidden_states.device, dtype=torch.long)
#                             )
#                             img_seq_indices.append(full_pos)

#                 if img_batch_indices:
#                     bi = torch.cat(img_batch_indices)
#                     si = torch.cat(img_seq_indices)

#                     img_hidden = hidden_states[bi, si]  # [N, D]
#                     img_logits = self.gen_head(img_hidden)  # [N, img_vocab_size]

#                     pad_cols = torch.full(
#                         (img_logits.size(0), vocab_size - img_vocab_size), -1e9,
#                         dtype=img_logits.dtype, device=img_logits.device
#                     )
#                     padded_img_logits = torch.cat([img_logits, pad_cols], dim=1)  # [N, V]

#                     logits[bi, si] = padded_img_logits

#                 # Step 3: CFG (only on image positions)
#                 logit_cond = logits[0::2]    # [B, total_len, V]
#                 logit_uncond = logits[1::2]  # [B, total_len, V]

#                 final_logits = logit_cond.clone()
#                 B = hidden_states.size(0) // 2

#                 for i in range(B):
#                     start = output_starts[i * 2]
#                     img_pos = seq_img_mask[i * 2].nonzero(as_tuple=True)[0]
#                     if len(img_pos) > 0:
#                         full_pos = start - 1 + img_pos
#                         valid = full_pos < logit_cond.size(1)
#                         if valid.any():
#                             full_pos = full_pos[valid]
#                             final_logits[i, full_pos] = (
#                                 logit_uncond[i, full_pos] +
#                                 self.guidance_scale * (logit_cond[i, full_pos] - logit_uncond[i, full_pos])
#                             )

#                 logits = final_logits

#             else:
#                 # Fallback: treat all as image (backward compatibility)
#                 logits = self.gen_head(hidden_states)
#                 logit_cond = logits[0::2, :]
#                 logit_uncond = logits[1::2, :]
#                 logits = logit_uncond + self.guidance_scale * (logit_cond - logit_uncond)

#             return CausalLMOutputWithPast(
#                 logits=logits,
#                 past_key_values=None,
#                 hidden_states=output.hidden_states if hasattr(output, 'hidden_states') else None,
#                 attentions=output.attentions if hasattr(output, 'attentions') else None,
#             )

#         elif task_id == 2:
#             # Task 2: Feedback Generation (text output)
#             return self.language_model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#             )

#         elif task_id == 3:
#             # Task 3: Regen Image Generation
#             output = self.language_model.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#             )
#             logits = self.gen_head(output.last_hidden_state)

#             logit_cond = logits[0::2, :]
#             logit_uncond = logits[1::2, :]
#             logits = logit_uncond + self.guidance_scale * (logit_cond-logit_uncond)

#             return CausalLMOutputWithPast(
#                 logits=logits,
#                 past_key_values=None,
#                 hidden_states=output.hidden_states if hasattr(output, 'hidden_states') else None,
#                 attentions=output.attentions if hasattr(output, 'attentions') else None,
#             )
#         else:
#             raise ValueError(f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")

#     def get_output_starts(self):
#         """Retrieve the output_starts from the last forward pass"""
#         return getattr(self, '_output_starts', None)


# AutoConfig.register("vision", VisionConfig)
# AutoConfig.register("aligner", AlignerConfig)
# AutoConfig.register("gen_vision", GenVisionConfig)
# AutoConfig.register("gen_aligner", GenAlignerConfig)
# AutoConfig.register("gen_head", GenHeadConfig)
# AutoConfig.register("multi_modality", MultiModalityConfig)
# AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)

# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig

from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector
import numpy as np
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist

def checkpoint_wrapper(fn, gradient_checkpointing_kwargs):
    def wrapper(*args, **kwargs):
        return checkpoint(fn, *args, **kwargs, **gradient_checkpointing_kwargs)
    return wrapper


class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)
        self.config = config
        self.enable_gradient_checkpointing = False

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))

    # only implement image mode for now
    @torch.inference_mode()
    def generate(self, 
                 input_ids, 
                 attention_mask, 
                 do_sample, 
                 max_new_tokens, 
                 eos_token_id, 
                 pad_token_id, 
                 generation_config=None, 
                 output_scores=False, 
                 return_dict_in_generate=True, 
                 use_cache=True
                ):
        if generation_config is None:
            generation_config = {}
        cfg_weight = generation_config.get("cfg_weight", 5.0)
        image_token_num_per_image = generation_config.get("image_token_num_per_image", 576)
        img_size = generation_config.get("img_size", 384)
        patch_size = generation_config.get("patch_size", 16)
        temperature = generation_config.get("temperature", 1.0)
        
        parallel_size = input_ids.shape[0]
        
        num_pad = torch.sum(input_ids == pad_token_id, dim=-1)
        last_pad_idx = num_pad
        sentence_start_token_id = input_ids[0, last_pad_idx[0]]
        assert sentence_start_token_id == 100000, f"sentence start token id should be 100000 for Janus Pro model, but got {sentence_start_token_id}"
        
        if cfg_weight == 1.0:
            tokens = input_ids
            duplicated_parallel_size = parallel_size
        else:
            duplicated_parallel_size = parallel_size*2
            tokens = torch.zeros((parallel_size*2, input_ids.shape[1]), dtype=torch.int).cuda()
            for i in range(parallel_size*2):
                tokens[i, :] = input_ids[i//2]
                if i % 2 != 0:
                    tokens[i, :-1] = pad_token_id
                    tokens[i, last_pad_idx[i//2]] = sentence_start_token_id
                assert attention_mask[i//2, last_pad_idx[i//2]] == 1, f"attention mask should be 1 at the sentence start token"
            attention_mask = attention_mask.repeat_interleave(2, dim=0)
                
        inputs_embeds = self.language_model.get_input_embeddings()(tokens)
                
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
        
        
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None).cuda()
        
        for i in range(image_token_num_per_image):
            outputs = self.language_model.model(inputs_embeds=inputs_embeds, 
                                                attention_mask=attention_mask,
                                                use_cache=use_cache, 
                                                past_key_values=outputs.past_key_values if i != 0 else None,
                                                position_ids=position_ids
                                                )
            hidden_states = outputs.last_hidden_state
            
            logits = self.gen_head(hidden_states[:, -1, :])
            if cfg_weight == 1.0:
                logits = logits
            else:
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                
                logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            
            if not do_sample:
                next_token = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            
            if cfg_weight == 1.0:
                next_token = next_token.view(-1)
            else:
                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.prepare_gen_img_embeds(next_token)
            
            inputs_embeds = img_embeds.unsqueeze(dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((duplicated_parallel_size, 1), dtype=torch.int, device=attention_mask.device)], dim=1)
            position_ids = (position_ids[:,-1:] + 1).cuda()
            
            
        
        dec = self.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
        
        sequences = torch.cat((input_ids, generated_tokens), dim=1)
        seq_img_mask = torch.zeros_like(sequences, dtype=torch.bool)
        seq_img_mask[:, input_ids.shape[1]:] = True
            
        output = AttrDict(
            generated_tokens=generated_tokens,
            input_ids=input_ids,
            sequences=sequences,
            seq_img_mask=seq_img_mask,
            gen_img=visual_img,
        )
        return output
    
    @torch.inference_mode()
    def text_generate(self,
                 input_ids, 
                 attention_mask, 
                 do_sample, 
                 max_new_tokens, 
                 eos_token_id, 
                 pad_token_id, 
                 image_start_token_id,
                 generation_config=None, 
                 output_scores=False,
                 return_dict_in_generate=True, 
                 use_cache=True,
                 early_stop_prob=0.0,
                ):
        if generation_config is None:
            generation_config = {}
        temperature = generation_config.get("temperature", 1.0)

        print(f"[TEMPERATURE]: {temperature}")
        print(f"[generation_config]: {generation_config}")
        generated_tokens = torch.zeros((input_ids.shape[0], max_new_tokens), dtype=torch.int).cuda()

        ended = torch.zeros((input_ids.shape[0],), dtype=torch.bool).cuda()
        
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None).cuda()
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        generated_length = 0
        early_stopped = torch.zeros((1,), dtype=torch.int).cuda()
        for i in range(max_new_tokens):
            outputs = self.language_model.model(inputs_embeds=inputs_embeds, 
                                                attention_mask=attention_mask,
                                                position_ids=position_ids,
                                                use_cache=use_cache, 
                                                past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = self.language_model.lm_head(hidden_states[:, -1, :])
            if not do_sample:
                next_token = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
            next_emb = self.language_model.get_input_embeddings()(next_token)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            inputs_embeds = next_emb.unsqueeze(dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.shape[0], 1), dtype=torch.int, device=attention_mask.device)], dim=1)
            position_ids = (position_ids[:,-1:] + 1).cuda()
            ended = ended | (next_token == eos_token_id)
            ended = ended | (next_token == image_start_token_id)
            generated_length += 1
            early_stoped = early_stopped | (torch.rand(1).cuda() < early_stop_prob)
            local_ended = ended.all().to(torch.int)
            local_ended = torch.maximum(local_ended, early_stoped.to(torch.int))
            if local_ended == 1:
                break
        
        generated_tokens = generated_tokens[:, :generated_length]
        output = AttrDict(
            generated_tokens=generated_tokens,
            input_ids=input_ids,
            sequences=torch.cat((input_ids, generated_tokens), dim=1),
        )
        return output
            
    @torch.inference_mode()
    def text_img_generate(self,
                 input_ids, 
                 attention_mask, 
                 do_sample, 
                 max_new_tokens, 
                 eos_token_id, 
                 pad_token_id, 
                 image_start_token_id,
                 generation_config=None, 
                 output_scores=False, 
                 return_dict_in_generate=True, 
                 use_cache=True,
                 text_early_stop_prob=0.0,
                ):
        if generation_config is None:
            generation_config = {}
        cfg_weight = generation_config.get("cfg_weight", 5.0)
        image_token_num_per_image = generation_config.get("image_token_num_per_image", 576)
        img_size = generation_config.get("img_size", 384)
        patch_size = generation_config.get("patch_size", 16)
        temperature = generation_config.get("temperature", 1.0)
        
        text_output = self.text_generate(input_ids,
                                        attention_mask,
                                        do_sample,
                                        max_new_tokens-image_token_num_per_image,
                                        eos_token_id=eos_token_id,
                                        pad_token_id=pad_token_id,
                                        image_start_token_id=image_start_token_id,
                                        generation_config=generation_config,
                                        output_scores=output_scores,
                                        return_dict_in_generate=return_dict_in_generate,
                                        use_cache=use_cache,
                                        early_stop_prob=text_early_stop_prob,
                                        )
        
        text_generated_tokens = text_output.generated_tokens
        
        new_input_ids = torch.concatenate((input_ids, text_generated_tokens), dim=1)
        max_length = new_input_ids.shape[1]
        cat_lengths = []
        # change to left padding
        for i in range(new_input_ids.shape[0]):
            new_input_ids[i, :] = torch.where(new_input_ids[i, :] == eos_token_id, image_start_token_id, new_input_ids[i, :])
            img_start_mask = new_input_ids[i] == image_start_token_id
            if not img_start_mask.any(): # no padding
                new_input_ids[i, -1] = image_start_token_id
                cat_length = 0
                # print("no image start token")
            else:
                img_start_idx = torch.argwhere(img_start_mask)[0][0]
                new_input_ids[i] = torch.cat([torch.ones((max_length-img_start_idx-1,), dtype=torch.long).cuda()*pad_token_id, new_input_ids[i, :img_start_idx+1]], dim=0)
                cat_length = max_length - img_start_idx - 1
            cat_lengths.append(cat_length)
            
        attention_mask = torch.where(new_input_ids == pad_token_id, 0, 1).to(torch.bool)
        
        img_output = self.generate(new_input_ids,
                                    attention_mask=attention_mask,
                                    do_sample=do_sample,
                                    max_new_tokens=max_new_tokens,
                                    eos_token_id=eos_token_id,
                                    pad_token_id=pad_token_id,
                                    generation_config=generation_config,
                                    output_scores=output_scores,
                                    return_dict_in_generate=return_dict_in_generate,
                                    use_cache=use_cache
                                    )
        
        generated_tokens = img_output.generated_tokens
        input_ids = img_output.input_ids
        sequences = img_output.sequences
        seq_img_mask = img_output.seq_img_mask
        
        # align the prompt and response
        for i in range(sequences.shape[0]):
            if cat_lengths[i] == 0: continue
            sequences[i, :-cat_lengths[i]] = sequences[i, cat_lengths[i]:].clone()
            sequences[i, -cat_lengths[i]:] = pad_token_id
            sequences[i, -cat_lengths[i]] = eos_token_id
            seq_img_mask[i, :-cat_lengths[i]] = seq_img_mask[i, cat_lengths[i]:].clone()
            seq_img_mask[i, -cat_lengths[i]:] = False
        gen_img = img_output.gen_img
        
        output = AttrDict(
            text_tokens=new_input_ids,
            text_gen_tokens=text_generated_tokens,
            img_tokens=generated_tokens,
            sequences=sequences,
            seq_img_mask=seq_img_mask,
            gen_img=gen_img,
        )        
        return output
    
    def text_img_generate_two_stage(self,
                 input_ids, 
                 img_input_ids,
                 attention_mask, 
                 do_sample, 
                 max_new_tokens, 
                 eos_token_id, 
                 pad_token_id, 
                 image_start_token_id,
                 generation_config=None, 
                 output_scores=False, 
                 return_dict_in_generate=True, 
                 use_cache=True):
        
        if generation_config is None:
            generation_config = {}
        cfg_weight = generation_config.get("cfg_weight", 5.0)
        image_token_num_per_image = generation_config.get("image_token_num_per_image", 576)
        img_size = generation_config.get("img_size", 384)
        patch_size = generation_config.get("patch_size", 16)
        temperature = generation_config.get("temperature", 1.0)
        
        text_output = self.text_generate(input_ids,
                        attention_mask,
                        do_sample,
                        max_new_tokens-image_token_num_per_image,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        image_start_token_id=image_start_token_id,
                        generation_config=generation_config,
                        output_scores=output_scores,
                        return_dict_in_generate=return_dict_in_generate,
                        use_cache=use_cache
                        )
        
        generated_tokens = text_output.generated_tokens
        input_ids_for_img = torch.cat([img_input_ids[:, :-4], generated_tokens, img_input_ids[:, -4:]], dim=1) # Last two tokens are <Assistant>:
        max_length = input_ids_for_img.shape[1]
        
        cat_lengths = []
        # change to left padding
        for i in range(input_ids_for_img.shape[0]):
            input_ids_for_img[i, :] = torch.where(input_ids_for_img[i, :] == eos_token_id, image_start_token_id, input_ids_for_img[i, :])
            img_start_mask = input_ids_for_img[i] == image_start_token_id
            if not img_start_mask.any(): # no padding
                input_ids_for_img[i, -1] = image_start_token_id
                cat_length = 0
                # print("no image start token")
            else:
                img_start_idx = torch.argwhere(img_start_mask)[0][0]
                input_ids_for_img[i] = torch.cat([torch.ones((max_length-img_start_idx-1,), dtype=torch.long).cuda()*pad_token_id, input_ids_for_img[i, :img_start_idx+1]], dim=0)
                cat_length = max_length - img_start_idx - 1
            cat_lengths.append(cat_length)
            
        attention_mask_for_img = torch.where(input_ids_for_img == pad_token_id, 0, 1).to(torch.bool)
        
        img_output = self.generate(input_ids_for_img,
                                    attention_mask=attention_mask_for_img,
                                    do_sample=do_sample,
                                    max_new_tokens=image_token_num_per_image,
                                    eos_token_id=eos_token_id,
                                    pad_token_id=pad_token_id,
                                    generation_config=generation_config,
                                    output_scores=output_scores,
                                    return_dict_in_generate=return_dict_in_generate,
                                    use_cache=use_cache
                                    )
        
        seq = img_output.sequences
        text_sequence = text_output.sequences
        seq_img_mask = img_output.seq_img_mask
        gen_img = img_output.gen_img
        
        
        output = AttrDict(
            text_tokens=input_ids_for_img,
            text_gen_tokens=generated_tokens,
            img_tokens=img_output.generated_tokens,
            sequences=seq,
            text_seq=text_sequence,
            seq_img_mask=seq_img_mask,
            gen_img=gen_img,
        )
        
        return output
                         
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.enable_gradient_checkpointing = True
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs
        print("Gradient checkpointing for llama backbone is enabled.")
        
    def text_only_forward(self, input_ids, attention_mask, position_ids):
        """
        Args:
            input_ids (torch.LongTensor): [b, T]
            attention_mask (torch.BoolTensor): [b, T]
            position_ids (torch.LongTensor): [b, T]
        Output:
            output (AttrDict): {
                text_logits (torch.FloatTensor): [b, T, V]
                logits (torch.FloatTensor): [b, T, V]
            }
        """
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        outputs = self.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = outputs.last_hidden_state
        text_logits = self.language_model.lm_head(hidden_states)
        
        output = AttrDict(
            text_logits=text_logits,
            logits=text_logits
        )
        return output
    
    def forward(self, 
                input_ids, 
                input_img_mask, 
                attention_mask, 
                position_ids, 
                cfg_weight=5.0,
                detach_uncond=False,
                use_cache=False,
                bos_token_id=100000,
                pad_token_id=100002,
                image_start_token_id=100003
               ):
        """
        Args:
            input_ids (torch.LongTensor): [b, T]
            input_img_mask (torch.BoolTensor): [b, T]
            attention_mask (torch.BoolTensor): [b, T]
            position_ids (torch.LongTensor): [b, T]
            cfg_weight (float): weight for the conditional generation
            detach_uncond (bool): whether to detach the unconditional generation logits
            use_cache (bool): whether to use cache for the transformer model
        Output:
            output (AttrDict): {
                text_logits (torch.FloatTensor): [b, T, V]
                img_logits (torch.FloatTensor): [b, T, V]
                logits (torch.FloatTensor): [b, T, V]
            }
        """
        if input_img_mask is None:
            return self.text_only_forward(input_ids, attention_mask, position_ids)
        
        parallel_size = input_ids.shape[0]
        if cfg_weight == 1.0:
            tokens = input_ids
            duplicated_img_mask = input_img_mask
            duplicated_attn_mask = attention_mask
            duplicated_position_ids = position_ids
            duplicated_parallel_size = parallel_size
        else:
            tokens = torch.zeros((parallel_size*2, input_ids.shape[1]), dtype=torch.int).cuda()
            duplicated_img_mask = torch.zeros((parallel_size*2, input_img_mask.shape[1]), dtype=torch.bool).cuda()
            duplicated_attn_mask = torch.zeros((parallel_size*2, attention_mask.shape[1]), dtype=torch.bool).cuda()
            duplicated_position_ids = torch.zeros((parallel_size*2, position_ids.shape[1]), dtype=torch.int).cuda()
            duplicated_parallel_size = parallel_size*2
            
            for i in range(parallel_size*2):
                tokens[i, :] = input_ids[i//2]
                duplicated_attn_mask[i, :] = attention_mask[i//2]
                duplicated_img_mask[i, :] = input_img_mask[i//2]
                duplicated_position_ids[i, :] = position_ids[i//2]
                if i % 2 != 0:
                    tokens[i, ~duplicated_img_mask[i]] = torch.where((tokens[i, ~duplicated_img_mask[i]] != bos_token_id) & (tokens[i, ~duplicated_img_mask[i]] != image_start_token_id), pad_token_id, tokens[i, ~duplicated_img_mask[i]])
                
        after_forward_duplicated_img_mask = torch.cat([duplicated_img_mask, torch.ones(duplicated_parallel_size, 1, device=duplicated_img_mask.device, dtype=torch.bool)], dim=-1)[:, 1:]
        after_forward_input_img_mask = torch.cat([input_img_mask, torch.ones(parallel_size, 1, device=input_img_mask.device, dtype=torch.bool)], dim=-1)[:, 1:]
        
        text_embeds = self.language_model.get_input_embeddings()(tokens[~duplicated_img_mask])
        text_embeds = text_embeds.reshape(duplicated_parallel_size, -1, text_embeds.shape[-1])

        inputs_embeds = torch.zeros((duplicated_parallel_size, input_ids.shape[1], text_embeds.shape[-1]), dtype=text_embeds.dtype).cuda()
        
        for i in range(duplicated_parallel_size):
            inputs_embeds[i, duplicated_img_mask[i], :] = self.prepare_gen_img_embeds(tokens[i, duplicated_img_mask[i]].view(-1))
            inputs_embeds[i, ~duplicated_img_mask[i], :] = text_embeds[i]
        
        if not self.enable_gradient_checkpointing:
            outputs = self.language_model.model(inputs_embeds=inputs_embeds, attention_mask=duplicated_attn_mask, position_ids=duplicated_position_ids, use_cache=use_cache)
        else:
            fn = lambda inputs_embeds, attention_mask, position_ids: self.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, use_cache=use_cache)
            outputs = checkpoint_wrapper(fn, self.gradient_checkpointing_kwargs)(inputs_embeds, duplicated_attn_mask, duplicated_position_ids)
            
        hidden_states = outputs.last_hidden_state
        
        text_hidden_states = hidden_states[~after_forward_duplicated_img_mask]
        img_hidden_states = hidden_states[after_forward_duplicated_img_mask]
        
        text_logits = self.language_model.lm_head(text_hidden_states)
        img_logits = self.gen_head(img_hidden_states)
        
        if cfg_weight == 1.0:
            pass
        else:
            logit_cond = img_logits[0::2, :]
            logit_uncond = img_logits[1::2, :]
            if detach_uncond:
                logit_uncond = logit_uncond.detach()
                img_logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            else:
                img_logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            text_logits = text_logits[0::2]
        logits = torch.zeros((parallel_size, input_ids.shape[1], text_logits.shape[-1]), dtype=text_logits.dtype, device=img_logits.device)
        
        # pad img logits
        dim_to_pad = logits.shape[-1] - img_logits.shape[-1]
        img_logits = torch.cat([img_logits, torch.ones(img_logits.shape[0], dim_to_pad, device=img_logits.device, dtype=img_logits.dtype)*(-1e9)], dim=-1)
        
        logits[~after_forward_input_img_mask] = text_logits
        logits[after_forward_input_img_mask] = img_logits
        
        output = AttrDict(
            text_logits=text_logits,
            img_logits=img_logits,
            logits=logits
        )
        return output
    
    def encode_img_gen(self, pixel_values):
        """
        Args:
            pixel_values (torch.FloatTensor): [b, 3, h, w]
        Output:
            img_embeds (torch.FloatTensor): [b, n_image_tokens, D]
        """
        b = pixel_values.shape[0]
        images = rearrange(pixel_values, "b c h w -> (b) c h w")
        _, _, info = self.gen_vision_model.encode(images)
        code = info[2]
        return code
        


AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
