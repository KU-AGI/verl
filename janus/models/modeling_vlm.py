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
from typing import List, Tuple, Dict, Any
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

    def set_processor(self, processor):
        """Set processor for image token handling"""
        self.processor = processor

    def _merge_text_and_image_embeds(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        all_image_start_indices: List[List[int]],
    ) -> torch.Tensor:
        """Merge text and image embeddings at specified positions (no in-place ops)."""
        B, T, D = text_embeds.shape
        num_img = len(all_image_start_indices[0])
        num_img_tokens = self.processor.num_image_tokens

        # image_embeds: [B * num_img * num_img_tokens, D] 형태라고 가정하고 있었음
        # 위에서 view 로 맞춰놨던 걸 그대로 반영
        image_embeds = image_embeds.view(B, num_img, num_img_tokens, D)  # [B, num_img, num_img_tokens, D]

        merged_list = []

        for b in range(B):
            pieces = []
            cursor = 0

            for img_idx in range(num_img):
                start = all_image_start_indices[b][img_idx]
                end = start + num_img_tokens

                # 텍스트 부분
                if start > cursor:
                    pieces.append(text_embeds[b, cursor:start])

                # 이미지 토큰 부분
                pieces.append(image_embeds[b, img_idx])

                cursor = end

            # 뒤에 남은 텍스트
            if cursor < T:
                pieces.append(text_embeds[b, cursor:])

            merged_seq = torch.cat(pieces, dim=0)  # (L_b, D)
            merged_list.append(merged_seq)

        # 각 시퀀스를 최대 길이에 맞춰 pad
        max_len = max(seq.size(0) for seq in merged_list)
        out = text_embeds.new_zeros((B, max_len, D))

        for b, seq in enumerate(merged_list):
            out[b, : seq.size(0)] = seq

        return out

    def _remove_padding_and_concat_with_embeds(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        output_embeds: torch.Tensor,
        output_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Remove padding and concatenate embeddings (no in-place ops)."""
        device = input_embeds.device
        dtype = input_embeds.dtype

        B, _, D = input_embeds.size()
        sequences = []
        masks = []
        output_starts: List[int] = []
        max_total_len = 0

        for i in range(B):
            # 1) 왼쪽 padding 제거 (input)
            input_valid_mask = attention_mask[i] == 1
            if input_valid_mask.any():
                first_valid = input_valid_mask.nonzero(as_tuple=False)[0].item()
                valid_input_embeds = input_embeds[i, first_valid:]        # (L_in, D)
                valid_input_mask = attention_mask[i, first_valid:]        # (L_in,)
            else:
                valid_input_embeds = input_embeds.new_zeros((0, D))
                valid_input_mask = attention_mask.new_zeros((0,))

            # 2) 오른쪽 padding 제거 (output)
            output_valid_mask = output_mask[i] == 1
            if output_valid_mask.any():
                last_valid = output_valid_mask.nonzero(as_tuple=False)[-1].item()
                valid_output_embeds = output_embeds[i, : last_valid + 1]  # (L_out, D)
                valid_output_mask = output_mask[i, : last_valid + 1]      # (L_out,)
            else:
                valid_output_embeds = output_embeds.new_zeros((0, D))
                valid_output_mask = output_mask.new_zeros((0,))

            # 3) concat
            concat_embeds = torch.cat([valid_input_embeds, valid_output_embeds], dim=0)  # (L_i, D)
            concat_mask = torch.cat([valid_input_mask, valid_output_mask], dim=0)        # (L_i,)

            sequences.append(concat_embeds)
            masks.append(concat_mask)
            output_starts.append(valid_input_embeds.size(0))
            max_total_len = max(max_total_len, concat_embeds.size(0))

        if max_total_len == 0:
            # 극단적으로 다 비어 있는 경우
            concat_embeds_padded = input_embeds.new_zeros((B, 1, D))
            concat_mask_padded = attention_mask.new_zeros((B, 1))
            position_ids = torch.zeros((B, 1), dtype=torch.long, device=device)
            return concat_embeds_padded, concat_mask_padded, position_ids, output_starts

        # 4) pad 도 in-place 없이 cat 로만 만들기
        padded_embeds = []
        padded_masks = []

        for embeds, mask in zip(sequences, masks):
            L = embeds.size(0)
            pad_len = max_total_len - L
            if pad_len > 0:
                pad_embeds = embeds.new_zeros((pad_len, D))
                pad_mask = mask.new_zeros((pad_len,))
                embeds = torch.cat([embeds, pad_embeds], dim=0)   # (max_total_len, D)
                mask = torch.cat([mask, pad_mask], dim=0)         # (max_total_len,)
            padded_embeds.append(embeds)
            padded_masks.append(mask)

        concat_embeds_padded = torch.stack(padded_embeds, dim=0)  # (B, max_total_len, D)
        concat_mask_padded = torch.stack(padded_masks, dim=0)     # (B, max_total_len)

        position_ids = (concat_mask_padded.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)
        position_ids = position_ids.masked_fill(concat_mask_padded == 0, 0)

        return concat_embeds_padded, concat_mask_padded, position_ids, output_starts

    def _process_task1_data(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Process Task 1 (Image Generation) data with CFG logic applied"""
        # 1. 원본 데이터 로드
        task1_input_ids = batch["task1_input_ids"].long() # [B, L_in]
        task1_attention_mask = batch["task1_attention_mask"] # [B, L_in]
        
        # 2. Conditional Embeddings 생성
        task1_input_embeds = self.language_model.get_input_embeddings()(task1_input_ids) # [B, L_in, D]
        
        # 3. Unconditional Embeddings 생성 (Masking 로직 적용)
        uncond_input_embeds = task1_input_embeds.clone()
        pad_id = torch.tensor(self.processor.pad_id, device=task1_input_ids.device)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # [주의] 할당 시 차원 불일치 방지를 위해 squeeze() 처리
            pad_embed = self.language_model.get_input_embeddings()(pad_id).squeeze() # [D]
            
        start_marker = torch.tensor([100601], device=task1_input_ids.device) # <|User|>
        end_marker = torch.tensor([100602], device=task1_input_ids.device)   # <|Assistant|>

        def find_sequence(inp_id, marker):
            len_needle = marker.shape[0]
            for i in range(inp_id.shape[0] - len_needle + 1):
                if torch.equal(inp_id[i:i+len_needle], marker):
                    return i
            return -1

        for i, row in enumerate(task1_input_ids):
            start_pos = find_sequence(row, start_marker)
            end_pos = find_sequence(row, end_marker)
            
            if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
                # User 프롬프트 구간을 pad_embed로 치환 (Uncond)
                uncond_input_embeds[i, start_pos : end_pos + 2] = pad_embed

        # 4. Batch Interleaving (B -> B*2)
        # Cond와 Uncond를 [C, U, C, U...] 순서로 섞습니다.
        batch_size, seq_len, embed_dim = task1_input_embeds.shape
        
        final_input_embeds = torch.zeros((batch_size * 2, seq_len, embed_dim), 
                                        dtype=task1_input_embeds.dtype, device=task1_input_embeds.device)
        final_input_embeds[0::2] = task1_input_embeds
        final_input_embeds[1::2] = uncond_input_embeds
        
        final_attention_mask = task1_attention_mask.repeat_interleave(2, dim=0) # [B*2, L_in]

        # 5. Output (Generated Image) 데이터 처리 및 2배 확장
        # 생성된 이미지는 Cond/Uncond 패스 모두에 동일하게 대응되도록 2배 확장합니다.
        task1_image_ids = batch["task1_gen_img_tokens"].long() # [B, L_out]
        task1_gen_img_embeds = self.gen_aligner(self.gen_embed(task1_image_ids)) # [B, L_out, D]
        
        # Interleave Output Embeds
        final_gen_img_embeds = task1_gen_img_embeds.repeat_interleave(2, dim=0) # [B*2, L_out, D]
        
        # Response mask 확장
        task1_response_mask = batch["task1_response_mask"] # [B, L_out]
        final_response_mask = task1_response_mask.repeat_interleave(2, dim=0) # [B*2, L_out]

        # 6. Padding 제거 및 Concatenate (최종 시퀀스 구성)
        # 
        concat_embeds, concat_mask, position_ids, output_starts = self._remove_padding_and_concat_with_embeds(
            final_input_embeds, final_attention_mask, final_gen_img_embeds, final_response_mask
        )

        return concat_embeds, concat_mask, position_ids, output_starts

    def _process_task2_data(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Process Task 2 (Feedback Generation) data"""
        # Input processing
        task2_input_ids = batch["task2_input_ids"]
        task2_attention_mask = batch["task2_attention_mask"]
        gen_imgs_pixel_values = batch["task1_gen_imgs_pixel_values"]

        task2_text_embeds = self.language_model.get_input_embeddings()(task2_input_ids)
        # Ensure vision_model encode is done with torch.no_grad() since it's frozen
        with torch.no_grad():
            vision_features = self.vision_model(gen_imgs_pixel_values)
        task2_image_embeds = self.aligner(vision_features)

        # Find image token positions
        pos_list = []
        invalid_mask = []
        for i, ids in enumerate(task2_input_ids):
            pos_tensor = (ids == self.processor.image_id).nonzero(as_tuple=False)
            if pos_tensor.numel() == 0:
                # No image placeholder found, mark this sample as invalid
                print(f"[WARNING] No image placeholder token (id={self.processor.image_id}) found in task2_input_ids[{i}]. Excluding from training.")
                pos_list.append([0])  # Use dummy position
                invalid_mask.append(i)
            else:
                pos = pos_tensor[0].item()
                pos_list.append([pos])
        task2_image_start_indices = pos_list

        # Zero out response_mask for invalid samples to exclude from loss calculation
        if len(invalid_mask) > 0 and "task2_response_mask" in batch:
            for idx in invalid_mask:
                batch["task2_response_mask"][idx] = 0

        task2_merged_embeds = self._merge_text_and_image_embeds(
            task2_text_embeds, task2_image_embeds, task2_image_start_indices
        )

        # Output processing
        feedback_ids = batch["task2_feedback_ids"]
        task2_feedback_embeds = self.language_model.get_input_embeddings()(feedback_ids)
        task2_response_mask = batch["task2_response_mask"]

        # Remove padding and concatenate
        concat_embeds, concat_mask, position_ids, output_starts = self._remove_padding_and_concat_with_embeds(
            task2_merged_embeds, task2_attention_mask, task2_feedback_embeds, task2_response_mask
        )

        return concat_embeds, concat_mask, position_ids, output_starts

    def _process_task3_data(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Process Task 3 (Regen Image Generation) data with CFG logic applied"""
        # 1. 원본 데이터 로드 (정수형 캐스팅 필수)
        task3_input_ids = batch["task3_input_ids"].long()
        task3_attention_mask = batch["task3_attention_mask"]
        task3_image_ids = batch["task1_gen_img_tokens"].long() # 이전 단계에서 생성된 이미지 토큰

        # 2. 텍스트 임베딩 및 CFG 마스킹 (Uncond 생성)
        task3_text_embeds = self.language_model.get_input_embeddings()(task3_input_ids)
        uncond_text_embeds = task3_text_embeds.clone()

        pad_id = torch.tensor(self.processor.pad_id, device=task3_input_ids.device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pad_embed = self.language_model.get_input_embeddings()(pad_id).squeeze() # [D]

        # Task 3용 마커: <end_of_image>\n 부터 <|Assistant|> 직전까지 마스킹
        start_marker = torch.tensor([100593, 185], device=task3_input_ids.device) 
        end_marker = torch.tensor([100602], device=task3_input_ids.device)

        def find_sequence(inp_id, marker):
            len_needle = marker.shape[0]
            for i in range(inp_id.shape[0] - len_needle + 1):
                if torch.equal(inp_id[i:i+len_needle], marker):
                    return i
            return -1

        for i, row in enumerate(task3_input_ids):
            start_pos = find_sequence(row, start_marker)
            end_pos = find_sequence(row, end_marker)
            if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
                # 피드백 텍스트 구간을 마스킹
                uncond_text_embeds[i, start_pos + 1 : end_pos + 2] = pad_embed

        # 3. 이미지 임베딩 준비
        task3_image_embeds = self.gen_aligner(self.gen_embed(task3_image_ids))

        # 4. 이미지 토큰 위치 찾기 (Cond/Uncond 공통 사용)
        pos_list = []
        invalid_mask = []
        for i, ids in enumerate(task3_input_ids):
            pos_tensor = (ids == self.processor.image_id).nonzero(as_tuple=False)
            if pos_tensor.numel() == 0:
                invalid_mask.append(i)
                pos_list.append([0])
            else:
                pos = pos_tensor[0].item()
                pos_list.append([pos])
        
        # 5. Merge (Cond/Uncond 각각 수행)
        cond_merged_embeds = self._merge_text_and_image_embeds(
            task3_text_embeds, task3_image_embeds, pos_list
        )
        uncond_merged_embeds = self._merge_text_and_image_embeds(
            uncond_text_embeds, task3_image_embeds, pos_list
        )

        # 6. Batch Interleaving (B -> B*2)
        batch_size, seq_len, embed_dim = cond_merged_embeds.shape
        final_merged_embeds = torch.zeros((batch_size * 2, seq_len, embed_dim), 
                                         dtype=cond_merged_embeds.dtype, device=cond_merged_embeds.device)
        final_merged_embeds[0::2] = cond_merged_embeds
        final_merged_embeds[1::2] = uncond_merged_embeds
        
        final_attention_mask = task3_attention_mask.repeat_interleave(2, dim=0)

        # 7. Output (Regenerated Image) 데이터 처리 및 2배 확장
        task3_regen_image_ids = batch["task3_regen_img_tokens"].long()
        task3_regen_img_embeds = self.gen_aligner(self.gen_embed(task3_regen_image_ids))
        final_regen_img_embeds = task3_regen_img_embeds.repeat_interleave(2, dim=0)

        # 8. Response Mask 처리
        task3_response_mask = batch["task3_response_mask"].clone()
        if len(invalid_mask) > 0:
            for idx in invalid_mask:
                task3_response_mask[idx] = 0
        final_response_mask = task3_response_mask.repeat_interleave(2, dim=0)

        # 9. 최종 시퀀스 구성
        concat_embeds, concat_mask, position_ids, output_starts = self._remove_padding_and_concat_with_embeds(
            final_merged_embeds, final_attention_mask, final_regen_img_embeds, final_response_mask
        )

        return concat_embeds, concat_mask, position_ids, output_starts

    def forward(
        self,
        task_id: int = 1,
        batch: Dict[str, Any] = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        output_starts: List[int] = None,
        **kwargs,
    ):
        """
        Unified forward pass for multi-task model.

        Args:
            task_id: Task identifier (1: Image Generation, 2: Feedback Generation, 3: Regen Image Generation)
            batch: Dict containing all task data (if provided, will process data internally)
            inputs_embeds: Pre-processed input embeddings (optional, if batch is not provided)
            attention_mask: Attention mask (optional, if batch is not provided)
            position_ids: Position indices (optional, if batch is not provided)
            output_starts: Output start positions (optional, if batch is not provided)

        Returns:
            Model output with logits (output_starts stored in self._output_starts)
        """
        from transformers.modeling_outputs import CausalLMOutputWithPast

        self.guidance_scale = 5

        # If batch is provided, process data internally
        if batch is not None:
            if task_id == 1:
                inputs_embeds, attention_mask, position_ids, output_starts = self._process_task1_data(batch)
                # Store output_starts in instance variable for retrieval
                self._output_starts = output_starts[0::2] if output_starts is not None else None
            elif task_id == 2:
                inputs_embeds, attention_mask, position_ids, output_starts = self._process_task2_data(batch)
                # Store output_starts in instance variable for retrieval
                self._output_starts = output_starts
            elif task_id == 3:
                inputs_embeds, attention_mask, position_ids, output_starts = self._process_task3_data(batch)
                # Store output_starts in instance variable for retrieval
                self._output_starts = output_starts[0::2] if output_starts is not None else None
            else:
                raise ValueError(f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")

        # Forward pass based on task_id
        if task_id == 1:
            # Task 1: Image Generation
            output = self.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            logits = self.gen_head(output.last_hidden_state)

            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + self.guidance_scale * (logit_cond-logit_uncond)

            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=None,
                hidden_states=output.hidden_states if hasattr(output, 'hidden_states') else None,
                attentions=output.attentions if hasattr(output, 'attentions') else None,
            )

        elif task_id == 2:
            # Task 2: Feedback Generation (text output)
            return self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        elif task_id == 3:
            # Task 3: Regen Image Generation
            output = self.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            logits = self.gen_head(output.last_hidden_state)

            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + self.guidance_scale * (logit_cond-logit_uncond)

            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=None,
                hidden_states=output.hidden_states if hasattr(output, 'hidden_states') else None,
                attentions=output.attentions if hasattr(output, 'attentions') else None,
            )
        else:
            raise ValueError(f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")

    def get_output_starts(self):
        """Retrieve the output_starts from the last forward pass"""
        return getattr(self, '_output_starts', None)


AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
