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

        image_embeds = image_embeds.view(B, num_img, num_img_tokens, D)

        merged_list = []

        for b in range(B):
            pieces = []
            cursor = 0

            for img_idx in range(num_img):
                start = all_image_start_indices[b][img_idx]
                end = start + num_img_tokens

                if start > cursor:
                    pieces.append(text_embeds[b, cursor:start])

                pieces.append(image_embeds[b, img_idx])

                cursor = end

            if cursor < T:
                pieces.append(text_embeds[b, cursor:])

            merged_seq = torch.cat(pieces, dim=0)
            merged_list.append(merged_seq)

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
        """Remove padding and concatenate embeddings (vectorized, no Python loops).

        Assumptions:
          - input is left-padded:  attention_mask = [0,...,0, 1,...,1]
          - output is right-padded: output_mask   = [1,...,1, 0,...,0]
        """
        B, T_in, D = input_embeds.shape
        T_out = output_embeds.shape[1]
        device = input_embeds.device

        assert torch.all((attention_mask.cumsum(1) > 0) <= attention_mask), "input mask not left-padded contiguous"
        assert torch.all((output_mask.cumsum(1) == output_mask.cumsum(1).clamp(max=output_mask.sum(1, keepdim=True))) ), "output mask not right-padded contiguous"

        # valid lengths per sample
        valid_input_len = attention_mask.sum(dim=1)   # (B,)  left-padded → sum = valid count
        valid_output_len = output_mask.sum(dim=1)      # (B,)  right-padded → sum = valid count
        total_len = valid_input_len + valid_output_len  # (B,)
        output_starts: List[int] = valid_input_len.tolist()

        max_total_len = int(total_len.max().item())

        if max_total_len == 0:
            concat_embeds_padded = input_embeds.new_zeros((B, 1, D))
            concat_mask_padded = attention_mask.new_zeros((B, 1))
            position_ids = torch.zeros((B, 1), dtype=torch.long, device=device)
            return concat_embeds_padded, concat_mask_padded, position_ids, output_starts

        # position indices in the output tensor: (1, max_total_len)
        pos_idx = torch.arange(max_total_len, device=device).unsqueeze(0)

        # boolean masks for input / output regions
        input_region  = pos_idx < valid_input_len.unsqueeze(1)   # (B, max_total_len)
        output_region = (pos_idx >= valid_input_len.unsqueeze(1)) & (pos_idx < total_len.unsqueeze(1))

        # gather input embeds: left-padded → position p maps to T_in - valid_input_len[i] + p
        input_src_idx = (T_in - valid_input_len.unsqueeze(1) + pos_idx).clamp(0, T_in - 1)  # (B, max_total_len)
        input_gathered = torch.gather(
            input_embeds, dim=1,
            index=input_src_idx.unsqueeze(-1).expand(-1, -1, D),
        )  # (B, max_total_len, D)

        # gather output embeds: right-padded → position p maps to p - valid_input_len[i]
        output_src_idx = (pos_idx - valid_input_len.unsqueeze(1)).clamp(0, T_out - 1)  # (B, max_total_len)
        output_gathered = torch.gather(
            output_embeds, dim=1,
            index=output_src_idx.unsqueeze(-1).expand(-1, -1, D),
        )  # (B, max_total_len, D)

        # assemble: input region → input_gathered, output region → output_gathered, else → 0
        zeros = input_embeds.new_zeros((B, max_total_len, D))
        concat_embeds_padded = torch.where(
            input_region.unsqueeze(-1), input_gathered,
            torch.where(output_region.unsqueeze(-1), output_gathered, zeros),
        )  # (B, max_total_len, D)

        concat_mask_padded = (pos_idx < total_len.unsqueeze(1)).to(attention_mask.dtype)  # (B, max_total_len)

        position_ids = (concat_mask_padded.cumsum(dim=-1) - 1).clamp(min=0).to(torch.long)
        position_ids = position_ids.masked_fill(concat_mask_padded == 0, 0)

        return concat_embeds_padded, concat_mask_padded, position_ids, output_starts

    def _forward_with_split_heads(self, hidden_states, output_starts):
        """
        Shared logic for Task 1 and Task 3:
        Split hidden states into text/image regions, apply lm_head and gen_head
        respectively, apply CFG on image logits only, and reassemble.

        Args:
            hidden_states: [B*2, T, D] from language model (cond/uncond interleaved)
            output_starts: List[int] of length B*2, where output (image) tokens begin

        Returns:
            logits: [B, T, V_text] with gen_head logits at image positions (padded to V_text)
            cond_output_starts: List[int] of length B (original batch)
        """
        B2 = hidden_states.shape[0]  # B*2 (cond + uncond interleaved)
        B = B2 // 2                  # original batch size
        T = hidden_states.shape[1]   # total sequence length

        # Step 1: Build image position mask (with next-token prediction shift)
        # Logit at position t predicts token t+1, so gen_head starts at output_start - 1
        img_mask = torch.zeros((B2, T), dtype=torch.bool, device=hidden_states.device)
        for i in range(B2):
            start = max(output_starts[i] - 1, 0)
            img_mask[i, start:] = True

        # Step 2: Apply different heads to text vs image positions
        text_logits = self.language_model.lm_head(hidden_states[~img_mask])
        img_logits = self.gen_head(hidden_states[img_mask])

        # Step 3: Apply CFG on image logits only
        img_per_sample = img_mask.sum(dim=1)
        cond_imgs, uncond_imgs = [], []
        offset = 0
        for i in range(B2):
            n = img_per_sample[i].item()
            chunk = img_logits[offset:offset + n]
            (cond_imgs if i % 2 == 0 else uncond_imgs).append(chunk)
            offset += n

        cfg_imgs = torch.cat([
            u[:min(len(c), len(u))] + self.guidance_scale * (c[:min(len(c), len(u))] - u[:min(len(c), len(u))])
            for c, u in zip(cond_imgs, uncond_imgs)
        ], dim=0)

        # Step 4: Extract text logits from cond samples only (discard uncond text)
        txt_per_sample = (~img_mask).sum(dim=1)
        cond_txts = []
        t_offset = 0
        for i in range(B2):
            n = txt_per_sample[i].item()
            chunk = text_logits[t_offset:t_offset + n]
            if i % 2 == 0:
                cond_txts.append(chunk)
            t_offset += n
        cond_text = torch.cat(cond_txts, dim=0) if cond_txts else text_logits.new_zeros((0, text_logits.shape[-1]))

        # Step 5: Reassemble logits into [B, T, V_text]
        cond_output_starts = [output_starts[i] for i in range(0, B2, 2)]

        orig_mask = torch.zeros((B, T), dtype=torch.bool, device=hidden_states.device)
        for i in range(B):
            orig_mask[i, max(cond_output_starts[i] - 1, 0):] = True

        V_text = cond_text.shape[-1] if cond_text.shape[0] > 0 else self.language_model.config.vocab_size
        V_img = cfg_imgs.shape[-1] if cfg_imgs.shape[0] > 0 else 1

        # Pad image logits to text vocab size (fill extra dims with -inf)
        pad_dim = V_text - V_img
        if pad_dim > 0:
            cfg_imgs = torch.cat([
                cfg_imgs,
                cfg_imgs.new_full((cfg_imgs.shape[0], pad_dim), -1e9)
            ], dim=-1)
        elif pad_dim < 0:
            cfg_imgs = cfg_imgs[:, :V_text]

        logits = torch.zeros((B, T, V_text), dtype=hidden_states.dtype, device=hidden_states.device)
        logits[~orig_mask] = cond_text
        logits[orig_mask] = cfg_imgs

        return logits, cond_output_starts

    def _process_task1_data(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Process Task 1 (Image Generation) data with CFG logic applied."""
        # 1. Load original data
        task1_input_ids = batch["task1_input_ids"].long()
        task1_attention_mask = batch["task1_attention_mask"]

        # 2. Create conditional embeddings
        task1_input_embeds = self.language_model.get_input_embeddings()(task1_input_ids)

        # 3. Create unconditional embeddings (mask user prompt region with pad embedding)
        uncond_input_embeds = task1_input_embeds.clone()
        pad_id = torch.tensor(self.processor.pad_id, device=task1_input_ids.device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pad_embed = self.language_model.get_input_embeddings()(pad_id).squeeze()

        start_marker = torch.tensor([100601], device=task1_input_ids.device)  # <|User|>
        end_marker = torch.tensor([100602], device=task1_input_ids.device)    # <|Assistant|>

        def find_sequence(inp_id, marker):
            len_needle = marker.shape[0]
            for i in range(inp_id.shape[0] - len_needle + 1):
                if torch.equal(inp_id[i:i + len_needle], marker):
                    return i
            return -1

        for i, row in enumerate(task1_input_ids):
            start_pos = find_sequence(row, start_marker)
            end_pos = find_sequence(row, end_marker)

            if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
                uncond_input_embeds[i, start_pos: end_pos + 2] = pad_embed

        # 4. Batch interleaving (B -> B*2): [Cond, Uncond, Cond, Uncond, ...]
        batch_size, seq_len, embed_dim = task1_input_embeds.shape

        final_input_embeds = torch.stack([task1_input_embeds, uncond_input_embeds], dim=1).reshape(batch_size * 2, seq_len, embed_dim)
        final_attention_mask = task1_attention_mask.repeat_interleave(2, dim=0)

        # 5. Output (generated image tokens) processing, duplicated for cond/uncond
        task1_image_ids = batch["task1_gen_img_tokens"].long()
        task1_gen_img_embeds = self.gen_aligner(self.gen_embed(task1_image_ids))

        final_gen_img_embeds = task1_gen_img_embeds.repeat_interleave(2, dim=0)

        task1_response_mask = batch["task1_response_mask"]
        final_response_mask = task1_response_mask.repeat_interleave(2, dim=0)

        # 6. Remove padding and concatenate into final sequence
        concat_embeds, concat_mask, position_ids, output_starts = self._remove_padding_and_concat_with_embeds(
            final_input_embeds, final_attention_mask, final_gen_img_embeds, final_response_mask,
        )

        return concat_embeds, concat_mask, position_ids, output_starts

    def _process_task2_data(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Process Task 2 (Feedback Generation) data."""
        task2_input_ids = batch["task2_input_ids"]
        task2_attention_mask = batch["task2_attention_mask"]
        gen_imgs_pixel_values = batch["task1_gen_imgs_pixel_values"]

        task2_text_embeds = self.language_model.get_input_embeddings()(task2_input_ids)
        with torch.no_grad():
            vision_features = self.vision_model(gen_imgs_pixel_values)
        task2_image_embeds = self.aligner(vision_features)

        # Find image placeholder token positions
        pos_list = []
        invalid_mask = []
        for i, ids in enumerate(task2_input_ids):
            pos_tensor = (ids == self.processor.image_id).nonzero(as_tuple=False)
            if pos_tensor.numel() == 0:
                print(f"[WARNING] No image placeholder token (id={self.processor.image_id}) "
                      f"found in task2_input_ids[{i}]. Excluding from training.")
                pos_list.append([0])
                invalid_mask.append(i)
            else:
                pos = pos_tensor[0].item()
                pos_list.append([pos])
        task2_image_start_indices = pos_list

        if len(invalid_mask) > 0 and "task2_response_mask" in batch:
            for idx in invalid_mask:
                batch["task2_response_mask"][idx] = 0

        task2_merged_embeds = self._merge_text_and_image_embeds(
            task2_text_embeds, task2_image_embeds, task2_image_start_indices,
        )

        feedback_ids = batch["task2_feedback_ids"]
        task2_feedback_embeds = self.language_model.get_input_embeddings()(feedback_ids)
        task2_response_mask = batch["task2_response_mask"]

        concat_embeds, concat_mask, position_ids, output_starts = self._remove_padding_and_concat_with_embeds(
            task2_merged_embeds, task2_attention_mask, task2_feedback_embeds, task2_response_mask,
        )

        return concat_embeds, concat_mask, position_ids, output_starts

    def _process_task3_data(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Process Task 3 (Regen Image Generation) data with CFG logic applied."""
        # 1. Load original data
        task3_input_ids = batch["task3_input_ids"].long()
        task3_attention_mask = batch["task3_attention_mask"]
        task3_image_ids = batch["task1_gen_img_tokens"].long()

        # 2. Create text embeddings and CFG-masked (unconditional) version
        task3_text_embeds = self.language_model.get_input_embeddings()(task3_input_ids)
        uncond_text_embeds = task3_text_embeds.clone()

        pad_id = torch.tensor(self.processor.pad_id, device=task3_input_ids.device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pad_embed = self.language_model.get_input_embeddings()(pad_id).squeeze()

        start_marker = torch.tensor([100593, 185], device=task3_input_ids.device)
        end_marker = torch.tensor([100602], device=task3_input_ids.device)

        def find_sequence(inp_id, marker):
            len_needle = marker.shape[0]
            for i in range(inp_id.shape[0] - len_needle + 1):
                if torch.equal(inp_id[i:i + len_needle], marker):
                    return i
            return -1

        for i, row in enumerate(task3_input_ids):
            start_pos = find_sequence(row, start_marker)
            end_pos = find_sequence(row, end_marker)
            if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
                uncond_text_embeds[i, start_pos + 1: end_pos + 2] = pad_embed

        # 3. Prepare image embeddings (via gen_embed -> gen_aligner)
        task3_image_embeds = self.gen_aligner(self.gen_embed(task3_image_ids))

        # 4. Find image placeholder positions (shared for cond/uncond)
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

        # 5. Merge text + image embeddings (separately for cond and uncond)
        cond_merged_embeds = self._merge_text_and_image_embeds(
            task3_text_embeds, task3_image_embeds, pos_list,
        )
        uncond_merged_embeds = self._merge_text_and_image_embeds(
            uncond_text_embeds, task3_image_embeds, pos_list,
        )

        # 6. Batch interleaving (B -> B*2)
        batch_size, seq_len, embed_dim = cond_merged_embeds.shape

        final_merged_embeds = torch.stack([cond_merged_embeds, uncond_merged_embeds], dim=1).reshape(batch_size * 2, seq_len, embed_dim)
        final_attention_mask = task3_attention_mask.repeat_interleave(2, dim=0)

        # 7. Output (regenerated image tokens), duplicated for cond/uncond
        task3_regen_image_ids = batch["task3_regen_img_tokens"].long()
        task3_regen_img_embeds = self.gen_aligner(self.gen_embed(task3_regen_image_ids))
        final_regen_img_embeds = task3_regen_img_embeds.repeat_interleave(2, dim=0)

        # 8. Response mask handling
        task3_response_mask = batch["task3_response_mask"].clone()
        if len(invalid_mask) > 0:
            for idx in invalid_mask:
                task3_response_mask[idx] = 0
        final_response_mask = task3_response_mask.repeat_interleave(2, dim=0)

        # 9. Final sequence construction
        concat_embeds, concat_mask, position_ids, output_starts = self._remove_padding_and_concat_with_embeds(
            final_merged_embeds, final_attention_mask, final_regen_img_embeds, final_response_mask,
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
            task_id: Task identifier (1: Image Gen, 2: Feedback Gen, 3: Regen Image Gen)
            batch: Dict containing all task data (processed internally if provided)
            inputs_embeds: Pre-processed input embeddings (optional)
            attention_mask: Attention mask (optional)
            position_ids: Position indices (optional)
            output_starts: Output start positions (optional)

        Returns:
            CausalLMOutputWithPast with logits.
            output_starts stored in self._output_starts for loss computation.
        """
        from transformers.modeling_outputs import CausalLMOutputWithPast

        self.guidance_scale = kwargs.get("cfg_weight", 5.0)
        self.temperature = kwargs.get("temperature", 1.0)
        self.txt_top_k = kwargs.get("txt_top_k", 0)
        self.txt_top_p = kwargs.get("txt_top_p", 1.0)
        self.img_top_k = kwargs.get("img_top_k", 0)
        self.img_top_p = kwargs.get("img_top_p", 1.0)

        # Data processing (if batch provided)
        if batch is not None:
            if task_id == 1:
                inputs_embeds, attention_mask, position_ids, output_starts = self._process_task1_data(batch)
            elif task_id == 2:
                inputs_embeds, attention_mask, position_ids, output_starts = self._process_task2_data(batch)
            elif task_id == 3:
                inputs_embeds, attention_mask, position_ids, output_starts = self._process_task3_data(batch)
            else:
                raise ValueError(f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")

        # Forward pass
        if task_id in (1, 3):
            # Task 1 & 3: Image generation with split heads (gen_head for image, lm_head for text)
            output = self.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            logits, cond_output_starts = self._forward_with_split_heads(
                output.last_hidden_state, output_starts,
            )

            self._output_starts = cond_output_starts

            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=None,
                hidden_states=output.hidden_states if hasattr(output, 'hidden_states') else None,
                attentions=output.attentions if hasattr(output, 'attentions') else None,
            )

        elif task_id == 2:
            # Task 2: Feedback generation (text-only output, use full language model)
            self._output_starts = output_starts

            return self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        else:
            raise ValueError(f"Invalid task_id: {task_id}. Must be 1, 2, or 3.")

    def get_output_starts(self):
        """Retrieve the output_starts from the last forward pass."""
        return getattr(self, '_output_starts', None)


AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)