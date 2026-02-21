import logging
import os
import contextlib
from typing import List, Tuple, Dict, Any

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from recipe.image_rl.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

from recipe.image_rl.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from recipe.image_rl.utils import FormattingEvaluator
from verl.utils.adaptive_entropy_coeff import AdaptiveEntropyCoefficient
import torch.distributed as dist

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def extract_output_logits(
    logits: torch.Tensor,
    output_start_positions: List[int],
    output_lengths: List[int]
) -> torch.Tensor:
    """
    Extract output logits for each sample based on their start positions and lengths.

    Args:
        logits: (batch_size, seq_len, vocab_size) - model output logits
        output_start_positions: List[int] - output start position for each sample
        output_lengths: List[int] - actual output length for each sample

    Returns:
        output_logits: (batch_size, max_output_len, vocab_size) - extracted and right-padded output logits
    """
    batch_size, seq_len, vocab_size = logits.size()
    max_output_len = max(output_lengths)

    # Create list to collect extracted logits
    extracted_logits_list = []

    for i, (start_pos, out_len) in enumerate(zip(output_start_positions, output_lengths)):
        if out_len > 0:
            # Extract logits for this sample's output tokens
            # Note: we need logits shifted by 1 position for next token prediction
            logit_start_pos = start_pos - 1
            logit_end_pos = logit_start_pos + out_len
            sample_output_logits = logits[i, logit_start_pos:logit_end_pos]  # (out_len, vocab_size)

            # Pad to max_output_len if needed
            if out_len < max_output_len:
                # Don't set requires_grad=False - let it inherit from logits
                padding = torch.zeros((max_output_len - out_len, vocab_size),
                                    dtype=logits.dtype, device=logits.device)
                sample_output_logits = torch.cat([sample_output_logits, padding], dim=0)
        else:
            # If no output tokens, create zero tensor that can still propagate gradients
            sample_output_logits = torch.zeros((max_output_len, vocab_size),
                                              dtype=logits.dtype, device=logits.device)

        extracted_logits_list.append(sample_output_logits.unsqueeze(0))

    # Stack all samples - this preserves gradient flow
    output_logits = torch.cat(extracted_logits_list, dim=0)

    return output_logits


def compute_image_generation_loss(
    old_log_prob: torch.Tensor,  # (bs, seq_len)
    log_prob: torch.Tensor,      # (bs, seq_len)
    rewards: torch.Tensor,       # (bs,)
    response_mask: torch.Tensor, # (bs, seq_len)
    eta: float = 1.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Image Generation Loss computation.
    """
    # Compute log-ratios over masked tokens
    log_prob_sum = (log_prob * response_mask).sum(dim=1)  # (bs,)
    old_log_prob_sum = (old_log_prob * response_mask).sum(dim=1)  # (bs,)
    log_ratios = log_prob_sum - old_log_prob_sum  # (bs,)

    scaled_rewards = eta * (rewards)
    loss_vec = (log_ratios - scaled_rewards) ** 2  # (bs,)

    if loss_agg_mode == "token-mean":
        sample_mask = response_mask.any(dim=1).float()  # (bs,)
        loss = verl_F.masked_mean(loss_vec, sample_mask)

    return loss, log_ratios, scaled_rewards


class DataParallelImageGenerationActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker"""

    def __init__(
        self,
        config: ActorConfig,
        processor: None,
        tokenizer: None,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__(config)
        self.processor = processor
        self.tokenizer = tokenizer
        self.formatter = FormattingEvaluator()
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False) # always True, not use this args
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            # entropy_from_logits = verl_F.entropy_from_logits_with_chunking
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking_for_2D # OURS
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        # Adaptive entropy coefficient
        adaptive_cfg = self.config.get('adaptive_entropy_coeff', {})
        if adaptive_cfg.get('enable', False):
            self.use_adaptive_entropy_coeff = True
            
            # 2. 'text'와 'image' 섹션을 가져옵니다 (없을 경우 빈 딕셔너리 기본값)
            text_cfg = adaptive_cfg.get('text', {})
            image_cfg = adaptive_cfg.get('image', {})

            # 3. Text 설정 (typo였던 gettext 수정 및 딕셔너리 접근 방식 적용)
            self.text_adaptive_entropy_coeff = AdaptiveEntropyCoefficient(
                # YAML에 'init_alpha'로 적으셨을 수도 있으니 둘 다 체크하도록 처리했습니다.
                initial_alpha=text_cfg.get('initial_alpha', text_cfg.get('init_alpha', 0.0)),
                target_entropy=text_cfg.get('target_entropy', -1.0),
                lr=text_cfg.get('lr', 1e-3),
                max_coeff=text_cfg.get('max_coeff', 1e-3),
                min_coeff=text_cfg.get('min_coeff', -1e-3),
            )
            
            # 4. Image 설정
            self.img_adaptive_entropy_coeff = AdaptiveEntropyCoefficient(
                initial_alpha=image_cfg.get('initial_alpha', image_cfg.get('init_alpha', 0.0)),
                target_entropy=image_cfg.get('target_entropy', -1.0),
                lr=image_cfg.get('lr', 1e-3),
                max_coeff=image_cfg.get('max_coeff', 1e-3),
                min_coeff=image_cfg.get('min_coeff', -1e-3),
            )
        else:
            self.use_adaptive_entropy_coeff = False

        self.cfg_weight = self.config.get('cfg_weight', 1.0)
        self.detach_uncond = self.config.get('detach_uncond', True)

        # Set processor in the model for unified forward pass
        if hasattr(self.actor_module, 'set_processor'):
            self.actor_module.set_processor(processor)
        elif hasattr(self.actor_module, 'module') and hasattr(self.actor_module.module, 'set_processor'):
            # For FSDP wrapped models
            self.actor_module.module.set_processor(processor)

    def _extract_valid_output_tokens(
        self,
        output_tokens: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract valid output tokens by removing right padding"""
        batch_size = output_tokens.size(0)
        max_valid_len = 0
        valid_tokens_list = []

        # Find valid tokens for each sample
        for i in range(batch_size):
            valid_mask = response_mask[i] == 1
            if valid_mask.any():
                last_valid = valid_mask.nonzero(as_tuple=False)[-1].item()
                valid_tokens = output_tokens[i, :last_valid + 1]
            else:
                valid_tokens = torch.tensor([], dtype=output_tokens.dtype, device=output_tokens.device)

            valid_tokens_list.append(valid_tokens)
            max_valid_len = max(max_valid_len, len(valid_tokens))

        # Right pad to max valid length
        if max_valid_len == 0:
            return torch.zeros((batch_size, 0), dtype=output_tokens.dtype, device=output_tokens.device)

        padded_tokens = torch.zeros((batch_size, max_valid_len),
                                   dtype=output_tokens.dtype, device=output_tokens.device)

        for i, valid_tokens in enumerate(valid_tokens_list):
            if len(valid_tokens) > 0:
                padded_tokens[i, :len(valid_tokens)] = valid_tokens

        return padded_tokens

    def _restore_log_probs_to_original_length(
        self,
        compact_log_probs: torch.Tensor,
        original_response_mask: torch.Tensor,
        pad_value: float = 0.0
    ) -> torch.Tensor:
        """
        Restore compact log_probs back to original response_mask length.
        This function must preserve gradient flow from compact_log_probs to restored_log_probs.

        Args:
            compact_log_probs: (batch_size, compact_len) - log probs from valid tokens only
            original_response_mask: (batch_size, original_len) - original response mask
            pad_value: value to use for padding positions

        Returns:
            restored_log_probs: (batch_size, original_len) - log probs padded to original length
        """
        batch_size, original_len = original_response_mask.size()
        compact_len = compact_log_probs.size(1)

        # Initialize restored tensor with zeros - this will hold our output
        # IMPORTANT: Don't set requires_grad explicitly, let it be inferred from operations
        restored_log_probs = torch.zeros(batch_size, original_len,
                                         dtype=compact_log_probs.dtype,
                                         device=compact_log_probs.device)

        # Process each batch item
        for i in range(batch_size):
            valid_mask = original_response_mask[i] == 1

            if valid_mask.any():
                valid_positions = valid_mask.nonzero(as_tuple=False).squeeze(-1)
                valid_len = len(valid_positions)
                restore_len = min(valid_len, compact_len)

                if restore_len > 0:
                    # Use advanced indexing which preserves gradients
                    # Place compact log probs at valid positions
                    restored_log_probs[i, valid_positions[:restore_len]] = compact_log_probs[i, :restore_len]

        return restored_log_probs

    def _forward_micro_batch(
        self,
        micro_batch: Dict[str, Any],
        temperature: float,
        calculate_entropy: bool = False,
        task_id: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass following modeling_vlm.py forward pattern.
        input_ids를 concat하고 model forward 후 logit slice하는 레퍼런스 방식.

        Returns:
            entropy: (bs, response_length) or None
            log_probs: (bs, response_length)
        """
        # ========== 1. task별 full sequence 구성 ==========
        if task_id == 1:
            responses = micro_batch['task1_gen_img_tokens']
            response_length = responses.size(1)

            input_ids = torch.cat([micro_batch['task1_input_ids'], responses], dim=1)
            attention_mask = torch.cat([
                micro_batch['task1_attention_mask'],
                micro_batch['task1_response_mask']
            ], dim=1)

            # seq_img_mask: prompt 부분은 False, response 부분은 task1_seq_img_mask
            prompt_img_mask = torch.zeros_like(micro_batch['task1_attention_mask'], dtype=torch.bool)
            seq_img_mask = torch.cat([prompt_img_mask, micro_batch['task1_seq_img_mask']], dim=1)

        elif task_id == 2:
            responses = micro_batch['task2_feedback_ids']
            response_length = responses.size(1)

            input_ids = torch.cat([micro_batch['task2_input_ids'], responses], dim=1)
            attention_mask = torch.cat([
                micro_batch['task2_attention_mask'],
                micro_batch['task2_response_mask']
            ], dim=1)

            # task2는 text generation이므로 seq_img_mask 전부 False
            seq_img_mask = torch.zeros_like(attention_mask, dtype=torch.bool)

        elif task_id == 3:
            responses = micro_batch['task3_regen_img_tokens']
            response_length = responses.size(1)

            input_ids = torch.cat([micro_batch['task3_input_ids'], responses], dim=1)
            attention_mask = torch.cat([
                micro_batch['task3_attention_mask'],
                micro_batch['task3_response_mask']
            ], dim=1)

            prompt_img_mask = torch.zeros_like(micro_batch['task3_attention_mask'], dtype=torch.bool)
            response_img_mask = micro_batch.get('task3_seq_img_mask', micro_batch['task3_response_mask'].bool())
            seq_img_mask = torch.cat([prompt_img_mask, response_img_mask], dim=1)
        else:
            raise ValueError(f"Invalid task_id: {task_id}")

        # ========== 2. position_ids 계산 ==========
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        # ========== 3. model forward (modeling_vlm.py forward 직접 호출) ==========
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.actor_module(
                input_ids=input_ids,
                input_img_mask=seq_img_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                image_start_token_id=self.processor.image_start_id,
                cfg_weight=self.cfg_weight,
                detach_uncond=self.detach_uncond,
                use_cache=False,
            )

            logits = output['logits'] if isinstance(output, dict) else output.logits
            logits.div_(temperature)

            # 레퍼런스 패턴: response 부분 logit만 slice
            logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
            logits = torch.clamp(logits, min=-30.0, max=30.0)

            log_probs = logprobs_from_logits(logits, responses)

            entropy = None
            if calculate_entropy:
                entropy = self.compute_entropy_from_logits(logits)  # (bsz, response_length)

        return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids"""
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        task_id = data.batch["task_id"].view(-1)[0].item()

        # Selected batch keys based on task_id
        if task_id == 1:
            select_batch_keys = [
                "task1_input_ids", "task1_attention_mask", "task1_gen_imgs_pixel_values",
                "task1_gen_img_tokens", "task1_response_mask", "task1_seq_img_mask", "task_id"
            ]
            non_tensor_batch_keys = []
        elif task_id == 2:
            select_batch_keys = [
                "task1_input_ids", "task1_attention_mask", "task1_gen_imgs_pixel_values", 
                "task1_gen_img_tokens", "task1_response_mask",
                "task2_input_ids", "task2_attention_mask", "task2_feedback_ids", 
                "task2_response_mask", "task_id"
            ]
            non_tensor_batch_keys = ["task2_feedback_texts"]
        elif task_id == 3:
            select_batch_keys = [
                "task1_input_ids", "task1_attention_mask", "task1_gen_imgs_pixel_values", 
                "task1_gen_img_tokens", "task1_response_mask",
                "task2_input_ids", "task2_attention_mask", "task2_feedback_ids", 
                "task2_response_mask",
                "task3_input_ids", "task3_attention_mask", "task3_regen_imgs_pixel_values", 
                "task3_regen_img_tokens", "task3_response_mask", "task_id"
            ]
            non_tensor_batch_keys = []
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

        data = data.select(batch_keys=select_batch_keys, non_tensor_batch_keys=non_tensor_batch_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, task_id=task_id
                )

            log_probs_lst.append(log_probs)

            if calculate_entropy and entropy is not None:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    def freeze_param(self):
        """Freeze and unfreeze parameters for training"""
        for n, p in self.actor_module.language_model.named_parameters():
            p.requires_grad = True
        self.actor_module.language_model.train()

        for n, p in self.actor_module.gen_embed.named_parameters():
            p.requires_grad = False
        self.actor_module.gen_embed.eval()

        for n, p in self.actor_module.gen_head.named_parameters():
            p.requires_grad = True
        self.actor_module.gen_head.train()

        for n, p in self.actor_module.gen_aligner.named_parameters():
            p.requires_grad = True
        self.actor_module.gen_aligner.train()

        for n, p in self.actor_module.aligner.named_parameters():
            p.requires_grad = True
        self.actor_module.aligner.train()

        for n, p in self.actor_module.vision_model.named_parameters():
            p.requires_grad = False
        self.actor_module.vision_model.eval()

        for n, p in self.actor_module.gen_vision_model.named_parameters():
            p.requires_grad = False
        self.actor_module.gen_vision_model.eval()

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        """Update policy with multi-task support"""
        self.actor_module.train()
        #self.freeze_param()
        loss_agg_mode = self.config.loss_agg_mode
        
        temperature = data.meta_info["temperature"]
        
        # Multi-task configuration
        multi_task_config = self.config.get("multi_task", {})
        enable_multi_task = multi_task_config.get("enable", True)
        task_weights = multi_task_config.get("task_weights", [1.0, 1.0, 1.0])
        task_selection = multi_task_config.get("task_selection", "all")
        
        # Determine which tasks to process
        if enable_multi_task:
            if task_selection == "all":
                task_ids = [1, 2, 3]
            elif task_selection == "weighted_sample":
                import random
                task_ids = random.choices([1, 2, 3], weights=task_weights, k=1)
            else:
                task_ids = [data.batch["task_id"].view(-1)[0].item()]
        else:
            task_ids = [data.batch["task_id"].view(-1)[0].item()]
        
        # ========== Select keys (레퍼런스 패턴) ==========
        select_keys = ['task_id']
        for tid in task_ids:
            select_keys.extend([
                f'task{tid}_input_ids', f'task{tid}_attention_mask',
                f'task{tid}_response_mask', f'task{tid}_old_log_probs',
                f'task{tid}_advantages',
            ])
            if tid == 1:
                select_keys.extend([
                    'task1_gen_imgs_pixel_values', 'task1_gen_img_tokens', 'task1_seq_img_mask'
                ])
            elif tid == 2:
                select_keys.extend(['task2_feedback_ids'])
            elif tid == 3:
                select_keys.extend([
                    'task3_regen_imgs_pixel_values', 'task3_regen_img_tokens'
                ])
            if self.config.use_kl_loss:
                select_keys.append(f'task{tid}_ref_log_prob')

        select_keys = list(set(select_keys))

        non_tensor_select_keys = ['uid']
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()
        if has_multi_modal_inputs:
            non_tensor_select_keys.append('multi_modal_inputs')

        # ========== 레퍼런스 패턴: chunk로 mini-batch 분할 ==========
        num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
        dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(dataloader):
                self.gradient_accumulation = (
                    self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                )
                num_micro_batches = (
                    mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                )
                micro_batches = mini_batch.select(
                    select_keys, non_tensor_select_keys
                ).chunk(num_micro_batches)

                self.actor_optimizer.zero_grad()

                for micro_data in micro_batches:
                    if isinstance(micro_data, DataProto):
                        data_dict = {
                            **micro_data.batch.to(torch.cuda.current_device()),
                            **micro_data.non_tensor_batch
                        }
                    else:
                        data_dict = micro_data.to(torch.cuda.current_device())

                    micro_metrics = {}

                    for task_id in task_ids:
                        responses_key = {
                            1: 'task1_gen_img_tokens',
                            2: 'task2_feedback_ids',
                            3: 'task3_regen_img_tokens',
                        }[task_id]

                        responses = data_dict[responses_key]
                        response_length = responses.size(1)
                        attention_mask = data_dict[f'task{task_id}_attention_mask']
                        response_mask = data_dict[f'task{task_id}_response_mask']
                        old_log_prob = data_dict[f'task{task_id}_old_log_probs']
                        advantages = data_dict[f'task{task_id}_advantages']

                        if task_id == 1 and 'task1_seq_img_mask' in data_dict:
                            seq_img_mask = data_dict['task1_seq_img_mask']
                        else:
                            seq_img_mask = None

                        if self.config.get('ignore_img_start', False) and task_id in [1, 3]:
                            img_start_mask = responses == self.config.image_start_token_id
                            response_mask = response_mask & (~img_start_mask)

                        uids = data_dict.get('uid', None)
                        clip_ratio = self.config.clip_ratio

                        # ========== Forward pass ==========
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data_dict,
                            temperature=temperature,
                            calculate_entropy=True,
                            task_id=task_id,
                        )

                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        rollout_is_weights = data_dict.get("rollout_is_weights", None)
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # ========== Policy loss ==========
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )

                        # ========== Entropy ==========
                        entropy_loss = verl_F.masked_mean(entropy, response_mask)

                        if seq_img_mask is not None:
                            img_entropy_loss = verl_F.masked_mean(
                                entropy, seq_img_mask & response_mask.bool()
                            )
                            text_entropy_loss = verl_F.masked_mean(
                                entropy, (~seq_img_mask) & response_mask.bool()
                            )
                        else:
                            img_entropy_loss = None
                            text_entropy_loss = None

                        # ========== Entropy coeff 적용 ==========
                        if not self.use_adaptive_entropy_coeff:
                            entropy_coeff = self.config.entropy_coeff
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            if img_entropy_loss is not None and text_entropy_loss is not None:
                                text_entropy_coeff = -self.text_adaptive_entropy_coeff.alpha.detach().item()
                                img_entropy_coeff = -self.img_adaptive_entropy_coeff.alpha.detach().item()
                                self.text_adaptive_entropy_coeff.update(entropy=text_entropy_loss.detach())
                                self.img_adaptive_entropy_coeff.update(entropy=img_entropy_loss.detach())
                                policy_loss = (
                                    pg_loss
                                    - text_entropy_loss * text_entropy_coeff
                                    - img_entropy_loss * img_entropy_coeff
                                )
                            else:
                                entropy_coeff = self.config.entropy_coeff
                                policy_loss = pg_loss - entropy_loss * entropy_coeff

                        # ========== KL loss ==========
                        if self.config.use_kl_loss:
                            ref_log_prob = data_dict[f'task{task_id}_ref_log_prob']
                            kld = core_algos.kl_penalty(
                                logprob=log_prob,
                                ref_logprob=ref_log_prob,
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_loss = verl_F.masked_mean(kld, response_mask)
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                        # ========== Scale & backward ==========
                        task_weight = task_weights[task_id - 1] if enable_multi_task else 1.0

                        if self.config.use_dynamic_bsz:
                            loss = policy_loss * task_weight * (
                                response_mask.size(0) / self.config.ppo_mini_batch_size
                            )
                        else:
                            loss = policy_loss * task_weight / self.gradient_accumulation

                        loss = torch.nan_to_num(loss, nan=0.0)
                        loss.backward()

                        # ========== Metrics ==========
                        task_metrics = {
                            f'actor/task{task_id}_entropy_loss': entropy_loss.detach().item(),
                            f'actor/task{task_id}_pg_loss': pg_loss.detach().item(),
                            f'actor/task{task_id}_pg_clipfrac': pg_clipfrac.detach().item(),
                            f'actor/task{task_id}_pg_clipfrac_lower': pg_clipfrac_lower.detach().item(),
                            f'actor/task{task_id}_ppo_kl': ppo_kl.detach().item(),
                            f'actor/task{task_id}_weight': task_weight,
                            f'actor/task{task_id}_loss': loss.detach().item(),
                        }

                        if text_entropy_loss is not None:
                            task_metrics[f'actor/task{task_id}_text_entropy_loss'] = (
                                text_entropy_loss.detach().item()
                            )
                        if img_entropy_loss is not None:
                            task_metrics[f'actor/task{task_id}_img_entropy_loss'] = (
                                img_entropy_loss.detach().item()
                            )
                        if self.use_adaptive_entropy_coeff and text_entropy_loss is not None:
                            task_metrics[f'actor/task{task_id}_text_entropy_coeff'] = text_entropy_coeff
                            task_metrics[f'actor/task{task_id}_img_entropy_coeff'] = img_entropy_coeff
                        if self.config.use_kl_loss:
                            task_metrics[f'actor/task{task_id}_kl_loss'] = kl_loss.detach().item()
                            task_metrics[f'actor/task{task_id}_kl_coef'] = self.config.kl_loss_coef

                        # ========== Pos/Neg log prob ==========
                        lp = log_prob.detach()
                        adv = advantages
                        mask = response_mask

                        seq_adv_sum = adv[:, 0].view(-1, 1)
                        is_pos_seq = (seq_adv_sum > 0)
                        is_neg_seq = (seq_adv_sum < 0)

                        pos_mask = is_pos_seq & mask.bool()
                        neg_mask = is_neg_seq & mask.bool()

                        task_metrics[f'actor/task{task_id}_pos_log_prob'] = lp[pos_mask].sum().item()
                        task_metrics[f'actor/task{task_id}_pos_log_prob_cnt'] = pos_mask.sum().item()
                        task_metrics[f'actor/task{task_id}_neg_log_prob'] = lp[neg_mask].sum().item()
                        task_metrics[f'actor/task{task_id}_neg_log_prob_cnt'] = neg_mask.sum().item()

                        micro_metrics.update(task_metrics)

                    # ========== Aggregated metrics across tasks ==========
                    if enable_multi_task and len(task_ids) > 1:
                        aggregated_metrics = {}

                        avg_pg_loss = sum([micro_metrics.get(f"actor/task{tid}_pg_loss", 0.0) for tid in task_ids]) / len(task_ids)
                        avg_pg_clipfrac = sum([micro_metrics.get(f"actor/task{tid}_pg_clipfrac", 0.0) for tid in task_ids]) / len(task_ids)
                        avg_ppo_kl = sum([micro_metrics.get(f"actor/task{tid}_ppo_kl", 0.0) for tid in task_ids]) / len(task_ids)
                        avg_pg_clipfrac_lower = sum([micro_metrics.get(f"actor/task{tid}_pg_clipfrac_lower", 0.0) for tid in task_ids]) / len(task_ids)
                        total_loss = sum([micro_metrics.get(f"actor/task{tid}_loss", 0.0) for tid in task_ids])

                        aggregated_metrics.update({
                            "actor/avg_pg_loss": avg_pg_loss,
                            "actor/avg_pg_clipfrac": avg_pg_clipfrac,
                            "actor/avg_ppo_kl": avg_ppo_kl,
                            "actor/avg_pg_clipfrac_lower": avg_pg_clipfrac_lower,
                            "actor/loss": total_loss,
                        })

                        if any(f"actor/task{tid}_kl_loss" in micro_metrics for tid in task_ids):
                            avg_kl_loss = sum([micro_metrics.get(f"actor/task{tid}_kl_loss", 0.0) for tid in task_ids]) / len(task_ids)
                            aggregated_metrics["actor/avg_kl_loss"] = avg_kl_loss

                        micro_metrics.update(aggregated_metrics)

                    append_to_dict(metrics, micro_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {'actor/grad_norm': grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics