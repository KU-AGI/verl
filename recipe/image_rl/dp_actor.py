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
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

from recipe.image_rl.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from recipe.image_rl.utils import FormattingEvaluator
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
            end_pos = start_pos + out_len
            sample_output_logits = logits[i, start_pos:end_pos]  # (out_len, vocab_size)

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
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)
            else entropy_from_logits
        )
        self.device_name = get_device_name()

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
        Unified forward pass using model's internal processing.
        FSDP will automatically unshard parameters during forward pass.

        Returns:
            entropy: (bs, response_len) or None
            log_probs: (bs, response_len)
        """
        # Get original response mask for length restoration
        original_response_mask = micro_batch[f"task{task_id}_response_mask"]

        # Extract valid output tokens (remove right padding)
        if task_id == 1:
            output_tokens = micro_batch["task1_gen_img_tokens"]
        elif task_id == 2:
            output_tokens = micro_batch["task2_feedback_ids"]
        elif task_id == 3:
            output_tokens = micro_batch["task3_regen_img_tokens"]
        else:
            raise ValueError(f"Invalid task_id: {task_id}")

        valid_output_tokens = self._extract_valid_output_tokens(output_tokens, original_response_mask)

        output_lengths = [original_response_mask[i].sum().item() for i in range(original_response_mask.size(0))]
        local_has_output = 1 if max(output_lengths) > 0 else 0

        # IMPORTANT: Always do forward pass even if no valid output
        # FSDP requires all ranks to participate in forward/backward for synchronization
        output = self.actor_module(
            task_id=task_id,
            batch=micro_batch,
        )

        if local_has_output == 0:
            # This rank has no output, but we still did the forward pass above
            # Keep gradient connection to forward pass for FSDP synchronization
            dummy_scalar = output.logits.flatten()[0] * 0.0
            log_probs = torch.zeros_like(original_response_mask, dtype=output.logits.dtype, device=output.logits.device) + dummy_scalar
            entropy = None
            if calculate_entropy:
                entropy = torch.zeros_like(original_response_mask, dtype=output.logits.dtype, device=output.logits.device) + dummy_scalar
            return entropy, log_probs

        # Extract output_starts from model
        output_starts = self.actor_module.get_output_starts()

        logits = output.logits
        task_logits = extract_output_logits(logits, output_starts, output_lengths)

        # Compute log probabilities
        compact_log_probs = logprobs_from_logits(task_logits, valid_output_tokens)
        log_probs = self._restore_log_probs_to_original_length(compact_log_probs, original_response_mask)

        # Calculate entropy if needed
        entropy = None
        if calculate_entropy:
            if not self.config.entropy_checkpointing:
                entropy = verl_F.entropy_from_logits(task_logits)
            else:
                entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, task_logits)

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

        # batch_keys
        select_batch_keys = [
            "task1_input_ids", "task1_attention_mask", "task1_gen_imgs_pixel_values", "task1_gen_img_tokens", "task1_response_mask",
            "task2_input_ids", "task2_attention_mask", "task2_feedback_ids", "task2_response_mask",
            "task3_input_ids", "task3_attention_mask", "task3_regen_imgs_pixel_values", "task3_regen_img_tokens", "task3_response_mask",
            "task_id"
        ]

        # non_tensor_batch_keys
        non_tensor_batch_keys = ["task2_feedback_texts"]

        data = data.select(batch_keys=select_batch_keys, non_tensor_batch_keys=non_tensor_batch_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        response_masks = []
        
        # Collect all response masks to determine global max length
        for micro_batch in micro_batches:
            micro_batch_device = micro_batch.to(get_device_id())
            response_mask = micro_batch_device.batch[f"task{task_id}_response_mask"]
            response_masks.append(response_mask)
        
        # Find global max response length
        all_response_masks = torch.cat(response_masks, dim=0)
        global_max_len = all_response_masks.size(1)  # Use original response mask length for consistency
        
        # Process each micro batch
        for i, micro_batch in enumerate(micro_batches):
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, task_id=task_id
                )

            need = torch.ones(log_probs.size(0), device=log_probs.device, dtype=torch.bool)

            # Pad log_probs to global_max_len for consistent concatenation
            current_len = log_probs.size(1)
            if current_len < global_max_len:
                padding = torch.zeros(log_probs.size(0), global_max_len - current_len, 
                                      device=log_probs.device, dtype=log_probs.dtype)
                log_probs = torch.cat([log_probs, padding], dim=1)
            elif current_len > global_max_len:
                # Truncate if somehow longer (shouldn't happen but safety check)
                log_probs = log_probs[:, :global_max_len]
            
            # Apply need mask
            log_probs = log_probs * need.unsqueeze(1)
            log_probs_lst.append(log_probs)
            
            if calculate_entropy and entropy is not None:
                # Pad entropy to global_max_len
                current_len = entropy.size(1)
                if current_len < global_max_len:
                    padding = torch.zeros(entropy.size(0), global_max_len - current_len, 
                                          device=entropy.device, dtype=entropy.dtype)
                    entropy = torch.cat([entropy, padding], dim=1)
                elif current_len > global_max_len:
                    entropy = entropy[:, :global_max_len]
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
        """Update policy"""
        # make sure we are in training mode
        self.actor_module.train()
        self.freeze_param()

        temperature = data.meta_info["temperature"]
        task_id = data.batch["task_id"].view(-1)[0].item()

        # batch_keys
        select_batch_keys = [
            "task1_input_ids", "task1_attention_mask", "task1_gen_imgs_pixel_values", "task1_gen_img_tokens", "task1_response_mask",
            "task2_input_ids", "task2_attention_mask", "task2_feedback_ids", "task2_response_mask",
            "task3_input_ids", "task3_attention_mask", "task3_regen_imgs_pixel_values", "task3_regen_img_tokens", "task3_response_mask",
            "task1_old_log_probs", "task2_old_log_probs", "task3_old_log_probs",
            "task1_advantages", "task2_advantages", "task3_advantages",
            "task_id"
        ]
        task_select_batch_keys = [key for key in select_batch_keys if str(task_id) in key]

        if self.config.use_kl_loss:
            task_select_batch_keys.append(f"task{task_id}_ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            task_select_batch_keys.append("rollout_is_weights")

        data = data.select(meta_info_keys=task_select_batch_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch}

                    old_log_prob = model_inputs[f"task{task_id}_old_log_probs"]
                    advantages = model_inputs[f"task{task_id}_advantages"]
                    response_mask = model_inputs[f"task{task_id}_response_mask"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, task_id=task_id
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs[f"task{task_id}_old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout importance sampling weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (all functions return 4 values)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs[f"task{task_id}_ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics[f"actor/task{task_id}_kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics[f"actor/task{task_id}_kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            f"actor/task{task_id}_pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            f"actor/task{task_id}_pg_clipfrac": pg_clipfrac.detach().item(),
                            f"actor/task{task_id}_ppo_kl": ppo_kl.detach().item(),
                            f"actor/task{task_id}_pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()

                mini_batch_metrics = {f"actor/task{task_id}_grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics