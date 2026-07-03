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
from recipe.image_rl.utils import FormattingEvaluatorV2
from verl.utils.adaptive_entropy_coeff import AdaptiveEntropyCoefficient
import torch.distributed as dist

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _response_mask_last_lengths(response_mask: torch.Tensor) -> List[int]:
    """Return last nonzero position + 1 for each response mask row."""
    lengths = []
    for row in response_mask:
        valid = row > 0
        if valid.any():
            lengths.append(int(valid.nonzero(as_tuple=False)[-1].item()) + 1)
        else:
            lengths.append(0)
    return lengths


def _format_logprob_shape_debug(
    *,
    task_id: int,
    logits_seq_len: int,
    output_starts: List[int],
    output_lengths: List[int],
    task_logits: torch.Tensor,
    valid_output_tokens: torch.Tensor,
    output_tokens: torch.Tensor,
    response_mask: torch.Tensor,
    micro_batch: Dict[str, Any],
) -> str:
    response_sums = response_mask.sum(dim=1).detach().cpu().to(torch.long).tolist()
    response_last = _response_mask_last_lengths(response_mask.detach().cpu())
    response_unique = torch.unique(response_mask.detach().cpu()).tolist()

    segment_unique = None
    if task_id == 2 and "task2_segment_mask" in micro_batch:
        segment_unique = torch.unique(micro_batch["task2_segment_mask"].detach().cpu()).tolist()

    row_infos = []
    for i, (start, out_len) in enumerate(zip(output_starts, output_lengths)):
        logit_start = int(start) - 1
        logit_end = logit_start + int(out_len)
        available = max(0, logits_seq_len - max(logit_start, 0))
        row_infos.append(
            {
                "row": i,
                "start": int(start),
                "out_len": int(out_len),
                "logit_start": int(logit_start),
                "logit_end": int(logit_end),
                "available_logits": int(available),
                "mask_sum": int(response_sums[i]) if i < len(response_sums) else None,
                "mask_last": int(response_last[i]) if i < len(response_last) else None,
                "hole": int(response_last[i] - response_sums[i]) if i < len(response_sums) else None,
            }
        )

    return (
        "[LOGPROB SHAPE MISMATCH] "
        f"task_id={task_id} "
        f"logits_seq_len={logits_seq_len} "
        f"task_logits_shape={tuple(task_logits.shape)} "
        f"valid_output_tokens_shape={tuple(valid_output_tokens.shape)} "
        f"output_tokens_shape={tuple(output_tokens.shape)} "
        f"response_mask_shape={tuple(response_mask.shape)} "
        f"response_mask_unique={response_unique} "
        f"segment_mask_unique={segment_unique} "
        f"rows={row_infos}"
    )


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
        self.formatter = FormattingEvaluatorV2()
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

        # Adaptive entropy coefficient (per-task, follows multi_task.task_ids)
        adaptive_cfg = self.config.get('adaptive_entropy_coeff', {})
        if adaptive_cfg.get('enable', False):
            self.use_adaptive_entropy_coeff = True
            multi_task_cfg = self.config.get('multi_task', {})
            task_ids = list(multi_task_cfg.get('task_ids', [1, 2, 3]))
            self.adaptive_entropy_coeffs = {}
            for tid in task_ids:
                task_cfg = adaptive_cfg.get(f'task{tid}', {})
                self.adaptive_entropy_coeffs[tid] = AdaptiveEntropyCoefficient(
                    initial_alpha=task_cfg.get('initial_alpha', 0.0),
                    target_entropy=task_cfg.get('target_entropy', -1.0),
                    lr=task_cfg.get('lr', 1e-3),
                    max_coeff=task_cfg.get('max_coeff', 1e-3),
                    min_coeff=task_cfg.get('min_coeff', -1e-3),
                )
            if torch.distributed.get_rank() == 0:
                print(f"Actor adaptive_entropy_coeff enabled for task_ids={task_ids}: {adaptive_cfg}")
        else:
            self.use_adaptive_entropy_coeff = False

        # Set processor in the model for unified forward pass
        if hasattr(self.actor_module, 'set_processor'):
            self.actor_module.set_processor(processor)
        elif hasattr(self.actor_module, 'module') and hasattr(self.actor_module.module, 'set_processor'):
            # For FSDP wrapped models
            self.actor_module.module.set_processor(processor)

    def adaptive_entropy_state_dict(self) -> Dict[str, Any]:
        if not getattr(self, "use_adaptive_entropy_coeff", False):
            return {}
        return {
            str(task_id): coeff.state_dict()
            for task_id, coeff in self.adaptive_entropy_coeffs.items()
        }

    def load_adaptive_entropy_state_dict(self, state: Dict[str, Any]) -> None:
        if not getattr(self, "use_adaptive_entropy_coeff", False) or not state:
            return
        for task_id, coeff in self.adaptive_entropy_coeffs.items():
            task_state = state.get(str(task_id), state.get(task_id))
            if task_state is not None:
                coeff.load_state_dict(task_state)

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

    def _build_output_lengths(self, response_mask: torch.Tensor) -> List[int]:
        # Match `_extract_valid_output_tokens`: length is last valid position + 1,
        # not mask sum, so non-contiguous masks cannot desync labels/logits.
        output_lengths = []
        for i in range(response_mask.size(0)):
            valid = (response_mask[i] == 1)
            if valid.any():
                output_lengths.append(int(valid.nonzero(as_tuple=False)[-1].item()) + 1)
            else:
                output_lengths.append(0)
        return output_lengths

    def _zero_missing_image_placeholder_rows(
        self,
        micro_batch: Dict[str, Any],
        response_mask: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        """Drop rows whose image-conditioned prompt lost its image placeholder."""
        if task_id not in (2, 3):
            return response_mask

        input_key = f"task{task_id}_input_ids"
        input_ids = micro_batch.get(input_key)
        image_id = getattr(self.processor, "image_id", None)
        if input_ids is None or image_id is None:
            return response_mask

        placeholder_counts = (input_ids == image_id).sum(dim=1)
        bad_rows = placeholder_counts == 0
        if not bool(bad_rows.any().item()):
            return response_mask

        response_mask = response_mask.clone()
        response_mask[bad_rows] = 0
        micro_batch[f"task{task_id}_response_mask"] = response_mask
        print(
            f"[DPActor] task{task_id}: zeroed rows with missing image placeholder "
            f"indices={bad_rows.nonzero(as_tuple=False).view(-1).detach().cpu().tolist()}",
            flush=True,
        )
        return response_mask

    def _zero_rows_with_insufficient_logits(
        self,
        micro_batch: Dict[str, Any],
        response_mask: torch.Tensor,
        output_starts: List[int],
        output_lengths: List[int],
        logits_seq_len: int,
        task_id: int,
    ) -> tuple[torch.Tensor, List[int]]:
        """Drop rows where model logits cannot cover the requested output span."""
        bad = []
        for i, (start, out_len) in enumerate(zip(output_starts, output_lengths)):
            if out_len <= 0:
                continue
            logit_start = max(int(start) - 1, 0)
            available = max(0, int(logits_seq_len) - logit_start)
            if available < int(out_len):
                bad.append((i, int(start), int(out_len), int(available)))

        if not bad:
            return response_mask, output_lengths

        response_mask = response_mask.clone()
        bad_indices = [item[0] for item in bad]
        response_mask[bad_indices] = 0
        micro_batch[f"task{task_id}_response_mask"] = response_mask
        output_lengths = self._build_output_lengths(response_mask)
        print(
            f"[DPActor] task{task_id}: zeroed rows with insufficient logits "
            f"logits_seq_len={logits_seq_len} rows={bad}",
            flush=True,
        )
        return response_mask, output_lengths

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
        task_id: int = 1,
        **kwargs
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
        original_response_mask = self._zero_missing_image_placeholder_rows(
            micro_batch, original_response_mask, task_id
        )

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

        # Must match `_extract_valid_output_tokens` exactly.
        output_lengths = self._build_output_lengths(original_response_mask)
        local_has_output = 1 if max(output_lengths) > 0 else 0

        # IMPORTANT: Always do forward pass even if no valid output
        # FSDP requires all ranks to participate in forward/backward for synchronization

        output = self.actor_module(
            task_id=task_id,
            batch=micro_batch,
            cfg_weight=kwargs.get("cfg_weight", None),
            temperature=temperature,
            txt_top_k=kwargs.get("txt_top_k", 0),
            txt_top_p=kwargs.get("txt_top_p", 1.0),
            img_top_k=kwargs.get("img_top_k", 0),
            img_top_p=kwargs.get("img_top_p", 1.0),
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
        logits_seq_len = int(logits.size(1))
        original_response_mask, output_lengths = self._zero_rows_with_insufficient_logits(
            micro_batch,
            original_response_mask,
            output_starts,
            output_lengths,
            logits_seq_len,
            task_id,
        )
        valid_output_tokens = self._extract_valid_output_tokens(output_tokens, original_response_mask)
        if max(output_lengths) == 0:
            dummy_scalar = logits.flatten()[0] * 0.0
            log_probs = torch.zeros_like(original_response_mask, dtype=logits.dtype, device=logits.device) + dummy_scalar
            entropy = None
            if calculate_entropy:
                entropy = torch.zeros_like(original_response_mask, dtype=logits.dtype, device=logits.device) + dummy_scalar
            return entropy, log_probs
        task_logits = extract_output_logits(logits, output_starts, output_lengths)

        del logits
        del output

        # Compute log probabilities
        if task_logits.shape[:2] != valid_output_tokens.shape:
            print(
                _format_logprob_shape_debug(
                    task_id=task_id,
                    logits_seq_len=logits_seq_len,
                    output_starts=output_starts,
                    output_lengths=output_lengths,
                    task_logits=task_logits,
                    valid_output_tokens=valid_output_tokens,
                    output_tokens=output_tokens,
                    response_mask=original_response_mask,
                    micro_batch=micro_batch,
                ),
                flush=True,
            )
        compact_log_probs = logprobs_from_logits(task_logits, valid_output_tokens)
        log_probs = self._restore_log_probs_to_original_length(compact_log_probs, original_response_mask)

        # Calculate entropy if needed
        entropy = None
        if calculate_entropy:
            if not self.config.entropy_checkpointing:
                compact_entropy = self.compute_entropy_from_logits(task_logits)
            else:
                compact_entropy = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, task_logits)
        
            entropy = self._restore_log_probs_to_original_length(compact_entropy, original_response_mask)

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
        cfg_weight = data.meta_info["cfg_weight"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        txt_top_k = data.meta_info.get("txt_top_k", 0)
        txt_top_p = data.meta_info.get("txt_top_p", 1.0)
        img_top_k = data.meta_info.get("img_top_k", 0)
        img_top_p = data.meta_info.get("img_top_p", 1.0)

        task_id = data.batch["task_id"].view(-1)[0].item()

        # Selected batch keys based on task_id
        if task_id == 1:
            select_batch_keys = [
                "task1_input_ids", "task1_attention_mask",
                "task1_gen_img_tokens", "task1_response_mask", "task_id"
            ]
            non_tensor_batch_keys = []
        elif task_id == 2:
            available_keys = set(data.batch.keys())
            select_batch_keys = [
                "task2_input_ids", "task2_attention_mask", "task2_feedback_ids",
                "task2_response_mask",
            ]
            if "task2_segment_mask" in available_keys:
                select_batch_keys.append("task2_segment_mask")
            if "task_id" in available_keys:
                select_batch_keys.append("task_id")
            if "task2_task1_gen_imgs_pixel_values" in available_keys:
                select_batch_keys.append("task2_task1_gen_imgs_pixel_values")
            elif "task1_gen_imgs_pixel_values" in available_keys:
                select_batch_keys.append("task1_gen_imgs_pixel_values")
            non_tensor_batch_keys = ["task2_feedback_texts"]
        elif task_id == 3:
            available_keys = set(data.batch.keys())
            select_batch_keys = [
                "task3_input_ids", "task3_attention_mask",
                "task3_regen_img_tokens", "task3_response_mask",
            ]
            if "task_id" in available_keys:
                select_batch_keys.append("task_id")
            if "task3_task1_gen_img_tokens" in available_keys:
                select_batch_keys.append("task3_task1_gen_img_tokens")
            elif "task3_input_img_tokens" in available_keys:
                select_batch_keys.append("task3_input_img_tokens")
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

            if task_id == 2 and "task2_task1_gen_imgs_pixel_values" in model_inputs:
                model_inputs["task1_gen_imgs_pixel_values"] = model_inputs["task2_task1_gen_imgs_pixel_values"]
            elif task_id == 3 and "task3_task1_gen_img_tokens" in model_inputs:
                model_inputs["task1_gen_img_tokens"] = model_inputs["task3_task1_gen_img_tokens"]
            elif task_id == 3 and "task3_input_img_tokens" in model_inputs:
                model_inputs["task1_gen_img_tokens"] = model_inputs["task3_input_img_tokens"]

            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, task_id=task_id,
                    cfg_weight=cfg_weight, txt_top_k=txt_top_k, txt_top_p=txt_top_p,
                    img_top_k=img_top_k, img_top_p=img_top_p,
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

    def _set_train_eval_modes(self):
        """Set train/eval modes per module.

        requires_grad is set once at model init in image_generation_worker._build_model_optimizer.
        This method only controls BN/Dropout behaviour via train()/eval().
        """
        self.actor_module.language_model.train()
        self.actor_module.gen_head.train()
        self.actor_module.gen_aligner.train()
        self.actor_module.aligner.train()
        self.actor_module.gen_embed.eval()
        self.actor_module.vision_model.eval()
        self.actor_module.gen_vision_model.eval()

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        """Update policy with task-mode or unified step-row batches."""
        self.actor_module.train()
        self._set_train_eval_modes()

        temperature = data.meta_info["temperature"]
        cfg_weight = data.meta_info["cfg_weight"]
        txt_top_k = data.meta_info.get("txt_top_k", 0)
        txt_top_p = data.meta_info.get("txt_top_p", 1.0)
        img_top_k = data.meta_info.get("img_top_k", 0)
        img_top_p = data.meta_info.get("img_top_p", 1.0)

        multi_task_config = self.config.get("multi_task", {})
        enable_multi_task = multi_task_config.get("enable", True)
        task_weights = multi_task_config.get("task_weights", [1.0, 1.0, 1.0])
        task_selection = multi_task_config.get("task_selection", "all")

        def _step_to_task(step_id: int) -> int:
            if step_id == 1:
                return 1
            if step_id in (2, 3, 4):
                return 2
            if step_id == 5:
                return 3
            raise ValueError(f"Invalid step_id: {step_id}")

        available_keys = set(data.batch.keys())
        step_ids = [sid for sid in data.meta_info.get("step_mode_step_ids", []) if f"step{int(sid)}_advantages" in available_keys]
        if not step_ids:
            step_ids = [sid for sid in range(1, 6) if f"step{sid}_advantages" in available_keys]
        step_ids = [int(sid) for sid in step_ids]
        step_mode = bool(step_ids)

        units = []
        all_select_batch_keys = []
        if step_mode:
            for step_id in step_ids:
                task_id = _step_to_task(step_id)
                prefix = f"step{step_id}_"
                units.append({"kind": "step", "id": step_id, "task_id": task_id, "prefix": prefix, "metric": f"actor/step{step_id}"})
                unit_keys = [
                    f"{prefix}input_ids",
                    f"{prefix}attention_mask",
                    f"{prefix}response_mask",
                    f"{prefix}old_log_probs",
                    f"{prefix}advantages",
                ]
                if step_id == 1:
                    unit_keys.append(f"{prefix}gen_img_tokens")
                elif step_id in (2, 3, 4):
                    unit_keys.append(f"{prefix}feedback_ids")
                    unit_keys.append(f"{prefix}segment_mask")
                    unit_keys.append(f"{prefix}task1_gen_imgs_pixel_values")
                elif step_id == 5:
                    unit_keys.append(f"{prefix}regen_img_tokens")
                    unit_keys.append(f"{prefix}task1_gen_img_tokens")
                    unit_keys.append(f"{prefix}input_img_tokens")
                if self.config.use_kl_loss:
                    unit_keys.append(f"{prefix}ref_log_prob")
                unit_keys.append(f"{prefix}rollout_is_weights")
                unit_keys.append(f"{prefix}unit_loss_weights")
                unit_keys.append(f"{prefix}stage_ids")
                unit_keys.append(f"{prefix}lane_ids")
                all_select_batch_keys.extend([k for k in unit_keys if k in available_keys])
        else:
            if enable_multi_task:
                configured_task_ids = multi_task_config.get("task_ids", None)
                if configured_task_ids is not None:
                    task_ids = list(configured_task_ids)
                elif task_selection == "all":
                    task_ids = [1, 2, 3]
                elif task_selection == "weighted_sample":
                    import random
                    task_ids = random.choices([1, 2, 3], weights=task_weights, k=1)
                else:
                    task_ids = [data.batch["task_id"].view(-1)[0].item()]
            else:
                task_ids = [data.batch["task_id"].view(-1)[0].item()]
            task_ids = [tid for tid in task_ids if f"task{tid}_advantages" in available_keys]
            for task_id in task_ids:
                prefix = f"task{task_id}_"
                units.append({"kind": "task", "id": task_id, "task_id": task_id, "prefix": prefix, "metric": f"actor/task{task_id}"})
                task_keys = [
                    f"task{task_id}_input_ids",
                    f"task{task_id}_attention_mask",
                    f"task{task_id}_response_mask",
                    f"task{task_id}_loss_mask",
                    f"task{task_id}_old_log_probs",
                    f"task{task_id}_advantages",
                ]
                if task_id == 1:
                    task_keys.append("task1_gen_img_tokens")
                elif task_id == 2:
                    task_keys.append("task2_feedback_ids")
                    task_keys.append("task2_segment_mask")
                    if "task2_task1_gen_imgs_pixel_values" in available_keys:
                        task_keys.append("task2_task1_gen_imgs_pixel_values")
                    else:
                        task_keys.append("task1_gen_imgs_pixel_values")
                elif task_id == 3:
                    task_keys.append("task3_regen_img_tokens")
                    if "task3_task1_gen_img_tokens" in available_keys:
                        task_keys.append("task3_task1_gen_img_tokens")
                    elif "task3_input_img_tokens" in available_keys:
                        task_keys.append("task3_input_img_tokens")
                if self.config.use_kl_loss:
                    task_keys.append(f"task{task_id}_ref_log_prob")
                task_keys.append(f"task{task_id}_rollout_is_weights")
                task_keys.append(f"task{task_id}_unit_loss_weights")
                all_select_batch_keys.extend([k for k in task_keys if k in available_keys])
            if "task_id" in available_keys:
                all_select_batch_keys.append("task_id")

        if not units:
            print("[dp_actor] WARNING: no units have computed advantages, skipping update_policy")
            return {}

        all_select_batch_keys = list(dict.fromkeys(all_select_batch_keys))
        non_tensor_available = set((getattr(data, "non_tensor_batch", None) or {}).keys())
        non_tensor_select_keys = [k for k in ["uid"] if k in non_tensor_available]
        data = data.select(all_select_batch_keys, non_tensor_select_keys)

        num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
        mini_batches = data.chunk(num_mini_batches)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        metrics = {}

        def _unit_inputs(base_inputs, unit):
            task_id = unit["task_id"]
            if unit["kind"] == "task":
                if task_id == 2 and "task2_task1_gen_imgs_pixel_values" in base_inputs:
                    return {**base_inputs, "task1_gen_imgs_pixel_values": base_inputs["task2_task1_gen_imgs_pixel_values"]}
                if task_id == 3 and "task3_task1_gen_img_tokens" in base_inputs:
                    return {**base_inputs, "task1_gen_img_tokens": base_inputs["task3_task1_gen_img_tokens"]}
                if task_id == 3 and "task3_input_img_tokens" in base_inputs:
                    return {**base_inputs, "task1_gen_img_tokens": base_inputs["task3_input_img_tokens"]}
                return base_inputs

            step_id = unit["id"]
            prefix = unit["prefix"]
            task_prefix = f"task{task_id}_"
            mapped = {**base_inputs}
            for suffix in (
                "input_ids", "attention_mask", "response_mask", "loss_mask", "old_log_probs", "advantages",
                "ref_log_prob", "rollout_is_weights", "unit_loss_weights", "stage_ids", "lane_ids", "gen_img_tokens",
                "feedback_ids", "regen_img_tokens",
            ):
                src = f"{prefix}{suffix}"
                if src in base_inputs:
                    mapped[f"{task_prefix}{suffix}"] = base_inputs[src]
            if step_id in (2, 3, 4):
                if f"{prefix}segment_mask" in base_inputs:
                    mapped["task2_segment_mask"] = base_inputs[f"{prefix}segment_mask"]
                if f"{prefix}task1_gen_imgs_pixel_values" in base_inputs:
                    mapped["task2_task1_gen_imgs_pixel_values"] = base_inputs[f"{prefix}task1_gen_imgs_pixel_values"]
                    mapped["task1_gen_imgs_pixel_values"] = base_inputs[f"{prefix}task1_gen_imgs_pixel_values"]
            elif step_id == 5:
                if f"{prefix}task1_gen_img_tokens" in base_inputs:
                    mapped["task3_task1_gen_img_tokens"] = base_inputs[f"{prefix}task1_gen_img_tokens"]
                    mapped["task1_gen_img_tokens"] = base_inputs[f"{prefix}task1_gen_img_tokens"]
                elif f"{prefix}input_img_tokens" in base_inputs:
                    mapped["task3_input_img_tokens"] = base_inputs[f"{prefix}input_img_tokens"]
                    mapped["task1_gen_img_tokens"] = base_inputs[f"{prefix}input_img_tokens"]
            return mapped

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = (
                        mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.select(all_select_batch_keys, non_tensor_select_keys).chunk(num_micro_batches)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    base_inputs = {**micro_batch.batch}
                    micro_batch_metrics = {}

                    for unit in units:
                        task_id = unit["task_id"]
                        metric_prefix = unit["metric"]
                        model_inputs = _unit_inputs(base_inputs, unit)

                        old_log_prob = model_inputs[f"task{task_id}_old_log_probs"]
                        advantages = model_inputs[f"task{task_id}_advantages"]
                        unit_loss_weights = model_inputs.get(f"task{task_id}_unit_loss_weights", None)
                        if unit_loss_weights is not None:
                            advantages = advantages * unit_loss_weights.to(device=advantages.device, dtype=advantages.dtype).view(-1, 1)
                        response_mask = model_inputs[f"task{task_id}_response_mask"]
                        loss_mask = model_inputs.get(f"task{task_id}_loss_mask", response_mask)
                        if task_id == 2 and "task2_segment_mask" in model_inputs:
                            segment_mask = model_inputs["task2_segment_mask"].to(device=response_mask.device)
                            valid_mask = response_mask > 0
                            weighted_mask = torch.zeros_like(response_mask, dtype=torch.float32)
                            for seg_id in (2, 3, 4):
                                seg_mask = ((segment_mask == seg_id) & valid_mask).to(dtype=torch.float32)
                                seg_len = seg_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                                weighted_mask = weighted_mask + seg_mask / seg_len
                            if weighted_mask.sum() > 0:
                                loss_mask = weighted_mask

                        if self.use_adaptive_entropy_coeff:
                            entropy_coeff = -self.adaptive_entropy_coeffs[task_id].get_alpha().item()
                        else:
                            entropy_coeff = self.config.entropy_coeff
                        loss_agg_mode = self.config.loss_agg_mode

                        if self.config.use_dynamic_bsz:
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation

                        calculate_entropy = entropy_coeff != 0 or self.use_adaptive_entropy_coeff
                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            task_id=task_id,
                            cfg_weight=cfg_weight,
                            txt_top_k=txt_top_k,
                            txt_top_p=txt_top_p,
                            img_top_k=img_top_k,
                            img_top_p=img_top_p,
                        )

                        if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                            old_log_prob = model_inputs[f"task{task_id}_old_log_probs"]
                        else:
                            old_log_prob = log_prob.detach() if on_policy else model_inputs[f"task{task_id}_old_log_probs"]

                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        rollout_is_weights = model_inputs.get(f"task{task_id}_rollout_is_weights", None)
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=loss_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )

                        if calculate_entropy:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)
                            if self.use_adaptive_entropy_coeff:
                                self.adaptive_entropy_coeffs[task_id].update(entropy=entropy_loss.detach())
                            micro_batch_metrics[f"{metric_prefix}_entropy"] = entropy_loss.detach().item()
                            micro_batch_metrics[f"{metric_prefix}_entropy_loss"] = (entropy_loss * entropy_coeff).detach().item() * loss_scale_factor
                            micro_batch_metrics[f"{metric_prefix}_entropy_coeff"] = entropy_coeff
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        kld = None
                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs[f"task{task_id}_ref_log_prob"]
                            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            micro_batch_metrics[f"{metric_prefix}_kl"] = kl_loss.detach().item()
                            micro_batch_metrics[f"{metric_prefix}_kl_loss"] = (kl_loss * self.config.kl_loss_coef).detach().item() * loss_scale_factor
                            micro_batch_metrics[f"{metric_prefix}_kl_coef"] = self.config.kl_loss_coef

                        task_weight = task_weights[task_id - 1] if enable_multi_task and not step_mode else 1.0
                        unit_weight_mean = 1.0
                        if unit_loss_weights is not None:
                            task_weight = 1.0
                            unit_weight_mean = float(unit_loss_weights.detach().float().mean().item())
                        weighted_loss = policy_loss * task_weight * loss_scale_factor

                        stage_ids_tensor = model_inputs.get(f"task{task_id}_stage_ids", None)
                        if stage_ids_tensor is not None:
                            stage_ids_tensor = stage_ids_tensor.to(device=response_mask.device).view(-1)
                            for raw_stage in torch.unique(stage_ids_tensor).detach().cpu().tolist():
                                stage_id = int(raw_stage)
                                stage_row_mask = (stage_ids_tensor == stage_id).to(dtype=loss_mask.dtype, device=loss_mask.device).view(-1, 1)
                                stage_loss_mask = loss_mask * stage_row_mask
                                if float(stage_loss_mask.sum().item()) <= 0.0:
                                    continue
                                s_pg_loss, s_pg_clipfrac, s_ppo_kl, s_pg_clipfrac_lower = policy_loss_fn(
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=advantages,
                                    response_mask=stage_loss_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    config=self.config,
                                    rollout_is_weights=rollout_is_weights,
                                )
                                s_policy_loss = s_pg_loss
                                if calculate_entropy:
                                    s_entropy_loss = agg_loss(loss_mat=entropy, loss_mask=stage_loss_mask, loss_agg_mode=loss_agg_mode)
                                    s_policy_loss = s_policy_loss - s_entropy_loss * entropy_coeff
                                    micro_batch_metrics[f"actor/stage{stage_id}_step{unit['id']}_entropy"] = s_entropy_loss.detach().item()
                                if self.config.use_kl_loss and kld is not None:
                                    s_kl_loss = agg_loss(loss_mat=kld, loss_mask=stage_loss_mask, loss_agg_mode=loss_agg_mode)
                                    s_policy_loss = s_policy_loss + s_kl_loss * self.config.kl_loss_coef
                                    micro_batch_metrics[f"actor/stage{stage_id}_step{unit['id']}_kl_loss"] = (
                                        s_kl_loss * self.config.kl_loss_coef
                                    ).detach().item() * loss_scale_factor
                                s_weighted_loss = s_policy_loss * task_weight * loss_scale_factor
                                micro_batch_metrics.update({
                                    f"actor/stage{stage_id}_step{unit['id']}_pg_loss": s_pg_loss.detach().item() * loss_scale_factor,
                                    f"actor/stage{stage_id}_step{unit['id']}_pg_clipfrac": s_pg_clipfrac.detach().item(),
                                    f"actor/stage{stage_id}_step{unit['id']}_ppo_kl": s_ppo_kl.detach().item(),
                                    f"actor/stage{stage_id}_step{unit['id']}_pg_clipfrac_lower": s_pg_clipfrac_lower.detach().item(),
                                    f"actor/stage{stage_id}_step{unit['id']}_loss": s_weighted_loss.detach().item(),
                                })

                        weighted_loss.backward()

                        task_grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.actor_module.parameters(), max_norm=float('inf')
                        ).item()
                        micro_batch_metrics.update({
                            f"{metric_prefix}_cum_grad_norm": task_grad_norm,
                            f"{metric_prefix}_pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            f"{metric_prefix}_pg_clipfrac": pg_clipfrac.detach().item(),
                            f"{metric_prefix}_ppo_kl": ppo_kl.detach().item(),
                            f"{metric_prefix}_pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            f"{metric_prefix}_weight": task_weight,
                            f"{metric_prefix}_unit_weight_mean": unit_weight_mean,
                            f"{metric_prefix}_loss": weighted_loss.detach().item(),
                        })

                        adv = model_inputs[f"task{task_id}_advantages"]
                        mask = model_inputs[f"task{task_id}_response_mask"]
                        lp = log_prob.detach()
                        seq_adv_sum = adv[:, 0].view(-1, 1)
                        is_pos_seq = seq_adv_sum > 0
                        is_neg_seq = seq_adv_sum < 0
                        pos_mask = is_pos_seq & mask.bool()
                        neg_mask = is_neg_seq & mask.bool()
                        micro_batch_metrics[f"{metric_prefix}_pos_log_prob"] = lp[pos_mask].sum().item()
                        micro_batch_metrics[f"{metric_prefix}_pos_log_prob_cnt"] = pos_mask.sum().item()
                        micro_batch_metrics[f"{metric_prefix}_neg_log_prob"] = lp[neg_mask].sum().item()
                        micro_batch_metrics[f"{metric_prefix}_neg_log_prob_cnt"] = neg_mask.sum().item()

                    if len(units) > 1:
                        prefixes = [u["metric"] for u in units]
                        aggregated_metrics = {
                            "actor/avg_pg_loss": sum(micro_batch_metrics.get(f"{p}_pg_loss", 0.0) for p in prefixes) / len(prefixes),
                            "actor/avg_pg_clipfrac": sum(micro_batch_metrics.get(f"{p}_pg_clipfrac", 0.0) for p in prefixes) / len(prefixes),
                            "actor/avg_ppo_kl": sum(micro_batch_metrics.get(f"{p}_ppo_kl", 0.0) for p in prefixes) / len(prefixes),
                            "actor/avg_pg_clipfrac_lower": sum(micro_batch_metrics.get(f"{p}_pg_clipfrac_lower", 0.0) for p in prefixes) / len(prefixes),
                            "actor/loss": sum(micro_batch_metrics.get(f"{p}_loss", 0.0) for p in prefixes),
                        }
                        if any(f"{p}_kl_loss" in micro_batch_metrics for p in prefixes):
                            aggregated_metrics["actor/avg_kl_loss"] = sum(micro_batch_metrics.get(f"{p}_kl_loss", 0.0) for p in prefixes) / len(prefixes)
                        if step_mode:
                            for stage_id in (1, 2):
                                stage_loss_keys = [f"actor/stage{stage_id}_step{u['id']}_loss" for u in units]
                                present_loss = [micro_batch_metrics[k] for k in stage_loss_keys if k in micro_batch_metrics]
                                if present_loss:
                                    aggregated_metrics[f"actor/stage{stage_id}_loss"] = sum(present_loss)
                                    pg_keys = [f"actor/stage{stage_id}_step{u['id']}_pg_loss" for u in units]
                                    pg_vals = [micro_batch_metrics[k] for k in pg_keys if k in micro_batch_metrics]
                                    if pg_vals:
                                        aggregated_metrics[f"actor/stage{stage_id}_avg_pg_loss"] = sum(pg_vals) / len(pg_vals)
                        micro_batch_metrics.update(aggregated_metrics)

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
