import logging
import os

import torch

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig
import contextlib
from recipe.image_rl.utils import FormattingEvaluator

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_image_generation_loss(
    old_log_prob: torch.Tensor,  # (bs, seq_len)
    log_prob: torch.Tensor,  # (bs, seq_len)
    rewards: torch.Tensor,  # (bs,)
    response_mask: torch.Tensor,  # (bs, seq_len)
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
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        processor (Processor): Processor
        tokenizer (Tokenizer): Tokenizer
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """
    def __init__(self, config: ActorConfig, processor: None, tokenizer: None, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config)
        self.processor = processor
        self.tokenizer = tokenizer
        self.formatter = FormattingEvaluator()
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
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
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

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

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, task_id: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        param_ctx = contextlib.nullcontext()
        if isinstance(self.actor_module, FSDP):
            param_ctx = FSDP.summon_full_params(self.actor_module, writeback=False, recurse=True)

        with param_ctx:
            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                
                if task_id == 1:
                    # Task 1: Image Generation: input
                    task1_input_ids = micro_batch["task1_input_ids"]
                    task1_attention_mask = micro_batch["task1_attention_mask"]
                    task1_input_embeds = self.actor_module.language_model.get_input_embeddings()(task1_input_ids)

                    # Task 1: Image Generation: output
                    gen_imgs_pixel_values = micro_batch["task1_gen_imgs_pixel_values"]
                    
                    _, _, all_image_ids = self.actor_module.gen_vision_model.encode(gen_imgs_pixel_values)
            
                    task1_image_ids = all_image_ids[2]
                    task1_image_ids = task1_image_ids.view(gen_imgs_pixel_values.size(0), -1)

                    task1_gen_img_embeds = self.actor_module.gen_aligner(self.module.gen_embed(task1_image_ids))
                    
                    # Labels
                    gen_img_tokens = micro_batch["task1_gen_img_tokens"]

                    # Response mask
                    task1_response_mask = micro_batch["task1_response_mask"]

                    # Task 1: Image Generation
                    task1_output = self.actor_module.language_model.model(
                        inputs_embeds=torch.cat([task1_input_embeds.to(self.device_name), task1_gen_img_embeds.to(self.device_name)], dim=1),
                        attention_mask=torch.cat([task1_attention_mask.to(self.device_name), task1_response_mask.to(self.device_name)], dim=1),
                    )  # prevent model thinks we are generating

                    # Response masking until EOS tokens
                    eos_pos = (task1_response_mask == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[1]
                    task1_response_mask = task1_response_mask[:, :eos_pos[0]]
                    
                    task1_logits = self.actor_module.gen_head(task1_output.last_hidden_state)
                    task1_response_length = task1_response_mask.size(1)
                    task1_logits = task1_logits[:, -task1_response_length - 1 : -1, :]

                    task1_log_probs = logprobs_from_logits(task1_logits, gen_img_tokens.to(self.device_name))

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            task1_entropy = verl_F.entropy_from_logits(task1_logits)  # (bsz, gen_img_tokens)
                        else:
                            task1_entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, task1_logits)
                    return task1_entropy, task1_log_probs

                if task_id == 2:
                    # Task 2: Feedback Generation: input
                    task2_input_ids = micro_batch["task2_input_ids"]
                    task2_attention_mask = micro_batch["task2_attention_mask"]

                    task2_text_embeds = self.actor_module.language_model.get_input_embeddings()(task2_input_ids)
                    task2_image_embeds = self.actor_module.aligner(self.actor_module.vision_model(gen_imgs_pixel_values))

                    task2_image_mask = (task2_input_ids == self.processor.image_id)
                    task2_merged_embeds = torch.where(task2_image_mask.unsqueeze(-1), task2_image_embeds.unsqueeze(-1), task2_text_embeds.clone())
                    
                    # Task 2: Feedback Generation: output
                    feedback_ids = micro_batch["task2_feedback_ids"]
                    task2_feedback_embeds = self.actor_module.language_model.get_input_embeddings()(feedback_ids)

                    # Response mask
                    task2_response_mask = micro_batch["task2_response_mask"]

                    # Task 2: Feedback Generation
                    task2_output = self.actor_module.language_model(
                        inputs_embeds=torch.cat([task2_merged_embeds.to(self.device_name), task2_feedback_embeds.to(self.device_name)], dim=1),
                        attention_mask=torch.cat([task2_attention_mask.to(self.device_name), task2_response_mask.to(self.device_name)], dim=1),
                    )  # prevent model thinks we are generating

                    # Response masking until EOS tokens
                    eos_pos = (task2_response_mask == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[1]
                    task2_response_mask = task2_response_mask[:, :eos_pos[0]]
                    
                    task2_logits = task2_output.logits
                    task2_response_length = task2_response_mask.size(1)
                    task2_logits = task2_logits[:, -task2_response_length - 1 : -1, :]

                    task2_log_probs = logprobs_from_logits(task2_logits, feedback_ids.to(self.device_name))

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            task2_entropy = verl_F.entropy_from_logits(task2_logits)  # (bsz, feedback_ids)
                    else:
                        task2_entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, task2_logits)
                    return task2_entropy, task2_log_probs

                if task_id == 3:
                    # Task 3: Regen Image Generation: input
                    task3_input_ids = micro_batch["task3_input_ids"]
                    task3_attention_mask = micro_batch["task3_attention_mask"]
                    
                    task3_text_embeds = self.actor_module.language_model.get_input_embeddings()(task3_input_ids)
                    task3_image_embeds = self.actor_module.aligner(self.actor_module.vision_model(gen_imgs_pixel_values))

                    task3_image_mask = (task3_input_ids == self.processor.image_id)
                    task3_merged_embeds = torch.where(task3_image_mask.unsqueeze(-1), task3_image_embeds.unsqueeze(-1), task3_text_embeds.clone())

                    # Task 3: Regen Image Generation: output
                    regen_imgs_pixel_values = micro_batch["task3_regen_imgs_pixel_values"]
                    
                    _, _, all_image_ids = self.actor_module.gen_vision_model.encode(regen_imgs_pixel_values)
            
                    task3_image_ids = all_image_ids[2]
                    task3_image_ids = task3_image_ids.view(regen_imgs_pixel_values.size(0), -1)

                    task3_regen_img_embeds = self.actor_module.gen_aligner(self.module.gen_embed(task3_image_ids))

                    # Labels
                    regen_img_tokens = micro_batch["task3_regen_img_tokens"]

                    # Response mask
                    task3_response_mask = micro_batch["task3_response_mask"]

                    # Task 3: Regen Image Generation
                    task3_output = self.actor_module.language_model.model(
                        inputs_embeds=torch.cat([task3_merged_embeds.to(self.device_name), task3_regen_img_embeds.to(self.device_name)], dim=1),
                        attention_mask=torch.cat([task3_attention_mask.to(self.device_name), task3_response_mask.to(self.device_name)], dim=1),
                    )  # prevent model thinks we are generating

                    # Response masking until EOS tokens
                    eos_pos = (task3_response_mask == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[1]
                    task3_response_mask = task3_response_mask[:, :eos_pos[0]]
                    
                    task3_logits = self.actor_module.gen_head(task3_output.last_hidden_state)
                    task3_response_length = task3_response_mask.size(1)
                    task3_logits = task3_logits[:, -task3_response_length - 1 : -1, :]

                    task3_log_probs = logprobs_from_logits(task3_logits, regen_img_tokens.to(self.device_name))

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            task3_entropy = verl_F.entropy_from_logits(task3_logits)  # (bsz, regen_img_tokens)
                        else:
                            task3_entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, task3_logits)
                    return task3_entropy, task3_log_probs 

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
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, task_id: int = 1) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        
        # batch_keys
        select_batch_keys = [
            "task1_input_ids", "task1_attention_mask", "task1_gen_imgs_pixel_values", "task1_gen_img_tokens", "task1_response_mask",
            "task2_input_ids", "task2_attention_mask", "task2_feedback_ids", "task2_response_mask",
            "task3_input_ids", "task3_attention_mask", "task3_regen_imgs_pixel_values", "task3_regen_img_tokens", "task3_response_mask"
        ]
        task_select_batch_keys = [key for key in select_batch_keys if str(task_id) in key]

        # non_tensor_batch_keys
        non_tensor_batch_keys = ["task2_feedback_texts"]

        data = data.select(batch_keys=task_select_batch_keys, non_tensor_batch_keys=non_tensor_batch_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            seq_len = micro_batch["task1_response_mask"].size(1)
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            
            # ### task2 -> 3 jump
            # micro_batch_feedback_text = model_inputs["task2_feedback_texts"]
            # feedback = self.formatter._split_text_into_parts(micro_batch_feedback_text)[-1]
            # if feedback is None or feedback == "No need to generate feedback.":
            #     log_probs_lst.append(torch.zeros(1, seq_len))
            #     if calculate_entropy:
            #         entropy_lst.append(torch.zeros(1, seq_len))
            #     continue

            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, task_id=task_id
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
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

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto, task_id: int = 1):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        # batch_keys
        select_batch_keys = [
            "task1_input_ids", "task1_attention_mask", "task1_gen_imgs_pixel_values", "task1_gen_img_tokens", "task1_response_mask",
            "task2_input_ids", "task2_attention_mask", "task2_feedback_ids", "task2_response_mask",
            "task3_input_ids", "task3_attention_mask", "task3_regen_imgs_pixel_values", "task3_regen_img_tokens", "task3_response_mask"
            "task1_old_log_probs", "task2_old_log_probs", "task3_old_log_probs",
            "task1_advantages", "task2_advantages", "task2_advantages",
        ]
        task_select_batch_keys = [key for key in select_batch_keys if str(task_id) in key]

        # non_tensor_batch_keys
        non_tensor_batch_keys = ["task2_feedback_texts"]

        if self.config.use_kl_loss:
            task_select_batch_keys.append(f"task{task_id}_ref_log_prob")
        if self.config.tis_imp_ratio_cap > 0:
            assert f"{task_id}_rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            task_select_batch_keys.append(f"task{task_id}_rollout_log_probs")

        data = data.select(meta_info_keys=task_select_batch_keys, non_tensor_batch_keys=non_tensor_batch_keys)

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
                    response_mask = model_inputs[f"task{task_id}_response_mask"]
                    old_log_prob = model_inputs[f"task{task_id}_old_log_probs"]
                    rollout_log_probs = model_inputs[f"task{task_id}_rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs[f"task{task_id}_advantages"]

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

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs[f"task{task_id}_old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_log_probs=rollout_log_probs,
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
                            f"actor/{task_id}_pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            f"actor/{task_id}_pg_clipfrac": pg_clipfrac.detach().item(),
                            f"actor/{task_id}_ppo_kl": ppo_kl.detach().item(),
                            f"actor/{task_id}_pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {f"actor/{task_id}_grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
