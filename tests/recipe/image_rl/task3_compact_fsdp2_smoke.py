"""Eight-GPU Task3 compact-logits memory smoke.

Run with the same environment as training, for example:

    torchrun --standalone --nproc-per-node=8 \
        tests/recipe/image_rl/task3_compact_fsdp2_smoke.py --prompt-length 2048

This intentionally reuses ``ImageGenerationActorRolloutRefWorker`` so model
loading, FSDP2 wrapping, mixed precision, gradient checkpointing, optimizer,
and the actor forward are the production implementations.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir

from recipe.image_rl.image_generation_worker import ImageGenerationActorRolloutRefWorker


DEFAULT_MODEL_PATH = (
    "/home/work/AGILAB/mllm_reasoning/data/experiments/ckpt/janus_sft/"
    "1223_v10_sft_warmup_constant_long_prompt/version_1/"
    "step=014000.ckpt/hf_model"
)


def _compose_worker_config(model_path: str):
    config_dir = os.path.abspath("recipe/fully_async_policy_image_rl/config")
    overrides = [
        f"actor_rollout_ref.model.path='{model_path}'",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_liger=True",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        "actor_rollout_ref.actor.task3_compact_logits=True",
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking=True",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=float32",
        "actor_rollout_ref.actor.fsdp_config.use_orig_params=True",
        "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=8",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        "actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LlamaDecoderLayer]",
        "actor_rollout_ref.actor.adaptive_entropy_coeff.enable=False",
        "actor_rollout_ref.rollout.n=16",
    ]
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="fully_async_ppo_trainer", overrides=overrides)
    return config.actor_rollout_ref


def _make_task3_micro_batch(worker, batch_size: int, prompt_length: int):
    processor = worker.processor
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    image_tokens = int(processor.num_image_tokens)
    if prompt_length < image_tokens + 2:
        raise ValueError(
            f"prompt_length={prompt_length} cannot contain {image_tokens} image placeholders"
        )

    fill_id = processor.tokenizer.eos_token_id
    input_ids = torch.full(
        (batch_size, prompt_length), fill_id, dtype=torch.long, device=device
    )
    image_start = min(128, prompt_length - image_tokens - 1)
    input_ids[:, image_start : image_start + image_tokens] = processor.image_id
    attention_mask = torch.ones_like(input_ids)

    generator = torch.Generator(device=device)
    generator.manual_seed(1234 + dist.get_rank())
    input_image_tokens = torch.randint(
        0, 16384, (batch_size, image_tokens), generator=generator, device=device
    )
    response_tokens = torch.randint(
        0, 16384, (batch_size, image_tokens), generator=generator, device=device
    )
    response_mask = torch.ones(
        (batch_size, image_tokens), dtype=torch.long, device=device
    )
    return {
        "task3_input_ids": input_ids,
        "task3_attention_mask": attention_mask,
        "task1_gen_img_tokens": input_image_tokens,
        "task3_regen_img_tokens": response_tokens,
        "task3_response_mask": response_mask,
    }


def _global_extreme(value: int, op: dist.ReduceOp) -> int:
    tensor = torch.tensor(value, dtype=torch.int64, device=torch.cuda.current_device())
    dist.all_reduce(tensor, op=op)
    return int(tensor.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    worker = ImageGenerationActorRolloutRefWorker(
        config=_compose_worker_config(args.model_path), role="actor"
    )
    worker.init_model()
    actor = worker.actor
    micro_batch = _make_task3_micro_batch(worker, args.micro_batch_size, args.prompt_length)

    dist.barrier()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    actor.actor_module.eval()
    with torch.no_grad():
        log_probs, entropy = actor._forward_micro_batch(
            micro_batch,
            temperature=1.0,
            calculate_entropy=True,
            task_id=3,
            cfg_weight=2.0,
        )
    if log_probs.shape != (args.micro_batch_size, 576):
        raise AssertionError(f"unexpected log_probs shape: {tuple(log_probs.shape)}")
    if entropy.shape != log_probs.shape:
        raise AssertionError(f"unexpected entropy shape: {tuple(entropy.shape)}")
    if not torch.isfinite(log_probs).all() or not torch.isfinite(entropy).all():
        raise AssertionError("non-finite Task3 log-prob or entropy")

    del log_probs, entropy
    torch.cuda.empty_cache()
    actor.actor_module.train()
    actor._set_train_eval_modes()
    actor.actor_optimizer.zero_grad()
    entropy, log_probs = actor._forward_micro_batch(
        micro_batch,
        temperature=1.0,
        calculate_entropy=True,
        task_id=3,
        cfg_weight=2.0,
    )
    loss = -log_probs.mean() - 1e-6 * entropy.mean()
    if not torch.isfinite(loss):
        raise AssertionError(f"non-finite Task3 smoke loss: {loss.item()}")
    loss.backward()
    grad_norm = actor._optimizer_step()
    actor.actor_optimizer.zero_grad()
    torch.cuda.synchronize()
    dist.barrier()

    peak_allocated = _global_extreme(torch.cuda.max_memory_allocated(), dist.ReduceOp.MAX)
    peak_reserved = _global_extreme(torch.cuda.max_memory_reserved(), dist.ReduceOp.MAX)
    total_memory = _global_extreme(torch.cuda.get_device_properties(local_rank).total_memory, dist.ReduceOp.MIN)
    min_headroom = total_memory - peak_allocated
    if dist.get_rank() == 0:
        print(
            "TASK3_COMPACT_FSDP2_SMOKE "
            + json.dumps(
                {
                    "prompt_length": args.prompt_length,
                    "micro_batch_size_per_gpu": args.micro_batch_size,
                    "world_size": dist.get_world_size(),
                    "loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "peak_allocated_gib": peak_allocated / 2**30,
                    "peak_reserved_gib": peak_reserved / 2**30,
                    "min_headroom_gib": min_headroom / 2**30,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
