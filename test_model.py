"""
Reproduce task2 (feedback generation) and task3 (regen image) from a rollout dump.

Usage:
    # 전체 task2 생성 (summary부터)
    python test_model.py --input-txt /verl/txt.txt

    # 어디서부터 생성할지 선택 (--rollout-from)
    #   summary  : 처음부터 전부 생성 (기본 full task2)
    #   decompose: 덤프 summary 고정, decompose부터 생성
    #   verify   : 덤프 summary+decompose 고정, verify부터 생성
    #   fourth   : 덤프 summary+decompose+verify 고정, fourth만 생성
    python test_model.py --input-txt /verl/txt.txt --rollout-from decompose --n-rollouts 4
    python test_model.py --input-txt /verl/txt.txt --rollout-from verify    --n-rollouts 4
    python test_model.py --input-txt /verl/txt.txt --rollout-from fourth    --n-rollouts 4

    # 덤프의 feedback 그대로 task3만 실행
    python test_model.py --input-txt /verl/txt.txt --use-dump-feedback --n-rollouts 8

    # 직접 지정
    python test_model.py --gen_img_path /path/to/gen.png --prompt "a photo of ..."
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from tensordict import TensorDict
from types import SimpleNamespace

from verl import DataProto

sys.path.insert(0, str(Path(__file__).parent))
from test_reward_server_finegrained import load_payload_from_txt


# ── stage constants ────────────────────────────────────────────────────────────

FIRST_HEADER  = "First, summarize the input prompt by keeping only explicitly stated visual facts\n"
SECOND_HEADER = "Second, Decompose summarize\n"
THIRD_HEADER  = "Third, Verify that the decomposed elements align with the image.\n"
FOURTH_HEADER = "Fourth, Generate corrective feedback.\n"


# ── helpers ───────────────────────────────────────────────────────────────────

def encode_image_to_gen_tokens(module, pil_image: PIL.Image.Image, device) -> torch.Tensor:
    """PIL image → VQ codebook indices (1, 576) via gen_vision_model.encode()."""
    img = pil_image.convert("RGB").resize((384, 384), PIL.Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0 * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, _, info = module.gen_vision_model.encode(tensor)
    return info[2].view(1, -1).long()  # (1, 576)


def build_rollout_from_config(rollout_from: str, payload: dict) -> tuple[str, str]:
    """(input_suffix, feedback_fixed_prefix) 반환.

    input_suffix        : <image_end_tag>\\n 이후 입력에 붙일 이미 생성된 prefix
    feedback_fixed_prefix: task3에 넘길 full_feedback의 고정 앞부분
                           (generated_text 를 뒤에 붙여 full_feedback 완성)
    """
    summary     = payload.get("summary", "")
    pred_tuples = payload.get("pred_tuples", "")
    vqa_results = payload.get("vqa_results", "")

    if rollout_from == "summary":
        # 아무것도 고정 안 함 → summary부터 전부 생성
        return FIRST_HEADER, ""

    elif rollout_from == "decompose":
        # summary 고정 → Second (decompose) 부터 생성
        fixed = FIRST_HEADER + summary + "\n\n" + SECOND_HEADER
        prefix = summary + "\n\n" + SECOND_HEADER
        return fixed, prefix

    elif rollout_from == "verify":
        # summary + decompose 고정 → Third (verify) 부터 생성
        fixed  = FIRST_HEADER + summary + "\n\n" + SECOND_HEADER + pred_tuples + "\n\n" + THIRD_HEADER
        prefix = summary + "\n\n" + SECOND_HEADER + pred_tuples + "\n\n" + THIRD_HEADER
        return fixed, prefix

    elif rollout_from == "fourth":
        # summary + decompose + verify 고정 → Fourth 만 생성
        fixed = (FIRST_HEADER + summary + "\n\n"
                 + SECOND_HEADER + pred_tuples + "\n\n"
                 + THIRD_HEADER + vqa_results + "\n\n"
                 + FOURTH_HEADER)
        prefix = (summary + "\n\n"
                  + SECOND_HEADER + pred_tuples + "\n\n"
                  + THIRD_HEADER + vqa_results + "\n\n"
                  + FOURTH_HEADER)
        return fixed, prefix

    else:
        raise ValueError(f"Unknown rollout_from: {rollout_from!r}")


def generate_from_stage(rollout, data: DataProto, input_suffix_per_sample: list[str]) -> list[str]:
    """<image_tag><image_end_tag>\\n 이후에 input_suffix를 붙이고 나머지를 생성."""
    input_format = []
    for prompt, suffix in zip(data.non_tensor_batch["prompt"], input_suffix_per_sample):
        sft = rollout.get_sft_format(prompt)  # ends with <image_start_tag>
        input_format.append(sft + rollout.image_tag + rollout.image_end_tag + "\n" + suffix)

    rollout.processor.tokenizer.pad_token_id = rollout.processor.pad_id
    inputs = rollout.processor.tokenizer(
        input_format, padding_side="left", padding=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(rollout.device)

    gen_imgs_pixel_values = data.batch.get("task1_gen_imgs_pixel_values", [])
    batched_total_ids, batched_attention_mask, _, all_image_start_indices, _ = \
        rollout.expand_image_placeholders(input_ids, gen_imgs_pixel_values)

    gen_imgs_pixel_values = gen_imgs_pixel_values.to(rollout.device, dtype=torch.bfloat16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        text_embeds  = rollout.module.language_model.get_input_embeddings()(batched_total_ids.long())
        image_embeds = rollout.module.aligner(rollout.module.vision_model(gen_imgs_pixel_values))

    merged_embeds = rollout.merge_text_and_image_embeds(text_embeds, image_embeds, all_image_start_indices)
    generated_texts, _, _ = rollout.generate_text(merged_embeds, batched_attention_mask)
    torch.cuda.empty_cache()
    return generated_texts


def build_dataproto(prompt: str, pil_image: PIL.Image.Image, processor, module, device,
                    n_rollouts: int = 1) -> DataProto:
    gen_imgs_tensor = processor.image_processor([pil_image]).pixel_values           # (1, C, H, W)
    gen_img_tokens  = encode_image_to_gen_tokens(module, pil_image, device)          # (1, 576)

    gen_imgs_tensor = gen_imgs_tensor.repeat(n_rollouts, 1, 1, 1)
    gen_img_tokens  = gen_img_tokens.repeat(n_rollouts, 1)

    batch = TensorDict(
        {
            "task1_gen_imgs_pixel_values": gen_imgs_tensor.cpu(),
            "task1_gen_img_tokens":        gen_img_tokens.cpu(),
            "task1_response_mask":         torch.ones((n_rollouts, 576), dtype=torch.long),
        },
        batch_size=[n_rollouts],
    )
    non_tensor_batch = {
        "prompt":               np.array([prompt]    * n_rollouts, dtype=object),
        "task1_gen_imgs_pil_list": np.array([pil_image] * n_rollouts, dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})


# ── arg parsing ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/home/work/AGILAB/mllm_reasoning/data/experiments/ckpt/janus_sft/1223_v10_sft_warmup_constant_long_prompt/version_1/step=014000.ckpt/hf_model")
    p.add_argument("--output_dir", default="./debug_rollout")
    p.add_argument("--n-rollouts", type=int, default=8,
                   help="같은 (prompt, image)로 생성할 rollout 수 (배치 크기). 기본 1.")

    # txt 덤프
    p.add_argument("--input-txt", type=Path, default=None,
                   help="롤아웃 리포트 txt 파일.")
    p.add_argument("--use-dump-feedback", action="store_true",
                   help="덤프 feedback을 task3 입력으로 사용 (task2 스킵).")
    p.add_argument("--rollout-from",
                   choices=["summary", "decompose", "verify", "fourth"],
                   default=None,
                   help=(
                       "어디서부터 생성할지 선택. --input-txt 필요.\n"
                       "  summary  : 처음부터 전부 생성 (기본 full task2)\n"
                       "  decompose: 덤프 summary 고정, decompose부터 생성\n"
                       "  verify   : 덤프 summary+decompose 고정, verify부터 생성\n"
                       "  fourth   : 덤프 summary+decompose+verify 고정, fourth만 생성"
                   ))

    # 직접 지정
    p.add_argument("--gen_img_path", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--feedback_text", default=None)
    return p.parse_args()


def resolve_inputs(args) -> tuple[str, str, str | None, dict | None]:
    if args.input_txt is not None:
        payload = load_payload_from_txt(args.input_txt)
        gen_img_path = payload["image_path"]
        prompt = payload["prompt"]
        feedback_text = payload.get("feedback") if args.use_dump_feedback else None
        print(f"[test_model] Parsed from txt:\n  prompt  : {prompt}\n  gen_img : {gen_img_path}")
        return gen_img_path, prompt, feedback_text, payload

    if args.rollout_from is not None:
        raise ValueError("--rollout-from 은 --input-txt 와 함께 사용해야 합니다.")
    if args.gen_img_path is None or args.prompt is None:
        raise ValueError("--input-txt 또는 --gen_img_path + --prompt 를 지정하세요.")
    return args.gen_img_path, args.prompt, args.feedback_text, None


# ── main ──────────────────────────────────────────────────────────────────────

def init_distributed():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl", init_method="env://", world_size=1, rank=0)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    gen_img_path, prompt, feedback_text, payload = resolve_inputs(args)

    init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    print(f"\n[test_model] Loading model from {args.model_path} ...")
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from transformers import AutoModelForCausalLM

    processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    module: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print("[test_model] Model loaded.\n")

    # ── rollout 객체 생성 ─────────────────────────────────────────────────────
    from recipe.image_rl.config import ImageGenerationRolloutConfig
    from verl.workers.rollout.image_unified_rollout import ImageUnifiedRollout

    rollout_cfg = ImageGenerationRolloutConfig(
        cfg_weight=2.0, temperature=1.3, txt_top_p=1.0, img_top_p=1.0,
        prompt_length=1024, response_length=512,
        image_token_num_per_image=576,
    )
    model_config_shim = SimpleNamespace(tokenizer=processor.tokenizer, processor=processor)
    rollout = ImageUnifiedRollout(
        module=module, config=rollout_cfg,
        model_config=model_config_shim, device_mesh=None,
    )
    rollout.is_validate = False
    rollout.image_start_tag = processor.image_start_tag
    rollout.image_end_tag   = processor.image_end_tag
    rollout.image_tag       = processor.image_tag

    n = args.n_rollouts

    # ── DataProto 구성 ────────────────────────────────────────────────────────
    print(f"[test_model] Loading gen image from {gen_img_path} (n_rollouts={n})")
    pil_image = PIL.Image.open(gen_img_path).convert("RGB")
    data = build_dataproto(prompt, pil_image, processor, module, device, n_rollouts=n)

    # ── TASK 2: feedback 생성 ─────────────────────────────────────────────────
    if args.rollout_from is not None:
        # 지정된 stage부터 생성
        input_suffix, feedback_fixed_prefix = build_rollout_from_config(args.rollout_from, payload)
        print(f"\n── TASK 2: {args.rollout_from}부터 생성 (배치={n}) ─────────────")
        generated_texts = generate_from_stage(rollout, data, [input_suffix] * n)
        full_feedbacks = [feedback_fixed_prefix + gen for gen in generated_texts]
        for i, gen in enumerate(generated_texts):
            print(f"[task2][{i}] {gen[:120]}...")
        data.non_tensor_batch["task2_feedback_texts"] = np.array(full_feedbacks, dtype=object)
        print()

    elif feedback_text is not None:
        # 덤프 feedback 그대로 사용 (task2 스킵)
        # _split_text_into_parts가 Fourth를 추출하려면 full 4-part 텍스트가 필요하므로 재구성
        if payload is not None:
            summary     = payload.get("summary", "")
            pred_tuples = payload.get("pred_tuples", "")
            vqa_results = payload.get("vqa_results", "")
            full_feedback = (
                summary + "\n\n"
                + SECOND_HEADER + pred_tuples + "\n\n"
                + THIRD_HEADER + vqa_results + "\n\n"
                + FOURTH_HEADER + feedback_text
            )
        else:
            full_feedback = feedback_text
        data.non_tensor_batch["task2_feedback_texts"] = np.array([full_feedback] * n, dtype=object)
        print(f"\n── TASK 2: 덤프 feedback 사용 (생성 스킵, {n}개 복제) ──────────")
        print(f"[task2] feedback:\n{feedback_text}\n")

    else:
        # 기본 full task2
        print(f"\n── TASK 2: full 생성 (배치={n}) ────────────────────────────────")
        data = rollout._generate_minibatch_text_generation(data)
        for i, fb in enumerate(data.non_tensor_batch["task2_feedback_texts"]):
            print(f"[task2][{i}] {fb[:120]}...")
        print()

    # ── TASK 3: 이미지 재생성 ──────────────────────────────────────────────────
    print(f"── TASK 3: 이미지 재생성 (배치={n}) ─────────────────────────────────")
    data = rollout._generate_minibatch_regen_image_generation(data)

    def to_pil(img) -> PIL.Image.Image:
        if isinstance(img, PIL.Image.Image):
            return img
        return PIL.Image.fromarray(np.array(img).astype(np.uint8))

    # gen.png
    to_pil(pil_image).save(os.path.join(args.output_dir, "gen.png"))

    # feedbacks.txt
    feedback_save_path = os.path.join(args.output_dir, "feedbacks.txt")
    with open(feedback_save_path, "w") as f:
        for i, fb in enumerate(data.non_tensor_batch["task2_feedback_texts"]):
            f.write(f"=== rollout_{i} ===\n{fb}\n\n")
    print(f"  feedbacks.txt → {feedback_save_path}")

    # regen_*.png
    for i, regen_pil in enumerate(data.non_tensor_batch["task3_regen_imgs_pil_list"]):
        out_path = os.path.join(args.output_dir, f"regen_{i}.png")
        to_pil(regen_pil).save(out_path)
        print(f"  regen_{i}.png → {out_path}")

    print(f"\n[test_model] Done. → {args.output_dir}/")


if __name__ == "__main__":
    main()
