"""Manual tester for Step 2 fine-grained reward judges.

This script is focused on the prompts defined under:
`recipe/image_rl/prompts_finegrained.py`
-> `############################ Step 2 Fine-Graine Reward ############################`

Usage examples:
    python test_reward_server_finegrained.py --stage stage1
    python test_reward_server_finegrained.py /path/to/txt.txt
    python test_reward_server_finegrained.py --stage stage3 --show-messages
    python test_reward_server_finegrained.py --stage all --input-json /path/to/payload.json
    python test_reward_server_finegrained.py --stage all --input-txt /path/to/txt.txt
    python test_reward_server_finegrained.py --dump-sample-json

    python /verl/test_reward_server_finegrained.py --input-txt /verl/txt.txt --stage stage1

    # Test TASK3_REGENERATION_FOLLOWED_BY_EDITING_SYSTEM_PROMPT
    # Uses gen.png as SOURCE_IMAGE and regen.png as EDITED_IMAGE
    python /verl/test_reward_server_finegrained.py --input-txt /verl/txt.txt --stage edit

    # Test TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE (VQA evaluator)
    # task1: uses gen.png + GT VQA questions
    # task3: uses regen.png + GT VQA questions
    python /verl/test_reward_server_finegrained.py --input-txt /verl/txt.txt --stage task1
    python /home/work/AGILAB/mllm_reasoning/verl/test_reward_server_finegrained.py --input-txt /home/work/AGILAB/mllm_reasoning/verl/txt.txt --stage task3
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import PIL.Image
from openai import AsyncOpenAI

from recipe.image_rl.prompts_finegrained_simple import (
    PROMPT_TO_SUMMARY_REWARD_SYSTEM_PROMPT,
    SUMMARY_TO_TUPLE_DECOMPOSITION_REWARD_SYSTEM_PROMPT,
    TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE,
    TASK3_REGENERATION_FOLLOWED_BY_EDITING_SYSTEM_PROMPT,
    TUPLE_DECOMPOSITION_TO_VQA_REWARD_SYSTEM_PROMPT,
    VQA_TO_FEEDBACK_REWARD_SYSTEM_PROMPT,
)


SERVER_BASE_URL = os.environ.get("REWARD_SERVER_BASE_URL", "http://10.100.65.6:8007/v1")
API_KEY = os.environ.get("REWARD_SERVER_API_KEY", "EMPTY")
MODEL_PATH = os.environ.get("REWARD_SERVER_MODEL", "Qwen/Qwen3.5-35B-A3B")
REQUEST_TIMEOUT = float(os.environ.get("REWARD_SERVER_TIMEOUT", "300"))


SAMPLE_PAYLOAD: dict[str, str] = {
    "image_path": "/data/verl/ckpts/mllm_reasoning/0227_our_model_our_dataset_task123_group16/rollout/132/rollouter_source_v132/reasonr1/dpgbench/02941/rollout_14/gen.png",
    "prompt": (
        "A tranquil mountain lake scene featuring a small wooden dock extending into "
        "crystal-clear waters. Nearby, a lone kayaker in a bright red life jacket "
        "paddles gently across the surface, creating ripples that shimmer under the "
        "soft sunlight. Towering pine trees line the shoreline, their reflection "
        "mirrored perfectly in the calm water, evoking a sense of peaceful solitude "
        "amidst nature's grandeur."
    ),
    "summary": (
        "A wooden dock extends into a mountain lake. A lone kayaker in a red life "
        "jacket is on the lake near the dock. Pine trees line the shoreline."
    ),
    "pred_tuples": """\
1 | entity - whole (dock)
2 | entity - whole (lake)
3 | entity - whole (kayaker)
4 | entity - whole (life jacket)
5 | entity - whole (pine trees)
6 | relation - spatial (dock, lake, in)
7 | relation - spatial (kayaker, dock, near)
8 | attribute - color (life jacket, red)""",
    "vqa_results": """\
1 | A wooden dock is clearly visible extending from the shore into the water. Answer: Yes
2 | The image shows a calm mountain lake surrounded by trees and mountains. Answer: Yes
3 | A single kayaker is visible on the water near the dock. Answer: Yes
4 | The kayaker is wearing a bright red life jacket. Answer: Yes
5 | Pine trees are visible along the shoreline. Answer: Yes
6 | The dock extends into the lake water, so the relation is satisfied. Answer: Yes
7 | The kayaker is positioned near the dock on the lake surface. Answer: Yes
8 | The visible life jacket on the kayaker is red. Answer: Yes""",
    "feedback": "No need to generate feedback.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual tester for Step 2 fine-grained reward prompts."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Optional txt.txt or JSON payload path.",
    )
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "stage3", "stage4", "edit", "task1", "task3", "all"),
        default="all",
        help=(
            "Which fine-grained reward judge to run. "
            "'edit' tests TASK3_REGENERATION_FOLLOWED_BY_EDITING_SYSTEM_PROMPT. "
            "'task1'/'task3' tests TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE "
            "using GT VQA questions from txt.txt (task1=gen.png, task3=regen.png)."
        ),
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Optional JSON file overriding SAMPLE_PAYLOAD keys.",
    )
    parser.add_argument(
        "--input-txt",
        type=Path,
        help="Rollout report txt.txt path. Fields are auto-extracted into payload.",
    )
    parser.add_argument(
        "--show-messages",
        action="store_true",
        help="Print the exact request messages before sending.",
    )
    parser.add_argument(
        "--dump-sample-json",
        action="store_true",
        help="Print the sample payload JSON and exit.",
    )
    return parser.parse_args()


def image_to_base64(image_path: str) -> str:
    image = PIL.Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def content_from_prompt_with_images(prompt: str, image_urls: list[str]) -> list[dict[str, Any]]:
    parts = prompt.split("<image>")
    if len(parts) - 1 != len(image_urls):
        raise ValueError(
            f"placeholder <image> count({len(parts) - 1}) != image count({len(image_urls)})"
        )

    content: list[dict[str, Any]] = []
    for idx, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if idx < len(image_urls):
            content.append({"type": "image_url", "image_url": {"url": image_urls[idx]}})
    return content


def normalize_line_number_prefix(text: str) -> str:
    if not text:
        return ""

    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    has_prefix = any(re.match(r"^\d+\s*\|\s*", line) for line in raw_lines)

    if has_prefix:
        return "\n".join(raw_lines)

    return "\n".join(f"{idx} | {line}" for idx, line in enumerate(raw_lines, start=1))


def load_payload(input_json: Path | None) -> dict[str, str]:
    payload = dict(SAMPLE_PAYLOAD)
    if input_json is None:
        return payload

    loaded = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{input_json} must contain a JSON object")

    for key, value in loaded.items():
        payload[key] = value

    payload["pred_tuples"] = normalize_line_number_prefix(payload.get("pred_tuples", ""))
    payload["vqa_results"] = normalize_line_number_prefix(payload.get("vqa_results", ""))
    return payload


def _extract_block(text: str, start_pattern: str, end_pattern: str) -> str:
    match = re.search(
        start_pattern + r"(.*?)" + end_pattern,
        text,
        re.DOTALL | re.MULTILINE,
    )
    if not match:
        raise ValueError(f"could not extract block: {start_pattern} ... {end_pattern}")
    return match.group(1).strip()


def _normalize_report_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_payload_from_txt(input_txt: Path) -> dict[str, str]:
    report_text = _normalize_report_text(input_txt.read_text(encoding="utf-8"))

    prompt = _extract_block(
        report_text,
        r"📝 \[PROMPT\]\n",
        r"\n=+\n",
    )

    # Extract gen.png path (TASK 1 INITIAL GEN)
    gen_match = re.search(
        r"\[TASK 1\] INITIAL GEN.*?-\s*Path:\s*(.+?)$",
        report_text,
        re.MULTILINE | re.DOTALL,
    )
    if not gen_match:
        # Fallback to first Path found
        gen_match = re.search(r"^\s*-\s*Path:\s*(.+)$", report_text, re.MULTILINE)
    if not gen_match:
        raise ValueError(f"could not find gen image path in {input_txt}")
    gen_image_path = gen_match.group(1).strip()

    # Extract regen.png path (TASK 3 RE-GENERATION)
    regen_match = re.search(
        r"\[TASK 3\] RE-GENERATION.*?-\s*Path:\s*(.+?)$",
        report_text,
        re.MULTILINE | re.DOTALL,
    )
    regen_image_path = regen_match.group(1).strip() if regen_match else ""

    # Extract feedback from Model Feedback section
    feedback_section_match = re.search(
        r"-\s*Model Feedback:\n(.*?)(?:\n\s*-\s*(?:Judge Alignment|Stage1 Response|Feedback Response))",
        report_text,
        re.DOTALL,
    )
    edit_feedback = ""
    if feedback_section_match:
        model_feedback_text = feedback_section_match.group(1).strip()
        # Extract the Fourth step feedback
        fourth_match = re.search(
            r"Fourth,\s*Generate corrective feedback\.\s*\n(.*)$",
            model_feedback_text,
            re.DOTALL,
        )
        if fourth_match:
            edit_feedback = fourth_match.group(1).strip()

    # Use fallback for model_feedback extraction
    try:
        model_feedback = _extract_block(
            report_text,
            r"^\s*-\s*Model Feedback:\n",
            r"\n\s*-\s*(?:Judge Alignment Gate Response|Stage1 Response|Feedback Response)",
        )
    except ValueError:
        model_feedback = ""

    summary = ""
    pred_tuples = ""
    vqa_results = ""
    feedback = "No need to generate feedback."

    if model_feedback:
        summary_match = re.search(
            r"^\s*(.*?)\n\s*Second,\s*Decompose summarize\s*$",
            model_feedback,
            re.DOTALL | re.MULTILINE,
        )
        tuple_match = re.search(
            r"Second,\s*Decompose summarize\s*\n(.*?)\n\s*Third,\s*Verify that the decomposed elements align with the image\.\s*$",
            model_feedback,
            re.DOTALL | re.MULTILINE,
        )
        vqa_match = re.search(
            r"Third,\s*Verify that the decomposed elements align with the image\.\s*\n(.*?)\n\s*Fourth,\s*Generate corrective feedback\.\s*(.*)$",
            model_feedback,
            re.DOTALL | re.MULTILINE,
        )

        if summary_match:
            summary = summary_match.group(1).strip()
        if tuple_match:
            pred_tuples = tuple_match.group(1).strip()
        if vqa_match:
            vqa_results = vqa_match.group(1).strip()
            feedback = vqa_match.group(2).strip() or "No need to generate feedback."

    # Extract GT VQA Questions from GROUND TRUTH REFERENCE section
    vqa_questions = ""
    gt_vqa_match = re.search(
        r"-\s*VQA Questions:\n(.*?)(?:\n={10,}|$)",
        report_text,
        re.DOTALL,
    )
    if gt_vqa_match:
        vqa_questions = gt_vqa_match.group(1).strip()

    return {
        "image_path": gen_image_path,
        "regen_image_path": regen_image_path,
        "prompt": prompt,
        "summary": summary,
        "pred_tuples": normalize_line_number_prefix(pred_tuples),
        "vqa_results": normalize_line_number_prefix(vqa_results),
        "feedback": feedback,
        "edit_feedback": edit_feedback,
        "vqa_questions": vqa_questions,
    }


def build_stage1_messages(payload: dict[str, str]) -> list[dict[str, Any]]:
    user_content = (
        f"PROMPT:\n{payload.get('prompt', '')}\n\n"
        f"SUMMARY:\n{payload.get('summary', '')}"
    )
    return [
        {"role": "system", "content": PROMPT_TO_SUMMARY_REWARD_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_stage2_messages(payload: dict[str, str]) -> list[dict[str, Any]]:
    user_content = (
        f"SUMMARY:\n{payload.get('summary', '')}\n\n"
        f"PRED_TUPLES:\n{payload.get('pred_tuples', '')}"
    )
    return [
        {"role": "system", "content": SUMMARY_TO_TUPLE_DECOMPOSITION_REWARD_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_stage3_messages(payload: dict[str, str]) -> list[dict[str, Any]]:
    image_path = payload.get("image_path", "")
    if not image_path:
        raise ValueError("stage3 requires `image_path`")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    user_content = (
        "IMAGE:\n<image>\n\n"
        f"PRED_TUPLES:\n{payload.get('pred_tuples', '')}\n\n"
        f"VQA_RESULTS:\n{payload.get('vqa_results', '')}"
    )
    return [
        {"role": "system", "content": TUPLE_DECOMPOSITION_TO_VQA_REWARD_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": content_from_prompt_with_images(
                user_content,
                [image_to_base64(image_path)],
            ),
        },
    ]


def build_stage4_messages(payload: dict[str, str]) -> list[dict[str, Any]]:
    feedback = payload.get("feedback", "") or "No need to generate feedback."
    user_content = (
        f"PRED_TUPLES:\n{payload.get('pred_tuples', '')}\n\n"
        f"VQA_RESULTS:\n{payload.get('vqa_results', '')}\n\n"
        f"FEEDBACK:\n{feedback}"
    )
    return [
        {"role": "system", "content": VQA_TO_FEEDBACK_REWARD_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_vqa_score(response: str) -> tuple[dict[int, bool], float]:
    """Parse VQA Yes/No answers and compute score."""
    ans_re = re.compile(r'(?:step\s+)?(\d+)\s*\|\s*(?:Answer:\s*)?(Yes|No)', re.IGNORECASE)
    idx_to_ans: dict[int, bool] = {}
    for idx_str, yn in ans_re.findall(response):
        idx_to_ans[int(idx_str)] = (yn.strip().lower() == "yes")
    total = len(idx_to_ans)
    score = sum(idx_to_ans.values()) / total if total else 0.0
    return idx_to_ans, score


def build_task1_messages(payload: dict[str, str]) -> list[dict[str, Any]]:
    """Task 1 VQA: gen.png + GT VQA questions → TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE."""
    image_path = payload.get("image_path", "")
    vqa_questions = payload.get("vqa_questions", "")
    if not image_path:
        raise ValueError("task1 requires `image_path` (gen.png)")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not vqa_questions:
        raise ValueError("task1 requires `vqa_questions` (from GT VQA Questions)")

    user_content = f"[IMAGE]:\n<image>\n\n[QUESTIONS]:\n{vqa_questions}"
    return [
        {"role": "system", "content": TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE},
        {
            "role": "user",
            "content": content_from_prompt_with_images(user_content, [image_to_base64(image_path)]),
        },
    ]


def build_task3_messages(payload: dict[str, str]) -> list[dict[str, Any]]:
    """Task 3 VQA: regen.png + GT VQA questions → TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE."""
    image_path = payload.get("regen_image_path", "")
    vqa_questions = payload.get("vqa_questions", "")
    if not image_path:
        raise ValueError("task3 requires `regen_image_path` (regen.png)")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not vqa_questions:
        raise ValueError("task3 requires `vqa_questions` (from GT VQA Questions)")

    user_content = f"[IMAGE]:\n<image>\n\n[QUESTIONS]:\n{vqa_questions}"
    return [
        {"role": "system", "content": TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE},
        {
            "role": "user",
            "content": content_from_prompt_with_images(user_content, [image_to_base64(image_path)]),
        },
    ]


def build_edit_messages(payload: dict[str, str]) -> list[dict[str, Any]]:
    """Build messages for TASK3_REGENERATION_FOLLOWED_BY_EDITING_SYSTEM_PROMPT.

    Uses gen.png as SOURCE_IMAGE and regen.png as EDITED_IMAGE.
    """
    source_image_path = payload.get("image_path", "")
    edited_image_path = payload.get("regen_image_path", "")
    edit_feedback = payload.get("edit_feedback", "") or payload.get("feedback", "")

    if not source_image_path:
        raise ValueError("edit stage requires `image_path` (gen.png as SOURCE_IMAGE)")
    if not Path(source_image_path).exists():
        raise FileNotFoundError(f"source image not found: {source_image_path}")
    if not edited_image_path:
        raise ValueError("edit stage requires `regen_image_path` (regen.png as EDITED_IMAGE)")
    if not Path(edited_image_path).exists():
        raise FileNotFoundError(f"edited image not found: {edited_image_path}")

    # Build user content with two images
    user_content = (
        "SOURCE_IMAGE:\n<image>\n\n"
        f"FEEDBACK:\n{edit_feedback}\n\n"
        "EDITED_IMAGE:\n<image>"
    )

    return [
        {"role": "system", "content": TASK3_REGENERATION_FOLLOWED_BY_EDITING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": content_from_prompt_with_images(
                user_content,
                [image_to_base64(source_image_path), image_to_base64(edited_image_path)],
            ),
        },
    ]


def extract_json(raw_text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


async def send_request(stage: str, messages: list[dict[str, Any]], show_messages: bool, is_vqa: bool = False) -> str | None:
    client = AsyncOpenAI(api_key=API_KEY, base_url=SERVER_BASE_URL)

    if show_messages:
        print(f"\n[{stage}] messages")
        print(json.dumps(messages, ensure_ascii=False, indent=2))

    print("=" * 100)
    print(f"Stage  : {stage}")
    print(f"Server : {SERVER_BASE_URL}")
    print(f"Model  : {MODEL_PATH}")
    print("=" * 100)

    try:
        start = time.time()
        response = await client.chat.completions.create(
            model=MODEL_PATH,
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
            top_p=1.0,
            extra_body={
                "top_k": -1,
                "min_p": 0.0,
                "best_of": 1,
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=REQUEST_TIMEOUT,
        )
        elapsed = time.time() - start
        raw = response.choices[0].message.content or ""

        print(f"\n[{stage}] response ({elapsed:.2f}s)")
        print("-" * 100)
        print(raw)
        print("-" * 100)

        if is_vqa:
            idx_to_ans, score = _parse_vqa_score(raw)
            yes_count = sum(idx_to_ans.values())
            total = len(idx_to_ans)
            print(f"\n[{stage}] VQA parsed answers: {idx_to_ans}")
            print(f"[{stage}] Score: {yes_count}/{total} = {score:.4f}")
        else:
            parsed = extract_json(raw)
            if parsed is None:
                print(f"[{stage}] JSON parse failed")
            else:
                print(f"[{stage}] parsed")
                print(json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True))

        return raw
    except Exception as exc:
        import traceback

        print(f"\n[{stage}] ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


async def main() -> None:
    args = parse_args()
    if args.dump_sample_json:
        print(json.dumps(SAMPLE_PAYLOAD, ensure_ascii=False, indent=2))
        return

    input_txt = args.input_txt
    input_json = args.input_json

    if args.input_path is not None:
        suffix = args.input_path.suffix.lower()
        if suffix == ".json":
            input_json = args.input_path
        else:
            input_txt = args.input_path

    if input_txt is not None:
        payload = load_payload_from_txt(input_txt)
    else:
        payload = load_payload(input_json)
        payload["pred_tuples"] = normalize_line_number_prefix(payload.get("pred_tuples", ""))
        payload["vqa_results"] = normalize_line_number_prefix(payload.get("vqa_results", ""))

    builders = {
        "stage1": build_stage1_messages,
        "stage2": build_stage2_messages,
        "stage3": build_stage3_messages,
        "stage4": build_stage4_messages,
        "edit":   build_edit_messages,
        "task1":  build_task1_messages,
        "task3":  build_task3_messages,
    }
    vqa_stages = {"task1", "task3"}

    # For "all", exclude "edit"/"task1"/"task3" since they require different inputs
    if args.stage == "all":
        stages = ["stage1", "stage2", "stage3", "stage4"]
    else:
        stages = [args.stage]
    for stage in stages:
        messages = builders[stage](payload)
        await send_request(stage, messages, args.show_messages, is_vqa=(stage in vqa_stages))


if __name__ == "__main__":
    asyncio.run(main())
