#!/usr/bin/env python3
"""
Offline reward evaluation script for TIIF benchmark dumps.

Computes fine-grained rewards (Task 1, 2, 3) on pre-generated TIIF data
using the same reward logic as reward_function_fine_grained.py.

Supports iterative correction chains via --step:
    step 0: curr=gen                          → Task 1 only
    step 1: prev=gen,          curr=correction_0  → Task 1,2,3
    step 2: prev=correction_0, curr=correction_1  → Task 1,2,3
    ...

TIIF folder structure:
    tiif/
    ├── long/
    │   ├── gen/{category}/{data_idx}/{metadata.jsonl, samples/}
    │   ├── correction_0/{category}/{data_idx}/{metadata.jsonl, rationales.json, samples/}
    │   └── ...
    └── short/
        └── (same structure)

Usage:
    # Step 0: gen only (Task 1), long description
    python -m bench.reward_tiif \
        --data_dir /data/mllm/data/for_down/tiif \
        --vqa_json /path/to/tiif_dsg_final.json \
        --desc long \
        --output_dir /data/mllm/data/for_down/tiif/reward_results \
        --step 0 --tasks 1

    # Step 1: gen → correction_0, both long and short
    python -m bench.reward_tiif \
        --data_dir /data/mllm/data/for_down/tiif \
        --vqa_json /path/to/tiif_dsg_final.json \
        --desc long short \
        --output_dir /data/mllm/data/for_down/tiif/reward_results \
        --step 1 --tasks 1 2 3

    # All steps at once
    python -m bench.reward_tiif \
        --data_dir /data/mllm/data/for_down/tiif \
        --vqa_json /path/to/tiif_dsg_final.json \
        --desc long short \
        --output_dir /data/mllm/data/for_down/tiif/reward_results \
        --step 0 1 2 --tasks 1 2 3
"""

import os
import sys
import json
import argparse
import asyncio
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple

import PIL.Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recipe.image_rl.reward_function_fine_grained import (
    compute_score_single_async,
    get_response,
    get_messages,
    get_messages_task2_stage1,
    get_messages_task2_stage2,
    get_messages_task2_stage3,
    get_messages_task2_stage4,
    get_messages_task3_edit,
    image_evaluator_parser,
    _parse_vqa_reward_score,
    _parse_json_score,
    _normalize_tuple_lines,
    _add_index_to_vqa_lines,
    convert_gen_img_to_base64,
    RM_VLM_MODEL_PATH,
    RM_LLM_MODEL_PATH,
)
from recipe.image_rl.utils import FormattingEvaluatorV2


# ──────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────

def load_vqa_questions(vqa_json_path: str) -> Dict[str, dict]:
    """Load VQA questions and build prompt -> entry mapping."""
    with open(vqa_json_path) as f:
        data = json.load(f)
    return {item["prompt"]: item for item in data}


def step_to_dirs(step: int) -> Tuple[Optional[str], str]:
    """Convert step number to (prev_subdir, curr_subdir) names.

    step 0: prev=None,             curr="gen"            → Task 1 only
    step 1: prev="gen",            curr="correction_0"   → Task 1,2,3
    step 2: prev="correction_0",   curr="correction_1"   → Task 1,2,3
    step N: prev=correction_{N-2}, curr=correction_{N-1}  → Task 1,2,3
    """
    if step == 0:
        return None, "gen"
    elif step == 1:
        return "gen", "correction_0"
    else:
        return f"correction_{step - 2}", f"correction_{step - 1}"


def discover_samples(desc_dir: str, curr_subdir: str) -> List[Tuple[str, str]]:
    """Discover all (category, data_idx) pairs under desc_dir/curr_subdir/.

    Returns list of (category, data_idx) tuples, sorted.
    """
    curr_path = os.path.join(desc_dir, curr_subdir)
    samples = []
    for category in sorted(os.listdir(curr_path)):
        cat_dir = os.path.join(curr_path, category)
        if not os.path.isdir(cat_dir):
            continue
        for data_idx in sorted(os.listdir(cat_dir)):
            idx_dir = os.path.join(cat_dir, data_idx)
            if not os.path.isdir(idx_dir):
                continue
            samples.append((category, data_idx))
    return samples


def load_sample(
    desc_dir: str,
    category: str,
    data_idx: str,
    prompt_to_vqa: Dict[str, dict],
    prev_subdir: Optional[str] = None,
    curr_subdir: str = "gen",
) -> dict:
    """Load a single sample's data for TIIF.

    TIIF path: desc_dir/{subdir}/{category}/{data_idx}/

    Args:
        desc_dir: e.g., /data/mllm/data/for_down/tiif/long
        category: e.g., "2d_spatial_relation"
        data_idx: e.g., "00000"
        prev_subdir: previous step's subdirectory. None for step 0.
        curr_subdir: current step's subdirectory.

    Returns dict with all fields needed for evaluation.
    """
    folder_id = f"{category}/{data_idx}"
    curr_dir = os.path.join(desc_dir, curr_subdir, category, data_idx)

    # Metadata — always read from curr (fallback to gen/ for metadata)
    meta_path = os.path.join(curr_dir, "metadata.jsonl")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(desc_dir, "gen", category, data_idx, "metadata.jsonl")
    with open(meta_path) as f:
        metadata = json.loads(f.readline())
    prompt = metadata["prompt"]

    # VQA question from DSG JSON (keyed by prompt)
    vqa_entry = prompt_to_vqa.get(prompt, {})
    vqa_question = vqa_entry.get("question", "")
    summarize = vqa_entry.get("tuple", "")
    feedback_tuple = vqa_entry.get("tuple", "")

    # Current images (4 per sample)
    curr_images = []
    curr_image_paths = []
    curr_samples_dir = os.path.join(curr_dir, "samples")
    for i in range(4):
        img_path = os.path.join(curr_samples_dir, f"{i:05d}.png")
        if os.path.exists(img_path):
            curr_images.append(PIL.Image.open(img_path).convert("RGB"))
            curr_image_paths.append(img_path)

    # Previous step images
    prev_images = []
    prev_image_paths = []
    if prev_subdir is not None:
        prev_dir = os.path.join(desc_dir, prev_subdir, category, data_idx)
        prev_samples_dir = os.path.join(prev_dir, "samples")
        for i in range(4):
            img_path = os.path.join(prev_samples_dir, f"{i:05d}.png")
            if os.path.exists(img_path):
                prev_images.append(PIL.Image.open(img_path).convert("RGB"))
                prev_image_paths.append(img_path)

    # Rationales from curr
    rationales = None
    rat_path = os.path.join(curr_dir, "rationales.json")
    if os.path.exists(rat_path):
        with open(rat_path) as f:
            rationales = json.load(f)

    return {
        "folder_id": folder_id,
        "category": category,
        "data_idx": data_idx,
        "prompt": prompt,
        "tag": metadata.get("type", category),  # TIIF uses 'type' field as tag
        "metadata": metadata,
        "curr_images": curr_images,
        "curr_image_paths": curr_image_paths,
        "prev_images": prev_images if prev_images else None,
        "prev_image_paths": prev_image_paths if prev_image_paths else None,
        "rationales": rationales,
        "vqa_question": vqa_question,
        "vqa_entry": vqa_entry,
        "summarize": summarize,
        "feedback_tuple": feedback_tuple,
    }


# ──────────────────────────────────────────────────────────────
# Per-image reward computation
# ──────────────────────────────────────────────────────────────

async def eval_task1_single(sample: dict, image_idx: int) -> dict:
    """Task 1: VQA reward on curr image."""
    curr_img = sample["curr_images"][image_idx]
    prompt = sample["prompt"]
    vqa_question = sample["vqa_question"]

    result = await compute_score_single_async(
        prompt=prompt,
        gen_img=curr_img,
        feedback_text=None,
        regen_img=None,
        ground_truth_img=None,
        summarize=sample["summarize"],
        feedback_tuple=sample["feedback_tuple"],
        predicted_summarize=None,
        predicted_tuple=None,
        predicted_answer=None,
        predicted_feedback=None,
        vqa_question=vqa_question,
        extra_info={},
        task_id=1,
    )
    return {
        "folder_id": sample["folder_id"],
        "category": sample["category"],
        "image_index": image_idx,
        "task": 1,
        "inputs": {
            "curr_image_path": sample["curr_image_paths"][image_idx],
            "prompt": prompt,
            "vqa_question": vqa_question,
        },
        **result,
    }


async def eval_task2_single(sample: dict, image_idx: int) -> dict:
    """Task 2: Reasoning quality reward for a specific image's rationale."""
    rationales = sample["rationales"]
    if rationales is None:
        return {
            "folder_id": sample["folder_id"],
            "category": sample["category"],
            "image_index": image_idx,
            "task": 2,
            "score": 0.0,
            "reward_extra_info": {"error": "no rationales found"},
        }

    # Find rationale for this image_index
    rationale = None
    for r in rationales:
        if r["image_index"] == image_idx:
            rationale = r
            break

    if rationale is None:
        return {
            "folder_id": sample["folder_id"],
            "category": sample["category"],
            "image_index": image_idx,
            "task": 2,
            "score": 0.0,
            "reward_extra_info": {"error": f"no rationale for image_index {image_idx}"},
        }

    # prev image = what the model looked at when generating this rationale
    prev_img = sample["prev_images"][image_idx] if sample["prev_images"] else None
    feedback_text = rationale["original_answer"]
    predicted_answer_str = "\n".join(rationale["pred_answer"]) if isinstance(rationale["pred_answer"], list) else rationale["pred_answer"]

    result = await compute_score_single_async(
        prompt=sample["prompt"],
        gen_img=prev_img,
        feedback_text=feedback_text,
        regen_img=None,
        ground_truth_img=None,
        summarize=sample["summarize"],
        feedback_tuple=sample["feedback_tuple"],
        predicted_summarize=rationale["pred_summarize"],
        predicted_tuple=rationale["pred_tuple"],
        predicted_answer=predicted_answer_str,
        predicted_feedback=rationale["pred_feedback"],
        vqa_question=sample["vqa_question"],
        extra_info={},
        task_id=2,
    )
    return {
        "folder_id": sample["folder_id"],
        "category": sample["category"],
        "image_index": image_idx,
        "task": 2,
        "inputs": {
            "prev_image_path": sample["prev_image_paths"][image_idx] if sample["prev_image_paths"] else None,
            "prompt": sample["prompt"],
            "vqa_question": sample["vqa_question"],
            "feedback_text": feedback_text,
            "predicted_summarize": rationale["pred_summarize"],
            "predicted_tuple": rationale["pred_tuple"],
            "predicted_answer": predicted_answer_str,
            "predicted_feedback": rationale["pred_feedback"],
            "summarize": sample["summarize"],
            "feedback_tuple": sample["feedback_tuple"],
        },
        **result,
    }


async def eval_task3_single(sample: dict, image_idx: int) -> dict:
    """Task 3: Correction quality (prev_img + feedback -> curr_img)."""
    if sample["prev_images"] is None or sample["rationales"] is None:
        return {
            "folder_id": sample["folder_id"],
            "category": sample["category"],
            "image_index": image_idx,
            "task": 3,
            "score": 0.0,
            "reward_extra_info": {"error": "no prev images or rationales"},
        }

    # Find rationale for this image
    rationale = None
    for r in sample["rationales"]:
        if r["image_index"] == image_idx:
            rationale = r
            break

    if rationale is None:
        return {
            "folder_id": sample["folder_id"],
            "category": sample["category"],
            "image_index": image_idx,
            "task": 3,
            "score": 0.0,
            "reward_extra_info": {"error": f"no rationale for image_index {image_idx}"},
        }

    prev_img = sample["prev_images"][image_idx] if image_idx < len(sample["prev_images"]) else None
    curr_img = sample["curr_images"][image_idx] if image_idx < len(sample["curr_images"]) else None

    if prev_img is None or curr_img is None:
        return {
            "folder_id": sample["folder_id"],
            "category": sample["category"],
            "image_index": image_idx,
            "task": 3,
            "score": 0.0,
            "reward_extra_info": {"error": "missing prev or curr image"},
        }

    feedback_text = rationale["original_answer"]
    predicted_answer_str = "\n".join(rationale["pred_answer"]) if isinstance(rationale["pred_answer"], list) else rationale["pred_answer"]

    result = await compute_score_single_async(
        prompt=sample["prompt"],
        gen_img=prev_img,
        feedback_text=feedback_text,
        regen_img=curr_img,
        ground_truth_img=None,
        summarize=sample["summarize"],
        feedback_tuple=sample["feedback_tuple"],
        predicted_summarize=rationale["pred_summarize"],
        predicted_tuple=rationale["pred_tuple"],
        predicted_answer=predicted_answer_str,
        predicted_feedback=rationale["pred_feedback"],
        vqa_question=sample["vqa_question"],
        extra_info={},
        task_id=3,
    )
    return {
        "folder_id": sample["folder_id"],
        "category": sample["category"],
        "image_index": image_idx,
        "task": 3,
        "inputs": {
            "prev_image_path": sample["prev_image_paths"][image_idx],
            "curr_image_path": sample["curr_image_paths"][image_idx],
            "prompt": sample["prompt"],
            "vqa_question": sample["vqa_question"],
            "feedback_text": feedback_text,
            "predicted_summarize": rationale["pred_summarize"],
            "predicted_tuple": rationale["pred_tuple"],
            "predicted_answer": predicted_answer_str,
            "predicted_feedback": rationale["pred_feedback"],
        },
        **result,
    }


# ──────────────────────────────────────────────────────────────
# Batch orchestration
# ──────────────────────────────────────────────────────────────

async def eval_sample(sample: dict, tasks: List[int], semaphore: asyncio.Semaphore) -> List[dict]:
    """Evaluate all tasks for one sample (all 4 images)."""
    results = []
    num_images = len(sample["curr_images"])

    for image_idx in range(num_images):
        coros = []
        for task_id in tasks:
            if task_id == 1:
                coros.append(eval_task1_single(sample, image_idx))
            elif task_id == 2:
                coros.append(eval_task2_single(sample, image_idx))
            elif task_id == 3:
                coros.append(eval_task3_single(sample, image_idx))

        for coro in coros:
            async with semaphore:
                try:
                    result = await coro
                    results.append(result)
                except Exception as e:
                    print(f"[ERROR] folder={sample['folder_id']} image={image_idx}: {e}")
                    results.append({
                        "folder_id": sample["folder_id"],
                        "category": sample.get("category", ""),
                        "image_index": image_idx,
                        "task": -1,
                        "score": 0.0,
                        "reward_extra_info": {"error": str(e)},
                    })

    return results


async def run_evaluation(
    desc_dir: str,
    desc: str,
    vqa_json_path: str,
    output_dir: str,
    tasks: List[int],
    max_concurrent: int = 64,
    sample_ids: Optional[List[str]] = None,
    step: int = 0,
):
    """Main evaluation loop for a single desc (long or short).

    Args:
        desc_dir: path to desc directory, e.g., /data/.../tiif/long
        desc: description type, "long" or "short"
        step: correction iteration step.
    """
    prev_subdir, curr_subdir = step_to_dirs(step)

    # Step 0: no prev images / rationales → Task 1 only
    if step == 0:
        valid_tasks = [t for t in tasks if t == 1]
        skipped = [t for t in tasks if t != 1]
        if skipped:
            print(f"[INFO] Step 0: skipping task {skipped} — no prev images/rationales")
    else:
        valid_tasks = list(tasks)
    tasks = valid_tasks

    if not tasks:
        print(f"[WARN] Step {step}: no valid tasks to run, skipping")
        return None

    print(f"\n{'='*60}")
    print(f"[{desc}] Step {step}: prev={prev_subdir}, curr={curr_subdir}, tasks={tasks}")
    print(f"{'='*60}")

    # Validate directories exist
    curr_path = os.path.join(desc_dir, curr_subdir)
    if not os.path.isdir(curr_path):
        print(f"[ERROR] Current directory not found: {curr_path}")
        return None
    if prev_subdir is not None:
        prev_path = os.path.join(desc_dir, prev_subdir)
        if not os.path.isdir(prev_path):
            print(f"[ERROR] Previous directory not found: {prev_path}")
            return None

    os.makedirs(output_dir, exist_ok=True)

    # Load VQA questions
    print(f"Loading VQA questions from {vqa_json_path}...")
    prompt_to_vqa = load_vqa_questions(vqa_json_path)
    print(f"  Loaded {len(prompt_to_vqa)} VQA entries")

    # Discover sample folders: list of (category, data_idx)
    all_samples = discover_samples(desc_dir, curr_subdir)
    if sample_ids is not None:
        sample_id_set = set(sample_ids)
        all_samples = [(cat, idx) for cat, idx in all_samples if f"{cat}/{idx}" in sample_id_set]
    print(f"Found {len(all_samples)} samples across categories, evaluating tasks: {tasks}")

    # Count per category
    cat_counts = defaultdict(int)
    for cat, _ in all_samples:
        cat_counts[cat] += 1
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt} samples")

    semaphore = asyncio.Semaphore(max_concurrent)
    all_results = []
    start_time = time.time()

    # Process in batches for memory efficiency
    BATCH_SIZE = 50
    for batch_start in range(0, len(all_samples), BATCH_SIZE):
        batch_sample_ids = all_samples[batch_start:batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(all_samples))
        print(f"\nProcessing samples {batch_start}-{batch_end-1} / {len(all_samples)}...")

        # Load samples
        samples = []
        for category, data_idx in batch_sample_ids:
            try:
                sample = load_sample(
                    desc_dir, category, data_idx, prompt_to_vqa,
                    prev_subdir=prev_subdir, curr_subdir=curr_subdir,
                )
                samples.append(sample)
            except Exception as e:
                print(f"[WARN] Failed to load sample {category}/{data_idx}: {e}")

        # Run evaluation
        batch_tasks = [eval_sample(s, tasks, semaphore) for s in samples]
        batch_results_nested = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for sample, res in zip(samples, batch_results_nested):
            if isinstance(res, Exception):
                print(f"[ERROR] Batch exception for {sample['folder_id']}: {res}")
            else:
                for r in res:
                    r["tag"] = sample["tag"]
                    r["prompt"] = sample["prompt"]
                all_results.extend(res)

        # Save per-sample results incrementally
        for sample in samples:
            sample_results = [r for r in all_results if r["folder_id"] == sample["folder_id"]]
            if sample_results:
                # Use category__data_idx as filename to avoid nested dirs
                safe_name = sample["folder_id"].replace("/", "__")
                sample_out = os.path.join(output_dir, "per_sample", safe_name + ".json")
                os.makedirs(os.path.dirname(sample_out), exist_ok=True)
                with open(sample_out, "w") as f:
                    json.dump({
                        "folder_id": sample["folder_id"],
                        "category": sample["category"],
                        "data_idx": sample["data_idx"],
                        "prompt": sample["prompt"],
                        "tag": sample["tag"],
                        "results": sample_results,
                    }, f, indent=2, default=str)

        # Close images to free memory
        for sample in samples:
            for img in sample["curr_images"]:
                img.close()
            if sample["prev_images"]:
                for img in sample["prev_images"]:
                    img.close()

        elapsed = time.time() - start_time
        done = batch_end
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(all_samples) - done) / rate if rate > 0 else 0
        print(f"  Progress: {done}/{len(all_samples)} samples, {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    # ── Compute summary ──
    print("\nComputing summary...")
    summary = compute_summary(all_results, tasks)
    summary["step"] = step
    summary["desc"] = desc
    summary["prev_subdir"] = prev_subdir
    summary["curr_subdir"] = curr_subdir

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save all results
    all_results_path = os.path.join(output_dir, "all_results.json")
    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_time = time.time() - start_time
    print(f"\nDone [{desc}] step {step}! Total time: {total_time:.1f}s")
    print(f"  Summary: {summary_path}")
    print(f"  All results: {all_results_path}")
    print(f"  Per-sample: {os.path.join(output_dir, 'per_sample/')}")

    # Print summary to stdout
    print_summary(summary)

    return summary


def compute_summary(all_results: List[dict], tasks: List[int]) -> dict:
    """Aggregate results into a summary."""
    summary = {
        "total_samples": 0,
        "total_evaluations": len(all_results),
    }

    # Collect unique folder_ids
    folder_ids = set(r["folder_id"] for r in all_results)
    summary["total_samples"] = len(folder_ids)

    # Group by task
    for task_id in tasks:
        task_results = [r for r in all_results if r.get("task") == task_id]
        if not task_results:
            continue

        scores = [r["score"] for r in task_results if r.get("score") is not None]
        task_summary = {
            "count": len(task_results),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "std_score": float(np.std(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
            "median_score": float(np.median(scores)) if scores else 0.0,
        }

        # Group by tag (category)
        tag_scores = defaultdict(list)
        for r in task_results:
            tag = r.get("tag", "unknown")
            if r.get("score") is not None:
                tag_scores[tag].append(r["score"])

        tag_summary = {}
        for tag, t_scores in sorted(tag_scores.items()):
            tag_summary[tag] = {
                "count": len(t_scores),
                "mean_score": float(np.mean(t_scores)),
                "std_score": float(np.std(t_scores)),
            }
        task_summary["by_tag"] = tag_summary

        # Task-specific extra info aggregation
        if task_id == 1:
            vqa_rewards = [r["reward_extra_info"].get("task1_vqa_reward", 0.0) for r in task_results if "reward_extra_info" in r]
            task_summary["mean_vqa_reward"] = float(np.mean(vqa_rewards)) if vqa_rewards else 0.0

        elif task_id == 2:
            for key in ["task2_rule_based_format_reward", "task2_rule_based_decompose_reward",
                        "task2_vlm_reward", "task2_prompt_to_summary_reward",
                        "task2_summary_to_tuple_reward", "task2_tuple_to_vqa_reward",
                        "task2_vqa_to_feedback_reward"]:
                vals = [r["reward_extra_info"].get(key, 0.0) for r in task_results if "reward_extra_info" in r]
                task_summary[f"mean_{key}"] = float(np.mean(vals)) if vals else 0.0

        elif task_id == 3:
            edited = [r for r in task_results if r.get("score") != -100]
            not_edited = [r for r in task_results if r.get("score") == -100]

            task_summary["edited_count"] = len(edited)
            task_summary["not_edited_count"] = len(not_edited)
            task_summary["edit_ratio"] = len(edited) / len(task_results) if task_results else 0.0

            edited_scores = [r["score"] for r in edited if r.get("score") is not None]
            task_summary["edited_mean_score"] = float(np.mean(edited_scores)) if edited_scores else 0.0
            task_summary["edited_std_score"] = float(np.std(edited_scores)) if edited_scores else 0.0
            task_summary["edited_median_score"] = float(np.median(edited_scores)) if edited_scores else 0.0

            for key in ["task3_vqa_reward", "task3_edit_reward"]:
                vals = [r["reward_extra_info"].get(key, 0.0) for r in edited
                        if "reward_extra_info" in r]
                task_summary[f"mean_{key}"] = float(np.mean(vals)) if vals else 0.0

            # Per-tag edit breakdown
            tag_edit_breakdown = defaultdict(lambda: {"edited": 0, "not_edited": 0, "edited_scores": []})
            for r in task_results:
                tag = r.get("tag", "unknown")
                if r.get("score") == -100:
                    tag_edit_breakdown[tag]["not_edited"] += 1
                else:
                    tag_edit_breakdown[tag]["edited"] += 1
                    if r.get("score") is not None:
                        tag_edit_breakdown[tag]["edited_scores"].append(r["score"])

            tag_edit_summary = {}
            for tag, info in sorted(tag_edit_breakdown.items()):
                total = info["edited"] + info["not_edited"]
                tag_edit_summary[tag] = {
                    "edited": info["edited"],
                    "not_edited": info["not_edited"],
                    "total": total,
                    "edit_ratio": info["edited"] / total if total > 0 else 0.0,
                    "edited_mean_score": float(np.mean(info["edited_scores"])) if info["edited_scores"] else 0.0,
                }
            task_summary["edit_breakdown_by_tag"] = tag_edit_summary

        summary[f"task{task_id}"] = task_summary

    # ── Cross-task analysis: Task 1 × Task 3 ──
    task1_results = [r for r in all_results if r.get("task") == 1]
    task3_results = [r for r in all_results if r.get("task") == 3]

    if task1_results and task3_results:
        t1_lookup = {(r["folder_id"], r["image_index"]): r for r in task1_results}
        t3_lookup = {(r["folder_id"], r["image_index"]): r for r in task3_results}

        common_keys = set(t1_lookup.keys()) & set(t3_lookup.keys())

        unnecessary_edits = []
        missed_edits = []

        for key in sorted(common_keys):
            t1 = t1_lookup[key]
            t3 = t3_lookup[key]
            t1_score = t1.get("score", 0)
            t3_score = t3.get("score", 0)

            if t1_score == 1.0 and t3_score != -100:
                unnecessary_edits.append({
                    "folder_id": key[0],
                    "image_index": key[1],
                    "tag": t1.get("tag", "unknown"),
                    "task1_score": t1_score,
                    "task3_score": t3_score,
                })
            elif t1_score < 1.0 and t3_score == -100:
                missed_edits.append({
                    "folder_id": key[0],
                    "image_index": key[1],
                    "tag": t1.get("tag", "unknown"),
                    "task1_score": t1_score,
                })

        def _agg_by_tag(items):
            by_tag = defaultdict(list)
            for item in items:
                by_tag[item["tag"]].append(item)
            return {tag: len(lst) for tag, lst in sorted(by_tag.items())}

        cross_analysis = {
            "total_paired": len(common_keys),
            "unnecessary_edit": {
                "description": "Task1 all-yes (score=1.0) but model still edited (task3 != -100)",
                "count": len(unnecessary_edits),
                "ratio": len(unnecessary_edits) / len(common_keys) if common_keys else 0.0,
                "by_tag": _agg_by_tag(unnecessary_edits),
                "samples": unnecessary_edits,
            },
            "missed_edit": {
                "description": "Task1 has-no (score<1.0) but model did not edit (task3 == -100)",
                "count": len(missed_edits),
                "ratio": len(missed_edits) / len(common_keys) if common_keys else 0.0,
                "by_tag": _agg_by_tag(missed_edits),
                "samples": missed_edits,
            },
        }
        summary["cross_task_analysis"] = cross_analysis

    return summary


def print_summary(summary: dict):
    """Pretty-print the summary."""
    print("\n" + "=" * 60)
    print("REWARD EVALUATION SUMMARY")
    print("=" * 60)
    if "step" in summary:
        prev = summary.get('prev_subdir') or '(none)'
        curr = summary.get('curr_subdir', '?')
        desc = summary.get('desc', '?')
        print(f"[{desc}] Step {summary['step']}: prev={prev}, curr={curr}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Total evaluations: {summary['total_evaluations']}")

    for task_id in [1, 2, 3]:
        key = f"task{task_id}"
        if key not in summary:
            continue
        ts = summary[key]
        print(f"\n── Task {task_id} ──")
        print(f"  Count: {ts['count']}")
        print(f"  Score: {ts['mean_score']:.4f} +/- {ts['std_score']:.4f} (median={ts['median_score']:.4f}, min={ts['min_score']:.4f}, max={ts['max_score']:.4f})")

        if task_id == 1:
            print(f"  Mean VQA reward: {ts.get('mean_vqa_reward', 0):.4f}")

        elif task_id == 2:
            print(f"  Mean format reward: {ts.get('mean_task2_rule_based_format_reward', 0):.4f}")
            print(f"  Mean decompose reward: {ts.get('mean_task2_rule_based_decompose_reward', 0):.4f}")
            print(f"  Mean VLM reward: {ts.get('mean_task2_vlm_reward', 0):.4f}")
            print(f"    Stage 1 (prompt->summary):  {ts.get('mean_task2_prompt_to_summary_reward', 0):.4f}")
            print(f"    Stage 2 (summary->tuple):   {ts.get('mean_task2_summary_to_tuple_reward', 0):.4f}")
            print(f"    Stage 3 (tuple->vqa):        {ts.get('mean_task2_tuple_to_vqa_reward', 0):.4f}")
            print(f"    Stage 4 (vqa->feedback):     {ts.get('mean_task2_vqa_to_feedback_reward', 0):.4f}")

        elif task_id == 3:
            print(f"  Edited: {ts.get('edited_count', 0)} | Not edited: {ts.get('not_edited_count', 0)} | Edit ratio: {ts.get('edit_ratio', 0):.4f}")
            print(f"  Edited-only mean score: {ts.get('edited_mean_score', 0):.4f} +/- {ts.get('edited_std_score', 0):.4f} (median={ts.get('edited_median_score', 0):.4f})")
            print(f"  Mean VQA reward (edited): {ts.get('mean_task3_vqa_reward', 0):.4f}")
            print(f"  Mean edit reward (edited): {ts.get('mean_task3_edit_reward', 0):.4f}")
            if ts.get("edit_breakdown_by_tag"):
                print(f"  Edit breakdown by tag:")
                for tag, info in ts["edit_breakdown_by_tag"].items():
                    print(f"    {tag:30s}: edited={info['edited']:>4d} | not_edited={info['not_edited']:>4d} | ratio={info['edit_ratio']:.4f} | edited_mean={info['edited_mean_score']:.4f}")

        if ts.get("by_tag") and task_id != 3:
            print(f"  By category:")
            for tag, tag_data in ts["by_tag"].items():
                print(f"    {tag:30s}: {tag_data['mean_score']:.4f} +/- {tag_data['std_score']:.4f} (n={tag_data['count']})")

    # Cross-task analysis
    cross = summary.get("cross_task_analysis")
    if cross:
        print(f"\n── Cross-Task Analysis (Task1 x Task3) ──")
        print(f"  Total paired: {cross['total_paired']}")

        ue = cross["unnecessary_edit"]
        print(f"\n  Unnecessary edits (Task1=1.0 but still edited):")
        print(f"    Count: {ue['count']} / {cross['total_paired']} ({ue['ratio']:.4f})")
        if ue["by_tag"]:
            for tag, cnt in ue["by_tag"].items():
                print(f"      {tag:30s}: {cnt}")

        me = cross["missed_edit"]
        print(f"\n  Missed edits (Task1<1.0 but not edited):")
        print(f"    Count: {me['count']} / {cross['total_paired']} ({me['ratio']:.4f})")
        if me["by_tag"]:
            for tag, cnt in me["by_tag"].items():
                print(f"      {tag:30s}: {cnt}")

    print("=" * 60)


def print_combined_summary(all_summaries: Dict[str, dict]):
    """Print a cross-step, cross-desc comparison table."""
    print("\n" + "=" * 70)
    print("CROSS-STEP / CROSS-DESC COMPARISON")
    print("=" * 70)

    # Group by desc
    by_desc = defaultdict(dict)
    for key, summary in all_summaries.items():
        desc = summary.get("desc", "unknown")
        step = summary.get("step", "?")
        by_desc[desc][f"step{step}"] = summary

    for desc in sorted(by_desc.keys()):
        steps_dict = by_desc[desc]
        steps = sorted(steps_dict.keys())

        print(f"\n{'─'*70}")
        print(f"  Description: {desc}")
        print(f"{'─'*70}")

        for task_id in [1, 2, 3]:
            key = f"task{task_id}"
            has_task = any(key in steps_dict[s] for s in steps)
            if not has_task:
                continue

            print(f"\n  ── Task {task_id} ──")
            print(f"  {'Step':<30s} {'Mean Score':>12s} {'Std':>10s} {'Count':>8s}")
            print(f"  {'-'*60}")
            for s in steps:
                ts = steps_dict[s].get(key, {})
                if not ts:
                    continue
                prev_sub = steps_dict[s].get("prev_subdir") or "(none)"
                curr_sub = steps_dict[s].get("curr_subdir", "?")
                label = f"{prev_sub}->{curr_sub}"
                print(f"  {label:<30s} {ts.get('mean_score', 0):>12.4f} {ts.get('std_score', 0):>10.4f} {ts.get('count', 0):>8d}")

    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Offline reward evaluation for TIIF benchmark dumps")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to TIIF data dir (contains long/, short/)")
    parser.add_argument("--vqa_json", type=str, required=True,
                        help="Path to VQA question JSON (tiif_dsg_final.json)")
    parser.add_argument("--desc", type=str, nargs="+", default=["long"],
                        choices=["long", "short"],
                        help="Description types to evaluate (default: long)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--step", type=int, nargs="+", default=[0],
                        help="Correction step(s). 0=gen only, 1=gen->correction_0, "
                             "2=correction_0->correction_1, etc.")
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3],
                        choices=[1, 2, 3], help="Which tasks to evaluate (default: 1 2 3)")
    parser.add_argument("--max_concurrent", type=int, default=64,
                        help="Max concurrent API requests (default: 64)")
    parser.add_argument("--sample_ids", type=str, nargs="*", default=None,
                        help="Specific sample IDs to evaluate (format: category/data_idx)")
    parser.add_argument("--vlm_base_urls", type=str, nargs="*", default=None,
                        help="Override VLM server URLs")
    parser.add_argument("--rm_vlm_model_path", type=str, default=None,
                        help="Override RM VLM model name")
    return parser.parse_args()


def main():
    args = parse_args()

    import recipe.image_rl.reward_function_fine_grained as rf

    # Override VLM base URLs if provided
    if args.vlm_base_urls:
        rf.VLM_BASE_URLS = args.vlm_base_urls
        rf.vlm_client_manager = rf.ClientManager(rf.VLM_BASE_URLS, name="VLM")
        rf._rm_slot_queues = {}
        print(f"Using VLM servers: {args.vlm_base_urls}")

    # Override model name if provided
    if args.rm_vlm_model_path:
        rf.RM_VLM_MODEL_PATH = args.rm_vlm_model_path
        print(f"Using VLM model: {args.rm_vlm_model_path}")

    all_summaries = {}
    for desc in args.desc:
        desc_dir = os.path.join(args.data_dir, desc)
        if not os.path.isdir(desc_dir):
            print(f"[ERROR] desc directory not found: {desc_dir}")
            continue

        for step in args.step:
            # Each (desc, step) pair gets its own output subdirectory
            step_output_dir = os.path.join(args.output_dir, desc, f"step{step}")

            summary = asyncio.run(run_evaluation(
                desc_dir=desc_dir,
                desc=desc,
                vqa_json_path=args.vqa_json,
                output_dir=step_output_dir,
                tasks=args.tasks,
                max_concurrent=args.max_concurrent,
                sample_ids=args.sample_ids,
                step=step,
            ))
            if summary is not None:
                all_summaries[f"{desc}_step{step}"] = summary

    # Save combined summary
    if len(all_summaries) > 1:
        combined_path = os.path.join(args.output_dir, "combined_summary.json")
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2, default=str)
        print(f"\nCombined summary saved to: {combined_path}")
        print_combined_summary(all_summaries)


if __name__ == "__main__":
    main()
