#!/usr/bin/env python3
"""
Offline reward evaluation script for t2icompbench benchmark dumps.

Computes fine-grained rewards (Task 1, 2, 3) on pre-generated t2icompbench data
using the same reward logic as reward_function_fine_grained.py.

Supports iterative correction chains via --step:
    step 0: curr=gen                          → Task 1 only
    step 1: prev=gen,          curr=correction_0  → Task 1,2,3
    step 2: prev=correction_0, curr=correction_1  → Task 1,2,3
    ...

Directory structure (differs from geneval by having a category level):
    data_dir/
        gen/{category}/{folder_id}/metadata.jsonl, samples/
        correction_0/{category}/{folder_id}/metadata.jsonl, rationales.json, samples/
        correction_1/{category}/{folder_id}/...

Categories: 3d_spatial, color, complex, non-spatial, numeracy, shape, spatial, texture

Usage:
    # Step 0 (gen only, Task 1)
    python -m bench.reward_t2icompbench \
        --data_dir /data/mllm/data/for_down/t2icompbench \
        --vqa_json "" \
        --output_dir /data/mllm/data/for_down/t2icompbench/reward_results/step0 \
        --tasks 1

    # Step 1: gen → correction_0 (Task 1,2,3)
    python -m bench.reward_t2icompbench \
        --data_dir /data/mllm/data/for_down/t2icompbench \
        --vqa_json "" \
        --output_dir /data/mllm/data/for_down/t2icompbench/reward_results/step1 \
        --step 1 --tasks 1 2 3

    # All steps 0-3 at once
    python -m bench.reward_t2icompbench \
        --data_dir /data/mllm/data/for_down/t2icompbench \
        --vqa_json "" \
        --output_dir /data/mllm/data/for_down/t2icompbench/reward_results \
        --step 0 1 2 3 --tasks 1 2 3
"""

import os
import sys
import json
import argparse
import asyncio
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any

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


def step_to_dirs(step: int) -> tuple:
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


def discover_samples(data_dir: str, subdir: str) -> List[tuple]:
    """Discover all (category, folder_id) pairs under data_dir/subdir/.

    t2icompbench structure: {subdir}/{category}/{folder_id}/
    Returns sorted list of (category, folder_id) tuples.
    """
    base_path = os.path.join(data_dir, subdir)
    samples = []
    for category in sorted(os.listdir(base_path)):
        cat_path = os.path.join(base_path, category)
        if not os.path.isdir(cat_path):
            continue
        for folder_id in sorted(os.listdir(cat_path)):
            folder_path = os.path.join(cat_path, folder_id)
            if not os.path.isdir(folder_path):
                continue
            samples.append((category, folder_id))
    return samples


def load_sample(
    data_dir: str,
    category: str,
    folder_id: str,
    prompt_to_vqa: Dict[str, dict],
    prev_subdir: Optional[str] = None,
    curr_subdir: str = "gen",
) -> dict:
    """Load a single sample's data.

    Args:
        category: t2icompbench category (e.g., "color", "spatial").
        prev_subdir: previous step's images (source for Task 2/3). None for step 0.
        curr_subdir: current step's images + rationales.

    Returns dict with:
        - prev_images / prev_image_paths: source images (from prev step)
        - curr_images / curr_image_paths: current images (Task 1 target, Task 3 edit target)
        - rationales: from curr_subdir (reasoning about prev images)
    """
    curr_dir = os.path.join(data_dir, curr_subdir, category, folder_id)

    # Metadata — always read from curr (or gen/ for fallback)
    meta_path = os.path.join(curr_dir, "metadata.jsonl")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(data_dir, "gen", category, folder_id, "metadata.jsonl")
    with open(meta_path) as f:
        metadata = json.loads(f.readline())
    prompt = metadata["prompt"]

    # VQA question
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
        prev_dir = os.path.join(data_dir, prev_subdir, category, folder_id)
        prev_samples_dir = os.path.join(prev_dir, "samples")
        for i in range(4):
            img_path = os.path.join(prev_samples_dir, f"{i:05d}.png")
            if os.path.exists(img_path):
                prev_images.append(PIL.Image.open(img_path).convert("RGB"))
                prev_image_paths.append(img_path)

    # Rationales from curr (reasoning about prev images)
    rationales = None
    rat_path = os.path.join(curr_dir, "rationales.json")
    if os.path.exists(rat_path):
        with open(rat_path) as f:
            rationales = json.load(f)

    return {
        "folder_id": folder_id,
        "category": category,
        "sample_key": f"{category}/{folder_id}",
        "prompt": prompt,
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
        "sample_key": sample["sample_key"],
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
            "sample_key": sample["sample_key"],
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
            "sample_key": sample["sample_key"],
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
        "sample_key": sample["sample_key"],
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
            "sample_key": sample["sample_key"],
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
            "sample_key": sample["sample_key"],
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
            "sample_key": sample["sample_key"],
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
        "sample_key": sample["sample_key"],
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
                    print(f"[ERROR] sample={sample['sample_key']} image={image_idx}: {e}")
                    results.append({
                        "sample_key": sample["sample_key"],
                        "folder_id": sample["folder_id"],
                        "category": sample["category"],
                        "image_index": image_idx,
                        "task": -1,
                        "score": 0.0,
                        "reward_extra_info": {"error": str(e)},
                    })

    return results


async def run_evaluation(
    data_dir: str,
    vqa_json_path: str,
    output_dir: str,
    tasks: List[int],
    max_concurrent: int = 64,
    sample_ids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    step: int = 0,
):
    """Main evaluation loop.

    Args:
        step: correction iteration step.
              0 = Task 1 only on gen images
              1 = gen→correction_0, 2 = correction_0→correction_1, etc.
        categories: filter to specific categories (e.g., ["color", "spatial"]).
        sample_ids: filter to specific sample keys ("category/folder_id").
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
    print(f"Step {step}: prev={prev_subdir}, curr={curr_subdir}, tasks={tasks}")
    print(f"{'='*60}")

    # Validate directories exist
    curr_path = os.path.join(data_dir, curr_subdir)
    if not os.path.isdir(curr_path):
        print(f"[ERROR] Current directory not found: {curr_path}")
        return None
    if prev_subdir is not None:
        prev_path = os.path.join(data_dir, prev_subdir)
        if not os.path.isdir(prev_path):
            print(f"[ERROR] Previous directory not found: {prev_path}")
            return None

    os.makedirs(output_dir, exist_ok=True)

    # Load VQA questions
    prompt_to_vqa = {}
    if vqa_json_path:
        print(f"Loading VQA questions from {vqa_json_path}...")
        prompt_to_vqa = load_vqa_questions(vqa_json_path)
        print(f"  Loaded {len(prompt_to_vqa)} VQA entries")
    else:
        print("[INFO] No VQA JSON provided — vqa_question/summarize/feedback_tuple will be empty")

    # Discover samples: (category, folder_id) pairs
    all_samples = discover_samples(data_dir, curr_subdir)

    # Filter by categories
    if categories is not None:
        all_samples = [(c, f) for c, f in all_samples if c in categories]

    # Filter by sample_ids (format: "category/folder_id")
    if sample_ids is not None:
        sample_id_set = set(sample_ids)
        all_samples = [(c, f) for c, f in all_samples if f"{c}/{f}" in sample_id_set]

    print(f"Found {len(all_samples)} samples across categories, evaluating tasks: {tasks}")

    # Show per-category counts
    cat_counts = defaultdict(int)
    for c, f in all_samples:
        cat_counts[c] += 1
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")

    semaphore = asyncio.Semaphore(max_concurrent)
    all_results = []
    start_time = time.time()

    # Process in batches for memory efficiency
    BATCH_SIZE = 50
    for batch_start in range(0, len(all_samples), BATCH_SIZE):
        batch_sample_keys = all_samples[batch_start:batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(all_samples))
        print(f"\nProcessing samples {batch_start}-{batch_end-1} / {len(all_samples)}...")

        # Load samples
        samples = []
        for category, folder_id in batch_sample_keys:
            try:
                sample = load_sample(
                    data_dir, category, folder_id, prompt_to_vqa,
                    prev_subdir=prev_subdir, curr_subdir=curr_subdir,
                )
                samples.append(sample)
            except Exception as e:
                print(f"[WARN] Failed to load sample {category}/{folder_id}: {e}")

        # Run evaluation
        batch_tasks = [eval_sample(s, tasks, semaphore) for s in samples]
        batch_results_nested = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for sample, res in zip(samples, batch_results_nested):
            if isinstance(res, Exception):
                print(f"[ERROR] Batch exception for {sample['sample_key']}: {res}")
            else:
                all_results.extend(res)

        # Save per-sample results incrementally
        for sample in samples:
            sample_results = [r for r in all_results if r.get("sample_key") == sample["sample_key"]]
            if sample_results:
                sample_out = os.path.join(
                    output_dir, "per_sample", sample["category"], sample["folder_id"] + ".json"
                )
                os.makedirs(os.path.dirname(sample_out), exist_ok=True)
                with open(sample_out, "w") as f:
                    json.dump({
                        "sample_key": sample["sample_key"],
                        "folder_id": sample["folder_id"],
                        "category": sample["category"],
                        "prompt": sample["prompt"],
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
    print(f"\nDone step {step}! Total time: {total_time:.1f}s")
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

    # Collect unique sample_keys
    sample_keys = set(r["sample_key"] for r in all_results)
    summary["total_samples"] = len(sample_keys)

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

        # Group by category
        cat_scores = defaultdict(list)
        for r in task_results:
            cat = r.get("category", "unknown")
            if r.get("score") is not None:
                cat_scores[cat].append(r["score"])

        cat_summary = {}
        for cat, c_scores in sorted(cat_scores.items()):
            cat_summary[cat] = {
                "count": len(c_scores),
                "mean_score": float(np.mean(c_scores)),
                "std_score": float(np.std(c_scores)),
            }
        task_summary["by_category"] = cat_summary

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
            # Split into edited vs not-edited (-100)
            edited = [r for r in task_results if r.get("score") != -100]
            not_edited = [r for r in task_results if r.get("score") == -100]

            task_summary["edited_count"] = len(edited)
            task_summary["not_edited_count"] = len(not_edited)
            task_summary["edit_ratio"] = len(edited) / len(task_results) if task_results else 0.0

            # Mean score only for edited samples
            edited_scores = [r["score"] for r in edited if r.get("score") is not None]
            task_summary["edited_mean_score"] = float(np.mean(edited_scores)) if edited_scores else 0.0
            task_summary["edited_std_score"] = float(np.std(edited_scores)) if edited_scores else 0.0
            task_summary["edited_median_score"] = float(np.median(edited_scores)) if edited_scores else 0.0

            for key in ["task3_vqa_reward", "task3_edit_reward"]:
                vals = [r["reward_extra_info"].get(key, 0.0) for r in edited
                        if "reward_extra_info" in r]
                task_summary[f"mean_{key}"] = float(np.mean(vals)) if vals else 0.0

            # Per-category edit breakdown
            cat_edit_breakdown = defaultdict(lambda: {"edited": 0, "not_edited": 0, "edited_scores": []})
            for r in task_results:
                cat = r.get("category", "unknown")
                if r.get("score") == -100:
                    cat_edit_breakdown[cat]["not_edited"] += 1
                else:
                    cat_edit_breakdown[cat]["edited"] += 1
                    if r.get("score") is not None:
                        cat_edit_breakdown[cat]["edited_scores"].append(r["score"])

            cat_edit_summary = {}
            for cat, info in sorted(cat_edit_breakdown.items()):
                total = info["edited"] + info["not_edited"]
                cat_edit_summary[cat] = {
                    "edited": info["edited"],
                    "not_edited": info["not_edited"],
                    "total": total,
                    "edit_ratio": info["edited"] / total if total > 0 else 0.0,
                    "edited_mean_score": float(np.mean(info["edited_scores"])) if info["edited_scores"] else 0.0,
                }
            task_summary["edit_breakdown_by_category"] = cat_edit_summary

        summary[f"task{task_id}"] = task_summary

    # ── Cross-task analysis: Task 1 × Task 3 ──
    task1_results = [r for r in all_results if r.get("task") == 1]
    task3_results = [r for r in all_results if r.get("task") == 3]

    if task1_results and task3_results:
        # Build lookup: (sample_key, image_index) -> result
        t1_lookup = {(r["sample_key"], r["image_index"]): r for r in task1_results}
        t3_lookup = {(r["sample_key"], r["image_index"]): r for r in task3_results}

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
                    "sample_key": key[0],
                    "image_index": key[1],
                    "category": t1.get("category", "unknown"),
                    "task1_score": t1_score,
                    "task3_score": t3_score,
                })
            elif t1_score < 1.0 and t3_score == -100:
                missed_edits.append({
                    "sample_key": key[0],
                    "image_index": key[1],
                    "category": t1.get("category", "unknown"),
                    "task1_score": t1_score,
                })

        def _agg_by_category(items):
            by_cat = defaultdict(list)
            for item in items:
                by_cat[item["category"]].append(item)
            return {cat: len(lst) for cat, lst in sorted(by_cat.items())}

        cross_analysis = {
            "total_paired": len(common_keys),
            "unnecessary_edit": {
                "description": "Task1 all-yes (score=1.0) but model still edited (task3 != -100)",
                "count": len(unnecessary_edits),
                "ratio": len(unnecessary_edits) / len(common_keys) if common_keys else 0.0,
                "by_category": _agg_by_category(unnecessary_edits),
                "samples": unnecessary_edits,
            },
            "missed_edit": {
                "description": "Task1 has-no (score<1.0) but model did not edit (task3 == -100)",
                "count": len(missed_edits),
                "ratio": len(missed_edits) / len(common_keys) if common_keys else 0.0,
                "by_category": _agg_by_category(missed_edits),
                "samples": missed_edits,
            },
        }
        summary["cross_task_analysis"] = cross_analysis

    return summary


def print_summary(summary: dict):
    """Pretty-print the summary."""
    print("\n" + "=" * 60)
    print("REWARD EVALUATION SUMMARY (t2icompbench)")
    print("=" * 60)
    if "step" in summary:
        prev = summary.get('prev_subdir') or '(none)'
        curr = summary.get('curr_subdir', '?')
        print(f"Step {summary['step']}: prev={prev}, curr={curr}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Total evaluations: {summary['total_evaluations']}")

    for task_id in [1, 2, 3]:
        key = f"task{task_id}"
        if key not in summary:
            continue
        ts = summary[key]
        print(f"\n── Task {task_id} ──")
        print(f"  Count: {ts['count']}")
        print(f"  Score: {ts['mean_score']:.4f} ± {ts['std_score']:.4f} (median={ts['median_score']:.4f}, min={ts['min_score']:.4f}, max={ts['max_score']:.4f})")

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
            print(f"  Edited-only mean score: {ts.get('edited_mean_score', 0):.4f} ± {ts.get('edited_std_score', 0):.4f} (median={ts.get('edited_median_score', 0):.4f})")
            print(f"  Mean VQA reward (edited): {ts.get('mean_task3_vqa_reward', 0):.4f}")
            print(f"  Mean edit reward (edited): {ts.get('mean_task3_edit_reward', 0):.4f}")
            if ts.get("edit_breakdown_by_category"):
                print(f"  Edit breakdown by category:")
                for cat, info in ts["edit_breakdown_by_category"].items():
                    print(f"    {cat:20s}: edited={info['edited']:>4d} | not_edited={info['not_edited']:>4d} | ratio={info['edit_ratio']:.4f} | edited_mean={info['edited_mean_score']:.4f}")

        if ts.get("by_category") and task_id != 3:
            print(f"  By category:")
            for cat, cat_data in ts["by_category"].items():
                print(f"    {cat:20s}: {cat_data['mean_score']:.4f} ± {cat_data['std_score']:.4f} (n={cat_data['count']})")

    # Cross-task analysis
    cross = summary.get("cross_task_analysis")
    if cross:
        print(f"\n── Cross-Task Analysis (Task1 x Task3) ──")
        print(f"  Total paired: {cross['total_paired']}")

        ue = cross["unnecessary_edit"]
        print(f"\n  Unnecessary edits (Task1=1.0 but still edited):")
        print(f"    Count: {ue['count']} / {cross['total_paired']} ({ue['ratio']:.4f})")
        if ue["by_category"]:
            for cat, cnt in ue["by_category"].items():
                print(f"      {cat:20s}: {cnt}")

        me = cross["missed_edit"]
        print(f"\n  Missed edits (Task1<1.0 but not edited):")
        print(f"    Count: {me['count']} / {cross['total_paired']} ({me['ratio']:.4f})")
        if me["by_category"]:
            for cat, cnt in me["by_category"].items():
                print(f"      {cat:20s}: {cnt}")

    print("=" * 60)


def print_combined_summary(all_summaries: Dict[str, dict]):
    """Print a cross-step comparison table."""
    print("\n" + "=" * 70)
    print("CROSS-STEP COMPARISON (t2icompbench)")
    print("=" * 70)

    steps = sorted(all_summaries.keys())
    for task_id in [1, 2, 3]:
        key = f"task{task_id}"
        has_task = any(key in all_summaries[s] for s in steps)
        if not has_task:
            continue

        print(f"\n── Task {task_id} ──")
        print(f"  {'Step':<20s} {'Mean Score':>12s} {'Std':>10s} {'Count':>8s}")
        print(f"  {'-'*50}")
        for s in steps:
            ts = all_summaries[s].get(key, {})
            if not ts:
                continue
            prev_sub = all_summaries[s].get("prev_subdir") or "(none)"
            curr_sub = all_summaries[s].get("curr_subdir", "?")
            label = f"{prev_sub}->{curr_sub}"
            print(f"  {label:<20s} {ts.get('mean_score', 0):>12.4f} {ts.get('std_score', 0):>10.4f} {ts.get('count', 0):>8d}")

    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Offline reward evaluation for t2icompbench dumps")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to t2icompbench data dir (contains gen/, correction_0/, ...)")
    parser.add_argument("--vqa_json", type=str, required=True,
                        help="Path to VQA question JSON. Leave empty to skip VQA ground-truth.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--step", type=int, nargs="+", default=[0],
                        help="Correction step(s). 0=gen only, 1=gen->correction_0, etc. "
                             "Multiple steps can be specified (e.g., --step 0 1 2 3)")
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3],
                        choices=[1, 2, 3], help="Which tasks to evaluate (default: 1 2 3)")
    parser.add_argument("--max_concurrent", type=int, default=64,
                        help="Max concurrent API requests (default: 64)")
    parser.add_argument("--categories", type=str, nargs="*", default=None,
                        help="Filter to specific categories (e.g., color spatial)")
    parser.add_argument("--sample_ids", type=str, nargs="*", default=None,
                        help="Specific sample keys to evaluate (e.g., color/00000 spatial/00001)")
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
    for step in args.step:
        # Each step gets its own output subdirectory
        if len(args.step) > 1:
            step_output_dir = os.path.join(args.output_dir, f"step{step}")
        else:
            step_output_dir = args.output_dir

        summary = asyncio.run(run_evaluation(
            data_dir=args.data_dir,
            vqa_json_path=args.vqa_json,
            output_dir=step_output_dir,
            tasks=args.tasks,
            max_concurrent=args.max_concurrent,
            sample_ids=args.sample_ids,
            categories=args.categories,
            step=step,
        ))
        if summary is not None:
            all_summaries[f"step{step}"] = summary

    # If multiple steps, save a combined summary
    if len(args.step) > 1 and all_summaries:
        combined_path = os.path.join(args.output_dir, "combined_summary.json")
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2, default=str)
        print(f"\nCombined summary saved to: {combined_path}")
        print_combined_summary(all_summaries)


if __name__ == "__main__":
    main()
