#!/usr/bin/env python3
"""Post-hoc DPGBench reward/benchmark alignment analysis.

Run from repo root after starting RM and detector servers:

    cd /home/work/AGILAB/mllm_reasoning/verl
    source ~/.bashrc >/dev/null 2>&1
    conda activate verl

    # Quick smoke run: 2 prompts x 4 samples x selected tasks/splits.
    python utils/analyze_dpg_reward_alignment.py --limit 2 --concurrency 2

    # Full 64-prompt DPGBench subset run. Safe to resume if interrupted.
    python utils/analyze_dpg_reward_alignment.py --concurrency 8 --resume

    # Override API endpoints if needed. Defaults are RM 8006/8007/8005 and detector 8083/8084.
    python utils/analyze_dpg_reward_alignment.py \
        --rm-base-urls http://10.100.85.6:8000/v1 http://10.100.85.6:8001/v1 http://10.100.85.6:8006/v1 http://10.100.85.6:8007/v1 http://10.100.85.6:8005/v1 \
        --detector-urls http://10.100.85.6:8083 http://10.100.85.6:8084 \
        --concurrency 32 --resume

Default output directory:

    /home/work/AGILAB/mllm_reasoning/data/experiments/eval/rl/0530_KT_v5_fine_grained_lr_1e_6_GAE_length_norm_gamma_0_6_sglang_v2/step_3900/dpgbench/reward_alignment_analysis

This script scores the 64 DPGBench rows in val_benchmark_v5.parquet using the
fine-grained reward function, then joins those reward logs with saved dpg-bench
benchmark detail files.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from recipe.image_rl import reward_function_fine_grained as reward_fn


DEFAULT_DUMP_ROOT = (
    "/home/work/AGILAB/mllm_reasoning/data/experiments/eval/rl/"
    "0530_KT_v5_fine_grained_lr_1e_6_GAE_length_norm_gamma_0_6_sglang_v2/"
    "step_500/dpgbench"
)
DEFAULT_PARQUET = "/home/work/AGILAB/mllm_reasoning/pimang62/data/val_benchmark_v5.parquet"
DEFAULT_RM_URLS = [
    "http://10.100.85.2:8006/v1",
    "http://10.100.85.2:8007/v1",
    "http://10.100.85.2:8005/v1",
]
DEFAULT_DETECTOR_URLS = [
    "http://10.100.85.2:8083",
    "http://10.100.85.2:8084",
]
STAGES = ("gen", "correction_0", "correction_1")


def normalize_prompt(text: Any) -> str:
    return str(text or "").strip()


def read_json_line(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: flatten_for_csv(row.get(k)) for k in keys})


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def load_dpg_rows(parquet_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    if "data_source" not in df.columns:
        raise ValueError(f"{parquet_path} has no data_source column")
    dpg = df[df["data_source"].astype(str) == "dpgbench"].copy()
    rows = []
    for row in dpg.to_dict(orient="records"):
        reward_model = row.get("reward_model") or {}
        if not isinstance(reward_model, dict):
            raise ValueError(f"reward_model is not a dict for {row.get('prompt_id')}")
        row["reward_tuple"] = reward_model.get("tuple") or row.get("tuple") or ""
        row["reward_vqa_question"] = reward_model.get("vqa_question") or row.get("question") or ""
        row["reward_summary"] = reward_model.get("summary") or row.get("summary") or ""
        rows.append(row)
    return rows


def load_dump_metadata(dump_root: Path) -> Dict[str, Dict[str, Any]]:
    by_prompt: Dict[str, Dict[str, Any]] = {}
    gen_root = dump_root / "gen"
    for metadata_path in sorted(gen_root.glob("*/metadata.jsonl")):
        metadata = read_json_line(metadata_path)
        prompt_key = normalize_prompt(metadata.get("prompt"))
        if not prompt_key:
            continue
        item_id = metadata_path.parent.name
        metadata["item_id"] = item_id
        metadata["grid_stem"] = str(metadata.get("file_name") or item_id)
        by_prompt[prompt_key] = metadata
    return by_prompt


def build_matches(dpg_rows: List[Dict[str, Any]], dump_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    metadata_by_prompt = load_dump_metadata(dump_root)
    matches: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for row in dpg_rows:
        prompt_key = normalize_prompt(row.get("prompt"))
        metadata = metadata_by_prompt.get(prompt_key)
        if metadata is None:
            missing.append({"prompt_id": row.get("prompt_id"), "prompt": row.get("prompt")})
            continue
        matches.append(
            {
                "prompt_id": row.get("prompt_id"),
                "prompt": row.get("prompt"),
                "category": row.get("category"),
                "data_source": row.get("data_source"),
                "item_id": metadata["item_id"],
                "grid_stem": metadata["grid_stem"],
                "metadata": metadata,
                "reward_tuple": row["reward_tuple"],
                "reward_vqa_question": row["reward_vqa_question"],
                "reward_summary": row["reward_summary"],
                "reward_model": row.get("reward_model") or {},
                "extra_info": row.get("extra_info") or {},
            }
        )
    return matches, missing


def sample_image_path(dump_root: Path, split: str, item_id: str, sample_index: int) -> Path:
    return dump_root / split / item_id / "samples" / f"{sample_index:05d}.png"


def load_rationales(dump_root: Path, split: str, item_id: str) -> Dict[int, Dict[str, Any]]:
    path = dump_root / split / item_id / "rationales.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    result: Dict[int, Dict[str, Any]] = {}
    for pos, entry in enumerate(data if isinstance(data, list) else []):
        if not isinstance(entry, dict):
            continue
        idx = int(entry.get("image_index", pos))
        result[idx] = entry
    return result


def rationale_answer_to_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value or "")


def configure_reward_backends(args: argparse.Namespace) -> None:
    reward_fn.VLM_BASE_URLS = list(args.rm_base_urls)
    reward_fn.LLM_BASE_URLS = []
    reward_fn.RM_VLM_MODEL_PATH = args.rm_model
    reward_fn.RM_LLM_MODEL_PATH = args.rm_model
    reward_fn.vlm_client_manager = reward_fn.ClientManager(reward_fn.VLM_BASE_URLS, name="VLM")
    reward_fn.llm_client_manager = reward_fn.ClientManager(reward_fn.LLM_BASE_URLS, name="LLM")
    reward_fn._rm_slot_queues.clear()

    reward_fn.DETECTOR_URLS = [] if args.disable_detector else list(args.detector_urls)
    reward_fn._det_slot_queues.clear()


def parse_result_line(line: str) -> Optional[Dict[str, Any]]:
    parts = line.strip().rsplit(", ", 5)
    if len(parts) != 6:
        return None
    path, s0, s1, s2, s3, avg = parts
    try:
        scores = [float(s0), float(s1), float(s2), float(s3)]
        avg_score = float(avg)
    except ValueError:
        return None
    return {
        "grid_path": path,
        "grid_stem": Path(path).stem,
        "crop_scores": scores,
        "benchmark_avg_score": avg_score,
    }


DETAIL_RE = re.compile(
    r"^(?P<path>.*\.png), \("
    r"(?P<x1>\d+), (?P<y1>\d+), (?P<x2>\d+), (?P<y2>\d+)"
    r"\), (?P<question>.*), (?P<answer>yes|no)\s*$",
    re.IGNORECASE,
)


def sample_index_from_crop(x1: int, y1: int, x2: int, y2: int) -> int:
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    col = 0 if x1 < width else 1
    row = 0 if y1 < height else 1
    return row * 2 + col


def parse_detail_line(line: str) -> Optional[Dict[str, Any]]:
    match = DETAIL_RE.match(line.strip())
    if not match:
        return None
    x1, y1, x2, y2 = (int(match.group(k)) for k in ("x1", "y1", "x2", "y2"))
    return {
        "grid_path": match.group("path"),
        "grid_stem": Path(match.group("path")).stem,
        "crop": [x1, y1, x2, y2],
        "sample_index": sample_index_from_crop(x1, y1, x2, y2),
        "benchmark_question": match.group("question"),
        "benchmark_answer": match.group("answer").lower(),
    }


def load_benchmark_records(dump_root: Path, splits: List[str], valid_grid_stems: set[str]) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, int], Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    by_sample: Dict[Tuple[str, str, int], Dict[str, Any]] = defaultdict(
        lambda: {"benchmark_questions": [], "benchmark_answers": []}
    )

    for split in splits:
        grid_root = dump_root / "final_outputs" / split / "grid"
        if not grid_root.exists():
            continue

        for result_path in sorted(grid_root.glob("*results.txt")):
            if result_path.name.endswith("_detail.txt"):
                continue
            with result_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parsed = parse_result_line(line)
                    if not parsed or parsed["grid_stem"] not in valid_grid_stems:
                        continue
                    for sample_index, crop_score in enumerate(parsed["crop_scores"]):
                        key = (split, parsed["grid_stem"], sample_index)
                        by_sample[key].update(
                            {
                                "benchmark_crop_score": crop_score,
                                "benchmark_avg_score": parsed["benchmark_avg_score"],
                                "benchmark_results_file": str(result_path),
                                "grid_path": parsed["grid_path"],
                            }
                        )

        for detail_path in sorted(grid_root.glob("*results_detail.txt")):
            with detail_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parsed = parse_detail_line(line)
                    if not parsed or parsed["grid_stem"] not in valid_grid_stems:
                        continue
                    parsed.update({"split": split, "benchmark_detail_file": str(detail_path)})
                    records.append(parsed)
                    key = (split, parsed["grid_stem"], parsed["sample_index"])
                    by_sample[key]["benchmark_questions"].append(parsed["benchmark_question"])
                    by_sample[key]["benchmark_answers"].append(parsed["benchmark_answer"])
                    by_sample[key].setdefault("benchmark_detail_file", str(detail_path))

    return records, dict(by_sample)


def detector_inputs(feedback_tuple: str) -> List[Dict[str, Any]]:
    return reward_fn.verify_detection_single(feedback_tuple)


def task_record_base(match: Dict[str, Any], split: str, sample_index: int, task: int) -> Dict[str, Any]:
    return {
        "prompt_id": match["prompt_id"],
        "prompt": match["prompt"],
        "item_id": match["item_id"],
        "grid_stem": match["grid_stem"],
        "split": split,
        "sample_index": sample_index,
        "task": task,
    }


def extra_value(result: Dict[str, Any], key: str, default: Any = None) -> Any:
    return (result.get("reward_extra_info") or {}).get(key, default)


async def safe_compute(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    try:
        return await reward_fn.compute_score_single_async(*args, **kwargs)
    except Exception as exc:
        return {"score": 0.0, "reward_extra_info": {"error": repr(exc)}}


async def score_one_sample(
    match: Dict[str, Any],
    dump_root: Path,
    sample_index: int,
    splits: List[str],
    semaphore: asyncio.Semaphore,
) -> List[Dict[str, Any]]:
    async with semaphore:
        records: List[Dict[str, Any]] = []
        prompt = match["prompt"]
        feedback_tuple = match["reward_tuple"]
        vqa_question = match["reward_vqa_question"]
        summary = match["reward_summary"]
        item_id = match["item_id"]

        gen_img = sample_image_path(dump_root, "gen", item_id, sample_index)
        source_vqa_by_split: Dict[str, float] = {}

        if "gen" in splits and gen_img.exists():
            det_list = detector_inputs(feedback_tuple)
            result = await safe_compute(
                prompt,
                str(gen_img),
                "",
                None,
                None,
                summary,
                feedback_tuple,
                None,
                None,
                None,
                None,
                vqa_question,
                {},
                1,
            )
            rec = task_record_base(match, "gen", sample_index, 1)
            rec.update(
                {
                    "image_path": str(gen_img),
                    "reward_question": vqa_question,
                    "detector_info_list": det_list,
                    "score": result.get("score", 0.0),
                    "reward_extra_info": result.get("reward_extra_info", {}),
                    "task_vqa_reward": extra_value(result, "task1_vqa_reward", 0.0),
                    "detector_reward": extra_value(result, "task1_detector_reward", 0.0),
                }
            )
            records.append(rec)
            source_vqa_by_split["gen"] = to_float(rec["task_vqa_reward"])

        previous_split = "gen"
        for split in ("correction_0", "correction_1"):
            if split not in splits:
                previous_split = split
                continue

            source_img = sample_image_path(dump_root, previous_split, item_id, sample_index)
            edited_img = sample_image_path(dump_root, split, item_id, sample_index)
            rationales = load_rationales(dump_root, split, item_id)
            rationale = rationales.get(sample_index)
            if not source_img.exists() or not edited_img.exists() or not rationale:
                rec = task_record_base(match, split, sample_index, 0)
                rec.update(
                    {
                        "source_image_path": str(source_img),
                        "edited_image_path": str(edited_img),
                        "error": "missing source image, edited image, or rationale",
                    }
                )
                records.append(rec)
                previous_split = split
                continue

            pred_tuple = rationale.get("pred_tuple")
            pred_answer = rationale_answer_to_text(rationale.get("pred_answer"))
            pred_feedback = rationale.get("pred_feedback")
            feedback_text = rationale.get("original_answer") or "\n\n".join(
                [str(pred_tuple or ""), pred_answer, str(pred_feedback or "")]
            )
            decision_vqa = source_vqa_by_split.get(previous_split, source_vqa_by_split.get("gen", 0.0))
            extra_info = {
                "task2_decision_vqa_reward": decision_vqa,
                "task2_decision_vqa_source": f"{previous_split}_vqa_reward",
            }

            task2_result = await safe_compute(
                prompt,
                str(source_img),
                feedback_text,
                None,
                None,
                summary,
                feedback_tuple,
                None,
                pred_tuple,
                pred_answer,
                pred_feedback,
                vqa_question,
                extra_info,
                2,
            )
            task2_rec = task_record_base(match, split, sample_index, 2)
            task2_rec.update(
                {
                    "source_image_path": str(source_img),
                    "edited_image_path": str(edited_img),
                    "reward_question": vqa_question,
                    "gt_tuple": feedback_tuple,
                    "rationale": rationale,
                    "pred_tuple": pred_tuple,
                    "pred_answer": pred_answer,
                    "pred_feedback": pred_feedback,
                    "decision_vqa_reward": decision_vqa,
                    "score": task2_result.get("score", 0.0),
                    "reward_extra_info": task2_result.get("reward_extra_info", {}),
                    "task2_prompt_to_tuple_reward": extra_value(task2_result, "task2_prompt_to_tuple_reward", 0.0),
                    "task2_tuple_to_vqa_reward": extra_value(task2_result, "task2_tuple_to_vqa_reward", 0.0),
                    "task2_vqa_to_feedback_reward": extra_value(task2_result, "task2_vqa_to_feedback_reward", 0.0),
                    "task2_vlm_reward": extra_value(task2_result, "task2_vlm_reward", 0.0),
                }
            )
            records.append(task2_rec)

            det_list = detector_inputs(feedback_tuple)
            task3_result = await safe_compute(
                prompt,
                str(source_img),
                feedback_text,
                str(edited_img),
                None,
                summary,
                feedback_tuple,
                None,
                pred_tuple,
                pred_answer,
                pred_feedback,
                vqa_question,
                {},
                3,
            )
            task3_rec = task_record_base(match, split, sample_index, 3)
            task3_rec.update(
                {
                    "source_image_path": str(source_img),
                    "edited_image_path": str(edited_img),
                    "reward_question": vqa_question,
                    "detector_info_list": det_list,
                    "pred_feedback": pred_feedback,
                    "score": task3_result.get("score", 0.0),
                    "reward_extra_info": task3_result.get("reward_extra_info", {}),
                    "task_vqa_reward": extra_value(task3_result, "task3_vqa_reward", 0.0),
                    "task3_edit_reward": extra_value(task3_result, "task3_edit_reward", 0.0),
                    "detector_reward": extra_value(task3_result, "task3_detector_reward", 0.0),
                }
            )
            records.append(task3_rec)
            source_vqa_by_split[split] = to_float(task3_rec["task_vqa_reward"])
            previous_split = split

        return records


def load_existing_reward_keys(path: Path) -> set[Tuple[str, str, int, int]]:
    keys = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((row["split"], row["item_id"], int(row["sample_index"]), int(row["task"])))
    return keys


def record_key(row: Dict[str, Any]) -> Tuple[str, str, int, int]:
    return (row["split"], row["item_id"], int(row["sample_index"]), int(row["task"]))


def expected_record_keys(match: Dict[str, Any], sample_index: int, splits: List[str]) -> set[Tuple[str, str, int, int]]:
    keys = set()
    item_id = match["item_id"]
    if "gen" in splits:
        keys.add(("gen", item_id, sample_index, 1))
    for split in ("correction_0", "correction_1"):
        if split in splits:
            keys.add((split, item_id, sample_index, 2))
            keys.add((split, item_id, sample_index, 3))
    return keys


async def score_all(args: argparse.Namespace, matches: List[Dict[str, Any]], output_dir: Path) -> List[Dict[str, Any]]:
    reward_path = output_dir / "reward_records.jsonl"
    skipped_keys = load_existing_reward_keys(reward_path) if args.resume else set()
    mode = "a" if args.resume and reward_path.exists() else "w"

    semaphore = asyncio.Semaphore(args.concurrency)
    selected = matches[: args.limit] if args.limit else matches

    all_records: List[Dict[str, Any]] = []
    if args.resume and reward_path.exists():
        with reward_path.open("r", encoding="utf-8") as f:
            all_records = [json.loads(line) for line in f if line.strip()]

    jobs = []
    for match in selected:
        for sample_index in range(4):
            expected = expected_record_keys(match, sample_index, args.splits)
            if args.resume and expected and expected.issubset(skipped_keys):
                continue
            jobs.append((match, sample_index))

    with reward_path.open(mode, encoding="utf-8") as out:
        tasks = [
            asyncio.create_task(score_one_sample(match, Path(args.dump_root), sample_index, args.splits, semaphore))
            for match, sample_index in jobs
        ]
        total = len(tasks)
        for idx, task in enumerate(asyncio.as_completed(tasks), start=1):
            records = await task
            new_records = [r for r in records if record_key(r) not in skipped_keys]
            for row in new_records:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            all_records.extend(new_records)
            if idx % args.progress_every == 0 or idx == total:
                print(f"[reward] completed {idx}/{total} sample jobs")
        if total == 0:
            print("[reward] no sample jobs to run")

    return all_records


def build_comparison_rows(
    reward_records: List[Dict[str, Any]],
    benchmark_by_sample: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in reward_records:
        if rec.get("task") == 0:
            continue
        key = (rec["split"], rec["grid_stem"], int(rec["sample_index"]))
        bench = benchmark_by_sample.get(key, {})
        row = {
            "prompt_id": rec.get("prompt_id"),
            "item_id": rec.get("item_id"),
            "grid_stem": rec.get("grid_stem"),
            "split": rec.get("split"),
            "sample_index": rec.get("sample_index"),
            "task": rec.get("task"),
            "reward_score": rec.get("score"),
            "benchmark_crop_score": bench.get("benchmark_crop_score"),
            "benchmark_avg_score": bench.get("benchmark_avg_score"),
            "benchmark_questions": bench.get("benchmark_questions", []),
            "benchmark_answers": bench.get("benchmark_answers", []),
            "reward_question": rec.get("reward_question"),
            "image_path": rec.get("image_path") or rec.get("edited_image_path"),
            "source_image_path": rec.get("source_image_path"),
            "edited_image_path": rec.get("edited_image_path"),
            "pred_feedback": rec.get("pred_feedback"),
            "task_vqa_reward": rec.get("task_vqa_reward"),
            "detector_reward": rec.get("detector_reward"),
            "task3_edit_reward": rec.get("task3_edit_reward"),
            "task2_vlm_reward": rec.get("task2_vlm_reward"),
            "task2_prompt_to_tuple_reward": rec.get("task2_prompt_to_tuple_reward"),
            "task2_tuple_to_vqa_reward": rec.get("task2_tuple_to_vqa_reward"),
            "task2_vqa_to_feedback_reward": rec.get("task2_vqa_to_feedback_reward"),
            "reward_extra_info": rec.get("reward_extra_info"),
        }
        rows.append(row)
    return rows


def summarize(comparison_rows: List[Dict[str, Any]], matches: List[Dict[str, Any]], missing: List[Dict[str, Any]]) -> Dict[str, Any]:
    def finite_mean(values: List[float]) -> Optional[float]:
        finite = [x for x in values if math.isfinite(x)]
        return sum(finite) / len(finite) if finite else None

    summary: Dict[str, Any] = {
        "matched_rows": len(matches),
        "missing_rows": len(missing),
        "missing": missing,
        "groups": {},
    }
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        grouped[(str(row.get("split")), int(row.get("task")))].append(row)

    for (split, task), rows in grouped.items():
        rewards = [to_float(r.get("reward_score"), float("nan")) for r in rows]
        bench_crop = [to_float(r.get("benchmark_crop_score"), float("nan")) for r in rows]
        bench_avg = [to_float(r.get("benchmark_avg_score"), float("nan")) for r in rows]
        valid_rewards = [x for x in rewards if math.isfinite(x)]
        key = f"{split}/task{task}"
        summary["groups"][key] = {
            "count": len(rows),
            "reward_mean": sum(valid_rewards) / len(valid_rewards) if valid_rewards else None,
            "benchmark_crop_mean": finite_mean(bench_crop),
            "benchmark_avg_mean": finite_mean(bench_avg),
            "pearson_reward_vs_benchmark_crop": pearson(rewards, bench_crop),
            "pearson_reward_vs_benchmark_avg": pearson(rewards, bench_avg),
            "missing_benchmark_count": sum(not math.isfinite(x) for x in bench_crop),
        }
    return summary


def build_mismatch_cases(comparison_rows: List[Dict[str, Any]], limit: int = 200) -> List[Dict[str, Any]]:
    cases = []
    for row in comparison_rows:
        reward_score = to_float(row.get("reward_score"), float("nan"))
        bench_score = to_float(row.get("benchmark_crop_score"), float("nan"))
        if not math.isfinite(reward_score) or not math.isfinite(bench_score):
            continue
        # Task1/3 scores can exceed 1 because detector bonus is additive; clamp for comparison only.
        comparable_reward = max(0.0, min(1.0, reward_score))
        diff = abs(comparable_reward - bench_score)
        case = dict(row)
        case["reward_score_clamped_for_diff"] = comparable_reward
        case["abs_reward_benchmark_diff"] = diff
        cases.append(case)
    cases.sort(key=lambda x: x["abs_reward_benchmark_diff"], reverse=True)
    return cases[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", default=DEFAULT_DUMP_ROOT)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--splits", nargs="+", default=list(STAGES), choices=list(STAGES))
    parser.add_argument("--rm-base-urls", nargs="+", default=DEFAULT_RM_URLS)
    parser.add_argument("--rm-model", default=os.environ.get("RM_VLM_MODEL_PATH", reward_fn.RM_VLM_MODEL_PATH))
    parser.add_argument("--detector-urls", nargs="+", default=DEFAULT_DETECTOR_URLS)
    parser.add_argument("--disable-detector", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    dump_root = Path(args.dump_root)
    output_dir = Path(args.output_dir) if args.output_dir else dump_root / "reward_alignment_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_reward_backends(args)

    dpg_rows = load_dpg_rows(Path(args.parquet))
    matches, missing = build_matches(dpg_rows, dump_root)
    valid_grid_stems = {m["grid_stem"] for m in matches}

    print(f"[match] parquet dpg rows={len(dpg_rows)} matched={len(matches)} missing={len(missing)}")
    if missing:
        print(f"[match] missing prompt_ids={[m['prompt_id'] for m in missing]}")

    benchmark_records, benchmark_by_sample = load_benchmark_records(dump_root, args.splits, valid_grid_stems)
    write_jsonl(output_dir / "benchmark_records.jsonl", benchmark_records)
    print(f"[benchmark] detail records={len(benchmark_records)} sample keys={len(benchmark_by_sample)}")

    reward_records = await score_all(args, matches, output_dir)
    comparison_rows = build_comparison_rows(reward_records, benchmark_by_sample)
    write_jsonl(output_dir / "comparison_per_sample.jsonl", comparison_rows)
    write_csv(output_dir / "comparison_per_sample.csv", comparison_rows)

    summary = summarize(comparison_rows, matches, missing)
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    mismatch_cases = build_mismatch_cases(comparison_rows)
    write_jsonl(output_dir / "mismatch_cases.jsonl", mismatch_cases)
    print(f"[done] wrote outputs to {output_dir}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
