# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert benchmark_v4 JSON files → verl-format parquet for RL training.

There is no single "correct" SMILES answer in this benchmark.
Reward is computed by calling chemistry tool servers to verify whether the
predicted SMILES satisfies the property + fragment constraints in `answer`.
(See chemistry_reward.py for the evaluation logic.)

Input files (benchmark_v4):
  scenarios_generation_balanced_rl_10k.json     → train
  scenarios_optimization_balanced_rl_10k.json   → train
  scenarios_generation_balanced_test_1k.json    → test
  scenarios_optimization_balanced_test_1k.json  → test

Output parquet columns:
  prompt        – [{"role": "system", ...}, {"role": "user", "content": question}]
  data_source   – "mol_generation" | "mol_optimization"
  reward_model  – {"ground_truth": <json of answer+infeasible>, "task_type": ...}
  extra_info    – {"id", "infeasible", "ref_smiles", "task_type", "meta_info"}

Usage:
  python create_chemagent_dataset.py \\
      --data_dir /path/to/benchmark_v4 \\
      --output_dir /data/verl/data/chemistry

  # generation only / optimization only:
  python create_chemagent_dataset.py --split generation
  python create_chemagent_dataset.py --split optimization
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_GENERATION = """\
You are an expert computational chemist. Your task is to design a molecule \
that satisfies the given molecular property constraints and functional group requirements.

You have access to chemistry tools to generate, edit, and validate molecules:
- Propose candidate SMILES structures
- Analyze properties (MW, logP, logD, logS, HBD, HBA, TPSA, rotB, rings, etc.)
- Check functional group presence and counts
- Edit and refine structures iteratively

Instructions:
1. Carefully read all property and fragment constraints in the task.
2. Use the available tools to propose and verify candidate molecules.
3. If the constraints are mutually contradictory and no valid molecule can exist, \
output exactly: <ANSWER>None</ANSWER>
4. Otherwise, output the final SMILES of your best candidate: <ANSWER>YOUR_SMILES</ANSWER>
5. Output exactly ONE <ANSWER> tag."""

SYSTEM_PROMPT_OPTIMIZATION = """\
You are an expert computational chemist specializing in molecular optimization. \
Your task is to modify a given seed molecule so that the result satisfies updated \
property and fragment constraints while preserving specified structural features.

You have access to chemistry tools to edit and validate molecules:
- Apply targeted modifications (bioisosteric replacement, fragment removal/addition, etc.)
- Analyze properties (MW, logP, logD, logS, HBD, HBA, TPSA, rotB, rings, etc.)
- Check functional group presence and counts
- Measure similarity to the seed molecule

Instructions:
1. The seed molecule (reference SMILES) is stated in the task.
2. Apply surgical modifications — do not redesign from scratch unless necessary.
3. Use tools to verify all property and fragment constraints are satisfied.
4. If the constraints are mutually contradictory and no valid molecule can exist, \
output exactly: <ANSWER>None</ANSWER>
5. Otherwise, output the final SMILES of your optimized molecule: <ANSWER>YOUR_SMILES</ANSWER>
6. Output exactly ONE <ANSWER> tag."""

SYSTEM_PROMPTS = {
    "generation":   SYSTEM_PROMPT_GENERATION,
    "optimization": SYSTEM_PROMPT_OPTIMIZATION,
}


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def item_to_verl(item: dict) -> dict:
    """
    Convert one benchmark_v4 item to a verl training row.

    ground_truth stores the constraint spec (properties + fragments) together
    with the infeasible flag. chemistry_reward.py calls tool servers to evaluate
    whether the predicted SMILES satisfies these constraints — there is no single
    reference answer to compare against.
    """
    task_type = item["task_type"]          # "generation" | "optimization"
    data_source = f"mol_{task_type}"

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPTS[task_type]},
        {"role": "user",   "content": item["question"]},
    ]

    # Embed `infeasible` into ground_truth so chemistry_reward.py can detect it
    # from either ground_truth or extra_info.
    answer_with_flag = {**item["answer"], "infeasible": item["infeasible"]}
    ground_truth = json.dumps(answer_with_flag, ensure_ascii=False)

    reward_model = {
        "ground_truth": ground_truth,
        "task_type": task_type,
    }

    extra_info = {
        "id":        item["id"],
        "infeasible": item["infeasible"],
        "ref_smiles": item.get("ref_smiles"),  # None for generation, seed for optimization
        "task_type":  task_type,
        "meta_info":  item.get("meta_info", {}),
    }

    return {
        "prompt":       prompt,
        "data_source":  data_source,
        "reward_model": reward_model,
        "extra_info":   extra_info,
    }


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def convert_and_save(
    items: list[dict],
    output_path: Path,
    shuffle: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    rows = [item_to_verl(item) for item in items]
    df = pd.DataFrame(rows)
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark_v4 JSON → verl parquet for RL training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/users/pimang62/chemistry_tool_agent_clean/data/benchmark_v4",
        help="Directory containing benchmark_v4 JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/verl/data/chemistry_agent",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--split",
        choices=["generation", "optimization", "both"],
        default="both",
        help="Task type(s) to include (default: both)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling train set",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    rl_files = {
        "generation":   data_dir / "scenarios_generation_balanced_rl_10k.json",
        "optimization": data_dir / "scenarios_optimization_balanced_rl_10k.json",
    }
    test_files = {
        "generation":   data_dir / "scenarios_generation_balanced_test_1k.json",
        "optimization": data_dir / "scenarios_optimization_balanced_test_1k.json",
    }

    splits = ["generation", "optimization"] if args.split == "both" else [args.split]

    # Load
    train_items, test_items = [], []
    for sp in splits:
        train_items.extend(load_json(rl_files[sp]))
        test_items.extend(load_json(test_files[sp]))

    print(f"Train: {len(train_items):,} items "
          f"(infeasible: {sum(d['infeasible'] for d in train_items):,})")
    print(f"Test:  {len(test_items):,} items "
          f"(infeasible: {sum(d['infeasible'] for d in test_items):,})")

    # Convert & save
    suffix = "" if args.split == "both" else f"_{args.split}"

    train_df = convert_and_save(
        train_items, output_dir / f"train{suffix}.parquet",
        shuffle=True, seed=args.seed,
    )
    test_df = convert_and_save(
        test_items, output_dir / f"test{suffix}.parquet",
        shuffle=False, seed=args.seed,
    )

    # Summary
    print(f"\nSaved → {output_dir}")
    for label, df in [("train", train_df), ("test", test_df)]:
        by_type = df["data_source"].value_counts().to_dict()
        infeasible = sum(r["infeasible"] for r in df["extra_info"])
        print(f"  {label}{suffix}.parquet : {len(df):,} rows | {by_type} | infeasible={infeasible:,}")

    # Spot-check
    print("\nSample row (train[0]):")
    r = train_df.iloc[0]
    print(f"  data_source : {r['data_source']}")
    print(f"  infeasible  : {r['extra_info']['infeasible']}")
    print(f"  question    : {r['prompt'][1]['content'][:100]}...")
    gt = json.loads(r['reward_model']['ground_truth'])
    print(f"  properties  : {gt.get('properties', [])[:2]} ...")
    print(f"  fragments   : {gt.get('fragments', [])[:2]}")


if __name__ == "__main__":
    main()
