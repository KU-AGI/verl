import os
import json
import argparse
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="/home/work/AGILAB/mllm_reasoning/data/ourdataset/train/rl_prompt/v3/train_val/train_162921.json")
    parser.add_argument("--val_path", default="/home/work/AGILAB/mllm_reasoning/data/ourdataset/train/rl_prompt/v3/train_val/val_288.json")
    parser.add_argument("--save_dir", default="/home/work/AGILAB/mllm_reasoning/pimang62/data")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset = datasets.load_dataset("json", data_files=args.train_path, split="train")
    val_dataset = datasets.load_dataset("json", data_files=args.val_path, split="train")

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt_id = example["prompt_id"]
            prompt = example["prompt"]
            
            summary = example.get('summarize', "")
            feedback_tuple = example.get("tuple", "")
            vqa_question = example.get("question", "") 

            d, s, i = prompt_id.split("_")
            if 'focusdiff' in prompt_id:
                d_source = 'focusdiff'
            elif 'aug' in prompt_id:
                d_source = f"long_{s}"
            else:
                d_source = s

            data = {
                "data_source": d_source,
                "prompt": prompt,
                "ability": "image_unified_generation",
                "reward_model": {
                    "style": "rule", 
                    "ground_truth": None, 
                    "summary": summary,
                    "tuple": feedback_tuple, 
                    "vqa_question": vqa_question
                },
                "extra_info": {
                    "split": split, 
                    "index": idx,
                    "prompt_id": prompt_id
                },
            }
            return data

        return process_fn

    print(f"Processing Train dataset ({len(train_dataset)} items)...")
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    
    print(f"Processing Val dataset ({len(val_dataset)} items)...")
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)

    train_save_path = os.path.join(args.save_dir, "train_v2.parquet")
    val_save_path = os.path.join(args.save_dir, "val_v2.parquet")

    train_dataset.to_parquet(train_save_path)
    val_dataset.to_parquet(val_save_path)

    print("-" * 30)
    print(f"Conversion Complete!")
    print(f"Train Parquet: {train_save_path}")
    print(f"Val Parquet: {val_save_path}")