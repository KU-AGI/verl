import os
import json
import argparse
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_path", default="/home/work/AGILAB/mllm_reasoning/data/ourdataset/train/rl_prompt/v5/final_train/val_288.json")
    parser.add_argument("--save_dir", default="/home/work/AGILAB/mllm_reasoning/pimang62/data")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    val_dataset = datasets.load_dataset("json", data_files=args.val_path, split="train")

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt_id = example["prompt_id"]
            prompt = example["prompt"]
            
            summary = example.get('summary', example.get('summarize', ""))
            feedback_tuple = example.get("tuple", "")
            vqa_question = example.get("question", "") 

            category = example.get("category", "")
            if 'geneval' in prompt_id:
                # geneval_counting -> geneval/counting, geneval_color_attr -> geneval/color_attr
                d_source = category.replace('geneval_', 'geneval/', 1)
            elif 't2icompbench' in prompt_id:
                d_source = category.replace('t2icompbench_', 't2icompbench/', 1)
            elif 'dpgbench' in prompt_id:
                d_source = 'dpgbench'
            elif 'focusdiff' in prompt_id:
                d_source = 'focusdiff'
            elif 'aug' in prompt_id:
                parts = prompt_id.split("_")
                d_source = f"long_{parts[1]}"
            elif 'conpair' in prompt_id:
                d_source = 'conpair'
            elif 'longalign' in prompt_id:
                d_source = 'longalign'
            else:
                d_source = category

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
    
    print(f"Processing Val dataset ({len(val_dataset)} items)...")
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)

    val_save_path = os.path.join(args.save_dir, "val_v5.parquet")

    val_dataset.to_parquet(val_save_path)

    print("-" * 30)
    print(f"Conversion Complete!")
    print(f"Val Parquet: {val_save_path}")