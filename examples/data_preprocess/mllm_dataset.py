import os
import json
import argparse
import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="/data/mllm/janus_r1_sft/data") # flow_grpo/images, reasonr1
    parser.add_argument("--data_path", default="/data/mllm/our_dataset/train/rl_prompt/genevalpp_target_bench_49504.json")
    parser.add_argument("--save_dir", default="/data/mllm/data")

    args = parser.parse_args()

    root_dir = args.root_dir
    data_path = args.data_path
    save_dir = args.save_dir

    dataset = datasets.load_dataset("json", data_files=data_path, split="train")
    train_dataset, val_dataset = dataset.train_test_split(test_size=0.1).values()

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt_id = example["prompt_id"]
            prompt = example["prompt"]
            
            feedback_tuple = example["tuple"]
            vqa_question = example["question"] # for task 2 eval

            data = {
                "data_source": "image_generation",
                "prompt": prompt,
                "ability": "image_unified_generation",
                "reward_model": {"style": "rule", "ground_truth": None, "tuple": feedback_tuple, "vqa_question": vqa_question},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_dataset.to_parquet(os.path.join(save_dir, "train.parquet"))

    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)
    val_dataset.to_parquet(os.path.join(save_dir, "val.parquet"))