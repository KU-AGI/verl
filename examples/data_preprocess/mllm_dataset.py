import os
import json
import argparse
import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="/data/mllm/janus_r1_sft/data")
    parser.add_argument("--data_path", default="/data/mllm/janus_r1_filter_final.json")
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
            aligned_image_path = os.path.join(root_dir, example["aligned_data"][0]["img_path"].lstrip("/"))

            data = {
                "data_source": "image_generation",
                "prompt": [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}],
                "ability": "image_unified_generation",
                "reward_model": {"style": "rule", "ground_truth": aligned_image_path},
                "extra_info": {"split": split, "index": idx, "prompt_id": prompt_id},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_dataset.to_parquet(os.path.join(save_dir, "train.parquet"))

    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)
    val_dataset.to_parquet(os.path.join(save_dir, "val.parquet"))