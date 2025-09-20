from datasets import load_dataset, load_from_disk, concatenate_datasets
import argparse
import os, json, re
from glob import glob

system_prompt_template = "You are a chemist."

user_prompt_template = {
    "forward": "[SMILES] Given the precursor, what is the expected product of the chemical reaction?",
    "retro": "Given the product [SMILES], what are some likely reactants that could have been used in its synthesis?",
    "reagent": "Please suggest some possible reagents that could have been used in the following chemical reaction [SMILES].",
}

smiles_map = {
    "forward": "{reactants}.{reagents}",
    "retro": "{products}",
    "reagent": "{reactants}>>{products}",
}

gt_map = {
    "forward": "products",
    "retro": "reactants",
    "reagent": "reagents",
}

def get_smiles(x, template):
    template_keys = re.findall(r'(?<!\{)\{([^{}]+)\}(?!\})', template)  # "{}" but not "{{}}"
    input_smiles = template

    elements = {}
    for key in template_keys:
        joined_key = ".".join(x[key]) # in dataset
        elements[key] = joined_key
       
    input_smiles = input_smiles.format(**elements)
    return input_smiles

def map_answer(x):
    react_type = x.get("task", None)
    output_smiles = ".".join(x.get(gt_map[react_type], []))
    if output_smiles:
        return {"answers": output_smiles}
    return {"answers": "None"}

def map_messages(x):
    react_type = x.get("task", None)
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt_template
            },
            {
                "role": "user",
                "content": user_prompt_template[react_type].replace("[SMILES]", get_smiles(x, smiles_map[react_type]))
            }
        ]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="/data/llm-reaction-reasoning/data/orderly/main_training")
    parser.add_argument("--val_dir", default="/data/llm-reaction-reasoning/data/orderly/balanced_val")
    parser.add_argument("--test_dir", default="/data/llm-reaction-reasoning/data/orderly/excluded_test")
    parser.add_argument("--output_dir", default="/data/verl/data")
    parser.add_argument("--dir_name", default="chem_dapo")
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    concatenated = []

    tasks = ["forward", "retro", "reagent"]
    for task in tasks:
        if args.train:
            dataset_paths = sorted(glob(os.path.join(f"{args.train_dir}/{task}", "*.json")))
        elif args.val:
            dataset_paths = sorted(glob(os.path.join(f"{args.val_dir}", f"balanced_{task}_val_v10_required.jsonl")))
        elif args.test:
            dataset_paths = sorted(glob(os.path.join(f"{args.test_dir}", f"excluded_{task}_test_v10_required.jsonl")))
        else:
            raise ValueError("Either --train or --val or --test must be specified")

        dataset = concatenate_datasets([load_dataset("json", data_files=dataset_path, split="train", keep_in_memory=False) for dataset_path in dataset_paths])

        dataset = dataset.add_column("task", [task] * len(dataset))
        dataset = dataset.map(map_messages, load_from_cache_file=False)

        if args.train:
            dataset = dataset.filter(lambda x: x["filtered"] == True, load_from_cache_file=False)
            print(f"Task {task} has {len(dataset)} filtered True examples")
        
        # dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(min(args.n_samples, len(dataset))))
        dataset = dataset.map(map_answer, load_from_cache_file=False)
        concatenated.append(dataset)

    concatenated = concatenate_datasets(concatenated)

    if args.train:
        concatenated.save_to_disk(f"{args.output_dir}/{args.dir_name}/train")
    elif args.val:
        concatenated.save_to_disk(f"{args.output_dir}/{args.dir_name}/val")
    elif args.test:
        concatenated.save_to_disk(f"{args.output_dir}/{args.dir_name}/test")
