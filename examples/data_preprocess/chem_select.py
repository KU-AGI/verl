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

task_ids_map = {
    "forward": 0,
    "retro": 1,
    "reagent": 2
}

id_to_react_type = {v: k for k, v in task_ids_map.items()}

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

def infer_react_type(x):
    tid = x.get("task_ids", None)
    return id_to_react_type.get(tid, None)

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
    react_type = infer_react_type(x)
    output_smiles = ".".join(x.get(gt_map[react_type], []))
    if output_smiles:
        return {"answers": output_smiles}
    return {"answers": "None"}

def map_messages(x):
    react_type = infer_react_type(x)
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
    parser.add_argument("--local_dir", default="/data/llm-reaction-reasoning/data/orderly/main_training")
    args = parser.parse_args()
    
    concatenated = []

    tasks = ["forward", "retro", "reagent"]
    for task in tasks:
        task_path = f"{args.local_dir}/{task}"
        dataset_paths = sorted(glob(os.path.join(f"{args.local_dir}/{task}", "*.json")))
        
        dataset = concatenate_datasets([load_dataset("json", data_files=dataset_path, split="train", keep_in_memory=False) for dataset_path in dataset_paths])
        # dataset = load_from_disk(f"{args.local_dir}/{task}")
        dataset = dataset.add_column("task_ids", [task_ids_map[task]] * len(dataset))
        dataset = dataset.map(map_messages, load_from_cache_file=False)

        dataset = dataset.filter(lambda x: x["filtered"] == True, load_from_cache_file=False)
        print(f"Task {task} has {len(dataset)} filtered False examples")
        
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(min(3000, len(dataset))))
        dataset = dataset.map(map_answer, load_from_cache_file=False)
        concatenated.append(dataset)

    concatenated = concatenate_datasets(concatenated)
    concatenated.save_to_disk(f"{args.local_dir}/chem_dapo")