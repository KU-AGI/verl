from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
import argparse
import os, json, re, random
from glob import glob
from tqdm import tqdm


def _sort_key(path):
    name = os.path.basename(path)

    # setN_ 형태면 N을 추출, 없으면 0
    m_set = re.match(r"set(\d+)_", name)
    set_num = int(m_set.group(1)) if m_set else 0

    # 파일명 안의 모든 숫자 추출
    nums = list(map(int, re.findall(r"(\d+)", name)))
    # 보통은 [setN, start, end] 또는 [start, end] 형태
    if m_set:
        start_num = nums[1]  # setN 다음 첫 숫자
    else:
        start_num = nums[0]  # 그냥 시작 숫자

    return (set_num, start_num)


system_prompt_template = "You are a chemist."

user_prompt_template = {
    "forward": [
        "[SMILES] Considering the given starting materials, what might be the resulting product in a chemical reaction?",
        "Consider that for a chemical reaction, if [SMILES] is/are the precursor, what can be the product?",
        "[SMILES] Given the above precursor, what could be a probable product of their reaction?",
        "Predict a possible product from the listed precursor. [SMILES]",
        "Using [SMILES] as the precursor, tell me the potential product.",
        "Please provide a feasible product that could be formed using these precursor: [SMILES] .",
        "A chemical reaction has started with the substance(s) [SMILES] as the precursor, what could be a probable product?",
        "Propose a potential product given these precursor. [SMILES]",
        "Can you tell me the potential product of a chemical reaction that uses [SMILES] as the precursor?",
        "Based on the given precursor: [SMILES], what product could potentially be produced?",
        "[SMILES] Based on the precursor given above, suggest a possible product.",
        "Given the following precursor, please provide a possible product. [SMILES]",
        "Predict the product of a chemical reaction with [SMILES] as the precursor.",
    ],
    "retro":[
        "With the given product [SMILES], suggest some likely reactants that were used in its synthesis.",
        "[SMILES] Given the product provided, propose some possible reactants that could have been employed in its formation.",
        "Can you list the reactants that might result in the chemical product [SMILES] ?",
        "To synthesis [SMILES], what are the possible reactants? Write in the SMILES representation.",
        "Can you identify the reactant(s) that might result in the given product [SMILES] ?",
        "Do retrosynthesis with the product [SMILES] .",
        "Based on the given product, provide some plausible reactants that might have been utilized to prepare it. [SMILES]",
        "Suggest possible substances that may have been involved in the synthesis of the presented compound. [SMILES]",
        "Identify possible reactants that could have been used to create the specified product. [SMILES]",
        "Given the following product, please provide possible reactants. [SMILES]",
        "Provide the potential reactants that may be used to produce the product [SMILES] .",
        "Could you tell which reactants might have been used to generate the following product? [SMILES]",
        "What reactants could lead to the production of the following product? [SMILES]",
    ],
}


smiles_map = {
    "forward": "{reactants}.{reagents}.{solvents}",
    "retro": "{products}",
}

gt_map = {
    "forward": "products",
    "retro": "reactants",
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


def unify_struct(x):
    x['rationale'] = x['rationale'][0]
    x['filtered'] = x['filtered'][0]

    info = x.pop('info')
    new_info = {}
    if x['task'] == 'forward':
        new_info['reactant_str'] = info['reactant_str']
        new_info['reagent_str'] = info['reagent_str']
        new_info['solvent_str'] = info['solvent_str']
        new_info['precursor_with_solvent_smiles_stat'] = info['precursor_with_solvent_smiles_stat']
        new_info['reactant_funcgroup_and_count'] = info['reactant_funcgroup_and_count']
        new_info['template'] = info['template']
    elif x['task'] == 'retro':
        new_info['bond_funcgroup_and_count'] = info['bond_funcgroup_and_count']
        new_info['product_smiles_stat'] = info['product_smiles_stat']
        new_info['bond_list'] = info['bond_list']
        new_info['synthons_list_new'] = info['synthons_list_new']
        new_info['synthetic_equivalents_list'] = info['synthetic_equivalents_list']
    x['info_json_str'] = json.dumps(new_info)
    return x


def map_messages(x, split):
    react_type = x.get("task", None)
    if split == "train":
        user_prompt = random.choice(user_prompt_template[react_type]).replace("[SMILES]", get_smiles(x, smiles_map[react_type]))
    else: # validation and test: use the first one
        user_prompt = user_prompt_template[react_type][0].replace("[SMILES]", get_smiles(x, smiles_map[react_type]))
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt_template
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    }


def append_info(data):
    try:
        messages = data.pop("messages", None)
        answer = data.pop("answers", None)
        rationale = data.pop("rationale", None)

        solution = ["<think>\n" + rationale + "</think>\n\n" + "<ANSWER>\n" + answer + "\n</ANSWER>"] if rationale is not None else ["<think>\n" + "DUMMY" + "</think>\n\n" + "<ANSWER>\n" + answer + "\n</ANSWER>"]
        solution = "".join(solution)

        new_data = {
            "data_source": "chem_dapo",
            "prompt": messages,
            "ability": "chemistry",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                k: v for k, v in data.items()
            },
        }
        return new_data
    except Exception as e:
        print(f"Error in extract_info: {e}")
        return {
            "data_source": None,
            "prompt": None,
            "ability": None,
            "reward_model": None,
            "extra_info": None,
        }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="/data/llm-reaction-reasoning/data/orderly/main_training_v11")
    parser.add_argument("--val_dir", default="/data/llm-reaction-reasoning/data/orderly/excluded_val_test")
    parser.add_argument("--test_dir", default="/data/llm-reaction-reasoning/data/orderly/excluded_val_test")
    parser.add_argument("--output_dir", default="/data/verl/data")
    parser.add_argument("--dir_name", default="chem_dapo")
    parser.add_argument("--n_samples", type=int, default=999999999)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    concatenated = []
    args.train = True

    tasks = ["forward", "retro"]
    for task in tasks:
        if args.train:
            dataset_paths = sorted(glob(os.path.join(f"{args.train_dir}/{task}", "*.json")), key=_sort_key)[-10:]
            split = "train"
        elif args.val:
            # dataset_paths = sorted(glob(os.path.join(f"{args.val_dir}", f"{task}*val.json")))
            dataset_paths = [
                f"/data/llm-reaction-reasoning/data/orderly/excluded_val_test/{task}_val.json",
            ]
            split = "val"
        elif args.test:
            # dataset_paths = sorted(glob(os.path.join(f"{args.test_dir}", f"{task}*test.json")))
            dataset_paths = [
                # f"/data/llm-reaction-reasoning/data/orderly/excluded_val_test/{task}_ood_carbon_test.json",
                # f"/data/llm-reaction-reasoning/data/orderly/excluded_val_test/{task}_ood_length_test.json",
                f"/data/llm-reaction-reasoning/data/orderly/excluded_val_test/{task}_test.json",
            ]
            split = "test"
        else:
            raise ValueError("Either --train or --val or --test must be specified")

        # try:
        #     dataset = concatenate_datasets([load_dataset("json", data_files=dataset_path, split="train", keep_in_memory=False) for dataset_path in dataset_paths])
        # except:
        data_list = []
        for dataset_path in tqdm(dataset_paths, desc=f"Loading {task} {split} datasets"):
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for d in data:
                d['dataset_path'] = dataset_path.split("/")[-1].split(".json")[0]
                if d['filtered'] == False:
                    continue
                for k, v in d["info"].items():
                    if isinstance(d["info"][k], list):
                        for key in d["info"][k]:
                            d["info"]
                    d["info"][k] = str(v)
            
                data_list.append(d)
        
        dataset = Dataset.from_list(data_list)

        # reverse dataset row order
        dataset = dataset.select(list(reversed(range(len(dataset)))))

        dataset = dataset.add_column("task", [task] * len(dataset))
        dataset = dataset.map(map_messages, fn_kwargs={"split": split}, load_from_cache_file=False)
        dataset = dataset.map(unify_struct, load_from_cache_file=False)

        dataset = dataset.remove_columns("filtered")
        print(f"Task {task} has {len(dataset)} filtered True examples")
        
        # dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(min(args.n_samples, len(dataset))))
        dataset = dataset.map(map_answer, load_from_cache_file=False)
        concatenated.append(dataset)

    concatenated = concatenate_datasets(concatenated)

    if args.train:
        dataset = concatenated.map(function=append_info, load_from_cache_file=False)
        dataset = dataset.filter(lambda x: x["data_source"] is not None)
        dataset.to_parquet(os.path.join(args.output_dir, f"{args.dir_name}", "syntheticreact_train.parquet"))
    elif args.val:
        dataset = concatenated.map(function=append_info, load_from_cache_file=False)
        # dataset = dataset.filter(lambda x: x["data_source"] is not None)
        dataset.to_parquet(os.path.join(args.output_dir, f"{args.dir_name}", "syntheticreact_val.parquet"))
    elif args.test:
        dataset = concatenated.map(function=append_info, load_from_cache_file=False)
        # dataset = dataset.filter(lambda x: x["data_source"] is not None)
        dataset.to_parquet(os.path.join(args.output_dir, f"{args.dir_name}", "syntheticreact_test.parquet"))
