from datasets import load_dataset, load_from_disk, concatenate_datasets
import argparse
import os, json, re, random
from glob import glob


def extract_forward_info(rationale: str):
    if "`**[atom]:[number]**` as follows." in rationale:
        step4_highlighted_smiles = rationale.split("## Step 4")[1].split("## Step 5")[0].split("`**[atom]:[number]**` as follows.")[1].strip()
        step4_highlighted_smiles = step4_highlighted_smiles.split("\n")[-1].strip()
        step4_highlighted_smiles = step4_highlighted_smiles.strip("*")
    else:
        step4_highlighted_smiles = "(No highlighted atoms)"

    step5_bond_pairs = rationale.split("## Step 5")[1].split("## Step 6")[0].split("the following bonds will be formed.")[1].strip()
    if "List the pairs of atoms that are likely to form bonds in the product, based on the highlighted atoms in the precursor:" in step5_bond_pairs:
        step5_bond_pairs = step5_bond_pairs.split("List the pairs of atoms that are likely to form bonds in the product, based on the highlighted atoms in the precursor:")[1].strip()
    if "List the pairs of atoms that are likely to form bonds in the product, based on the highlighted atoms in the precursor." in step5_bond_pairs:
        step5_bond_pairs = step5_bond_pairs.split("List the pairs of atoms that are likely to form bonds in the product, based on the highlighted atoms in the precursor.")[1].strip()

    step6_intermediate_smiles = rationale.split("## Step 6")[1].split("## Step 7")[0].split("it can be expressed as")[-1].split("If this is canonicalized")[0].strip().strip(".").strip()
    step6_canonical_smiles = rationale.split("## Step 6")[1].split("## Step 7")[0].split("If this is canonicalized, it becomes ")[-1].split(". From the canonicalized SMILES")[0].strip()
    step6_tagged_smiles = rationale.split("## Step 6")[1].split("## Step 7")[0].split("we get the following:")[-1].strip()

    step5_bond_pairs_list = []
    if step5_bond_pairs != '':
        for bond_pair in step5_bond_pairs.split("\n"):
            bond_pair_str = bond_pair.strip('`').strip('*').strip("(").strip(")").strip()
            atom1, atom2, bond_type = bond_pair_str.split(",")
            atom1 = int(atom1.strip())
            atom2 = int(atom2.strip())
            bond_type = bond_type.strip()
            step5_bond_pairs_list.append([str(atom1), str(atom2), bond_type.strip()])


    info = {
        "forward": {
            "generated_substructure_from_precursor": ["DUMMY"],
            "masked_smiles": ["DUMMY"],
            "reactive_atoms_smiles_str": step4_highlighted_smiles,
            "reactive_atom_bonds": step5_bond_pairs_list,
            "product_changes_tagged": step6_tagged_smiles,
            "intermediate_smiles": step6_intermediate_smiles,
            "canonical_intermediate": step6_canonical_smiles,
        },

        "retro": {
            "masked_smiles": ["DUMMY"],
            "atom_mapping": "DUMMY",
            "bond_list": [["", "", ""]],
            "synthons_list": [""],
            "synthons_list_new": [""],
            "synthetic_equivalents": [""],
        },

        "reagent": {
            "masked_smiles": ["DUMMY"],
            "generated_substructure_list": ["DUMMY"],
            "removed_substructure_list": ["DUMMY"],
            "reagents": [""],
            "correct_reagent_number": "",
        }


    }
    return info


def extract_retro_info(rationale: str):
    step4_rationale = rationale.split("## Step 4")[1].split("## Step 5")[0]
    step5_rationale = rationale.split("## Step 5")[1].split("## Step 6")[0].split("\n\n")[0]
    step6_rationale = rationale.split("## Step 6")[1].split("## Step 7")[0]
    step7_rationale = rationale.split("## Step 7")[1]

    if "it does not appear that" in step5_rationale:
        bond_disconnections_str = "(No bond disconnections)"
        bond_disconnections_list = []
    else:
        bond_disconnections_list = []
        for bond_disconnection in step5_rationale.split('\n')[2:]:
            atom1, atom2, bond_type = re.split(",|:", bond_disconnection)
            atom1 = int(atom1.strip())
            atom2 = int(atom2.strip())
            bond_type = bond_type.strip()
            # bond_disconnections_list.append([atom1, atom2, bond_type])
            bond_disconnections_list.append([str(atom1), str(atom2), bond_type.strip()])
    pattern = re.compile(r"\b\d+\.\s*([^\n\r]+)")
    if "These synthons can be grouped as follows." in step6_rationale:
        single_part = step6_rationale.split("These synthons can be grouped as follows.")[0].strip()
        multi_part = step6_rationale.split("These synthons can be grouped as follows.")[1].strip()
        # group synthons
        synthons = [m.group(1).strip() for m in pattern.finditer(single_part)]
        # synthons = list(set(synthons))
        synthon_str = "\n".join([f"{i+1}. {syn}" for i, syn in enumerate(synthons)])

        # group synthons
        grouped_synthons = [m.group(1).strip() for m in pattern.finditer(multi_part)]
        grouped_synthon_str = "\n".join([f"{i+1}. {syn}" for i, syn in enumerate(grouped_synthons)])

        synthon_full_str = f"{synthon_str}\n\nThese synthons can be grouped as follows.\n{grouped_synthon_str}"
    elif "These synthons can be grouped as follows:" in step6_rationale:
        single_part = step6_rationale.split("These synthons can be grouped as follows:")[0].strip()
        multi_part = step6_rationale.split("These synthons can be grouped as follows:")[1].strip()
        # group synthons
        synthons = [m.group(1).strip() for m in pattern.finditer(single_part)]
        # synthons = list(set(synthons))
        synthon_str = "\n".join([f"{i+1}. {syn}" for i, syn in enumerate(synthons)])

        # group synthons
        grouped_synthons = [m.group(1).strip() for m in pattern.finditer(multi_part)]
        grouped_synthon_str = "\n".join([f"{i+1}. {syn}" for i, syn in enumerate(grouped_synthons)])

        synthon_full_str = f"{synthon_str}\n\nThese synthons can be grouped as follows.\n{grouped_synthon_str}"
    else:
        synthons = [m.group(1).strip() for m in pattern.finditer(step6_rationale)]
        grouped_synthons = synthons
        # synthons = list(set(synthons))
        synthon_full_str = "\n".join([f"{i+1}. {syn}" for i, syn in enumerate(synthons)])

    pattern = re.compile(r"\b\d+\.\s*([^\n\r]+)")
    synthetic_equivalents = [m.group(1).strip() for m in pattern.finditer(step7_rationale)]
    synthetic_equivalents_str = "\n".join([f"{i+1}. {syn}" for i, syn in enumerate(synthetic_equivalents)])

    assert len(grouped_synthons) == len(synthetic_equivalents), f"len(grouped_synthons) != len(synthetic_equivalents): {len(grouped_synthons)} != {len(synthetic_equivalents)}"

    return {
        "forward": {
            "generated_substructure_from_precursor": ["DUMMY"],
            "masked_smiles": ["DUMMY"],
            "reactive_atoms_smiles_str": "",
            "reactive_atom_bonds": [["", "", ""]],
            "product_changes_tagged": "",
            "intermediate_smiles": "",
            "canonical_intermediate": "",
        },
        "retro": {
            "masked_smiles": ["DUMMY"],
            "atom_mapping": "DUMMY",
            "bond_list": bond_disconnections_list,
            "synthons_list": synthons,
            "synthons_list_new": grouped_synthons,
            "synthetic_equivalents": synthetic_equivalents,
        },
        "reagent": {
            "masked_smiles": ["DUMMY"],
            "generated_substructure_list": ["DUMMY"],
            "removed_substructure_list": ["DUMMY"],
            "reagents": [""],
            "correct_reagent_number": "",
        }
    }


def extract_reagent_info(rationale: str):
    step6_rationale = rationale.split("## Step 6")[1].split("## Step 7")[0]
    step7_rationale = rationale.split("## Step 7")[1]

    # Extract the candidate reagents block and parse as list of SMILES
    candidate_reagents_block = step6_rationale.split("The candidate reagents are as follows:")[-1].strip()
    # Each line is like "1. SMILES"
    candidate_reagents_list = []
    for line in candidate_reagents_block.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match "number. SMILES"
        m = re.match(r"^\d+\.\s*(.+)$", line)
        if m:
            candidate_reagents_list.append(m.group(1).strip())
    
    candidate_reagents_list_str = "\n".join([f"{i+1}. {reagent}" for i, reagent in enumerate(candidate_reagents_list)])
    selected_reagent_number = "(No selected reagent)"
    match = re.search(r"reagent\s*(\d+)", step7_rationale, re.IGNORECASE)
    if match:
        selected_reagent_number = int(match.group(1))
    return {
        "forward": {
            "generated_substructure_from_precursor": ["DUMMY"],
            "masked_smiles": ["DUMMY"],
            "reactive_atoms_smiles_str": "",
            "reactive_atom_bonds": [["", "", ""]],
            "product_changes_tagged": "",
            "intermediate_smiles": "",
            "canonical_intermediate": "",
        },

        "retro": {
            "masked_smiles": ["DUMMY"],
            "atom_mapping": "DUMMY",
            "bond_list": [["", "", ""]],
            "synthons_list": [""],
            "synthons_list_new": [""],
            "synthetic_equivalents": [""],
        },

        "reagent": {
            "masked_smiles": ["DUMMY"],
            "generated_substructure_list": ["DUMMY"],
            "removed_substructure_list": ["DUMMY"],
            "reagents": candidate_reagents_list,
            "correct_reagent_number": str(selected_reagent_number),
        }
    }


system_prompt_template = "You are a chemist."

user_prompt_template = {
    "forward": [
        "[SMILES] Considering the given starting materials, what might be the resulting product in a chemical reaction?",
        "Consider that for a chemical reaction, if [SMILES] is/are the reactants and reagents, what can be the product?",
        "[SMILES] Given the above reactants and reagents, what could be a probable product of their reaction?",
        "Predict a possible product from the listed reactants and reagents. [SMILES]",
        "Using [SMILES] as the reactants and reagents, tell me the potential product.",
        "Please provide a feasible product that could be formed using these reactants and reagents: [SMILES] .",
        "A chemical reaction has started with the substance(s) [SMILES] as the reactants and reagents, what could be a probable product?",
        "Propose a potential product given these reactants and reagents. [SMILES]",
        "Can you tell me the potential product of a chemical reaction that uses [SMILES] as the reactants and reagents?",
        "Based on the given reactants and reagents: [SMILES], what product could potentially be produced?",
        "[SMILES] Based on the reactants and reagents given above, suggest a possible product.",
        "Given the following reactants and reagents, please provide a possible product. [SMILES]",
        "Predict the product of a chemical reaction with [SMILES] as the reactants and reagents.",
    ],
    "retro": [
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
    "reagent": [
        "Please suggest some possible reagents that could have been used in the following chemical reaction [SMILES].",
        "Given this chemical reaction [SMILES], what are some reagents that could have been used?",
        "Given the following reaction [SMILES], what are some possible reagents that could have been utilized?",
        "[SMILES] Based on the given chemical reaction, can you propose some likely reagents that might have been utilized?",
        "[SMILES] From the provided chemical reaction, propose some possible reagents that could have been used.",
        "Can you provide potential reagents for the following chemical reaction? [SMILES]",
        "[SMILES] Please propose potential reagents that might have been utilized in the provided chemical reaction.",
        "What reagents could have been utilized in the following chemical reaction? [SMILES]",
        "Based on the given chemical reaction [SMILES], suggest some possible reagents.",
        "Given the following chemical reaction [SMILES], what are some potential reagents that could have been employed?",
        "Please provide possible reagents based on the following chemical reaction [SMILES].",
        "Can you suggest some reagents that might have been used in the given chemical reaction? [SMILES]",
    ],
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
        messages = data.pop("messages")
        answer = data.pop("answers")
        rationale = data.pop("rationale")

        solution = ["<think>\n" + rationale + "</think>\n\n" + "<ANSWER>\n" + answer + "\n</ANSWER>"]
        solution = "".join(solution)

        if data["task"] == "forward":
            supporting_info = extract_forward_info(rationale)
        elif data["task"] == "retro":
            supporting_info = extract_retro_info(rationale)
        elif data["task"] == "reagent":
            supporting_info = extract_reagent_info(rationale)
        else:
            raise ValueError(f"Unknown task: {data['task']}")

        # Choose extra columns
        extra_columns = ["task", "rxn_str", "reactants", "reagents", "products", "solvent", "yields", "class_name"]
        
        data = {
            "data_source": "chem_dapo",
            "prompt": messages,
            "ability": "chemistry",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                k: v for k, v in data.items() if k in extra_columns
            },
        }
        data["extra_info"]["supporting_info"] = supporting_info
        return data
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
    parser.add_argument("--train_dir", default="/data/llm-reaction-reasoning/data/orderly/main_training")
    parser.add_argument("--val_dir", default="/data/llm-reaction-reasoning/data/orderly/balanced_val")
    parser.add_argument("--test_dir", default="/data/llm-reaction-reasoning/data/orderly/excluded_test")
    parser.add_argument("--output_dir", default="/data/verl/data")
    parser.add_argument("--dir_name", default="chem_dapo")
    parser.add_argument("--n_samples", type=int, default=999999999)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    concatenated = []
    args.test = True

    tasks = ["forward", "retro", "reagent"]
    for task in tasks:
        if args.train:
            dataset_paths = sorted(glob(os.path.join(f"{args.train_dir}/{task}", "*.json")))
            split = "train"
        elif args.val:
            dataset_paths = sorted(glob(os.path.join(f"{args.val_dir}", f"balanced_{task}_val_v10_required.jsonl")))
            split = "val"
        elif args.test:
            dataset_paths = sorted(glob(os.path.join(f"{args.test_dir}", f"excluded_{task}_test_v10_required.jsonl")))
            split = "test"
        else:
            raise ValueError("Either --train or --val or --test must be specified")

        dataset = concatenate_datasets([load_dataset("json", data_files=dataset_path, split="train", keep_in_memory=False) for dataset_path in dataset_paths])

        dataset = dataset.add_column("task", [task] * len(dataset))
        dataset = dataset.map(map_messages, fn_kwargs={"split": split}, load_from_cache_file=False)

        if args.train:
            dataset = dataset.filter(lambda x: x["filtered"] == True, load_from_cache_file=False)
            dataset = dataset.remove_columns("filtered")
            dataset = dataset.filter(lambda x: "For example" not in x["rationale"])
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
