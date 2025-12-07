import json, re

from tqdm import tqdm
from datasets import load_from_disk
from glob import glob
from pprint import pprint
from rdkit import Chem, RDLogger
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
RDLogger.DisableLog('rdApp.*')  # Disable RDKit warnings


# # Load /data/llm-reaction-reasoning/data/orderly/main_training_full_3000/reagent/incorrect_v4_all/*.arrow
# train_datasets = load_from_disk("/data/llm-reaction-reasoning/data/orderly/main_training_full_3000/reagent/incorrect_v4_all/")
# train_reagent_pool = set()

# for train_entry in tqdm(train_datasets):
#     input_ids = train_entry['input_ids']
#     raw_prompts = tokenizer.decode(input_ids).split("</ANSWER>")[:-1] # Remove the last empty split
#     for raw_prompt in raw_prompts:
#         step6_rationale = raw_prompt.split("## Step 6")[1].split("## Step 7")[0].split("<REFLECTION>")[0].strip()
#         candidate_blocks_set = set()
#         for line in step6_rationale.splitlines():
#             candidate_block = line.strip()
#             if not candidate_block:
#                 continue
#             # Match "number. SMILES"
#             m = re.match(r"^\d+\.\s*(.+)$", line)
#             if m:
#                 candidate_blocks_set.add(m.group(1).strip())
#         train_reagent_pool.add(tuple(candidate_blocks_set))
    







def exact_match(pred_smi: str, gt_smi: str) -> bool:
    """Compare two SMILES strings for chemical equivalence."""
    try:
        mol_pred = Chem.MolFromSmiles(pred_smi)
        mol_gt = Chem.MolFromSmiles(gt_smi)
        
        if mol_pred is None or mol_gt is None:
            return False
            
        return Chem.MolToInchi(mol_pred) == Chem.MolToInchi(mol_gt)
    except Exception:
        return False



task = "reagent"

test_dump_path = "/data/verl/dumps/verl-dapo/reagent_naiverwd_naivecredit_naivespl_GRPO_auxiliary/test/*.jsonl"
test_files = glob(test_dump_path)
test_files = sorted(test_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))[-1:]

val_dump_path = "/data/verl/dumps/verl-dapo/reagent_naiverwd_naivecredit_naivespl_GRPO_auxiliary/val/*.jsonl"
val_files = glob(val_dump_path)
val_files = sorted(val_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))


test_reagent_pool = set()
for test_file in test_files:
    with open(test_file, "r") as f:
        test_data = [json.loads(line) for line in f.readlines()]
    if task == "forward":
        test_data = test_data[0:1000]
    elif task == "retro":
        test_data = test_data[1000:2000]
    elif task == "reagent":
        test_data = test_data[2000:3000]
    for test_entry in test_data:
        gt = test_entry['gts'].split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        try:
            step6_rationale = test_entry["output"].split("## Step 6")[1].split("## Step 7")[0].strip()
        except:
            continue
        predict = test_entry["output"].split("<ANSWER>")[1].split("</ANSWER>")[0].strip()

        # if not exact_match(predict, gt):
        if "<REFLECTION>" not in step6_rationale:
            step6_initial = step6_rationale.split("<REFLECTION>")[0].strip()
            candidate_blocks_set = set()
            for line in step6_initial.splitlines():
                candidate_block = line.strip()
                if not candidate_block:
                    continue
                # Match "number. SMILES"
                m = re.match(r"^\d+\.\s*(.+)$", line)
                if m:
                    candidate_blocks_set.add(m.group(1).strip())
            if len(candidate_blocks_set.intersection({gt})) == 0:
                test_reagent_pool.add(tuple(candidate_blocks_set))



# Load /data/llm-reaction-reasoning/data/orderly/main_training_full_3000/reagent/incorrect_v4_all/*.arrow
train_datasets = load_from_disk("/data/llm-reaction-reasoning/data/orderly/main_training_full_3000/reagent/incorrect_v4_all/")
train_reagent_pool = set()

for train_entry in tqdm(train_datasets):
    input_ids = train_entry['input_ids']
    raw_prompts = tokenizer.decode(input_ids).split("</ANSWER>")[:-1] # Remove the last empty split
    for raw_prompt in raw_prompts:
        step6_rationale = raw_prompt.split("## Step 6")[1].split("## Step 7")[0].split("<REFLECTION>")[0].strip()
        candidate_blocks_set = set()
        for line in step6_rationale.splitlines():
            candidate_block = line.strip()
            if not candidate_block:
                continue
            # Match "number. SMILES"
            m = re.match(r"^\d+\.\s*(.+)$", line)
            if m:
                candidate_blocks_set.add(m.group(1).strip())
        train_reagent_pool.add(tuple(candidate_blocks_set))

    # common_reagent_pools = test_reagent_pool.intersection(tuple(candidate_blocks_set))
    tmp_set = set()
    tmp_set.add(tuple(candidate_blocks_set))
    common_reagent_pools = test_reagent_pool.intersection(tmp_set)
    if common_reagent_pools:
        print(f"Common reagent pools between train and test: {common_reagent_pools}")

