import datasets
import ast

from pprint import pprint
from tqdm import tqdm




def fix_forward_supporting_info(example):
    si = example["extra_info"]["supporting_info"]
    reactive_atom_bonds = ast.literal_eval(si["reactive_atom_bonds"])
    converted_reactive_atom_bonds = [
        [str(x) if isinstance(x, int) else x for x in sub]
        for sub in reactive_atom_bonds
    ]

    example["extra_info"]["supporting_info"] = {
        "forward": {
            "canonical_intermediate": si["canonical_intermediate"],
            "generated_substructure_from_precursor": si["generated_substructure_from_precursor"],
            "intermediate_smiles": si["intermediate_smiles"],
            "masked_smiles": si["masked_smiles"],
            "product_changes_tagged": si["product_changes_tagged"],
            "reactive_atom_bonds": converted_reactive_atom_bonds,
            "reactive_atoms_smiles_str": si["reactive_atoms_smiles_str"],
        },
        "retro": {
            "atom_mapping": "DUMMY",
            "bond_list": [["", "", ""]],
            "masked_smiles": ["DUMMY"],
            "synthetic_equivalents": [""],
            "synthons_list": [""],
            "synthons_list_new": [""],
        },
        "reagent": {
            "correct_reagent_number": "",
            "generated_substructure_list": ["DUMMY"],
            "masked_smiles": ["DUMMY"],
            "reagents": [""],
            "removed_substructure_list": ["DUMMY"],
        }
    }
    return example

data = datasets.load_dataset("parquet", data_files="/data/verl/data/chem_dapo/syntheticreact_remained_forward_samples_for_train.parquet", split="train")
data_fixed = data.map(fix_forward_supporting_info)
# resave data with fixed supporting_info
data_fixed.to_parquet("/data/verl/data/chem_dapo/syntheticreact_remained_forward_samples_for_train_fixed.parquet")

data = datasets.load_dataset("parquet", data_files="/data/verl/data/chem_dapo/syntheticreact_extra_3to7_30k_forward_samples_for_train.parquet", split="train")
data_fixed = data.map(fix_forward_supporting_info)
# resave data with fixed supporting_info
data_fixed.to_parquet("/data/verl/data/chem_dapo/syntheticreact_extra_3to7_30k_forward_samples_for_train_fixed.parquet")



# Load /data/verl/data/chem_dapo/syntheticreact_extra_3to7_20k_allfalut_10k_reagent_samples_for_train.parquet
# data1 = datasets.load_dataset("parquet", data_files="/data/verl/data/chem_dapo/syntheticreact_remained_retro_samples_for_train.parquet", split="train")


# # Load /data/verl/data/chem_dapo/syntheticreact_auxiliary_reagent_train.parquet
# data2 = datasets.load_dataset("parquet", data_files="/data/verl/data/chem_dapo/syntheticreact_retro_train.parquet", split="train")


# for i in tqdm(range(len(data1))):
#     fixed_supporting_info = {
#         "forward": {
#             'canonical_intermediate': data1[i]['extra_info']['supporting_info']['canonical_intermediate'],
#             'generated_substructure_from_precursor': data1[i]['extra_info']['supporting_info']['generated_substructure_from_precursor'],
#             'intermediate_smiles': data1[i]['extra_info']['supporting_info']['intermediate_smiles'],
#             'masked_smiles': data1[i]['extra_info']['supporting_info']['masked_smiles'],
#             'product_changes_tagged': data1[i]['extra_info']['supporting_info']['product_changes_tagged'],
#             'reactive_atom_bonds': data1[i]['extra_info']['supporting_info']['reactive_atom_bonds'],
#             'reactive_atoms_smiles_str': data1[i]['extra_info']['supporting_info']['reactive_atoms_smiles_str'],
#         },
#         "retro": {
#             'atom_mapping': 
#             'bond_list': 
#             'masked_smiles': 
#             'synthetic_equivalents': [''],
#             'synthons_list': [''],
#             'synthons_list_new': [''],
#         },
#         "reagent": {
#             'correct_reagent_number': '',
#             'generated_substructure_list': ['DUMMY'],
#             'masked_smiles': ['DUMMY'],
#             'reagents': [''],
#             'removed_substructure_list': ['DUMMY']
#         }
#     }
#     data1[i]['extra_info']['supporting_info'] = fixed_supporting_info

# # resave data1 with fixed supporting_info
# data1.to_parquet("/data/verl/data/chem_dapo/syntheticreact_remained_forward_samples_for_train_fixed.parquet")


