from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset, DatasetDict
import argparse
import os, json, re, random
from glob import glob



"""
{'ability': 'chemistry',
 'class_name': 'Fischer-Speier esterification',
 'data_source': 'chem_dapo',
 'extra_info': {'class_name': 'Fischer-Speier esterification',
                'products': ['COC(=O)/C=C/c1ccc(O)c(OC)c1'],
                'reactants': ['CO', 'COc1cc(C=CC(=O)O)ccc1O'],
                'reagents': ['O=S(=O)(O)O'],
                'rxn_str': 'CO.COc1cc(C=CC(=O)O)ccc1O.O=S(=O)(O)O>>COC(=O)/C=C/c1ccc(O)c(OC)c1',
                'supporting_info': {'forward': {'canonical_intermediate': '',
                                                'generated_substructure_from_precursor': ['DUMMY'],
                                                'intermediate_smiles': '',
                                                'masked_smiles': ['DUMMY'],
                                                'product_changes_tagged': '',
                                                'reactive_atom_bonds': [['',
                                                                         '',
                                                                         '']],
                                                'reactive_atoms_smiles_str': ''},
                                    'reagent': {'correct_reagent_number': '3',
                                                'generated_substructure_list': ['DUMMY'],
                                                'masked_smiles': ['DUMMY'],
                                                'reagents': ['O=S(=O)(O)C(F)(F)F',
                                                             'O=S(=O)([O-])O',
                                                             'O=S(=O)(O)O'],
                                                'removed_substructure_list': ['DUMMY']},
                                    'retro': {'atom_mapping': 'DUMMY',
                                              'bond_list': [['', '', '']],
                                              'masked_smiles': ['DUMMY'],
                                              'synthetic_equivalents': [''],
                                              'synthons_list': [''],
                                              'synthons_list_new': ['']}},
                'task': 'reagent',
                'yields': []},
 'products': ['COC(=O)/C=C/c1ccc(O)c(OC)c1'],
 'prompt': [{'content': 'You are a chemist.', 'role': 'system'},
            {'content': 'Please suggest some possible reagents that could have '
                        'been used in the following chemical reaction '
                        'CO.COc1cc(C=CC(=O)O)ccc1O>>COC(=O)/C=C/c1ccc(O)c(OC)c1.',
             'role': 'user'}],
 'reactants': ['CO', 'COc1cc(C=CC(=O)O)ccc1O'],
 'reagents': ['O=S(=O)(O)O'],
 'reward_model': {'ground_truth': '<think>\n'
                                  '## Step 1: Understanding molecular roles  \n'
                                  'The reactant is CO.COc1cc(C=CC(=O)O)ccc1O '
                                  'and the product is '
                                  'COC(=O)/C=C/c1ccc(O)c(OC)c1.\n'
                                  '\n'
                                  '## Step 2: Analysis of main functional '
                                  'groups  \n'
                                  'The reactant contains a carboxylic acid '
                                  'group (masked SMILES: `CO_`) and a hydroxyl '
                                  'group (masked SMILES: `C_`). The product '
                                  'contains an ester group (masked SMILES: '
                                  '`_C(=O)O_`) and an epoxide ring (masked '
                                  'SMILES: `COC(=O)/C_`).\n'
                                  '\n'
                                  '## Step 3: Predict possible reaction '
                                  'mechanisms  \n'
                                  'The transformation suggests an '
                                  'esterification reaction, where a carboxylic '
                                  'acid reacts with an alcohol to form an '
                                  'ester. This mechanism involves the '
                                  'protonation of the carbonyl oxygen, '
                                  'nucleophilic attack by the alcohol, '
                                  'formation of a tetrahedral intermediate, '
                                  'and subsequent elimination of water to '
                                  'yield the ester.\n'
                                  '\n'
                                  '## Step 4: Δ analysis  \n'
                                  'The functional group/substructure change '
                                  '(Δ) between reactant and product involves '
                                  'the removal of a carboxylic acid group '
                                  '(masked SMILES: `CO_`) and a hydroxyl group '
                                  '(masked SMILES: `C_`), while new '
                                  'substructures such as an ester group '
                                  '(masked SMILES: `_C(=O)O_`) and an epoxide '
                                  'ring (masked SMILES: `COC(=O)/C_`) are '
                                  'introduced.\n'
                                  '\n'
                                  '## Step 5: Derivation of required '
                                  'functions  \n'
                                  'The reaction requires a strong acid '
                                  'catalyst to protonate the carbonyl oxygen, '
                                  'facilitating nucleophilic attack by the '
                                  'alcohol. The catalyst must also be capable '
                                  'of removing water to drive the equilibrium '
                                  'toward ester formation.\n'
                                  '\n'
                                  '## Step 6: Generation of candidate '
                                  'reagents  \n'
                                  'The candidate reagents are as follows:  \n'
                                  '1. O=S(=O)(O)C(F)(F)F  \n'
                                  '2. O=S(=O)([O-])O  \n'
                                  '3. O=S(=O)(O)O  \n'
                                  '\n'
                                  '## Step 7: Selection of reagents  \n'
                                  'Reagent 3 is the most suitable reagent for '
                                  'the reaction, since it is sulfuric acid, '
                                  'which acts as a strong acid catalyst in '
                                  'esterification reactions. It provides the '
                                  'necessary protonation for carbonyl '
                                  'activation and facilitates water removal, '
                                  'both critical for driving the reaction '
                                  'forward.</think>\n'
                                  '\n'
                                  '<ANSWER>\n'
                                  'O=S(=O)(O)O\n'
                                  '</ANSWER>',
                  'style': 'rule'},
 'rxn_str': 'CO.COc1cc(C=CC(=O)O)ccc1O.O=S(=O)(O)O>>COC(=O)/C=C/c1ccc(O)c(OC)c1',
 'solvents': [],
 'task': 'reagent',
 'yields': []}
"""


def append_reagent_info(data):
    ability = "chemistry"
    class_name = data.get('class_name', 'DUMMY')
    data_source = "chem_dapo"
    extra_info = {
        'class_name': class_name,
        'products': data['products'],
        'reactants': data['reactants'],
        'reagents': data['reagents'],
        'rxn_str': data['rxn_str'],
        'supporting_info': {
            'forward': {
                'canonical_intermediate': 'DUMMY',
                'generated_substructure_from_precursor': ['DUMMY'],
                'intermediate_smiles': 'DUMMY',
                'masked_smiles': ['DUMMY'],
                'product_changes_tagged': 'DUMMY',
                'reactive_atom_bonds': [['', '', '']],
                'reactive_atoms_smiles_str': ''
            },
            'retro': {
                'atom_mapping': 'DUMMY',
                'bond_list': [['', '', '']],
                'masked_smiles': ['DUMMY'],
                'synthetic_equivalents': [''],
                'synthons_list': [''],
                'synthons_list_new': ['']
            },
            'reagent': {
                'correct_reagent_number': '-1',
                'generated_substructure_list': ['DUMMY'],
                'masked_smiles': ['DUMMY'],
                'reagents': ['CO', 'CO', 'CO'],
                'removed_substructure_list': ['DUMMY']
            }
        },
        'task': 'reagent',
        'yields': data.get('yields', []),
    }
    products = data['products']
    prompt = [
        {'role': 'system', 'content': 'You are a chemist.'},
        {'role': 'user', 'content': f"""Please suggest some possible reagents that could have been used in the following chemical reaction {'.'.join(data["reactants"])}>>{'.'.join(data["products"])}."""}
    ],
    reactants = data['reactants']
    reagents = data['reagents']
    reward_model = {
        'ground_truth': 'DUMMY',
        'style': 'rule'
    }
    rxn_str = data['rxn_str']
    solvents = data.get('solvents', [])
    task = "reagent"
    yields = data.get('yields', [])

    new_data = {
        'ability': ability,
        'class_name': class_name,
        'data_source': data_source,
        'extra_info': extra_info,
        'products': products,
        'prompt': prompt,
        'reactants': reactants,
        'reagents': reagents,
        'reward_model': reward_model,
        'rxn_str': rxn_str,
        'solvents': solvents,
        'task': task,
        'yields': yields
    }

    return new_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", default="/data/llm-reaction-reasoning/data/orderly/balanced_val")
    parser.add_argument("--test_dir", default="/data/llm-reaction-reasoning/data/orderly/excluded_test")
    parser.add_argument("--output_dir", default="/data/verl/data")
    parser.add_argument("--dir_name", default="chem_dapo")
    parser.add_argument("--n_samples", type=int, default=999999999)
    args = parser.parse_args()

    task = "reagent"

    concatenated = []
    valtest_paths = [os.path.join(f"{args.val_dir}", f"balanced_{task}_val_v10_required.jsonl"), os.path.join(f"{args.test_dir}", f"excluded_{task}_test_v10_required.jsonl")]
    for dataset_path in valtest_paths:
        d = load_dataset("json", split="train", data_files=dataset_path)
        concatenated.append(d)
    valtest_data = concatenate_datasets(concatenated)
    all_rxn_strs = set(valtest_data["rxn_str"])

    with open("/data/llm-reaction-reasoning/data/orderly/condition_test.json") as f:
        base_data = json.load(f)

    new_data = []
    for item in base_data:
        rxn_str = item["rxn_str"]
        if rxn_str in all_rxn_strs:
            continue
        if len(item['reagents']) == 0:
            continue
        item['rationale'] = "DUMMY"
        new_data.append(item)


    # Change to parquet format and save "/data/verl/data/chem_dapo/syntheticreact_auxiliary_reagent_train.parquet"
    new_dataset = Dataset.from_list(new_data)
    new_dataset = new_dataset.map(function=append_reagent_info, load_from_cache_file=False)
    new_dataset.to_parquet(os.path.join(args.output_dir, f"{args.dir_name}", "syntheticreact_auxiliary_reagent_train.parquet"))

