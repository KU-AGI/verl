import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AddedToken

def load_model(base_model_path, model_path):
    if model_path.endswith(".pt"):
        _model = torch.load(model_path, map_location="cpu", weights_only=False)
        
        new_state_dict = {}
        for key in _model["module"].keys():
            if key.startswith("model."):
                new_state_dict[key[len("model."):]] = _model["module"][key]

        model = AutoModelForCausalLM.from_pretrained(base_model_path)
        model.load_state_dict(new_state_dict)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        reactant_funcgroup_start = "<REACTANT_FUNCGROUP>" # 151669
        reactant_funcgroup_end = "</REACTANT_FUNCGROUP>" # 151670
        product_funcgroup_start = "<PRODUCT_FUNCGROUP>" # 151671
        product_funcgroup_end = "</PRODUCT_FUNCGROUP>" # 151672
        molecular_role_start = "<MOLECULAR_ROLE>" # 151673
        molecular_role_end = "</MOLECULAR_ROLE>" # 151674
        condition_start = "<CONDITION>" # 151675
        condition_end = "</CONDITION>" # 151676
        precursor_stat_start = "<PRECURSOR_STAT>" # 151677
        precursor_stat_end = "</PRECURSOR_STAT>" # 151678
        reactant_stat_start = "<REACTANT_STAT>" # 151679
        reactant_stat_end = "</REACTANT_STAT>" # 151680
        product_stat_start = "<PRODUCT_STAT>" # 151681
        product_stat_end = "</PRODUCT_STAT>" # 151682
        template_start = "<TEMPLATE>" # 151683
        template_end = "</TEMPLATE>" # 151684
        bond_disconnect_start = "<BOND_DISCONNECT>" # 151685
        bond_disconnect_end = "</BOND_DISCONNECT>" # 151686
        synthon_start = "<SYNTHON>" # 151687
        synthon_end = "</SYNTHON>" # 151688
        synthetic_equivalent_start = "<SYNTHETIC_EQUIVALENT>" # 151689
        synthetic_equivalent_end = "</SYNTHETIC_EQUIVALENT>" # 151690
        reactant_removed_start = "<REACTANT_REMOVED_FUNCGROUP>" # 151691
        reactant_removed_end = "</REACTANT_REMOVED_FUNCGROUP>"# 151692
        product_added_start = "<PRODUCT_ADDED_FUNCGROUP>" # 151693
        product_added_end = "</PRODUCT_ADDED_FUNCGROUP>" # 151694
        special_tokens = [
            reactant_funcgroup_start, reactant_funcgroup_end,
            product_funcgroup_start, product_funcgroup_end,
            molecular_role_start, molecular_role_end,
            condition_start, condition_end,
            precursor_stat_start, precursor_stat_end,
            reactant_stat_start, reactant_stat_end,
            product_stat_start, product_stat_end,
            template_start, template_end,
            bond_disconnect_start, bond_disconnect_end,
            synthon_start, synthon_end,
            synthetic_equivalent_start, synthetic_equivalent_end,
            reactant_removed_start, reactant_removed_end,
            product_added_start, product_added_end,
        ]
        added = tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    AddedToken(t, special=True) for t in special_tokens
                ]
            }
        )
        return model, tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        reactant_funcgroup_start = "<REACTANT_FUNCGROUP>" # 151669
        reactant_funcgroup_end = "</REACTANT_FUNCGROUP>" # 151670
        product_funcgroup_start = "<PRODUCT_FUNCGROUP>" # 151671
        product_funcgroup_end = "</PRODUCT_FUNCGROUP>" # 151672
        molecular_role_start = "<MOLECULAR_ROLE>" # 151673
        molecular_role_end = "</MOLECULAR_ROLE>" # 151674
        condition_start = "<CONDITION>" # 151675
        condition_end = "</CONDITION>" # 151676
        precursor_stat_start = "<PRECURSOR_STAT>" # 151677
        precursor_stat_end = "</PRECURSOR_STAT>" # 151678
        reactant_stat_start = "<REACTANT_STAT>" # 151679
        reactant_stat_end = "</REACTANT_STAT>" # 151680
        product_stat_start = "<PRODUCT_STAT>" # 151681
        product_stat_end = "</PRODUCT_STAT>" # 151682
        template_start = "<TEMPLATE>" # 151683
        template_end = "</TEMPLATE>" # 151684
        bond_disconnect_start = "<BOND_DISCONNECT>" # 151685
        bond_disconnect_end = "</BOND_DISCONNECT>" # 151686
        synthon_start = "<SYNTHON>" # 151687
        synthon_end = "</SYNTHON>" # 151688
        synthetic_equivalent_start = "<SYNTHETIC_EQUIVALENT>" # 151689
        synthetic_equivalent_end = "</SYNTHETIC_EQUIVALENT>" # 151690
        reactant_removed_start = "<REACTANT_REMOVED_FUNCGROUP>" # 151691
        reactant_removed_end = "</REACTANT_REMOVED_FUNCGROUP>"# 151692
        product_added_start = "<PRODUCT_ADDED_FUNCGROUP>" # 151693
        product_added_end = "</PRODUCT_ADDED_FUNCGROUP>" # 151694
        special_tokens = [
            reactant_funcgroup_start, reactant_funcgroup_end,
            product_funcgroup_start, product_funcgroup_end,
            molecular_role_start, molecular_role_end,
            condition_start, condition_end,
            precursor_stat_start, precursor_stat_end,
            reactant_stat_start, reactant_stat_end,
            product_stat_start, product_stat_end,
            template_start, template_end,
            bond_disconnect_start, bond_disconnect_end,
            synthon_start, synthon_end,
            synthetic_equivalent_start, synthetic_equivalent_end,
            reactant_removed_start, reactant_removed_end,
            product_added_start, product_added_end,
        ]
        added = tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    AddedToken(t, special=True) for t in special_tokens
                ]
            }
        )

        return model, tokenizer 

if __name__ == "__main__":
    model, tokenizer = load_model("Qwen/Qwen3-8B", "/data/verl/ckpts/reflection_v4_fullft_all/best.ckpt/checkpoint/mp_rank_00_model_states.pt")
    model.save_pretrained("/data/verl/ckpts/reflection_v4_fullft_all")
    tokenizer.save_pretrained("/data/verl/ckpts/reflection_v4_fullft_all")
    print(model)
    print(tokenizer)