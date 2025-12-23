import os, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AddedToken
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

def main(args):
    print(f"Loading base model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

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
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA adapter from {args.adapter_path}")
    
    # Load PEFT model with LoRA adapter
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    print("Merging LoRA weights with base model...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Model merging and saving completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="./ReactionReasoner_stage123_lora_merged")
    parser.add_argument("--adapter_path", type=str, default="./ReactionReasoner_lora_adapter")
    parser.add_argument("--output_dir", type=str, default="./ReactionReasoner_stage123_lora_adapter_merged")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)