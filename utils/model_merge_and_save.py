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

    # Forward
    molecule_info_start = "<MOLECULE_INFO>" # 151669
    molecule_info_end = "</MOLECULE_INFO>" # 151670
    main_funcgroup_info_start = "<MAIN_FUNCGROUP_INFO>" # 151671
    main_funcgroup_info_end = "</MAIN_FUNCGROUP_INFO>" # 151672
    template_start = "<TEMPLATE>" # 151673
    template_end = "</TEMPLATE>" # 151674
    reaction_start = "<REACTION>" # 151675
    reaction_end = "</REACTION>" # 151676
    # Retro
    product_info_start = "<PRODUCT_INFO>" # 151677
    product_info_end = "</PRODUCT_INFO>" # 151678
    candidate_structure_start = "<CANDIDATE_STRUCTURE>" # 151679
    candidate_structure_end = "</CANDIDATE_STRUCTURE>" # 151680
    strategic_bond_disconnect_start = "<STRATEGIC_BOND_DISCONNECTION>" # 151681
    strategic_bond_disconnect_end = "</STRATEGIC_BOND_DISCONNECTION>" # 151682
    synthetic_equivalent_start = "<SYNTHETIC_EQUIVALENT>" # 151683
    synthetic_equivalent_end = "</SYNTHETIC_EQUIVALENT>" # 151684

    special_tokens = [
        molecule_info_start, molecule_info_end,
        main_funcgroup_info_start, main_funcgroup_info_end,
        template_start, template_end,
        reaction_start, reaction_end,
        product_info_start, product_info_end,
        candidate_structure_start, candidate_structure_end,
        strategic_bond_disconnect_start, strategic_bond_disconnect_end,
        synthetic_equivalent_start, synthetic_equivalent_end,
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