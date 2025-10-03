import os, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

def main(args):
    print(f"Loading base model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
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