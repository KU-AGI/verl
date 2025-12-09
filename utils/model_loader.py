import torch
from transformers import AutoModelForCausalLM, AutoConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor
from collections import OrderedDict
import os
import json
from typing import Dict
from collections import defaultdict
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint, get_fp32_state_dict_from_zero_checkpoint

def load_deepspeed_ckpt(model, ckpt_path, dtype='bf16'):
    sft_state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)

    corrected_state_dict = {}
    keys_were_corrected = False

    prefix_to_remove = 'model.'

    print(f"Attempting to remove prefix: '{prefix_to_remove}'")

    for k, v in sft_state_dict.items():
        if k.startswith(prefix_to_remove):
            new_key = k[len(prefix_to_remove):] 
            corrected_state_dict[new_key] = v
            keys_were_corrected = True
        else:
            corrected_state_dict[k] = v 
            print(f"WARNING: Found a key without expected prefix: {k}")

    if keys_were_corrected:
        print(f"Successfully corrected '{prefix_to_remove}' prefix from SFT keys.")
    else:
        print("ERROR: No keys were corrected. The prefix 'model.' was not found.")

    print("Loading corrected state_dict into model (strict=False)...")
    load_result = model.load_state_dict(corrected_state_dict, strict=False)
    
    # 4. 로드 결과 확인
    print("--- Load Result (after correction) ---")
    print(f"Missing Keys : {load_result.missing_keys}")
    print(f"Unexpected Keys : {load_result.unexpected_keys}")

    model_dtype = torch.bfloat16 if dtype == 'bf16' else torch.float32
    model = model.to(model_dtype)

    return model


if __name__ == "__main__":
    base_model_path = "/data/mllm/checkpoints/Janus-Pro-7B"
    finetuned_ckpt_dir = "/data/mllm/ckpt/step=012000.ckpt" # latest, zero_to_fp32.py
    save_path = "/data/mllm/ckpt/step=012000.ckpt/hf_model"

    # 1. Base processor 로드
    print("Loading base processor...")
    processor = VLChatProcessor.from_pretrained(base_model_path)
    
    # 2. Base model 로드
    print("Loading base model...")
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    
    # 모델 로드
    model = load_deepspeed_ckpt(model, finetuned_ckpt_dir)

    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)