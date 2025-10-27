# import torch
# import json
# from transformers import AutoModelForCausalLM, AutoConfig
# from janus.models import MultiModalityCausalLM, VLChatProcessor

# def _preview_missing_unexpected(missing, unexpected, limit=30):
#     print(f"[load_state_dict] missing: {len(missing)} keys, unexpected: {len(unexpected)} keys")
#     if missing:
#         print("  missing (preview):", missing[:limit])
#     if unexpected:
#         print("  unexpected (preview):", unexpected[:limit])

# def load_sharded_state_dict(ckpt_dir):
#     """
#     Sharded checkpoint에서 state_dict 로드
#     """
#     index_file = f"{ckpt_dir}/pytorch_model.bin.index.json"
    
#     print(f"Loading sharded checkpoint from: {ckpt_dir}")
    
#     # index.json 파일 읽기
#     with open(index_file, 'r') as f:
#         index = json.load(f)
    
#     print(f"Found {len(index['weight_map'])} weights across {len(set(index['weight_map'].values()))} shards")
    
#     # 모든 shard 파일들 로드해서 합치기
#     state_dict = {}
#     shard_files = set(index['weight_map'].values())
    
#     for shard_file in shard_files:
#         shard_path = f"{ckpt_dir}/{shard_file}"
#         print(f"Loading shard: {shard_file}")
#         shard = torch.load(shard_path, map_location="cpu")
#         if shard.keys().startswith("model."):
#            shard = {k.replace("model.", ""): v for k, v in shard.items()}
#         state_dict.update(shard)
    
#     print(f"Loaded {len(state_dict)} weights total")
#     return state_dict

# def load_finetuned_model_sharded(base_model_path, finetuned_ckpt_dir):
#     """
#     Sharded fine-tuned checkpoint로부터 모델 로드
#     """
#     # 1. Base processor 로드
#     print("Loading base processor...")
#     vl_chat_processor = VLChatProcessor.from_pretrained(base_model_path)
    
#     # 2. Base model 로드
#     print("Loading base model...")
#     vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
#         base_model_path,
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16,
#         device_map=None
#     )
    
#     # 3. Sharded fine-tuned weights 로드
#     print("Loading fine-tuned weights...")
#     state_dict = load_sharded_state_dict(finetuned_ckpt_dir)
    
#     # 4. State dict 로드
#     print("Applying fine-tuned weights to model...")
#     missing, unexpected = vl_gpt.load_state_dict(state_dict, strict=False)
#     _preview_missing_unexpected(missing, unexpected, limit=30)
    
#     return vl_gpt, vl_chat_processor

# def save_model(model, processor, save_path):
#     """
#     모델과 processor 저장
#     """
#     print(f"Saving model and processor to: {save_path}")
#     model.save_pretrained(save_path)
#     processor.save_pretrained(save_path)
#     print("✅ Model and processor saved successfully!")

# # 사용 예시
# if __name__ == "__main__":
#     base_model_path = "/data/mllm/checkpoints/Janus-Pro-7B"
#     finetuned_ckpt_dir = "/data/mllm/ckpt"  # pytorch_model.bin.index.json이 있는 디렉토리
#     save_path = "/data/mllm/ckpt/pretrained"
    
#     # 모델 로드
#     model, processor = load_finetuned_model_sharded(base_model_path, finetuned_ckpt_dir)
    
#     # 저장
#     save_model(model, processor, save_path)

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor
from collections import OrderedDict
import os
import json
from typing import Dict
from collections import defaultdict

def load_sharded_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load Sharded .bin (Optimized for speed and memory)
    """
    index_path = os.path.join(checkpoint_path, "pytorch_model.bin.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found at {index_path}")

    with open(index_path, "r") as f:
        index = json.load(f)
    
    # 1. weight_map을 뒤집어서 shard_file -> [tensor_names] 구조로 변경
    # 이렇게 하면 각 샤드를 한 번만 로드할 수 있습니다.
    shards_to_tensors = defaultdict(list)
    for tensor_name, shard_file in index["weight_map"].items():
        shards_to_tensors[shard_file].append(tensor_name)

    full_state_dict = OrderedDict()

    # 2. 샤드 파일별로 순회
    for shard_file, tensor_names in shards_to_tensors.items():
        shard_path = os.path.join(checkpoint_path, shard_file)
        
        # 3. 단일 샤드 파일만 메모리에 로드
        loaded_shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        
        # 4. 해당 샤드에 포함된 텐서들만 full_state_dict에 추가
        for tensor_name in tensor_names:
            # 'model.' 접두사 제거
            new_key = tensor_name.replace("model.", "", 1)
            full_state_dict[new_key] = loaded_shard[tensor_name]
        
        # 5. 작업이 끝난 샤드는 메모리에서 즉시 해제 (가비지 컬렉션 유도)
        del loaded_shard

    return full_state_dict


if __name__ == "__main__":
    base_model_path = "/data/mllm/checkpoints/Janus-Pro-7B"
    finetuned_ckpt_dir = "/data/mllm/ckpt"  # pytorch_model.bin.index.json이 있는 디렉토리
    save_path = "/data/mllm/ckpt/pretrained"

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
    full_state_dict = load_sharded_checkpoint(finetuned_ckpt_dir)

    incompatible_keys = model.load_state_dict(full_state_dict, strict=False)
    
    print("✅ Finetuned weights loaded successfully.")

    if incompatible_keys.missing_keys:
        print("\n--- Missing Keys (were not in the checkpoint, kept original weights) ---")
        print(f"Found {len(incompatible_keys.missing_keys)} missing keys.")
    if incompatible_keys.unexpected_keys:
        print("\n--- Unexpected Keys (were in the checkpoint but not in the model) ---")
        print(f"Found {len(incompatible_keys.unexpected_keys)} unexpected keys.")
    
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)