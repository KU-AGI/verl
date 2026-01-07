from collections import defaultdict
import numpy as np
import torch
from typing import Any

def _to_object_array(vs: list[Any]) -> np.ndarray:
    return np.fromiter(vs, dtype=object, count=len(vs))

def collate_fn(data_list: list[dict]) -> dict:
    data_list = [d for d in data_list if d is not None]
    if len(data_list) == 0:
        raise RuntimeError("All samples in this batch were None (invalid/filtered).")
    
    grouped_batch = defaultdict(list)
    grouped_non = defaultdict(list)
    grouped_meta = defaultdict(list)
    other = defaultdict(list)
    
    # ✨ 첫 번째 샘플의 meta_info 키들을 미리 수집
    meta_info_keys = set()
    if data_list and isinstance(data_list[0].get("meta_info"), dict):
        meta_info_keys = set(data_list[0]["meta_info"].keys())
    
    for sample in data_list:
        b = sample.get("batch")
        if isinstance(b, dict):
            for k, v in b.items():
                grouped_batch[k].append(v)
        
        nt = sample.get("non_tensor_batch")
        if isinstance(nt, dict):
            for k, v in nt.items():
                grouped_non[k].append(v)
        
        mi = sample.get("meta_info")
        if isinstance(mi, dict):
            for k, v in mi.items():
                grouped_meta[k].append(v)
        
        for k, v in sample.items():
            if k in ("batch", "non_tensor_batch", "meta_info"):
                continue
            # ✨ meta_info에서 온 키는 other에 넣지 않음
            if k in meta_info_keys:
                continue
            other[k].append(v)
    
    out = {}
    
    # A) batch: tensor-only
    for k, vs in grouped_batch.items():
        if len(vs) == 0:
            continue
        if not all(isinstance(v, torch.Tensor) for v in vs):
            raise TypeError(
                f"batch[{k}] contains non-tensor values. Example type={type(vs[0])}. "
                "Fix __getitem__ conversion so batch fields are torch.Tensor."
            )
        out[k] = torch.stack(vs, dim=0)
    
    # B) non_tensor_batch: object arrays
    for k, vs in grouped_non.items():
        out[k] = _to_object_array(vs)
    
    # C) ✨ meta_info: 리스트로 유지
    meta_info = {}
    for k, vs in grouped_meta.items():
        if all(v == vs[0] for v in vs):
            meta_info[k] = vs[0]
        else:
            meta_info[k] = vs  # 리스트 유지
    
    out.update(meta_info)
    
    # D) other: numpy object array로 변환
    for k, vs in other.items():
        if len(vs) == 0:
            continue
        
        if all(isinstance(v, torch.Tensor) for v in vs):
            out[k] = torch.stack(vs, dim=0)
        elif all(v == vs[0] for v in vs):
            out[k] = vs[0]
        else:
            out[k] = _to_object_array(vs)

    return out