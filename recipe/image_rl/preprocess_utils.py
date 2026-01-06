import json
import numpy as np
import torch
from PIL import Image
import io

def restore_field_type_static(key, value, schema_metadata, nested_prefix=""):
    if schema_metadata is None:
        return value
    full_key = f"{nested_prefix}.{key}" if nested_prefix else key
    if full_key not in schema_metadata:
        return value

    meta = schema_metadata[full_key]
    value_type = meta.get("value_type")
    dtype = meta.get("dtype")
    shape = meta.get("shape")
    is_tensor = meta.get("is_tensor", False)

    if value is None:
        return None

    if value_type == "PIL.Image":
        if isinstance(value, dict) and "bytes" in value:
            return Image.open(io.BytesIO(value["bytes"]))
        if hasattr(value, "convert"):
            return value

    if value_type == "List[PIL.Image]" and isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict) and "bytes" in item:
                out.append(Image.open(io.BytesIO(item["bytes"])))
            else:
                out.append(item)
        return out

    if is_tensor and isinstance(value, list):
        t = torch.tensor(value)
        if dtype:
            if "int" in dtype: t = t.long()
            elif "float" in dtype: t = t.float()
            elif "bool" in dtype: t = t.bool()
        if shape:
            try: t = t.reshape(shape)
            except: pass
        return t

    if value_type == "ndarray" and isinstance(value, list):
        arr = np.array(value)
        if dtype and dtype.startswith("numpy."):
            try: arr = arr.astype(dtype.replace("numpy.", ""))
            except: pass
        if shape:
            try: arr = arr.reshape(shape)
            except: pass
        return arr

    return value


def restore_nested_dict_static(nested_dict, prefix, schema_metadata):
    if not isinstance(nested_dict, dict):
        return nested_dict
    restored = {}
    for k, v in nested_dict.items():
        full_key = f"{prefix}.{k}"
        if schema_metadata and full_key in schema_metadata:
            meta = schema_metadata[full_key]
            if meta.get("is_batch_level") and isinstance(v, str):
                try: v = json.loads(v)
                except: pass
        restored[k] = restore_field_type_static(k, v, schema_metadata, nested_prefix=prefix)
    return restored


def preprocess_row_wrapper(row_dict, schema_metadata, is_nested_structure):
    if is_nested_structure:
        if isinstance(row_dict.get("batch"), dict):
            row_dict["batch"] = restore_nested_dict_static(row_dict["batch"], "batch", schema_metadata)
        if isinstance(row_dict.get("non_tensor_batch"), dict):
            row_dict["non_tensor_batch"] = restore_nested_dict_static(row_dict["non_tensor_batch"], "non_tensor_batch", schema_metadata)
        if isinstance(row_dict.get("meta_info"), dict):
            row_dict["meta_info"] = restore_nested_dict_static(row_dict["meta_info"], "meta_info", schema_metadata)

        out = {}
        if row_dict.get("batch"): out.update(row_dict["batch"])
        if row_dict.get("non_tensor_batch"): out.update(row_dict["non_tensor_batch"])
        if row_dict.get("meta_info"): out.update(row_dict["meta_info"])
        for k in ["global_step", "param_version", "local_trigger_step"]:
            if k in row_dict: out[k] = row_dict[k]
        return out
    else:
        for k, v in list(row_dict.items()):
            row_dict[k] = restore_field_type_static(k, v, schema_metadata)
        return row_dict


def _make_cache_key(paths: list[str]) -> str:
    import hashlib, os
    parts = []
    for p in sorted(paths):
        try:
            st = os.stat(p)
            parts.append(f"{p}:{st.st_size}:{int(st.st_mtime)}")
        except:
            parts.append(p)
    s = "|".join(parts).encode("utf-8")
    return hashlib.md5(s).hexdigest()