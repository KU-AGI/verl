# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import List, Optional, Union, Tuple, Any

import numpy as np
import torch
import pyarrow as pa
from datasets import load_dataset, load_from_disk
from omegaconf import DictConfig, ListConfig
from PIL import Image
import io as _io

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.getLogger(__name__)


class ImageRLDataset(Dataset):
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(list(data_files))
        self.original_data_files = copy.deepcopy(list(data_files))
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        self.schema_metadata = None
        self.is_nested_structure = False

        self.use_sample_cache = config.get("use_sample_cache", False)
        self.cache_size = config.get("sample_cache_size", 1000)
        self._sample_cache = {} if self.use_sample_cache else None

        self.decode_tensors = config.get("decode_tensors", True)
        self.decode_images = config.get("decode_images", True)
        self.device = config.get("device", "cpu")
        self.max_cached_chunks = config.get("max_cached_chunks", 5)

        self._chunk_cache = OrderedDict()
        self.chunk_types: List[str] = []
        self.chunk_dirs: List[Path] = []

        self._download()
        self._read_files_and_build_index()

    # ----------------------------- IO / discovery -----------------------------

    def _download(self, use_origin_parquet: bool = False):
        from verl.utils.fs import copy_to_local
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, f in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=f, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _load_schema_metadata(self, base_dir: str):
        import json
        meta_filepath = os.path.join(base_dir, "schema_metadata.json")
        if os.path.exists(meta_filepath):
            with open(meta_filepath, "r") as f:
                meta = json.load(f)
            self.schema_metadata = meta.get("structure", {})
            self.is_nested_structure = True
            print(f"[ImageRLDataset] Loaded schema metadata with {len(self.schema_metadata)} fields from {base_dir}")

    def _is_parquet_file(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in [".parquet", ".pq"]

    def _is_arrow_dataset(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        if (path / "dataset_info.json").exists():
            return True
        if list(path.glob("*.arrow")):
            return True
        if list(path.glob("data-*")):
            return True
        return False

    def _discover_arrow_dataset_under(self, container: Path, max_depth: int = 2) -> List[Path]:
        found: List[Path] = []
        if not container.is_dir():
            return found

        frontier = [(container, 0)]
        seen = set()
        while frontier:
            p, d = frontier.pop()
            if str(p) in seen:
                continue
            seen.add(str(p))

            if self._is_arrow_dataset(p):
                found.append(p)
                continue

            if d >= max_depth:
                continue

            try:
                for child in p.iterdir():
                    if child.is_dir():
                        frontier.append((child, d + 1))
            except PermissionError:
                continue

        uniq = []
        seen2 = set()
        for p in found:
            if str(p) not in seen2:
                uniq.append(p)
                seen2.add(str(p))
        return uniq

    def _find_chunks(self, data_path: Path) -> List[Tuple[Path, str, Path]]:
        chunks: List[Tuple[Path, str, Path]] = []

        if not data_path.exists():
            logger.error(f"Path does not exist: {data_path}")
            return chunks

        if self._is_parquet_file(data_path):
            return [(data_path, "parquet", data_path.parent)]

        if self._is_arrow_dataset(data_path):
            return [(data_path, "arrow", data_path)]

        if not data_path.is_dir():
            logger.error(f"Path is not a directory: {data_path}")
            return chunks

        chunk_pattern = re.compile(r"^chunk_\d+$")
        chunk_dirs: List[Path] = []
        try:
            idx = 0
            for item in data_path.iterdir():
                if idx >= 8: break
                if item.is_dir() and chunk_pattern.match(item.name):
                    chunk_dirs.append(item)
                    idx += 1
        except PermissionError:
            logger.error(f"Permission denied: {data_path}")
            return chunks

        if chunk_dirs:
            chunk_dirs = sorted(chunk_dirs, key=lambda x: int(x.name.split("_")[1]))
            for chunk_dir in chunk_dirs:
                found_in_chunk = False

                if self._is_arrow_dataset(chunk_dir):
                    chunks.append((chunk_dir, "arrow", chunk_dir))
                    found_in_chunk = True

                parquet_files = list(chunk_dir.glob("*.parquet")) + list(chunk_dir.glob("*.pq"))
                if parquet_files:
                    for pq_file in parquet_files:
                        chunks.append((pq_file, "parquet", chunk_dir))
                    found_in_chunk = True

                if not found_in_chunk:
                    arrow_files = list(chunk_dir.glob("*.arrow"))
                    if arrow_files:
                        chunks.append((chunk_dir, "arrow_files", chunk_dir))
                        found_in_chunk = True

                if not found_in_chunk:
                    nested_arrow = self._discover_arrow_dataset_under(chunk_dir, max_depth=2)
                    if nested_arrow:
                        for adir in nested_arrow:
                            chunks.append((adir, "arrow", chunk_dir))
                        found_in_chunk = True

                if not found_in_chunk:
                    logger.warning(f"Skipping {chunk_dir} (no data found)")
        else:
            parquet_files = sorted(data_path.glob("*.parquet")) + sorted(data_path.glob("*.pq"))
            for pq_file in parquet_files:
                chunks.append((pq_file, "parquet", data_path))

            for subdir in data_path.iterdir():
                if subdir.is_dir() and self._is_arrow_dataset(subdir):
                    chunks.append((subdir, "arrow", subdir))

        return chunks

    # ----------------------- indexing / length computation --------------------

    def _get_chunk_length(self, chunk_path: Path, chunk_type: str) -> int:
        if chunk_type == "parquet":
            ds = load_dataset("parquet", data_files=str(chunk_path), split="train")
            return int(len(ds))

        if chunk_type == "arrow":
            try:
                ds = load_from_disk(str(chunk_path))
                return int(len(ds))
            except Exception:
                arrow_files = list(Path(chunk_path).glob("*.arrow"))
                if not arrow_files:
                    raise
                import pyarrow.dataset as pads
                dset = pads.dataset(arrow_files, format="arrow")
                if hasattr(dset, "count_rows"):
                    return int(dset.count_rows())
                return int(dset.to_table().num_rows)

        if chunk_type == "arrow_files":
            arrow_files = list(chunk_path.glob("*.arrow"))
            import pyarrow.dataset as pads
            dset = pads.dataset(arrow_files, format="arrow")
            if hasattr(dset, "count_rows"):
                return int(dset.count_rows())
            return int(dset.to_table().num_rows)

        raise ValueError(f"Unknown chunk type: {chunk_type}")

    def _read_files_and_build_index(self):
        all_chunks: List[Tuple[Path, str, Path]] = []

        for data_path in self.data_files:
            path = Path(data_path)
            if path.is_dir() and self.schema_metadata is None:
                self._load_schema_metadata(str(path))
            all_chunks.extend(self._find_chunks(path))

        if not all_chunks:
            raise ValueError(f"No valid data files found in {self.data_files}")

        if self.schema_metadata is None:
            for p, t, _d in all_chunks:
                if t == "arrow" and p.is_dir():
                    self._load_schema_metadata(str(p))
                    if self.schema_metadata is not None:
                        break

        self.chunk_files = [c[0] for c in all_chunks]
        self.chunk_types = [c[1] for c in all_chunks]
        self.chunk_dirs = [c[2] for c in all_chunks]

        print(f"[ImageRLDataset] Found {len(self.chunk_files)} chunks total:")
        for p, t in zip(self.chunk_files, self.chunk_types):
            print(f"  - {p} ({t})")

        self.chunk_offsets = []
        total_length = 0
        failed: List[Tuple[str, str, str]] = []

        for chunk_idx, (chunk_path, chunk_type) in enumerate(zip(self.chunk_files, self.chunk_types)):
            try:
                chunk_len = self._get_chunk_length(chunk_path, chunk_type)
                if chunk_len <= 0:
                    failed.append((str(chunk_path), chunk_type, "len==0"))
                    continue
                self.chunk_offsets.append((chunk_idx, total_length, chunk_len))
                total_length += chunk_len
            except Exception as e:
                failed.append((str(chunk_path), chunk_type, repr(e)))
                continue

        self.total_length = int(total_length)
        print(f"[ImageRLDataset] Total samples: {self.total_length}")

        if self.total_length == 0:
            preview = "\n".join([f"  - {p} ({t}) -> {err}" for p, t, err in failed[:80]])
            raise ValueError(
                "All chunks failed to load or had zero length; dataset would be empty.\n"
                "First failures:\n" + preview
            )

        if self.max_samples > 0 and self.max_samples < self.total_length:
            self.total_length = int(self.max_samples)
            print(f"[ImageRLDataset] Limited to {self.max_samples} samples")

    # ------------------------------ chunk cache ------------------------------

    def _get_chunk_for_index(self, idx: int) -> tuple:
        for chunk_idx, start_idx, chunk_len in self.chunk_offsets:
            if start_idx <= idx < start_idx + chunk_len:
                return chunk_idx, idx - start_idx
        raise IndexError(f"Index {idx} out of range")

    # ============= 최적화된 컬럼 단위 로드 (핵심 변경) =============
    @staticmethod
    def _as_array(col):
        """ChunkedArray를 Array로 변환"""
        if isinstance(col, pa.ChunkedArray):
            combined = col.combine_chunks()
            if isinstance(combined, pa.ChunkedArray):
                return combined.chunk(0)
            return combined
        return col

    @staticmethod
    def _flatten_fully(table: pa.Table, max_iter: int = 16) -> pa.Table:
        """struct 타입을 완전히 flatten"""
        t = table
        for _ in range(max_iter):
            if any(pa.types.is_struct(f.type) for f in t.schema):
                t = t.flatten()
            else:
                break
        return t

    def _extract_group_columns_fast(
        self,
        table: pa.Table,
        group: str,
    ) -> dict[str, list]:
        """
        group struct를 컬럼 단위로 한 번에 로드 (row dict 접근 없음)
        Returns: dict[field_name] = list(len=chunk_size)
        """
        if group not in table.column_names:
            return {}

        gt = self._flatten_fully(table.select([group]))
        out = {}

        prefix = group + "."
        for colname in gt.column_names:
            if not colname.startswith(prefix):
                continue
            
            field = colname[len(prefix):]

            if field in {"attention_mask", "task_id"}:
                continue

            arr = self._as_array(gt[colname])
            # ✨ 핵심: to_pylist()를 컬럼당 1번만 실행
            out[field] = arr.to_pylist()

        return out

    def _load_chunk(self, chunk_idx: int) -> tuple:
        """청크를 로드하고 컬럼 단위로 전처리"""
        if chunk_idx in self._chunk_cache:
            self._chunk_cache.move_to_end(chunk_idx)
            return self._chunk_cache[chunk_idx]

        chunk_path = self.chunk_files[chunk_idx]
        chunk_type = self.chunk_types[chunk_idx]
        chunk_dir = self.chunk_dirs[chunk_idx]

        # 1. 데이터셋 로드
        if chunk_type == "arrow":
            try:
                ds = load_from_disk(str(chunk_path))
            except Exception as e:
                logger.warning(f"load_from_disk failed for {chunk_path}, trying arrow-shard fallback: {e}")
                arrow_files = list(Path(chunk_path).glob("*.arrow"))
                if not arrow_files:
                    raise
                import pyarrow.dataset as pads
                dset = pads.dataset(arrow_files, format="arrow")
                from datasets import Dataset as HFDataset
                table = dset.to_table()
                ds = HFDataset(table)

        elif chunk_type == "arrow_files":
            arrow_files = list(Path(chunk_path).glob("*.arrow"))
            if not arrow_files:
                raise ValueError(f"No arrow files found in {chunk_path}")
            import pyarrow.dataset as pads
            dset = pads.dataset(arrow_files, format="arrow")
            from datasets import Dataset as HFDataset
            table = dset.to_table()
            ds = HFDataset(table)

        elif chunk_type == "parquet":
            ds = load_dataset("parquet", data_files=str(chunk_path), split="train")
        else:
            raise ValueError(f"Unknown chunk type: {chunk_type}")

        ds.reset_format()
        table = ds.data

        # 2. 컬럼 단위로 전체 데이터 추출 (한 번에!)
        chunk_data = {
            "table": table,
            "chunk_type": chunk_type,
            "chunk_dir": chunk_dir,
            "top_level": {},
            "batch": {},
            "non_tensor_batch": {},
            "meta_info": {},
            "is_flat": False  # ✨ flat 구조 여부 추적
        }

        # top-level 컬럼들
        for k in ["global_step", "param_version", "local_trigger_step", "images_dir"]:
            if k in table.column_names:
                arr = self._as_array(table[k])
                chunk_data["top_level"][k] = arr.to_pylist()

        # ✨ nested structure 시도 (arrow와 parquet 공통)
        batch_cols = self._extract_group_columns_fast(table, "batch")
        non_tensor_cols = self._extract_group_columns_fast(table, "non_tensor_batch")
        meta_cols = self._extract_group_columns_fast(table, "meta_info")
        
        if batch_cols or non_tensor_cols or meta_cols:
            # nested structure 존재
            chunk_data["batch"] = batch_cols
            chunk_data["non_tensor_batch"] = non_tensor_cols
            chunk_data["meta_info"] = meta_cols
            chunk_data["is_flat"] = False
        else:
            # flat structure - 모든 컬럼을 직접 로드
            chunk_data["is_flat"] = True
            processed_cols = set(chunk_data["top_level"].keys())
            
            for col_name in table.column_names:
                if col_name in processed_cols:
                    continue
                
                arr = self._as_array(table[col_name])
                values = arr.to_pylist()
                
                # ✨ 컬럼 이름으로 분류 (개선된 휴리스틱)
                # 1. prompt와 메타데이터
                if col_name in [self.prompt_key, "uid", "index"]:
                    chunk_data["non_tensor_batch"][col_name] = values
                # 2. 딕셔너리 타입 (extra_info, tools_kwargs 등)
                elif col_name in ["extra_info", "tools_kwargs", "interaction_kwargs"]:
                    chunk_data["non_tensor_batch"][col_name] = values
                # 3. 이미지 관련
                elif col_name.endswith(("_pil", "_pil_list", "_image", "_images")):
                    chunk_data["non_tensor_batch"][col_name] = values
                elif "image" in col_name.lower() or "pil" in col_name.lower():
                    chunk_data["non_tensor_batch"][col_name] = values
                # 4. tensor 관련
                elif col_name.endswith(("_ids", "_mask", "_tokens", "_pixel_values", "_scores")):
                    chunk_data["batch"][col_name] = values
                # 5. 메타 정보 (step, version 등)
                elif col_name.endswith(("_step", "_version", "_dir")):
                    chunk_data["meta_info"][col_name] = values
                # 6. 기타는 non_tensor_batch에
                else:
                    chunk_data["non_tensor_batch"][col_name] = values
            
            logger.info(
                f"[Flat parquet] {chunk_path}: "
                f"batch={len(chunk_data['batch'])}, "
                f"non_tensor={len(chunk_data['non_tensor_batch'])}, "
                f"meta={len(chunk_data['meta_info'])}"
            )
        
        # 3. 캐시에 저장
        result = chunk_data
        self._chunk_cache[chunk_idx] = result
        if len(self._chunk_cache) > self.max_cached_chunks:
            self._chunk_cache.popitem(last=False)

        return result

    # ----------------------------- decode helpers ----------------------------

    def _convert_to_pil_if_needed(self, value):
        if value is None:
            return None
        if isinstance(value, (list, np.ndarray)):
            arr = np.array(value)
            if arr.shape == (384, 384, 3):
                return Image.fromarray(arr.astype(np.uint8))
        return value

    def _convert_list_to_tensor(self, value):
        if value is None:
            return None
        if isinstance(value, list):
            try:
                arr = np.array(value)
                t = torch.from_numpy(arr)
                return t.to(self.device) if self.device != "cpu" else t
            except Exception as e:
                logger.warning(f"Failed to convert list to tensor: {e}")
                return value
        return value

    # ------------------------------ torch Dataset ----------------------------

    def __len__(self):
        return int(self.total_length)

    def _resolve_image_path(self, value, chunk_dir: Path):
        """이미지 경로를 절대 경로로 변환"""
        if value is None or value == "N/A":
            return value
        
        # 문자열인 경우 (상대 경로)
        if isinstance(value, str):
            # 이미 절대 경로면 그대로 반환
            if os.path.isabs(value):
                return value
            # 상대 경로면 chunk_dir/images/와 결합
            return str(chunk_dir / "images" / value)
        
        # 리스트인 경우 재귀적으로 처리
        elif isinstance(value, list):
            return [self._resolve_image_path(item, chunk_dir) for item in value]
        
        return value
    
    def _resolve_image_paths_in_dict(self, d: dict, chunk_dir: Path, prefix: str = "") -> dict:
        """딕셔너리 내의 모든 이미지 경로를 재귀적으로 변환"""
        if not isinstance(d, dict):
            return d
        
        result = {}
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            # 이미지 필드 판별 (스키마 또는 키 이름 기반)
            is_image_field = False
            if self.schema_metadata:
                meta = self.schema_metadata.get(full_key, {})
                v_type = meta.get("value_type", "")
                is_image_field = v_type in ["PIL.Image", "List[PIL.Image]"]
            
            # 키 이름으로도 판별
            if not is_image_field:
                is_image_field = ("image" in key.lower() or 
                                 "pil" in key.lower() or
                                 key in ["gen", "regen", "gt", "ground_truth"])
            
            # 이미지 경로 변환
            if is_image_field and isinstance(value, str):
                result[key] = self._resolve_image_path(value, chunk_dir)
            # 중첩된 딕셔너리 처리
            elif isinstance(value, dict):
                result[key] = self._resolve_image_paths_in_dict(value, chunk_dir, full_key)
            # 리스트 처리 (이미지 리스트일 수 있음)
            elif isinstance(value, list) and is_image_field:
                result[key] = self._resolve_image_path(value, chunk_dir)
            else:
                result[key] = value
        
        return result

    def __getitem__(self, item: int):
        """최적화된 샘플 추출: 캐시된 컬럼 리스트에서 인덱싱만"""
        if self._sample_cache is not None and item in self._sample_cache:
            return copy.deepcopy(self._sample_cache[item])

        if item < 0 or item >= self.total_length:
            raise IndexError(f"Index {item} out of range [0, {self.total_length})")

        chunk_idx, local_idx = self._get_chunk_for_index(item)
        chunk_data = self._load_chunk(chunk_idx)

        chunk_type = chunk_data["chunk_type"]
        chunk_dir = chunk_data["chunk_dir"]
        is_flat = chunk_data.get("is_flat", False)
        row_dict: dict[str, Any] = {}

        # top-level fields
        for k, values in chunk_data["top_level"].items():
            row_dict[k] = values[local_idx]

        # batch 그룹 (tensor로 변환)
        batch_dict = {}
        batch_data = chunk_data.get("batch", {})
        if not isinstance(batch_data, dict):
            logger.error(f"batch is not dict: type={type(batch_data)}, item={item}")
            batch_data = {}
        
        for field, values in batch_data.items():
            try:
                value = values[local_idx]
                
                # tensor 변환
                if field.endswith(("_ids", "_mask", "_tokens", "_pixel_values", "_scores")):
                    value = self._convert_list_to_tensor(value)
                
                batch_dict[field] = value
            except Exception as e:
                logger.warning(f"Failed to process batch field {field} at item {item}: {e}")
                continue

        # non_tensor_batch 그룹
        non_tensor_dict = {}
        non_tensor_data = chunk_data.get("non_tensor_batch", {})
        if not isinstance(non_tensor_data, dict):
            logger.error(f"non_tensor_batch is not dict: type={type(non_tensor_data)}, item={item}")
            non_tensor_data = {}
        
        for field, values in non_tensor_data.items():
            try:
                value = values[local_idx]
                
                # ✨ 이미지 경로 변환 (PIL 변환 전에)
                if isinstance(value, str) and ("image" in field.lower() or "pil" in field.lower()):
                    value = self._resolve_image_path(value, chunk_dir)
                
                # PIL 변환
                if field.endswith("_pil_list") or field.endswith("_pil") or "image" in field.lower():
                    if isinstance(value, str) and os.path.exists(value):
                        try:
                            value = Image.open(value).convert("RGB")
                        except Exception as e:
                            logger.warning(f"Failed to load image from {value}: {e}")
                    else:
                        value = self._convert_to_pil_if_needed(value)
                
                non_tensor_dict[field] = value
            except Exception as e:
                logger.warning(f"Failed to process non_tensor field {field} at item {item}: {e}")
                continue

        # meta_info 그룹
        meta_dict = {}
        meta_data = chunk_data.get("meta_info", {})
        if not isinstance(meta_data, dict):
            logger.error(f"meta_info is not dict: type={type(meta_data)}, item={item}")
            meta_data = {}
        
        for field, values in meta_data.items():
            try:
                value = values[local_idx]
                meta_dict[field] = value
            except Exception as e:
                logger.warning(f"Failed to process meta field {field} at item {item}: {e}")
                continue

        # ✨ 이미지 경로 변환 적용
        try:
            batch_dict = self._resolve_image_paths_in_dict(batch_dict, chunk_dir, "batch")
            non_tensor_dict = self._resolve_image_paths_in_dict(non_tensor_dict, chunk_dir, "non_tensor_batch")
        except Exception as e:
            logger.warning(f"Failed to resolve image paths at item {item}: {e}")
        
        # ✨ row_dict 구성 - 빈 딕셔너리도 명시적으로 생성
        row_dict["batch"] = batch_dict if batch_dict else {}
        row_dict["non_tensor_batch"] = non_tensor_dict if non_tensor_dict else {}
        row_dict["meta_info"] = meta_dict if meta_dict else {}
        
        # top-level로도 노출 (batch만)
        row_dict.update(batch_dict)
        # meta_info도 top-level로 노출
        row_dict.update(meta_dict)

        # Promote prompt
        nt = row_dict["non_tensor_batch"]
        prompt = row_dict.get(self.prompt_key, None)
        if prompt is None and isinstance(nt, dict):
            prompt = nt.get(self.prompt_key, None)
        row_dict["prompt"] = prompt

        # Promote extra_info
        if ("extra_info" not in row_dict or row_dict["extra_info"] is None) and isinstance(nt, dict):
            row_dict["extra_info"] = nt.get("extra_info", {}) or {}
        elif "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}

        # Compute index/tools/interaction
        extra_info = row_dict.get("extra_info", {}) or {}
        index = extra_info.get("index", item) if isinstance(extra_info, dict) else item

        row_dict["index"] = index
        # Ensure uid
        if isinstance(nt, dict):
            if "uid" not in nt or nt["uid"] is None:
                nt["uid"] = f"uid_{index}"

        # Always provide a tensor
        if "dummy_tensor" not in row_dict:
            row_dict["dummy_tensor"] = torch.zeros(1)

        # Cache
        if self._sample_cache is not None and len(self._sample_cache) < self.cache_size:
            self._sample_cache[item] = copy.deepcopy(row_dict)

        return row_dict