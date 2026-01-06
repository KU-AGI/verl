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
from collections import defaultdict
from typing import List, Union, Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from datasets import load_dataset, load_from_disk, concatenate_datasets
from functools import partial
from recipe.image_rl.preprocess_utils import preprocess_row_wrapper, _make_cache_key


logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class ImageRLDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet/Arrow files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet/Arrow file(s) or directory.
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
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
        
        # Schema metadata for arrow data restoration
        self.schema_metadata = None
        self.is_nested_structure = False  # Flag to check if data has nested structure

        # Load fromo cache
        self.preprocessed_cache_dir = config.get("preprocessed_cache_dir", None)
        self.force_rebuild_preprocessed = config.get("force_rebuild_preprocessed", False)

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _load_schema_metadata(self, arrow_dir):
        """Load schema metadata from arrow directory"""
        import json
        
        # Try new format first (schema_metadata.json)
        meta_filepath = os.path.join(arrow_dir, "schema_metadata.json")
        if os.path.exists(meta_filepath):
            with open(meta_filepath, 'r') as f:
                meta = json.load(f)
                self.schema_metadata = meta.get('structure', {})
                self.is_nested_structure = True  # New format uses nested structure
                print(f"[ImageRLDataset] Loaded schema metadata (nested structure) with {len(self.schema_metadata)} fields")
            return

    def _read_files_and_tokenize(self):
        cache_dir = None
        if self.preprocessed_cache_dir:
            cache_key = _make_cache_key(self.data_files)
            cache_dir = os.path.join(self.preprocessed_cache_dir, f"preprocessed_{cache_key}")

        if cache_dir and os.path.exists(os.path.join(cache_dir, "dataset_info.json")) and not self.force_rebuild_preprocessed:
            print(f"[ImageRLDataset] Loading preprocessed dataset from cache: {cache_dir}")
            self.dataframe = load_from_disk(cache_dir)
            # 이미 flatten + 타입복원까지 끝난 데이터로 간주
            self.preprocessed_data = None
            return

        dataframes = []

        for data_path in self.data_files:
            # Check if it's an arrow directory (contains dataset_info.json)
            if os.path.isdir(data_path):
                dataset_info_path = os.path.join(data_path, "dataset_info.json")
                if os.path.exists(dataset_info_path):
                    # Load arrow dataset
                    print(f"[ImageRLDataset] Loading arrow dataset from {data_path}")
                    dataframe = load_from_disk(data_path)
                    dataframes.append(dataframe)
                    
                    # Load schema metadata from parent directory
                    parent_dir = os.path.dirname(data_path)
                    if self.schema_metadata is None:
                        self._load_schema_metadata(parent_dir)
                else:
                    # Directory containing multiple arrow datasets
                    subdirs = [os.path.join(data_path, d) for d in os.listdir(data_path) 
                              if os.path.isdir(os.path.join(data_path, d))]
                    for subdir in subdirs:
                        if os.path.exists(os.path.join(subdir, "dataset_info.json")):
                            print(f"[ImageRLDataset] Loading arrow dataset from {subdir}")
                            dataframe = load_from_disk(subdir)
                            dataframes.append(dataframe)
                    
                    # Load schema metadata from parent directory
                    if self.schema_metadata is None:
                        self._load_schema_metadata(data_path)
            else:
                # Load parquet file
                dataframe = datasets.load_dataset("parquet", data_files=data_path)["train"]
                dataframes.append(dataframe)
        
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        preprocess_func = partial(
            preprocess_row_wrapper,
            schema_metadata=self.schema_metadata,
            is_nested_structure=self.is_nested_structure
        )

        self.dataframe = self.dataframe.map(
            preprocess_func,
            num_proc=64, # self.num_workers,
            desc="Restoring data types",
            load_from_cache_file=False,
        )

        if cache_dir:
            print(f"[ImageRLDataset] Saving preprocessed dataset to: {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)
            self.dataframe.save_to_disk(cache_dir)
            try:
                import json
                meta_path = os.path.join(cache_dir, "schema_metadata.json")
                with open(meta_path, "w") as f:
                    json.dump({"structure": self.schema_metadata}, f)
            except Exception as e:
                print(f"[ImageRLDataset] Warning: failed to save schema_metadata.json: {e}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        prompt = row_dict.pop(self.prompt_key, None)

        row_dict["prompt"] = prompt

        # for tensor batch in DataProto
        row_dict["dummy_tensor"] = torch.zeros(1)

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict.get("data_source"))
        
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()