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


logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
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


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
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
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
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

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    images = (
                        [process_image(image) for image in doc[image_key]]
                        if image_key in doc and doc[image_key]
                        else None
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]]
                        if video_key in doc and doc[video_key]
                        else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs
                        )
                    )

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

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

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos = [process_video(video) for video in row_dict_videos]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
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


class JanusTextOnlyRLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key=None,
                 image_key=None,
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir=None,
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False,
                 system_prompt="",
                 prompt_template=None,
                 cot_generate=False,
                 ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts
        self.system_prompt = system_prompt
        self.cot_generate = cot_generate
        self.prompt_template = prompt_template
        self.data_source_list = []
        if self.prompt_template is None:
            self.prompt_template = "A photo of {}."
        
        self.prompts = []
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = os.path.expanduser(parquet_file)
            file_name = os.path.basename(self.parquet_files[i]).replace('.txt', '')
            if self.parquet_files[i].endswith('.txt'):
                with open(self.parquet_files[i], 'r') as f:
                    prompts = f.readlines()
                for prompt in prompts:
                    if self.filter_overlong_prompts:
                        if len(tokenizer.encode(prompt, add_special_tokens=False))+10 > self.max_prompt_length: # +5 for <user>, <assistant>, <img_start>, etc..
                            continue
                    self.prompts.append(prompt.strip())
                    self.data_source_list.append(file_name)
                        
                # self.prompts.extend([prompt.strip() for prompt in prompts])
                # self.data_source_list.extend([file_name] * len(prompts))
            else:
                raise ValueError(f"Unsupported file format: {self.parquet_files[i]}")
        
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = {}

        chat = [
            {
                "role": "<|User|>",
                "content": self.prompt_template.format(self.prompts[item]),
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=chat,
            sft_format=self.processor.sft_format,
            system_prompt=self.system_prompt,
        )
        
        if not self.cot_generate:
            prompt = sft_format + self.processor.image_start_tag
        else:
            prompt = sft_format
        
        raw_prompt = prompt

        is_multi_modal = False
        assert not is_multi_modal, "JanusTextOnlyRLHFDataset only supports t2i data"
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        input_ids = input_ids[0]
        attention_mask = attention_mask[0]
        sentence_start_token, image_start_token = self.tokenizer.encode(self.processor.image_start_tag)
        input_ids = torch.cat([torch.LongTensor([self.tokenizer.pad_token_id]), input_ids])
        attention_mask = torch.cat([torch.LongTensor([0]), attention_mask])

        num_pad = torch.sum(input_ids == self.tokenizer.pad_token_id, dim=-1)
        last_pad_idx = num_pad - 1
        
        input_ids[last_pad_idx] = sentence_start_token
        attention_mask[last_pad_idx] = 1
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict['data_source'] = self.data_source_list[item]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = self.prompts[item]

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
    
    
class DummyJanusDPORLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False,
                 system_prompt="",
                 prompt_template=None,
                 cot_generate=False,
                 ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts
        self.system_prompt = system_prompt
        self.cot_generate = cot_generate
        self.prompt_template = prompt_template
        self.dummy_prompts = [
            'One apple and two bananas on a plate',
            'A plate on the left of the cup',
            'A lamp on the left side of the bed, and a clock on the right',
            'A red apple on the left and a green apple on the right, both on a plate with a blue rim',
            'a plant in front of a jar',
            'a person above a leaf',
            'three suitcases and one bottle'
            'one screen and two watchs',
            'A dog lying on a rug in front of a fireplace',
            'two wines and two eggs',
            'A white mug in front of a green notebook on a desk',
            'a green bed and a khaki dish',
            'a brown bag and a pink car',
            'A pencil and an eraser beside an open notebook',
            'A cat sitting on the windowsill with a plant to its left',
            'A white book on top of a black book, which is on top of a red book',
            'A teddy bear on the right side of the pillow on a bed'
        ]
        
    def __len__(self):
        return 32

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = {}

        chat = [
            {
                "role": "<|User|>",
                "content": self.prompt_template.format(self.dummy_prompts[item%len(self.dummy_prompts)]),
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=chat,
            sft_format=self.processor.sft_format,
            system_prompt=self.system_prompt,
        )
        if not self.cot_generate:
            prompt = sft_format + self.processor.image_start_tag
        else:
            prompt = sft_format
        
        raw_prompt = prompt

        is_multi_modal = False
        assert not is_multi_modal, "JanusRLHFDataset only supports t2i data"
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        input_ids = input_ids[0]
        attention_mask = attention_mask[0]
        sentence_start_token, image_start_token = self.tokenizer.encode(self.processor.image_start_tag)
        input_ids = torch.cat([torch.LongTensor([self.tokenizer.pad_token_id]), input_ids, torch.LongTensor([image_start_token])])
        attention_mask = torch.cat([torch.LongTensor([0]), attention_mask, torch.LongTensor([1])])

        num_pad = torch.sum(input_ids == self.tokenizer.pad_token_id, dim=-1)
        last_pad_idx = num_pad - 1
        
        input_ids[last_pad_idx] = sentence_start_token
        attention_mask[last_pad_idx] = 1
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()

class InterleaveDataset():
    def __init__(self, data_dir:str, split:str=None):
        self.data_dir = data_dir
        self.datasets = {}
        self.dataset_names = []  # keep track of the order of datasets
        self.test_datasets = {}
        self.split = split
        self.debug = False
        
        self.load_datasets()
        self.lengths = {name: len(dataset) for name, dataset in self.datasets.items()}
        self.total_length = sum(self.lengths.values())
        self.dataset_start_idx = {name: sum([self.lengths[n] for n in self.dataset_names[:i]]) for i, name in enumerate(self.dataset_names)}
        
    def load_single_dataset(self, data_path):
        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"
            
        def load(data_path):
            try:
                dataset = load_dataset(data_path, split=data_split)
                print(f"loading dataset: {data_path}")
            except Exception as e:
                print(f"loading dataset from disk: {data_path}")
                dataset = load_from_disk(data_path)
            return dataset
        
        if 'batch_0' in os.listdir(data_path): # if the dataset is splited into batches
            datasets = []
            for batch in os.listdir(data_path):
                path = os.path.join(data_path, batch)
                datasets.append(load(path))
            dataset = concatenate_datasets(datasets)
        else:
            dataset = load(data_path)
        
        return dataset
    
    def split_for_test(self):
        test_datasets = {}
        for dataset_name in self.dataset_names:
            dataset = self.datasets[dataset_name]
            dataset = dataset.train_test_split(
                test_size=0.9,
                shuffle=False,
                seed=0,
            )
            test_datasets[dataset_name] = dataset['test']
            self.datasets[dataset_name] = dataset['train']
            if self.debug:
                print("Debugging dataset")
                test_datasets[dataset_name] = test_datasets[dataset_name].select(range(8))
                self.datasets[dataset_name] = self.datasets[dataset_name].select(range(8))
        return self.datasets, test_datasets
        
    def load_datasets(self):
        if '@' in self.data_dir:
            data_path = self.data_dir.split('@')[0]
            name = os.path.basename(data_path)
            self.datasets[name] = self.load_single_dataset(data_path)
            self.dataset_names = [name]
            return
        
        sub_dirs = os.listdir(self.data_dir)
        json_files = [f for f in sub_dirs if f.endswith('.json')]
        if len(json_files) > 0:
            self.datasets[self.data_dir] = self.load_single_dataset(self.data_dir)
            self.dataset_names = [os.path.basename(self.data_dir)]
        else:
            for dataset_name in os.listdir(self.data_dir):
                data_path = os.path.join(self.data_dir, dataset_name)
                self.datasets[dataset_name] = self.load_single_dataset(data_path)
                self.dataset_names.append(dataset_name)
            if self.split is not None:
                self.datasets, self.test_datasets = self.split_for_test()
            if self.split == 'test':
                self.datasets = self.test_datasets
            
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index):
        for dataset_name in self.dataset_names:
            if index < self.lengths[dataset_name]:
                item = self.datasets[dataset_name][index]
                item['data_source'] = dataset_name
                return item
            else:
                index -= self.lengths[dataset_name]
        raise IndexError("Index out of range")