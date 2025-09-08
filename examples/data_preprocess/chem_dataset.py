# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data/verl/data")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "chem_dapo"

    dataset = datasets.load_from_disk("/data/llm-reaction-reasoning/data/orderly/main_training/data_dir/chem_dapo", "main")

    train_dataset, test_dataset = dataset.train_test_split(test_size=0.1).values()

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            messages = example.pop("messages")
            solution = example.pop("answers")
            
            # Choose extra columns
            extra_columns = ["rxn_str", "reactants", "reagents", "products", "solvent", "yields", "class_name"]
            
            data = {
                "data_source": data_source,
                "prompt": messages,
                "ability": "chemistry",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    k: v for k, v in example.items() if k in extra_columns
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, load_from_cache_file=False)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, load_from_cache_file=False)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
