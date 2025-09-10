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
Preprocess the syntheticreact dataset to parquet format
"""

import argparse
import os
import re

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data/verl/data/chem_dapo")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir)

    data_source = "chem_dapo"

    if args.train:
        dataset = datasets.load_from_disk(f"{args.local_dir}/train", "main")
    elif args.test:
        dataset = datasets.load_from_disk(f"{args.local_dir}/test", "main")
    else:
        raise ValueError("Either --train or --test must be specified")

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

    if args.train:
        dataset = dataset.map(function=make_map_fn("train"), with_indices=True, load_from_cache_file=False)
        dataset.to_parquet(os.path.join(args.local_dir, "syntheticreact_9k_train.parquet"))
    elif args.test:
        dataset = dataset.map(function=make_map_fn("test"), with_indices=True, load_from_cache_file=False)
        dataset.to_parquet(os.path.join(args.local_dir, "syntheticreact_3k_test.parquet"))
