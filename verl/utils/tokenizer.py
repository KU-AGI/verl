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
"""Utils for tokenization."""

import warnings

__all__ = ["hf_tokenizer", "hf_processor"]


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer, AddedToken

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)

    reactant_funcgroup_start = "<REACTANT_FUNCGROUP>" # 151669
    reactant_funcgroup_end = "</REACTANT_FUNCGROUP>" # 151670
    product_funcgroup_start = "<PRODUCT_FUNCGROUP>" # 151671
    product_funcgroup_end = "</PRODUCT_FUNCGROUP>" # 151672
    molecular_role_start = "<MOLECULAR_ROLE>" # 151673
    molecular_role_end = "</MOLECULAR_ROLE>" # 151674
    condition_start = "<CONDITION>" # 151675
    condition_end = "</CONDITION>" # 151676
    precursor_stat_start = "<PRECURSOR_STAT>" # 151677
    precursor_stat_end = "</PRECURSOR_STAT>" # 151678
    reactant_stat_start = "<REACTANT_STAT>" # 151679
    reactant_stat_end = "</REACTANT_STAT>" # 151680
    product_stat_start = "<PRODUCT_STAT>" # 151681
    product_stat_end = "</PRODUCT_STAT>" # 151682
    template_start = "<TEMPLATE>" # 151683
    template_end = "</TEMPLATE>" # 151684
    bond_disconnect_start = "<BOND_DISCONNECT>" # 151685
    bond_disconnect_end = "</BOND_DISCONNECT>" # 151686
    synthon_start = "<SYNTHON>" # 151687
    synthon_end = "</SYNTHON>" # 151688
    synthetic_equivalent_start = "<SYNTHETIC_EQUIVALENT>" # 151689
    synthetic_equivalent_end = "</SYNTHETIC_EQUIVALENT>" # 151690
    reactant_removed_start = "<REACTANT_REMOVED_FUNCGROUP>" # 151691
    reactant_removed_end = "</REACTANT_REMOVED_FUNCGROUP>"# 151692
    product_added_start = "<PRODUCT_ADDED_FUNCGROUP>" # 151693
    product_added_end = "</PRODUCT_ADDED_FUNCGROUP>" # 151694
    special_tokens = [
        reactant_funcgroup_start, reactant_funcgroup_end,
        product_funcgroup_start, product_funcgroup_end,
        molecular_role_start, molecular_role_end,
        condition_start, condition_end,
        precursor_stat_start, precursor_stat_end,
        reactant_stat_start, reactant_stat_end,
        product_stat_start, product_stat_end,
        template_start, template_end,
        bond_disconnect_start, bond_disconnect_end,
        synthon_start, synthon_end,
        synthetic_equivalent_start, synthetic_equivalent_end,
        reactant_removed_start, reactant_removed_end,
        product_added_start, product_added_end,
    ]
    added = tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                AddedToken(t, special=True) for t in special_tokens
            ]
        }
    )

    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
