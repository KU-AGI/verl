# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from typing import Optional
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


ANSWER_TAG = r"<ANSWER>(.*?)</ANSWER>"


def exact_match(ot_smi, gt_smi):
    """SMILES exact match"""
    try:
        m_out = Chem.MolFromSmiles(ot_smi)
        m_gt = Chem.MolFromSmiles(gt_smi)
        
        if m_out is None or m_gt is None:
            return 0
            
        if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
            return 1
    except:
        pass
    return 0


def remove_tag(s: str) -> str:
    """Remove LaTeX tags from a string.

    Args:
        tags: String with LaTeX tags

    Returns:
        String without LaTeX tags
    """
    smiles_match = re.search(ANSWER_TAG, s, re.DOTALL)
    if smiles_match:
        tags = smiles_match.group(1)
    else:
        tags = s
    return tags.strip()


def is_correct_strict_tag(
    pred: str, gt: str
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract and check the boxed answer
    extracted_pred = remove_tag(pred) if pred is not None else None

    return 1 if exact_match(extracted_pred, gt) else 0, extracted_pred


def verify(
    solution_str: str, answer: str
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer

    Returns:
        True if the solution is correct, False otherwise
    """
    correct, pred = is_correct_strict_tag(solution_str, answer)
    return correct == 1, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_tag_verify: Whether to use strict tag verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, 0.0 for incorrect)
    """
    # Verify the solution
    correct, pred = verify(solution_str, ground_truth)

    reward = 1.0 if correct else 0.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }
