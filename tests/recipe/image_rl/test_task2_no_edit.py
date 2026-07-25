import importlib.util
import math
import sys
import types

import pytest
import torch

from recipe.image_rl.utils import (
    CANONICAL_NO_EDIT_FEEDBACK,
    FormattingEvaluatorV2,
    build_segment_response_mask,
    classify_task2_feedback,
    extract_task2_feedback,
    should_route_task3,
)


@pytest.mark.parametrize(
    ("feedback", "classification", "route_task3"),
    [
        (CANONICAL_NO_EDIT_FEEDBACK, "canonical_no_edit", False),
        (" No need. ", "noncanonical_no_edit", False),
        ("No need to generate feedback", "noncanonical_no_edit", False),
        ("no need to generate feedback.", "noncanonical_no_edit", False),
        ("No feedback needed.", "noncanonical_no_edit", False),
        ("No correction needed, but change the sky.", "noncanonical_no_edit", False),
        ("No need to change X, but change Y.", "noncanonical_no_edit", False),
        ("Step 1: Change the sky.", "other", True),
        ("There is no need to change X, but change Y.", "other", True),
        ("", "other", False),
        (None, "other", False),
    ],
)
def test_no_edit_classification_and_routing(feedback, classification, route_task3):
    assert classify_task2_feedback(feedback) == classification
    assert should_route_task3(feedback) is route_task3


def test_extract_feedback_from_v2_and_v3_responses():
    v2 = (
        "summary\nSecond, Decompose summarize\ntuples\n"
        "Third, Verify that the decomposed elements align with the image.\nvqa\n"
        "Fourth, Generate corrective feedback.\nNo need."
    )
    v3 = (
        "tuples\nSecond, Verify that the decomposed elements align with the image.\nvqa\n"
        "Third, Generate corrective feedback.\nStep 1: Change the sky."
    )
    assert extract_task2_feedback(v2) == "No need."
    assert extract_task2_feedback(v3) == "Step 1: Change the sky."
    assert should_route_task3(v2) is False
    assert should_route_task3(v3) is True


class _CharTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_v2_segment_mask_has_four_local_segments():
    tokenizer = _CharTokenizer()
    response = (
        "summary"
        + FormattingEvaluatorV2.SECOND_PATTERN
        + "tuples"
        + FormattingEvaluatorV2.THIRD_PATTERN
        + "vqa"
        + FormattingEvaluatorV2.FOURTH_PATTERN
        + "feedback"
    )
    ids = torch.tensor([tokenizer.encode(response)], dtype=torch.long)
    mask = build_segment_response_mask(
        ids,
        tokenizer,
        second_pattern=FormattingEvaluatorV2.SECOND_PATTERN,
        third_pattern=FormattingEvaluatorV2.THIRD_PATTERN,
        fourth_pattern=FormattingEvaluatorV2.FOURTH_PATTERN,
    )
    assert set(mask[0].tolist()) == {2, 3, 4, 5}


def _finalize_task2_reward_extra_info(*args, **kwargs):
    if "openai" not in sys.modules and importlib.util.find_spec("openai") is None:
        openai_stub = types.ModuleType("openai")

        class _AsyncOpenAI:
            def __init__(self, *args, **kwargs):
                pass

        openai_stub.AsyncOpenAI = _AsyncOpenAI
        sys.modules["openai"] = openai_stub

    if "mathruler" not in sys.modules and importlib.util.find_spec("mathruler") is None:
        mathruler_stub = types.ModuleType("mathruler")
        grader_stub = types.ModuleType("mathruler.grader")
        grader_stub.extract_boxed_content = lambda text: text
        mathruler_stub.grader = grader_stub
        sys.modules["mathruler"] = mathruler_stub
        sys.modules["mathruler.grader"] = grader_stub

    from recipe.image_rl.reward_function_fine_grained import (
        finalize_task2_reward_extra_info as finalize,
    )

    return finalize(*args, **kwargs)


def _stage_info(**overrides):
    info = {
        "task2_rule_based_format_reward": 1.0,
        "task2_prompt_to_summary_reward": 2.0,
        "task2_summary_to_tuple_reward": 2.0,
        "task2_tuple_to_vqa_reward": 2.0,
        "task2_vqa_to_feedback_content_reward": 1.0,
        "task2_no_feedback_needed": 0,
        "task2_noncanonical_no_edit": 0,
    }
    info.update(overrides)
    return info


def test_canonical_no_edit_reward_requires_target_match():
    matched, matched_info = _finalize_task2_reward_extra_info(
        _stage_info(task2_no_feedback_needed=1),
        decision_vqa=1.0,
        decision_source="task1",
    )
    mismatched, mismatch_info = _finalize_task2_reward_extra_info(
        _stage_info(task2_no_feedback_needed=1),
        decision_vqa=0.0,
        decision_source="task1",
    )
    assert matched == pytest.approx(2.0)
    assert matched_info["task2_vqa_to_feedback_reward"] == 2.0
    assert mismatched == 0.0
    assert mismatch_info["task2_vqa_to_feedback_reward"] == 0.0


def test_noncanonical_no_edit_zeros_only_feedback_stage():
    reward, info = _finalize_task2_reward_extra_info(
        _stage_info(task2_noncanonical_no_edit=1),
        decision_vqa=0.0,
        decision_source="task1",
    )
    assert reward == 0.0
    assert info["task2_prompt_to_summary_reward"] == 2.0
    assert info["task2_summary_to_tuple_reward"] == 2.0
    assert info["task2_tuple_to_vqa_reward"] == 2.0
    assert info["task2_vqa_to_feedback_reward"] == 0.0


def test_edit_feedback_uses_judge_content_when_target_requires_edit():
    reward, info = _finalize_task2_reward_extra_info(
        _stage_info(task2_vqa_to_feedback_content_reward=1.0),
        decision_vqa=0.0,
        decision_source="task1",
    )
    assert reward == pytest.approx(math.prod([2.0, 2.0, 2.0, 1.0]) ** 0.25)
    assert info["task2_vqa_to_feedback_reward"] == 1.0


def test_image_prompt_valid_mask_rejects_partial_placeholders():
    from recipe.image_rl.utils import image_prompt_valid_mask
    micro_batch = {
        "task3_input_ids": torch.tensor(
            [
                [99, 99, 99, 99, 0],
                [99, 99, 99, 0, 0],
                [99, 99, 99, 99, 0],
            ],
            dtype=torch.long,
        ),
        "task3_attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
            ],
            dtype=torch.long,
        ),
    }
    valid_rows = image_prompt_valid_mask(
        micro_batch["task3_input_ids"],
        micro_batch["task3_attention_mask"],
        image_id=99,
        expected_count=4,
    )
    assert valid_rows.tolist() == [True, False, False]


def test_actor_zeroes_all_invalid_task3_rows_in_place():
    from types import SimpleNamespace

    from recipe.image_rl.dp_actor import DataParallelImageGenerationActor

    actor = DataParallelImageGenerationActor.__new__(DataParallelImageGenerationActor)
    actor.processor = SimpleNamespace(image_id=99)
    actor.config = {"image_token_num_per_image": 4}
    response_mask = torch.ones((2, 3), dtype=torch.long)
    micro_batch = {
        "task3_input_ids": torch.tensor(
            [
                [1, 2, 3, 0],
                [99, 99, 1, 0],
            ],
            dtype=torch.long,
        ),
        "task3_attention_mask": torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
            ],
            dtype=torch.long,
        ),
        "task3_response_mask": response_mask,
    }
