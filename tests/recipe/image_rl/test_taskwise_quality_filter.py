from types import SimpleNamespace

import numpy as np
import torch

from verl import DataProto
from recipe.fully_async_policy_image_rl.detach_utils import quality_filter_rollout_sample


def _task_batch(task_id, scores, *, uid="sample", turns=None, segment_mask=None):
    scores = torch.tensor(scores, dtype=torch.float32)
    n_rows, response_len = scores.shape
    if turns is None:
        turns = [0] * n_rows
    tensors = {
        f"task{task_id}_token_level_scores": scores,
        f"task{task_id}_response_mask": torch.ones((n_rows, response_len), dtype=torch.long),
        "turn_idx": torch.tensor(turns, dtype=torch.int32),
    }
    if segment_mask is not None:
        tensors["task2_segment_mask"] = torch.tensor(segment_mask, dtype=torch.long)
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "uid": np.array([uid] * n_rows, dtype=object),
            "phase": np.ones(n_rows, dtype=np.int64),
        },
    )


def _rollout_sample(task_batches):
    full_batch = next(batch for batch in task_batches.values() if batch is not None)
    n_rows = len(full_batch)
    return SimpleNamespace(
        full_batch=full_batch,
        task_batches=task_batches,
        processing_times=[0.0] * n_rows,
        param_version_start=[0] * n_rows,
        param_version_end=[0] * n_rows,
    )


def test_phase1_filters_each_task_group_independently():
    task1 = _task_batch(1, [[1.0], [1.0], [1.0], [1.0]])
    task2 = _task_batch(
        2,
        [
            [1.0, 0.0, 2.0, 3.0],
            [1.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 2.0, 3.0],
            [1.0, 1.0, 2.0, 3.0],
        ],
        segment_mask=[[2, 3, 4, 5]] * 4,
    )
    task3 = _task_batch(3, [[0.0], [1.0], [0.0], [1.0]])
    sample = _rollout_sample({1: task1, 2: task2, 3: task3})

    quality_filter_rollout_sample(sample, group_size=4, task_ids=[1, 2, 3])

    assert sample.task_batches[1] is None
    assert len(sample.task_batches[2]) == 4
    assert len(sample.task_batches[3]) == 4
    assert sample.full_batch.meta_info["fully_async/filter_groups/phase1_total_groups"] == 3
    assert sample.full_batch.meta_info["fully_async/filter_groups/phase1_kept_groups"] == 2


def test_task2_uses_per_segment_std_and_per_turn_groups():
    turns = [0] * 4 + [1] * 4
    task2 = _task_batch(
        2,
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ],
        turns=turns,
        segment_mask=[[2, 3, 4, 5]] * 8,
    )
    sample = _rollout_sample({2: task2})

    quality_filter_rollout_sample(sample, group_size=4, task_ids=[2])

    kept = sample.task_batches[2]
    assert kept is not None
    assert len(kept) == 4
    assert kept.batch["turn_idx"].tolist() == [1, 1, 1, 1]
    # Turn 1 has a constant total reward of 3, but segments 2 and 3 vary.
    assert kept.batch["task2_token_level_scores"].sum(dim=-1).tolist() == [3.0] * 4
