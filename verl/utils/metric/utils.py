# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Metrics utils.
"""

from typing import Any

import numpy as np


def _flatten_metric_values(val: Any) -> np.ndarray:
    """Flatten nested numeric metric values into a 1-D numpy array."""
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()

    if isinstance(val, np.ndarray):
        if val.dtype == object:
            parts = [_flatten_metric_values(v) for v in val.tolist()]
            parts = [p for p in parts if p.size > 0]
            return np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)
        return val.reshape(-1)

    if isinstance(val, (list, tuple)):
        parts = [_flatten_metric_values(v) for v in val]
        parts = [p for p in parts if p.size > 0]
        return np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)

    return np.asarray([val], dtype=np.float64)


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        val = _flatten_metric_values(val)
        if "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics
