import random
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from verl.protocol import DataProto
from recipe.fully_async_policy_image_rl.detach_utils import (
    _slice_dataproto_with_meta,
    _concat_dataprotos_with_meta,
)


@dataclass
class BufferEntry:
    """Single uid-group entry in the replay buffer."""
    uid: str
    data: DataProto
    param_version: int
    mean_reward: float
    reward_std: float = 0.0
    max_reward: float = 0.0
    used: bool = False
    use_count: int = 0


class ReplayBuffer:
    """Per-task replay buffer with post-use quality eviction.

    Thread-safe: all public methods acquire an internal lock.

    Design:
      1. All newly generated groups are pushed unconditionally (no score filtering).
      2. Training batches are sampled from the buffer (group-level).
      3. After training, used groups are evaluated:
         - mean_reward >= threshold  -> keep
         - mean_reward <  threshold  -> evict
      4. When the buffer is full, oldest entries are dropped (FIFO).
    """

    def __init__(
        self,
        task_ids: list[int],
        score_thresholds: dict[int, float],
        max_size_per_task: int = -1,
        max_version_gap: int = -1,
        max_use_count: int = -1,
        filter_mode: str = "mean",
        reward_history_size: int = 100,
        max_quantile: float = 0.25,
        std_quantile: float = 0.25,
    ):
        self.task_ids = task_ids
        self.score_thresholds = score_thresholds  # {task_id: min_score}
        self.max_size_per_task = max_size_per_task
        self.max_version_gap = max_version_gap
        self.max_use_count = max_use_count  # -1 means unlimited
        self.filter_mode = filter_mode  # "mean", "std", "max", or "max_and_std"
        self.reward_history_size = reward_history_size
        self.max_quantile = max_quantile
        self.std_quantile = std_quantile
        self._lock = threading.Lock()
        # per-task buffers: task_id -> list[BufferEntry]
        self.buffers: dict[int, list[BufferEntry]] = {
            tid: [] for tid in task_ids
        }
        # per-task FIFO reward histories for dynamic threshold ("max_and_std" mode)
        self.max_reward_history: dict[int, deque[float]] = {
            tid: deque(maxlen=reward_history_size) for tid in task_ids
        }
        self.std_history: dict[int, deque[float]] = {
            tid: deque(maxlen=reward_history_size) for tid in task_ids
        }

    # ------------------------------------------------------------------
    # Push: store ALL uid groups (no score filtering)
    # ------------------------------------------------------------------

    def push(self, batch: DataProto, task_id: int) -> int:
        """Store ALL uid groups from *batch* into the task buffer.

        Each uid group becomes an individual BufferEntry with its mean reward
        pre-computed (used later for post-use eviction).
        Returns the total number of rows stored.
        """
        score_key = f"task{task_id}_token_level_scores"
        if score_key not in batch.batch:
            return 0

        scores = batch.batch[score_key].sum(-1)  # [batch_size]
        uids = batch.non_tensor_batch["uid"]
        param_versions = batch.non_tensor_batch["param_version"]

        # Group indices by uid
        uid_idxs: dict[str, list[int]] = {}
        for i, uid in enumerate(uids):
            uid_idxs.setdefault(uid, []).append(i)

        task_prefix = f"task{task_id}_"
        total_stored = 0

        entries_to_add: list[BufferEntry] = []

        allowed_prefixes = tuple(f"task{t}_" for t in range(1, task_id + 1))
        base_ntb = {"uid", "param_version", "data_source", "prompt_id", "prompt", "reward_model"}

        for uid, idxs in uid_idxs.items():
            # Skip groups that contain any invalid (-100) sample
            uid_scores = [scores[i].item() for i in idxs]
            if any(s < 0 for s in uid_scores):
                continue
            mean_reward = sum(uid_scores) / len(uid_scores)
            max_reward = max(uid_scores)
            if len(uid_scores) > 1:
                variance = sum((s - mean_reward) ** 2 for s in uid_scores) / len(uid_scores)
                reward_std = variance ** 0.5
            else:
                reward_std = 0.0

            # Representative param version (min across group)
            version = min(int(param_versions[i]) for i in idxs)

            # Slice the uid group from the full batch
            filtered = _slice_dataproto_with_meta(batch, idxs)

            # --- key filtering (for per-task storage) ---
            # (a) batch keys: task-prefixed + cross-task context
            cross_task_vals = {}
            if task_id == 2:
                if "task1_gen_imgs_pixel_values" in filtered.batch.keys():
                    cross_task_vals["task2_task1_gen_imgs_pixel_values"] = filtered.batch["task1_gen_imgs_pixel_values"]
                if "task1_token_level_scores" in filtered.batch.keys():
                    cross_task_vals["task2_task1_token_level_scores"] = filtered.batch["task1_token_level_scores"]
            elif task_id == 3:
                if "task1_gen_img_tokens" in filtered.batch.keys():
                    cross_task_vals["task3_task1_gen_img_tokens"] = filtered.batch["task1_gen_img_tokens"]
                if "task1_token_level_scores" in filtered.batch.keys():
                    cross_task_vals["task3_task1_token_level_scores"] = filtered.batch["task1_token_level_scores"]
                if "task2_token_level_scores" in filtered.batch.keys():
                    cross_task_vals["task3_task2_token_level_scores"] = filtered.batch["task2_token_level_scores"]
            for k in list(filtered.batch.keys()):
                if not k.startswith(task_prefix):
                    del filtered.batch[k]
            for k, val in cross_task_vals.items():
                filtered.batch[k] = val

            # (b) non_tensor_batch keys
            cross_ntb_vals = {}
            if task_id == 2:
                if "task1_gen_imgs_pil_list" in filtered.non_tensor_batch:
                    cross_ntb_vals["task2_task1_gen_imgs_pil_list"] = filtered.non_tensor_batch["task1_gen_imgs_pil_list"]
            elif task_id == 3:
                if "task1_gen_imgs_pil_list" in filtered.non_tensor_batch:
                    cross_ntb_vals["task3_task1_gen_imgs_pil_list"] = filtered.non_tensor_batch["task1_gen_imgs_pil_list"]
                if "task2_feedback_texts" in filtered.non_tensor_batch:
                    cross_ntb_vals["task3_task2_feedback_texts"] = filtered.non_tensor_batch["task2_feedback_texts"]
            filtered.non_tensor_batch = {
                k: v for k, v in filtered.non_tensor_batch.items()
                if k in base_ntb or k.startswith(task_prefix)
            }
            filtered.non_tensor_batch.update(cross_ntb_vals)
            if "data_source" in filtered.non_tensor_batch:
                filtered.non_tensor_batch[f"task{task_id}_data_source"] = filtered.non_tensor_batch.pop("data_source")

            # (c) meta_info: cumulative task-prefixed keys
            filtered.meta_info = {
                k: v for k, v in filtered.meta_info.items()
                if k.startswith(allowed_prefixes)
            }

            entries_to_add.append(BufferEntry(
                uid=uid,
                data=filtered,
                param_version=version,
                mean_reward=mean_reward,
                reward_std=reward_std,
                max_reward=max_reward,
                used=False,
            ))
            total_stored += len(idxs)

        with self._lock:
            self.buffers[task_id].extend(entries_to_add)
            # Record max_reward / std into FIFO histories for dynamic threshold
            for entry in entries_to_add:
                self.max_reward_history[task_id].append(entry.max_reward)
                self.std_history[task_id].append(entry.reward_std)
            # FIFO overflow: remove oldest entries if over capacity
            if self.max_size_per_task > 0:
                self._enforce_capacity(task_id)

        return total_stored

    # ------------------------------------------------------------------
    # Sample: draw group-level samples from buffer
    # ------------------------------------------------------------------

    def sample_task(self, task_id: int, n_samples: int,
                    current_version: int = -1) -> DataProto | None:
        """Sample up to *n_samples* rows from *task_id*'s buffer.

        Optionally evicts stale entries first (if max_version_gap is set).
        Marks sampled entries as ``used`` for post-training eviction.
        """
        with self._lock:
            if self.max_version_gap >= 0 and current_version >= 0:
                self._evict_stale(task_id, current_version)

            buf = self.buffers[task_id]
            if not buf:
                return None

            # Shuffle and pick groups until we have enough rows
            indices = list(range(len(buf)))
            random.shuffle(indices)
            selected_indices: list[int] = []
            collected = 0
            for idx in indices:
                if collected >= n_samples:
                    break
                selected_indices.append(idx)
                collected += len(buf[idx].data)

            if not selected_indices:
                return None

            # Mark selected entries as used and tag with version for downstream tracking
            for idx in selected_indices:
                buf[idx].used = True
                buf[idx].use_count += 1
                buf[idx].data.non_tensor_batch["sample_param_version"] = buf[idx].data.non_tensor_batch["param_version"]
                buf[idx].data.non_tensor_batch["entry_use_count"] = np.full(len(buf[idx].data), buf[idx].use_count)

            all_data = [buf[idx].data for idx in selected_indices]
            result = _concat_dataprotos_with_meta(all_data)
            # Truncate to exactly n_samples to guarantee consistent batch sizes
            # across tasks (needed for TensorDict when merging task batches).
            if len(result) > n_samples:
                result = _slice_dataproto_with_meta(result, list(range(n_samples)))
            return result

    # ------------------------------------------------------------------
    # Post-use eviction: evaluate used groups, remove low quality
    # ------------------------------------------------------------------

    def evict_after_use(self, task_id: int) -> tuple[int, int, dict]:
        """Evaluate used entries and evict low quality ones.

        For "max_and_std" mode, thresholds are computed dynamically from
        the FIFO reward histories using per-metric quantiles.
        For other modes, static ``score_thresholds`` are used.

        Returns ``(kept_count, evicted_count, info)`` among used entries.
        ``info`` contains the dynamic thresholds and history sizes for logging.
        Unused entries are always kept.
        """
        with self._lock:
            buf = self.buffers[task_id]
            kept: list[BufferEntry] = []
            evicted_count = 0
            kept_used_count = 0
            info: dict = {}

            # Compute thresholds
            if self.filter_mode == "max_and_std":
                max_hist = self.max_reward_history[task_id]
                std_hist = self.std_history[task_id]
                
                history_half = len(max_hist) >= self.reward_history_size // 2
                max_thr = float(np.quantile(list(max_hist), self.max_quantile)) if history_half else 0.0
                std_thr = float(np.quantile(list(std_hist), self.std_quantile)) if history_half else 0.0
                info = {
                    "max_threshold": max_thr,
                    "std_threshold": std_thr,
                    "max_history_len": len(max_hist),
                    "std_history_len": len(std_hist),
                    "history_half": int(history_half),
                }
            else:
                static_thr = self.score_thresholds.get(task_id, 0.0)

            for entry in buf:
                if entry.used:
                    over_use_limit = (self.max_use_count >= 0 and entry.use_count >= self.max_use_count)
                    if self.filter_mode == "max_and_std":
                        keep = (not over_use_limit
                                and entry.max_reward >= max_thr
                                and entry.reward_std >= std_thr)
                    elif self.filter_mode == "std":
                        keep = not over_use_limit and entry.reward_std >= static_thr
                    elif self.filter_mode == "max":
                        keep = not over_use_limit and entry.max_reward >= static_thr
                    else:  # "mean"
                        keep = not over_use_limit and entry.mean_reward >= static_thr
                    if keep:
                        entry.used = False  # reset for next round
                        kept.append(entry)
                        kept_used_count += 1
                    else:
                        evicted_count += 1
                else:
                    kept.append(entry)

            self.buffers[task_id] = kept
            return kept_used_count, evicted_count, info

    # ------------------------------------------------------------------
    # Internal helpers (caller must hold self._lock)
    # ------------------------------------------------------------------

    def _enforce_capacity(self, task_id: int):
        """Remove oldest entries (FIFO) when buffer exceeds max_size_per_task uid groups."""
        buf = self.buffers[task_id]
        if len(buf) <= self.max_size_per_task:
            return
        del buf[:len(buf) - self.max_size_per_task]

    def _evict_stale(self, task_id: int, current_version: int):
        """Remove entries exceeding max_version_gap for the given task."""
        if self.max_version_gap < 0:
            return
        self.buffers[task_id] = [
            e for e in self.buffers[task_id]
            if current_version - e.param_version <= self.max_version_gap
        ]

    # ------------------------------------------------------------------
    # Size / stats helpers (thread-safe)
    # ------------------------------------------------------------------

    def task_size(self, task_id: int) -> int:
        """Total rows in task buffer."""
        with self._lock:
            return sum(len(e.data) for e in self.buffers[task_id])

    def total_size(self) -> int:
        with self._lock:
            return sum(
                sum(len(e.data) for e in self.buffers[tid])
                for tid in self.task_ids
            )

    def num_entries(self) -> int:
        """Total number of uid-group entries across all tasks."""
        with self._lock:
            return sum(len(buf) for buf in self.buffers.values())

    def size_per_task(self) -> dict[int, int]:
        with self._lock:
            return {
                tid: sum(len(e.data) for e in self.buffers[tid])
                for tid in self.task_ids
            }

    def entries_per_task(self) -> dict[int, int]:
        with self._lock:
            return {tid: len(buf) for tid, buf in self.buffers.items()}

    def stats_per_task(self) -> dict[int, tuple[int, int]]:
        """Return (row_count, entry_count) per task under a single lock."""
        with self._lock:
            return {
                tid: (sum(len(e.data) for e in self.buffers[tid]), len(self.buffers[tid]))
                for tid in self.task_ids
            }
