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
    """Single (uid, turn_idx)-group entry in the replay buffer.

    Multi-turn rollouts produce multiple rows per uid (one per turn for task2
    and task3). We key each entry by the `(uid, turn_idx)` pair so:
      * Pushes coming from different turns of the same uid produce distinct
        entries rather than collapsing into one big per-uid group.
      * Sampling naturally picks per-(uid, turn) groups, which matches the
        GRPO group-size assumption on downstream training.
    `turn_idx=0` for task1 pushes (task1 only runs in turn 0) and for any
    batch that does not carry a `turn_idx` column (single-turn legacy path).
    """
    uid: str
    turn_idx: int
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

    filter_mode options:
      - "mean"                : keep if mean_reward >= score_thresholds[task_id]
      - "std"                 : keep if reward_std  >= score_thresholds[task_id]
      - "max"                 : keep if max_reward  >= score_thresholds[task_id]
      - "max_and_std_constant": keep if max_reward  >= score_thresholds[task_id]
                                        AND reward_std >= std_thresholds[task_id]
      - "mean_std_constant"   : keep if mean_reward >= score_thresholds[task_id]
                                        AND reward_std >= std_thresholds[task_id]
      - "max_and_std_adaptive": keep if max_reward  >= quantile(max_history, max_quantile)
                                        AND reward_std >= quantile(std_history, std_quantile)
    """

    def __init__(
        self,
        task_ids: list[int],
        score_thresholds: dict[int, float],
        max_size_per_task: int = -1,
        max_version_gap: int = -1,
        max_use_count: int = -1,
        filter_mode: str = "mean",
        # --- constant modes (max_and_std_constant, mean_std_constant) ---
        score_std_thresholds: dict[int, float] | None = None,
        # --- adaptive mode (max_and_std_adaptive) ---
        reward_history_size: int = 100,
        max_quantile: float = 0.25,
        std_quantile: float = 0.25,
    ):
        self.task_ids = task_ids
        self.score_thresholds = score_thresholds        # {task_id: max or mean threshold}
        self.std_thresholds = score_std_thresholds or {}  # {task_id: std threshold}
        self.max_size_per_task = max_size_per_task
        self.max_version_gap = max_version_gap
        self.max_use_count = max_use_count
        self.filter_mode = filter_mode
        # adaptive
        self.reward_history_size = reward_history_size
        self.max_quantile = max_quantile
        self.std_quantile = std_quantile

        self._lock = threading.Lock()
        self.buffers: dict[int, list[BufferEntry]] = {
            tid: [] for tid in task_ids
        }
        # FIFO reward histories — only populated in max_and_std_adaptive mode
        self.max_reward_history: dict[int, deque[float]] = {
            tid: deque(maxlen=reward_history_size) for tid in task_ids
        }
        self.std_history: dict[int, deque[float]] = {
            tid: deque(maxlen=reward_history_size) for tid in task_ids
        }

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a thread-safe snapshot suitable for torch.save."""
        with self._lock:
            return {
                "task_ids": list(self.task_ids),
                "score_thresholds": dict(self.score_thresholds),
                "std_thresholds": dict(self.std_thresholds),
                "max_size_per_task": self.max_size_per_task,
                "max_version_gap": self.max_version_gap,
                "max_use_count": self.max_use_count,
                "filter_mode": self.filter_mode,
                "reward_history_size": self.reward_history_size,
                "max_quantile": self.max_quantile,
                "std_quantile": self.std_quantile,
                "buffers": self.buffers,
                "max_reward_history": {
                    tid: list(hist) for tid, hist in self.max_reward_history.items()
                },
                "std_history": {
                    tid: list(hist) for tid, hist in self.std_history.items()
                },
            }

    def load_state_dict(self, state: dict) -> None:
        """Restore a replay-buffer snapshot created by state_dict."""
        with self._lock:
            self.task_ids = list(state.get("task_ids", self.task_ids))
            self.score_thresholds = dict(state.get("score_thresholds", self.score_thresholds))
            self.std_thresholds = dict(state.get("std_thresholds", self.std_thresholds))
            self.max_size_per_task = state.get("max_size_per_task", self.max_size_per_task)
            self.max_version_gap = state.get("max_version_gap", self.max_version_gap)
            self.max_use_count = state.get("max_use_count", self.max_use_count)
            self.filter_mode = state.get("filter_mode", self.filter_mode)
            self.reward_history_size = state.get("reward_history_size", self.reward_history_size)
            self.max_quantile = state.get("max_quantile", self.max_quantile)
            self.std_quantile = state.get("std_quantile", self.std_quantile)

            restored_buffers = state.get("buffers", {})
            self.buffers = {tid: list(restored_buffers.get(tid, [])) for tid in self.task_ids}

            restored_max_hist = state.get("max_reward_history", {})
            restored_std_hist = state.get("std_history", {})
            self.max_reward_history = {
                tid: deque(restored_max_hist.get(tid, []), maxlen=self.reward_history_size)
                for tid in self.task_ids
            }
            self.std_history = {
                tid: deque(restored_std_hist.get(tid, []), maxlen=self.reward_history_size)
                for tid in self.task_ids
            }

    def push(self, batch: DataProto, task_id: int) -> int:
        score_key = f"task{task_id}_token_level_scores"
        outcome_key = "outcome_token_level_scores"
        if score_key not in batch.batch and outcome_key not in batch.batch:
            return 0

        uids = batch.non_tensor_batch["uid"]
        param_versions = batch.non_tensor_batch["param_version"]
        phase_arr = None
        if "phase" in batch.non_tensor_batch:
            phase_arr = np.asarray(batch.non_tensor_batch["phase"])

        # Per-turn grouping: rows are keyed by (uid, turn_idx) so each turn's
        # rollout of the same prompt lives in its own group entry. turn_idx
        # may come from the non_tensor_batch (multi-turn) or from the batch
        # tensor (trainer stamping). Single-turn or missing → default to 0.
        turn_idx_arr = None
        if "turn_idx" in batch.non_tensor_batch:
            turn_idx_arr = batch.non_tensor_batch["turn_idx"]
        elif "turn_idx" in batch.batch:
            t = batch.batch["turn_idx"]
            turn_idx_arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

        key_idxs: dict[tuple[str, int], list[int]] = {}
        for i, uid in enumerate(uids):
            t_idx = int(turn_idx_arr[i]) if turn_idx_arr is not None else 0
            key_idxs.setdefault((uid, t_idx), []).append(i)

        task_prefix = f"task{task_id}_"
        total_stored = 0
        entries_to_add: list[BufferEntry] = []

        allowed_prefixes = tuple(f"task{t}_" for t in range(1, task_id + 1))
        base_ntb = {
            "uid",
            "turn_idx",
            "trajectory_id",
            "branch_id",
            "parent_branch_id",
            "param_version",
            "data_source",
            "prompt_id",
            "prompt",
            "reward_model",
            "phase",
            "grpo_group_id",
            "outcome_A_T",
            "outcome_IF_bar",
            "outcome_P_bar",
            "outcome_T",
        }

        for (uid, turn_idx), idxs in key_idxs.items():
            phases = None
            if phase_arr is not None:
                phases = {int(phase_arr[i]) for i in idxs}
                if len(phases) != 1:
                    # Mixed-phase groups indicate an upstream grouping bug.
                    # Skip rather than storing semantically inconsistent stats.
                    continue
                phase_id = next(iter(phases))
            else:
                # Legacy fallback: treat missing phase as local-reward path.
                phase_id = 0

            stats_key = score_key
            if stats_key not in batch.batch:
                # Legacy fallback for older batches. New MDP training stores
                # discounted returns directly in task{id}_token_level_scores,
                # so replay filtering should rank by the same task score that
                # the actor loss will consume.
                stats_key = outcome_key if outcome_key in batch.batch else None
            if stats_key is None:
                continue

            score_tensor = batch.batch[stats_key].sum(-1)
            uid_scores = [float(score_tensor[i].item()) for i in idxs]
            if (
                not uid_scores
                or not np.all(np.isfinite(uid_scores))
                or any(s == -100 for s in uid_scores)
            ):
                continue
            # Task2 V3 has three fixed reward segments (2/3/4). Normalize its
            # replay quality statistic to the per-step average while keeping
            # legacy outcome-only batches on their original scalar scale.
            if task_id == 2 and stats_key == score_key:
                uid_scores = [score / 3.0 for score in uid_scores]
            mean_reward = sum(uid_scores) / len(uid_scores)
            max_reward = max(uid_scores)
            if len(uid_scores) > 1:
                variance = sum((s - mean_reward) ** 2 for s in uid_scores) / len(uid_scores)
                reward_std = variance ** 0.5
            else:
                reward_std = 0.0

            version = min(int(param_versions[i]) for i in idxs)
            filtered = _slice_dataproto_with_meta(batch, idxs)

            cross_task_vals = {}
            # Multi-turn: task2/task3 training should consume the image the
            # policy actually saw at rollout time.
            if task_id == 2:
                img_src = (
                    "current_imgs_pixel_values"
                    if "current_imgs_pixel_values" in filtered.batch.keys()
                    else "task1_gen_imgs_pixel_values"
                )
                if img_src in filtered.batch.keys():
                    cross_task_vals["task2_task1_gen_imgs_pixel_values"] = filtered.batch[img_src]
                if "task1_token_level_scores" in filtered.batch.keys():
                    cross_task_vals["task2_task1_token_level_scores"] = filtered.batch["task1_token_level_scores"]
            elif task_id == 3:
                if "task3_input_img_tokens" in filtered.batch.keys():
                    cross_task_vals["task3_task1_gen_img_tokens"] = filtered.batch["task3_input_img_tokens"]
                if "task1_token_level_scores" in filtered.batch.keys():
                    cross_task_vals["task3_task1_token_level_scores"] = filtered.batch["task1_token_level_scores"]
                if "task2_token_level_scores" in filtered.batch.keys():
                    cross_task_vals["task3_task2_token_level_scores"] = filtered.batch["task2_token_level_scores"]
            # Preserve legacy outcome fields and `turn_idx` for diagnostics /
            # backward-compatible dumps. MDP training and replay filtering use
            # task{id}_token_level_scores as the source of truth.
            keep_non_task_prefix = {
                "outcome_token_level_scores",
                "outcome_task_id",
                "turn_idx",
            }
            for k in list(filtered.batch.keys()):
                if k in keep_non_task_prefix:
                    continue
                if not k.startswith(task_prefix):
                    del filtered.batch[k]
            for k, val in cross_task_vals.items():
                filtered.batch[k] = val

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

            filtered.meta_info = {
                k: v for k, v in filtered.meta_info.items()
                if k.startswith(allowed_prefixes)
            }

            entries_to_add.append(BufferEntry(
                uid=uid,
                turn_idx=int(turn_idx),
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
            # history는 adaptive 모드에서만 기록
            if self.filter_mode == "max_and_std_adaptive":
                for entry in entries_to_add:
                    self.max_reward_history[task_id].append(entry.max_reward)
                    self.std_history[task_id].append(entry.reward_std)
            if self.max_size_per_task > 0:
                self._enforce_capacity(task_id)

        return total_stored

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample_task(self, task_id: int, n_samples: int,
                    current_version: int = -1) -> DataProto | None:
        with self._lock:
            if self.max_version_gap >= 0 and current_version >= 0:
                self._evict_stale(task_id, current_version)

            buf = self.buffers[task_id]
            if not buf:
                return None

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

            for idx in selected_indices:
                buf[idx].used = True
                buf[idx].use_count += 1
                buf[idx].data.non_tensor_batch["sample_param_version"] = buf[idx].data.non_tensor_batch["param_version"]
                buf[idx].data.non_tensor_batch["entry_use_count"] = np.full(len(buf[idx].data), buf[idx].use_count)

            all_data = [buf[idx].data for idx in selected_indices]
            result = _concat_dataprotos_with_meta(all_data)
            if len(result) > n_samples:
                result = _slice_dataproto_with_meta(result, list(range(n_samples)))
            return result

    # ------------------------------------------------------------------
    # Post-use eviction
    # ------------------------------------------------------------------

    def evict_after_use(self, task_id: int) -> tuple[int, int, dict]:
        with self._lock:
            buf = self.buffers[task_id]
            kept: list[BufferEntry] = []
            evicted_count = 0
            kept_used_count = 0
            info: dict = {}

            # ── threshold 계산 ──────────────────────────────────────────
            if self.filter_mode == "max_and_std_adaptive":
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

            elif self.filter_mode == "max_and_std_constant":
                max_thr = self.score_thresholds.get(task_id, 0.0)
                std_thr = self.std_thresholds.get(task_id, 0.0)
                info = {"max_threshold": max_thr, "std_threshold": std_thr}

            elif self.filter_mode == "mean_std_constant":
                mean_thr = self.score_thresholds.get(task_id, 0.0)
                std_thr  = self.std_thresholds.get(task_id, 0.0)
                info = {"mean_threshold": mean_thr, "std_threshold": std_thr}

            else:  # "mean" / "std" / "max"
                static_thr = self.score_thresholds.get(task_id, 0.0)

            # ── 필터링 ──────────────────────────────────────────────────
            for entry in buf:
                if entry.used:
                    over_use_limit = (self.max_use_count >= 0 and entry.use_count >= self.max_use_count)

                    if self.filter_mode == "max_and_std_adaptive":
                        keep = (not over_use_limit
                                and entry.max_reward  >= max_thr
                                and entry.reward_std  >= std_thr)
                    elif self.filter_mode == "max_and_std_constant":
                        keep = (not over_use_limit
                                and entry.max_reward  >= max_thr
                                and entry.reward_std  >= std_thr)
                    elif self.filter_mode == "mean_std_constant":
                        keep = (not over_use_limit
                                and entry.mean_reward >= mean_thr
                                and entry.reward_std  >= std_thr)
                    elif self.filter_mode == "std":
                        keep = not over_use_limit and entry.reward_std  >= static_thr
                    elif self.filter_mode == "max":
                        keep = not over_use_limit and entry.max_reward  >= static_thr
                    else:  # "mean"
                        keep = not over_use_limit and entry.mean_reward >= static_thr

                    if keep:
                        entry.used = False
                        kept.append(entry)
                        kept_used_count += 1
                    else:
                        evicted_count += 1
                else:
                    kept.append(entry)

            self.buffers[task_id] = kept
            return kept_used_count, evicted_count, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_capacity(self, task_id: int):
        buf = self.buffers[task_id]
        if len(buf) <= self.max_size_per_task:
            return
        del buf[:len(buf) - self.max_size_per_task]

    def _evict_stale(self, task_id: int, current_version: int):
        if self.max_version_gap < 0:
            return
        self.buffers[task_id] = [
            e for e in self.buffers[task_id]
            if current_version - e.param_version <= self.max_version_gap
        ]

    # ------------------------------------------------------------------
    # Size / stats helpers
    # ------------------------------------------------------------------

    def task_size(self, task_id: int) -> int:
        with self._lock:
            return sum(len(e.data) for e in self.buffers[task_id])

    def total_size(self) -> int:
        with self._lock:
            return sum(
                sum(len(e.data) for e in self.buffers[tid])
                for tid in self.task_ids
            )

    def num_entries(self) -> int:
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
        with self._lock:
            return {
                tid: (sum(len(e.data) for e in self.buffers[tid]), len(self.buffers[tid]))
                for tid in self.task_ids
            }
