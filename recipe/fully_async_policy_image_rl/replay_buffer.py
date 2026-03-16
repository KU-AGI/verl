import random
from collections import deque

from verl.protocol import DataProto


class ReplayBuffer:
    """Per-task replay buffer with score-based filtering.

    Stores samples separately per task_id. On push, only samples whose
    task score >= threshold are kept. On sample, draws equally from all
    task buffers and merges into a single DataProto.
    """

    def __init__(
        self,
        max_version_gap: int,
        task_ids: list[int],
        score_thresholds: dict[int, float],
        max_size_per_task: int = -1,
    ):
        self.max_version_gap = max_version_gap
        self.task_ids = task_ids
        self.score_thresholds = score_thresholds  # {task_id: min_score}
        self.max_size_per_task = max_size_per_task
        # per-task buffers: task_id -> deque of (DataProto, param_version)
        # When max_size_per_task > 0, oldest entries are automatically dropped (FIFO).
        maxlen = max_size_per_task if max_size_per_task > 0 else None
        self.buffers: dict[int, deque] = {
            tid: deque(maxlen=maxlen) for tid in task_ids
        }

    def push(self, batch: DataProto, task_id: int):
        """Store score-filtered samples for a specific task (uid-group-aware).

        Filters by per-uid mean score >= threshold, keeping entire uid groups intact.
        Uses rollout param_version (not trainer version) for staleness tracking.
        """
        score_key = f"task{task_id}_token_level_scores"
        if score_key not in batch.batch:
            return 0

        scores = batch.batch[score_key].sum(-1)  # [batch_size]
        uids = batch.non_tensor_batch["uid"]
        param_versions = batch.non_tensor_batch["param_version"]
        threshold = self.score_thresholds.get(task_id, 0.0)

        # Compute per-uid mean score (only valid scores, ignoring -100 markers)
        uid_scores: dict[str, list[float]] = {}
        for i, uid in enumerate(uids):
            uid_scores.setdefault(uid, []).append(scores[i].item())

        good_uids = set()
        for uid, s_list in uid_scores.items():
            valid_scores = [s for s in s_list if s >= 0]
            if valid_scores and sum(valid_scores) / len(valid_scores) >= threshold:
                good_uids.add(uid)

        if not good_uids:
            # Debug: show why no uids passed
            uid_means = {}
            for uid, s_list in uid_scores.items():
                valid = [s for s in s_list if s >= 0]
                uid_means[uid] = sum(valid) / len(valid) if valid else -1.0
            print(f"[ReplayBuffer] task{task_id}: no uids passed threshold={threshold}, "
                  f"uid_means={uid_means}")
            return 0

        # Group by uid, use min version as representative (keep uid groups intact)
        uid_idxs: dict[str, list[int]] = {}
        uid_version: dict[str, int] = {}
        for i, uid in enumerate(uids):
            if uid in good_uids:
                uid_idxs.setdefault(uid, []).append(i)
                v = int(param_versions[i])
                uid_version[uid] = min(uid_version.get(uid, v), v)

        # Group uids by their representative version, store as complete uid groups
        version_groups: dict[int, list[int]] = {}
        for uid, idxs in uid_idxs.items():
            v = uid_version[uid]
            version_groups.setdefault(v, []).extend(idxs)

        total_kept = 0
        for v, idxs in version_groups.items():
            filtered = batch.select_idxs(idxs)
            self.buffers[task_id].append((filtered, v))
            total_kept += len(idxs)
        return total_kept

    def sample_task(self, current_version: int, task_id: int, n_samples: int,
                    exclude_uids: set | None = None) -> DataProto | None:
        """Sample n_samples rows from a specific task's buffer (uid-group-aware).

        Evicts stale entries first, then samples uid groups.
        exclude_uids: uids already in fresh batch, to prevent self-duplication.
        """
        evicted = self.evict_stale(current_version)
        if evicted > 0:
            per_task = self.size_per_task()
            print(f"[ReplayBuffer] evict_stale: evicted {evicted} samples (current_version={current_version}), remaining={per_task}")

        buf = list(self.buffers[task_id])
        if not buf:
            return None

        # Tag each entry's samples with its param version before concat
        for b, v in buf:
            b.non_tensor_batch["sample_param_version"] = b.non_tensor_batch["param_version"]
        all_batches = [b for b, _v in buf]
        combined = DataProto.concat(all_batches)
        if len(combined) == 0:
            return None

        # Sample uid groups, excluding fresh batch uids
        uids = combined.non_tensor_batch["uid"]
        uid_to_idxs: dict[str, list[int]] = {}
        for i, uid in enumerate(uids):
            if exclude_uids and uid in exclude_uids:
                continue
            uid_to_idxs.setdefault(uid, []).append(i)

        all_uid_keys = list(uid_to_idxs.keys())
        random.shuffle(all_uid_keys)
        selected_idxs = []
        for uid in all_uid_keys:
            if len(selected_idxs) >= n_samples:
                break
            selected_idxs.extend(uid_to_idxs[uid])

        if not selected_idxs:
            return None

        return combined.select_idxs(selected_idxs)

    def evict_stale(self, current_version: int):
        """Remove all entries that exceed max_version_gap from every task buffer.

        If max_version_gap == -1, no entries are evicted (unlimited retention).
        """
        if self.max_version_gap < 0:
            return 0
        total_evicted = 0
        maxlen = self.max_size_per_task if self.max_size_per_task > 0 else None
        for tid in self.task_ids:
            before = len(self.buffers[tid])
            self.buffers[tid] = deque(
                ((b, v) for b, v in self.buffers[tid]
                 if current_version - v <= self.max_version_gap),
                maxlen=maxlen,
            )
            total_evicted += before - len(self.buffers[tid])
        return total_evicted

    def task_eligible(self, current_version: int, task_id: int) -> int:
        """Count eligible rows for a specific task buffer.

        If max_version_gap == -1, all entries are eligible.
        """
        if self.max_version_gap < 0:
            return sum(len(b) for b, _ in self.buffers[task_id])
        return sum(
            len(b) for b, v in self.buffers[task_id]
            if current_version - v <= self.max_version_gap
        )

    def total_eligible(self, current_version: int) -> int:
        """Count total rows that would pass the version gap filter (without evicting)."""
        return sum(
            self.task_eligible(current_version, tid) for tid in self.task_ids
        )

    def total_size(self) -> int:
        return sum(
            sum(len(b) for b, _ in buf)
            for buf in self.buffers.values()
        )

    def num_entries(self) -> int:
        """Total number of stored batch entries (not rows)."""
        return sum(len(buf) for buf in self.buffers.values())

    def size_per_task(self) -> dict[int, int]:
        return {
            tid: sum(len(b) for b, _ in buf)
            for tid, buf in self.buffers.items()
        }
