# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import logging
import time

import ray
from ray.util.collective import collective
import asyncio
import numpy as np

from verl.utils.device import get_nccl_backend

logger = logging.getLogger(__name__)

def _resolve_method_by_suffix(worker, suffix: str, prefer_prefixes=()):
    """
    worker actor 내부 dir(self)에서 suffix로 끝나는 메서드명을 찾아 반환.
    prefer_prefixes 순서대로 우선 선택.
    """
    names = ray.get(worker.__ray_call__.remote(
        lambda self: [m for m in dir(self) if callable(getattr(self, m, None)) and m.endswith(suffix)]
    ))
    if not names:
        all_export = ray.get(worker.__ray_call__.remote(
            lambda self: [m for m in dir(self) if "export" in m]
        ))
        raise RuntimeError(f"No method endswith '{suffix}'. export-like={all_export[:50]}")

    for p in prefer_prefixes:
        cand = f"{p}_{suffix}"
        if cand in names:
            return cand
    return names[0]

@ray.remote
class ParameterSynchronizer:
    """
    Unified parameter synchronizer, responsible for synchronizing model parameters between actor and rollout
    Based on the mature synchronization mode implementation of one_step_off_policy
    Merges the functions of the original multiple synchronizer classes
    """

    def __init__(self, config, trainer, rollouter, mq):
        self.config = config
        self.trainer = trainer
        self.rollouter = rollouter
        self.mq_client = mq
        self.actor_wg = ray.get(trainer.get_actor_wg.remote())
        self.rollout_wg = ray.get(rollouter.get_rollout_wg.remote())

        # Basic attributes
        self.weights_info = None
        self.sync_group_initialized = False
        self.sync_group_name = "actor_rollout"
        self.wait_last_update = None
        self.wait_last_resume = None

        # Weight Version
        self.latest_version = 0
        self.latest_weights_ref = None
        self.keep_last_n = getattr(config.async_training, "keep_last_n_weights", 1)
        self._weights_refs = {}

        self.relays = self._setup_node_relays()

        # Statistics
        self.current_version = 0

        # Store self handle for passing to rollouter
        self.self_handle = None

        self._export_method_name = None
        self._rank_method_name = None
        self._actor_rank0_idx = self._setup_rank_info()

        self._init_weights_info()
        self._init_sync_group()

    def _setup_rank_info(self):
    
        rank_method = _resolve_method_by_suffix(
            self.actor_wg.workers[0], "get_dist_rank",
            prefer_prefixes=("actor", "rollout")
        )
        ranks = ray.get([getattr(w, rank_method).remote() for w in self.actor_wg.workers])
        return ranks.index(0)

    def _setup_node_relays(self):
        from recipe.fully_async_policy_image_rl.weight_relay import WeightRelayActor
        
        method_name = _resolve_method_by_suffix(
            self.rollout_wg.workers[0], 
            "get_node_id", 
            prefer_prefixes=("rollout", "actor", "actor_rollout")
        )
        print(f"[ParamSync] Resolved node_id method name: {method_name}")

        node_ids = ray.get([
            getattr(w, method_name).remote() for w in self.rollout_wg.workers
        ])
        
        unique_nodes = list(set(node_ids))
        
        relays = {}
        for node_id in unique_nodes:
            relays[node_id] = WeightRelayActor.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                )
            ).remote(node_id)
        
        print(f"[ParamSync] Initialized {len(relays)} RelayActors on nodes: {unique_nodes}")
        return relays

    def get_latest_published(self):
        return self.latest_version, self.latest_weights_ref

    def get_current_param_version(self) -> int:
        """Get current parameter version number"""
        return self.current_version

    def get_weights_info(self):
        """Get weights info"""
        return self.weights_info

    def _init_weights_info(self):
        self.weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(self.weights_info)

    def _init_sync_group(self):
        print("[ParameterSynchronizer] Initializing parameter synchronization group...")
        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        collective.create_collective_group(
            actor_rollout_workers,
            len(actor_rollout_workers),
            list(range(0, len(actor_rollout_workers))),
            backend=get_nccl_backend(),
            group_name=self.sync_group_name,
        )

    def set_self_handle(self, handle):
        """Store the actor handle for this ParameterSynchronizer"""
        self.self_handle = handle

        print(f"[ParameterSynchronizer] Linking self handle to Rollouter...")
        self.rollouter.set_param_synchronizer.remote(handle)

    async def _publish_weights(self, version: int):
    
        if self._export_method_name is None:
            self._export_method_name = _resolve_method_by_suffix(
                self.actor_wg.workers[0], "export_rollout_weights",
                prefer_prefixes=("actor", "rollout")
            )

        export_refs = [getattr(w, self._export_method_name).remote() for w in self.actor_wg.workers]
        
        weights_ref = export_refs[self._actor_rank0_idx]
        self.latest_weights_ref = weights_ref
        self.latest_version = version
        return weights_ref

    async def sync_weights(self, version, validate=False, global_steps=0):

        start_time = time.time()
        self.current_version = version
        
        # 1. 메시지 큐 버전 즉시 업데이트
        self.mq_client.update_param_version_sync(version)

        # 2. 트레이너 가중치(통짜 1D 텐서) 가져오기
        weights_ref = await self._publish_weights(version)

        # 노드 수만큼만 네트워크 전송이 발생합니다.
        prefetch_tasks = [r.prefetch_to_shm.remote(version, weights_ref) for r in self.relays.values()]
        
        print(f"[DEBUG 2-1] Synchronizer triggering prefetch for v{version} to {len(self.relays)} nodes", flush=True)
        t0 = time.time()
        await asyncio.gather(*prefetch_tasks)
        print(f"[DEBUG 2-2] All nodes finished SHM prefetch for v{version} in {time.time()-t0:.2f}s", flush=True)
        
        print(f"[ParamSync] v{version} Prefetch to all nodes SHM done. Time: {time.time()-start_time:.2f}s")

        # Update rollout version metadata & trigger validation asynchronously
        self.wait_last_update = self.rollouter.update_param_version.remote(version, validate, global_steps)
        # No need to pause/resume - generation continues with old weights until async update completes
        self.wait_last_resume = None

    def apply_nccl_weight_sync(self, version: int):
        """
        Perform NCCL weight synchronization between actor and rollout workers.
        Called by FullyAsyncRollouter when all servers are idle and ready for sync.

        This method has access to both actor_wg and rollout_wg, which is required for
        NCCL collective operations where all workers in the "actor_rollout" group must participate.
        """
        print(f"[ParameterSynchronizer] Performing NCCL weight sync for version {version}...")

        try:
            # Trigger NCCL weight sync on BOTH actor and rollout workers simultaneously
            # This is critical - NCCL collective requires all workers in "actor_rollout" group
            # Actor workers (rank 0) will prepare and broadcast weights
            # Rollout workers will receive and load weights
            actor_sync_refs = self.actor_wg.sync_rollout_weights()
            rollout_sync_refs = self.rollout_wg.sync_rollout_weights()

            # Wait for all workers to complete sync
            import ray
            ray.get(actor_sync_refs + rollout_sync_refs)

            print(f"[ParameterSynchronizer] NCCL weight sync completed for version {version}")

        except Exception as e:
            print(f"[ParameterSynchronizer] NCCL weight sync failed for version {version}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def wait_last_valid(self):
        print("[ParameterSynchronizer] Waiting last sync and validate...")
        start_time = time.time()
        if self.wait_last_update:
            ray.get(self.wait_last_update)
        if self.wait_last_resume:
            ray.get(self.wait_last_resume)
        print(f"[ParameterSynchronizer] Wait last validate cost: {time.time() - start_time:.2f} seconds")

    def rollouter_save_checkpoint(self, local_global_step_folder: str):
        """Trigger rollout to save checkpoint(dataloader)"""
        print(f"[ParameterSynchronizer] Triggering checkpoint save at {local_global_step_folder} ...")
        return ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
