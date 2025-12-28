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

from verl.utils.device import get_nccl_backend

logger = logging.getLogger(__name__)


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

        # Statistics
        self.current_version = 0

        # Store self handle for passing to rollouter
        self.self_handle = None

        self._init_weights_info()
        self._init_sync_group()

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

    def sync_weights(self, version, validate=False, global_steps=0):
        """
        Async weight sync: queues weight update on rollouter without blocking.
        The actual NCCL broadcast happens later when rollouter detects idle state.
        """
        start_time = time.time()

        self.current_version = version
        print(f"[ParameterSynchronizer] Starting ASYNC weight synchronization (version {self.current_version})...")

        # Update MQ version immediately (samples generated with old weights will be tagged with old version)
        self.mq_client.update_param_version_sync(version)

        # Queue async weight update on rollouter (non-blocking)
        # Pass self handle as param_synchronizer so rollouter can call us back when ready
        # This only queues the version - actual NCCL broadcast happens in background task
        # when servers become idle
        ray.get(self.rollouter.queue_async_weight_update.remote(version, self.self_handle))

        end_time = time.time()
        print(f"[ParameterSynchronizer] Queued async weight update to version {version}. cost {end_time - start_time:.2f} seconds")

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
