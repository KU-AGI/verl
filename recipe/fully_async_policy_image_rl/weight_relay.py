import time
import os
import time
import asyncio
import ray
import torch
import numpy as np

@ray.remote(num_cpus=1)
class WeightRelayActor:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.latest_version = -1

    async def prefetch_to_shm(self, version: int, weights_ref: ray.ObjectRef):

        print(f"[DEBUG 3-1] Node {self.node_id} starting network download of v{version}...", flush=True)
        t0 = time.time()

        file_path = f"/dev/shm/weights_v{version}.pt"
        if os.path.exists(file_path):
            return file_path

        if isinstance(weights_ref, ray.ObjectRef):
            weights_numpy = await asyncio.to_thread(ray.get, weights_ref)
        else:
            weights_numpy = weights_ref
        
        tmp_path = file_path + ".tmp"   
        try:
            weights_tensor = torch.from_numpy(weights_numpy).view(torch.bfloat16)
            torch.save(weights_tensor, tmp_path)
            os.replace(tmp_path, file_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(f"Failed to torch.save weights: {e}")

        self.latest_version = version
        
        # 3. 이전 버전 정리
        old_path = f"/dev/shm/weights_v{version-1}.pt"
        if os.path.exists(old_path):
            try: os.remove(old_path)
            except: pass
            
        print(f"[DEBUG 3-2] Node {self.node_id} saved v{version} to SHM file in {time.time()-t0:.2f}s", flush=True)
        print(f"[Relay][Node {self.node_id}] v{version} prefetch done.")
        return file_path