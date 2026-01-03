import time
import os
import time
import asyncio
import ray
import torch
import numpy as np
import glob

@ray.remote(num_cpus=1)
class WeightRelayActor:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.latest_version = -1

    async def prefetch_to_shm(self, version: int, weights_ref: ray.ObjectRef):

        print(f"[WeightRelayActor] Node {self.node_id} starting network download of v{version}...", flush=True)
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
        
        try:
            # /dev/shm에 있는 모든 가중치 파일 목록을 가져옵니다.
            all_weight_files = glob.glob("/dev/shm/weights_v*.pt")
            for f in all_weight_files:
                # 방금 저장한 파일이 아니라면 삭제
                if os.path.abspath(f) != os.path.abspath(file_path):
                    try:
                        os.remove(f)
                        print(f"[Cleanup] Deleted old version: {f}")
                    except:
                        pass
        except Exception as e:
            print(f"[Cleanup] Error during glob/remove: {e}")
            
        print(f"[WeightRelayActor][Node {self.node_id}] v{version} prefetch done. (Only v{version} kept)")
        return file_path