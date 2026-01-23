import time
import os
import time
import asyncio
import ray
import torch
import numpy as np
import glob
import mmap
from ray.util.collective import collective

def _tree_parent_children(r: int, ws: int, fo: int):
    if fo <= 0:
        raise ValueError(f"fanout must be > 0, got {fo}")
    parent_ = (r - 1) // fo if r > 0 else -1
    children_ = []
    base = r * fo + 1
    for i in range(fo):
        c = base + i
        if c < ws:
            children_.append(c)
    return parent_, children_

def _open_shm_u8(path: str, nbytes: int):
    fd_ = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
    os.ftruncate(fd_, nbytes)
    mm_ = mmap.mmap(fd_, nbytes, access=mmap.ACCESS_WRITE)
    arr_ = np.ndarray((nbytes,), dtype=np.uint8, buffer=mm_)
    ten_ = torch.from_numpy(arr_)  # uint8 tensor backed by /dev/shm mmap
    return fd_, mm_, ten_

@ray.remote(num_cpus=1)
class WeightRelayActor:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.latest_version = -1

    def configure_stream(self, rank: int, world_size: int, group_name: str, fanout: int, chunk_bytes: int):
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        self.fanout = fanout
        self.chunk_bytes = chunk_bytes
        return True

    async def prefetch_to_shm(self, version: int, weights_ref: ray.ObjectRef):
        raise RuntimeError("prefetch_to_shm CALLED (should not happen)")

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

    async def stream_to_shm_gloo(self, version: int, weights_ref=None):
        """
        Deadlock-safe Gloo tree streaming into /dev/shm.

        - No early return on "file exists". Even if already cached, the relay must
        participate in ALL send/recv to avoid collective mismatch deadlocks.
        - If the file already exists on a non-root node, we still recv/forward but skip disk writes.
        """

        print("STREAM GLOO START")
        t_total0 = time.time()

        group = self.group_name
        rank = int(self.rank)
        world = int(self.world_size)
        fanout = int(self.fanout)
        chunk = int(self.chunk_bytes)

        file_path = f"/dev/shm/weights_v{version}.pt"

        # If this node already has the file, we MUST still participate in recv/send,
        # but can skip writing/saving.
        skip_write = (rank != 0 and os.path.exists(file_path))

        parent, children = _tree_parent_children(rank, world, fanout)

        # -------- 1) Header propagation (nbytes, chunk_bytes) --------
        header = torch.empty((2,), dtype=torch.int64)

        if rank == 0:
            if weights_ref is None:
                raise ValueError("rank0 requires weights_ref, got None")

            if isinstance(weights_ref, ray.ObjectRef):
                w0 = await weights_ref
            else:
                w0 = weights_ref
            if not isinstance(w0, np.ndarray):
                raise TypeError(f"exported weights must be numpy.ndarray, got {type(w0)}")

            w0 = np.ascontiguousarray(w0).reshape(-1)
            w_u8 = (w0.view(np.uint8) if w0.dtype != np.uint8 else w0).reshape(-1)
            nbytes = int(w_u8.nbytes)

            header[0] = nbytes
            header[1] = chunk

            for c in children:
                collective.send(header, dst_rank=c, group_name=group)
        else:
            collective.recv(header, src_rank=parent, group_name=group)
            nbytes = int(header[0].item())
            chunk = int(header[1].item())

            for c in children:
                collective.send(header, dst_rank=c, group_name=group)

        if nbytes <= 0:
            raise ValueError(f"Invalid nbytes={nbytes}")
        if chunk <= 0:
            raise ValueError(f"Invalid chunk_bytes={chunk}")

        # -------- 2) Prepare SHM buffers on non-root (only if we will write) --------
        pid = os.getpid()
        tmp_pt  = f"/dev/shm/.wtmp_v{version}_{pid}.pt"
        tmp_bin = f"/dev/shm/.wtmp_v{version}_{pid}.bin"

        fd = mm = dst_u8 = None
        if rank != 0 and not skip_write:
            fd, mm, dst_u8 = _open_shm_u8(tmp_bin, nbytes)

        buf = torch.empty((chunk,), dtype=torch.uint8)

        t_net0 = time.time()
        # -------- 3) Body streaming + forward --------
        if rank == 0:
            src_u8 = torch.from_numpy(w_u8)
            for off in range(0, nbytes, chunk):
                n = min(chunk, nbytes - off)

                buf[:n].copy_(src_u8[off:off + n])
                if n < chunk:
                    buf[n:].zero_()

                for c in children:
                    collective.send(buf, dst_rank=c, group_name=group)
                    
            t_net = time.time() - t_net0
            t_save0 = time.time()

            # 1. Rank 0의 메모리에 있는 w0를 활용해 .pt 파일 저장
            # w0는 위에서 이미 numpy array로 준비되어 있습니다.
            weights_tensor = torch.from_numpy(w0).view(torch.bfloat16)
            
            # 2. 원자적(Atomic) 저장을 위해 임시 파일 사용
            tmp_pt = f"/dev/shm/.wtmp_v{version}_rank0.pt"
            torch.save(weights_tensor, tmp_pt)
            os.replace(tmp_pt, file_path)

            # 3. 이전 버전 파일 정리 (Rank 0 노드 관리용)
            try:
                for f in glob.glob("/dev/shm/weights_v*.pt"):
                    base = os.path.basename(f)
                    if (base.startswith("weights_v")
                        and base.endswith(".pt")
                        and base.count(".") == 1 
                        and os.path.abspath(f) != os.path.abspath(file_path)):
                        try:
                            os.remove(f)
                        except:
                            pass
            except:
                pass

            t_save = time.time() - t_save0
            t_total = time.time() - t_total0
            self.latest_version = version

            return {
                "node": self.node_id, 
                "rank": rank, 
                "file": file_path,  # 기존 None에서 file_path로 변경
                "t_net": t_net, 
                "t_save_pt": t_save, 
                "t_total": t_total,
                "skip_write": False,
            }

        # Non-root: always recv; optionally write; always forward
        try:
            for off in range(0, nbytes, chunk):
                n = min(chunk, nbytes - off)

                collective.recv(buf, src_rank=parent, group_name=group)

                if not skip_write:
                    dst_u8[off:off + n].copy_(buf[:n])

                for c in children:
                    collective.send(buf, dst_rank=c, group_name=group)
        finally:
            if mm is not None:
                mm.flush()
                mm.close()
            if fd is not None:
                os.close(fd)

        t_net = time.time() - t_net0

        # If we already had the file, do not overwrite; we only participated to avoid deadlock.
        if skip_write:
            self.latest_version = version
            t_total = time.time() - t_total0
            return {
            "node": self.node_id, "rank": rank, "file": file_path,
            "t_net": t_net, "t_save_pt": 0.0, "t_total": t_total,
            "skip_write": True,
            }

        t_save0 = time.time()

        # -------- 4) Materialize .pt for existing rollouter path --------
        mm = np.memmap(tmp_bin, dtype=np.uint8, mode="r", shape=(nbytes,))
        weights_tensor = torch.from_numpy(mm).view(torch.bfloat16).clone()
        del mm

        torch.save(weights_tensor, tmp_pt)
        os.replace(tmp_pt, file_path)

        # Remove tmp_bin to avoid /dev/shm accumulation
        try:
            os.remove(tmp_bin)
        except:
            pass

        # Cleanup older versions (.pt only; keep behavior)
        try:
            for f in glob.glob("/dev/shm/weights_v*.pt"):
                # weights_v{number}.pt 만 지우기 (tmp, suffix 붙은 것 제외)
                base = os.path.basename(f)
                if (base.startswith("weights_v")
                    and base.endswith(".pt")
                    and base.count(".") == 1   # weights_v15.pt 처럼 점이 1개인 것만
                    and os.path.abspath(f) != os.path.abspath(file_path)):
                    try:
                        os.remove(f)
                    except:
                        pass
        except:
            pass
        print("EXPORT ROLLOUT WEIGHT Done")
        t_save = time.time() - t_save0
        t_total = time.time() - t_total0
        self.latest_version = version

        return {
            "node": self.node_id, "rank": rank, "file": file_path,
            "t_net": t_net, "t_save_pt": t_save, "t_total": t_total,
            "skip_write": False,
        }