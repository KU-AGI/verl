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

def _write_rollout_weight_metadata(metadata):
    if metadata is None:
        return None
    metadata_tmp = "/dev/shm/rollout_weight_metadata.pt.tmp"
    metadata_path = "/dev/shm/rollout_weight_metadata.pt"
    torch.save(metadata, metadata_tmp)
    os.replace(metadata_tmp, metadata_path)
    return metadata_path

@ray.remote(num_cpus=1)
class WeightRelayActor:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.latest_version = -1

        # NIXL state (initialized via configure_nixl)
        self._nixl_agent = None
        self._nixl_recv_desc_bytes = None  # serialized recv descriptors for sender
        self._nixl_sender_name = None  # name of the sender agent (for receivers)

    def configure_stream(self, rank: int, world_size: int, group_name: str, fanout: int, chunk_bytes: int):
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        self.fanout = fanout
        self.chunk_bytes = chunk_bytes
        return True

    async def write_rollout_weight_metadata(self, metadata_ref=None):
        if metadata_ref is None:
            return {"node": self.node_id, "metadata_file": None}
        if isinstance(metadata_ref, ray.ObjectRef):
            metadata = await metadata_ref
        else:
            metadata = metadata_ref
        metadata_path = _write_rollout_weight_metadata(metadata)
        return {"node": self.node_id, "metadata_file": metadata_path}

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

    # ==================== NIXL Methods ====================

    def configure_nixl(self, role: str, listen_port: int = 0, recv_buf_bytes: int = 0):
        """
        Initialize NIXL agent on this relay.

        Args:
            role: "sender" (trainer node) or "receiver" (rollout node)
            listen_port: port for NIXL metadata exchange (receiver should use non-zero)
            recv_buf_bytes: pre-allocate receive buffer of this size (receiver only)

        Returns:
            dict with "name" and "metadata" (bytes) for peer registration
        """
        from recipe.fully_async_policy_image_rl.nixl_utils import NixlWeightAgent

        agent_name = f"relay_{self.node_id[:8]}_{role}"
        self._nixl_agent = NixlWeightAgent(agent_name, listen_port=listen_port)

        result = {
            "name": agent_name,
            "metadata": self._nixl_agent.get_metadata(),
            "node_id": self.node_id,
        }

        if role == "receiver" and recv_buf_bytes > 0:
            shm_path = f"/dev/shm/.nixl_recv_{self.node_id[:8]}"
            self._nixl_recv_desc_bytes = self._nixl_agent.allocate_recv_buffer(
                recv_buf_bytes, shm_path
            )
            result["recv_descs"] = self._nixl_recv_desc_bytes

        print(f"[WeightRelayActor][NIXL] Node {self.node_id} configured as {role}, "
              f"agent={agent_name}", flush=True)
        return result

    def nixl_add_remote(self, metadata: bytes):
        """Register a remote NIXL agent."""
        return self._nixl_agent.add_remote(metadata)

    def nixl_get_recv_descs(self, nbytes: int = 0):
        """
        Get (or resize) receiver descriptor bytes. Called by sender to know
        where to WRITE.
        """
        if nbytes > 0 and nbytes != self._nixl_agent._recv_nbytes:
            shm_path = f"/dev/shm/.nixl_recv_{self.node_id[:8]}"
            self._nixl_recv_desc_bytes = self._nixl_agent.resize_recv_buffer(nbytes)
        return self._nixl_recv_desc_bytes

    async def nixl_send_to_peers(self, version: int, weights_ref, peer_infos: list):
        """
        Sender (trainer relay): send weights to all receiver relays via NIXL WRITE.

        Args:
            version: weight version
            weights_ref: Ray ObjectRef or numpy array of exported weights
            peer_infos: list of dicts with "name", "recv_descs" for each receiver

        Returns:
            dict with timing stats
        """
        t_total0 = time.time()

        # Resolve weights
        if isinstance(weights_ref, ray.ObjectRef):
            w0 = await weights_ref
        else:
            w0 = weights_ref
        if not isinstance(w0, np.ndarray):
            raise TypeError(f"exported weights must be numpy.ndarray, got {type(w0)}")

        w0 = np.ascontiguousarray(w0).reshape(-1)
        nbytes = int(w0.view(np.uint8).nbytes)

        # Register source buffer with NIXL
        self._nixl_agent.register_source(w0)

        t_net0 = time.time()

        # Post WRITE transfers to all peers in parallel
        handles = []
        for info in peer_infos:
            remote_name = info["name"]
            recv_descs_bytes = info["recv_descs"]
            remote_descs = self._nixl_agent.agent.deserialize_descs(recv_descs_bytes)

            notif_msg = f"v{version}:{nbytes}".encode()
            handle = self._nixl_agent.send_to_receiver(remote_name, remote_descs, notif_msg)
            handles.append((handle, remote_name))

        # Poll all transfers to completion
        for handle, remote_name in handles:
            success = self._nixl_agent.poll_xfer(handle)
            if not success:
                print(f"[WeightRelayActor][NIXL] TIMEOUT sending to {remote_name}", flush=True)
            self._nixl_agent.release_handle(handle)

        t_net = time.time() - t_net0

        # Save .pt file locally (trainer node also needs it)
        t_save0 = time.time()
        file_path = f"/dev/shm/weights_v{version}.pt"
        tmp_pt = f"/dev/shm/.nixl_wtmp_v{version}_sender.pt"
        weights_tensor = torch.from_numpy(w0).view(torch.bfloat16)
        torch.save(weights_tensor, tmp_pt)
        os.replace(tmp_pt, file_path)

        # Cleanup older versions
        self._cleanup_old_weights(file_path)

        t_save = time.time() - t_save0
        t_total = time.time() - t_total0
        self.latest_version = version

        print(f"[WeightRelayActor][NIXL] Sender v{version} done: "
              f"net={t_net:.2f}s save={t_save:.2f}s total={t_total:.2f}s "
              f"peers={len(peer_infos)}", flush=True)

        return {
            "node": self.node_id, "role": "sender", "file": file_path,
            "t_net": t_net, "t_save_pt": t_save, "t_total": t_total,
            "skip_write": False, "nbytes": nbytes,
        }

    async def nixl_recv_and_save(self, version: int, sender_name: str):
        """
        Receiver (rollout relay): wait for NIXL notification, then materialize .pt file.

        The actual data transfer is initiated by the sender via WRITE.
        This method waits for the completion notification and then saves the
        received raw bytes as a .pt file.

        Args:
            version: weight version
            sender_name: name of the sender's NIXL agent

        Returns:
            dict with timing stats
        """
        t_total0 = time.time()
        file_path = f"/dev/shm/weights_v{version}.pt"

        # Wait for sender's notification that WRITE is complete
        t_net0 = time.time()
        notif = self._nixl_agent.wait_for_notif(sender_name, timeout_s=120.0)
        t_net = time.time() - t_net0

        # Parse nbytes from notification
        # notif format: b"v{version}:{nbytes}"
        notif_str = notif.decode()
        nbytes = int(notif_str.split(":")[1])

        # Materialize .pt file from receive buffer
        t_save0 = time.time()
        self._nixl_agent._recv_nbytes = nbytes
        file_path = self._nixl_agent.materialize_pt(version)

        # Cleanup older versions
        self._cleanup_old_weights(file_path)

        t_save = time.time() - t_save0
        t_total = time.time() - t_total0
        self.latest_version = version

        print(f"[WeightRelayActor][NIXL] Receiver v{version} done: "
              f"wait={t_net:.2f}s save={t_save:.2f}s total={t_total:.2f}s",
              flush=True)

        return {
            "node": self.node_id, "role": "receiver", "file": file_path,
            "t_net": t_net, "t_save_pt": t_save, "t_total": t_total,
            "skip_write": False,
        }

    def _cleanup_old_weights(self, keep_path: str):
        """Remove older weight .pt files from /dev/shm, keeping only keep_path."""
        try:
            for f in glob.glob("/dev/shm/weights_v*.pt"):
                base = os.path.basename(f)
                if (base.startswith("weights_v")
                    and base.endswith(".pt")
                    and base.count(".") == 1
                    and os.path.abspath(f) != os.path.abspath(keep_path)):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
        except Exception:
            pass
