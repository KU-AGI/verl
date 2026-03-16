"""
NIXL-based weight transfer utilities.

Provides NixlWeightAgent wrapper around the NIXL Python API for
high-performance RDMA point-to-point weight transfers between
trainer and rollout nodes, replacing Gloo collective streaming.
"""

import os
import time
import mmap
import numpy as np
import torch

try:
    from nixl._api import nixl_agent, nixl_agent_config
    NIXL_AVAILABLE = True
except ImportError:
    NIXL_AVAILABLE = False


def _open_shm_mmap(path: str, nbytes: int):
    """Open (or create) a file in /dev/shm and return (fd, mmap, numpy_u8_view)."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, nbytes)
    mm = mmap.mmap(fd, nbytes, access=mmap.ACCESS_WRITE)
    arr = np.ndarray((nbytes,), dtype=np.uint8, buffer=mm)
    return fd, mm, arr


class NixlWeightAgent:
    """
    Wraps a nixl_agent for weight transfer.

    On the sender (trainer) side:
      - register_source_buffer() registers the exported numpy weight array
      - send_to_receivers() posts WRITE transfers to all receiver agents

    On the receiver (rollout) side:
      - allocate_recv_buffer() pre-allocates an mmap buffer in /dev/shm
      - materialize_pt() converts the received raw bytes to a .pt file
    """

    def __init__(self, name: str, listen_port: int = 0):
        if not NIXL_AVAILABLE:
            raise ImportError("nixl is not installed. Install with: pip install nixl")

        config = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=True,
            listen_port=listen_port,
            backends=["UCX"],
        )
        self.agent = nixl_agent(name, config)
        self.name = name

        # Registered memory tracking
        self._src_reg = None  # registration handle for source buffer
        self._recv_reg = None  # registration handle for receive buffer
        self._recv_fd = None
        self._recv_mm = None
        self._recv_arr = None  # numpy uint8 view of recv buffer
        self._recv_nbytes = 0
        self._recv_shm_path = None

    def get_metadata(self) -> bytes:
        return self.agent.get_agent_metadata()

    def add_remote(self, metadata: bytes) -> str:
        return self.agent.add_remote_agent(metadata)

    # ---- Sender side ----

    def register_source(self, w_numpy: np.ndarray):
        """Register a numpy array (the exported weights) as DRAM source."""
        w_u8 = np.ascontiguousarray(w_numpy).view(np.uint8).reshape(-1)
        tensor = torch.from_numpy(w_u8)

        if self._src_reg is not None:
            try:
                self.agent.deregister_memory(self._src_reg)
            except Exception:
                pass

        self._src_reg = self.agent.register_memory(tensor, mem_type="cpu")
        self._src_tensor = tensor
        return self._src_reg

    def send_to_receiver(self, remote_name: str, remote_descs, notif_msg: bytes = b""):
        """
        Post a WRITE transfer to a remote receiver's pre-registered buffer.

        Args:
            remote_name: name of the remote nixl_agent
            remote_descs: deserialized descriptor list from the receiver
            notif_msg: notification message sent on completion

        Returns:
            transfer handle
        """
        local_descs = self.agent.get_xfer_descs(self._src_tensor, mem_type="cpu")
        handle = self.agent.initialize_xfer(
            "WRITE",
            local_descs,
            remote_descs,
            remote_name,
            notif_msg=notif_msg,
        )
        self.agent.transfer(handle)
        return handle

    def poll_xfer(self, handle, timeout_s: float = 120.0) -> bool:
        """Poll until transfer completes or times out."""
        t0 = time.time()
        while True:
            state = self.agent.check_xfer_state(handle)
            if state == "DONE":
                return True
            if time.time() - t0 > timeout_s:
                return False
            time.sleep(0.001)

    def release_handle(self, handle):
        self.agent.release_xfer_handle(handle)

    # ---- Receiver side ----

    def allocate_recv_buffer(self, nbytes: int, shm_path: str = "/dev/shm/.nixl_recv_buf"):
        """
        Pre-allocate an mmap buffer in /dev/shm and register it with NIXL.
        Returns serialized descriptor bytes for the sender to target.
        """
        # Cleanup previous buffer if exists
        self._cleanup_recv_buffer()

        self._recv_shm_path = shm_path
        self._recv_nbytes = nbytes
        self._recv_fd, self._recv_mm, self._recv_arr = _open_shm_mmap(shm_path, nbytes)

        recv_tensor = torch.from_numpy(self._recv_arr)
        self._recv_reg = self.agent.register_memory(recv_tensor, mem_type="cpu")

        # Create and serialize descriptors for the sender
        recv_descs = self.agent.get_xfer_descs(recv_tensor, mem_type="cpu")
        return self.agent.get_serialized_descs(recv_descs)

    def resize_recv_buffer(self, nbytes: int):
        """Resize the receive buffer if the weight size changed."""
        if nbytes != self._recv_nbytes:
            return self.allocate_recv_buffer(nbytes, self._recv_shm_path)
        # Return current descriptors
        recv_tensor = torch.from_numpy(self._recv_arr)
        recv_descs = self.agent.get_xfer_descs(recv_tensor, mem_type="cpu")
        return self.agent.get_serialized_descs(recv_descs)

    def wait_for_notif(self, sender_name: str, timeout_s: float = 120.0) -> bytes:
        """Wait for a notification from the sender indicating transfer is complete."""
        t0 = time.time()
        while True:
            notifs = self.agent.get_new_notifs()
            if sender_name in notifs and len(notifs[sender_name]) > 0:
                return notifs[sender_name][0]
            if time.time() - t0 > timeout_s:
                raise TimeoutError(
                    f"[NixlWeightAgent] Timeout waiting for notif from {sender_name}"
                )
            time.sleep(0.005)

    def materialize_pt(self, version: int) -> str:
        """
        Convert the raw bytes in the receive buffer to a .pt file.
        Returns the file path.
        """
        file_path = f"/dev/shm/weights_v{version}.pt"
        tmp_pt = f"/dev/shm/.nixl_wtmp_v{version}_{os.getpid()}.pt"

        weights_tensor = torch.from_numpy(
            self._recv_arr[: self._recv_nbytes].copy()
        ).view(torch.bfloat16)

        torch.save(weights_tensor, tmp_pt)
        os.replace(tmp_pt, file_path)
        return file_path

    def _cleanup_recv_buffer(self):
        if self._recv_reg is not None:
            try:
                self.agent.deregister_memory(self._recv_reg)
            except Exception:
                pass
            self._recv_reg = None
        if self._recv_mm is not None:
            try:
                self._recv_mm.close()
            except Exception:
                pass
            self._recv_mm = None
        if self._recv_fd is not None:
            try:
                os.close(self._recv_fd)
            except Exception:
                pass
            self._recv_fd = None
        if self._recv_shm_path and os.path.exists(self._recv_shm_path):
            try:
                os.remove(self._recv_shm_path)
            except Exception:
                pass

    def cleanup(self):
        """Cleanup all resources."""
        self._cleanup_recv_buffer()
        if self._src_reg is not None:
            try:
                self.agent.deregister_memory(self._src_reg)
            except Exception:
                pass
            self._src_reg = None
