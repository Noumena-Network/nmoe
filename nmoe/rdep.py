import os

import torch
import torch.distributed as dist
import numpy as np

# Import C extension (built in csrc/)
from .csrc import rdep as _C
from .moe import _MoEBf16Fused, _MoEBlockscaledFused


def _get_local_world_size() -> int:
    """Get local world size (GPUs on this node)."""
    return int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))

_CPU_PG = None
_CPU_PG_WORLD: int | None = None


def _cpu_pg():
    """CPU-only process group for bootstrap collectives (Gloo)."""
    global _CPU_PG, _CPU_PG_WORLD
    if not dist.is_initialized():
        return None
    world = int(dist.get_world_size())
    if world <= 1:
        return None
    if _CPU_PG is None or _CPU_PG_WORLD != world:
        _CPU_PG = dist.new_group(backend="gloo")
        _CPU_PG_WORLD = world
    return _CPU_PG


class Rdep:
    #TODO(xjdr): this is an odd global here
    PROFILES = {'bf16': -1, 'fp8': 0, 'nvfp4': 1}

    def __init__(self, dim: int, n_local: int, topk: int, profile: str = 'nvfp4',
                 capacity: int = 65536):
        assert profile in self.PROFILES, f"profile must be one of {list(self.PROFILES.keys())}"

        self.dim = dim
        self.n_local = n_local
        self.topk = topk
        self.profile = profile
        self.capacity = capacity
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_world = _get_local_world_size()
        _C.init(self.rank, self.world, self.local_world)
        mode_int = _C.get_mode()
        self._mode = {0: 'single', 1: 'ipc', 2: 'hybrid'}[mode_int]
        if self._mode == 'hybrid' and not _C.has_nvshmem():
            raise RuntimeError(
                f"Multi-node configuration (world={self.world} > local_world={self.local_world}) "
                "requires NVSHMEM support. Rebuild rdep with NVSHMEM or use single-node."
            )

        if self._mode == 'hybrid':
            self._setup_hybrid()
        elif self._mode == 'ipc':
            _C.alloc_bf16(capacity, dim, n_local)
            if profile != 'bf16':
                _C.alloc_blockscaled(capacity, dim, n_local, self.PROFILES[profile])
            self._setup_ipc()
        elif self._mode == 'single':
            _C.alloc_bf16(capacity, dim, n_local)
            _C.sync_buffer_ptrs_bf16()
            if profile != 'bf16':
                _C.alloc_blockscaled(capacity, dim, n_local, self.PROFILES[profile])
                _C.sync_buffer_ptrs_blockscaled()

    def _setup_hybrid(self):
        cpu_pg = _cpu_pg()
        if cpu_pg is None:
            raise RuntimeError("[RDEP] internal error: expected dist to be initialized for hybrid bootstrap")

        uid_size = _C.nvshmem_get_uid_size()
        if self.rank == 0:
            print(f"[RDEP] rank={self.rank}: Getting UID (size={uid_size})...", flush=True)
            uid = _C.nvshmem_get_uid()
            print(f"[RDEP] rank={self.rank}: Got UID", flush=True)
        else:
            uid = None

        if self.rank == 0:
            print(f"[RDEP] rank={self.rank}: Broadcasting UID via CPU...", flush=True)
        uid_list = [uid]
        dist.broadcast_object_list(uid_list, src=0, group=cpu_pg)
        uid = uid_list[0]
        if self.rank == 0:
            print(f"[RDEP] rank={self.rank}: UID broadcast complete", flush=True)
        if self.rank == 0:
            print(f"[RDEP] rank={self.rank}: Initializing NVSHMEM...", flush=True)
        _C.nvshmem_init(uid, self.rank, self.world, self.local_world)
        if self.rank == 0:
            print(f"[RDEP] rank={self.rank}: NVSHMEM initialized!", flush=True)

        _C.nvshmem_alloc_bf16(self.capacity, self.dim, self.n_local)
        node_id = self.rank // self.local_world
        local_handle_bf16 = _C.nvshmem_get_ipc_handle_bf16()
        all_handles_bf16 = [None] * self.world
        dist.all_gather_object(all_handles_bf16, local_handle_bf16, group=cpu_pg)
        local_handles_bf16 = []
        for r in range(self.world):
            if r // self.local_world == node_id:
                local_handles_bf16.append(all_handles_bf16[r])
        local_handles_bf16_np = np.concatenate(local_handles_bf16)
        _C.nvshmem_open_ipc_handles_bf16(local_handles_bf16_np, self.local_world)
        _C.nvshmem_sync_ipc_buffer_ptrs_bf16()
        dist.barrier(group=cpu_pg)

    def _setup_ipc(self):
        """Exchange IPC handles via NCCL all_gather (one-time at init)."""
        local_handle_bf16 = _C.get_ipc_handle_bf16()
        handle_tensor_bf16 = torch.from_numpy(local_handle_bf16).cuda()

        all_handles_bf16 = [torch.zeros_like(handle_tensor_bf16) for _ in range(self.world)]
        dist.all_gather(all_handles_bf16, handle_tensor_bf16)

        all_handles_bf16_np = np.concatenate([h.cpu().numpy() for h in all_handles_bf16])
        _C.open_ipc_handles_bf16(all_handles_bf16_np, self.world)
        _C.sync_buffer_ptrs_bf16()

        if self.profile != 'bf16':
            local_handle_block = _C.get_ipc_handle_blockscaled()
            handle_tensor_block = torch.from_numpy(local_handle_block).cuda()
            all_handles_block = [torch.zeros_like(handle_tensor_block) for _ in range(self.world)]
            dist.all_gather(all_handles_block, handle_tensor_block)
            all_handles_block_np = np.concatenate([h.cpu().numpy() for h in all_handles_block])
            _C.open_ipc_handles_blockscaled(all_handles_block_np, self.world)
            _C.sync_buffer_ptrs_blockscaled()

    def moe_bf16(self, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
                W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
        if self.profile != 'bf16':
            raise RuntimeError("moe_bf16() requires profile='bf16'")
        return _MoEBf16Fused.apply(self, x, eid, gates, W1, W3, W2)

    def moe_blockscaled(self, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
                        W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor,
                        W_cache) -> torch.Tensor:
        if self.profile == 'bf16':
            raise RuntimeError("moe_blockscaled() requires profile in {'fp8','nvfp4'}")
        return _MoEBlockscaledFused.apply(self, x, eid, gates, W1, W3, W2, W_cache)
