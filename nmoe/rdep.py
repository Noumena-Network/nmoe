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
        self._mode = {0: 'single', 1: 'ipc'}[mode_int]

        if self._mode == 'ipc':
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
                W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor,
                activation: str = "swiglu") -> torch.Tensor:
        if self.profile != 'bf16':
            raise RuntimeError("moe_bf16() requires profile='bf16'")
        return _MoEBf16Fused.apply(self, x, eid, gates, W1, W3, W2, activation)

    def moe_blockscaled(self, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
                        W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor,
                        W_cache, activation: str = "swiglu", forward_ablation: str = "off") -> torch.Tensor:
        if self.profile == 'bf16':
            raise RuntimeError("moe_blockscaled() requires profile in {'fp8','nvfp4'}")
        return _MoEBlockscaledFused.apply(self, x, eid, gates, W1, W3, W2, W_cache, activation, forward_ablation)
