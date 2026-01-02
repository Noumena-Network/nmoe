"""Runtime initialization and cleanup for nmoe training.

Handles platform checks, distributed setup, and seeds.
Seamlessly supports single GPU, single-node multi-GPU, and multi-node training.
"""
import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist


def _require_b200():
  """Hard-target NVIDIA B200 (sm_100a). Off-target is not supported."""
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device required (B200, sm_100a). Off-target is not supported.")
  major, minor = torch.cuda.get_device_capability()
  if (major, minor) != (10, 0):
    raise RuntimeError(
      f"This repo targets NVIDIA B200 (sm_100a). Detected compute capability {major}.{minor}. "
      "Off-target is not supported."
    )


def _ensure_third_party_imports() -> None:
  """Ensure vendored deps are importable (container-first contract)."""
  root = Path(__file__).resolve().parents[1]
  flash_attn = root / "third_party" / "flash_attn"
  if flash_attn.exists():
    p = str(flash_attn)
    if p not in sys.path:
      sys.path.insert(0, p)


def init(seed: int = 42) -> tuple[int, int]:
  """Initialize runtime for training. Returns (rank, world).

  Handles:
  - Platform check (B200 required)
  - Seeds and TF32
  - Device assignment (LOCAL_RANK env var)
  - Distributed init (automatic for multi-GPU)

  Works seamlessly for:
  - Single GPU: rank=0, world=1
  - Single-node multi-GPU: torchrun sets LOCAL_RANK, init NCCL
  - Multi-node: same as single-node, world > local_world
  """
  _require_b200()
  _ensure_third_party_imports()

  # Seeds and TF32
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  # Device assignment (torchrun sets LOCAL_RANK; single-process defaults to 0)
  local_rank = int(os.environ.get('LOCAL_RANK', '0'))
  torch.cuda.set_device(local_rank)

  # Distributed init (only when launched under torchrun)
  world_env = int(os.environ.get('WORLD_SIZE', '1'))
  if world_env > 1 and not dist.is_initialized():
    dist.init_process_group("nccl")

  # Get rank and world (or default to single GPU)
  rank = dist.get_rank() if dist.is_initialized() else 0
  world = dist.get_world_size() if dist.is_initialized() else 1

  return rank, world


def finalize():
  """Cleanup distributed state."""
  if dist.is_initialized():
    dist.destroy_process_group()
