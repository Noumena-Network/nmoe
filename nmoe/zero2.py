"""ZeRO-2 for replicated (dense) parameters.

Elegant minimal implementation:
- step_dense_adamw: RS(AVG) grads → AdamW shard update → AG params

State lives in caller-provided dict to keep this module stateless.
Works seamlessly for single GPU (no-op) and multi-GPU (ZeRO-2).
"""
import os
from contextlib import nullcontext
import math
import torch
import torch.distributed as dist


def _ceil_div(a: int, b: int) -> int:
  return (a + b - 1) // b


def _rs_chunk_elems(*, dtype: torch.dtype, world: int, shard_size: int) -> tuple[int, int]:
  """Return (shard_chunk, total_chunk_elems) for chunked reduce-scatter buffers.

  total_chunk_elems == world * shard_chunk.
  """
  if world <= 0:
    raise RuntimeError(f"ZeRO-2: invalid world_size={world}")
  if shard_size <= 0:
    raise RuntimeError(f"ZeRO-2: invalid shard_size={shard_size}")

  # Default: 2 GiB per dtype/group. Net win vs allocating padded_total for flat_grad.
  chunk_mb = int(os.getenv("NMOE_ZERO2_RS_CHUNK_MB", "2048"))
  if chunk_mb <= 0:
    raise RuntimeError(f"NMOE_ZERO2_RS_CHUNK_MB must be > 0 (got {chunk_mb})")
  bytes_per_elem = torch.empty((), dtype=dtype).element_size()
  target_elems_total = (chunk_mb * 1024 * 1024) // max(1, bytes_per_elem)
  # Must be divisible by world, and at least 1 elem per rank.
  target_shard_chunk = max(1, int(target_elems_total) // int(world))
  # Prefer an alignment that is friendly to both memcpy and NCCL (elements, not bytes).
  align = 256
  if shard_size <= align:
    shard_chunk = shard_size
  else:
    shard_chunk = max(align, (target_shard_chunk // align) * align)
    shard_chunk = min(shard_chunk, shard_size)
  total_chunk_elems = int(shard_chunk) * int(world)
  return int(shard_chunk), int(total_chunk_elems)


def _get_or_init_flat_group(
  group: dict,
  *,
  params: list[torch.nn.Parameter],
  rank: int,
  world: int,
  dtype: torch.dtype,
) -> dict:
  """Return (and lazily initialize) a persistent flat buffer plan for a param group/dtype.

  Stored on the *group dict* (not in optimizer state) to avoid checkpointing large buffers.
  """
  cache = group.setdefault('_zero2_flat', {})
  key = (dtype, int(world))
  if key in cache:
    flat = cache[key]
    # Re-init if param list changed (should not happen in training).
    if flat.get('n_params') == len(params):
      return flat

  if not params:
    raise RuntimeError("ZeRO-2: empty param list")

  dev = params[0].device
  for p in params:
    if p.device != dev:
      raise RuntimeError("ZeRO-2: params must be on a single device per group")
    if p.dtype != dtype:
      raise RuntimeError("ZeRO-2: dtype grouping invariant violated")

  offsets: list[tuple[int, int]] = []
  total = 0
  for p in params:
    n = int(p.numel())
    offsets.append((total, total + n))
    total += n

  shard_size = _ceil_div(total, world)
  padded_total = shard_size * world

  flat_param = torch.empty(padded_total, dtype=dtype, device=dev)
  # Initialize from current param values.
  for p, (a, b) in zip(params, offsets):
    flat_param[a:b].copy_(p.detach().view(-1))
  if total < padded_total:
    flat_param[total:padded_total].zero_()

  # Re-point params to views into flat_param (no per-step unpack copies).
  for p, (a, b) in zip(params, offsets):
    p.data = flat_param[a:b].view_as(p)  # type: ignore[assignment]

  # Scratch (persistent):
  # - Chunked reduce-scatter buffers (avoid allocating flat_grad of size padded_total).
  # - Local param shard (the only shard we update).
  shard_chunk, total_chunk_elems = _rs_chunk_elems(dtype=dtype, world=world, shard_size=shard_size)
  rs_in = torch.empty(total_chunk_elems, dtype=dtype, device=dev)
  rs_out = torch.empty(shard_chunk, dtype=dtype, device=dev)
  if world == 1:
    param_shard = flat_param[:shard_size]  # view (no comm)
  else:
    param_shard = torch.empty(shard_size, dtype=dtype, device=dev)
    a = rank * shard_size
    b = a + shard_size
    param_shard.copy_(flat_param[a:b])

  flat = {
    'n_params': len(params),
    'offsets': offsets,
    'total': total,
    'padded_total': padded_total,
    'shard_size': shard_size,
    'flat_param': flat_param,
    'param_shard': param_shard,
    'rs_in': rs_in,
    'rs_out': rs_out,
    'shard_chunk': shard_chunk,
  }
  cache[key] = flat
  return flat


@torch.no_grad()
def step_dense_adamw(
  param_groups: list[dict],
  *,
  state: dict,
  pg: dist.ProcessGroup | None = None,
  betas: tuple[float, float] = (0.9, 0.95),
  eps: float = 1e-8,
) -> None:
  """ZeRO-2 shard step for dense param groups using AdamW.

  For each group:
  1. Flatten grads and params per dtype
  2. Reduce-scatter(AVG) grads → local shard
  3. AdamW update on shard with shard-local state (exp_avg/exp_avg_sq)
  4. All-gather updated param shards

  Single GPU: just runs AdamW on full params (no RS/AG).
  Multi-GPU: full ZeRO-2 with sharding.
  """
  timers_on = os.getenv('NMOE_TIMERS', '1') not in ('0', 'false', 'False')
  if timers_on:
    try:
      from nmoe.metrics import cuda_time as _cuda_time  # local import to avoid hard dependency
      time_ctx = _cuda_time
    except Exception:
      time_ctx = lambda _tag: nullcontext()
  else:
    time_ctx = lambda _tag: nullcontext()

  world = dist.get_world_size(pg) if (dist.is_available() and dist.is_initialized()) else 1

  rank = dist.get_rank(pg) if (dist.is_available() and dist.is_initialized()) else 0

  for group_idx, group in enumerate(param_groups):
    params = list(group['params'])
    if not params:
      continue

    # Group by dtype
    by_dtype = {}
    for p in params:
      if p.dtype not in by_dtype:
        by_dtype[p.dtype] = []
      by_dtype[p.dtype].append(p)

    for dtype, group_params in by_dtype.items():
      flat = _get_or_init_flat_group(group, params=group_params, rank=rank, world=world, dtype=dtype)
      flat_param = flat['flat_param']
      param_shard = flat['param_shard']
      rs_in = flat['rs_in']
      rs_out = flat['rs_out']
      offsets: list[tuple[int, int]] = flat['offsets']
      total = int(flat['total'])
      padded_total = int(flat['padded_total'])
      shard_size = int(flat['shard_size'])
      shard_chunk = int(flat['shard_chunk'])

      # Get or initialize shard state
      state_key = f"shard_{rank}_{group_idx}_{dtype}"
      if state_key not in state:
        state[state_key] = {
          'step': 0,
          'exp_avg': torch.zeros(shard_size, dtype=dtype, device=param_shard.device),
          'exp_avg_sq': torch.zeros(shard_size, dtype=dtype, device=param_shard.device),
        }

      s = state[state_key]
      s['step'] += 1

      # AdamW update on shard (streamed per reduce-scatter chunk).
      beta1, beta2 = betas
      exp_avg = s['exp_avg']
      exp_avg_sq = s['exp_avg_sq']

      # Bias correction
      bias_correction1 = 1 - beta1 ** s['step']
      bias_correction2 = 1 - beta2 ** s['step']
      lr = float(group['lr'])
      wd = float(group.get('weight_decay', 0.0))
      step_size = lr / bias_correction1
      inv_bc2_sqrt = 1.0 / math.sqrt(bias_correction2)

      # Precompute flat views of grads (avoid repeated reshape work).
      grad_views: list[torch.Tensor | None] = []
      for p in group_params:
        if p.grad is None:
          grad_views.append(None)
        else:
          grad_views.append(p.grad.detach().reshape(-1))

      # Chunked reduce-scatter:
      # Treat conceptual flat_grad as [world, shard_size] laid out row-major.
      # For each shard offset, build rs_in[r, :] for each rank r, then RS(AVG) into rs_out.
      rs_in2d = rs_in.view(world, shard_chunk)
      cursors = [0 for _ in range(world)]

      with time_ctx('time_ms/zero2_reduce_scatter'):
        for shard_off in range(0, shard_size, shard_chunk):
          chunk_len = min(shard_chunk, shard_size - shard_off)
          # Zero rs_in for this chunk (required to avoid leaking previous chunk values).
          rs_in2d.zero_()

          for src_rank in range(world):
            g0 = src_rank * shard_size + shard_off
            if g0 >= total:
              continue
            g1 = min(g0 + chunk_len, total)
            row = rs_in2d[src_rank]

            i = cursors[src_rank]
            while i < len(offsets) and offsets[i][1] <= g0:
              i += 1
            cursors[src_rank] = i

            while i < len(offsets) and offsets[i][0] < g1:
              a, b = offsets[i]
              o0 = g0 if g0 > a else a
              o1 = g1 if g1 < b else b
              if o1 > o0:
                gv = grad_views[i]
                if gv is not None:
                  row[(o0 - g0):(o1 - g0)].copy_(gv[(o0 - a):(o1 - a)])
              i += 1

          if world == 1:
            rs_out[:chunk_len].copy_(rs_in2d[0, :chunk_len])
          else:
            dist.reduce_scatter_tensor(rs_out, rs_in, op=dist.ReduceOp.AVG, group=pg)

          g = rs_out[:chunk_len]
          p = param_shard[shard_off:(shard_off + chunk_len)]
          m = exp_avg[shard_off:(shard_off + chunk_len)]
          v = exp_avg_sq[shard_off:(shard_off + chunk_len)]

          m.mul_(beta1).add_(g, alpha=1 - beta1)
          v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

          denom = v.sqrt().mul_(inv_bc2_sqrt).add_(eps)
          if wd > 0.0:
            p.mul_(1.0 - lr * wd)
          p.addcdiv_(m, denom, value=-step_size)

      # Sync updated params to all ranks (no-op when world==1).
      if world > 1:
        with time_ctx('time_ms/zero2_all_gather'):
          dist.all_gather_into_tensor(flat_param, param_shard, group=pg)
      elif total < padded_total:
        # Keep padding clean for determinism.
        flat_param[total:padded_total].zero_()
