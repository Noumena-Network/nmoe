import os
import time
import math
import duckdb
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional
import subprocess

import torch
from torch.nn.utils import get_total_norm
from contextlib import contextmanager, nullcontext
import torch.distributed as dist
from collections import defaultdict
import os as _os

# Optional top‑level imports to avoid in‑function imports
try:  # model class references used by register_model_timers
    from nmoe.model import MLP  # type: ignore
except Exception:  # pragma: no cover
    MLP = None  # type: ignore

try:  # GPU metrics poller (C++ extension), optional
    from nmoe.csrc import gpu  # type: ignore
except Exception:  # pragma: no cover
    gpu = None  # type: ignore

# ------------------------------------------------------------
# NVIDIA B200 peak Tensor Core throughput (per GPU, dense)
#
# BF16: 2.25 PFLOPS  (dense)  ← 36 PFLOPS (sparse) per HGX ÷2 (dense) ÷8 GPUs
# FP8 : 4.50 PFLOPS  (dense)  ← 72 PFLOPS (sparse) per HGX ÷2 (dense) ÷8 GPUs
# NVFP4/FP4: 9.00 PFLOPS (dense)  ← 144|72 PFLOPS (sparse|dense) per HGX ÷8 GPUs
#
# Source (accessed Nov 27, 2025):
# - NVIDIA HGX Platform, HGX B200 specs table
#   Footnotes indicate specs listed as sparse; dense is one‑half; HGX = 8 GPUs.
#   https://www.nvidia.com/en-us/data-center/hgx/
#
# Fixed per‑GPU dense peaks for B200 (no env overrides).
# Units are TFLOPS (1e12 FLOP/s).
# ------------------------------------------------------------
B200_BF16_PEAK_TFLOPS: float = 2250.0
B200_FP8_PEAK_TFLOPS: float = 4500.0
B200_NVFP4_PEAK_TFLOPS: float = 9000.0

def b200_peak_tflops(dtype: str) -> float:
    """Return per‑GPU dense peak TFLOPS for B200 by dtype.

    Recognized keys: 'bf16', 'fp8', 'nvfp4'/'fp4'. Defaults to BF16.
    """
    d = dtype.lower()
    if d == "fp8":
        return B200_FP8_PEAK_TFLOPS
    if d in ("nvfp4", "fp4"):
        return B200_NVFP4_PEAK_TFLOPS
    return B200_BF16_PEAK_TFLOPS


def _crc32_u32(s: str) -> int:
    # Stable across runs/processes (unlike Python's hash()) and fast.
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


@dataclass
class _Nvfp4FlipAcc:
    # Previous sampled bytes from a packed nvfp4 cache tensor (uint8 on GPU)
    prev: torch.Tensor | None = None
    # GPU accumulators (avoid .item() per step); flushed on log cadence
    flips_sum: torch.Tensor | None = None  # int64 scalar on GPU
    denom_sum: torch.Tensor | None = None  # int64 scalar on GPU
    # Deterministic sample offset within the flattened byte view (cached per key)
    offset: int | None = None


class Nvfp4NibbleFlipTracker:
    """Estimate nvfp4 code flip rate from packed expert weight caches.

    Purpose (now): telemetry to correlate loss spikes with nvfp4 code churn/stall.
    Purpose (soon): a minimal feedback signal for "resonant" nvfp4 training where
    quantization dynamics act as intentional perturbations (no extra machinery).

    We sample a fixed-size contiguous byte window from packed nvfp4 caches and
    count how often each nibble changes vs the previous step.

    - Constant overhead wrt model size.
    - GPU-only accumulation; sync only on do_log steps.
    """

    SAMPLE_BYTES: int = 256 * 1024  # 256 KiB per tensor
    TARGET_FRAC: float = 0.5        # "resonant" heuristic (no tuning knob yet)

    def __init__(self, *, seed: int, keys: list[tuple[int, str]]):
        self.seed = int(seed) & 0xFFFFFFFF
        self.keys = keys  # (layer_idx, "W13_q"|"W2_q")
        self._acc: dict[tuple[int, str], _Nvfp4FlipAcc] = {k: _Nvfp4FlipAcc() for k in keys}

    def _offset(self, layer_idx: int, name: str, total_bytes: int) -> int:
        if total_bytes <= self.SAMPLE_BYTES:
            return 0
        h = _crc32_u32(f"{self.seed}/{layer_idx}/{name}")
        return int(h % (total_bytes - self.SAMPLE_BYTES + 1))

    @torch.no_grad()
    def _sample_u8(self, t: torch.Tensor, offset: int) -> torch.Tensor:
        flat = t.view(torch.uint8).reshape(-1)
        n = int(flat.numel())
        if n <= self.SAMPLE_BYTES:
            return flat.clone()
        return flat.narrow(0, int(offset), int(self.SAMPLE_BYTES)).clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for (layer_idx, name), slot in self._acc.items():
            try:
                blk = model.blocks[layer_idx]
                ffn = getattr(blk, "ffn", None)
                cache = getattr(ffn, "_W_cache", None) if ffn is not None else None
                if cache is None or getattr(cache, "profile", None) != "nvfp4":
                    continue
                t = getattr(cache, name, None)
                if t is None or (not t.is_cuda):
                    continue
                if t.dtype != torch.uint8:
                    continue
            except Exception:
                continue

            # Cache deterministic sample offset once (tensor size is static within a run).
            if slot.offset is None:
                try:
                    total_bytes = int(t.view(torch.uint8).numel())
                except Exception:
                    continue
                slot.offset = self._offset(layer_idx, name, total_bytes)

            cur = self._sample_u8(t, int(slot.offset))
            prev = slot.prev
            slot.prev = cur
            if prev is None or prev.numel() != cur.numel():
                continue

            xor = prev ^ cur
            # Count "nibble changed" events (two nibbles per byte) without host sync.
            flips = torch.count_nonzero(xor & 0x0F) + torch.count_nonzero(xor & 0xF0)
            denom = torch.full((), 2 * xor.numel(), device=xor.device, dtype=torch.int64)
            if slot.flips_sum is None:
                slot.flips_sum = flips.to(dtype=torch.int64)
                slot.denom_sum = denom
            else:
                slot.flips_sum += flips.to(dtype=torch.int64)
                slot.denom_sum += denom

    @torch.no_grad()
    def flush_means(self) -> dict[tuple[int, str], float]:
        out: dict[tuple[int, str], float] = {}
        for k, slot in self._acc.items():
            if slot.flips_sum is None or slot.denom_sum is None:
                continue
            frac = (slot.flips_sum.float() / slot.denom_sum.float()).clamp_(0.0, 1.0)
            out[k] = float(frac.item())  # sync only on do_log steps
            slot.flips_sum.zero_()
            slot.denom_sum.zero_()
        return out


_NVFP4_FLIP: Nvfp4NibbleFlipTracker | None = None
_NVFP4_FLIP_DISABLED: bool = False
_PENDING_TRAIN_SIGNAL_OVERRIDES: dict[str, dict[str, float]] | None = None


@dataclass
class MetricsState:
    ntokens_since_log: int
    last_log_time: float
    total_mem: int
    num_flops_per_token: float
    gpu_peak_flops: float


def publish_train_signal_overrides(overrides: Mapping[str, Mapping[str, float | int | None]]) -> None:
    global _PENDING_TRAIN_SIGNAL_OVERRIDES
    merged = dict(_PENDING_TRAIN_SIGNAL_OVERRIDES or {})
    for group, values in overrides.items():
        group_items = dict(merged.get(group, {}))
        for key, value in values.items():
            if value is None:
                continue
            group_items[str(key)] = float(value)
        if group_items:
            merged[str(group)] = group_items
    _PENDING_TRAIN_SIGNAL_OVERRIDES = merged or None


def pop_train_signal_overrides() -> dict[str, dict[str, float]]:
    global _PENDING_TRAIN_SIGNAL_OVERRIDES
    out = dict(_PENDING_TRAIN_SIGNAL_OVERRIDES or {})
    _PENDING_TRAIN_SIGNAL_OVERRIDES = None
    return out



def _num_flops_per_token(model: torch.nn.Module, seq_len: int) -> float:
    """Compute per-token FLOPs for this GPU, accounting for expert parallelism.

    With RDEP (expert parallelism):
    - Dense params are replicated: each GPU does full dense compute on local batch
    - Expert params are sharded: each GPU owns E/world experts
    - Per token, K experts are activated (same compute regardless of sharding)
    - Shared experts are replicated: full compute on each GPU

    The 6x multiplier accounts for forward + backward + optimizer (AdamW).
    """
    cfg = model.config
    H = cfg.dim
    E = cfg.n_routed_experts or 0
    K = cfg.n_activated_experts or 0
    Dff_moe = getattr(cfg, 'moe_inter_dim', 0) or 0
    L = cfg.n_layers
    L_moe = max(0, cfg.n_layers - cfg.n_dense_layers)
    S = getattr(cfg, 'n_shared_experts', 0) or 0

    # Get world_size to handle expert parallelism
    world = 1
    try:
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
    except Exception:
        pass

    # Local expert params (sharded across GPUs)
    n_local = E // world if E > 0 else 0
    per_moe_expert_params_local = n_local * (3 * H * Dff_moe)

    # Replicated params: router gate + shared experts
    per_moe_router_params = H * E
    per_shared_params = S * (3 * H * Dff_moe)

    # Total MoE-related params on this GPU
    routed_total_local = L_moe * (per_moe_expert_params_local + per_moe_router_params)
    shared_total = L_moe * per_shared_params

    # Dense params = total local params - MoE params
    nparams_total = sum(p.numel() for p in model.parameters())
    nparams_emb = model.embedding.weight.numel()
    nparams_dense = nparams_total - routed_total_local - shared_total

    # Active expert params per token: K experts activated, each has 3*H*Dff params
    # This is the same regardless of expert parallelism (we activate K experts per token)
    nparams_expert_active_per_token = K * (3 * H * Dff_moe)
    nparams_sparse_active = L_moe * (per_moe_router_params + nparams_expert_active_per_token)
    nparams_sparse_active += shared_total  # shared experts always active

    # Attention term (quadratic in seq_len for self-attention)
    head_dims = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim
    attn_term = 6.0 * L * cfg.n_heads * head_dims * seq_len

    return 6.0 * (nparams_dense - nparams_emb + nparams_sparse_active) + attn_term


def _get_peak_flops(device_name: str) -> int:
    # Copied from Torchtitan (bf16 peaks)
    try:
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        filtered = [line for line in result.stdout.splitlines() if "NVIDIA" in line and "H100" in line]
        device_name = " ".join(filtered) or device_name
    except FileNotFoundError:
        pass
    if "A100" in device_name:
        return int(312e12)
    elif "H100" in device_name:
        if "NVL" in device_name:
            return int(835e12)
        elif "PCIe" in device_name:
            return int(756e12)
        else:
            return int(989e12)
    elif "H200" in device_name:
        return int(989e12)
    elif "B200" in device_name:
        return int(2250e12)
    elif "MI355X" in device_name:
        return int(2500e12)
    elif "MI300X" in device_name or "MI325X" in device_name:
        return int(1300e12)
    elif "MI250X" in device_name:
        return int(191_500_000_000_000)
    elif "l40s" in device_name:
        return int(362e12)
    else:
        return int(312e12)


# -----------------------------
# Lightweight step aggregator
# -----------------------------

class _SegAgg:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.times: dict[str, float] = defaultdict(float)
        self.counts: dict[str, float] = defaultdict(float)

    def add_time_ms(self, tag: str, ms: float) -> None:
        # Accept fully-qualified tags like 'time_ms/comm_dispatch'
        self.times[tag] += float(ms)

    def add_count(self, tag: str, value: float) -> None:
        self.counts[tag] += float(value)

    def snapshot_and_reset(self) -> list[tuple[str, float]]:
        items: list[tuple[str, float]] = []
        for k, v in self.times.items():
            items.append((k, float(v)))
        for k, v in self.counts.items():
            items.append((k, float(v)))
        self.reset()
        return items


_SEG_AGG = _SegAgg()


def seg_reset() -> None:
    _SEG_AGG.reset()


def seg_add_time(tag: str, ms: float) -> None:
    _SEG_AGG.add_time_ms(tag, ms)


def seg_add_count(tag: str, value: float) -> None:
    _SEG_AGG.add_count(tag, value)


def _seg_peek() -> list[tuple[str, float]]:
    """Return current segment items without resetting (for derived metrics)."""
    out: list[tuple[str, float]] = []
    for k, v in _SEG_AGG.times.items():
        out.append((k, float(v)))
    for k, v in _SEG_AGG.counts.items():
        out.append((k, float(v)))
    return out


# -----------------------------
# Rank / timing utilities
# -----------------------------

def _is_rank0() -> bool:
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True  # single-GPU or non-dist treated as rank 0


# Stash CUDA event pairs for deferred timing at log-flush
_PENDING_TIMERS: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []


@contextmanager
def cuda_time(tag: str):
    """Record elapsed CUDA time for the current stream under `tag`.

    - Runs on all ranks; values are prefixed as per-rank series at flush time
      (e.g., `time_ms/r{rank}/...`).
    - Defers synchronization and elapsed_time() until log flush to avoid mid-step
      stalls.
    - Adds `time_ms/<name>` entries to the per-step aggregator.
    """
    if not torch.cuda.is_available():
        # No-op context
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        _PENDING_TIMERS.append((tag, start, end))


def _flush_timers_into_segments() -> None:
    if not _PENDING_TIMERS:
        return
    keep: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    for tag, s, e in _PENDING_TIMERS:
        try:
            # Non-blocking: only consume timers whose end event is complete.
            if not e.query():
                keep.append((tag, s, e))
                continue
            ms = float(s.elapsed_time(e))
            seg_add_time(tag, ms)
        except Exception:
            keep.append((tag, s, e))
    _PENDING_TIMERS[:] = keep


# ---------------------------------
# Module hook based segment timers
# ---------------------------------

def _install_timers_on_module(mod: torch.nn.Module, tag: str) -> None:
    """Install forward and backward timers on a module.

    Uses CUDA events recorded on the current stream; elapsed time is collected at log flush.
    No-ops when CUDA is unavailable or NMOE_TIMERS=0.
    """
    if not torch.cuda.is_available():
        return
    if _os.getenv('NMOE_TIMERS', '1') in ('0', 'false', 'False'):
        return

    key_f = f"_nmoe_timer_{tag}_fwd_start"
    key_b = f"_nmoe_timer_{tag}_bwd_start"

    def fwd_pre(_m, _inp):
        try:
            setattr(mod, key_f, torch.cuda.Event(enable_timing=True))
            getattr(mod, key_f).record()
        except Exception:
            pass

    def fwd_post(_m, _inp, _out):
        try:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start = getattr(mod, key_f, None)
            if start is not None:
                _PENDING_TIMERS.append((f"time_ms/{tag}_fwd", start, end))
        except Exception:
            pass

    def bwd_pre(_m, _grad_in):
        try:
            setattr(mod, key_b, torch.cuda.Event(enable_timing=True))
            getattr(mod, key_b).record()
        except Exception:
            pass

    def bwd_post(_m, _grad_in, _grad_out):
        try:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start = getattr(mod, key_b, None)
            if start is not None:
                _PENDING_TIMERS.append((f"time_ms/{tag}_bwd", start, end))
        except Exception:
            pass

    try:
        mod.register_forward_pre_hook(fwd_pre)
        mod.register_forward_hook(fwd_post)
        # full backward hooks for autograd modules
        if hasattr(mod, 'register_full_backward_pre_hook'):
            mod.register_full_backward_pre_hook(bwd_pre)
        if hasattr(mod, 'register_full_backward_hook'):
            mod.register_full_backward_hook(bwd_post)
    except Exception:
        pass


def register_model_timers(model: torch.nn.Module) -> None:
    """Register timing hooks on dense MLP and LM head.

    This keeps model code untouched and preserves minimalism. Hooks can be disabled
    by setting NMOE_TIMERS=0 in the environment.
    """
    if MLP is not None:
        for m in model.modules():
            if isinstance(m, MLP):
                _install_timers_on_module(m, 'dense_mlp')
    # LM head: final linear; tag as 'head'
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and getattr(m, 'out_features', None) == getattr(model, 'config', None).vocab_size:
            _install_timers_on_module(m, 'head')


def comm_counts(*, stage: str | None = None,
                M_recv: float | None = None,
                M_back: float | None = None,
                dropped: float | None = None,
                T: int | None = None,
                K: int | None = None) -> None:
    """Convenience to log comm counters; rank-0 only.

    Computes capacity and utilization if T,K given.
    """
    if M_recv is not None:
        seg_add_count('comm/M_recv', float(M_recv))
    if M_back is not None:
        seg_add_count('comm/M_back', float(M_back))
    if dropped is not None:
        seg_add_count('comm/dropped_rows', float(dropped))
    if T is not None and K is not None:
        world = 1
        try:
            if dist.is_available() and dist.is_initialized():
                world = dist.get_world_size()
        except Exception:
            pass
        capacity = float(T * K * world)
        seg_add_count('comm/capacity', capacity)
        if capacity > 0:
            if M_recv is not None:
                seg_add_count('comm/capacity_utilization', float(M_recv) / capacity)
            elif M_back is not None:
                seg_add_count('comm/capacity_utilization', float(M_back) / capacity)


def init_metrics(model: torch.nn.Module, seq_len: int) -> MetricsState:
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.reset_peak_memory_stats()
    dev_name = torch.cuda.get_device_name(0)
    return MetricsState(
        ntokens_since_log=0,
        last_log_time=time.perf_counter(),
        total_mem=torch.cuda.get_device_properties(0).total_memory,
        num_flops_per_token=_num_flops_per_token(model, seq_len),
        gpu_peak_flops=float(_get_peak_flops(dev_name)),
    )


def param_counts(model: torch.nn.Module) -> tuple[int, int]:
    cfg = model.config
    nparams_total = sum(p.numel() for p in model.parameters())
    H = cfg.dim
    E = cfg.n_routed_experts
    K = cfg.n_activated_experts
    Dff_moe = cfg.moe_inter_dim
    L = cfg.n_layers
    L_moe = max(0, cfg.n_layers - cfg.n_dense_layers)
    S = getattr(cfg, 'n_shared_experts', 0)
    per_moe_expert_params = E * (3 * H * Dff_moe)
    per_moe_router_params = H * E
    per_shared_params = S * (3 * H * Dff_moe)
    routed_total = L_moe * (per_moe_expert_params + per_moe_router_params)
    shared_total = L_moe * per_shared_params
    nparams_dense = nparams_total - (routed_total + shared_total)
    nparams_sparse_active = L_moe * (per_moe_router_params + per_moe_expert_params * (K / E)) + shared_total
    nparams_active = int(nparams_dense + nparams_sparse_active)
    return int(nparams_total), nparams_active


def log_param_summary(model: torch.nn.Module, print_fn: Callable[[str], None]) -> None:
    total, active = param_counts(model)
    b = 1_000_000_000
    print_fn(f"({total / b:.2f}B params {active / b:.2f}B active)")


def log_step_torchtitan(
    step: int,
    model: torch.nn.Module,
    loss_value: torch.Tensor,
    tokens_this_step: int,
    state: MetricsState,
    print_fn: Callable[[str], None],
    *,
    do_log: bool = True,
) -> dict:
    state.ntokens_since_log += tokens_this_step
    if not do_log:
        return {}
    now = time.perf_counter()
    dt = now - state.last_log_time
    tps = state.ntokens_since_log / dt
    # World size for optional node_tps print (rank0 prints via print_fn)
    try:
        world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    except Exception:
        world = 1

    is_rank0 = _is_rank0()

    # Loss: mean across ranks (cheap scalar allreduce) for stable logging.
    loss_t = loss_value.detach().float()
    if world > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        loss_t.div_(float(world))
    loss_f = float(loss_t.item())

    # Grad L2 norm
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if grads:
        local_norm = get_total_norm(grads, norm_type=2.0, foreach=True)
        norm2 = local_norm.float() * local_norm.float()
        if world > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(norm2, op=dist.ReduceOp.SUM)
        grad_norm = float(norm2.sqrt().item())
    else:
        grad_norm = 0.0

    reserved_bytes = torch.cuda.max_memory_reserved(0)
    reserved_gib = reserved_bytes / (1024**3)
    reserved_pct = reserved_bytes / state.total_mem * 100.0
    alloc_bytes = torch.cuda.memory_allocated(0)
    alloc_gib = alloc_bytes / (1024**3)
    max_alloc_bytes = torch.cuda.max_memory_allocated(0)
    max_alloc_gib = max_alloc_bytes / (1024**3)

    tflops = state.num_flops_per_token * tps / 1e12

    # Derive ms/step from measured tokens/s and per-rank tokens_per_step (BS × Seq)
    tokens_per_step = max(1, int(tokens_this_step))
    ms_per_step = (tokens_per_step / max(tps, 1e-9)) * 1000.0

    line = (
        f"step: {step:2}  "
        f"loss: {loss_f:7.4f}  "
        f"grad_norm: {grad_norm:7.4f}  "
        f"memory: {reserved_gib:5.2f}GiB({reserved_pct:.2f}%)  "
        f"tps: {round(tps):,}  "
        f"tflops: {tflops:,.2f}  "
        f"ms/step: {ms_per_step:.1f}"
    )
    if world > 1:
        try:
            node_tps = round(tps * world)
            line += f"  node_tps: {node_tps:,}"
        except Exception:
            pass
    print_fn(line)

    out = {
        'tps': float(tps),
        'tflops': float(tflops),
        'ms_per_step': float(ms_per_step),
        'grad_norm': float(grad_norm),
        'loss': float(loss_f),
        'reserved_gib': float(reserved_gib),
        'reserved_pct': float(reserved_pct),
        'alloc_gib': float(alloc_gib),
        'max_alloc_gib': float(max_alloc_gib),
    }

    state.ntokens_since_log = 0
    state.last_log_time = now
    torch.cuda.reset_peak_memory_stats()
    return out


def log_router_stats(model: torch.nn.Module, print_fn: Callable[[str], None]) -> None:
    # Use guarded collector to avoid NaNs / div0 in console output
    layers = [blk.ffn for blk in getattr(model, "blocks") if hasattr(blk, "ffn") and hasattr(blk.ffn, "last_aux_loss")]
    if not layers:
        return
    # Drop aux by default; compute only if non-zero alpha is enabled
    aux_vals = []
    for ffn in layers:
        alpha = float(getattr(ffn, 'aux_loss_alpha', 0.0)) if hasattr(ffn, 'aux_loss_alpha') else 0.0
        if alpha > 0 and hasattr(ffn, 'last_aux_loss'):
            aux_vals.append(float(ffn.last_aux_loss.detach().cpu()))
    aux = sum(aux_vals) if aux_vals else 0.0
    per, agg = collect_router_stats(model)
    mean_cv = agg.get('mean_cv') if isinstance(agg, dict) else None
    max_loads = [p.get('max_load') for p in per if p.get('max_load') is not None]
    bmins = [p.get('bias_min') for p in per if p.get('bias_min') is not None]
    bmaxs = [p.get('bias_max') for p in per if p.get('bias_max') is not None]
    mx = (sum(max_loads) / len(max_loads)) if max_loads else 0.0
    bmin = min(bmins) if bmins else 0.0
    bmax = max(bmaxs) if bmaxs else 0.0
    cv_str = f"{mean_cv:.2f}%" if mean_cv is not None else "--"
    if aux_vals:
        line = f"router: aux {aux:.4f}  cv {cv_str}  max {mx:.2f}%  bias[{bmin:.2f},{bmax:.2f}]"
    else:
        line = f"router: cv {cv_str}  max {mx:.2f}%  bias[{bmin:.2f},{bmax:.2f}]"
    print_fn(line)


# ==========================
# Metrics Writer (Parquet)
# ==========================

class MetricsWriter:
    """Parquet-per-step metrics writer for training telemetry.

    Contract (authoritative live store):
      /data/metrics/{run_id}/step_XXXXXXXX.parquet

    Schema (single source of truth):
      metrics(run TEXT, tag TEXT, step INTEGER, ts_ms BIGINT, value DOUBLE)
    """

    def __init__(self, run_dir: str, run: str) -> None:
        self.run_dir = os.path.abspath(str(run_dir))
        os.makedirs(self.run_dir, exist_ok=True)
        self.run = str(run)
        # Use DuckDB in-memory as a tiny buffer for the current step, then flush to parquet.
        self._conn = duckdb.connect(":memory:")
        self._dirty_steps: set[int] = set()
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS buf (
              run   TEXT NOT NULL,
              tag   TEXT NOT NULL,
              step  INTEGER NOT NULL,
              ts_ms BIGINT NOT NULL,
              value DOUBLE NOT NULL,
              PRIMARY KEY (run, tag, step)
            )
            """
        )

    def close(self) -> None:
        try:
            for s in sorted(list(self._dirty_steps)):
                self.flush_parquet(s)
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass

    def _now_ts_ms(self) -> int:
        return int(time.time() * 1000)

    def insert(self, step: int, tag: str, value: float, ts_ms: Optional[int] = None) -> None:
        ts_val = int(self._now_ts_ms() if ts_ms is None else ts_ms)
        self._conn.execute(
            "INSERT OR REPLACE INTO buf(run, tag, step, ts_ms, value) VALUES (?,?,?,?,?)",
            [self.run, tag, int(step), ts_val, float(value)],
        )
        self._dirty_steps.add(int(step))

    def insert_many(self, step: int, items: Iterable[tuple[str, float]], ts_ms: Optional[int] = None) -> None:
        ts_val = int(self._now_ts_ms() if ts_ms is None else ts_ms)
        rows = [(self.run, tag, int(step), ts_val, float(val)) for tag, val in items]
        if not rows:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO buf(run, tag, step, ts_ms, value) VALUES (?,?,?,?,?)",
            rows,
        )
        self._dirty_steps.add(int(step))

    def flush_parquet(self, step: int) -> None:
        """Flush current step data to parquet for concurrent readers. Call on rank 0 only."""
        try:
            s = int(step)
            if s not in self._dirty_steps:
                return
            parquet_path = os.path.join(self.run_dir, f"step_{s:08d}.parquet")
            tmp_path = os.path.join(self.run_dir, f"step_{s:08d}.parquet.tmp")
            parquet_sql = parquet_path.replace("'", "''")
            tmp_sql = tmp_path.replace("'", "''")
            if os.path.exists(parquet_path):
                existing_sql = f"SELECT run, tag, step, ts_ms, value FROM read_parquet('{parquet_sql}')"
            else:
                existing_sql = "SELECT run, tag, step, ts_ms, value FROM buf WHERE 0=1"
            self._conn.execute(
                "COPY ("
                "  SELECT run, tag, step, ts_ms, value FROM ("
                "    SELECT *, row_number() OVER (PARTITION BY run, tag, step ORDER BY ts_ms DESC) AS rn"
                "    FROM ("
                f"      {existing_sql}"
                "      UNION ALL"
                f"      SELECT run, tag, step, ts_ms, value FROM buf WHERE step = {s}"
                "    )"
                "  ) WHERE rn = 1"
                "  ORDER BY tag"
                f") TO '{tmp_sql}' (FORMAT PARQUET)"
            )
            os.replace(tmp_path, parquet_path)  # Atomic rename
            # Keep the in-memory buffer bounded.
            self._conn.execute(f"DELETE FROM buf WHERE step = {s}")
            self._dirty_steps.discard(s)
        except Exception:
            pass

    # ---- Convenience helpers ----
    def log_core(self, step: int, *, loss: float, lr_dense: float,
                 lr_router: Optional[float] = None,
                 lr_expert: Optional[float] = None,
                 lr_muon: Optional[float] = None,
                 grad_norm: Optional[float] = None,
                 tokens_seen: Optional[float] = None,
                 tokens_per_s_gpu: Optional[float] = None,
                 ms_per_step: Optional[float] = None,
                 tflops: Optional[float] = None,
                 loader_wait_ms: Optional[float] = None,
                 memory_current_alloc_gib: Optional[float] = None,
                 memory_max_alloc_gib: Optional[float] = None) -> None:
        items: list[tuple[str, float]] = [("train/loss", loss), ("optimizer/lr", lr_dense), ("optimizer/lr_dense", lr_dense)]
        if lr_router is not None:
            items.append(("optimizer/lr_router", lr_router))
        if lr_expert is not None:
            items.append(("optimizer/lr_expert", lr_expert))
        if lr_muon is not None:
            items.append(("optimizer/lr_muon", lr_muon))
        if grad_norm is not None:
            items.append(("train/grad_norm", grad_norm))
        if tokens_seen is not None:
            items.append(("train/tokens_seen", tokens_seen))
        if tokens_per_s_gpu is not None:
            items.append(("throughput/tokens_per_s_gpu", tokens_per_s_gpu))
        if ms_per_step is not None:
            items.append(("throughput/ms_per_step", ms_per_step))
        if tflops is not None:
            items.append(("efficiency/tflops", tflops))
        if loader_wait_ms is not None:
            items.append(("throughput/loader_wait_ms", loader_wait_ms))
        if memory_current_alloc_gib is not None:
            items.append(("memory/current_alloc_gib", memory_current_alloc_gib))
        if memory_max_alloc_gib is not None:
            items.append(("memory/max_alloc_gib", memory_max_alloc_gib))
        self.insert_many(step, items)

    def log_train_signal_group(self, step: int, group: str, *,
                               grad_norm: Optional[float] = None,
                               param_norm: Optional[float] = None,
                               grad_to_param: Optional[float] = None,
                               param_count: Optional[int] = None,
                               pre_param_norm: Optional[float] = None,
                               update_norm: Optional[float] = None,
                               update_to_pre_param: Optional[float] = None,
                               optimizer_update_norm: Optional[float] = None,
                               optimizer_update_to_pre_param: Optional[float] = None,
                               weight_decay_update_norm: Optional[float] = None) -> None:
        prefix = f"train_signal/{group}"
        items: list[tuple[str, float]] = []
        if grad_norm is not None:
            items.append((f"{prefix}/grad_norm", grad_norm))
        if param_norm is not None:
            items.append((f"{prefix}/param_norm", param_norm))
        if grad_to_param is not None:
            items.append((f"{prefix}/grad_to_param", grad_to_param))
        if param_count is not None:
            items.append((f"{prefix}/param_count", float(param_count)))
        if pre_param_norm is not None:
            items.append((f"{prefix}/pre_param_norm", pre_param_norm))
        if update_norm is not None:
            items.append((f"{prefix}/update_norm", update_norm))
        if update_to_pre_param is not None:
            items.append((f"{prefix}/update_to_pre_param", update_to_pre_param))
        if optimizer_update_norm is not None:
            items.append((f"{prefix}/optimizer_update_norm", optimizer_update_norm))
        if optimizer_update_to_pre_param is not None:
            items.append((f"{prefix}/optimizer_update_to_pre_param", optimizer_update_to_pre_param))
        if weight_decay_update_norm is not None:
            items.append((f"{prefix}/weight_decay_update_norm", weight_decay_update_norm))
        if items:
            self.insert_many(step, items)

    def log_router_layer(self, step: int, layer_idx: int, *,
                         cv: Optional[float] = None,
                         entropy: Optional[float] = None,
                         experts_active: Optional[int] = None,
                         bias_range: Optional[float] = None,
                         max_load: Optional[float] = None,
                         min_load: Optional[float] = None,
                         p10_load: Optional[float] = None,
                         p50_load: Optional[float] = None,
                         p90_load: Optional[float] = None,
                         importance_cv: Optional[float] = None,
                         importance_entropy: Optional[float] = None,
                         max_importance: Optional[float] = None,
                         min_importance: Optional[float] = None,
                         p10_importance: Optional[float] = None,
                         p50_importance: Optional[float] = None,
                         p90_importance: Optional[float] = None) -> None:
        prefix = f"router/layer_{layer_idx:02d}"
        items: list[tuple[str, float]] = []
        if cv is not None:
            items.append((f"{prefix}/cv", cv))
        if entropy is not None:
            items.append((f"{prefix}/entropy", entropy))
        if experts_active is not None:
            items.append((f"{prefix}/experts_active", float(experts_active)))
        if bias_range is not None:
            items.append((f"{prefix}/bias_range", bias_range))
        if max_load is not None:
            items.append((f"{prefix}/max_load", max_load))
        if min_load is not None:
            items.append((f"{prefix}/min_load", min_load))
        if p10_load is not None:
            items.append((f"{prefix}/p10_load", p10_load))
        if p50_load is not None:
            items.append((f"{prefix}/p50_load", p50_load))
        if p90_load is not None:
            items.append((f"{prefix}/p90_load", p90_load))
        if importance_cv is not None:
            items.append((f"{prefix}/importance_cv", importance_cv))
        if importance_entropy is not None:
            items.append((f"{prefix}/importance_entropy", importance_entropy))
        if max_importance is not None:
            items.append((f"{prefix}/max_importance", max_importance))
        if min_importance is not None:
            items.append((f"{prefix}/min_importance", min_importance))
        if p10_importance is not None:
            items.append((f"{prefix}/p10_importance", p10_importance))
        if p50_importance is not None:
            items.append((f"{prefix}/p50_importance", p50_importance))
        if p90_importance is not None:
            items.append((f"{prefix}/p90_importance", p90_importance))
        if items:
            self.insert_many(step, items)

    def log_router_agg(self, step: int, **metrics: float) -> None:
        items = [(f"router_agg/{k}", float(v)) for k, v in metrics.items() if v is not None]
        if items:
            self.insert_many(step, items)

    def log_comm(self, step: int, **metrics: float) -> None:
        # Example: a2a_ms, pack_ms, unpack_ms, clone_ms, gate_ms
        items = [(f"comm/{k}", float(v)) for k, v in metrics.items() if v is not None]
        if items:
            self.insert_many(step, items)

    def log_gpu_snapshot(self, step: int, snapshot: Iterable[Mapping[str, float]]) -> None:
        # Per-GPU series
        base_items: list[tuple[str, float]] = []
        agg = {
            'mean_utilization_gpu': 0.0,
            'total_memory_used_gib': 0.0,
            'total_power_w': 0.0,
            'max_temperature_c': 0.0,
        }
        cnt = 0
        for g in snapshot:
            idx = int(g.get('index', cnt))
            def val(name: str) -> Optional[float]:
                v = g.get(name)
                try:
                    return float(v) if v is not None else None
                except Exception:
                    return None
            pairs = [
                (f"gpu/{idx}/utilization_gpu", val('utilization_gpu')),
                (f"gpu/{idx}/memory_used_gib", val('memory_used_gib')),
                (f"gpu/{idx}/memory_total_gib", val('memory_total_gib')),
                (f"gpu/{idx}/power_draw_w", val('power_draw_w')),
                (f"gpu/{idx}/power_limit_w", val('power_limit_w')),
                (f"gpu/{idx}/temperature_c", val('temperature_c')),
                (f"gpu/{idx}/clocks_sm_mhz", val('clocks_sm_mhz')),
                (f"gpu/{idx}/throttle_thermal", val('throttle_thermal')),
                (f"gpu/{idx}/throttle_power", val('throttle_power')),
                (f"gpu/{idx}/throttle_hw_slowdown", val('throttle_hw_slowdown')),
                (f"gpu/{idx}/throttle_apps", val('throttle_apps')),
            ]
            # ECC counters may be large; cast to float for storage
            ecc_c = g.get('ecc_corrected')
            if ecc_c is not None:
                try:
                    pairs.append((f"gpu/{idx}/ecc_corrected", float(ecc_c)))
                except Exception:
                    pass
            ecc_u = g.get('ecc_uncorrected')
            if ecc_u is not None:
                try:
                    pairs.append((f"gpu/{idx}/ecc_uncorrected", float(ecc_u)))
                except Exception:
                    pass
            for tag, v in pairs:
                if v is not None:
                    base_items.append((tag, v))
            # Aggregates
            u = val('utilization_gpu') or 0.0
            m_used = val('memory_used_gib') or 0.0
            p = val('power_draw_w') or 0.0
            t = val('temperature_c') or 0.0
            agg['mean_utilization_gpu'] += u
            agg['total_memory_used_gib'] += m_used
            agg['total_power_w'] += p
            agg['max_temperature_c'] = max(agg['max_temperature_c'], t)
            cnt += 1
        if base_items:
            self.insert_many(step, base_items)
        if cnt > 0:
            self.insert_many(step, [
                ("gpu_agg/mean_utilization_gpu", agg['mean_utilization_gpu'] / cnt),
                ("gpu_agg/total_memory_used_gib", agg['total_memory_used_gib']),
                ("gpu_agg/total_power_w", agg['total_power_w']),
                ("gpu_agg/max_temperature_c", agg['max_temperature_c']),
            ])

# ------------------------------
# Training-run orchestration API
# ------------------------------

@dataclass
class MetricsContext:
    writer: Optional[MetricsWriter]
    gpu_snapshot_fn: Optional[Callable[[], Iterable[Mapping[str, float]]]]
    gpu_stop_fn: Optional[Callable[[], None]]
    last_gpu_log: float = 0.0


def start_metrics(run_id: Optional[str] = None,
                  metrics_dir: Optional[str] = None,
                  poll_interval_ms: int = 1000) -> MetricsContext:
    """Initialize metrics writer and GPU poller.

    Safe to call in environments without NVML or extension; returns a context
    with writer/snapshot set to None on failures.
    """
    writer: Optional[MetricsWriter] = None
    gpu_snapshot_fn = None
    gpu_stop_fn = None
    try:
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    except Exception:
        rank = 0

    try:
        # Rank-0-only writer. Live metrics are written as per-step parquet files under:
        #   {metrics_dir}/{run_id}/step_XXXXXXXX.parquet
        rid = run_id or os.getenv('NMOE_RUN') or time.strftime('%Y%m%d-%H%M%S')
        mdir = metrics_dir or os.getenv('NMOE_METRICS_DIR', '/data/metrics')
        run_dir = os.path.join(mdir, rid)
        os.makedirs(run_dir, exist_ok=True)
        writer = MetricsWriter(run_dir, run=rid) if rank == 0 else None
    except Exception:
        writer = None

    # GPU poller via csrc extension (optional)
    try:
        # Start GPU poller on rank 0 only
        if rank == 0 and gpu is not None:
            try:
                gpu.start(poll_interval_ms)
                gpu_snapshot_fn = gpu.snapshot
                gpu_stop_fn = gpu.stop
            except Exception:
                gpu_snapshot_fn = None
                gpu_stop_fn = None
        else:
            gpu_snapshot_fn = None
            gpu_stop_fn = None
    except Exception:
        gpu_snapshot_fn = None
        gpu_stop_fn = None

    return MetricsContext(writer=writer, gpu_snapshot_fn=gpu_snapshot_fn, gpu_stop_fn=gpu_stop_fn, last_gpu_log=0.0)


def stop_metrics(ctx: Optional[MetricsContext]) -> None:
    if ctx is None:
        return
    try:
        if ctx.gpu_stop_fn is not None:
            ctx.gpu_stop_fn()
    except Exception:
        pass
    try:
        if ctx.writer is not None:
            ctx.writer.close()
    except Exception:
        pass


def collect_param_group_stats(model: torch.nn.Module):
    """Return global grad/param norms for dense, router, and expert groups."""
    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named_params:
        return {}

    expert_ids: set[int] = set()
    if hasattr(model, "param_sets"):
        try:
            expert_params, _ = model.param_sets()  # type: ignore[attr-defined]
            expert_ids = {id(p) for p in expert_params}
        except Exception:
            expert_ids = set()

    device = named_params[0][1].device
    # Per group: grad_sq, param_sq, param_count
    acc = torch.zeros(9, device=device, dtype=torch.float64)
    group_index = {"dense": 0, "router": 1, "expert": 2}

    for name, p in named_params:
        if id(p) in expert_ids:
            group = "expert"
        elif "router" in name:
            group = "router"
        else:
            group = "dense"
        base = group_index[group] * 3
        data = p.detach().float()
        acc[base + 1] += (data * data).sum(dtype=torch.float64)
        acc[base + 2] += float(p.numel())
        if p.grad is not None:
            grad = p.grad.detach().float()
            acc[base + 0] += (grad * grad).sum(dtype=torch.float64)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)

    out = {}
    for group, idx in group_index.items():
        base = idx * 3
        grad_sq = float(acc[base + 0].item())
        param_sq = float(acc[base + 1].item())
        param_count = int(round(float(acc[base + 2].item())))
        grad_norm = math.sqrt(grad_sq) if grad_sq > 0.0 else 0.0
        param_norm = math.sqrt(param_sq) if param_sq > 0.0 else 0.0
        grad_to_param = (grad_norm / param_norm) if param_norm > 0.0 else None
        out[group] = {
            'grad_norm': grad_norm,
            'param_norm': param_norm,
            'grad_to_param': grad_to_param,
            'param_count': param_count,
        }
    return out


def collect_router_stats(model: torch.nn.Module):
    """Return (per_layer_stats, agg_stats) for router metrics.

    per_layer_stats: list of dict(li, cv, entropy, load quantiles, bias_range)
    agg_stats: dict(mean_cv, std_cv, mean_entropy, min_entropy, dead_experts_count, ...)
    """
    layers = [blk.ffn for blk in getattr(model, "blocks") if hasattr(blk, "ffn") and hasattr(blk.ffn, "last_aux_loss")]
    per = []
    cvs, ents, maxps = [], [], []
    minps, p10s, p50s, p90s = [], [], [], []
    importance_cvs, importance_ents, importance_maxps = [], [], []
    importance_minps, importance_p10s, importance_p50s, importance_p90s = [], [], [], []
    dead_total = 0
    experts_active_list = []
    for li, ffn in enumerate(layers):
        cv = ent = mx = mn = p10 = p50 = p90 = None
        imp_cv = imp_ent = imp_mx = imp_mn = imp_p10 = imp_p50 = imp_p90 = None
        bmin = bmax = None
        experts_active = None
        if hasattr(ffn, "last_loads"):
            l = ffn.last_loads.detach().float()
            m = l.mean()
            if m.item() != 0.0:
                cv = (l.std(unbiased=False) / m * 100.0).item()
            mn = (l.min() * 100.0).item()
            p10 = (torch.quantile(l, 0.10) * 100.0).item()
            p50 = (torch.quantile(l, 0.50) * 100.0).item()
            p90 = (torch.quantile(l, 0.90) * 100.0).item()
            mx = (l.max() * 100.0).item()
            p = (l / l.sum().clamp_min(1e-9)).clamp_min(1e-12)
            ent = float((-p * p.log()).sum().item())
            dead_total += int((l <= 0).sum().item())
            experts_active = int((l > 0).sum().item())
            experts_active_list.append(float(experts_active))
        if hasattr(ffn, "last_importance") and ffn.last_importance is not None:
            imp = ffn.last_importance.detach().float()
            imp_mean = imp.mean()
            if imp_mean.item() != 0.0:
                imp_cv = (imp.std(unbiased=False) / imp_mean * 100.0).item()
            imp_mn = (imp.min() * 100.0).item()
            imp_p10 = (torch.quantile(imp, 0.10) * 100.0).item()
            imp_p50 = (torch.quantile(imp, 0.50) * 100.0).item()
            imp_p90 = (torch.quantile(imp, 0.90) * 100.0).item()
            imp_mx = (imp.max() * 100.0).item()
            imp_p = (imp / imp.sum().clamp_min(1e-12)).clamp_min(1e-12)
            imp_ent = float((-imp_p * imp_p.log()).sum().item())
        if hasattr(ffn, "router") and hasattr(ffn.router, "bias"):
            b = ffn.router.bias.detach().float()
            bmin = float(b.min().item())
            bmax = float(b.max().item())
        if cv is not None:
            cvs.append(cv)
        if ent is not None:
            ents.append(ent)
        if mx is not None:
            maxps.append(mx)
        if mn is not None:
            minps.append(mn)
        if p10 is not None:
            p10s.append(p10)
        if p50 is not None:
            p50s.append(p50)
        if p90 is not None:
            p90s.append(p90)
        if imp_cv is not None:
            importance_cvs.append(imp_cv)
        if imp_ent is not None:
            importance_ents.append(imp_ent)
        if imp_mx is not None:
            importance_maxps.append(imp_mx)
        if imp_mn is not None:
            importance_minps.append(imp_mn)
        if imp_p10 is not None:
            importance_p10s.append(imp_p10)
        if imp_p50 is not None:
            importance_p50s.append(imp_p50)
        if imp_p90 is not None:
            importance_p90s.append(imp_p90)
        per.append({
            'layer': li,
            'cv': cv,
            'entropy': ent,
            'max_load': mx,
            'min_load': mn,
            'p10_load': p10,
            'p50_load': p50,
            'p90_load': p90,
            'importance_cv': imp_cv,
            'importance_entropy': imp_ent,
            'max_importance': imp_mx,
            'min_importance': imp_mn,
            'p10_importance': imp_p10,
            'p50_importance': imp_p50,
            'p90_importance': imp_p90,
            'bias_min': bmin,
            'bias_max': bmax,
            'experts_active': experts_active,
        })

    agg = {
        'mean_cv': (sum(cvs) / len(cvs)) if cvs else None,
        'std_cv': (float(torch.tensor(cvs).std(unbiased=False).item()) if cvs else None),
        'mean_entropy': (sum(ents) / len(ents)) if ents else None,
        'min_entropy': (min(ents) if ents else None),
        'dead_experts_count': float(dead_total),
        'experts_active_mean': (sum(experts_active_list) / len(experts_active_list)) if experts_active_list else None,
        'mean_min_load': (sum(minps) / len(minps)) if minps else None,
        'mean_p10_load': (sum(p10s) / len(p10s)) if p10s else None,
        'mean_p50_load': (sum(p50s) / len(p50s)) if p50s else None,
        'mean_p90_load': (sum(p90s) / len(p90s)) if p90s else None,
        'mean_max_load': (sum(maxps) / len(maxps)) if maxps else None,
        'mean_importance_cv': (sum(importance_cvs) / len(importance_cvs)) if importance_cvs else None,
        'mean_importance_entropy': (sum(importance_ents) / len(importance_ents)) if importance_ents else None,
        'mean_min_importance': (sum(importance_minps) / len(importance_minps)) if importance_minps else None,
        'mean_p10_importance': (sum(importance_p10s) / len(importance_p10s)) if importance_p10s else None,
        'mean_p50_importance': (sum(importance_p50s) / len(importance_p50s)) if importance_p50s else None,
        'mean_p90_importance': (sum(importance_p90s) / len(importance_p90s)) if importance_p90s else None,
        'mean_max_importance': (sum(importance_maxps) / len(importance_maxps)) if importance_maxps else None,
    }
    return per, agg


def log_training_step(step: int,
                      *,
                      model: torch.nn.Module,
                      loss: torch.Tensor,
                      lr_dense: float,
                      lr_router: Optional[float] = None,
                      lr_expert: Optional[float] = None,
                      lr_muon: Optional[float] = None,
                      tokens_seen: int,
                      tokens_this_step: int,
                      state: MetricsState,
                      print_fn: Callable[[str], None],
                      ctx: Optional[MetricsContext] = None,
                      log_every_seconds: float = 1.0,
                      loader_wait_ms: Optional[float] = None) -> dict:
    """Print Torchtitan-style step log, persist core + router metrics, and snapshot GPU ~1 Hz.

    Returns the dict from log_step_torchtitan.
    """
    # Log gating: do work only every cfg.log_every steps (and step 1).
    log_every = 1
    try:
        cfg = getattr(model, 'config', None)
        if cfg is not None:
            log_every = int(getattr(cfg, 'log_every', 1))
    except Exception:
        log_every = 1
    log_every = max(1, log_every)
    do_log = (step == 1) or ((step % log_every) == 0)
    try:
        if cfg is not None and int(getattr(cfg, 'steps', 0)) == int(step):
            do_log = True
    except Exception:
        pass

    # ------------------------------------------------------------
    # NVFP4 flip-rate tracker: update every step, flush on do_log.
    # Auto-enables only when nvfp4 MoE caches exist AND DuckDB writer exists.
    # ------------------------------------------------------------
    global _NVFP4_FLIP, _NVFP4_FLIP_DISABLED
    if _NVFP4_FLIP is None and (not _NVFP4_FLIP_DISABLED) and ctx is not None and ctx.writer is not None:
        try:
            # Find first+last nvfp4 MoE layers as canaries.
            moe_layer_idxs: list[int] = []
            for i, blk in enumerate(getattr(model, "blocks", [])):
                ffn = getattr(blk, "ffn", None)
                if getattr(ffn, "_dtype", None) != "nvfp4":
                    continue
                if not bool(getattr(ffn, "_use_blockscaled", False)):
                    continue
                moe_layer_idxs.append(int(i))
            if moe_layer_idxs:
                canaries = [moe_layer_idxs[0], moe_layer_idxs[-1]]
                seed = _crc32_u32(str(ctx.writer.run))
                keys = [(i, "W13_q") for i in canaries] + [(i, "W2_q") for i in canaries]
                _NVFP4_FLIP = Nvfp4NibbleFlipTracker(seed=seed, keys=keys)
            else:
                _NVFP4_FLIP_DISABLED = True
        except Exception:
            _NVFP4_FLIP_DISABLED = True

    if _NVFP4_FLIP is not None:
        try:
            _NVFP4_FLIP.update(model)
        except Exception:
            pass

    # Console prints rank0 only.
    is_rank0 = _is_rank0()
    print_out = print_fn if is_rank0 else (lambda *_args, **_kwargs: None)

    out = log_step_torchtitan(
        step,
        model,
        loss,
        tokens_this_step,
        state,
        print_out,
        do_log=do_log,
    )
    if not do_log:
        return out

    signal_stats = collect_param_group_stats(model)
    for group, extra in pop_train_signal_overrides().items():
        signal_stats.setdefault(group, {}).update(extra)

    # Persist core metrics (rank 0 only)
    rank = 0
    try:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
    except Exception:
        rank = 0

    if ctx is not None and ctx.writer is not None and rank == 0:
        try:
            # Flush deferred CUDA timers before snapshotting segments
            _flush_timers_into_segments()
            # Derive per-dtype TFLOP/s using measured segments (best-effort)
            # Collect current segment times (non-reset)
            seg_now = dict(_seg_peek())
            ms_attn = seg_now.get('time_ms/attn_fwd', 0.0) + seg_now.get('time_ms/attn_bwd', 0.0)
            ms_dense = seg_now.get('time_ms/dense_mlp_fwd', 0.0) + seg_now.get('time_ms/dense_mlp_bwd', 0.0)
            ms_head = seg_now.get('time_ms/head_fwd', 0.0) + seg_now.get('time_ms/head_bwd', 0.0)
            ms_fp8 = seg_now.get('time_ms/expert_mlp_fp8', 0.0)
            # Flops per token split (approx): fp8 = 6 * 3*H*Dff * L_moe * K; bf16 = total - fp8
            try:
                cfg = getattr(model, 'config', None)
                H = int(cfg.dim)
                Dff = int(cfg.moe_inter_dim)
                L = int(cfg.n_layers)
                L_moe = max(0, L - int(cfg.n_dense_layers))
                K = int(cfg.n_activated_experts)
                flops_token_total = _num_flops_per_token(model, cfg.seq_len)
                flops_token_fp8 = 6.0 * (3.0 * H * Dff) * L_moe * K
                flops_token_bf16 = max(0.0, flops_token_total - flops_token_fp8)
                # Per-step FLOPs
                toks = float(tokens_this_step)
                flops_fp8_step = flops_token_fp8 * toks
                flops_bf16_step = flops_token_bf16 * toks
                # Achieved TFLOPs (compute-only for the dtype segments we timed)
                tflops_fp8_ach = (flops_fp8_step / (ms_fp8 / 1000.0) / 1e12) if ms_fp8 > 0 else None
                ms_bf16 = ms_attn + ms_dense + ms_head
                tflops_bf16_ach = (flops_bf16_step / (ms_bf16 / 1000.0) / 1e12) if ms_bf16 > 0 else None
            except Exception:
                tflops_fp8_ach = None
                tflops_bf16_ach = None
            ctx.writer.log_core(
                step,
                loss=float(out.get('loss', 0.0)),
                lr_dense=float(lr_dense),
                lr_router=(float(lr_router) if lr_router is not None else None),
                lr_expert=(float(lr_expert) if lr_expert is not None else None),
                lr_muon=(float(lr_muon) if lr_muon is not None else None),
                grad_norm=(float(out['grad_norm']) if out.get('grad_norm') is not None else None),
                tokens_seen=float(tokens_seen),
                tokens_per_s_gpu=out.get('tps'),
                ms_per_step=out.get('ms_per_step'),
                tflops=out.get('tflops'),
                loader_wait_ms=loader_wait_ms,
                memory_current_alloc_gib=out.get('alloc_gib'),
                memory_max_alloc_gib=out.get('max_alloc_gib'),
            )
            for group, stats in signal_stats.items():
                ctx.writer.log_train_signal_group(
                    step,
                    group,
                    grad_norm=stats.get('grad_norm'),
                    param_norm=stats.get('param_norm'),
                    grad_to_param=stats.get('grad_to_param'),
                    param_count=stats.get('param_count'),
                    pre_param_norm=stats.get('pre_param_norm'),
                    update_norm=stats.get('update_norm'),
                    update_to_pre_param=stats.get('update_to_pre_param'),
                    optimizer_update_norm=stats.get('optimizer_update_norm'),
                    optimizer_update_to_pre_param=stats.get('optimizer_update_to_pre_param'),
                    weight_decay_update_norm=stats.get('weight_decay_update_norm'),
                )
            # Persist per-dtype achieved TFLOP/s (rank 0 only)
            items = []
            if tflops_fp8_ach is not None:
                items.append(("efficiency/fp8_tflops", float(tflops_fp8_ach)))
            if tflops_bf16_ach is not None:
                items.append(("efficiency/bf16_tflops", float(tflops_bf16_ach)))
            if items:
                ctx.writer.insert_many(step, items)
        except Exception:
            pass

    # Router metrics (per-layer + aggregates)
    if ctx is not None and ctx.writer is not None and rank == 0:
        try:
            per, agg = collect_router_stats(model)
            for item in per:
                cv = item['cv']
                ent = item['entropy']
                mx = item['max_load']
                bmin = item['bias_min']
                bmax = item['bias_max']
                ctx.writer.log_router_layer(
                    step,
                    item['layer'],
                    cv=cv,
                    entropy=ent,
                    experts_active=item.get('experts_active'),
                    bias_range=(bmax - bmin) if (bmin is not None and bmax is not None) else None,
                    max_load=mx,
                    min_load=item.get('min_load'),
                    p10_load=item.get('p10_load'),
                    p50_load=item.get('p50_load'),
                    p90_load=item.get('p90_load'),
                    importance_cv=item.get('importance_cv'),
                    importance_entropy=item.get('importance_entropy'),
                    max_importance=item.get('max_importance'),
                    min_importance=item.get('min_importance'),
                    p10_importance=item.get('p10_importance'),
                    p50_importance=item.get('p50_importance'),
                    p90_importance=item.get('p90_importance'),
                )
            if any(v is not None for v in agg.values()):
                ctx.writer.log_router_agg(
                    step,
                    mean_cv=agg['mean_cv'],
                    std_cv=agg['std_cv'],
                    mean_entropy=agg['mean_entropy'],
                    min_entropy=agg['min_entropy'],
                    dead_experts_count=agg['dead_experts_count'],
                    experts_active_mean=agg.get('experts_active_mean'),
                    mean_min_load=agg.get('mean_min_load'),
                    mean_p10_load=agg.get('mean_p10_load'),
                    mean_p50_load=agg.get('mean_p50_load'),
                    mean_p90_load=agg.get('mean_p90_load'),
                    mean_max_load=agg.get('mean_max_load'),
                    mean_importance_cv=agg.get('mean_importance_cv'),
                    mean_importance_entropy=agg.get('mean_importance_entropy'),
                    mean_min_importance=agg.get('mean_min_importance'),
                    mean_p10_importance=agg.get('mean_p10_importance'),
                    mean_p50_importance=agg.get('mean_p50_importance'),
                    mean_p90_importance=agg.get('mean_p90_importance'),
                    mean_max_importance=agg.get('mean_max_importance'),
                )
        except Exception:
            pass

    # Flush timers for this rank, then flush step-level segment and comm metrics, if any (all ranks, per-rank tags)
    _flush_timers_into_segments()
    if ctx is not None and ctx.writer is not None:
        try:
            seg_items = _SEG_AGG.snapshot_and_reset()
            items: list[tuple[str, float]] = []
            # Per-rank throughput for debugging
            if out.get('tps') is not None:
                items.append((f"throughput/r{rank}/tokens_per_s", float(out['tps'])))
            # Prefix tags with r{rank}/ for per-rank visibility
            for tag, val in seg_items:
                if tag.startswith('time_ms/'):
                    items.append((f"time_ms/r{rank}/" + tag[len('time_ms/'):], val))
                elif tag.startswith('comm/'):
                    items.append((f"comm/r{rank}/" + tag[len('comm/'):], val))
            # NVFP4 nibble flip fraction (per-rank, canary MoE layers)
            if _NVFP4_FLIP is not None:
                for (layer_idx, name), frac in _NVFP4_FLIP.flush_means().items():
                    items.append((f"quant/r{rank}/nvfp4/layer_{layer_idx:02d}/{name}_nibble_flip_frac", float(frac)))
                    items.append((f"quant/r{rank}/nvfp4/layer_{layer_idx:02d}/{name}_resonance_error", float(frac - Nvfp4NibbleFlipTracker.TARGET_FRAC)))
            if items:
                ctx.writer.insert_many(step, items)
        except Exception:
            pass

    # GPU snapshot ~1 Hz
    if ctx is not None and ctx.writer is not None and ctx.gpu_snapshot_fn is not None:
        now = time.perf_counter()
        if (now - ctx.last_gpu_log) >= log_every_seconds:
            try:
                snap = ctx.gpu_snapshot_fn()
                ctx.writer.log_gpu_snapshot(step, snap)
                ctx.last_gpu_log = now
            except Exception:
                pass

    # Console router summary
    if is_rank0:
        try:
            log_router_stats(model, print_out)
        except Exception:
            pass

    # Flush parquet for concurrent readers (rank 0 only, at log cadence)
    if is_rank0 and ctx is not None and ctx.writer is not None:
        ctx.writer.flush_parquet(step)

    return out
