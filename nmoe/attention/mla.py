import os
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.profiler import record_function

from nmoe.attention.rope import rotate_pe
from nmoe.config import Config
from nmoe.norm import RMSNorm


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise RuntimeError(msg)


def _sm100_only(device: torch.device) -> None:
  _require(torch.cuda.is_available(), "MLA requires CUDA (B200 / SM100).")
  major, minor = torch.cuda.get_device_capability(device)
  _require(major == 10, f"MLA requires SM100 (B200). Got compute capability {major}.{minor}.")


def _nvtx(tag: str):
  if os.getenv('NMOE_NVTX', '0') not in ('1', 'true', 'True'):
    return nullcontext()
  if torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx') and hasattr(torch.cuda.nvtx, 'range'):
    return torch.cuda.nvtx.range(tag)
  return nullcontext()


# Module-level workspace cache for MLA backward pass.
#
# Design goals:
# - Avoid per-backward allocation churn (token/s stability, allocator pressure)
# - Bounded growth: one buffer per device, grown in-place as needed
# - Scratch-only: not part of module state / checkpoints
_mla_workspace_cache: dict[torch.device, torch.Tensor] = {}


def _get_mla_workspace(device: torch.device, workspace_bytes: int) -> torch.Tensor:
  """Get a cached workspace buffer for MLA backward.

  Returns a `torch.uint8` CUDA tensor with `numel() >= workspace_bytes`.
  """
  buf = _mla_workspace_cache.get(device)
  if buf is None or buf.numel() < workspace_bytes:
    buf = torch.empty((workspace_bytes,), device=device, dtype=torch.uint8)
    _mla_workspace_cache[device] = buf
  return buf


class _MlaFa4FwdFlashMlaBwd(torch.autograd.Function):
  @staticmethod
  def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    _sm100_only(q.device)
    _require(q.is_cuda and k.is_cuda and v.is_cuda, "MLA FA4+FlashMLA requires CUDA tensors.")
    _require(q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16,
             "MLA FA4+FlashMLA requires BF16 inputs.")
    _require(q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "Expected q/k/v with shape [B, S, H, D].")

    bsz, seqlen, n_heads, d_qk = q.shape
    _require(k.shape == (bsz, seqlen, n_heads, d_qk), "k must match q shape.")
    d_v = v.shape[-1]
    _require(v.shape == (bsz, seqlen, n_heads, d_v), "v must have shape [B, S, H, Dv].")
    _require(d_qk == 192 and d_v == 128, f"Only (d_qk, d_v) = (192, 128) is supported. Got ({d_qk}, {d_v}).")

    # Hard requirement: do not proceed without FA4 forward + FlashMLA backward.
    # Environment contract: `third_party/flash_attn` is on PYTHONPATH (so `flash_attn.cute` is importable),
    # and `nmoe.csrc.flashmla_sm100` is built.
    from flash_attn.cute.interface import _flash_attn_fwd  # type: ignore
    from nmoe.csrc import flashmla_sm100 as _flashmla  # type: ignore

    total = bsz * seqlen
    q_ = q.reshape(total, n_heads, d_qk).contiguous()
    k_ = k.reshape(total, n_heads, d_qk).contiguous()
    v_ = v.reshape(total, n_heads, d_v).contiguous()

    cu = torch.arange(0, (bsz + 1) * seqlen, step=seqlen, device=q.device, dtype=torch.int32)

    with _nvtx("attn/fa4_fwd"):
      out, lse = _flash_attn_fwd(
          q_,
          k_,
          v_,
          cu_seqlens_q=cu,
          cu_seqlens_k=cu,
          softmax_scale=float(softmax_scale),
          causal=True,
          return_lse=True,
      )

    # FlashMLA expects lse as [total, H] float32 with stride(0) == 1.
    lse_t = lse.T

    ctx.save_for_backward(q_, k_, v_, out, lse_t, cu)
    ctx.softmax_scale = float(softmax_scale)
    ctx.seqlen = int(seqlen)
    ctx._flashmla = _flashmla
    return out.reshape(bsz, seqlen, n_heads, d_v)

  @staticmethod
  def backward(ctx, d_out: torch.Tensor):
    q, k, v, out, lse_t, cu = ctx.saved_tensors
    softmax_scale = ctx.softmax_scale
    seqlen = ctx.seqlen
    flashmla = ctx._flashmla

    bsz = cu.numel() - 1
    total, n_heads, d_qk = q.shape
    d_v = v.shape[-1]

    d_o = d_out.reshape(total, n_heads, d_v).contiguous()

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    max_seqlen_aligned = ((seqlen + 7) // 8) * 8
    workspace_bytes = 0
    workspace_bytes += 4 * bsz * max_seqlen_aligned * n_heads * d_qk  # dQ_acc
    workspace_bytes += 4 * max_seqlen_aligned * bsz * n_heads * 2  # sum_OdO + scaled_lse
    # Use cached workspace to avoid allocation churn
    workspace = _get_mla_workspace(q.device, workspace_bytes)

    with _nvtx("attn/flashmla_bwd"):
      flashmla.dense_prefill_bwd(
          workspace,
          d_o,
          q,
          k,
          v,
          out,
          lse_t,
          cu,
          cu,
          dq,
          dk,
          dv,
          1,  # causal
          softmax_scale,
          seqlen,
          seqlen,
          True,  # is_varlen
      )

    return dq.reshape(bsz, seqlen, n_heads, d_qk), dk.reshape(bsz, seqlen, n_heads, d_qk), dv.reshape(bsz, seqlen, n_heads, d_v), None


class MLA(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.dim = config.dim
    self.n_heads = config.n_heads
    self.q_lora_rank = config.q_lora_rank
    self.kv_lora_rank = config.kv_lora_rank
    self.qk_nope_head_dim = config.qk_nope_head_dim
    self.qk_rope_head_dim = config.qk_rope_head_dim
    self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    self.v_head_dim = config.v_head_dim
    self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=torch.bfloat16)
    self.q_norm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
    self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False, dtype=torch.bfloat16)
    self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False, dtype=torch.bfloat16)
    self.kv_norm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
    self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False, dtype=torch.bfloat16)
    self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False, dtype=torch.bfloat16)
    self.softmax_scale = self.qk_head_dim ** -0.5

  def init_weights(self, init_std: float = 0.02):
    for proj in [self.wq_a, self.wq_b, self.wkv_a, self.wkv_b]:
      nn.init.trunc_normal_(proj.weight, mean=0.0, std=0.02)
    nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
    self.q_norm.weight.data.fill_(1.0)
    self.kv_norm.weight.data.fill_(1.0)

  @record_function("attn")
  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    bsz, seqlen, _ = x.size()
    q = self.wq_b(self.q_norm(self.wq_a(x)))
    q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    q_pe = rotate_pe(q_pe, cos, sin)
    q[..., self.qk_nope_head_dim:].copy_(q_pe)
    kv = self.wkv_a(x)
    kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = rotate_pe(k_pe.unsqueeze(2), cos, sin)
    kv = self.wkv_b(self.kv_norm(kv))
    kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k = torch.empty((bsz, seqlen, self.n_heads, self.qk_head_dim), device=x.device, dtype=k_nope.dtype)
    k[..., :self.qk_nope_head_dim].copy_(k_nope)
    k[..., self.qk_nope_head_dim:].copy_(k_pe.expand(-1, -1, self.n_heads, -1))
    with record_function("attn.kernel[mla]"):
      output = _MlaFa4FwdFlashMlaBwd.apply(q, k, v, self.softmax_scale)
    output = output.contiguous().view(bsz, seqlen, self.n_heads * self.v_head_dim)
    return self.wo(output)
