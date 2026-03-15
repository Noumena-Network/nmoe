"""MoE computation: grouped GEMM and fused autograd functions.

This module contains:
- expert(): BF16 grouped GEMM for expert MLP
- _MoEBf16Fused: Autograd function for BF16 MoE forward/backward
- _MoEBlockscaledFused: Autograd function for FP8/NVFP4 MoE forward/backward

The dispatch/combine infrastructure is in rdep.py.
The Router and MoE nn.Module classes are in model.py.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.distributed as dist

from nmoe.csrc import rdep as _C

if TYPE_CHECKING:
  from nmoe.rdep import Rdep


def expert(
  Xe_pad: torch.Tensor,
  W1: torch.Tensor,
  W3: torch.Tensor,
  W2: torch.Tensor,
  offs_pad: torch.Tensor,
  activation: str = "swiglu",
) -> torch.Tensor:
  """Expert MLP with configurable activation.

  Activations:
    - swiglu: Y = (SiLU(X @ W1) * (X @ W3)) @ W2
    - relu_squared: Y = ReLU(X @ W1)² @ W2
    - squared_reglu: Y = (ReLU(X @ W1)² * (X @ W3)) @ W2

  BF16 path using torch._grouped_mm.

  Args:
    Xe_pad: [M_pad, H] pre-padded BF16 input from rdep.dispatch_sorted
    W1, W3: [E, H, Dff] gate/up weights
    W2: [E, Dff, H] down weight
    offs_pad: [E] cumulative padded offsets from rdep
    activation: activation function name

  Returns:
    [M_pad, H] BF16 output (caller uses dest to select valid rows)
  """
  if Xe_pad.size(0) == 0:
    return Xe_pad

  H1 = torch._grouped_mm(Xe_pad, W1, offs=offs_pad)

  if activation == "swiglu":
    H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)
    A = F.silu(H1) * H3
  elif activation == "relu_squared":
    A = F.relu(H1) ** 2
  elif activation == "squared_reglu":
    H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)
    A = F.relu(H1) ** 2 * H3
  else:
    raise ValueError(f"Unknown activation: {activation}")

  return torch._grouped_mm(A, W2, offs=offs_pad)


def _decode_e8m0(scale_bytes: torch.Tensor) -> torch.Tensor:
  scale_i32 = scale_bytes.to(dtype=torch.int32)
  return torch.ldexp(torch.ones_like(scale_i32, dtype=torch.float32), scale_i32 - 127)


def _decode_nvfp4_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
  nib_i32 = nibbles.to(dtype=torch.int32)
  sign = 1.0 - 2.0 * ((nib_i32 >> 3) & 0x1).to(dtype=torch.float32)
  exp = (nib_i32 >> 1) & 0x3
  mant = (nib_i32 & 0x1).to(dtype=torch.float32)
  normal = torch.ldexp(1.0 + 0.5 * mant, exp - 1)
  subnormal = mant * 0.5
  return sign * torch.where(exp == 0, subnormal, normal)


def _dequant_fp8(q: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
  q_rows = q.squeeze(-1).to(dtype=torch.float32)
  scales = _decode_e8m0(scale_bytes.squeeze(-1)).repeat_interleave(32, dim=1)
  return (q_rows * scales[:, : q_rows.shape[1]]).to(dtype=torch.bfloat16)


def _dequant_nvfp4(q: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
  q_u8 = q.squeeze(-1).contiguous()
  lo = q_u8[:, 0::2].to(dtype=torch.int32)
  hi = q_u8[:, 1::2].to(dtype=torch.int32)
  packed = lo | (hi << 8)
  nibbles = torch.stack([
    packed & 0xF,
    (packed >> 4) & 0xF,
    (packed >> 8) & 0xF,
    (packed >> 12) & 0xF,
  ], dim=-1)
  values = _decode_nvfp4_nibbles(nibbles).reshape(q_u8.shape[0], q_u8.shape[1] * 2)
  scales = _decode_e8m0(scale_bytes.squeeze(-1)).repeat_interleave(32, dim=1)
  return (values * scales[:, : values.shape[1]]).to(dtype=torch.bfloat16)


def _quant_dequant_rows(x: torch.Tensor, profile: str) -> torch.Tensor:
  from nmoe.quant import quantize_fp8, quantize_nvfp4

  if profile == 'fp8':
    q, sfa = quantize_fp8(x)
    return _dequant_fp8(q, sfa)
  if profile == 'nvfp4':
    q, sfa = quantize_nvfp4(x)
    return _dequant_nvfp4(q, sfa)
  raise ValueError(f"unsupported profile: {profile}")


def _quant_dequant_w13(W: torch.Tensor, profile: str) -> torch.Tensor:
  E, H, Dff = W.shape
  rows = W.transpose(1, 2).contiguous().view(E * Dff, H)
  dq = _quant_dequant_rows(rows, profile)
  return dq.view(E, Dff, H).transpose(1, 2).contiguous()


def _quant_dequant_w2(W2: torch.Tensor, profile: str) -> torch.Tensor:
  E, Dff, H = W2.shape
  rows = W2.transpose(1, 2).contiguous().view(E * H, Dff)
  dq = _quant_dequant_rows(rows, profile)
  return dq.view(E, H, Dff).transpose(1, 2).contiguous()


def _blockscaled_ablation_profiles(rdep_profile: str, forward_ablation: str) -> tuple[str | None, str | None, str | None, str | None]:
  if forward_ablation == 'off':
    return None, None, None, None
  if forward_ablation == 'w13_bf16':
    return rdep_profile, None, rdep_profile, rdep_profile
  if forward_ablation == 'stage1_bf16':
    return None, None, None, rdep_profile
  if forward_ablation == 'full_bf16':
    return None, None, None, None
  if forward_ablation == 'w13_fp8':
    return rdep_profile, 'fp8', rdep_profile, rdep_profile
  if forward_ablation == 'w2_fp8':
    return rdep_profile, rdep_profile, rdep_profile, 'fp8'
  if forward_ablation == 'stage1_fp8':
    return 'fp8', 'fp8', 'fp8', rdep_profile
  if forward_ablation == 'full_fp8':
    return 'fp8', 'fp8', 'fp8', 'fp8'
  raise ValueError(f"unsupported blockscaled forward ablation: {forward_ablation}")


def _blockscaled_backward_profiles(rdep_profile: str, forward_ablation: str) -> tuple[str | None, str | None, str | None, str | None]:
  # The default blockscaled backward should follow the same fake-quant contract
  # as the live forward path instead of silently recomputing a clean BF16 expert.
  if forward_ablation == 'off':
    return rdep_profile, rdep_profile, rdep_profile, rdep_profile
  return _blockscaled_ablation_profiles(rdep_profile, forward_ablation)


def _materialize_blockscaled_ablation_operands(
  Xe_pad: torch.Tensor,
  W1: torch.Tensor,
  W3: torch.Tensor,
  W2: torch.Tensor,
  *,
  input_profile: str | None,
  w13_profile: str | None,
  w2_profile: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  Xe_use = Xe_pad
  if input_profile is not None:
    Xe_use = _quant_dequant_rows(Xe_pad.contiguous(), input_profile)

  W1_use = _quant_dequant_w13(W1.detach(), w13_profile) if w13_profile is not None else W1
  W3_use = _quant_dequant_w13(W3.detach(), w13_profile) if w13_profile is not None else W3
  W2_use = _quant_dequant_w2(W2.detach(), w2_profile) if w2_profile is not None else W2
  return Xe_use, W1_use, W3_use, W2_use


def _apply_blockscaled_postact_profile(A: torch.Tensor, postact_profile: str | None) -> torch.Tensor:
  if postact_profile is None:
    return A
  return _quant_dequant_rows(A.contiguous(), postact_profile)


def _expert_blockscaled_ablation(
  Xe_pad: torch.Tensor,
  W1: torch.Tensor,
  W3: torch.Tensor,
  W2: torch.Tensor,
  offs_pad: torch.Tensor,
  activation: str,
  *,
  input_profile: str | None,
  w13_profile: str | None,
  postact_profile: str | None,
  w2_profile: str | None,
) -> torch.Tensor:
  Xe_use, W1_use, W3_use, W2_use = _materialize_blockscaled_ablation_operands(
    Xe_pad, W1, W3, W2,
    input_profile=input_profile,
    w13_profile=w13_profile,
    w2_profile=w2_profile,
  )

  H1 = torch._grouped_mm(Xe_use, W1_use, offs=offs_pad)

  if activation == "swiglu":
    H3 = torch._grouped_mm(Xe_use, W3_use, offs=offs_pad)
    A = F.silu(H1) * H3
  elif activation == "relu_squared":
    A = F.relu(H1) ** 2
  elif activation == "squared_reglu":
    H3 = torch._grouped_mm(Xe_use, W3_use, offs=offs_pad)
    A = F.relu(H1) ** 2 * H3
  else:
    raise ValueError(f"Unknown activation: {activation}")

  A = _apply_blockscaled_postact_profile(A, postact_profile)
  return torch._grouped_mm(A, W2_use, offs=offs_pad)


def _forward_blockscaled_ablation(
  rdep: Rdep,
  x: torch.Tensor,
  eid: torch.Tensor,
  gates_fp32: torch.Tensor,
  W1: torch.Tensor,
  W3: torch.Tensor,
  W2: torch.Tensor,
  activation: str,
  forward_ablation: str,
) -> torch.Tensor:
  device = x.device
  stream = torch.cuda.current_stream(device)
  T, H = x.shape
  K = int(eid.shape[1])
  E = int(rdep.n_local)
  is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

  offs_pad = torch.empty(E, device=device, dtype=torch.int32)
  M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()
  align = 128

  M_recv = _C.dispatch_meta_bf16(
    x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
    int(T), int(K), align,
    offs_pad.data_ptr(), M_host.data_ptr(),
    stream,
  )

  out_f32 = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
  if M_recv <= 0:
    if is_dist:
      dummy_ye_pad = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
      _C.return_scatter_from_pad_bf16(dummy_ye_pad.data_ptr(), out_f32.data_ptr(), 0, int(T), int(K), stream)
    return out_f32.to(dtype=torch.bfloat16)

  max_pad = (int(M_recv) + int(E) * (align - 1) + (align - 1)) // align * align
  offs_pad[-1] = int(max_pad)

  Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
  _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

  if forward_ablation in ('w2_fp8', 'stage1_fp8', 'full_fp8'):
    from nmoe.quant import quantize_fp8, quantize_nvfp4
    from nmoe.blockscaled.grouped import _swizzle_sf_to_mma, expert_blockscaled, quantize_weights

    if forward_ablation == 'w2_fp8':
      if rdep.profile == 'fp8':
        Xe_q_nat, Xe_sf_nat = quantize_fp8(Xe_pad.contiguous())
      elif rdep.profile == 'nvfp4':
        Xe_q_nat, Xe_sf_nat = quantize_nvfp4(Xe_pad.contiguous())
      else:
        raise ValueError(f"unsupported RDEP profile: {rdep.profile}")
      Xe_q_use, Xe_sf_use = Xe_q_nat, _swizzle_sf_to_mma(Xe_sf_nat)
      w13_profile = rdep.profile
      w2_profile = 'fp8'
    else:
      Xe_q_fp8, Xe_sf_fp8 = quantize_fp8(Xe_pad.contiguous())
      Xe_q_use, Xe_sf_use = Xe_q_fp8, _swizzle_sf_to_mma(Xe_sf_fp8)
      w13_profile = 'fp8'
      w2_profile = 'fp8' if forward_ablation == 'full_fp8' else rdep.profile
    W_cache_fp8 = quantize_weights(W1, W3, W2, w13_profile=w13_profile, w2_profile=w2_profile)
    Ye_pad = expert_blockscaled(
      Xe_q_use,
      Xe_sf_use,
      W_cache_fp8,
      offs_pad,
      capacity_rows=int(rdep.capacity),
    )
  else:
    input_profile, w13_profile, postact_profile, w2_profile = _blockscaled_ablation_profiles(rdep.profile, forward_ablation)
    Ye_pad = _expert_blockscaled_ablation(
      Xe_pad, W1, W3, W2, offs_pad, activation,
      input_profile=input_profile,
      w13_profile=w13_profile,
      postact_profile=postact_profile,
      w2_profile=w2_profile,
    )

  _C.return_scatter_from_pad_bf16(Ye_pad.data_ptr(), out_f32.data_ptr(), int(M_recv), int(T), int(K), stream)
  return out_f32.to(dtype=torch.bfloat16)


class _MoEBf16Fused(torch.autograd.Function):
  @staticmethod
  def forward(ctx, rdep: Rdep, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
              W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor, activation: str = "swiglu") -> torch.Tensor:
    device = x.device
    stream = torch.cuda.current_stream(device)

    x = x.contiguous().bfloat16()
    eid = eid.contiguous().int()
    gates = gates.contiguous().bfloat16()
    gates_fp32 = gates.detach().float()

    T, H = x.shape
    K = int(eid.shape[1])
    is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if is_dist:
      need = int(T) * int(K) * int(rdep.world)
      if rdep.capacity < need:
        raise RuntimeError(
          f"[RDEP] capacity too small: capacity={rdep.capacity:,} need>={need:,} (T={T:,} K={K} world={rdep.world}). "
          "Set capacity to worst-case T*K*world (no silent truncation)."
        )

    offs_pad = torch.empty(rdep.n_local, device=device, dtype=torch.int32)
    # dispatch_meta_bf16 uses this host int32 (pinned) as scratch to read back M_recv.
    M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()

    # BF16 fused path uses align=128 for consistent GEMM padding
    align = 128

    M_recv = _C.dispatch_meta_bf16(
      x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
      int(T), int(K), align,
      offs_pad.data_ptr(), M_host.data_ptr(),
      stream,
    )

    out_f32 = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
    if M_recv <= 0:
      # DeepEP collectiveness: every rank must participate in return_scatter even if it sends nothing,
      # because other ranks may be returning outputs for *our* local tokens, and IPC barriers must match.
      if is_dist:
        dummy_ye_pad = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        _C.return_scatter_from_pad_bf16(
          dummy_ye_pad.data_ptr(),
          out_f32.data_ptr(),
          0, int(T), int(K),
          stream,
        )
      ctx.rdep = rdep
      ctx.activation = activation
      ctx.save_for_backward(x, eid, gates, W1, W3, W2)
      return out_f32.to(dtype=torch.bfloat16)

    # Avoid a second host sync for exact M_pad:
    # - Exact padded total is sum_e align_up(cnt_e, align) and depends on routing.
    # - For BF16 grouped GEMM we only need per-expert offsets to be aligned.
    # - Over-allocate to a deterministic upper bound and extend the *last* expert's
    #   padded region. Extra rows are zeroed and therefore compute to zero.
    max_pad = (int(M_recv) + int(rdep.n_local) * (align - 1) + (align - 1)) // align * align
    # Ensure the last expert's padded segment reaches max_pad (keeps per-expert alignment).
    offs_pad[-1] = int(max_pad)

    Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
    _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

    Ye_pad = expert(Xe_pad, W1, W3, W2, offs_pad, activation)
    _C.return_scatter_from_pad_bf16(
      Ye_pad.data_ptr(),
      out_f32.data_ptr(),
      int(M_recv), int(T), int(K),
      stream,
    )

    ctx.rdep = rdep
    ctx.activation = activation
    ctx.save_for_backward(x, eid, gates, W1, W3, W2)
    return out_f32.to(dtype=torch.bfloat16)

  @staticmethod
  def backward(ctx, dOut: torch.Tensor):
    x, eid, gates, W1, W3, W2 = ctx.saved_tensors
    rdep: Rdep = ctx.rdep
    activation: str = ctx.activation
    device = x.device
    stream = torch.cuda.current_stream(device)

    x = x.contiguous().bfloat16()
    eid = eid.contiguous().int()
    gates = gates.contiguous().bfloat16()
    gates_fp32 = gates.detach().float()
    dOut = dOut.contiguous().bfloat16()

    T, H = x.shape
    K = int(eid.shape[1])
    is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if is_dist:
      need = int(T) * int(K) * int(dist.get_world_size())
      if rdep.capacity < need:
        raise RuntimeError(
          f"[RDEP] capacity too small: capacity={rdep.capacity:,} need>={need:,} (T={T:,} K={K} world={dist.get_world_size()}). "
          "Set capacity to worst-case T*K*world (no silent truncation)."
        )

    offs_pad = torch.empty(int(W1.size(0)), device=device, dtype=torch.int32)
    M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()

    # BF16 fused path uses align=128 for consistent GEMM padding
    align = 128

    M_recv = _C.dispatch_meta_bf16(
      x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
      int(T), int(K), align,
      offs_pad.data_ptr(), M_host.data_ptr(),
      stream,
    )

    if M_recv <= 0:
      dW1 = torch.zeros_like(W1)
      dW3 = torch.zeros_like(W3)
      dW2 = torch.zeros_like(W2)
      dX = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)

      # DeepEP collectiveness: still run distributed gather/scatter so we:
      # (1) send dY for our local tokens, (2) receive dGate/dX from other ranks.
      if is_dist:
        dGates_tk_f32 = torch.zeros(int(T), int(K), device=device, dtype=torch.float32)
        dummy_row_id = torch.empty(1, device=device, dtype=torch.int64)
        dummy_gate_sorted = torch.empty(1, device=device, dtype=torch.float32)
        dummy_ye_sorted = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        dummy_dye_sorted = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        dummy_dgate_sorted = torch.empty(1, device=device, dtype=torch.float32)
        _C.gather_dy_dist_bf16(
          dOut.data_ptr(),
          eid.data_ptr(),
          dummy_ye_sorted.data_ptr(),
          dummy_row_id.data_ptr(),
          dummy_gate_sorted.data_ptr(),
          dummy_dye_sorted.data_ptr(),
          dummy_dgate_sorted.data_ptr(),
          dGates_tk_f32.data_ptr(),
          0, int(T), int(H), int(K),
          stream,
        )
        dummy_dxe_sorted = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        _C.scatter_dx_dist_bf16(
          dummy_dxe_sorted.data_ptr(),
          dummy_row_id.data_ptr(),
          dX.data_ptr(),
          0, int(T), int(H), int(K),
          stream,
        )
        dGates = dGates_tk_f32.to(dtype=torch.bfloat16)
      else:
        dGates = torch.zeros(int(T), int(K), device=device, dtype=torch.bfloat16)

      return None, dX, None, dGates, dW1, dW3, dW2, None

    max_pad = (int(M_recv) + int(offs_pad.numel()) * (align - 1) + (align - 1)) // align * align
    offs_pad[-1] = int(max_pad)

    Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
    _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

    row_id = torch.empty(int(M_recv), device=device, dtype=torch.int64)
    gate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
    _C.gather_meta_sorted_bf16(row_id.data_ptr(), gate_sorted.data_ptr(), int(M_recv), stream)

    with torch.enable_grad():
      Xe_pad = Xe_pad.requires_grad_(True)
      Ye_pad = expert(Xe_pad, W1, W3, W2, offs_pad, activation)

    dYe_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
    dGate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
    dGates_tk_f32 = torch.zeros(int(T), int(K), device=device, dtype=torch.float32)

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
      _C.gather_dy_dist_bf16(
        dOut.data_ptr(),
        eid.data_ptr(),
        Ye_pad.detach().data_ptr(),
        row_id.data_ptr(),
        gate_sorted.data_ptr(),
        dYe_sorted.data_ptr(),
        dGate_sorted.data_ptr(),
        dGates_tk_f32.data_ptr(),
        int(M_recv), int(T), int(H), int(K),
        stream,
      )
    else:
      _C.gather_dy_bf16(
        dOut.data_ptr(),
        Ye_pad.detach().data_ptr(),
        row_id.data_ptr(),
        gate_sorted.data_ptr(),
        dYe_sorted.data_ptr(),
        dGate_sorted.data_ptr(),
        int(M_recv), int(T), int(H), int(K),
        stream,
      )
      _C.scatter_gate_bf16(
        dGate_sorted.data_ptr(),
        row_id.data_ptr(),
        dGates_tk_f32.data_ptr(),
        int(M_recv), int(T), int(K),
        stream,
      )

    dYe_pad = torch.zeros(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
    _C.scatter_sorted_to_pad_bf16(dYe_sorted.data_ptr(), dYe_pad.data_ptr(), int(M_recv), int(H), stream)

    dXe_pad, dW1, dW3, dW2 = torch.autograd.grad(
      outputs=Ye_pad,
      inputs=(Xe_pad, W1, W3, W2),
      grad_outputs=dYe_pad,
      retain_graph=False,
      create_graph=False,
      allow_unused=False,
    )

    dX = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
    dXe_pad_bf16 = dXe_pad.to(dtype=torch.bfloat16)

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
      dXe_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
      _C.gather_from_pad_bf16(dXe_pad_bf16.data_ptr(), dXe_sorted.data_ptr(), int(M_recv), int(H), stream)
      _C.scatter_dx_dist_bf16(
        dXe_sorted.data_ptr(),
        row_id.data_ptr(),
        dX.data_ptr(),
        int(M_recv), int(T), int(H), int(K),
        stream,
      )
    else:
      _C.scatter_dx_bf16_internal(
        dXe_pad_bf16.data_ptr(),
        row_id.data_ptr(),
        dX.data_ptr(),
        int(M_recv), int(T), int(H), int(K),
        stream,
      )

    dGates = dGates_tk_f32.to(dtype=torch.bfloat16)
    return None, dX, None, dGates, dW1, dW3, dW2, None


class _MoEBlockscaledFused(torch.autograd.Function):
  @staticmethod
  def forward(ctx, rdep: Rdep, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
              W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor, W_cache,
              activation: str = "swiglu", forward_ablation: str = "off") -> torch.Tensor:
    device = x.device
    stream = torch.cuda.current_stream(device)

    x = x.contiguous().bfloat16()
    eid = eid.contiguous().int()
    gates = gates.contiguous().bfloat16()
    gates_fp32 = gates.detach().float()

    T, H = x.shape
    K = int(eid.shape[1])
    E = int(rdep.n_local)
    is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if forward_ablation not in ('off', 'w13_bf16', 'stage1_bf16', 'full_bf16', 'w13_fp8', 'w2_fp8', 'stage1_fp8', 'full_fp8'):
      raise ValueError(f"unsupported blockscaled forward ablation: {forward_ablation}")
    if is_dist:
      need = int(T) * int(K) * int(rdep.world)
      if rdep.capacity < need:
        raise RuntimeError(
          f"[RDEP] capacity too small: capacity={rdep.capacity:,} need>={need:,} (T={T:,} K={K} world={rdep.world}). "
          "Set capacity to worst-case T*K*world (no silent truncation)."
        )

    if forward_ablation != 'off':
      out = _forward_blockscaled_ablation(rdep, x, eid, gates_fp32, W1, W3, W2, activation, forward_ablation)
      ctx.rdep = rdep
      ctx.W_cache = W_cache
      ctx.activation = activation
      ctx.forward_ablation = forward_ablation
      ctx.T = int(T)
      ctx.H = int(H)
      ctx.K = int(K)
      ctx.save_for_backward(x, eid, gates, W1, W3, W2)
      return out

    # Option A: Use BF16 dispatch + local quantization
    # This ensures Xe_pad (BF16) is available for backward STE
    offs_pad = torch.empty(E, device=device, dtype=torch.int32)
    M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()
    align = 128  # Required for blockscaled SF swizzle

    M_recv = _C.dispatch_meta_blockscaled(
      x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
      int(T), int(K),
      offs_pad.data_ptr(), M_host.data_ptr(),
      stream,
    )

    out_f32 = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
    if M_recv <= 0:
      # DeepEP collectiveness: every rank must participate in return_scatter
      if is_dist:
        dummy_ye_pad = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        _C.return_scatter_from_pad_blockscaled(dummy_ye_pad.data_ptr(), out_f32.data_ptr(), 0, int(T), int(K), stream)
      ctx.rdep = rdep
      ctx.W_cache = W_cache
      ctx.activation = activation
      ctx.forward_ablation = forward_ablation
      ctx.T = int(T)
      ctx.H = int(H)
      ctx.K = int(K)
      ctx.save_for_backward(x, eid, gates, W1, W3, W2)
      return out_f32.to(dtype=torch.bfloat16)

    M_pad = int(M_host.item())

    # Gather blockscaled activations into padded layout (quantized + packed SF)
    pack_factor = 2 if rdep.profile == 'fp8' else 4
    Hp = H // pack_factor
    sf_k = H // 32
    sf_k_pad = ((sf_k + 3) // 4) * 4
    Xe_q = torch.empty(int(M_pad), Hp, device=device, dtype=torch.uint16)
    Xe_sf = torch.empty(int(M_pad), sf_k_pad, device=device, dtype=torch.uint8)
    _C.gather_xe_blockscaled(Xe_q.data_ptr(), Xe_sf.data_ptr(), int(M_recv), int(M_pad), stream)

    # Expert compute (blockscaled)
    from nmoe.blockscaled.grouped import expert_blockscaled
    Ye_pad = expert_blockscaled(Xe_q, Xe_sf, W_cache, offs_pad, capacity_rows=int(rdep.capacity))

    _C.return_scatter_from_pad_blockscaled(
      Ye_pad.data_ptr(),
      out_f32.data_ptr(),
      int(M_recv), int(T), int(K),
      stream,
    )

    ctx.rdep = rdep
    ctx.W_cache = W_cache
    ctx.activation = activation
    ctx.forward_ablation = forward_ablation
    ctx.T = int(T)
    ctx.H = int(H)
    ctx.K = int(K)
    ctx.save_for_backward(x, eid, gates, W1, W3, W2)
    return out_f32.to(dtype=torch.bfloat16)

  @staticmethod
  def backward(ctx, dOut: torch.Tensor):
    x, eid, gates, W1, W3, W2 = ctx.saved_tensors
    rdep: Rdep = ctx.rdep
    W_cache = ctx.W_cache
    activation: str = ctx.activation
    forward_ablation: str = getattr(ctx, "forward_ablation", "off")

    device = dOut.device
    stream = torch.cuda.current_stream(device)

    dOut = dOut.contiguous().bfloat16()
    x = x.contiguous().bfloat16()
    eid = eid.contiguous().int()
    gates = gates.contiguous().bfloat16()
    gates_fp32 = gates.detach().float()

    T = int(ctx.T)
    H = int(ctx.H)
    K = int(ctx.K)
    E = int(rdep.n_local)
    is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if is_dist:
      need = int(T) * int(K) * int(dist.get_world_size())
      if rdep.capacity < need:
        raise RuntimeError(
          f"[RDEP] capacity too small: capacity={rdep.capacity:,} need>={need:,} (T={T:,} K={K} world={dist.get_world_size()}). "
          "Set capacity to worst-case T*K*world (no silent truncation)."
        )

    # Option A: Use BF16 dispatch to get correct Xe_pad from all ranks
    # This fixes the distributed bug where local x was used for remote rows
    offs_pad = torch.empty(E, device=device, dtype=torch.int32)
    M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()
    align = 128  # Required for blockscaled SF swizzle

    M_recv = _C.dispatch_meta_bf16(
      x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
      int(T), int(K), align,
      offs_pad.data_ptr(), M_host.data_ptr(),
      stream,
    )

    if M_recv <= 0:
      dW1 = torch.zeros_like(W1)
      dW3 = torch.zeros_like(W3)
      dW2 = torch.zeros_like(W2)
      dX = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)

      # DeepEP collectiveness: still run distributed gather/scatter
      if is_dist:
        dGates_tk_f32 = torch.zeros(int(T), int(K), device=device, dtype=torch.float32)
        dummy_row_id = torch.empty(1, device=device, dtype=torch.int64)
        dummy_gate_sorted = torch.empty(1, device=device, dtype=torch.float32)
        dummy_ye_sorted = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        dummy_dye_sorted = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        dummy_dgate_sorted = torch.empty(1, device=device, dtype=torch.float32)
        _C.gather_dy_dist_bf16(
          dOut.data_ptr(),
          eid.data_ptr(),
          dummy_ye_sorted.data_ptr(),
          dummy_row_id.data_ptr(),
          dummy_gate_sorted.data_ptr(),
          dummy_dye_sorted.data_ptr(),
          dummy_dgate_sorted.data_ptr(),
          dGates_tk_f32.data_ptr(),
          0, int(T), int(H), int(K),
          stream,
        )
        dummy_dxe_sorted = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        _C.scatter_dx_dist_bf16(
          dummy_dxe_sorted.data_ptr(),
          dummy_row_id.data_ptr(),
          dX.data_ptr(),
          0, int(T), int(H), int(K),
          stream,
        )
        dGates = dGates_tk_f32.to(dtype=torch.bfloat16)
      else:
        dGates = torch.zeros(int(T), int(K), device=device, dtype=torch.bfloat16)

      return None, dX, None, dGates, dW1, dW3, dW2, None, None, None

    # Compute max_pad and extend last expert's padded region
    max_pad = (int(M_recv) + E * (align - 1) + (align - 1)) // align * align
    offs_pad[-1] = int(max_pad)

    # Gather BF16 activations (correct from all source ranks via IPC buffer!)
    Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
    _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

    # Get row_id and gate_sorted for backward computation
    row_id = torch.empty(int(M_recv), device=device, dtype=torch.int64)
    gate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
    _C.gather_meta_sorted_bf16(row_id.data_ptr(), gate_sorted.data_ptr(), int(M_recv), stream)

    # SonicMoE optimization: compute dGate via ⟨A, dA'⟩ instead of ⟨dOut, Ye⟩.
    # This eliminates the expensive Ye_pad recomputation for dGate - we compute
    # dGate = ⟨A, dA⟩ directly from activations, not ⟨dOut, Ye⟩ from outputs.
    # A = post-SwiGLU activation, dA = dYe @ W2.T (ungated gradient)
    #
    # TODO(perf): The gather_dy kernels still compute dGate internally (dot product of Ye*dOut).
    # This is wasted compute (~negligible). To fully remove, modify CUDA kernels in rdep.cu.
    dYe_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
    dGate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
    dGates_tk_f32 = torch.zeros(int(T), int(K), device=device, dtype=torch.float32)

    is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if is_dist:
      # Distributed path: gather dYe across ranks with gate scaling (no dGate yet)
      _C.gather_dy_nogate_dist_bf16(
        dOut.data_ptr(),
        eid.data_ptr(),
        row_id.data_ptr(),
        gate_sorted.data_ptr(),
        dYe_sorted.data_ptr(),
        int(M_recv), int(T), int(H), int(K),
        stream,
      )
    else:
      # Single-GPU: gather dY with gate scaling (no dGate yet)
      _C.gather_dy_nogate_bf16(
        dOut.data_ptr(),
        row_id.data_ptr(),
        gate_sorted.data_ptr(),
        dYe_sorted.data_ptr(),
        int(M_recv), int(T), int(H), int(K),
        stream,
      )

    dYe_pad = torch.zeros(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
    _C.scatter_sorted_to_pad_bf16(
      dYe_sorted.data_ptr(),
      dYe_pad.data_ptr(),
      int(M_recv), int(H),
      stream,
    )

    offs_pinned = torch.empty(E, dtype=torch.int32, device='cpu', pin_memory=True)
    offs_pinned.copy_(offs_pad, non_blocking=True)
    copy_event = torch.cuda.Event()
    copy_event.record(stream)
    Dff = int(W2.size(1))

    input_profile, w13_profile, postact_profile, w2_profile = _blockscaled_backward_profiles(rdep.profile, forward_ablation)
    Xe_use, W1_use, W3_use, W2_use = _materialize_blockscaled_ablation_operands(
      Xe_pad, W1, W3, W2,
      input_profile=input_profile,
      w13_profile=w13_profile,
      w2_profile=w2_profile,
    )

    H1 = torch._grouped_mm(Xe_use, W1_use, offs=offs_pad)
    dA = torch._grouped_mm(dYe_pad, W2_use.transpose(1, 2), offs=offs_pad)

    # Activation-specific forward and backward
    if activation == "swiglu":
      H3 = torch._grouped_mm(Xe_use, W3_use, offs=offs_pad)
      A = torch.empty_like(H1)
      dH1 = torch.empty_like(H1)
      dH3 = torch.empty_like(H3)
      _C.swiglu_bwd_bf16(
        H1.data_ptr(), int(Dff),
        H3.data_ptr(), int(Dff),
        dA.data_ptr(), int(Dff),
        A.data_ptr(), int(Dff),
        dH1.data_ptr(), int(Dff),
        dH3.data_ptr(), int(Dff),
        int(max_pad), int(Dff),
        stream,
      )
      A = _apply_blockscaled_postact_profile(A, postact_profile)
    elif activation == "relu_squared":
      # A = relu(H1)²; dH1 = 2 * relu(H1) * dA
      relu_H1 = F.relu(H1)
      A = relu_H1 ** 2
      A = _apply_blockscaled_postact_profile(A, postact_profile)
      dH1 = 2 * relu_H1 * dA
      H3 = torch.empty(0, device=device, dtype=H1.dtype)  # unused
      dH3 = torch.empty(0, device=device, dtype=H1.dtype)  # unused
    elif activation == "squared_reglu":
      # A = relu(H1)² * H3; dH1 = 2 * relu(H1) * H3 * dA; dH3 = relu(H1)² * dA
      H3 = torch._grouped_mm(Xe_use, W3_use, offs=offs_pad)
      relu_H1 = F.relu(H1)
      relu_H1_sq = relu_H1 ** 2
      A = relu_H1_sq * H3
      A = _apply_blockscaled_postact_profile(A, postact_profile)
      dH1 = 2 * relu_H1 * H3 * dA
      dH3 = relu_H1_sq * dA
    else:
      raise ValueError(f"Unknown activation: {activation}")

    # SonicMoE dGate identity: dGate = ⟨A, dA⟩ instead of ⟨dOut, Ye⟩
    # This avoids recomputing Ye_pad in both single-GPU and distributed modes.
    _C.dgate_from_adA_bf16(
      A.data_ptr(),
      dA.data_ptr(),
      dGate_sorted.data_ptr(),
      int(M_recv), int(Dff),
      stream,
    )
    # Correct SonicMoE identity: it requires dA_ungated = dY @ W2.T.
    # We formed dA from dYe_pad where dYe = gate * dY (needed for weight grads),
    # so dA = gate * dA_ungated and ⟨A, dA⟩ = gate * dGate_true.
    gate_sorted.clamp_min_(1e-12)
    dGate_sorted.div_(gate_sorted)
    if is_dist:
      # Distributed: send dGate back to source ranks via IPC
      _C.send_dgate_dist_bf16(
        row_id.data_ptr(),
        dGate_sorted.data_ptr(),
        dGates_tk_f32.data_ptr(),
        int(M_recv), int(T), int(K),
        stream,
      )
    else:
      # Single-GPU: scatter dGate directly
      _C.scatter_gate_bf16(
        dGate_sorted.data_ptr(),
        row_id.data_ptr(),
        dGates_tk_f32.data_ptr(),
        int(M_recv), int(T), int(K),
        stream,
      )

    copy_event.synchronize()
    offs_host = offs_pinned
    dW2 = torch.empty_like(W2)
    _C.bf16_wgrad_w2_cublaslt(
      A.data_ptr(),
      dYe_pad.data_ptr(),
      dW2.data_ptr(),
      offs_host.data_ptr(),
      int(E), int(H), int(Dff),
      stream,
    )

    dW1 = torch.empty_like(W1)
    _C.bf16_wgrad_w13_cublaslt(
      Xe_use.data_ptr(),
      dH1.data_ptr(),
      dW1.data_ptr(),
      offs_host.data_ptr(),
      int(E), int(H), int(Dff),
      stream,
    )

    # relu_squared doesn't use W3, so dW3/dH3 gradients are zero
    if activation == "relu_squared":
      dW3 = torch.zeros_like(W3)
      dX_pad = torch._grouped_mm(dH1, W1_use.transpose(1, 2), offs=offs_pad)
    else:
      dW3 = torch.empty_like(W3)
      _C.bf16_wgrad_w13_cublaslt(
        Xe_use.data_ptr(),
        dH3.data_ptr(),
        dW3.data_ptr(),
        offs_host.data_ptr(),
        int(E), int(H), int(Dff),
        stream,
      )
      dX_pad = torch._grouped_mm(dH1, W1_use.transpose(1, 2), offs=offs_pad)
      dX_pad.add_(torch._grouped_mm(dH3, W3_use.transpose(1, 2), offs=offs_pad))
    dX = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
      dX_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
      _C.gather_from_pad_bf16(dX_pad.data_ptr(), dX_sorted.data_ptr(), int(M_recv), int(H), stream)
      _C.scatter_dx_dist_bf16(
        dX_sorted.data_ptr(),
        row_id.data_ptr(),
        dX.data_ptr(),
        int(M_recv), int(T), int(H), int(K),
        stream,
      )
    else:
      _C.scatter_dx_bf16_internal(
        dX_pad.data_ptr(),
        row_id.data_ptr(),
        dX.data_ptr(),
        int(M_recv), int(T), int(H), int(K),
        stream,
      )

    dGates = dGates_tk_f32.to(dtype=torch.bfloat16)
    return None, dX, None, dGates, dW1, dW3, dW2, None, None, None
