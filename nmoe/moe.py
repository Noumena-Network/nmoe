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
) -> torch.Tensor:
  """Expert MLP: Y = (SiLU(X @ W1) * (X @ W3)) @ W2

  BF16 path using torch._grouped_mm.

  Args:
    Xe_pad: [M_pad, H] pre-padded BF16 input from rdep.dispatch_sorted
    W1, W3: [E, H, Dff] gate/up weights
    W2: [E, Dff, H] down weight
    offs_pad: [E] cumulative padded offsets from rdep

  Returns:
    [M_pad, H] BF16 output (caller uses dest to select valid rows)
  """
  if Xe_pad.size(0) == 0:
    return Xe_pad

  H1 = torch._grouped_mm(Xe_pad, W1, offs=offs_pad)
  H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)
  return torch._grouped_mm(F.silu(H1).mul_(H3), W2, offs=offs_pad)


class _MoEBf16Fused(torch.autograd.Function):
  @staticmethod
  def forward(ctx, rdep: Rdep, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
              W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
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
        dummy_ye = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        _C.return_scatter(
          dummy_ye.data_ptr(),
          out_f32.data_ptr(),
          0, int(T), int(K),
          stream,
        )
      ctx.rdep = rdep
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

    Ye_pad = expert(Xe_pad, W1, W3, W2, offs_pad)
    Ye_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
    _C.gather_from_pad_bf16(Ye_pad.data_ptr(), Ye_sorted.data_ptr(), int(M_recv), int(H), stream)

    _C.return_scatter(
      Ye_sorted.data_ptr(),
      out_f32.data_ptr(),
      int(M_recv), int(T), int(K),
      stream,
    )

    ctx.rdep = rdep
    ctx.save_for_backward(x, eid, gates, W1, W3, W2)
    return out_f32.to(dtype=torch.bfloat16)

  @staticmethod
  def backward(ctx, dOut: torch.Tensor):
    x, eid, gates, W1, W3, W2 = ctx.saved_tensors
    rdep: Rdep = ctx.rdep
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

      return None, dX, None, dGates, dW1, dW3, dW2

    max_pad = (int(M_recv) + int(offs_pad.numel()) * (align - 1) + (align - 1)) // align * align
    offs_pad[-1] = int(max_pad)

    Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
    _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

    row_id = torch.empty(int(M_recv), device=device, dtype=torch.int64)
    gate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
    _C.gather_meta_sorted_bf16(row_id.data_ptr(), gate_sorted.data_ptr(), int(M_recv), stream)

    with torch.enable_grad():
      Xe_pad = Xe_pad.requires_grad_(True)
      Ye_pad = expert(Xe_pad, W1, W3, W2, offs_pad)

    Ye_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
    _C.gather_from_pad_bf16(Ye_pad.detach().data_ptr(), Ye_sorted.data_ptr(), int(M_recv), int(H), stream)

    dYe_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
    dGate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
    dGates_tk_f32 = torch.zeros(int(T), int(K), device=device, dtype=torch.float32)

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
      _C.gather_dy_dist_bf16(
        dOut.data_ptr(),
        eid.data_ptr(),
        Ye_sorted.data_ptr(),
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
        Ye_sorted.data_ptr(),
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
    return None, dX, None, dGates, dW1, dW3, dW2


class _MoEBlockscaledFused(torch.autograd.Function):
  @staticmethod
  def forward(ctx, rdep: Rdep, x: torch.Tensor, eid: torch.Tensor, gates: torch.Tensor,
              W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor, W_cache) -> torch.Tensor:
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
    if is_dist:
      need = int(T) * int(K) * int(rdep.world)
      if rdep.capacity < need:
        raise RuntimeError(
          f"[RDEP] capacity too small: capacity={rdep.capacity:,} need>={need:,} (T={T:,} K={K} world={rdep.world}). "
          "Set capacity to worst-case T*K*world (no silent truncation)."
        )

    # Option A: Use BF16 dispatch + local quantization
    # This ensures Xe_pad (BF16) is available for backward STE
    offs_pad = torch.empty(E, device=device, dtype=torch.int32)
    M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()
    align = 128  # Required for blockscaled SF swizzle

    M_recv = _C.dispatch_meta_bf16(
      x.data_ptr(), eid.data_ptr(), gates_fp32.data_ptr(),
      int(T), int(K), align,
      offs_pad.data_ptr(), M_host.data_ptr(),
      stream,
    )

    out_f32 = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
    if M_recv <= 0:
      # DeepEP collectiveness: every rank must participate in return_scatter
      if is_dist:
        dummy_ye = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        _C.return_scatter(dummy_ye.data_ptr(), out_f32.data_ptr(), 0, int(T), int(K), stream)
      ctx.rdep = rdep
      ctx.W_cache = W_cache
      ctx.T = int(T)
      ctx.H = int(H)
      ctx.K = int(K)
      ctx.save_for_backward(x, eid, gates, W1, W3, W2)
      return out_f32.to(dtype=torch.bfloat16)

    # Compute max_pad and extend last expert's padded region
    max_pad = (int(M_recv) + E * (align - 1) + (align - 1)) // align * align
    offs_pad[-1] = int(max_pad)

    # Gather BF16 activations
    Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
    _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

    # Quantize locally: BF16 -> FP8/NVFP4
    pack_factor = 2 if rdep.profile == 'fp8' else 4
    Hp = H // pack_factor
    sf_k = H // 32
    sf_k_pad = ((sf_k + 3) // 4) * 4
    M_e_stride = ((rdep.capacity + 127) // 128) * 128  # 128-aligned capacity per expert

    Xe_q = torch.empty(int(max_pad), Hp, device=device, dtype=torch.uint16)
    Xe_sf = torch.empty(E, M_e_stride, sf_k_pad, device=device, dtype=torch.uint8)

    # Quant kernels expect offs_with0 [E+1] with leading 0: [0, offs_pad[0], offs_pad[1], ...]
    offs_with0 = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), offs_pad])

    if rdep.profile == 'fp8':
      _C.quant_fp8_sf_strided_mma(
        Xe_pad.data_ptr(), int(H),
        Xe_q.data_ptr(), Hp,
        Xe_sf.data_ptr(),
        offs_with0.data_ptr(),
        E, M_e_stride,
        int(max_pad), int(H),
        stream,
      )
    else:  # nvfp4
      _C.quant_nvfp4_sf_strided_mma(
        Xe_pad.data_ptr(), int(H),
        Xe_q.data_ptr(), Hp,
        Xe_sf.data_ptr(),
        offs_with0.data_ptr(),
        E, M_e_stride,
        int(max_pad), int(H),
        stream,
      )

    # Expert compute (blockscaled)
    from nmoe.blockscaled.grouped import expert_blockscaled
    Ye_pad = expert_blockscaled(Xe_q, Xe_sf, W_cache, offs_pad)

    # Gather sorted and return scatter
    Ye_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
    _C.gather_from_pad_bf16(Ye_pad.data_ptr(), Ye_sorted.data_ptr(), int(M_recv), int(H), stream)

    _C.return_scatter(
      Ye_sorted.data_ptr(),
      out_f32.data_ptr(),
      int(M_recv), int(T), int(K),
      stream,
    )

    ctx.rdep = rdep
    ctx.W_cache = W_cache
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

      return None, dX, None, dGates, dW1, dW3, dW2, None

    # Compute max_pad and extend last expert's padded region
    max_pad = (int(M_recv) + E * (align - 1) + (align - 1)) // align * align
    offs_pad[-1] = int(max_pad)

    # Gather BF16 activations (correct from all source ranks via IPC buffer!)
    Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
    _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

    # Get row_id and gate_sorted for dGate computation
    row_id = torch.empty(int(M_recv), device=device, dtype=torch.int64)
    gate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
    _C.gather_meta_sorted_bf16(row_id.data_ptr(), gate_sorted.data_ptr(), int(M_recv), stream)

    # Quantize and run expert forward for Ye recomputation (needed for dGate)
    pack_factor = 2 if rdep.profile == 'fp8' else 4
    Hp = H // pack_factor
    sf_k = H // 32
    sf_k_pad = ((sf_k + 3) // 4) * 4
    M_e_stride = ((rdep.capacity + 127) // 128) * 128

    Xe_q = torch.empty(int(max_pad), Hp, device=device, dtype=torch.uint16)
    Xe_sf = torch.empty(E, M_e_stride, sf_k_pad, device=device, dtype=torch.uint8)

    # Quant kernels expect offs_with0 [E+1] with leading 0: [0, offs_pad[0], offs_pad[1], ...]
    offs_with0 = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), offs_pad])

    if rdep.profile == 'fp8':
      _C.quant_fp8_sf_strided_mma(
        Xe_pad.data_ptr(), int(H),
        Xe_q.data_ptr(), Hp,
        Xe_sf.data_ptr(),
        offs_with0.data_ptr(),
        E, M_e_stride,
        int(max_pad), int(H),
        stream,
      )
    else:  # nvfp4
      _C.quant_nvfp4_sf_strided_mma(
        Xe_pad.data_ptr(), int(H),
        Xe_q.data_ptr(), Hp,
        Xe_sf.data_ptr(),
        offs_with0.data_ptr(),
        E, M_e_stride,
        int(max_pad), int(H),
        stream,
      )

    from nmoe.blockscaled.grouped import expert_blockscaled
    Ye_pad = expert_blockscaled(Xe_q, Xe_sf, W_cache, offs_pad)

    # Gather sorted Ye for dGate
    Ye_sorted_unscaled = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
    _C.gather_from_pad_bf16(Ye_pad.data_ptr(), Ye_sorted_unscaled.data_ptr(), int(M_recv), int(H), stream)

    # TODO(perf): The gather_dy kernels still compute dGate internally (dot product of Ye*dOut).
    # This is wasted compute (~negligible). To fully remove, modify CUDA kernels in rdep.cu.
    dYe_sorted = torch.empty(int(M_recv), int(H), device=device, dtype=torch.bfloat16)
    dGate_sorted = torch.empty(int(M_recv), device=device, dtype=torch.float32)
    dGates_tk_f32 = torch.zeros(int(T), int(K), device=device, dtype=torch.float32)

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
      _C.gather_dy_dist_bf16(
        dOut.data_ptr(),
        eid.data_ptr(),
        Ye_sorted_unscaled.data_ptr(),  # Ye only used for dGate which we discard
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
        Ye_sorted_unscaled.data_ptr(),  # Ye only used for dGate which we discard
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
    H1 = torch._grouped_mm(Xe_pad, W1, offs=offs_pad)
    H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)
    dA = torch._grouped_mm(dYe_pad, W2.transpose(1, 2), offs=offs_pad)
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
      Xe_pad.data_ptr(),
      dH1.data_ptr(),
      dW1.data_ptr(),
      offs_host.data_ptr(),
      int(E), int(H), int(Dff),
      stream,
    )

    dW3 = torch.empty_like(W3)
    _C.bf16_wgrad_w13_cublaslt(
      Xe_pad.data_ptr(),
      dH3.data_ptr(),
      dW3.data_ptr(),
      offs_host.data_ptr(),
      int(E), int(H), int(Dff),
      stream,
    )

    dX_pad = torch._grouped_mm(dH1, W1.transpose(1, 2), offs=offs_pad)
    dX_pad.add_(torch._grouped_mm(dH3, W3.transpose(1, 2), offs=offs_pad))
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
    return None, dX, None, dGates, dW1, dW3, dW2, None
