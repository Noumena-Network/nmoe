"""Grouped blockscaled GEMM for SM100 (B200).

Used by MoE expert MLP compute. Production contract:
- Inputs are pre-quantized (FP8 E4M3FN or NVFP4 E2M1 packed) with E8M0 scale factors.
- Scale factors are already swizzled to CUTLASS MMA layout (raw uint8 bytes).
- Single path only: strided grouped launch with GPU-built metadata (no list-based API).
"""

from __future__ import annotations

from dataclasses import dataclass, astuple
from typing import Tuple, Type, Union
from inspect import isclass

import torch


# -- CuTeDSL imports (runtime + typing) --
try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as ctorch
    # Additional CuTeDSL subsystems used by the vendored kernel
    from cutlass.cute.nvgpu import cpasync, tcgen05
    import cutlass.utils as utils
    import cutlass.pipeline as pipeline
    import cutlass.utils.blackwell_helpers as sm100_utils
    import cutlass.utils.blockscaled_layout as blockscaled_utils
    from cutlass.cutlass_dsl import dsl_user_op
    from cutlass._mlir.dialects import arith, builtin, llvm
    from cutlass.base_dsl.typing import T
    from cutlass.cute.typing import (
        Float32,
        Float4E2M1FN,
        Float8E4M3FN,
        Int32,
        Int4,
        Uint16,
        Uint8,
    )
except Exception as e:  # pragma: no cover - environment guard
    raise RuntimeError(
        "CuTeDSL (nvidia-cutlass-dsl) is required. Install >= 4.3.1.\n"
        f"Import error: {e}"
    )


# -- CUDA driver bindings for CUstream interop --
try:
    import cuda.bindings.driver as cuda
except Exception as e:  # pragma: no cover - environment guard
    raise RuntimeError(
        "cuda.bindings.driver not available. Ensure CUDA Python bindings are installed.\n"
        f"Import error: {e}"
    )


# Scale factor swizzle + grouped GEMM metadata (from in-repo CUDA extension)
from nmoe.csrc import rdep

def _swizzle_sf_to_mma(sf_mkl: torch.Tensor) -> torch.Tensor:
  """Swizzle scale factors from MKL row-major to MMA layout.

  IMPORTANT: Returns the full padded tensor to preserve the swizzle pattern.
  The GEMM kernel uses offs array to know actual bounds.
  """
  assert sf_mkl.dtype == torch.uint8 and sf_mkl.ndim == 3
  M, sf_k, _ = sf_mkl.shape

  # Pad M to 128 and sf_k to 4 (required by swizzle)
  M_pad = ((M + 127) // 128) * 128
  sf_k_pad = ((sf_k + 3) // 4) * 4

  if M != M_pad or sf_k != sf_k_pad:
    sf_mkl_pad = torch.zeros(M_pad, sf_k_pad, dtype=torch.uint8, device=sf_mkl.device)
    sf_mkl_pad[:M, :sf_k] = sf_mkl.squeeze(-1)
  else:
    sf_mkl_pad = sf_mkl.squeeze(-1).contiguous()

  sf_mma_flat = torch.empty(M_pad * sf_k_pad, dtype=torch.uint8, device=sf_mkl.device)
  stream = torch.cuda.current_stream(sf_mkl.device)
  rdep.swizzle_sf_mkl_to_mma(sf_mkl_pad.data_ptr(), sf_mma_flat.data_ptr(), M_pad, sf_k_pad, stream)

  # Return full padded tensor - DO NOT slice or call .contiguous() as that would
  # copy data back to row-major layout, destroying the MMA swizzle pattern!
  return sf_mma_flat.view(M_pad, sf_k_pad, 1)
# No external loader — kernel is vendored below in this file.


# -- Local helpers --


def _num_sms(device: int | None = None) -> int:
    dev = torch.cuda.current_device() if device is None else int(device)
    props = torch.cuda.get_device_properties(dev)
    return int(props.multi_processor_count)


def _require_sm100(device_index: int | None = None) -> None:
    """Fail fast off-target. nmoe is B200-only (SM100)."""
    dev = torch.cuda.current_device() if device_index is None else int(device_index)
    cap = torch.cuda.get_device_capability(dev)
    if tuple(cap) != (10, 0):
        raise RuntimeError(f"blockscaled GEMM requires SM100 (B200). Got capability={cap}.")


# -----------------------------------------------------------------------------
# Lightweight workspace cache to avoid per-call GPU allocations of metadata
# tensors in run_grouped_blockscaled_strided().
# Keyed by (device_index, group_count, max_clusters_capacity).
# No public knobs; single-path behavior.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Fused epilogue helpers (CuTeDSL).
# These are used only when fuse_swiglu_quant=True.
# -----------------------------------------------------------------------------

def _tanh_approx_f32(x: cutlass.Float32) -> cutlass.Float32:
    # Exact tanh is acceptable here; input is already half-scaled (x = 0.5 * gate).
    return cute.tanh(x)


@dsl_user_op
def _e8m0_encode_from_pos_f32(scale: Float32, *, loc=None, ip=None):
    """Encode a positive FP32 scale to E8M0 (uint8 exponent byte).

    Matches `nmoe/csrc/ptx.cu:e8m0_encode_from_pos_f32` exactly using bit ops
    (no log2/exp2), avoiding numeric edge-case drift and fixing CuTeDSL scalar
    math codegen issues.

    For normalized FP32: `scale = 2^(E-127) * (1.mantissa)`.
    Then `ceil(log2(scale)) = (E-127) + (mantissa != 0)`, so the E8M0 byte is:
      `byte = E + (mantissa != 0)` (clamped to 254).
    """
    # Bitcast float32 to i32 and extract exponent/mantissa.
    scale_v = scale.ir_value(loc=loc, ip=ip) if hasattr(scale, 'ir_value') else scale
    bits_i32 = arith.bitcast(Int32.mlir_type, scale_v, loc=loc, ip=ip)
    c23 = arith.constant(Int32.mlir_type, 23, loc=loc, ip=ip)
    c0xff = arith.constant(Int32.mlir_type, 0xFF, loc=loc, ip=ip)
    e_i32 = arith.andi(arith.shrui(bits_i32, c23, loc=loc, ip=ip), c0xff, loc=loc, ip=ip)

    c_mant_mask = arith.constant(Int32.mlir_type, 0x7FFFFF, loc=loc, ip=ip)
    mant_i32 = arith.andi(bits_i32, c_mant_mask, loc=loc, ip=ip)

    c0 = arith.constant(Int32.mlir_type, 0, loc=loc, ip=ip)
    c1 = arith.constant(Int32.mlir_type, 1, loc=loc, ip=ip)
    mant_nonzero = arith.cmpi(arith.CmpIPredicate.ne, mant_i32, c0, loc=loc, ip=ip)
    e_i32 = arith.addi(e_i32, arith.select(mant_nonzero, c1, c0, loc=loc, ip=ip), loc=loc, ip=ip)

    # Clamp to [0, 254].
    c254 = arith.constant(Int32.mlir_type, 254, loc=loc, ip=ip)
    e_i32 = arith.minui(e_i32, c254, loc=loc, ip=ip)

    return arith.trunci(Uint8.mlir_type, e_i32, loc=loc, ip=ip)


@dsl_user_op
def _e8m0_inv_decode_to_f32(scale_byte: Uint8, *, loc=None, ip=None):
    """Decode inv_scale from an E8M0 byte (uint8): inv_scale = 2^(127 - byte).

    Matches `nmoe/csrc/ptx.cu:e8m0_inv_decode_to_f32`:
      - inv_byte = 254 - min(byte, 254)
      - float bits = (inv_byte == 0 ? 0x00400000 : inv_byte << 23)
    """
    scale_byte_v = scale_byte.ir_value(loc=loc, ip=ip) if hasattr(scale_byte, 'ir_value') else scale_byte
    b_i32 = arith.extui(Int32.mlir_type, scale_byte_v, loc=loc, ip=ip)
    c254 = arith.constant(Int32.mlir_type, 254, loc=loc, ip=ip)
    b_i32 = arith.minui(b_i32, c254, loc=loc, ip=ip)
    inv_i32 = arith.subi(c254, b_i32, loc=loc, ip=ip)

    c23 = arith.constant(Int32.mlir_type, 23, loc=loc, ip=ip)
    bits_normal = arith.shli(inv_i32, c23, loc=loc, ip=ip)

    c0 = arith.constant(Int32.mlir_type, 0, loc=loc, ip=ip)
    is_zero = arith.cmpi(arith.CmpIPredicate.eq, inv_i32, c0, loc=loc, ip=ip)
    c_subnormal = arith.constant(Int32.mlir_type, 0x00400000, loc=loc, ip=ip)
    bits = arith.select(is_zero, c_subnormal, bits_normal, loc=loc, ip=ip)

    return arith.bitcast(Float32.mlir_type, bits, loc=loc, ip=ip)


@dsl_user_op
def _fp8_pack2_e4m3(x0: Float32, x1: Float32, *, loc=None, ip=None):
    """Pack two FP8 E4M3 values into a uint16 (low byte = x0, high byte = x1).

    Uses inline PTX: cvt.rn.satfinite.e4m3x2.f32 (sm_89+).
    Matches nmoe/csrc/ptx.cu behavior.

    TODO: Explore CUTLASS EVT (Epilogue Visitor Tree) for native FP8 support
    which may provide better codegen and avoid inline PTX.
    """
    x0_v = x0.ir_value(loc=loc, ip=ip) if hasattr(x0, 'ir_value') else x0
    x1_v = x1.ir_value(loc=loc, ip=ip) if hasattr(x1, 'ir_value') else x1
    # cvt.rn.satfinite.e4m3x2.f32 packs two f32 -> one b16 (two FP8 bytes)
    # PTX uses .b16 register for packed output. Order: low byte = $2, high byte = $1.
    result = llvm.inline_asm(
        Uint16.mlir_type,
        [x1_v, x0_v],  # Swapped to get x0 in low byte
        "{ .reg.b16 tmp; cvt.rn.satfinite.e4m3x2.f32 tmp, $1, $2; mov.u16 $0, tmp; }",
        "=h,f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return result


@dsl_user_op
def _nvfp4_pack2_e2m1(x0: Float32, x1: Float32, *, loc=None, ip=None):
    """Pack two NVFP4 E2M1 values into a uint8 (low nibble = x0, high nibble = x1).

    Uses inline PTX: cvt.rn.satfinite.e2m1x2.f32 (sm_100a / Blackwell).
    Matches nmoe/csrc/ptx.cu behavior.

    TODO: Explore CUTLASS EVT (Epilogue Visitor Tree) for native NVFP4 support
    which may provide better codegen and avoid inline PTX.
    """
    x0_v = x0.ir_value(loc=loc, ip=ip) if hasattr(x0, 'ir_value') else x0
    x1_v = x1.ir_value(loc=loc, ip=ip) if hasattr(x1, 'ir_value') else x1
    # cvt.rn.satfinite.e2m1x2.f32 packs two f32 -> one .b8 (two E2M1 nibbles)
    # PTX uses .b8 register for packed output. Order: low nibble = $2, high nibble = $1.
    # Output to u32 via mov.b32 then extract low byte, since LLVM lacks direct b8 support.
    result_i32 = llvm.inline_asm(
        Int32.mlir_type,
        [x1_v, x0_v],  # Swapped to get x0 in low nibble
        "{ .reg.b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2; mov.b32 $0, {tmp, tmp, tmp, tmp}; }",
        "=r,f,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    # Extract low byte
    return arith.trunci(Uint8.mlir_type, result_i32, loc=loc, ip=ip)


_STRIDED_WORKSPACE_CACHE: dict[tuple, tuple] = {}

def _get_strided_workspace(device, E: int, max_clusters: int, KernelCls):
    key = (int(device.index if hasattr(device, 'index') else int(device)), int(E), int(max(1, max_clusters)))
    ws = _STRIDED_WORKSPACE_CACHE.get(key)
    if ws is not None:
        return ws
    tensormap_shape = (
        max(1, int(max_clusters)),
        KernelCls.GROUPED_NUM_TENSOR_MAPS,
        KernelCls.GROUPED_TENSOR_MAP_BYTES // 8,
    )
    tensormap_cute, _ = ctorch.cute_tensor_like(
        torch.empty(tensormap_shape, dtype=torch.int64, device=device),
        cutlass.Int64, is_dynamic_layout=False,
    )
    dim_size_mnkl_cute, dim_size_mnkl_torch = ctorch.cute_tensor_like(
        torch.empty((E, 4), dtype=torch.int32, device=device),
        cutlass.Int32, is_dynamic_layout=False, assumed_align=16,
    )
    strides_abc_cute, strides_abc_torch = ctorch.cute_tensor_like(
        torch.empty((E, 3, 2), dtype=torch.int32, device=device),
        cutlass.Int32, is_dynamic_layout=False, assumed_align=16,
    )
    ptrs_abc_cute, ptrs_abc_torch = ctorch.cute_tensor_like(
        torch.empty((E, 3), dtype=torch.int64, device=device),
        cutlass.Int64, is_dynamic_layout=False, assumed_align=16,
    )
    ptrs_sfasfb_cute, ptrs_sfasfb_torch = ctorch.cute_tensor_like(
        torch.empty((E, 2), dtype=torch.int64, device=device),
        cutlass.Int64, is_dynamic_layout=False, assumed_align=16,
    )
    ws = (
        tensormap_cute,
        dim_size_mnkl_cute, dim_size_mnkl_torch,
        strides_abc_cute, strides_abc_torch,
        ptrs_abc_cute, ptrs_abc_torch,
        ptrs_sfasfb_cute, ptrs_sfasfb_torch,
    )
    _STRIDED_WORKSPACE_CACHE[key] = ws
    return ws

def _create_initial_cute_tensor(
    l: int,
    mode0: int,
    mode1: int,
    is_mode0_major: bool,
    dtype: type,
    divisibility: int = 16,
) -> Tuple[cute.Tensor, torch.Tensor]:
    """Create a dynamic-layout CuTe tensor for JIT compilation.

    Follows the Example 92 pattern: creates a GPU tensor with proper CuTe
    layout metadata. The tensor is allocated but NOT initialized - the kernel
    uses tensor_of_ptrs_abc for actual data access.

    Parameters
    ----------
    l : int
        Batch dimension (always 1 for grouped GEMM)
    mode0 : int
        First matrix dimension (M for A/C, N for B)
    mode1 : int
        Second matrix dimension (K for A/B, N for C)
    is_mode0_major : bool
        True if mode0 has unit stride (column-major)
        False if mode1 has unit stride (row-major, standard for PyTorch)
    dtype : type
        CuTe dtype (e.g., cutlass.Float8E4M3FN)
    divisibility : int
        Element divisibility for alignment (16 for FP8, 32 for FP4)

    Returns
    -------
    Tuple[cute.Tensor, torch.Tensor]
        CuTe tensor with dynamic layout and underlying torch tensor
    """
    # Create reference CPU tensor with proper shape/layout encoding per Example 92
    ref_cpu = ctorch.matrix(l, mode0, mode1, is_mode0_major, cutlass.Float32)

    # Create GPU CuTe tensor with dynamic layout
    cute_tensor, torch_tensor = ctorch.cute_tensor_like(
        ref_cpu, dtype, is_dynamic_layout=True, assumed_align=16
    )

    # Mark compact shape dynamic per Example 92 pattern
    cute_tensor.mark_compact_shape_dynamic(
        mode=0 if is_mode0_major else 1,
        stride_order=(2, 1, 0) if is_mode0_major else (2, 0, 1),
        divisibility=divisibility,
    )

    return cute_tensor, torch_tensor


def _wrap_uint8_as_sf(
    tensor: torch.Tensor,
    sf_dtype: type[cutlass.Numeric],
) -> Tuple[cute.Tensor, torch.Tensor]:
    """Wrap a uint8 tensor as a properly-typed E8M0 CuTe tensor.

    E8M0 is byte-identical to uint8 (each byte = 2^(byte-127)), so we can
    directly wrap the existing uint8 buffer as E8M0 without any conversion.
    This is the fast path: ~0.2ms vs ~109ms for FP32→E8M0 conversion.
    """
    assert tensor.dtype == torch.uint8, "Expected uint8 tensor for pre-converted SF"
    assert tensor.is_cuda, "Scale-factor tensor must be CUDA resident"
    # Create CuTe tensor with E8M0 dtype interpretation
    cute_tensor, torch_back = ctorch.cute_tensor_like(
        tensor, sf_dtype, is_dynamic_layout=True, assumed_align=16
    )
    # Copy the uint8 bytes to the backing tensor (E8M0 is byte-identical to uint8)
    torch_back.view(torch.uint8).copy_(tensor.view(torch.uint8))
    return cute_tensor, torch_back


__all__ = ["quantize_weights", "expert_blockscaled"]


# -----------------------------------------------------------------------------
# Fused SwiGLU + quantization helpers (CuTeDSL / PTX)
# -----------------------------------------------------------------------------
#
# The grouped blockscaled kernel includes a fused epilogue that computes:
#   (gate, up) -> silu(gate) * up -> quantize -> packed activation + E8M0 SFA
#
# We use inline PTX for FP8/NVFP4 packing and bitwise E8M0 encode/decode to keep
# this path allocation-free and avoid materializing BF16 intermediates.


@cute.jit
def _silu_f32(x: cutlass.Float32) -> cutlass.Float32:
    # Match nmoe/csrc/quant.cu: x / (1 + exp(-x)).
    one = cutlass.Float32(1.0)
    return x / (one + cute.math.exp(-x, fastmath=True))


@dsl_user_op
def _e8m0_encode_from_pos_f32(s: Float32, *, loc=None, ip=None) -> Uint8:
    # Match nmoe/csrc/ptx.cu: ceil(log2(s)) via exponent+mantissa.
    # Caller guarantees s > 0.
    u16 = Uint16(
        llvm.inline_asm(
            Uint16.mlir_type,
            [Float32(s).ir_value(loc=loc, ip=ip)],
            """{\n\t
            .reg .b32 bits; \n\t
            .reg .b32 e;    \n\t
            .reg .b32 mant; \n\t
            .reg .pred p;   \n\t
            mov.b32 bits, $1;       \n\t
            shr.u32 e, bits, 23;    \n\t
            and.b32 e, e, 0xff;     \n\t
            and.b32 mant, bits, 0x7fffff; \n\t
            setp.ne.u32 p, mant, 0; \n\t
            @p add.u32 e, e, 1;     \n\t
            min.u32 e, e, 254;      \n\t
            cvt.u16.u32 $0, e;      \n\t
            }""",
            "=h,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    u8 = llvm.trunc(T.i8(), u16.ir_value(loc=loc, ip=ip), llvm.IntegerOverflowFlags.none, loc=loc, ip=ip)
    return Uint8(u8)


@dsl_user_op
def _e8m0_inv_decode_to_f32(b: Uint8, *, loc=None, ip=None) -> Float32:
    # Match nmoe/csrc/ptx.cu: inv_scale = 2^(127-b) via exponent-only float bits.
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [llvm.zext(T.i16(), Uint8(b).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)],
            """{\n\t
            .reg .b32 inv;  \n\t
            .reg .b32 bits; \n\t
            .reg .pred p;   \n\t
            cvt.u32.u16 inv, $1;    \n\t
            sub.u32 inv, 254, inv;  \n\t
            shl.b32 bits, inv, 23;  \n\t
            setp.eq.u32 p, inv, 0;  \n\t
            @p mov.b32 bits, 0x00400000; \n\t
            mov.b32 $0, bits;       \n\t
            }""",
            "=f,h",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _fp8_pack2_e4m3(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint16:
    # Pack two FP8 E4M3 bytes into uint16. Use argument order (b, a) so `a`
    # becomes the low byte in memory (matches existing packed activation layout).
    #
    # PTX returns the packed 16b value in a predicate-friendly register; move it
    # through a 32b integer output and truncate to avoid constraint mismatches.
    u32 = llvm.inline_asm(
        T.i32(),
        [
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(a).ir_value(loc=loc, ip=ip),
        ],
        """{\n\t
        .reg .b16 t; \n\t
        cvt.rn.satfinite.e4m3x2.f32 t, $1, $2; \n\t
        mov.b32 $0, {t, 0}; \n\t
        }""",
        "=r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    u16 = llvm.trunc(T.i16(), u32, llvm.IntegerOverflowFlags.none, loc=loc, ip=ip)
    return Uint16(u16)


@dsl_user_op
def _nvfp4_pack2_e2m1(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint8:
    # Pack two NVFP4 E2M1 nibbles into one byte. Use argument order (b, a) to
    # match the packed layout consumed by CUTLASS FP4 kernels.
    u32 = llvm.inline_asm(
        T.i32(),
        [
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(a).ir_value(loc=loc, ip=ip),
        ],
        """{\n\t
        .reg .b8 t; \n\t
        cvt.rn.satfinite.e2m1x2.f32 t, $1, $2; \n\t
        mov.b32 $0, {t, 0, 0, 0}; \n\t
        }""",
        "=r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    u8 = llvm.trunc(T.i8(), u32, llvm.IntegerOverflowFlags.none, loc=loc, ip=ip)
    return Uint8(u8)


@dsl_user_op
def _nvfp4_pack4_e2m1_u16(a0: Float32, a1: Float32, a2: Float32, a3: Float32, *, loc=None, ip=None) -> Uint16:
    # Match nmoe/csrc/ptx.cu + nmoe/csrc/quant.cu:
    #  - pack 4 E2M1 nibbles into uint16 (two bytes)
    #  - swap element pairs to match CUTLASS / TE MMA lane order.
    u32 = llvm.inline_asm(
        T.i32(),
        [
            Float32(a1).ir_value(loc=loc, ip=ip),
            Float32(a0).ir_value(loc=loc, ip=ip),
            Float32(a3).ir_value(loc=loc, ip=ip),
            Float32(a2).ir_value(loc=loc, ip=ip),
        ],
        """{\n\t
        .reg .b8 f0; \n\t
        .reg .b8 f1; \n\t
        cvt.rn.satfinite.e2m1x2.f32 f0, $1, $2; \n\t
        cvt.rn.satfinite.e2m1x2.f32 f1, $3, $4; \n\t
        mov.b32 $0, {f0, f1, f0, f1}; \n\t
        }""",
        "=r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    u16 = llvm.trunc(T.i16(), u32, llvm.IntegerOverflowFlags.none, loc=loc, ip=ip)
    return Uint16(u16)


# ===============================
# Vendored kernel implementation
# ===============================

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


class NmoeGroupedScaledGemmKernel:
    """
    Vendored minimal Grouped Blockscaled GEMM kernel for SM100.
    This is a direct inlining of the essential structures from CUTLASS CuTeDSL
    example 92, trimmed to remove CLI/benchmark scaffolding and external deps.
    """

    @dataclass
    class Params:
        a_dtype: Type[cutlass.Numeric]
        b_dtype: Type[cutlass.Numeric]
        acc_dtype: Type[cutlass.Numeric]
        c_dtype: Type[cutlass.Numeric]
        sf_dtype: Type[cutlass.Numeric]
        sf_vec_size: int
        a_layout: utils.LayoutEnum
        b_layout: utils.LayoutEnum
        c_layout: utils.LayoutEnum
        use_2cta_instrs: bool
        mma_tiler_mn: tuple[int, int]
        cluster_shape_mn: tuple[int, int]
        update_tensormaps_in_smem: bool
        group_count: int
        problem_sizes_mnkl_host: tuple[tuple[int, int, int, int]]
        max_active_clusters: int
        fuse_swiglu_quant: bool = False

        @property
        def hash_key(self):
            return astuple(self)

    # Constants used by the kernel
    reserved_smem_bytes = 1024
    GROUPED_TENSOR_MAP_BYTES = 128
    GROUPED_NUM_TENSOR_MAPS = 6
    tensor_memory_management_bytes = 12

    def __init__(self, params: Params):
        self.a_dtype = params.a_dtype
        self.b_dtype = params.b_dtype
        self.acc_dtype = params.acc_dtype
        self.c_dtype = params.c_dtype
        self.sf_dtype = params.sf_dtype
        self.sf_vec_size = params.sf_vec_size
        self.a_major_mode = params.a_layout.mma_major_mode()
        self.b_major_mode = params.b_layout.mma_major_mode()
        self.c_layout = params.c_layout
        self.use_2cta_instrs = params.use_2cta_instrs
        self.cluster_shape_mn = params.cluster_shape_mn
        self.mma_tiler = (*params.mma_tiler_mn, 1)
        self.group_count = params.group_count
        self.max_active_clusters = params.max_active_clusters
        self.fuse_swiglu_quant = bool(params.fuse_swiglu_quant)
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.tensormap_update_mode = (
            utils.TensorMapUpdateMode.SMEM if params.update_tensormaps_in_smem else utils.TensorMapUpdateMode.GMEM
        )
        self.occupancy = 1
        # Warp roles
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len((self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id))
        # Named barriers
        self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * len(self.epilog_warp_id))
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)))
        self.tensormap_ab_init_barrier = pipeline.NamedBarrier(barrier_id=4, num_threads=64)
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = 512

    def _setup_attributes(self):
        # Compute instruction shapes
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler[:2],
        )
        # Build an SFB MMA placeholder; will be rebuilt after sfb inst shapes are computed
        # Under 2CTA, SFB M-mode must be 128; using AB tiler (M=256) here would fail.
        placeholder_shape_mn = (
            128,
            cute.round_up(self.mma_tiler[1], 128),
        ) if self.use_2cta_instrs else self.mma_tiler[:2]
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            placeholder_shape_mn,
        )
        # Match example 92 logic
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        # Rebuild SFB MMA with the correct instruction shape (M=128 under 2CTA)
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            (self.mma_inst_shape_mn_sfb[0], self.mma_inst_shape_mn_sfb[1]),
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_tile_shape_mnk = tuple(x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1)))
        if self.fuse_swiglu_quant:
            # Fused SwiGLU quantization consumes two adjacent H13 subtiles at once
            # (2x32 H13 cols => 32 activation cols), so the epilogue must have an
            # even number of 32-col subtiles along N.
            if int(self.cluster_tile_shape_mnk[1]) % 64 != 0:
                raise ValueError(
                    "fuse_swiglu_quant requires cluster N tile to be a multiple of 64 "
                    f"(got {int(self.cluster_tile_shape_mnk[1])})."
                )

        # Compute cluster layouts and multicast flags
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,))
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma_sfb.thr_id.shape,))
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype
        )

        # Stage counts and smem layouts
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )
        if self.fuse_swiglu_quant:
            # Fused SwiGLU quantization reads two adjacent H13 subtiles at once.
            # We need at least 2 C stages so we can keep both subtiles resident in
            # shared memory simultaneously (no intermediate BF16 tensor).
            self.num_c_stage = max(int(self.num_c_stage), 2)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        # Use SFB-specific MMA and tiler for SFB staging
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_sfb, self.mma_tiler_sfb, self.sf_vec_size, self.num_ab_stage
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage)

        # Fused SwiGLU epilogue: store BF16 activations (Dff) via TMA.
        # Each H13 subtile covers 32 columns => activation subtile is 16 columns.
        # This is tied to our fixed tiler (128x128) and BF16 epilogue on SM100.
        self.act_epi_tile = (self.cluster_tile_shape_mnk[0], 16)
        self.act_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.act_epi_tile, self.num_c_stage
        )

        mbar_smem_bytes = self._get_mbar_smem_bytes(
            num_acc_stage=self.num_acc_stage, num_ab_stage=self.num_ab_stage, num_c_stage=self.num_c_stage
        )
        tensormap_smem_bytes = (
            NmoeGroupedScaledGemmKernel.GROUPED_TENSOR_MAP_BYTES * NmoeGroupedScaledGemmKernel.GROUPED_NUM_TENSOR_MAPS
        )
        if mbar_smem_bytes + tensormap_smem_bytes + NmoeGroupedScaledGemmKernel.tensor_memory_management_bytes > self.reserved_smem_bytes:
            raise ValueError(
                "smem consumption for mbar and tensormap exceeds reserved bytes"
            )

    @cute.jit
    def __call__(
        self,
        initial_a: cute.Tensor,
        initial_b: cute.Tensor,
        initial_c: cute.Tensor,
        initial_sfa: cute.Tensor,
        initial_sfb: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_address_sfasfb: cute.Tensor,
        a_base_ptr: cutlass.Int64,
        a_row_bytes: cutlass.Int32,
        out_act_base_ptr: cutlass.Int64,
        out_sf_base_ptr: cutlass.Int64,
        out_sf_expert_bytes: cutlass.Int32,
        total_num_clusters: int,
        tensormap_cute_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        """Execute grouped blockscaled GEMM - matching Example 92 exactly.

        Initial tensors carry dtype and layout info. Actual data comes from
        tensor_address_abc/sfasfb arrays indexed by group.
        """
        # Extract dtypes and layout info from initial tensors (Example 92 lines 425-431)
        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b.element_type
        self.sf_dtype = initial_sfa.element_type
        self.c_dtype = initial_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(initial_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(initial_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(initial_c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes dependent on inputs
        self._setup_attributes()

        # Setup SF tensors by filling A/B shape to scale factor atom layout (Example 92 lines 438-449)
        # CRITICAL: Use initial_sfa.iterator to preserve the original pointer
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(initial_a.shape, self.sf_vec_size)
        initial_sfa = cute.make_tensor(initial_sfa.iterator, sfa_layout)

        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(initial_b.shape, self.sf_vec_size)
        initial_sfb = cute.make_tensor(initial_sfb.iterator, sfb_layout)

        # SF layouts derived above for initial_sfa/sfb

        # Build mma definitions
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # TMA load atoms for A/B/SFA/SFB
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, initial_a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, initial_b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op, initial_sfa, sfa_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma_sfb.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op, initial_sfb, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb, self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16
        )

        # Compute num_tma_load_bytes for pipeline tx_count
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size

        # TMA store atom for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), initial_c, epi_smem_layout, self.epi_tile
        )

        # TMA store atom for fused SwiGLU activations (BF16), N_act = N/2.
        # Use initial_c iterator as backing storage; only layout/shape matter.
        act_smem_layout = cute.slice_(self.act_smem_layout_staged.outer, (None, None, 0))
        n_act = cute.size(initial_c.shape, mode=[1]) // 2
        # NOTE: The TMA tensor used for partitioning must be statically large enough
        # for all tile coordinates the scheduler may generate. Our production RDEP
        # capacity is 65536 and padding is 128-aligned.
        m_act = 512 * self.cluster_tile_shape_mnk[0]  # 512*128 = 65536 rows
        initial_act = cute.make_tensor(
            initial_c.iterator,
            cute.make_layout((m_act, n_act), stride=(n_act, 1)),
        )
        tma_atom_act, tma_tensor_act = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), initial_act, act_smem_layout, self.act_epi_tile
        )

        # Grid setup
        self.tile_sched_params, grid = self._compute_grid(
            total_num_clusters, self.cluster_shape_mn, max_active_clusters
        )

        self.buffer_align_bytes = 1024
        self.size_tensormap_in_i64 = (
            NmoeGroupedScaledGemmKernel.GROUPED_NUM_TENSOR_MAPS * NmoeGroupedScaledGemmKernel.GROUPED_TENSOR_MAP_BYTES // 8
        )

        # Shared storage declaration
        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[cutlass.Int64, self.size_tensormap_in_i64]
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)], self.buffer_align_bytes
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)], self.buffer_align_bytes
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)], self.buffer_align_bytes
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)], self.buffer_align_bytes
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)], self.buffer_align_bytes
            ]

        # Define shared storage for kernel and launch
        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.shared_storage.size_in_bytes() > self.smem_capacity):
            raise ValueError(
                f"Shared storage uses {self.shared_storage.size_in_bytes()} bytes, "
                f"exceeds SM100 limit {self.smem_capacity} bytes."
            )

        # Launch the GPU kernel via CuTe DSL
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_act,
            tma_tensor_act,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.act_smem_layout_staged,
            self.epi_tile,
            self.act_epi_tile,
            self.tile_sched_params,
            group_count,
            problem_shape_mnkl,
            strides_abc,
            tensor_address_abc,
            tensor_address_sfasfb,
            a_base_ptr,
            a_row_bytes,
            out_act_base_ptr,
            out_sf_base_ptr,
            out_sf_expert_bytes,
            tensormap_cute_tensor,
        ).launch(
            grid=grid,
            block=(int(self.threads_per_cta), 1, 1),
            cluster=(int(self.cluster_shape_mn[0]), int(self.cluster_shape_mn[1]), 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )
        return

    #  GPU device kernel (vendored from CUTLASS example 92)
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tma_atom_act: cute.CopyAtom,
        mAct_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        act_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        act_epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        group_count: cutlass.Constexpr,
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_address_sfasfb: cute.Tensor,
        a_base_ptr: cutlass.Int64,
        a_row_bytes: cutlass.Int32,
        out_act_base_ptr: cutlass.Int64,
        out_sf_base_ptr: cutlass.Int64,
        out_sf_expert_bytes: cutlass.Int32,
        tensormaps: cute.Tensor,
    ):
        # Prefetch descriptors on TMA warp
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == self.tma_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfb)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)
            if cutlass.const_expr(self.fuse_swiglu_quant):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_act)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # CTA/thread coordinates
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        # Shared storage
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
        tensormap_a_smem_ptr = tensormap_smem_ptr
        tensormap_b_smem_ptr = (
            tensormap_a_smem_ptr + NmoeGroupedScaledGemmKernel.GROUPED_TENSOR_MAP_BYTES // 8
        )
        tensormap_sfa_smem_ptr = (
            tensormap_b_smem_ptr + NmoeGroupedScaledGemmKernel.GROUPED_TENSOR_MAP_BYTES // 8
        )
        tensormap_sfb_smem_ptr = (
            tensormap_sfa_smem_ptr + NmoeGroupedScaledGemmKernel.GROUPED_TENSOR_MAP_BYTES // 8
        )
        tensormap_c_smem_ptr = (
            tensormap_sfb_smem_ptr + NmoeGroupedScaledGemmKernel.GROUPED_TENSOR_MAP_BYTES // 8
        )
        tensormap_act_smem_ptr = (
            tensormap_c_smem_ptr + NmoeGroupedScaledGemmKernel.GROUPED_TENSOR_MAP_BYTES // 8
        )

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        # AB pipeline (TMA load)
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # ACC pipeline (UMMA accumulate)
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)
        cute.arch.mbarrier_init_fence()

        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        # Setup smem tensors
        # NOTE: CuTeDSL does not support scalar dereference of swizzled smem. For the
        # fused SwiGLU path we need scalar loads of (gate, up) pairs from smem, so
        # we use the *non-swizzled* view of the epilogue buffer.
        if cutlass.const_expr(self.fuse_swiglu_quant):
            sC = storage.sC.get_tensor(c_smem_layout_staged.outer)
        else:
            sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # Multicast masks
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        # Local tiles and partitions (global)
        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gSFA_mkl = cute.local_tile(mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFB_nkl = cute.local_tile(mSFB_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # TMA partition S/D for A/B/SF
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3)
        )
        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa, block_in_cluster_coord_vmnk[2], sfa_cta_layout, cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3)
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout, cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3)
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # Fragments and ACC tiler
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Cluster wait before alloc
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

        # Tensormap workspaces
        grid_dim = cute.arch.grid_dim()
        tensormap_workspace_idx = bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx
        tensormap_manager = utils.TensorMapManager(
            utils.TensorMapUpdateMode.SMEM, NmoeGroupedScaledGemmKernel.GROUPED_TENSOR_MAP_BYTES
        )
        tensormap_a_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 0, None)].iterator
        )
        tensormap_b_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 1, None)].iterator
        )
        tensormap_sfa_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 2, None)].iterator
        )
        tensormap_sfb_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 3, None)].iterator
        )
        tensormap_c_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 4, None)].iterator
        )
        tensormap_act_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 5, None)].iterator
        )

        # Specialized TMA warp: main producer loop
        if warp_idx == self.tma_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), grid_dim)
            group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
                self.group_count, tile_sched_params, self.cluster_tile_shape_mnk, utils.create_initial_search_state()
            )
            # Exact number of work tiles (clusters) from runtime problem sizes.
            # Host passes a capacity-based upper bound; we early-exit here.
            total_num_clusters_real = cutlass.Int32(0)
            tmp_problem_mnkl = cute.make_rmem_tensor(cute.make_layout(4), problem_sizes_mnkl.element_type)
            for g in cutlass.range(self.group_count, unroll_full=True):
                cute.autovec_copy(problem_sizes_mnkl[(g, None)], tmp_problem_mnkl)
                ntile_m = (tmp_problem_mnkl[0] + self.cluster_tile_shape_mnk[0] - 1) // self.cluster_tile_shape_mnk[0]
                ntile_n = (tmp_problem_mnkl[1] + self.cluster_tile_shape_mnk[1] - 1) // self.cluster_tile_shape_mnk[1]
                total_num_clusters_real += ntile_m * ntile_n
            tensormap_init_done = cutlass.Boolean(False)
            last_group_idx = cutlass.Int32(-1)
            work_tile = tile_sched.initial_work_tile_info()
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
            while work_tile.is_valid_tile & (work_tile.tile_idx[2] < total_num_clusters_real):
                cur_tile_coord = work_tile.tile_idx
                grouped_gemm_cta_tile_info = group_gemm_ts_helper.delinearize_z(cur_tile_coord, problem_sizes_mnkl)
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                is_group_changed = cur_group_idx != last_group_idx
                if is_group_changed:
                    real_tensor_a = self.make_tensor_abc_for_tensormap_update(
                        cur_group_idx,
                        self.a_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        tensor_address_abc,
                        0,
                    )
                    real_tensor_b = self.make_tensor_abc_for_tensormap_update(
                        cur_group_idx,
                        self.b_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        tensor_address_abc,
                        1,
                    )
                    real_tensor_sfa = self.make_tensor_sfasfb_for_tensormap_update(
                        cur_group_idx,
                        self.sf_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        tensor_address_sfasfb,
                        0,
                    )
                    real_tensor_sfb = self.make_tensor_sfasfb_for_tensormap_update(
                        cur_group_idx,
                        self.sf_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        tensor_address_sfasfb,
                        1,
                    )
                    if tensormap_init_done == False:
                        self.tensormap_ab_init_barrier.arrive_and_wait()
                        tensormap_init_done = True
                    tensormap_manager.update_tensormap(
                        (real_tensor_a, real_tensor_b, real_tensor_sfa, real_tensor_sfb),
                        (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb),
                        (tensormap_a_gmem_ptr, tensormap_b_gmem_ptr, tensormap_sfa_gmem_ptr, tensormap_sfb_gmem_ptr),
                        self.tma_warp_id,
                        (tensormap_a_smem_ptr, tensormap_b_smem_ptr, tensormap_sfa_smem_ptr, tensormap_sfb_smem_ptr),
                    )

                # Slice per MMA tile index
                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgSFB_slice = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # Peek AB buffer empty
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < cur_k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                if is_group_changed:
                    tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_b_gmem_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_sfa_gmem_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_sfb_gmem_ptr)

                # TMA load loop
                for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_a_gmem_ptr, cute.AddressSpace.generic
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_b_gmem_ptr, cute.AddressSpace.generic
                        ),
                    )
                    # SFA/SFB loads
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_sfa_gmem_ptr, cute.AddressSpace.generic
                        ),
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_sfb_gmem_ptr, cute.AddressSpace.generic
                        ),
                    )

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                # Advance to next tile (no producer_commit - producer_tail handles it)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            # Wait for all AB buffer empty
            ab_pipeline.producer_tail(ab_producer_state)

        # Specialized epilogue warps
        if warp_idx < self.mma_warp_id:
            # C tensormap is only needed for the non-fused path (TMA store).
            if cutlass.const_expr(not self.fuse_swiglu_quant):
                tensormap_manager.init_tensormap_from_atom(tma_atom_c, tensormap_c_smem_ptr, self.epilog_warp_id[0])

            # Alloc tensor memory buffer
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(self.num_tmem_alloc_cols, tmem_holding_buf, is_two_cta=use_2cta_instrs)

            # Bar sync for retrieve tensor memory ptr from shared memory
            self.tmem_alloc_barrier.arrive_and_wait()

            # Retrieve tensor memory ptr and make accumulator tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Partition for epilogue
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rC, epi_tidx, sC)
            if cutlass.const_expr(not self.fuse_swiglu_quant):
                tma_atom_c, bSG_sC, bSG_gC_partitioned = self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                )

            # Persistent tile scheduling loop
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), grid_dim)
            group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
                self.group_count, tile_sched_params, self.cluster_tile_shape_mnk, utils.create_initial_search_state()
            )
            work_tile = tile_sched.initial_work_tile_info()
            total_num_clusters_real = cutlass.Int32(0)
            tmp_problem_mnkl = cute.make_rmem_tensor(cute.make_layout(4), problem_sizes_mnkl.element_type)
            for g in cutlass.range(self.group_count, unroll_full=True):
                cute.autovec_copy(problem_sizes_mnkl[(g, None)], tmp_problem_mnkl)
                ntile_m = (tmp_problem_mnkl[0] + self.cluster_tile_shape_mnk[0] - 1) // self.cluster_tile_shape_mnk[0]
                ntile_n = (tmp_problem_mnkl[1] + self.cluster_tile_shape_mnk[1] - 1) // self.cluster_tile_shape_mnk[1]
                total_num_clusters_real += ntile_m * ntile_n
            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            # Threads/warps participating in TMA store pipeline
            c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilog_warp_id))
            c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)
            last_group_idx = cutlass.Int32(-1)

            while work_tile.is_valid_tile & (work_tile.tile_idx[2] < total_num_clusters_real):
                cur_tile_coord = work_tile.tile_idx
                grouped_gemm_cta_tile_info = group_gemm_ts_helper.delinearize_z(cur_tile_coord, problem_sizes_mnkl)
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                is_group_changed = cur_group_idx != last_group_idx

                if cutlass.const_expr(not self.fuse_swiglu_quant):
                    if is_group_changed:
                        # Construct tensor C based on real shape, stride information
                        real_tensor_c = self.make_tensor_abc_for_tensormap_update(
                            cur_group_idx,
                            self.c_dtype,
                            (
                                grouped_gemm_cta_tile_info.problem_shape_m,
                                grouped_gemm_cta_tile_info.problem_shape_n,
                                grouped_gemm_cta_tile_info.problem_shape_k,
                            ),
                            strides_abc,
                            tensor_address_abc,
                            2,
                        )
                        tensormap_manager.update_tensormap(
                            (real_tensor_c,), (tma_atom_c,), (tensormap_c_gmem_ptr,),
                            self.epilog_warp_id[0], (tensormap_c_smem_ptr,)
                        )

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k

                if cutlass.const_expr(self.fuse_swiglu_quant):
                    # Fused SwiGLU + quantization: write quantized post-activation
                    # (FP8 E4M3FN or NVFP4 E2M1 packed) + per-expert MMA-layout E8M0 SF
                    # directly to global memory (no intermediate BF16 activation tensor).
                    ptrA_i64 = tensor_address_abc[(cur_group_idx, 0)]
                    group_row0 = (ptrA_i64 - a_base_ptr) // cutlass.Int64(a_row_bytes)
                    n_h13 = grouped_gemm_cta_tile_info.problem_shape_n
                    dff = n_h13 // 2
                    sf_k = dff // cutlass.Int32(32)
                    c1 = cutlass.Int32(1)
                    c0 = cutlass.Int32(0)

                    # Output scale factors: packed-by-offs MMA layout (byte-swizzled, opaque).
                    out_sf_ptr = cute.make_ptr(
                        cutlass.Uint8, out_sf_base_ptr, cute.AddressSpace.gmem, assumed_align=16
                    )
                    # Packed layout: expert chunk starts at offs[e] * sf_k (row_bytes).
                    out_sf_ptr = out_sf_ptr + cutlass.Int64(group_row0) * cutlass.Int64(out_sf_expert_bytes)
                    # Flat view for explicit CUTLASS swizzle addressing.
                    sf_flat_len = grouped_gemm_cta_tile_info.problem_shape_m * (dff // cutlass.Int32(32))
                    gSF_flat = cute.make_tensor(
                        out_sf_ptr,
                        cute.make_layout((sf_flat_len,), stride=(c1,)),
                    )

                    # Output activations:
                    # - FP8:  [M_e, dff/2] uint16 (2 FP8 bytes per u16)
                    # - NVFP4:[M_e, dff/4] uint16 (4 FP4 nibbles per u16)
                    if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                        dff_u16 = dff // cutlass.Int32(2)
                        out_act_u16_ptr = cute.make_ptr(
                            cutlass.Uint16, out_act_base_ptr, cute.AddressSpace.gmem, assumed_align=16
                        )
                        out_act_u16_ptr = out_act_u16_ptr + cutlass.Int64(group_row0) * cutlass.Int64(dff_u16)
                        gAct_u16 = cute.make_tensor(
                            out_act_u16_ptr,
                            cute.make_layout(
                                (grouped_gemm_cta_tile_info.problem_shape_m, dff_u16),
                                stride=(dff_u16, c1),
                            ),
                        )
                    else:
                        dff_u16 = dff // cutlass.Int32(4)
                        out_act_u16_ptr = cute.make_ptr(
                            cutlass.Uint16, out_act_base_ptr, cute.AddressSpace.gmem, assumed_align=16
                        )
                        out_act_u16_ptr = out_act_u16_ptr + cutlass.Int64(group_row0) * cutlass.Int64(dff_u16)
                        gAct_u16 = cute.make_tensor(
                            out_act_u16_ptr,
                            cute.make_layout(
                                (grouped_gemm_cta_tile_info.problem_shape_m, dff_u16),
                                stride=(dff_u16, c1),
                            ),
                        )
                    m_tile0 = mma_tile_coord_mnl[0] * cutlass.Int32(self.cluster_tile_shape_mnk[0])
                    n_tile0_h13 = mma_tile_coord_mnl[1] * cutlass.Int32(self.cluster_tile_shape_mnk[1])
                else:
                    # Slice to per mma tile index
                    # ((ATOM_V, REST_V), EPI_M, EPI_N)
                    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

                # Wait for accumulator buffer full
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                if cutlass.const_expr(not self.fuse_swiglu_quant):
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                if cutlass.const_expr(not self.fuse_swiglu_quant):
                    if is_group_changed:
                        if warp_idx == self.epilog_warp_id[0]:
                            tensormap_manager.fence_tensormap_update(tensormap_c_gmem_ptr)

                # Store accumulator to global memory in subtiles
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                if cutlass.const_expr(self.fuse_swiglu_quant):
                    # Fused path: compute SwiGLU, compute one E8M0 SF per 32 outputs,
                    # and write quantized outputs directly to gmem.
                    if (out_act_base_ptr != cutlass.Int64(0)) & (out_sf_base_ptr != cutlass.Int64(0)):
                        lane = cute.arch.lane_idx()
                        lane16 = lane & cutlass.Int32(15)

                        # Shuffle within half-warps (two independent 16-lane groups).
                        # CuTeDSL shuffles take a packed `mask_and_clamp`:
                        #   - mask = (warp_size - width)
                        #   - clamp = (width - 1)
                        # packed as (mask << 8) | clamp.
                        # width=16 partitions the warp into two independent half-warps
                        # matching our 2-rows-per-warp mapping.
                        mask_and_clamp = cutlass.Int32((16 << 8) | 15)

                        # Process 128 rows: 4 epilogue warps * (2 rows/warp) * 16 iters = 128 rows.
                        iters = self.cluster_tile_shape_mnk[0] // (2 * len(self.epilog_warp_id))

                        for subtile_pair in range(subtile_cnt // 2):
                            subtile0 = subtile_pair * 2
                            subtile1 = subtile0 + 1

                            # Each subtile covers 32 H13 columns = 16 act cols.
                            # We quantize over 32 act cols (2 subtiles) to match sf_vec_size=32.
                            act_col0 = (n_tile0_h13 + cutlass.Int32(subtile0 * 32)) // 2
                            k_sf = act_col0 // cutlass.Int32(32)

                            # --- Load subtile0 into sC[cbuf0] ---
                            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile0)]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                            tRS_rC.store(acc_vec.to(self.c_dtype))
                            cbuf0 = (num_prev_subtiles + subtile0) % self.num_c_stage
                            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, cbuf0)])

                            # --- Load subtile1 into sC[cbuf1] ---
                            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile1)]
                            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                            tRS_rC.store(acc_vec.to(self.c_dtype))
                            cbuf1 = (num_prev_subtiles + subtile1) % self.num_c_stage
                            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, cbuf1)])

                            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                            self.epilog_sync_barrier.arrive_and_wait()

                            for it in range(iters):
                                row0 = warp_idx * cutlass.Int32(2) + cutlass.Int32(it * (2 * len(self.epilog_warp_id)))
                                hi = lane // 16  # 0 or 1
                                j = lane - hi * 16
                                row = row0 + hi
                                row_in_group = m_tile0 + row

                                v0 = cutlass.Float32(0.0)
                                v1 = cutlass.Float32(0.0)
                                v2 = cutlass.Float32(0.0)
                                v3 = cutlass.Float32(0.0)
                                gate0 = cutlass.Float32(0.0)
                                up0 = cutlass.Float32(0.0)
                                gate1 = cutlass.Float32(0.0)
                                up1 = cutlass.Float32(0.0)
                                gate2 = cutlass.Float32(0.0)
                                up2 = cutlass.Float32(0.0)
                                gate3 = cutlass.Float32(0.0)
                                up3 = cutlass.Float32(0.0)
                                if row_in_group < grouped_gemm_cta_tile_info.problem_shape_m:
                                    if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                                        # FP8: 16 lanes cover 32 outputs (2 outputs per lane).
                                        if j < cutlass.Int32(8):
                                            base = j * cutlass.Int32(4)
                                            gate0 = sC[(row, base + 0, cbuf0)].to(cutlass.Float32)
                                            up0   = sC[(row, base + 1, cbuf0)].to(cutlass.Float32)
                                            gate1 = sC[(row, base + 2, cbuf0)].to(cutlass.Float32)
                                            up1   = sC[(row, base + 3, cbuf0)].to(cutlass.Float32)
                                        else:
                                            jj = j - cutlass.Int32(8)
                                            base = jj * cutlass.Int32(4)
                                            gate0 = sC[(row, base + 0, cbuf1)].to(cutlass.Float32)
                                            up0   = sC[(row, base + 1, cbuf1)].to(cutlass.Float32)
                                            gate1 = sC[(row, base + 2, cbuf1)].to(cutlass.Float32)
                                            up1   = sC[(row, base + 3, cbuf1)].to(cutlass.Float32)
                                        v0 = _silu_f32(gate0) * up0
                                        v1 = _silu_f32(gate1) * up1
                                    else:
                                        # NVFP4: 8 lanes cover 32 outputs (4 outputs per lane),
                                        # avoiding cross-lane shuffles in the quant hot path.
                                        if j < cutlass.Int32(4):
                                            base = j * cutlass.Int32(8)
                                            gate0 = sC[(row, base + 0, cbuf0)].to(cutlass.Float32)
                                            up0   = sC[(row, base + 1, cbuf0)].to(cutlass.Float32)
                                            gate1 = sC[(row, base + 2, cbuf0)].to(cutlass.Float32)
                                            up1   = sC[(row, base + 3, cbuf0)].to(cutlass.Float32)
                                            gate2 = sC[(row, base + 4, cbuf0)].to(cutlass.Float32)
                                            up2   = sC[(row, base + 5, cbuf0)].to(cutlass.Float32)
                                            gate3 = sC[(row, base + 6, cbuf0)].to(cutlass.Float32)
                                            up3   = sC[(row, base + 7, cbuf0)].to(cutlass.Float32)
                                        elif j < cutlass.Int32(8):
                                            base = (j - cutlass.Int32(4)) * cutlass.Int32(8)
                                            gate0 = sC[(row, base + 0, cbuf1)].to(cutlass.Float32)
                                            up0   = sC[(row, base + 1, cbuf1)].to(cutlass.Float32)
                                            gate1 = sC[(row, base + 2, cbuf1)].to(cutlass.Float32)
                                            up1   = sC[(row, base + 3, cbuf1)].to(cutlass.Float32)
                                            gate2 = sC[(row, base + 4, cbuf1)].to(cutlass.Float32)
                                            up2   = sC[(row, base + 5, cbuf1)].to(cutlass.Float32)
                                            gate3 = sC[(row, base + 6, cbuf1)].to(cutlass.Float32)
                                            up3   = sC[(row, base + 7, cbuf1)].to(cutlass.Float32)

                                        v0 = _silu_f32(gate0) * up0
                                        v1 = _silu_f32(gate1) * up1
                                        v2 = _silu_f32(gate2) * up2
                                        v3 = _silu_f32(gate3) * up3

                                # Per-row amax over this 32-wide block.
                                abs0 = cute.arch.fmax(v0, -v0)
                                abs1 = cute.arch.fmax(v1, -v1)
                                abs2 = cute.arch.fmax(v2, -v2)
                                abs3 = cute.arch.fmax(v3, -v3)
                                amax = cute.arch.fmax(cute.arch.fmax(abs0, abs1), cute.arch.fmax(abs2, abs3))
                                for off in (8, 4, 2, 1):
                                    other = cute.arch.shuffle_sync_down(amax, offset=cutlass.Int32(off), mask_and_clamp=mask_and_clamp)
                                    amax = cute.arch.fmax(amax, other)

                                # Lane16==0 in each half-warp computes scale.
                                scale_byte = cutlass.Uint8(127)
                                inv_scale = cutlass.Float32(1.0)
                                if lane16 == cutlass.Int32(0):
                                    dtype_max = cutlass.Float32(448.0) if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN) else cutlass.Float32(6.0)
                                    scale = amax / dtype_max
                                    if scale <= cutlass.Float32(0.0):
                                        scale = cutlass.Float32(1.0)
                                    scale_byte = cutlass.Uint8(_e8m0_encode_from_pos_f32(scale))
                                    inv_scale = cutlass.Float32(_e8m0_inv_decode_to_f32(scale_byte))

                                inv_scale = cute.arch.shuffle_sync(inv_scale, offset=0, mask_and_clamp=mask_and_clamp)

                                if (lane16 == cutlass.Int32(0)) & (row_in_group < grouped_gemm_cta_tile_info.problem_shape_m):
                                    # Match nmoe/csrc/quant.cu: cutlass_sf_swizzle_offset(m, k_sf, M_e, sf_k).
                                    # This swizzle is the source of truth for the per-expert MMA SF layout.
                                    m = row_in_group
                                    k = k_sf
                                    m_32 = m % cutlass.Int32(32)
                                    m_4 = (m // cutlass.Int32(32)) % cutlass.Int32(4)
                                    m_rest = m // cutlass.Int32(128)
                                    k_4 = k % cutlass.Int32(4)
                                    k_rest = k // cutlass.Int32(4)
                                    rest_k = (dff // cutlass.Int32(32)) // cutlass.Int32(4)  # sf_k/4
                                    atom_offset = m_32 * cutlass.Int32(16) + m_4 * cutlass.Int32(4) + k_4
                                    atom_idx = m_rest * rest_k + k_rest
                                    dst = atom_idx * cutlass.Int32(512) + atom_offset
                                    gSF_flat[dst] = scale_byte

                                q0 = v0 * inv_scale
                                q1 = v1 * inv_scale
                                if row_in_group < grouped_gemm_cta_tile_info.problem_shape_m:
                                    if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                                        u16_idx = (act_col0 // cutlass.Int32(2)) + j
                                        if u16_idx < (dff // cutlass.Int32(2)):
                                            gAct_u16[(row_in_group, u16_idx)] = _fp8_pack2_e4m3(q0, q1)
                                    else:
                                        u16_idx = (act_col0 // cutlass.Int32(4)) + j
                                        if (j < cutlass.Int32(8)) & (u16_idx < (dff // cutlass.Int32(4))):
                                            q2 = v2 * inv_scale
                                            q3 = v3 * inv_scale
                                            gAct_u16[(row_in_group, u16_idx)] = _nvfp4_pack4_e2m1_u16(q0, q1, q2, q3)

                    # Ensure all stores are visible before proceeding.
                    self.epilog_sync_barrier.arrive_and_wait()
                else:
                    # Non-fused path: TMA store C to global memory.
                    for subtile_idx in range(subtile_cnt):
                        # Load accumulator from tensor memory buffer to register
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        # Convert to C type
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))

                        # Store C to shared memory
                        c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])

                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        self.epilog_sync_barrier.arrive_and_wait()

                        # TMA store C to global memory
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, subtile_idx)],
                                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                    tensormap_c_gmem_ptr, cute.AddressSpace.generic
                                ),
                            )
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                # Async arrive accumulator buffer empty
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            # Dealloc the tensor memory buffer
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            self.epilog_sync_barrier.arrive_and_wait()
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs)

            # Wait for C store complete
            c_pipeline.producer_tail()

        # Specialized MMA warp
        if warp_idx == self.mma_warp_id:
            # Initialize tensormaps for A, B, SFA and SFB
            tensormap_manager.init_tensormap_from_atom(tma_atom_a, tensormap_a_smem_ptr, self.mma_warp_id)
            tensormap_manager.init_tensormap_from_atom(tma_atom_b, tensormap_b_smem_ptr, self.mma_warp_id)
            tensormap_manager.init_tensormap_from_atom(tma_atom_sfa, tensormap_sfa_smem_ptr, self.mma_warp_id)
            tensormap_manager.init_tensormap_from_atom(tma_atom_sfb, tensormap_sfb_smem_ptr, self.mma_warp_id)
            # Signal tensormap initialization has finished
            self.tensormap_ab_init_barrier.arrive_and_wait()

            # Bar sync for retrieve tensor memory ptr from shared mem
            self.tmem_alloc_barrier.arrive_and_wait()

            # Retrieve tensor memory ptr and make accumulator/SFA/SFB tensors
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # Partition for S2T copy of SFA/SFB
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            # Persistent tile scheduling loop
            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), grid_dim)
            group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
                self.group_count, tile_sched_params, self.cluster_tile_shape_mnk, utils.create_initial_search_state()
            )
            work_tile = tile_sched.initial_work_tile_info()
            total_num_clusters_real = cutlass.Int32(0)
            tmp_problem_mnkl = cute.make_rmem_tensor(cute.make_layout(4), problem_sizes_mnkl.element_type)
            for g in cutlass.range(self.group_count, unroll_full=True):
                cute.autovec_copy(problem_sizes_mnkl[(g, None)], tmp_problem_mnkl)
                ntile_m = (tmp_problem_mnkl[0] + self.cluster_tile_shape_mnk[0] - 1) // self.cluster_tile_shape_mnk[0]
                ntile_n = (tmp_problem_mnkl[1] + self.cluster_tile_shape_mnk[1] - 1) // self.cluster_tile_shape_mnk[1]
                total_num_clusters_real += ntile_m * ntile_n
            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            while work_tile.is_valid_tile & (work_tile.tile_idx[2] < total_num_clusters_real):
                cur_tile_coord = work_tile.tile_idx
                # MMA warp is only interested in number of tiles along K dimension
                cur_k_tile_cnt, cur_group_idx = group_gemm_ts_helper.search_cluster_tile_count_k(
                    cur_tile_coord, problem_sizes_mnkl
                )

                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < cur_k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                # Wait for accumulator buffer empty
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                # Reset the ACCUMULATE field for each tile
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                # MMA mainloop
                for k_tile in range(cur_k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        # Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged, tCtSFA_compact_s2t)
                        cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_staged, tCtSFB_compact_s2t)

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)

                            cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < cur_k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                # Async arrive accumulator buffer full
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                # Advance to next tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Wait for accumulator buffer empty
            acc_pipeline.producer_tail(acc_producer_state)

    @cute.jit
    def make_tensor_abc_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_index: int,
    ):
        ptr_i64 = tensor_address_abc[(group_idx, tensor_index)]
        if cutlass.const_expr(not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)):
            raise TypeError(f"dtype must be a type of cutlass.Numeric, got {type(dtype)}")
        tensor_gmem_ptr = cute.make_ptr(dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16)
        strides_tensor_gmem = strides_abc[(group_idx, tensor_index, None)]
        strides_tensor_reg = cute.make_rmem_tensor(cute.make_layout(2), strides_abc.element_type)
        cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
        stride_mn = strides_tensor_reg[0]
        stride_k = strides_tensor_reg[1]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)
        if cutlass.const_expr(tensor_index == 0):
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr, cute.make_layout((m, k, c1), stride=(stride_mn, stride_k, c0))
            )
        elif cutlass.const_expr(tensor_index == 1):
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr, cute.make_layout((n, k, c1), stride=(stride_mn, stride_k, c0))
            )
        else:
            m = problem_shape_mnk[0]
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr, cute.make_layout((m, n, c1), stride=(stride_mn, stride_k, c0))
            )

    @cute.jit
    def make_tensor_sfasfb_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        tensor_address_sfasfb: cute.Tensor,
        tensor_index: int,
    ):
        ptr_i64 = tensor_address_sfasfb[(group_idx, tensor_index)]
        if cutlass.const_expr(not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)):
            raise TypeError(f"dtype must be a type of cutlass.Numeric, got {type(dtype)}")
        tensor_gmem_ptr = cute.make_ptr(dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16)
        c1 = cutlass.Int32(1)
        if cutlass.const_expr(tensor_index == 0):
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            sfa_layout = blockscaled_utils.tile_atom_to_shape_SF((m, k, c1), self.sf_vec_size)
            return cute.make_tensor(tensor_gmem_ptr, sfa_layout)
        else:
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF((n, k, c1), self.sf_vec_size)
            return cute.make_tensor(tensor_gmem_ptr, sfb_layout)

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype, epi_tile, use_2cta_instrs
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        gC_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tma_atom_c = atom
        sC_for = cute.group_modes(sC, 0, 2)
        gC_for = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(tma_atom_c, 0, cute.make_layout(1), sC_for, gC_for)
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2
        num_c_stage = 2
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage
        num_ab_stage = (smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)) // ab_bytes_per_stage
        num_c_stage += (
            smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        total_num_clusters: int,
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: int,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )
        tile_sched_params = utils.PersistentTileSchedulerParams(problem_shape_ntile_mnl, (*cluster_shape_mn, 1))
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)
        return tile_sched_params, grid

    @staticmethod
    def _get_mbar_smem_bytes(**kwargs_stages: int) -> int:
        num_barriers_per_stage = 2
        num_bytes_per_barrier = 8
        return sum(num_barriers_per_stage * num_bytes_per_barrier * stage for stage in kwargs_stages.values())


# (legacy weight-cache path removed; single production path is below)


# ==============================================================================
# Strided API: GPU-only metadata building, no Python loops
# ==============================================================================

# Separate compile cache for strided path (different compile key structure)
# NOTE: max_clusters is NOT included - the compiled kernel works for any cluster
# count up to the workspace size. Workspace is separately managed.
@dataclass(frozen=True)
class _StridedCompileKey:
    device_index: int
    sm_arch: Tuple[int, int]
    profile: str
    c_dtype_name: str
    group_count: int
    mma_tiler_mn: tuple[int, int]
    cluster_shape_mn: tuple[int, int]
    use_2cta: bool
    fuse_swiglu_quant: bool

_STRIDED_COMPILE_CACHE: dict[_StridedCompileKey, Tuple] = {}

# ==============================================================================
# Expert MLP scratch (blockscaled)
# ==============================================================================

@dataclass
class _ExpertScratch:
    """Blockscaled expert scratch (capacity-bounded, reuse-only)."""

    M_cap: int
    # Fused W13 epilogue outputs
    # fp8:   uint16 [M_cap, Dff//2] (2 FP8 bytes per u16)
    # nvfp4: uint8  [M_cap, Dff//2] (2 FP4 codes per byte)
    A_act: torch.Tensor
    # Post-activation SFA (packed by offs): uint8 [M_cap, sf_k_out] MMA layout
    A_sf: torch.Tensor


_EXPERT_SCRATCH: dict[tuple[int, str, int, int, int], _ExpertScratch] = {}


def run_grouped_blockscaled_strided(
    A_pad: torch.Tensor,           # [M_pad, K_packed, 1] quantized activations
    SFA_pad: torch.Tensor,         # [M_pad, sf_k_pad] uint8 E8M0 scale factors (packed by offs, MMA swizzled)
    B_stacked: torch.Tensor,       # [E, N, K_packed, 1] stacked weights
    SFB_stacked: torch.Tensor,     # [E, N, sf_k, 1] uint8 E8M0 scale factors (MMA swizzled)
    C_pad: torch.Tensor,           # [M_pad, N, 1] output (pre-allocated)
    offs: torch.Tensor,            # [E+1] int32 cumulative offsets (GPU)
    *,
    profile: str,                  # "fp8" or "nvfp4"
    N: int,                        # Output dimension (same for all experts)
    K: int,                        # Input dimension (in elements, not packed)
    fuse_swiglu_quant: bool = False,
    out_act: torch.Tensor | None = None,     # fp8: [M_pad, Dff//2] uint16; nvfp4: [M_pad, Dff//2] uint8
    out_sf_mma: torch.Tensor | None = None,  # [M_pad, sf_k] uint8 MMA layout (packed by offs)
) -> None:
    """Strided grouped blockscaled GEMM (GPU metadata builder).

    This function builds GEMM metadata on GPU using a CUDA kernel. Tile
    scheduling uses a capacity-based upper bound and exits early based on
    runtime problem sizes (no CPU synchronization).

    Requirements:
    - Weights must be stacked: B_stacked[e] is the weight for expert e
    - Scale factors must be pre-swizzled to MMA layout (uint8 E8M0)
    - Output C_pad must be CUDA bf16/fp16; it may be a small dummy when fuse_swiglu_quant=True.
    - All experts have same N, K dimensions (only M varies per expert)

    When fuse_swiglu_quant=True, the kernel uses the vendored fused epilogue
    path to compute SwiGLU + quantize/pack + E8M0 SF in one pass, writing:
      - out_act: quantized post-activation (FP8 or NVFP4 packed)
      - out_sf_mma: per-expert MMA-layout E8M0 scale factors per 32 outputs
    This mode is only valid for the W13 GEMM (interleaved gate/up columns).
    """
    E = B_stacked.shape[0]
    M_pad = A_pad.shape[0]
    device = A_pad.device

    if E == 0 or M_pad == 0:
        return

    # Get CUDA stream (must match current PyTorch stream)
    torch_stream = torch.cuda.current_stream(device)
    cu_stream = cuda.CUstream(torch_stream.cuda_stream)

    # Profile-specific settings
    if profile == "fp8":
        ab_dtype = cutlass.Float8E4M3FN
        sf_dtype = cutlass.Float8E8M0FNU
        sf_vec_size = 32
        pack_factor = 1
    elif profile == "nvfp4":
        ab_dtype = cutlass.Float4E2M1FN
        sf_dtype = cutlass.Float8E8M0FNU
        # Our NVFP4 quantization uses per-32 BF16 (E8M0) scale factors.
        sf_vec_size = 32
        pack_factor = 2  # 2 FP4 elements per byte
    else:
        raise ValueError("profile must be 'fp8' or 'nvfp4'")

    c_dtype_cutlass = cutlass.BFloat16 if C_pad.dtype == torch.bfloat16 else cutlass.Float16

    # One clear path: single tiler + cluster shape + tensormap strategy.
    mma_tiler_mn = (128, 128)
    cluster_shape_mn = (1, 1)
    tensormap_update_smem = True
    use_2cta = False

    # Tile sizes used for total_num_clusters sizing.
    ct_m = mma_tiler_mn[0] * cluster_shape_mn[0]
    ct_n = mma_tiler_mn[1] * cluster_shape_mn[1]

    # Packed SFA layout (production): [M_pad, sf_k_pad] MMA-swizzled bytes, where
    # each expert's swizzled chunk is packed back-to-back using offs (no E*capacity allocation).
    if not (SFA_pad.is_cuda and SFA_pad.dtype == torch.uint8 and SFA_pad.ndim in (2, 3)):
        raise ValueError(
            "SFA_pad must be a uint8 CUDA tensor (packed by offs, MMA layout). "
            f"Got device={SFA_pad.device} dtype={SFA_pad.dtype} shape={tuple(SFA_pad.shape)}."
        )
    if SFA_pad.ndim == 3:
        if int(SFA_pad.shape[2]) != 1:
            raise ValueError(f"SFA_pad must have trailing singleton dim if 3D. Got shape={tuple(SFA_pad.shape)}.")
        if int(SFA_pad.shape[0]) != int(M_pad):
            raise ValueError(f"SFA_pad.shape[0] must equal M_pad={M_pad}. Got {int(SFA_pad.shape[0])}.")
        # Drop last dim for simpler checks/strides (no copy).
        SFA_pad = SFA_pad.squeeze(-1)
    else:
        if int(SFA_pad.shape[0]) != int(M_pad):
            raise ValueError(f"SFA_pad.shape[0] must equal M_pad={M_pad}. Got {int(SFA_pad.shape[0])}.")
    SFA_row_bytes = int(SFA_pad.stride(0))  # uint8 => element stride is byte stride

    out_act_base_ptr = 0
    out_sf_base_ptr = 0
    out_sf_expert_bytes = 0
    if fuse_swiglu_quant:
        if (N & 1) != 0:
            raise ValueError(f"fuse_swiglu_quant requires N=2*Dff. Got N={N}.")
        dff = N // 2
        sf_k_out = (dff + 31) // 32
        if (sf_k_out & 3) != 0:
            raise ValueError(f"fuse_swiglu_quant requires Dff multiple of 128. Got Dff={dff}.")

        if out_act is None:
            raise ValueError("fuse_swiglu_quant=True requires out_act.")
        if out_sf_mma is None:
            raise ValueError("fuse_swiglu_quant=True requires out_sf_mma.")

        if profile == "fp8":
            exp_dtype = torch.uint16
        else:
            exp_dtype = torch.uint8
        exp_shape1 = dff // 2
        if not (
            out_act.is_cuda
            and out_act.dtype == exp_dtype
            and out_act.ndim == 2
            and int(out_act.shape[0]) == int(M_pad)
            and int(out_act.shape[1]) == int(exp_shape1)
            and out_act.is_contiguous()
        ):
            raise ValueError(f"out_act must be contiguous CUDA {exp_dtype} tensor with shape [M_pad, {exp_shape1}].")

        if not (
            out_sf_mma.is_cuda
            and out_sf_mma.dtype == torch.uint8
            and out_sf_mma.ndim == 2
            and int(out_sf_mma.shape[0]) == int(M_pad)
            and int(out_sf_mma.shape[1]) == int(sf_k_out)
            and out_sf_mma.is_contiguous()
        ):
            raise ValueError("out_sf_mma must be contiguous uint8 CUDA tensor with shape [M_pad, sf_k_out].")

        out_act_base_ptr = int(out_act.data_ptr())
        out_sf_base_ptr = int(out_sf_mma.data_ptr())
        out_sf_expert_bytes = int(out_sf_mma.stride(0))  # uint8 => row byte stride (packed by offs)

    # Tile-scheduler upper bound (for tensormap workspace sizing).
    #
    # RDEP bounds total received tokens by `capacity` (<= M_e_stride, which is
    # capacity-aligned), then pads each expert to 128 rows. Therefore total
    # padded M across all experts is:
    #   M_pad <= capacity + E*(128-1) <= M_e_stride + E*(128-1)
    #
    # The blockscaled scheduler iterates clusters over (M_tiles, N_tiles). For
    # grouped GEMM, the worst-case sum of ceil(M_e/ct_m) across experts is
    # upper-bounded by:
    #   sum_e ceil(M_e/ct_m) <= ceil((M_pad + E*(ct_m-1)) / ct_m)
    #
    # This is much tighter than E*ceil(M_e_stride/ct_m) and avoids pathological
    # JIT compile times for small runs on large-capacity buffers.
    tiles_n = (N + ct_n - 1) // ct_n
    # Tensormap workspace is indexed by CTA coordinates (gridDim.x*y*z), not by total tiles.
    # Keep this bounded by active CTAs to avoid multi-minute JITs / allocator spikes.
    max_clusters_cap = max(1, int(_num_sms()))

    # Exact number of work tiles for the persistent scheduler.
    # With RDEP-style padding, M_pad is 128-aligned and concatenates all experts
    # back-to-back, so the total number of M tiles is just M_pad / ct_m.
    if M_pad % ct_m != 0:
        raise ValueError(f"M_pad must be a multiple of {ct_m}, got {M_pad}.")
    tiles_m_total = M_pad // ct_m
    total_num_clusters = max(1, int(tiles_m_total) * int(tiles_n))

    # Device index for compile key
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    _require_sm100(device_index)
    sm_arch = torch.cuda.get_device_capability(device_index)

    # Build compile key (stable across batches - max_clusters NOT included).
    ckey = _StridedCompileKey(
        device_index=int(device_index),
        sm_arch=tuple(sm_arch),
        profile=profile,
        c_dtype_name=str(c_dtype_cutlass),
        group_count=int(E),
        mma_tiler_mn=tuple(mma_tiler_mn),
        cluster_shape_mn=tuple(cluster_shape_mn),
        use_2cta=bool(use_2cta),
        fuse_swiglu_quant=bool(fuse_swiglu_quant),
    )

    # Compile kernel if not cached
    compiled_tuple = _STRIDED_COMPILE_CACHE.get(ckey)
    if compiled_tuple is None:
        KernelCls = NmoeGroupedScaledGemmKernel

        # Create representative problem sizes for compilation (shapes are dynamic);
        # use a conservative reference M equal to one CTA tile in M.
        M_ref = max(ct_m, 1)
        problem_sizes = tuple((M_ref, N, K, 1) for _ in range(E))

        params = KernelCls.Params(
            a_dtype=ab_dtype,
            b_dtype=ab_dtype,
            acc_dtype=cutlass.Float32,
            c_dtype=c_dtype_cutlass,
            sf_dtype=sf_dtype,
            sf_vec_size=sf_vec_size,
            a_layout=cutlass.utils.LayoutEnum.ROW_MAJOR,
            b_layout=cutlass.utils.LayoutEnum.ROW_MAJOR,
            c_layout=cutlass.utils.LayoutEnum.ROW_MAJOR,
            use_2cta_instrs=bool(use_2cta),
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=tuple(cluster_shape_mn),
            update_tensormaps_in_smem=tensormap_update_smem,
            group_count=E,
            problem_sizes_mnkl_host=problem_sizes,
            max_active_clusters=_num_sms(),
            fuse_swiglu_quant=bool(fuse_swiglu_quant),
        )
        kernel = KernelCls(params)

        divisibility_ab = 32 if ab_dtype == cutlass.Float4E2M1FN else 16

        # Create initial tensors for compilation
        K_packed = K // pack_factor if profile == "nvfp4" else K
        init_a, _ = _create_initial_cute_tensor(1, M_ref, K_packed, False, ab_dtype, divisibility_ab)
        init_b, _ = _create_initial_cute_tensor(1, N, K_packed, False, ab_dtype, divisibility_ab)
        init_c, _ = _create_initial_cute_tensor(1, M_ref, N, False, c_dtype_cutlass, 16)

        # Create dummy SF tensors for compilation signature
        sf_k = (K + 31) // 32
        dummy_sfa = torch.zeros(M_ref, sf_k, 1, dtype=torch.uint8, device=device)
        dummy_sfb = torch.zeros(N, sf_k, 1, dtype=torch.uint8, device=device)
        init_sfa, _ = _wrap_uint8_as_sf(dummy_sfa, sf_dtype)
        init_sfb, _ = _wrap_uint8_as_sf(dummy_sfb, sf_dtype)

        # Metadata tensor shapes
        tensormap_shape = (
            max(1, max_clusters_cap),
            KernelCls.GROUPED_NUM_TENSOR_MAPS,
            KernelCls.GROUPED_TENSOR_MAP_BYTES // 8,
        )
        tensor_of_tensormap, _ = ctorch.cute_tensor_like(
            torch.empty(tensormap_shape, dtype=torch.int64, device=device),
            cutlass.Int64, is_dynamic_layout=False,
        )
        tensor_of_dim_size_mnkl, _ = ctorch.cute_tensor_like(
            torch.empty((E, 4), dtype=torch.int32, device=device),
            cutlass.Int32, is_dynamic_layout=False, assumed_align=16,
        )
        tensor_of_strides_abc, _ = ctorch.cute_tensor_like(
            torch.empty((E, 3, 2), dtype=torch.int32, device=device),
            cutlass.Int32, is_dynamic_layout=False, assumed_align=16,
        )
        tensor_of_ptrs_abc, _ = ctorch.cute_tensor_like(
            torch.empty((E, 3), dtype=torch.int64, device=device),
            cutlass.Int64, is_dynamic_layout=False, assumed_align=16,
        )
        tensor_of_ptrs_sfasfb, _ = ctorch.cute_tensor_like(
            torch.empty((E, 2), dtype=torch.int64, device=device),
            cutlass.Int64, is_dynamic_layout=False, assumed_align=16,
        )

        compiled = cute.compile(
            kernel, init_a, init_b, init_c, init_sfa, init_sfb,
            E, tensor_of_dim_size_mnkl, tensor_of_strides_abc,
            tensor_of_ptrs_abc, tensor_of_ptrs_sfasfb,
            cutlass.Int64(0), cutlass.Int32(0), cutlass.Int64(0), cutlass.Int64(0), cutlass.Int32(0),
            max_clusters_cap,
            tensor_of_tensormap, params.max_active_clusters, cu_stream,
            options="--opt-level 2",
        )
        _STRIDED_COMPILE_CACHE[ckey] = (compiled, init_a, init_b, init_c, init_sfa, init_sfb, params.max_active_clusters)
        compiled_tuple = _STRIDED_COMPILE_CACHE[ckey]

    compiled, init_a, init_b, init_c, init_sfa, init_sfb, max_active_clusters = compiled_tuple

    # Allocate metadata tensors on GPU — reuse cached workspace
    KernelCls = NmoeGroupedScaledGemmKernel
    (
        tensormap_cute,
        dim_size_mnkl_cute, dim_size_mnkl_torch,
        strides_abc_cute, strides_abc_torch,
        ptrs_abc_cute, ptrs_abc_torch,
        ptrs_sfasfb_cute, ptrs_sfasfb_torch,
    ) = _get_strided_workspace(device, E, max_clusters_cap, KernelCls)

    # =========================================================================
    # KEY OPTIMIZATION: Build metadata on GPU using CUDA kernel - NO CPU SYNC!
    # =========================================================================

    # Compute strides for the CUDA kernel
    # IMPORTANT: The kernel needs TWO types of strides:
    # 1. BYTE strides for pointer arithmetic (adding offsets to base pointers)
    # 2. ELEMENT strides for CUTLASS (stored in strides_abc output)
    #
    # PyTorch tensor.stride() returns element counts.
    # - For uint8-backed tensors (FP8, NVFP4, SF), element=byte, so stride IS byte stride
    # - For bf16-backed tensors (C), need to multiply by 2 to get byte stride

    # Element strides (for CUTLASS strides_abc)
    if profile == "nvfp4":
        # NVFP4: stored as uint8 with 2 elements per byte, CUTLASS expects element strides
        A_stride0_elem = A_pad.stride(0) * 2  # Convert packed stride to FP4 element units
        B_stride0_elem = B_stacked.stride(1) * 2
    else:
        # FP8: 1 byte per element
        A_stride0_elem = A_pad.stride(0)
        B_stride0_elem = B_stacked.stride(1)

    # stride(1) is the column stride, always 1 element (no multiplier needed)
    A_stride1_elem = A_pad.stride(1)
    B_stride1_elem = B_stacked.stride(2)
    C_stride0_elem = C_pad.stride(0)  # bf16 - already element stride
    C_stride1_elem = C_pad.stride(1)  # bf16 - already element stride

    # Byte strides (for pointer arithmetic in CUDA kernel)
    # For uint8-backed (A, B, SFA, SFB): stride IS byte stride
    # For bf16-backed (C): byte_stride = element_stride * 2
    A_row_bytes = A_pad.stride(0)  # uint8 backed, stride is bytes
    B_expert_bytes = B_stacked.stride(0)  # uint8 backed, stride is bytes
    # C is ignored when fuse_swiglu_quant=True; use 0 strides to keep pointer arithmetic in-bounds.
    C_row_bytes = 0 if fuse_swiglu_quant else (C_pad.stride(0) * 2)
    SFB_expert_bytes = SFB_stacked.stride(0)  # uint8 backed

    # Call our CUDA kernel to build metadata on GPU
    # Argument order must match binding:
    # offs_ptr, E,
    # A_base, A_row_bytes, B_base, B_expert_bytes, C_base, C_row_bytes,
    # SFA_base, SFA_row_bytes, SFB_base, SFB_expert_bytes,
    # A_stride0_elem, A_stride1_elem, B_stride0_elem, B_stride1_elem, C_stride0_elem, C_stride1_elem,
    # N, K, sizes_ptr, strides_ptr, ptrs_abc_ptr, ptrs_sfasfb_ptr, stream

    rdep.build_grouped_gemm_metadata(
        offs.data_ptr(), E,
        # Byte strides for pointer arithmetic
        A_pad.data_ptr(), A_row_bytes,
        B_stacked.data_ptr(), B_expert_bytes,
        C_pad.data_ptr(), C_row_bytes,
        SFA_pad.data_ptr(), SFA_row_bytes,  # packed SFA: base + offs[e]*row_bytes
        SFB_stacked.data_ptr(), SFB_expert_bytes,
        # Element strides for CUTLASS
        A_stride0_elem, A_stride1_elem,
        B_stride0_elem, B_stride1_elem,
        (0 if fuse_swiglu_quant else C_stride0_elem),
        (0 if fuse_swiglu_quant else C_stride1_elem),
        N, K,
        dim_size_mnkl_torch.data_ptr(),
        strides_abc_torch.data_ptr(),
        ptrs_abc_torch.data_ptr(),
        ptrs_sfasfb_torch.data_ptr(),
        torch_stream,
    )

    # NOTE: CuTeDSL 4.3.4 requires explicit typing for scalar args to avoid
    # pointer truncation/mis-marshalling in the JIT launcher.
    compiled(
        init_a, init_b, init_c, init_sfa, init_sfb,
        E, dim_size_mnkl_cute, strides_abc_cute,
        ptrs_abc_cute, ptrs_sfasfb_cute,
        cutlass.Int64(A_pad.data_ptr()), cutlass.Int32(A_row_bytes),
        cutlass.Int64(out_act_base_ptr),
        cutlass.Int64(out_sf_base_ptr),
        cutlass.Int32(out_sf_expert_bytes),
        cutlass.Int32(total_num_clusters),
        tensormap_cute, max_active_clusters, cu_stream,
    )


@dataclass
class QuantizedWeightsFused:
    """Quantized weights with fused W13 for Option B architecture.

    W13 interleaves W1 and W3 columns: [gate0, up0, gate1, up1, ...]
    This enables single GEMM for both gate and up projections.
    """
    # Fused gate+up weights [E, 2*Dff, H_packed, 1] interleaved
    W13_q: torch.Tensor
    W13_sf_mma: torch.Tensor
    # Down projection [E, H, Dff_packed, 1]
    W2_q: torch.Tensor
    W2_sf_mma: torch.Tensor
    # Dimensions
    E: int
    H: int
    Dff: int
    profile: str


def quantize_weights(
    W1: torch.Tensor,
    W3: torch.Tensor,
    W2: torch.Tensor,
    profile: str = 'nvfp4',
) -> QuantizedWeightsFused:
    """Quantize weights with W13 interleaved for fused GEMM+SwiGLU path.

    Creates interleaved W13 where columns alternate: [gate0, up0, gate1, up1, ...]
    This enables single GEMM followed by fused SwiGLU+Quantize.

    Args:
        W1: [E, H, Dff] BF16 gate weights
        W3: [E, H, Dff] BF16 up weights
        W2: [E, Dff, H] BF16 down weight
        profile: 'fp8' or 'nvfp4'

    Returns:
        QuantizedWeightsFused with interleaved W13 and W2
    """
    # imports resolved at module top

    E, H, Dff = W1.shape
    if H % 128 != 0:
        raise ValueError(f"H must be a multiple of 128. Got H={H}.")
    if Dff % 128 != 0:
        raise ValueError(f"Dff must be a multiple of 128. Got Dff={Dff}.")
    if profile not in ("fp8", "nvfp4"):
        raise ValueError("profile must be 'fp8' or 'nvfp4'")

    # Use the fused quantize+pack kernel that writes SFA directly into the per-expert
    # MMA swizzle layout (no per-expert Python loops, no separate swizzle kernel).
    # We treat weights as expert-concatenated with fixed per-expert row count.
    stream = torch.cuda.current_stream(W1.device)

    # -------------------------------------------------------------------------
    # W13: interleave W1/W3 and transpose to [E, 2*Dff, H]
    # -------------------------------------------------------------------------
    M13 = 2 * Dff
    K13 = H
    sf_k13 = K13 // 32
    if (K13 & 127) != 0:
        raise ValueError(f"H must be a multiple of 128 for MMA SF swizzle. Got H={H}.")
    if (M13 & 127) != 0:
        raise ValueError(f"2*Dff must be a multiple of 128 for MMA SF swizzle. Got 2*Dff={M13}.")

    # [E, H, Dff] + [E, H, Dff] -> [E, H, 2*Dff] (interleaved last dim)
    # then transpose to [E, 2*Dff, H] for B in A @ B.T
    W13_t = torch.stack((W1, W3), dim=-1).view(E, H, M13).transpose(1, 2).contiguous()
    W13_x = W13_t.view(E * M13, K13)
    offs13 = torch.arange(0, (E + 1) * M13, step=M13, device=W1.device, dtype=torch.int32)

    if profile == "fp8":
        W13_u16 = torch.empty((E * M13, K13 // 2), device=W1.device, dtype=torch.uint16)
    else:
        W13_u16 = torch.empty((E * M13, K13 // 4), device=W1.device, dtype=torch.uint16)

    W13_sf_mma = torch.empty((E, M13, sf_k13), device=W1.device, dtype=torch.uint8)
    if profile == "fp8":
        rdep.quant_fp8_sf_strided_mma(
            W13_x.data_ptr(), K13,
            W13_u16.data_ptr(), K13 // 2,
            W13_sf_mma.data_ptr(),
            offs13.data_ptr(),
            E, M13,
            E * M13, K13,
            stream,
        )
        W13_q = W13_u16.view(torch.uint8).view(E, M13, K13, 1).view(torch.float8_e4m3fn)
    else:
        rdep.quant_nvfp4_sf_strided_mma(
            W13_x.data_ptr(), K13,
            W13_u16.data_ptr(), K13 // 4,
            W13_sf_mma.data_ptr(),
            offs13.data_ptr(),
            E, M13,
            E * M13, K13,
            stream,
        )
        W13_q = W13_u16.view(torch.uint8).view(E, M13, K13 // 2, 1)

    W13_sf_mma = W13_sf_mma.view(E, M13, sf_k13, 1)

    # -------------------------------------------------------------------------
    # W2: transpose to [E, H, Dff] for B in A @ B.T
    # -------------------------------------------------------------------------
    M2 = H
    K2 = Dff
    sf_k2 = K2 // 32
    if (K2 & 127) != 0:
        raise ValueError(f"Dff must be a multiple of 128 for MMA SF swizzle. Got Dff={Dff}.")
    if (M2 & 127) != 0:
        raise ValueError(f"H must be a multiple of 128 for MMA SF swizzle. Got H={H}.")

    W2_t = W2.transpose(1, 2).contiguous()  # [E, H, Dff]
    W2_x = W2_t.view(E * M2, K2)
    offs2 = torch.arange(0, (E + 1) * M2, step=M2, device=W1.device, dtype=torch.int32)

    if profile == "fp8":
        W2_u16 = torch.empty((E * M2, K2 // 2), device=W1.device, dtype=torch.uint16)
    else:
        W2_u16 = torch.empty((E * M2, K2 // 4), device=W1.device, dtype=torch.uint16)

    W2_sf_mma = torch.empty((E, M2, sf_k2), device=W1.device, dtype=torch.uint8)
    if profile == "fp8":
        rdep.quant_fp8_sf_strided_mma(
            W2_x.data_ptr(), K2,
            W2_u16.data_ptr(), K2 // 2,
            W2_sf_mma.data_ptr(),
            offs2.data_ptr(),
            E, M2,
            E * M2, K2,
            stream,
        )
        W2_q = W2_u16.view(torch.uint8).view(E, M2, K2, 1).view(torch.float8_e4m3fn)
    else:
        rdep.quant_nvfp4_sf_strided_mma(
            W2_x.data_ptr(), K2,
            W2_u16.data_ptr(), K2 // 4,
            W2_sf_mma.data_ptr(),
            offs2.data_ptr(),
            E, M2,
            E * M2, K2,
            stream,
        )
        W2_q = W2_u16.view(torch.uint8).view(E, M2, K2 // 2, 1)

    W2_sf_mma = W2_sf_mma.view(E, M2, sf_k2, 1)

    return QuantizedWeightsFused(
        W13_q=W13_q,
        W13_sf_mma=W13_sf_mma,
        W2_q=W2_q,
        W2_sf_mma=W2_sf_mma,
        E=E,
        H=H,
        Dff=Dff,
        profile=profile,
    )


def expert_blockscaled(
    Xe_q_pad: torch.Tensor,
    Xe_sf_pad: torch.Tensor,
    W_cache: QuantizedWeightsFused,
    offs_pad: torch.Tensor,
    *,
    capacity_rows: int | None = None,
) -> torch.Tensor:
    """Expert MLP (blockscaled) for RDEP-produced packed activations + swizzled SFA.

    Contract (production):
      - Xe_q_pad: [M_pad, Hp] uint16 packed activations from RDEP dispatch
      - Xe_sf_pad: [M_pad, sf_k_pad] uint8 E8M0 SFA in MMA layout (packed by offs)
      - offs_pad: [E] int32 cumulative padded offsets (no leading 0)

    Implementation:
      - GEMM1+2 (W13): produces BF16 H13 = [gate, up] interleaved
      - Fused SwiGLU + quantize/pack: produces packed activations + packed-by-offs MMA SFA
      - GEMM3 (W2): consumes packed activations + MMA SFA and produces BF16 output
    """
    M_pad = int(Xe_q_pad.shape[0])
    E = int(W_cache.E)
    H = int(W_cache.H)
    Dff = int(W_cache.Dff)
    profile = W_cache.profile

    if M_pad == 0:
        return torch.zeros(0, H, device=Xe_q_pad.device, dtype=torch.bfloat16)

    # Production (moonlight): Xe_sf_pad is packed by offs as a 2D swizzled buffer
    # [M_pad, sf_k_pad], where each expert chunk is laid out back-to-back.
    if not (Xe_sf_pad.is_cuda and Xe_sf_pad.dtype == torch.uint8 and Xe_sf_pad.ndim in (2, 3)):
        raise ValueError(
            "Xe_sf_pad must be a uint8 CUDA tensor with shape [M_pad, sf_k_pad] (packed by offs, MMA layout). "
            f"Got device={Xe_sf_pad.device} dtype={Xe_sf_pad.dtype} shape={tuple(Xe_sf_pad.shape)}."
        )
    if Xe_sf_pad.ndim == 3:
        if int(Xe_sf_pad.shape[2]) != 1:
            raise ValueError(f"Xe_sf_pad must have trailing singleton dim if 3D. Got shape={tuple(Xe_sf_pad.shape)}.")
        Xe_sf_pad = Xe_sf_pad.squeeze(-1)
    if int(Xe_sf_pad.shape[0]) != int(M_pad):
        raise ValueError(f"Xe_sf_pad.shape[0] must equal M_pad={M_pad}. Got {int(Xe_sf_pad.shape[0])}.")

    device_index = Xe_q_pad.device.index if Xe_q_pad.device.index is not None else torch.cuda.current_device()
    # Input SFA is for K=H.
    sf_k_in = H // 32
    sf_k_in_pad = ((sf_k_in + 3) // 4) * 4
    if int(Xe_sf_pad.shape[1]) != int(sf_k_in_pad):
        raise ValueError(f"Xe_sf_pad.shape[1] must equal sf_k_in_pad={sf_k_in_pad}. Got {int(Xe_sf_pad.shape[1])}.")

    # Post-activation SFA is for K=Dff.
    sf_k_out = Dff // 32
    if (sf_k_out & 3) != 0:
        raise ValueError(f"Dff must be a multiple of 128 (sf_k_out%4==0). Got Dff={Dff}.")

    # Scratch is cached per (device, profile, E, H, Dff) and must not accumulate
    # multiple large buffers across routing/layers (HBM stability contract).
    #
    # When capacity_rows is provided, size scratch by the configured no-drop
    # capacity (routing-independent) to avoid per-step allocator growth when
    # M_pad fluctuates.
    align_m = 128
    M_cap_rows = int(M_pad)
    if capacity_rows is not None:
        cap = int(capacity_rows)
        if cap < 0:
            raise ValueError(f"capacity_rows must be >=0, got {cap}.")
        # RDEP pads each expert to 128 rows; total padded M is bounded by
        # capacity + E*(align_m-1). Add one extra (align_m-1) for final rounding.
        M_cap_rows = ((cap + E * (align_m - 1) + (align_m - 1)) // align_m) * align_m

    M_cap_needed = ((int(M_cap_rows) + (align_m - 1)) // align_m) * align_m
    scratch_key = (int(device_index), str(profile), int(E), int(H), int(Dff))
    scratch = _EXPERT_SCRATCH.get(scratch_key)
    if scratch is None or int(scratch.M_cap) < int(M_cap_needed):
        # Reuse-only when capacity_rows is set (allocator stability contract).
        if scratch is not None and capacity_rows is not None:
            raise RuntimeError(
                f"blockscaled expert scratch too small for configured capacity: "
                f"have M_cap={int(scratch.M_cap)} need>={int(M_cap_needed)} "
                f"(capacity_rows={int(capacity_rows)}, E={E}, align_m={align_m})."
            )

        # Otherwise, grow once to the needed upper bound (and replace in-cache),
        # making sure we drop the old buffer first to avoid transient double
        # allocation spikes.
        if scratch is not None:
            del _EXPERT_SCRATCH[scratch_key]
            scratch = None

        if profile == "fp8":
            A_act = torch.empty((M_cap_needed, Dff // 2), device=Xe_q_pad.device, dtype=torch.uint16)
        elif profile == "nvfp4":
            A_act = torch.empty((M_cap_needed, Dff // 2), device=Xe_q_pad.device, dtype=torch.uint8)
        else:
            raise ValueError(f"Unsupported profile: {profile}")
        A_sf = torch.empty((M_cap_needed, sf_k_out), device=Xe_q_pad.device, dtype=torch.uint8)
        scratch = _ExpertScratch(M_cap=M_cap_needed, A_act=A_act, A_sf=A_sf)
        _EXPERT_SCRATCH[scratch_key] = scratch

    # offs: [E+1] with leading 0.
    #
    # NOTE: Avoid CUDA scalar/slice assignment here: it is surprisingly expensive
    # (can introduce ms-scale launch-side overhead). A tiny cat is faster and
    # keeps the GPU fed.
    offs_pad_i32 = offs_pad if offs_pad.dtype == torch.int32 else offs_pad.to(torch.int32)
    offs = torch.cat((offs_pad_i32.new_zeros((1,)), offs_pad_i32), dim=0)

    # Convert packed activations to CUTLASS operand format
    if profile == "fp8":
        A_q = Xe_q_pad.view(torch.uint8).view(M_pad, H, 1).view(torch.float8_e4m3fn)
    elif profile == "nvfp4":
        A_q = Xe_q_pad.view(torch.uint8).view(M_pad, H // 2, 1)
    else:
        raise ValueError(f"Unsupported profile: {profile}")

    # GEMM 1+2 (W13) with fused SwiGLU + quantization (no BF16 H13 materialization).
    dummy_c = torch.empty((1, 1, 1), device=Xe_q_pad.device, dtype=torch.bfloat16)
    run_grouped_blockscaled_strided(
        A_q, Xe_sf_pad, W_cache.W13_q, W_cache.W13_sf_mma, dummy_c, offs,
        profile=profile, N=2 * Dff, K=H,
        fuse_swiglu_quant=True,
        out_act=scratch.A_act[:M_pad],
        out_sf_mma=scratch.A_sf[:M_pad],
    )

    if profile == "fp8":
        A_q_3 = scratch.A_act[:M_pad].view(torch.uint8).view(M_pad, Dff, 1).view(torch.float8_e4m3fn)
    elif profile == "nvfp4":
        A_q_3 = scratch.A_act[:M_pad].view(torch.uint8).view(M_pad, Dff // 2, 1)
    else:
        raise ValueError(f"Unsupported profile: {profile}")

    # GEMM 3: Y = A @ W2.T
    Y_pad = torch.empty((M_pad, H, 1), device=Xe_q_pad.device, dtype=torch.bfloat16)
    run_grouped_blockscaled_strided(
        A_q_3, scratch.A_sf[:M_pad], W_cache.W2_q, W_cache.W2_sf_mma, Y_pad, offs,
        profile=profile, N=H, K=Dff,
    )

    return Y_pad.squeeze(-1)
