"""Python wrapper for BF16 -> FP8/NVFP4 quantization kernels.

Uses quant.cu kernels via the rdep C-extension (nmoe.rdep._C).

Output formats:
  - FP8: out [M, K/2] uint16 (packed), sfa [M, ceil(K/32)] uint8
  - NVFP4: out [M, K/4] uint16 (packed), sfa [M, ceil(K/32)] uint8
"""
from __future__ import annotations

import torch
from nmoe.rdep import _C


def quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 tensor to FP8 E4M3 with E8M0 scale factors.

    Args:
        x: [M, K] BF16 tensor, K must be multiple of 32

    Returns:
        out: [M, K, 1] float8_e4m3fn for blockscaled runners
        sfa: [M, K//32, 1] uint8 E8M0 scale factors
    """
    assert x.dtype == torch.bfloat16, f"Expected BF16, got {x.dtype}"
    assert x.is_cuda and x.is_contiguous()
    M, K = x.shape
    assert K % 32 == 0, f"K must be multiple of 32, got {K}"

    out_u16 = torch.empty(M, K // 2, dtype=torch.uint16, device=x.device)
    sfa = torch.empty(M, K // 32, dtype=torch.uint8, device=x.device)

    _C.quant_fp8(
        x.data_ptr(), K,
        out_u16.data_ptr(), K // 2,
        sfa.data_ptr(), K // 32,
        M, K, torch.cuda.current_stream(x.device)
    )

    out = out_u16.view(torch.uint8).view(M, K, 1).view(torch.float8_e4m3fn)
    sfa = sfa.unsqueeze(-1)
    return out, sfa


def quantize_nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 tensor to NVFP4 E2M1 with E8M0 scale factors.

    Args:
        x: [M, K] BF16 tensor, K must be multiple of 32

    Returns:
        out: [M, K//2, 1] uint8 packed bytes (2 FP4 codes per byte)
        sfa: [M, K//32, 1] uint8 E8M0 scale factors
    """
    assert x.dtype == torch.bfloat16, f"Expected BF16, got {x.dtype}"
    assert x.is_cuda and x.is_contiguous()
    M, K = x.shape
    assert K % 32 == 0, f"K must be multiple of 32, got {K}"

    out_u16 = torch.empty(M, K // 4, dtype=torch.uint16, device=x.device)
    sfa = torch.empty(M, K // 32, dtype=torch.uint8, device=x.device)

    _C.quant_nvfp4(
        x.data_ptr(), K,
        out_u16.data_ptr(), K // 4,
        sfa.data_ptr(), K // 32,
        M, K, torch.cuda.current_stream(x.device)
    )

    out = out_u16.view(torch.uint8).view(M, K // 2, 1)
    sfa = sfa.unsqueeze(-1)
    return out, sfa
