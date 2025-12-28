import math

import pytest
import torch


def _require_sm100_and_deps() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required.")
    if torch.cuda.get_device_capability(0) != (10, 0):
        pytest.skip("SM100 (B200) required.")
    try:
        import cutlass  # noqa: F401
        import cuda.bindings.driver  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Required runtime deps missing: {e}")


@pytest.mark.parametrize(
    "profile",
    [
        "fp8",
        "nvfp4",
    ],
)
def test_expert_blockscaled_worst_case_routing_forward_matches_reference(profile: str) -> None:
    _require_sm100_and_deps()
    try:
        from nmoe.blockscaled.grouped import expert_blockscaled, quantize_weights, run_grouped_blockscaled_strided
        from nmoe.csrc import rdep
    except Exception as e:  # pragma: no cover
        pytest.skip(f"nmoe CUDA extension not available: {e}")

    torch.manual_seed(0)
    device = torch.device("cuda", 0)

    # Small, deterministic shapes that satisfy all alignment constraints.
    E = 4
    H = 256
    Dff = 512
    M_pad = 256  # must be multiple of 128

    x = torch.randn(M_pad, H, device=device, dtype=torch.bfloat16)

    # Pathological routing: all rows belong to expert 0; others get 0 rows.
    offs_pad = torch.full((E,), M_pad, device=device, dtype=torch.int32)

    # Quantize activations to packed format + packed MMA-layout SFA.
    sf_k_in = H // 32
    if profile == "fp8":
        Xe_q = torch.empty(M_pad, H // 2, device=device, dtype=torch.uint16)
        Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
        rdep.quant_fp8(
            x.data_ptr(),
            H,
            Xe_q.data_ptr(),
            H // 2,
            Xe_sf_mkl.data_ptr(),
            sf_k_in,
            M_pad,
            H,
            torch.cuda.current_stream(device),
        )
    else:
        Xe_q = torch.empty(M_pad, H // 4, device=device, dtype=torch.uint16)
        Xe_sf_mkl = torch.empty(M_pad, sf_k_in, device=device, dtype=torch.uint8)
        rdep.quant_nvfp4(
            x.data_ptr(),
            H,
            Xe_q.data_ptr(),
            H // 4,
            Xe_sf_mkl.data_ptr(),
            sf_k_in,
            M_pad,
            H,
            torch.cuda.current_stream(device),
        )

    Xe_sf_mma = torch.empty_like(Xe_sf_mkl)
    rdep.swizzle_sf_mkl_to_mma(
        Xe_sf_mkl.data_ptr(),
        Xe_sf_mma.data_ptr(),
        int(M_pad),
        int(sf_k_in),
        torch.cuda.current_stream(device),
    )
    Xe_sf_pad = Xe_sf_mma

    # Random expert weights; reference uses expert 0 only.
    W1 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
    W3 = torch.randn(E, H, Dff, device=device, dtype=torch.bfloat16) / math.sqrt(H)
    W2 = torch.randn(E, Dff, H, device=device, dtype=torch.bfloat16) / math.sqrt(Dff)

    W_cache = quantize_weights(W1, W3, W2, profile=profile)

    y = expert_blockscaled(Xe_q, Xe_sf_pad, W_cache, offs_pad, capacity_rows=M_pad)
    assert y.shape == (M_pad, H)
    assert torch.isfinite(y).all()

    # Reference path: materialize H13 (BF16), then run the production SwiGLU+quant kernel
    # to produce packed activations + MMA-layout SFA, then GEMM3.
    offs = torch.cat((offs_pad.new_zeros((1,)), offs_pad), dim=0)
    if profile == "fp8":
        A_q = Xe_q.view(torch.uint8).view(M_pad, H, 1).view(torch.float8_e4m3fn)
    else:
        A_q = Xe_q.view(torch.uint8).view(M_pad, H // 2, 1)

    H13 = torch.empty((M_pad, 2 * Dff, 1), device=device, dtype=torch.bfloat16)
    run_grouped_blockscaled_strided(
        A_q,
        Xe_sf_pad,
        W_cache.W13_q,
        W_cache.W13_sf_mma,
        H13,
        offs,
        profile=profile,
        N=2 * Dff,
        K=H,
    )

    sf_k_out = Dff // 32
    # Reference quant kernel writes SF to a per-expert strided buffer; production
    # GEMM3 consumes SF packed-by-offs. Build both.
    M_e_stride = M_pad  # test uses worst-case: expert0 has all rows, others 0
    A_sf_strided = torch.zeros((E, M_e_stride, sf_k_out), device=device, dtype=torch.uint8)
    A_sf_ref = torch.zeros((M_pad, sf_k_out), device=device, dtype=torch.uint8)
    if profile == "fp8":
        A_act_ref_u16 = torch.empty((M_pad, Dff // 2), device=device, dtype=torch.uint16)
        rdep.swiglu_quant_fp8_sf_strided_mma(
            H13.data_ptr(),
            2 * Dff,
            A_act_ref_u16.data_ptr(),
            Dff // 2,
            A_sf_strided.data_ptr(),
            offs.data_ptr(),
            E,
            M_e_stride,
            M_pad,
            Dff,
            torch.cuda.current_stream(device),
        )
        A_q_3 = A_act_ref_u16.view(torch.uint8).view(M_pad, Dff, 1).view(torch.float8_e4m3fn)
    else:
        A_act_ref_u16 = torch.empty((M_pad, Dff // 4), device=device, dtype=torch.uint16)
        rdep.swiglu_quant_nvfp4_sf_strided_mma(
            H13.data_ptr(),
            2 * Dff,
            A_act_ref_u16.data_ptr(),
            Dff // 4,
            A_sf_strided.data_ptr(),
            offs.data_ptr(),
            E,
            M_e_stride,
            M_pad,
            Dff,
            torch.cuda.current_stream(device),
        )
        A_q_3 = A_act_ref_u16.view(torch.uint8).view(M_pad, Dff // 2, 1)

    # Pack per-expert SF into packed-by-offs layout for GEMM3.
    offs_cpu = offs.to("cpu")
    for e in range(E):
        row0 = int(offs_cpu[e].item())
        row1 = int(offs_cpu[e + 1].item())
        if row1 <= row0:
            continue
        m = row1 - row0
        A_sf_ref[row0:row1].copy_(A_sf_strided[e, :m])

    y_ref = torch.empty((M_pad, H, 1), device=device, dtype=torch.bfloat16)
    run_grouped_blockscaled_strided(
        A_q_3,
        A_sf_ref,
        W_cache.W2_q,
        W_cache.W2_sf_mma,
        y_ref,
        offs,
        profile=profile,
        N=H,
        K=Dff,
    )
    y_ref = y_ref.squeeze(-1)
    assert torch.isfinite(y_ref).all()

    # Fused epilogue vs reference (materialize H13 + quant kernel) is expected to
    # differ slightly due to rounding/packing, especially under NVFP4.
    if profile == "fp8":
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 2e-2, 2e-2
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)
