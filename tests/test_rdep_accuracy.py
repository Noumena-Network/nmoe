import torch
import torch.nn.functional as F

from nmoe.rdep import Rdep


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("SKIP: CUDA not available")


def _require_sm100() -> None:
    _require_cuda()
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (10, 0):
        raise SystemExit(f"SKIP: requires sm_100 (got sm_{major}{minor})")


def _seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ref_moe_k1(
    x: torch.Tensor,  # [T, H] bf16
    eid: torch.Tensor,  # [T, 1] int32
    gates: torch.Tensor,  # [T, 1] bf16
    W1: torch.Tensor,  # [E, H, Dff] bf16
    W3: torch.Tensor,  # [E, H, Dff] bf16
    W2: torch.Tensor,  # [E, Dff, H] bf16
) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    assert eid.dtype == torch.int32
    assert gates.dtype == torch.bfloat16
    assert eid.shape[1] == 1
    assert gates.shape[1] == 1

    T, H = x.shape
    E = int(W1.shape[0])
    eid1 = eid[:, 0]
    g = gates[:, 0].float()

    out = torch.zeros((T, H), device=x.device, dtype=torch.float32)
    for e in range(E):
        idx = (eid1 == e).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        x_e = x.index_select(0, idx)
        h1 = x_e @ W1[e]
        h3 = x_e @ W3[e]
        y = (F.silu(h1).mul_(h3)) @ W2[e]
        out.index_add_(0, idx, y.float() * g.index_select(0, idx).unsqueeze(1))

    return out.to(dtype=torch.bfloat16)


def main() -> None:
    _seed(0)
    _require_cuda()

    device = torch.device("cuda")
    # Keep shapes tiny but aligned to the kernel contracts (H multiple of 128).
    T, H, Dff, E, K = 32, 128, 256, 4, 1

    # Scale inputs to realistic magnitudes (like bench_moe_e2e.py) to avoid
    # huge intermediate values that amplify quantization error in blockscaled tests.
    # Uses ~Xavier init scale (1/sqrt(fan_in)) to match production model behavior.
    # Scale before requires_grad to keep tensors as leaves for gradient tests.
    x = (torch.randn((T, H), device=device, dtype=torch.bfloat16) * 0.1).requires_grad_(True)
    eid = torch.randint(0, E, (T, K), device=device, dtype=torch.int32)
    gates = torch.ones((T, K), device=device, dtype=torch.bfloat16)
    W1 = (torch.randn((E, H, Dff), device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    W3 = (torch.randn((E, H, Dff), device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    W2 = (torch.randn((E, Dff, H), device=device, dtype=torch.bfloat16) * 0.02).requires_grad_(True)

    rdep = Rdep(dim=H, n_local=E, topk=K, profile="bf16", capacity=T * K)

    out = rdep.moe_bf16(x, eid, gates, W1, W3, W2)
    ref = _ref_moe_k1(x, eid, gates, W1, W3, W2)
    if not torch.equal(out, ref):
        max_abs = (out.float() - ref.float()).abs().max().item()
        raise AssertionError(f"BF16 forward mismatch: max_abs={max_abs}")

    loss = out.float().sum()
    loss.backward()

    x2 = x.detach().clone().requires_grad_(True)
    W12 = W1.detach().clone().requires_grad_(True)
    W32 = W3.detach().clone().requires_grad_(True)
    W22 = W2.detach().clone().requires_grad_(True)
    ref2 = _ref_moe_k1(x2, eid, gates, W12, W32, W22)
    ref2.float().sum().backward()

    torch.testing.assert_close(x.grad, x2.grad, atol=0, rtol=0)
    torch.testing.assert_close(W1.grad, W12.grad, atol=0, rtol=0)
    torch.testing.assert_close(W3.grad, W32.grad, atol=0, rtol=0)
    torch.testing.assert_close(W2.grad, W22.grad, atol=0, rtol=0)

    # Blockscaled forward sanity (requires sm_100).
    try:
        _require_sm100()
    except SystemExit:
        return

    from nmoe.blockscaled.grouped import quantize_weights

    for profile, atol in (("fp8", 1e-2), ("nvfp4", 5e-2)):
        rdep_bs = Rdep(dim=H, n_local=E, topk=K, profile=profile, capacity=T * K)
        W_cache = quantize_weights(W1.detach(), W3.detach(), W2.detach(), profile=profile)
        out_bs = rdep_bs.moe_blockscaled(x.detach(), eid, gates, W1.detach(), W3.detach(), W2.detach(), W_cache)
        torch.testing.assert_close(out_bs, ref, atol=atol, rtol=0.0)


if __name__ == "__main__":
    main()
