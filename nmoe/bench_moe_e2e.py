#!/usr/bin/env python3
"""End-to-end MoE benchmark: throughput, latency, and accuracy."""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from nmoe.rdep import Rdep
from nmoe.moe import _MoEBlockscaledFused
from nmoe.blockscaled.grouped import quantize_weights


def _ms(events: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> list[float]:
    return [s.elapsed_time(e) for s, e in events]


def _p(pct: float, xs: list[float]) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    idx = int(round((pct / 100.0) * (len(xs) - 1)))
    return xs[idx]


def reference_moe_forward(x, eids, gates, W1, W3, W2):
    """Reference MoE forward in BF16 for accuracy comparison."""
    T, H = x.shape
    K = eids.shape[1]
    E = W1.shape[0]
    Dff = W1.shape[2]

    # Gather and compute per-expert
    out = torch.zeros_like(x)
    for e in range(E):
        mask = (eids == e).any(dim=1)
        if not mask.any():
            continue
        xe = x[mask]  # [Me, H]
        # Get gates for this expert
        gate_mask = (eids == e)
        gate_vals = (gates * gate_mask.float()).sum(dim=1)[mask]  # [Me]

        # Expert computation: SwiGLU MLP
        h1 = xe @ W1[e]  # [Me, Dff]
        h3 = xe @ W3[e]  # [Me, Dff]
        a = F.silu(h1) * h3  # SwiGLU
        ye = a @ W2[e]  # [Me, H]

        # Weighted output
        out[mask] += ye * gate_vals.unsqueeze(-1)

    return out


def bench_forward(rdep, x, eids, gates, W1, W3, W2, W_cache, warmup=10, iters=100):
    """Benchmark forward pass."""
    # Warmup
    for _ in range(warmup):
        out = _MoEBlockscaledFused.apply(rdep, x, eids, gates, W1, W3, W2, W_cache)
    torch.cuda.synchronize()

    # Timed
    events = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = _MoEBlockscaledFused.apply(rdep, x, eids, gates, W1, W3, W2, W_cache)
        end.record()
        events.append((start, end))

    torch.cuda.synchronize()
    return _ms(events), out


def bench_backward(rdep, x, eids, gates, W1, W3, W2, W_cache, warmup=10, iters=100):
    """Benchmark forward + backward pass."""
    # Warmup
    for _ in range(warmup):
        x_grad = x.clone().requires_grad_(True)
        out = _MoEBlockscaledFused.apply(rdep, x_grad, eids, gates, W1, W3, W2, W_cache)
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()

    # Timed
    events = []
    for _ in range(iters):
        x_grad = x.clone().requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = _MoEBlockscaledFused.apply(rdep, x_grad, eids, gates, W1, W3, W2, W_cache)
        loss = out.sum()
        loss.backward()
        end.record()
        events.append((start, end))

    torch.cuda.synchronize()
    return _ms(events)


def compute_accuracy(out_test, out_ref):
    """Compute accuracy metrics."""
    diff = (out_test.float() - out_ref.float()).abs()
    rel_err = diff / (out_ref.float().abs() + 1e-6)

    return {
        "max_abs_err": diff.max().item(),
        "mean_abs_err": diff.mean().item(),
        "max_rel_err": rel_err.max().item(),
        "mean_rel_err": rel_err.mean().item(),
        "cosine_sim": F.cosine_similarity(
            out_test.float().flatten().unsqueeze(0),
            out_ref.float().flatten().unsqueeze(0)
        ).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["fp8", "nvfp4"], default="nvfp4")
    parser.add_argument("--T", type=int, default=4096, help="Tokens")
    parser.add_argument("--H", type=int, default=2048, help="Hidden dim")
    parser.add_argument("--Dff", type=int, default=1408, help="FFN intermediate dim")
    parser.add_argument("--E", type=int, default=8, help="Experts")
    parser.add_argument("--K", type=int, default=2, help="Top-K")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--no-accuracy", action="store_true", help="Skip accuracy check")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print(f"\n{'='*60}")
    print(f"MoE E2E Benchmark: profile={args.profile}")
    print(f"Config: T={args.T}, H={args.H}, Dff={args.Dff}, E={args.E}, K={args.K}")
    print(f"{'='*60}\n")

    # Initialize with blockscaled buffers for quantized path
    capacity = args.T * args.K * 2  # 2x for headroom
    rdep = Rdep(dim=args.H, n_local=args.E, topk=args.K, profile=args.profile,
                capacity=capacity, alloc_blockscaled_buffers=True)

    # Create inputs
    x = torch.randn((args.T, args.H), device=device, dtype=torch.bfloat16) * 0.1
    eids = torch.randint(0, args.E, (args.T, args.K), device=device, dtype=torch.int32)
    gates = torch.softmax(torch.randn((args.T, args.K), device=device), dim=-1).to(torch.bfloat16)

    # Create weights
    W1 = torch.randn((args.E, args.H, args.Dff), device=device, dtype=torch.bfloat16) * 0.02
    W3 = torch.randn((args.E, args.H, args.Dff), device=device, dtype=torch.bfloat16) * 0.02
    W2 = torch.randn((args.E, args.Dff, args.H), device=device, dtype=torch.bfloat16) * 0.02

    # Create weight cache
    W_cache = quantize_weights(W1, W3, W2, profile=args.profile)

    # Accuracy check
    if not args.no_accuracy:
        print("Computing reference output...")
        with torch.no_grad():
            out_ref = reference_moe_forward(x, eids, gates, W1, W3, W2)

        print("Computing test output...")
        with torch.no_grad():
            out_test = _MoEBlockscaledFused.apply(rdep, x, eids, gates, W1, W3, W2, W_cache)

        acc = compute_accuracy(out_test, out_ref)
        print(f"\n--- Accuracy vs BF16 Reference ---")
        print(f"  Max absolute error:  {acc['max_abs_err']:.6f}")
        print(f"  Mean absolute error: {acc['mean_abs_err']:.6f}")
        print(f"  Max relative error:  {acc['max_rel_err']:.6f}")
        print(f"  Mean relative error: {acc['mean_rel_err']:.6f}")
        print(f"  Cosine similarity:   {acc['cosine_sim']:.6f}")

    # Forward benchmark
    print(f"\n--- Forward Pass Benchmark (iters={args.iters}) ---")
    fwd_ms, _ = bench_forward(rdep, x, eids, gates, W1, W3, W2, W_cache,
                               warmup=args.warmup, iters=args.iters)

    fwd_p50 = _p(50, fwd_ms)
    fwd_p99 = _p(99, fwd_ms)
    fwd_mean = statistics.mean(fwd_ms)

    # Throughput: tokens/second
    tokens_per_iter = args.T
    fwd_throughput = tokens_per_iter / (fwd_mean / 1000)  # tokens/sec

    print(f"  Latency p50:  {fwd_p50:.3f} ms")
    print(f"  Latency p99:  {fwd_p99:.3f} ms")
    print(f"  Latency mean: {fwd_mean:.3f} ms")
    print(f"  Throughput:   {fwd_throughput/1e6:.2f} M tokens/sec")

    # Forward+Backward benchmark
    print(f"\n--- Forward+Backward Benchmark (iters={args.iters}) ---")
    fwdbwd_ms = bench_backward(rdep, x, eids, gates, W1, W3, W2, W_cache,
                                warmup=args.warmup, iters=args.iters)

    fwdbwd_p50 = _p(50, fwdbwd_ms)
    fwdbwd_p99 = _p(99, fwdbwd_ms)
    fwdbwd_mean = statistics.mean(fwdbwd_ms)
    fwdbwd_throughput = tokens_per_iter / (fwdbwd_mean / 1000)

    print(f"  Latency p50:  {fwdbwd_p50:.3f} ms")
    print(f"  Latency p99:  {fwdbwd_p99:.3f} ms")
    print(f"  Latency mean: {fwdbwd_mean:.3f} ms")
    print(f"  Throughput:   {fwdbwd_throughput/1e6:.2f} M tokens/sec")

    # Memory stats
    print(f"\n--- Memory ---")
    print(f"  Peak allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"  Peak reserved:  {torch.cuda.max_memory_reserved()/1e9:.2f} GB")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
