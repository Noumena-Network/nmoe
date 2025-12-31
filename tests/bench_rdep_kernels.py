import argparse
import statistics

import torch

from nmoe.rdep import Rdep
from nmoe.csrc import rdep as _C


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")


def _ms(events: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> list[float]:
    out = []
    for start, end in events:
        out.append(start.elapsed_time(end))
    return out


def _p(pct: float, xs: list[float]) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    idx = int(round((pct / 100.0) * (len(xs) - 1)))
    return xs[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["bf16", "fp8", "nvfp4"], default="nvfp4")
    parser.add_argument("--T", type=int, default=32768)
    parser.add_argument("--H", type=int, default=5120)
    parser.add_argument("--E", type=int, default=8)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    _require_cuda()
    device = torch.device("cuda")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    rdep = Rdep(dim=args.H, n_local=args.E, topk=args.K, profile=args.profile, capacity=args.T * args.K)

    x = torch.randn((args.T, args.H), device=device, dtype=torch.bfloat16)
    eid = torch.randint(0, args.E, (args.T, args.K), device=device, dtype=torch.int32)
    gates = torch.ones((args.T, args.K), device=device, dtype=torch.bfloat16)
    gates_fp32 = gates.float()

    stream = torch.cuda.current_stream(device)
    offs_pad = torch.empty(args.E, device=device, dtype=torch.int32)
    M_host = torch.zeros(1, device="cpu", dtype=torch.int32).pin_memory()

    if args.profile == "bf16":
        dispatch = lambda: _C.dispatch_meta_bf16(
            x.data_ptr(),
            eid.data_ptr(),
            gates_fp32.data_ptr(),
            int(args.T),
            int(args.K),
            128,
            offs_pad.data_ptr(),
            M_host.data_ptr(),
            stream,
        )
    else:
        dispatch = lambda: _C.dispatch_meta_blockscaled(
            x.data_ptr(),
            eid.data_ptr(),
            gates_fp32.data_ptr(),
            int(args.T),
            int(args.K),
            offs_pad.data_ptr(),
            M_host.data_ptr(),
            stream,
        )

    # Warmup
    for _ in range(args.warmup):
        M_recv = int(dispatch())
        if args.profile != "bf16" and M_recv > 0:
            M_pad = int(M_host.item())
            pack_factor = 2 if args.profile == "fp8" else 4
            Hp = args.H // pack_factor
            sf_k = args.H // 32
            sf_k_pad = ((sf_k + 3) // 4) * 4
            Xe_q = torch.empty((M_pad, Hp), device=device, dtype=torch.uint16)
            Xe_sf = torch.empty((M_pad, sf_k_pad), device=device, dtype=torch.uint8)
            _C.gather_xe_blockscaled(Xe_q.data_ptr(), Xe_sf.data_ptr(), int(M_recv), int(M_pad), stream)
    torch.cuda.synchronize()

    # Timed
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        M_recv = int(dispatch())
        if args.profile != "bf16" and M_recv > 0:
            M_pad = int(M_host.item())
            pack_factor = 2 if args.profile == "fp8" else 4
            Hp = args.H // pack_factor
            sf_k = args.H // 32
            sf_k_pad = ((sf_k + 3) // 4) * 4
            Xe_q = torch.empty((M_pad, Hp), device=device, dtype=torch.uint16)
            Xe_sf = torch.empty((M_pad, sf_k_pad), device=device, dtype=torch.uint8)
            _C.gather_xe_blockscaled(Xe_q.data_ptr(), Xe_sf.data_ptr(), int(M_recv), int(M_pad), stream)
        end.record()
        events.append((start, end))

    torch.cuda.synchronize()
    ms = _ms(events)
    print(
        f"profile={args.profile} T={args.T} H={args.H} E={args.E} K={args.K} iters={args.iters} "
        f"p50_ms={_p(50, ms):.4f} p99_ms={_p(99, ms):.4f} mean_ms={statistics.mean(ms):.4f}"
    )


if __name__ == "__main__":
    main()
