# nmoe

```
   _ __   _ __ ___   ___   ___
  | '_ \ | '_ ` _ \ / _ \ / _ \
  | | | || | | | | | (_) |  __/
  |_| |_||_| |_| |_|\___/ \___|
```

> No all-to-all. No tensor parallel. B200-first.

This repo is an opinionated Mixture-of-Experts trainer for NVIDIA Blackwell B200 (`sm_100a`).
MoE expert parallelism is implemented via **RDEP**: direct dispatch/return using CUDA IPC (single-node),
instead of NCCL all-to-all collectives on the expert path.

## Quick start

This repository is **container-first** (via `docker/`), with an explicit opt-in host bootstrap for cloud GPU instances.

### Speedrun (happy path)

On an 8×GPU machine (B200 or H100):

```bash
bash scripts/bootstrap.sh
n speedrun
```

What `n speedrun` does:
- Defaults to `super` (MoE-256).
- Prepares the canonical speedrun dataset under `${DATA_DIR:-/data}/speedrun/{train,val}` (if missing).
- Prepares the Karpathy CORE eval bundle under `${DATA_DIR:-/data}/eval/eval_bundle` (if missing).
- Trains until `target_loss=3.28`, then runs the full CORE suite (22 tasks).
- Appends an entry (including CORE) to `LEADERBOARD.json` (the official, tracked scoreboard).

Dtype defaults:
- **B200 / sm_100a**: `fp8` (default)
- **H100 / sm_90**: `bf16` (bring-up path; blockscaled + MLA are not supported)

View the leaderboard:

```bash
n speedrun --leaderboard
cat LEADERBOARD.json
```

### Docker quick start

Boot a machine with B200 GPUs and run a minimal single-GPU smoke test (`moonlet`) inside the training image:

```bash
# Build the training image (builds base → system → deps → train)
make -C docker train

# Run single-GPU training (mount /data for datasets, checkpoints, metrics)
docker run --gpus all -v /data:/data xjdr/nmoe_train:latest \
  python -m nmoe.train configs/moonlet.toml
```

## Multi-GPU (single-node)

Single-node (8×GPU) training:

```bash
torchrun --standalone --nproc_per_node=8 -m nmoe.train configs/moonlight.toml
```

Kubernetes manifests in `k8s/` are templates for single-node training, NVIZ, and profiling; edit hostnames, images, and storage before deploying.

## Configs

| Config | Model | Experts | GPUs | Use Case |
|--------|-------|---------|------|----------|
| `configs/speedrun/` | Speedrun suite | dense/MoE | 8 | Speedruns + leaderboard |
| `moonlet.toml` | Moonlet | 64 (6 active) | 1 | Single-GPU research |
| `moonlight.toml` | Moonlight | 64 (6 active) | 8 | Single-node RDEP |

## Why RDEP

Traditional MoE uses NCCL all-to-all: every GPU waits for every other GPU.
RDEP replaces this with direct dispatch/return: each GPU sends tokens directly
to the expert owner and receives outputs back. No all-to-all on the MoE path.

```
Source rank                       Owner rank
───────────                       ──────────
tokens ──▶ dispatch ─────────────▶ owner buffer
              │                         │
              │   direct write          ▼
              │                    expert GEMM
              │                         │
output ◀── scatter ◀───────────── return
```

## Data

Training consumes pre-tokenized `.npy` shards.

**Preprocess from HuggingFace:**

```bash
python -m nmoe.data.cli prep \
    --source hf \
    --dataset HuggingFaceFW/fineweb-edu \
    --output /data/fineweb_edu \
    --name fineweb_edu
```

Two workflows:
- **Direct shards** (fast): set `data_path` in config
- **Mixtures**: set `flow_mode`, `mixture_toml`, `flow_profiles_toml` and prep sources via `prep-mixture`

See `nmoe/data/README.md` for the data contract and golden-path commands.

## Research Reproducibility

Research journals are mirrored in this repo:
- posts: `content/`
- figures: `static/figures/`

When a post includes runnable commands, the command surface should use `python -m nmoe...` with configs from `configs/` so results are reproducible from this repo boundary.

## Metrics & NVIZ

Training writes:
- Experiments → SQLite (`/data/experiments.db`)
- Metrics → Parquet per step (`/data/metrics/{run_id}/step_*.parquet`)

NVIZ is the included dashboard. See `nviz/README.md`.

## Architecture

```
nmoe/
├── train.py          # Training loop
├── model.py          # Transformer + MoE
├── moe.py            # Fused MoE autograd
├── rdep.py           # RDEP orchestration
├── checkpoint.py     # Split checkpoints
├── config.py         # TOML config
├── metrics.py        # DuckDB writer
├── csrc/             # CUDA kernels
├── data/             # Data pipeline (HF → shards) + loader
├── attention/        # MLA, DSA, SWA, NSA, KDA
└── eval/             # Evaluation hooks
```

## Attention

Default is MLA (Multi-head Latent Attention) for all layers. Global/local mixing uses MLA for global layers and SWA (Sliding Window) for local layers:

```toml
attn = "mla"              # Global attention type
attn_local = "swa"        # Local attention type
attn_global_every = 6     # Pattern: 5 local + 1 global, repeating
attn_local_window = 128   # SWA window size
```

With `attn_global_every = 6`, layers 0-4 use SWA, layer 5 uses MLA, layers 6-10 use SWA, layer 11 uses MLA, etc. The final layer is always global.

## What's Inside

**RDEP Kernels** — Fused dispatch/return using CUDA IPC (single-node).
BF16 and blockscaled (FP8/NVFP4) paths (B200-first).

**Grouped GEMMs** — cuBLASLt with per-expert scaling. SM100-optimized via CuTe DSL.

**Deterministic Resume** — Checkpoint includes RNG state, shard cursor, config fingerprint.

## Tests

The project is primarily validated via end-to-end training runs. Some Triton kernels include optional `pytest`-guarded tests
inside the module (e.g. `nmoe/triton/nsa.py`, `nmoe/triton/swa.py`).

## Contributing

nmoe is intentionally narrow and opinionated: B200-first (`sm_100a`), RDEP expert parallelism, TOML configs, and no NCCL all-to-all on the MoE path.
We prefer one clear way to do each supported job over many interchangeable stacks.

## Acknowledgements

This codebase borrows ideas from and interoperates with upstream ecosystems including PyTorch, Triton, CUTLASS, and the DeepSeek family of MoE architectures.
See `THIRD_PARTY_NOTICES.md` for license attributions.

## Cite

```bibtex
@misc{nmoe,
  title = {nmoe: B200-targeted MoE training with RDEP},
  year = {2025},
  publisher = {GitHub}
}
```

## Non-Goals

- Tensor parallel (ever)
- NCCL all-to-all for MoE (ever)
- Multi-node training (this release)
- A100 support
- Fallback paths (silent downshifts)

Primary target: B200. Limited H100 support exists for BF16 speedruns only.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `sm_100a` errors | You are running a B200-only path. For H100, use BF16 + SDPA (e.g. `n speedrun`). |
| Multi-node launch attempted | Multi-node is out-of-scope for this release. Use single-node IPC (`torchrun --standalone --nproc_per_node=<N>`). |
| OOM | Reduce `batch_size` or `seq_len` |

## License

Apache-2.0. See `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`.
