# NMoE

```
   _ __   _ __ ___   ___   ___
  | '_ \ | '_ ` _ \ / _ \ / _ \
  | | | || | | | | | (_) |  __/
  |_| |_||_| |_| |_|\___/ \___|
```

> No all-to-all. No tensor parallel. B200-only.

MoE training on NVIDIA B200 using RDEP—direct GPU-to-GPU NVSHMEM puts instead of
NCCL collectives. One fused kernel per direction. Zero collective synchronization
on the expert path.

## Prerequisites

- NVIDIA B200 GPU(s) (`sm_100a`)
- CUDA 12.8+ / PyTorch nightly (cu128)
- NVSHMEM 3.5+ (for multi-node RDEP)

## Quick Start (Docker)

This repository is **container-first**. Build and run via the Dockerfiles in `docker/`.

```bash
# Build base image
docker build -f docker/Dockerfile.base -t xjdr/nmoe:base .

# Build training image
docker build -f docker/Dockerfile.train -t xjdr/nmoe_train:latest .

# Run single-GPU training
docker run --gpus all -v /data:/data xjdr/nmoe_train:latest \
    python -m nmoe.train configs/moonlet.toml
```

For multi-node with NVSHMEM:

```bash
docker build -f docker/Dockerfile.dist -t xjdr/nmoe_dist:latest .
```

## Configs

| Config | Model | Experts | GPUs | Use Case |
|--------|-------|---------|------|----------|
| `moonlet.toml` | 7B | 64 (6 active) | 1 | Single-GPU research |
| `moonlight.toml` | 16B | 64 (6 active) | 8 | Single-node RDEP |
| `dsv2.toml` | DeepSeek-V2 | 160 (6 active) | 8+ | Multi-node |
| `dsv3.toml` | DeepSeek-V3 | 256 (8 active) | 32+ | Production |

## Training

```bash
# Single GPU
python -m nmoe.train configs/moonlet.toml

# Multi-GPU (single node)
torchrun --standalone --nproc_per_node=8 -m nmoe.train configs/moonlight.toml

# Multi-node
torchrun --nnodes=N --nproc_per_node=8 --node_rank=R \
    --master_addr=ADDR --master_port=PORT \
    -m nmoe.train configs/dsv2.toml

# Override config values
python -m nmoe.train configs/moonlet.toml --steps=500 --dtype=bf16
```

## Why RDEP

Traditional MoE uses NCCL all-to-all: every GPU waits for every other GPU.
RDEP replaces this with direct NVSHMEM puts—each GPU writes tokens directly
into the expert owner's buffer. No collective. No barrier. No waiting.

```
Source rank                       Owner rank
───────────                       ──────────
tokens ──▶ dispatch ─────────────▶ symmetric buffer
              │                         │
              │   nvshmem_putmem        │
              │   + atomic slot         ▼
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
- **Direct shards** (research): set `data_path` in config
- **Flows** (production): set `flow_mode`, `mixture_toml`, `flow_profiles_toml`

See `nmoe/data/README.md` for the full data pipeline.

## Metrics & NVIZ

Training writes:
- Experiments → SQLite (`/data/experiments.db`)
- Metrics → DuckDB (`/data/metrics/{run_id}/rank_{rank}.duckdb`)

NVIZ is the included dashboard. See `nviz/README.md`.

## Kubernetes

Example manifests in `k8s/`:

```bash
kubectl apply -f k8s/train.yaml      # Training job
kubectl apply -f k8s/nviz.yaml       # Metrics dashboard
kubectl apply -f k8s/lab.yaml        # Jupyter environment
```

Edit hostnames, images, and storage before deploying.

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
├── data/             # Data pipeline, HYDRA
├── attention/        # MLA, DSA, SWA
└── eval/             # Evaluation hooks
```

## What's Inside

**RDEP Kernels** — Fused dispatch/return using NVSHMEM (inter-node) and IPC (intra-node).
BF16 and blockscaled (FP8/NVFP4) paths.

**Grouped GEMMs** — cuBLASLt with per-expert scaling. SM100-optimized via CuTe DSL.

**Deterministic Resume** — Checkpoint includes RNG state, shard cursor, config fingerprint.

**HYDRA** — LLM-as-judge data quality pipeline. See `nmoe/data/HYDRA.md`.
This repo includes `nmoe/data/hydra_judge.pt` (a small judge head `state_dict`); see `nmoe/data/HYDRA_JUDGE_HEAD.md`.

## Non-Goals

- Tensor parallel (ever)
- NCCL all-to-all for MoE (ever)
- H100/A100 support
- Fallback paths

One hardware target. One distribution strategy. B200 or bust.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `sm_100a` errors | You need B200. No workarounds. |
| NVSHMEM init fails | Use IPC mode for single-node, or check bootstrap config |
| OOM | Reduce `batch_size` or `seq_len` |

## License

Apache-2.0. See `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`.
