# nmoe

```
   _ __   _ __ ___   ___   ___
  | '_ \ | '_ ` _ \ / _ \ / _ \
  | | | || | | | | | (_) |  __/
  |_| |_||_| |_| |_|\___/ \___|
```

> No all-to-all. No tensor parallel. B200-only.

This repo is an opinionated Mixture-of-Experts trainer hard-targeted to NVIDIA Blackwell B200 (`sm_100a`).
MoE expert parallelism is implemented via **RDEP**: direct dispatch/return using CUDA IPC (intra-node) and NVSHMEM (inter-node),
instead of NCCL all-to-all collectives on the expert path.

## Quick start

This repository is **container-first**. The supported way to build and run is via the Dockerfiles in `docker/`.

Boot a machine with B200 GPUs and run a minimal single-GPU smoke test (`moonlet`) inside the training image:

```bash
# Build base image (Dockerfile.train expects this tag)
docker build -f docker/Dockerfile.base -t xjdr/nmoe:base .

# Build training image
docker build -f docker/Dockerfile.train -t xjdr/nmoe_train:latest .

# Run single-GPU training (mount /data for datasets, checkpoints, metrics)
docker run --gpus all -v /data:/data xjdr/nmoe_train:latest \
  python -m nmoe.train configs/moonlet.toml
```

## Multi-GPU and multi-node

Single-node (8×GPU) training:

```bash
torchrun --standalone --nproc_per_node=8 -m nmoe.train configs/moonlight.toml
```

Multi-node runs require NVSHMEM. Build the NVSHMEM-enabled image:

```bash
docker build -f docker/Dockerfile.dist -t xjdr/nmoe_dist:latest .
```

Kubernetes manifests in `k8s/` are templates for training, NVIZ, and profiling; edit hostnames, images, and storage before deploying.

## Configs

| Config | Model | Experts | GPUs | Use Case |
|--------|-------|---------|------|----------|
| `moonlet.toml` | 7B | 64 (6 active) | 1 | Single-GPU research |
| `moonlight.toml` | 16B | 64 (6 active) | 8 | Single-node RDEP |
| `dsv2.toml` | DeepSeek-V2 | 160 (6 active) | 8+ | Multi-node |
| `dsv3.toml` | DeepSeek-V3 | 256 (8 active) | 32+ | Production |

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

**RDEP Kernels** — Fused dispatch/return using NVSHMEM (inter-node) and IPC (intra-node).
BF16 and blockscaled (FP8/NVFP4) paths.

**Grouped GEMMs** — cuBLASLt with per-expert scaling. SM100-optimized via CuTe DSL.

**Deterministic Resume** — Checkpoint includes RNG state, shard cursor, config fingerprint.

**HYDRA** — LLM-as-judge data quality pipeline. See `nmoe/data/HYDRA.md`.
This repo includes `nmoe/data/hydra_judge.pt` (a small judge head `state_dict`); see `nmoe/data/HYDRA_JUDGE_HEAD.md`.

## Tests

The project is primarily validated via end-to-end training runs. Some Triton kernels include optional `pytest`-guarded tests
inside the module (e.g. `nmoe/triton/nsa.py`, `nmoe/triton/swa.py`).

## Contributing

nmoe is intentionally narrow and opinionated: B200-only (`sm_100a`), RDEP expert parallelism, TOML configs, and no NCCL all-to-all on the MoE path.
We prefer one clear way to do each supported job over many interchangeable stacks.

## Acknowledgements

This codebase borrows ideas from and interoperates with upstream ecosystems including PyTorch, Triton, NVSHMEM, CUTLASS, and the DeepSeek family of MoE architectures.
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
