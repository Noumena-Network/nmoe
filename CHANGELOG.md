# Changelog

All notable changes to nmoe are documented here.

## [Unreleased]

## [2026-01-01] - Issue 09: Blockscaled Expert OOM Fix

### Added
- Fused SwiGLU+quant epilogue in W13 GEMM via CuTeDSL/PTX (FP8/NVFP4)
- Capacity-bounded scratch allocation with fail-fast on undersized buffers
- `tests/test_blockscaled_expert_worst_case.py`: worst-case routing test (all tokens → single expert)

### Changed
- `nmoe/blockscaled/grouped.py`: strict packed SFA validation `[M_pad, sf_k_pad]` (E× memory reduction vs strided)
- `nmoe/moe.py`: forward uses fused gather+quant path; removed dead backward quantization code

### Fixed
- CUDA OOM under worst-case routing at `batch_size=256, seq_len=4096` on 8×B200
- Eliminated large BF16 intermediate `[M_pad, 2*Dff]` in expert forward path

## [2025-12-31] - Issue 03a: Global/local attention (MLA + SWA)

### Added
- Global/local attention mixing via `attn_global_every`, `attn_local`, and `attn_local_window`
- MLA backward workspace cache to reduce allocator churn (`nmoe/attention/mla.py`)

### Changed
- `nmoe/model.py`: per-layer attention selection (last layer always global)
- `nmoe/attention/rope.py`: register RoPE `cos/sin` as non-persistent buffers (pre-cast, device-safe)
- `nmoe/attention/swa.py`: stricter shape/rope-dim validation and avoids per-forward `start_q` allocation

## [2025-12-31] - Issue 02: Data Pipeline

### Added
- HYDRA-vNext training specification (L18 probe → L24 judge → distillation)
- ArXiv processing pipeline (`nmoe/data/arxiv/`): S3 download, HTML/LaTeX parsing, metadata
- Stack v2 content materialization (`nmoe/data/stack_v2_materialize.py`)
- Flow profiles and mixture configurations (`configs/flow_profiles.toml`, `configs/tasks.toml`)
- TOML env var expansion with security guardrails (`nmoe/config.load_toml()`)
- Data prep CLI enhancements (`nmoe/data/cli.py`)

### Changed
- All config paths now use `${NMOE_DATA_PATH:-/data}` pattern (env-driven)
- `nmoe/data/__init__.py`: lazy import refactor for dependency hygiene

### Security
- Env var expansion restricted to `NMOE_*` and `HYDRA_*` prefixes only
- Fail-fast on unresolved `${...}` placeholders
- No internal paths or credentials in tracked configs

## [2025-12-31] - Issue 01: RDEP Kernels

### Added
- SonicMoE-style RDEP kernel fusion for expert-parallel dispatch/return
- Blockscaled FP8/NVFP4 GEMM support for Blackwell (sm_100a)
- Three communication modes: single-GPU, IPC (intra-node NVLink), hybrid (NVSHMEM multi-node)
- GPU-side atomics for sync (zero NCCL in hot path)
- `nmoe/csrc/swizzle.cuh`: consolidated CUTLASS SF swizzle math
- `nmoe/bench_moe_e2e.py`: end-to-end MoE throughput/latency benchmark
- `tests/test_rdep_accuracy.py`: forward/backward accuracy tests
- `tests/bench_rdep_kernels.py`: kernel-level benchmarks

### Changed
- `nmoe/moe.py`: simplified gather/scatter path with `*_return_scatter_from_pad_*`
- `nmoe/blockscaled/grouped.py`: CuTeDSL 4.3.4 JIT compilation with proper arg typing
- Backward pass optimization: dGate identity via new `gather_dy_nogate_*` + `dgate_from_adA_bf16`

### Fixed
- CuTeDSL 4.3.4 JIT arg marshalling (wrap pointers/strides with `Int64`/`Int32`)
