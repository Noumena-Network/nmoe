# Changelog

All notable changes to nmoe are documented here.

## [2026-02-07] - v1.0: Public Release

### Added
- **Speedrun leaderboard** (`LEADERBOARD.json`): Canonical benchmarks for super (256 experts) and ultra (4096 experts) configs on B200 and H100
- **Muon optimizer** (`nmoe/opt.py`): Moonlight recipe (SGD-Nesterov + Newton-Schulz orthogonalization + consistent update RMS)
- **CLI** (`n` command): `n speedrun`, `n train`, `n mon`, `n viz`, `n research`
- **Bootstrap script** (`scripts/bootstrap.sh`): One-command setup for cloud GPU instances
- **CORE eval integration**: 22-task eval suite runs automatically after speedrun
- **nmon TUI** (`tools/nmon`): Terminal monitoring for training runs
- **nviz dashboard** (`nviz/`): Web-based metrics visualization

### Changed
- Default attention: MLA (Multi-head Latent Attention) for all layers
- Tokenizer: `o200k_harmony` (vocab_size=201088)
- Expert optimizer: Muon for fp8/nvfp4, AdamW for bf16

### Golden Path
```bash
git clone https://github.com/Noumena-Network/nmoe.git && cd nmoe
bash scripts/bootstrap.sh
n speedrun
```

## [2026-01-02] - Issue 07: Eval Framework

### Added
- Choice-based eval (`nmoe/eval/choices.py`): MMLU, MMLU-pro, ARC, HellaSwag, BoolQ, OpenBookQA, WinoGrande, COPA
- Generation-based eval (`nmoe/eval/run_gen.py`): GSM8K, HumanEval, GPQA
- Code execution sandbox (`nmoe/eval/execution.py`) for HumanEval
- HF dataset adapters (`nmoe/eval/adapters.py`) for all choice tasks
- Training integration via `eval_every` config in `nmoe/train.py`

### Notes
- Choice-based eval uses chunked logsumexp to avoid materializing `(seq, vocab)` logits
- Multi-GPU eval keeps ranks in lockstep by construction (no barriers in inner loops)
- Requires `datasets` library: `uv pip install datasets`

## [2026-01-02] - Issue 08: ZeRO-2 Chunked Reduce-Scatter and Checkpoint Reliability

### Added
- Chunked reduce-scatter in ZeRO-2 (`nmoe/zero2.py`): configurable via `NMOE_ZERO2_RS_CHUNK_MB` (default 2GB)
- Third-party import path setup in `nmoe/runtime.py`

### Changed
- `nmoe/checkpoint.py`: eager CPU copy before background queue; clean thread shutdown
- `nmoe/train.py`: support `--resume` / `--no-resume` boolean flags
- `nmoe/opt.py`: return dense LR for logging (multi-group compatibility)
- `nmoe/quant.py`: direct csrc import to avoid circular dependencies

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
- Flow profiles and mixture configurations (`configs/flow_profiles.toml`, `configs/tasks.toml`)
- TOML env var expansion with security guardrails (`nmoe/config.load_toml()`)
- Data prep (HF → shards) utilities and CLI (`nmoe/data/*`)

### Changed
- All config paths now use `${NMOE_DATA_PATH:-/data}` pattern (env-driven)
- `nmoe/data/__init__.py`: lazy import refactor for dependency hygiene

### Security
- Env var expansion restricted to `NMOE_*` prefix only
- Fail-fast on unresolved `${...}` placeholders
- No internal paths or credentials in tracked configs

## [2025-12-31] - Issue 01: RDEP Kernels

### Added
- SonicMoE-style RDEP kernel fusion for expert-parallel dispatch/return
- Blockscaled FP8/NVFP4 GEMM support for Blackwell (sm_100a)
- Single-node communication modes: single-GPU and IPC (intra-node NVLink)
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
