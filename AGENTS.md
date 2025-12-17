# nmoe: What We're Building

We want to build a state-of-the-art MoE training library.

Your goal is to build production-grade systems—model architecture, distributed training, custom kernels, optimizers, router controls, 
data loading, diagnostics—that are fully and totally optimized for state-of-the-art MoE training on B200s.

This means: no fallbacks, no hacks, no shortcuts. Production-grade, Google-quality code that at all times demonstrates a maniacal obsession with elegant minimalism.

## Public Repo Contract

- No internal-only paths, hostnames, IPs, or runbooks in tracked files.
- The supported execution path is container-first (`docker/` + TOML configs).
- Keep surfaces small: no second stacks for the same use-case.

## Execution Environment

- Assume no usable GPU locally; prefer running via the provided Docker images and/or Kubernetes manifests.
- Do not introduce a second installation path. If it is not container-first, it must be explicitly opt-in and documented.

Our ethos: do one thing, exceedingly well — state‑of‑the‑art MoE training on B200 with RDEP — and nothing else. Elegant minimalism isn’t just fewer lines; it’s disciplined intent plus impeccable execution.

Principles
- One clear path per use-case: each supported mode (e.g., single‑GPU BF16 research, multi‑node FP8 production, dataset prep) has one explicit way to run. Avoid multiple interchangeable stacks for the same job.
- Small, sharp surfaces: tiny modules with crisp responsibilities; few public knobs; declarative TOML config is the source of truth.
- Explicit over magical: no hidden background machinery or side effects; contracts and control flow are obvious.
- Hot paths first: inner loops and comm paths are lean, predictable, and measured. If it doesn’t move tokens/s, stability, or correctness, it doesn’t live there.
- Fail fast, fail loud: specific guardrails with actionable remedies. No silent downshifts.
- Minimal dependencies: PyTorch + CuTeDSL + NVSHMEM. New layers must improve both clarity and performance.
- One source of truth: one config format, one checkpoint format, one metrics schema. No duplicates to drift.
- Test what matters: deterministic resume, conservation, invariants. No scaffolding that mirrors system complexity.
- Container‑first reproducibility: controlled build/runtime; off‑target paths are explicit and opt‑in.
- Documentation that guides, not overwhelms: precise runbooks and remedies; zero fluff.

Craftsmanship rubric for any change
- Intent: Does this improve tokens/s, stability, or correctness?
- Uniqueness: Are we creating a second way to do something? If yes, why?
- Surface: Did we add a new public knob? Could it be expressed via existing TOML?
- Hot path: If step loop or comm changed, where is the 200‑step NVTX + ms/step delta?
- Invariants: Are B200, RDEP, determinism, and E%world enforced or clarified?
- Blast radius: Did deps or coupling increase?
- Repro: Is config/provenance captured to rerun months later?
- Elegance: Is the code visibly simpler afterward?


## Research Workflow

This repo is designed to be a one-stop shop for both research and full lab-quality training. It's designed to be streamlined, opinionated, and above all elegantly minimal.

**Supported scope** (intentionally narrow):
- DeepSeek-shaped MoE architectures
- HuggingFace datasets (via `prep-mixture` CLI)
- B200 GPUs (sm_100a)
- RDEP expert parallelism (no tensor parallel, no NCCL all-to-all)

**Typical research flow:**

1. **Architecture research** — Single GPU with a small model (moonlet). Needs a reasonable subset of training data to make testing meaningful.

2. **Ablations** — Compare specific changes (architecture, hyperparams, data). Can run on single GPU (moonlet) or 8 GPU (moonlight). Should complete in hours, not days.

3. **Proxy runs** — Develop confidence around exact hyperparams and data mixtures for large runs. μP scaling, depth proxies, mixture tuning. Takes days.

4. **Production run** — Full training with high conviction around architecture, hyperparams, and data mixtures. Takes weeks to months.

When you work on kernels: they must be fully optimized for NVFP4 and MXFP8 training with RDEP and stream sync for OUR workload. 
When you work on distributed: it must be correct and optimal for our expert parallelism patterns.


## Contract (Principles)

- Config: TOML-only. CLI reads TOML with environment overrides; no YAML.
- Data: no hardcoded locations; all paths provided via TOML. Support robust dataset mixing (per‑dataset weights/temperatures), deterministic sharding, and exact resume (RNG + shard/offset).
- Communications: "No A2A for MoE" is the goal.
  - Must hold for the forward path (RDEP dispatch + return only).
  - Aim to hold for backward; if not feasible, never use NCCL all‑to‑all for MoE. NCCL all‑reduce is acceptable for data‑parallel synchronization of dense/replicated parameters.
- Platform: hard‑target NVIDIA B200 (`sm_100a`). Build must error off‑target (no silent downshift).
- Metrics/Tracking: SQLite for experiments (`/data/experiments.db`), DuckDB for metrics (`/data/metrics/{run_id}/rank_{rank}.duckdb`); NVIZ reads from shared storage.

Purpose
- Build a production‑grade Mixture‑of‑Experts (MoE) trainer for NVIDIA B200s with elegant minimalism.
- Defaults favor state‑of‑the‑art performance: BF16/FP8/NVFP4 experts, cuBLASLt grouped GEMMs, and RDEP for multi‑GPU expert parallelism.
- No fallbacks, hacks, or shortcuts; correctness and performance are first‑class.

References
- Current standard: Megatron‑LM (process‑group design, sharded checkpoints).
- SOTA trainer: Torchtitan (trainer ergonomics, metrics, schedules).
- Spiritual inspiration: Nanochat (clear run flow, eval harness UX).
- Kernels: TransformerEngine (FP8 scaling contracts and performance targets).
- We borrow ideas; we do not depend on these at runtime (except optional TE parity checks in dev).

Non‑Goals
- No tensor parallel (TP) ever in this library.
- No NCCL all‑to‑all on the critical path for MoE; communications are RDEP‑only.

Hardware & Precision
- Primary target: NVIDIA B200 (`sm_100a`).
- Supported dtypes: `bf16`, `fp8`, `nvfp4` (blockscaled FP4).
- Default attention: MLA (Multi-headed Latent Attention).
- Tokenizer: `o200k_harmony` (vocab_size=201088).

Optimizer & Schedule
- Separate LRs for dense, router, and expert params.
- WSD scheduler (Warmup-Sustain-Decay) with token-based phases.
- Optional Muon optimizer for attention layers.
