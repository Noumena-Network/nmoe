# lbl_audit.md

Notes captured during the line-by-line “elegant minimalism” audit (B200-only, RDEP, fail-fast/loud, one clear path).

## Batch 1: core trainer + infra (high-signal issues)

- `nmoe/train.py:34` Import-time side effect (`PYTORCH_CUDA_ALLOC_CONF`); violates “explicit over magical” unless it’s an intentional global contract.
- `nmoe/train.py:40` Config validation is duplicated/ad-hoc (also in `nmoe/model.py`); minimalism wants one `validate_config(cfg, rank, world)` with a single precedence story.
- `nmoe/train.py:115-117` Token accounting inconsistent (per-rank `inputs.numel()` vs global `cfg.batch_size * cfg.seq_len`); either log both explicitly or make them consistent.
- `nmoe/train.py:191-214` Env/CLI override parsing duplicated and inconsistent; one parser, one precedence statement.
- `nmoe/config.py:29-45` Fields labeled “REQUIRED” default to `None` and are typed non-Optional; not fail-fast and not type-honest.
- `nmoe/config.py:95-191` Config is a grab bag (train/data/eval/profiling/RL/SFT) + untyped backend dict knobs; violates “small surface” unless split into strict sub-blocks with unknown-key hard errors.
- `nmoe/runtime.py:11-20` SM100 capability check is good; ensure the docstring matches the real contract (“SM100-only” vs “B200-only”).
- `nmoe/model.py:26-30` `get_attention()` `KeyError` is not user-grade; raise a `ValueError` listing allowed backends.
- `nmoe/model.py:235-243` `all_reduce(loads)` every forward + recomputing MoE layer list per call is hot-path overhead; gate (rank0/every N) and precompute in `__init__`.
- `nmoe/rdep.py:35-40` `assert` for user-facing contracts is not fail-loud in optimized runs; raise explicit errors.
- `nmoe/rdep.py:73-112` Bring-up printing + global `_CPU_PG` caching is hidden global state.
- `nmoe/checkpoint.py:563-567` `_split_param_names()` “fallback: everything is dense” is a direct “no silent downshifts” violation; should hard error if `param_sets()` missing when MoE/world>1.
- `nmoe/checkpoint.py:251-252,788-789` Bare `except` swallow actionable failures.
- `nmoe/opt.py:152-183` Name-pattern “no decay” rules are magical/fragile; prefer type-based rules with an explicit exception list. Also silently degrades when `param_sets()` absent.
- `nmoe/metrics.py:128-159` Multi-GPU detection + shelling out to `lspci` violates “B200-only” + adds nondeterministic external deps.
- `nmoe/log.py:56-65` Import-time logging config + ANSI coloring is a side effect; policy should live in the entrypoint.
- `nmoe/experiments.py:122` `config_json = json.dumps(vars(cfg))` is not provenance-grade; prefer stable TOML snapshot or explicit canonical serialization.

## Batch 2: attention/data/eval/NVIZ (high-signal issues)

- `nmoe/data/mixture.py:125-128` `_fixed_point_weights()` uses `max(1, ...)`, forcing nonzero sampling weight even for zero-weight sources (correctness + minimalism red flag).
- `nmoe/data/loader.py:173-197` Prefetcher thread swallows exceptions; violates “fail loud”.
- `nmoe/data/loader.py:433-445` Comment says `.next()` returns `batch_size` sequences, but code shards across `dp_world_size` so per-rank microbatch differs; contract mismatch.
- `nmoe/attention/rope.py` + `nmoe/model.py` RoPE cos/sin are built on CPU then copied to GPU every forward (`.to(tokens.device)`); hot-path host→device traffic.
- `nmoe/attention/dsa.py` Builds dense `[B,T,T]` masks despite claiming `O(T·k)`; explodes at long seq.
- `nmoe/attention/nsa.py` / `nmoe/attention/dsa.py` `torch.compile(...)` wrapped in `try/except` with silent eager fallback; conflicts with “no fallbacks”.
- `nmoe/attention/mla.py:47-52` Hard dependency on `flash_attn` (via `third_party/flash_attn` in Dockerfile); acceptable only if treated as strict, version-pinned runtime dependency everywhere.
- `nmoe/attention/swa.py:2-4` Unused import (`torch.nn.functional as F`).
- `nmoe/attention/swa.py:30-33` `attn_swa` is an untyped dict knob; expands public surface outside TOML schema enforcement.
- `nmoe/attention/swa.py:66` Runtime `assert` in forward path; should be explicit errors for production.
- `nmoe/attention/swa.py:69-71` `start_q` tensor created every forward; should be a buffer or a scalar path if supported.
- `nmoe/quant.py:25-28,55-58` Uses `assert` for runtime contracts; should raise explicit errors.
- `nviz/components/data-table.tsx` appears unused (no imports found); dead code violates “do one thing exceedingly well”.
- `nviz/components/gpu-monitor.tsx` hardcodes driver/CUDA versions in the UI header; should be derived or omitted.
- `nviz/components/router-health-heatmap.tsx` hardcodes total experts; should come from run metadata.
- `nviz/components/runs-sidebar.tsx` does per-run `api.summary()` calls in a loop (N+1 fetch pattern) and has an unused import.
- `nviz/hooks/use-visibility-poll.ts:9-17` `catch {}` hides repeated failures; if polling is critical, failures should be observable.

## Batch 3: blockscaled + Triton (in progress; high-signal issues)

- `nmoe/blockscaled/grouped.py:102-143,1814-1815,1828-1829` Global caches (`_STRIDED_WORKSPACE_CACHE`, `_STRIDED_COMPILE_CACHE`, `_EXPERT_SCRATCH`) are unbounded; needs an explicit sizing policy.
- `nmoe/blockscaled/grouped.py:82-86,1982` `_num_sms()` uses “current device” even when compiling for `device_index`; can compile with the wrong SM count if another device is active.
- `nmoe/blockscaled/grouped.py` Wrapper is mixed with a large vendored kernel body (license block at `:225-251`); minimalism would isolate vendored code from the public surface.
- `nmoe/triton/swa.py:137-151` Forward `repeat_interleave` of `k/v` and per-forward padding allocates/copies in the hot path.
- `nmoe/triton/dsa.py:87-128` Materializes full `[B,M,N]` FP32 scores before `topk`; O(BMN) memory/bandwidth is likely unacceptable at long seq.
- `nmoe/triton/dsa.py:291-337` Packed top-k assumes indices fit in 16 bits; no guard that `N_PAD <= 65535` → potential silent index corruption.
- `nmoe/triton/nsa.py:343-344,187-191` `BLOCK_S = min(Ns, 256)` truncates selection blocks with no error; `Ns>256` silently produces zero tail blocks (correctness landmine).
- `nmoe/triton/nsa.py:349-354` Per-forward padding of Q/K/V allocates/copies; hot-path overhead.
- `nmoe/triton/kda.py:26-38` Import-time global `triton.set_allocator` with CPU fallbacks violates B200-only + “no fallbacks” and adds hidden side effects.
- `nmoe/triton/kda.py` Contains duplicated helpers/sections (e.g., duplicate `is_tma_supported()`), and appears to re-declare helpers near EOF; indicates vendored drift and non-minimal structure.
- `nmoe/triton/kda.py:5737-5781` Duplicated `input_guard` uses a generator for `contiguous_args` (not a tuple); likely correctness bug.
- `nmoe/triton/kda.py:2452-2467` Defaults to `device=torch.device('cpu')` and uses Python loops (`tolist()`, `range`) to build tensors; slow + off-scope for B200-only hot paths.
- `nmoe/csrc/rdep_nvshmem.cu:20-22` Includes `rdep/configs.cuh`, `rdep/utils.cuh`, `rdep/ibgda_device.cuh`, but those headers are not present under `nmoe/csrc/`; build depends on an external include path (violates “minimal deps” unless made explicit).
- `nmoe/csrc/Makefile:62-74` Build hard-requires `third_party/flashmla` and errors if missing; this should be an explicit, version-pinned dependency contract or removed.
- `nmoe/csrc/gpu.cpp:42-161` Code has “NVML optional” compilation, but `nmoe/csrc/Makefile:155-157` always links `-lnvidia-ml`; current behavior is inconsistent and may fail at link-time rather than fail-loud in Python.
- `nmoe/csrc/gemm.cu:271-336` Per-expert loop launches one cuBLASLt matmul per expert (wgrad path); likely too much launch overhead for large E.
- `nmoe/csrc/muon.cu` Errors print+return (void) without propagating to Python; violates “fail loud”.

## Infra/config (contract mismatches)

- `k8s/dist_train.yaml` sets `WORLD_SIZE=2`, while `configs/dsv3.toml` describes 4 nodes × 8 GPUs; mismatch breaks “obvious contracts”.
- `docker/Dockerfile.train` sets `PYTHONPATH` to include `third_party/flash_attn`, reinforcing that MLA currently isn’t “minimal deps”.
- `nmoe/data/prep.py` has multiple execution stacks (local + multiprocessing + Apache Beam) and non-streaming RAM spikes (e.g., materializing `list(source)` in Beam path).
- `nmoe/data/sinks.py` writes `.bin.tmp` then reads the whole shard back to build `.npy` and recomputes checksums; large avoidable memory+IO.

## Repo hygiene (working tree)

- Some tracked files appear deleted/modified in your working tree (e.g., `nmoe/blockscaled/dense.py`, `nmoe/blockscaled/ggemm.py`); if unintentional, restore before interpreting audit deltas.
