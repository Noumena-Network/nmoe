---
title: "What We Built"
subtitle: "A production-grade MoE training system, because reproducibility is the experiment"
date: 2026-03-14
author: "xjdr"
number: "0001"
series: "systems"
status: "framing"
---

> Post 0001: what `nmoe` is and why we built it the way we did.

I've been burned by training infrastructure enough times to know one thing: **if you can't trust resume, eval, and logging, you're not doing research. You're telling stories.**

The prototypes that led to `nmoe` taught that lesson repeatedly. In proto work, configs drifted. Someone changed the data pipeline and forgot to update the checkpoint format. We resumed a run and got different results. We compared two experiments and realized they were using different tokenizers.

So when we rebuilt from scratch, we started with a simpler question: what is the smallest system in which we can actually trust the results?

One lesson hit harder than it should have. In an MoE stack, **eval belongs to the distributed system**. If you get it wrong, you do not merely get noisier numbers. You get deadlocks, silent collapse, and fake results that look plausible until they fail to reproduce. That is why “production-grade” matters here. It is a research prerequisite.

All source anchors in this post are relative to the external `nmoe` repo.

## Evidence Scope

This post is about architecture and invariants. The transport and kernel measurements land in `0009`. The measurement receipts and result loops start in `0002` and later. Read the claims here as implementation contracts backed by source anchors in the external `nmoe` tree.

## The Constraints We Chose

### One path per use-case

Every time you add a second way to do the same thing, you create drift. The paths evolve independently. Eventually one breaks and nobody notices because most people are using the other one. Then somebody tries to reproduce an old result and cannot.

So we chose one config format, one checkpoint format, one metrics schema, and one resume path. If you want a different behavior, you change the code. You do not add a mystery flag and hope everyone remembers what it means six months later.

### Fail fast, fail loud

Silent fallbacks are how research turns into folklore. If you are training on the wrong GPU, the run should stop immediately; a hidden `10x` slowdown is how bad results sneak in. If the config does not match the checkpoint, the process should fail before it writes another token.

We chose a fail-loud default path: no silent downshifts, no hidden “works on my machine” branch. Leave the supported envelope and the system should stop with an actionable message. Narrow bring-up relaxations exist, but they are explicit opt-ins.

### Determinism is non-negotiable

Exact resume sounds simple until you write down what it really requires. You need to checkpoint

- model state
- optimizer state
- data position
- every RNG source the training loop actually depends on

In the current public `nmoe` code, the checkpoint path explicitly persists optimizer state, torch RNG, CUDA RNG, loader state, config and plan fingerprints, and any separately stepped Muon state. If we introduce Python- or NumPy-driven randomness into the training loop, those RNG sources need to be checkpointed too.

### The stale-batch failure

We learned this the annoying way. An async loader worker had already fetched one more batch under the pre-resume offsets, so after resume the run looked healthy but the token stream had shifted by one batch.

The symptom was subtle. Loss kept going down. Nothing exploded. But experiments that should have matched no longer matched, because the data stream had quietly diverged.

The public `nmoe` path takes the conservative route on purpose: the exact-resume training loader runs with prefetch disabled unless queue state is checkpointed too. That sounds boring. Boring is the point. Preserve the loader state, preserve the token order, and only then talk about exact resume.

That is why deterministic resume is a validity gate. If you cannot show that a resumed run matches a fresh one, you should not trust any result that depended on stopping and restarting.

## The Contracts We Refuse To Compromise On

| Surface | Contract | Why it matters |
|---------|----------|----------------|
| Config | TOML only. If it matters, it lives in the config. Overrides are explicit and constrained, and there is no hidden background mutation. | One source of truth beats a pile of near-equivalent entry points. |
| Data | Paths come from config. Deterministic loading is a function of the plan, the world, and the checkpointed loader state. The exact-resume path keeps prefetch off unless queue state is checkpointed too. | Resume only means something if token order survives it. |
| Distributed | The public MoE token dispatch/return story is built around `RDEP` rather than `NCCL` all-to-all. In the current public tree, the directly auditable `RDEP` path is single-node `IPC`; the broader multi-node transport story comes in `0009`. Standard data-parallel sync still uses `NCCL`, and this stack does not use tensor parallel. | MoE transport is part of the scientific contract. It determines whether the performance story and the correctness story line up. |
| Platform | The default contract is `B200` (`sm_100a`). Off-target paths should fail loudly unless an explicit bring-up path is enabled. Precision (`bf16`, `fp8`, `nvfp4`) is always explicit. | The system should tell you what world you are in. |

Those contracts are narrow by design. They are there to make the research surface legible.

## What We Ended Up With

<figure>
  <img src="/figures/nmoe_system_overview.svg" alt="High-level nmoe system overview." />
  <figcaption>The hot path stays intentionally small: config -> train -> measure -> resume.</figcaption>
</figure>

At the center, `nmoe` is a transformer/MoE training stack with MoE layers, `MLA` and sliding-window attention, and experts that can run in `bf16`, `fp8`, or `nvfp4`. Expert parallelism runs through `RDEP`, with the public tree directly exposing the single-node `IPC` path today and the broader transport story deferred to `0009`.

Optimization is `AdamW` with separate learning rates for dense, router, and expert parameters. Eligible dense `2D` weights can use `Muon`, and expert matrices can opt into an explicit `ExpertMuon` override when requested. The schedule is `WSD` with token-based phases.

Data paths are config-driven. The public tree includes both HuggingFace-backed dataset surfaces and pre-tokenized shard paths. Metrics land in per-step parquet under `<metrics_dir>/{run}/step_XXXXXXXX.parquet`, written from a DuckDB-backed in-memory buffer. Router health, throughput, loss, and gradients all live there. Checkpointing preserves the public optimizer state with config fingerprinting, and resume validates the config before it loads weights.

## What We Left Out

We left out tensor parallel because, for this lab, it solved a different problem than the one we were trying to solve. We left out multiple config formats because `TOML` already gives us one clean source of truth, and `YAML` plus `JSON` would only create more places for drift. We left out plugin-style framework mode because it makes for a cute demo and a terrible lab notebook.

The theme is the same in each case: a broader surface would have made the system feel more flexible while making the science less trustworthy.

## What's Next

With the infrastructure in place, we can start doing science. Post `0002` covers the eval loop: `bpb`, `CORE`, and the router-health signals that tell you when an MoE is quietly collapsing.
