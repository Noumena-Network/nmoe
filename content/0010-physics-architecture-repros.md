---
title: "Reproducing Canon, mHC, and Engram"
subtitle: "A research narrative: wrong starts, PhysicsLM4 alignment, and one real polysemy failure"
date: 2026-03-14
author: "xjdr"
number: "0010"
series: "research"
status: "result"
draft: false
receipts:
  - "nmoe/repro/0010.receipts.json"
---

> Post 0010: we tried to measure architecture primitives, built the wrong evaluation setup first, then rebuilt it until the results were actually legible.

We wanted to answer a simple question: **which 2025-era architecture primitives actually help, and on what?**

Canon adds local convolutions at specific positions in the transformer block. mHC constrains residual flow with manifold-aware hyper-connections. Engram adds hashed bigram memory with learned gating.

What we did not know at the start was whether our evaluation setup was measuring the thing each primitive claimed to help. That ended up being the whole story.

## The first evaluation setup looked fine, and that was the problem

Our first pass sounded rigorous enough on paper. We had a `depo` task for definition-to-property QA, a `lano` task for symbolic grammar, a `mano` task for arithmetic, and the usual loss, token-accuracy, and exact-match metrics.

We ran the matrix. The results were muddy. Canon helped a little. mHC helped a little. Engram was mostly neutral. Nothing really separated.

That mismatch forced a fork. Either the papers were wrong, or we were asking the wrong questions of the model.

## What the feedback changed

A colleague reviewed the setup against the PhysicsLM4 reference implementation and found four places where we had made the problem easier or blurrier than the paper intended.

| Mismatch | Why it mattered |
|:--|:--|
| `depo` was too easy | We were using single-token “words.” The reference uses multi-token words, a mini-vocab, an end-of-word marker, and multiple QA pairs per sample. Our version let the model cheat with shallow pattern matching. |
| We were not measuring Canon on its own terms | Canon is about structured reasoning: taking token-level substructure and turning it into something usable. On this suite, DP-KL on `lano_cfg` is a more faithful readout than plain loss or token accuracy. |
| We were not using Canon the way the paper defines it | The paper has four placements (`A`, `B`, `C`, `D`). We were only using one, and Canon-`D` assumes SwiGLU semantics that we did not have. |
| We were averaging Engram over the wrong tasks | Engram is a memory primitive. Folding it into structured-reasoning tasks where it should be orthogonal just adds noise. |

This was the humbling part. The setup looked disciplined while still being badly matched to the claims we wanted to test.

## Fixing the substrate

We rebuilt the substrate so it tracked PhysicsLM4 much more closely.

| Change | What it bought us |
|:--|:--|
| `depo_v2` | multi-token words, multiple QA pairs per sample, answer-only loss masking |
| `lano_cfg` + DP-KL | a context-free grammar with DP-computable ground-truth next-token distributions, reported with the metric Canon actually needs |
| SwiGLU MLP | the semantics Canon-`D` expects |
| Canon `ABCD` | the full placement set from the paper rather than one borrowed hook |

This took longer than the first pass. After that, the results were finally legible.

## What the structured suite actually showed

With the fixed substrate, we ran 6 variants across 3 seeds:

| Variant | Loss | Token Acc | DP-KL |
|:--|--:|--:|--:|
| baseline | 0.789±0.043 | 0.625±0.002 | 0.196±0.021 |
| engram | 0.792±0.060 | 0.626±0.012 | 0.197±0.043 |
| mhc | 0.762±0.045 | 0.640±0.002 | 0.163±0.033 |
| canon (ABCD) | 0.518±0.031 | 0.737±0.005 | 0.040±0.007 |
| mhc + canon | 0.521±0.034 | 0.737±0.005 | 0.045±0.013 |
| mhc + canon + engram | 0.517±0.030 | 0.738±0.004 | 0.042±0.008 |

<figure>
  <img src="/figures/physicslm4_dpkl.svg" alt="DP-KL by variant" />
  <figcaption>DP-KL on the `lano_cfg` slice of the PhysicsLM4-faithful structured suite. Canon dominates there; mHC helps modestly; Engram is orthogonal.</figcaption>
</figure>

This 3-seed table is aggregated from the seed-0 `physics/physicslm4_reval_s0/*` runs and the seed-1/2 `physics/validation_3seed/*` runs listed in `repro/0010.receipts.json`. Loss and token accuracy come from `analysis/summary.json`; DP-KL comes from `analysis/lano_cfg_dp_valid.json`.

This was the first really clean signal in the whole project. Canon is the dominant lever on this DP-KL slice: DP-KL drops `80%` (`0.196 → 0.040`), and the gain survives the ground-truth distribution metric itself. mHC helps modestly on its own (`0.196 → 0.163`, `-17%`) but is not additive with Canon at this scale. Engram stays orthogonal on this suite (`0.196 → 0.197`), which is exactly what you would expect when the task is not memory-limited.

Our working read is that the old “everything overlaps” result mostly came from not isolating what each primitive was supposed to help.

## Engram only made sense on its own turf

Engram looked useless on the structured suite. That turned out to be the wrong test.

Engram is a memory primitive, so we built a dedicated suite.

| Task / slice | What it asks |
|:--|:--|
| `ngram` | bigram transition table, where memory should help directly |
| `ngram_polysemy` | the same structure, but with two modes (`A` and `B`) that share hash addresses and require different answers |
| `ngram_scrambled` | the same format, but transitions are randomized per sample, so memory should learn to stay out of the way |

Results, seed 0, token accuracy on the answer region:

| Task/Slice | Baseline | Engram | Δ |
|:--|--:|--:|--:|
| ngram/all | 0.159 | 0.281 | +77% |
| ngram_polysemy/mode=A | 0.162 | 0.288 | +78% |
| ngram_polysemy/mode=B | 0.096 | 0.039 | −59% |
| ngram_scrambled/all | 0.002 | 0.002 | 0% |

These exact seed-0 slice values come from `physics/engram_repro_ngram_d24_s0_matrix/analysis/slices_valid.json`. The layerwise collision diagnostic below uses the matching `physics/engram_repro_ngram_d24_s0_matrix_layerce/*` export.

That pattern was far more informative than the earlier average. Engram helps when memory is well-posed, stays neutral when memory is useless, and fails badly on polysemy mode `B`.

That last row was the real surprise. Same task family as mode `A`, but in this seed-0 slice Engram makes it worse.

<figure>
  <img src="/figures/engram_repro_d24/mem_gate_heatmap_width_fixed_residual_vanilla_memory_engram_attn_global.svg" alt="Engram gate heatmap" />
  <figcaption>Engram's gate policy: open early, close late. On scrambled, it closes harder. The conditionality is real, but it still does not explain mode `B`.</figcaption>
</figure>

## The mode=B failure was the most useful negative result

Once mode `B` failed this badly, the next question was obvious: was the model getting the answer right early and then drifting off course, or was memory retrieval wrong from the start?

| Hypothesis | Signature |
|:--|:--|
| late overwrite | the model gets mode `B` right in early layers, then later layers overwrite the correct answer with the mode `A` answer |
| early collision | the hashed memory retrieves the wrong value from the start because mode `A` and mode `B` collide on the same addresses, and the model never recovers |

Standard LogitLens was not enough here. We needed per-layer CE against the **actual correct answer**, not just “does this hidden state look like a plausible token.”

So we built **LogitScope**: project each layer to logits and compute CE against the true label.

<figure>
  <img src="/figures/engram_repro_d24/slices_layer_ce_ngram_polysemy_width_fixed_residual_vanilla_memory_none_attn_global.svg" alt="LogitScope baseline" />
  <figcaption>Baseline: mode `B` is harder than mode `A`, but the layerwise CE still improves in a steady way.</figcaption>
</figure>

<figure>
  <img src="/figures/engram_repro_d24/slices_layer_ce_ngram_polysemy_width_fixed_residual_vanilla_memory_engram_attn_global.svg" alt="LogitScope Engram" />
  <figcaption>Engram: mode `A` improves layer by layer, while mode `B` stays worse than baseline from the start. The signature points to early collision.</figcaption>
</figure>

On this seed-0 diagnostic slice, the evidence strongly favors **early collision**. Engram's mode `B` CE is worse than baseline at essentially every layer. There is no “got it right early, lost it late” signature.

<figure>
  <img src="/figures/engram_repro_d24/slices_layer_ce_ngram_polysemy_width_fixed_residual_vanilla_memory_ple_ngrammer_attn_global.svg" alt="LogitScope PLE" />
  <figcaption>PLE+Ngrammer shows the same pattern. Hash collision without disambiguation seems broader than Engram alone.</figcaption>
</figure>

## Attention span was not the first-order lever here

We expected attention topology to matter more, so we ran a simple 4k sweep at `seq_len = 4096`, `window = 64`, with the same base stack `mHC + Canon-ABCD + Engram` and four global-layer schedules: `100%`, `50%`, `10%`, `0%`.

Result, 3-seed aggregate: the differences are small and mostly within noise. Having *some* global layers helps optimization a bit; beyond that, it is not decisive on this suite.

<figure>
  <img src="/figures/physics_attn_ratio_4k.svg" alt="Attention ratio sweep at seq_len=4096" />
  <figcaption>Attention ratio sweep at long context: the effect exists, but it is not first-order on this suite.</figcaption>
</figure>

This figure is the 3-seed `physics/mixed_ratio_3seed/*` sweep listed in `repro/0010.receipts.json`, with rows corresponding to the `100%` (`attn=global`), `50%` (`attn=mixed:G1L1:64`), `10%` (`attn=mixed:G1L9:64`), and `0%` (`attn=local:64`) schedules.

My conservative read is that the task mix may not isolate the truly global regime cleanly enough, or Canon and mHC are already doing most of the representational work here.

## What I think we learned

On method, the biggest lesson was that matching the measurement contract mattered more than running more experiments. DP-KL is the most informative metric on the `lano_cfg` structured slice so far; loss and token accuracy are easier to satisfy with shallow shortcuts. Slice-level metrics catch failures that averages hide, and per-layer diagnostics are what let us distinguish late overwrite from early collision.

On the primitives, the cleanest summary is this:

| Primitive | What it helps | Evidence | Failure mode |
|:--|:--|:--|:--|
| Canon (ABCD) | Structured reasoning / grammar | −80% DP-KL (0.196 → 0.040) | None observed on this suite |
| mHC | Residual flow / stability | −17% DP-KL alone (0.196 → 0.163) | Subsumed by Canon at this scale |
| Engram | N-gram memory | seed-0 ngram token acc +77% (0.159 → 0.281) | Hash collision / polysemy mode `B` |
| PLE+Ngrammer | N-gram memory (always-on) | Worse than baseline on the seed-0 polysemy slice | Same collision issue |
| attention ratio | attention span / efficiency | small effect at `seq_len=4096` | suite may not isolate the truly global regime |

On Engram specifically, the mode `B` failure looks less like a bug than a limitation of hash-based memory without disambiguation in this setup. The obvious fixes are a larger hash table, context-aware hashing, or a pre-injection gate that refuses memory before it corrupts the answer. The simplest operational lesson may be narrower still: do not use Engram on tasks with unresolved polysemy and expect it to sort itself out.

## Where I would look first

Once the measurement story got cleaner, the obvious next question was implementation. These are the first external references I would audit.

| Primitive | Reference | Why start there |
|:--|:--|:--|
| Canon | [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | Canon-style local mixing |
| mHC | [AndreSlavescu/mHC.cu](https://github.com/AndreSlavescu/mHC.cu) | mHC-style residual constraints |

Both are worth auditing against our B200 target. For now I treat them as references to inspect. I would not drop either one straight into production.

## If you want to rerun the public slice

```bash
# Structured-suite public entry point (single variant; the full cited 6-variant, 3-seed bundle still requires multiple runs plus the export step recorded in the receipt)
python -m nmoe.research.physics.arch_ablations   --output ./out/physics_structured_single   --steps 2000 --seed 0 --init-seed 0   --dim 256 --n-layers 6 --seq-len 2048   --mlp-type swiglu   --lano-cfg-kl   --tasks     "depo_v2:0.4"     "lano_cfg:0.3:depth=6,max_len=1024"     "mano:0.3"   --variant width=fixed,residual=vanilla,precond=canon,canon_set=ABCD,memory=none,attn=global

# Engram memory suite public matrix (baseline + Engram + PLE/Ngrammer)
python -m nmoe.research.physics.arch_ablations   --output ./out/engram_repro   --steps 2000 --seed 0 --init-seed 0   --dim 256 --n-layers 24 --seq-len 256   --slice-metrics --slice-metrics-n 512   --logitlens --logitlens-n 256   --cka --cka-n 256   --layer-ce --layer-ce-n 256   --tasks     "ngram:1.0:n_symbols=512,n_steps=128,table_seed=0"     "ngram_polysemy:1.0:n_symbols=512,n_steps=128,table_seed=0"     "ngram_scrambled:1.0:n_symbols=512,n_steps=128,table_seed=0"   --matrix engram_repro

# Render slice figures
python -m nmoe.research.physics.viz_slices --runs ./out/engram_repro

# Render LogitLens figures
python -m nmoe.research.physics.viz_logitlens --runs ./out/engram_repro

# Render CKA figures
python -m nmoe.research.physics.viz_cka --runs ./out/engram_repro
```

These are the real public entry points today. They regenerate representative local `./out/...` runs plus the Engram diagnostic figures. The full cited `6`-variant, `3`-seed export still lives behind the receipt bundle's provenance map.

## Receipts

The bundle is `repro/0010.receipts.json`.

It carries the multiseed structured-suite exports, the seed-0 Engram matrix and layer-CE diagnostics, the 4k mixed-ratio sweep, figure provenance, and the public entry-point commands for the structured suite, the Engram suite, and the attention sweep.

Read it as provenance and figure mapping for the cited bundle. A one-shot public regeneration script is still follow-up work.

Schema-validation command:

```bash
python scripts/repro/verify_post_receipts.py   --repo-root .   --receipts-dir repro   --post 0010
```
