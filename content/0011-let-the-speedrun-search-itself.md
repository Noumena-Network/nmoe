---
title: "Let the Speedrun Search Itself"
subtitle: "Eval-gated config-only autoresearch on the canonical super fp8 lane"
date: 2026-03-14
author: "xjdr"
number: "0011"
series: "research"
status: "result"
draft: false
receipts:
  - "nmoe/repro/0011.receipts.json"
---

Recently, I kept finding myself circling the same question: what did I actually want from all the new autoresearch repos?

I did not want another training stack, another config system, or another place for experiments to disappear and become folklore. What I wanted was the discipline: a written program, a narrow mutable surface, fixed budgets, explicit keep-or-discard decisions, and receipts for every attempt.

I wanted the agent to roam, but only inside a box that I trusted.

For `nmoe`, that box had to stay small. The whole point of this repo is that there is one real training path, one real metrics surface, and one real eval loop. If “autoresearch” meant bolting on a second stack, the exercise would miss the point.

This post is the first result from doing it the stricter way.

## Evidence scope

This is a result post for one bounded campaign on the public `nmoe` surface.

| Surface | Contract |
|:--|:--|
| mutation tier | config-only |
| lane | canonical super fp8 speedrun |
| data | canonical speedrun train/val |
| primary objective | `final_valid_loss` |
| veto | CORE cannot drop by more than `0.002` |
| cluster shape | `4` GPU workers in parallel |
| receipt boundary | `repro/0011.receipts.json` |

So the claim here is narrow on purpose. This is not code-editing autoresearch, and it is not a general statement that the LLM proposer always beats the deterministic fallback. It is one real campaign, on one real lane, with one machine-readable truth boundary.

## What I wanted the controller to do

I did not want an “AI scientist” demo. I wanted something much more boring, which is why I trust it more.

The controller had to reuse the canonical `nmoe.train` path, touch only allowlisted config fields, spend a fixed budget per candidate, promote only when validation improved *and* CORE stayed inside budget, and write a receipt whether the candidate won, lost, or crashed.

The core contract in `campaigns/speedrun_super_research.toml` is:

```toml
[objective]
primary_metric = "final_valid_loss"
direction = "min"
min_delta_abs = 0.001

[objective.constraints]
required_metrics = ["core"]
max_core_drop = 0.002

[budget.benchmark]
steps = 512

[mutation]
tier = "config_only"
```

In practice that meant one canonical runner (`python -m nmoe.cli.main campaign auto ...`), one canonical data source (`/data/speedrun/train` and `/data/speedrun/val`), a `512`-step benchmark budget per candidate, and a small override surface centered on `aux_loss_alpha`, `lr_dense`, `lr_router`, `warmup_steps`, and a few nearby dials.

The cluster contract mattered just as much as the metric contract. We ran `4` workers in parallel, with one candidate claim per worker and unique checkpoint roots per experiment. If workers share receipts or checkpoint roots, the whole thing stops being research and turns into scheduler noise.

## The first useful negative result showed up immediately

The first wake-up was on `aux_loss_alpha`.

| candidate | final validation loss | CORE | decision |
|:--|--:|--:|:--|
| `seed` (`aux_loss_alpha=0.0001`) | `5.1987` | `-0.0169` | keep |
| `aux_loss_alpha=0.00015` | `5.1950` | `-0.0183` | keep |
| `aux_loss_alpha=0.0005` | `5.1920` | `-0.0208` | discard |

That third row is the whole reason the eval gate exists.

If I had optimized validation loss alone, `0.0005` would have looked like the next champion. CORE fell past the allowed budget, so the controller rejected it. A seductive wrong answer is still a wrong answer.

## The first 4-worker wave found the real dense-LR move

Once the controller had a viable wake-up point, I let four workers go at once.

That is where the campaign stopped feeling like a toy. It was no longer one scalar hill-climb. It had to survive concurrency, stale local baselines, and the fact that more than one candidate can be locally “kept” in the same wave.

| candidate | final validation loss | CORE | tokens/s/GPU | mean CV | decision |
|:--|--:|--:|--:|--:|:--|
| `lr_dense=0.0016` | `5.2729` | `-0.0208` | `94.7k` | `237.5` | discard |
| `lr_dense=0.0020` | `5.1571` | `-0.0153` | `97.8k` | `211.6` | keep |
| `lr_dense=0.0022` | `5.1270` | `-0.0156` | `98.1k` | `199.8` | keep |
| `lr_router=0.0021` | `5.1932` | `-0.0167` | `91.9k` | `242.3` | keep |

<figure>
  <img src="/figures/0011_autoresearch_progression.svg" alt="Autoresearch champion progression showing final validation loss and throughput across the kept global winners." />
  <figcaption>Global champion updates only. The controller walks from the seed to the final `aux_loss_alpha=0.00012, lr_dense=0.0022` winner while improving both validation loss and throughput.</figcaption>
</figure>

`lr_dense=0.0022` was the first move that felt like a regime change rather than noise. Relative to the wake-up baseline (`aux_loss_alpha=0.00015`), it improved final validation loss by `0.0679` nats while also improving CORE. It ran a bit faster and lowered mean router CV too.

There is a subtle systems point hiding in this table. In a parallel wave, more than one candidate can be kept against the same older baseline. That is fine. The global champion still has to be chosen from the best kept receipt instead of whichever worker happened to finish last.

## The refinement wave gave the cleanest answer of the whole run

The next wave refined around the new dense-LR champion.

| candidate | final validation loss | CORE | tokens/s/GPU | mean CV | decision |
|:--|--:|--:|--:|--:|:--|
| `aux_loss_alpha=0.00018`, `lr_dense=0.0022` | `5.1174` | `-0.0193` | `97.8k` | `210.7` | discard |
| `aux_loss_alpha=0.00012`, `lr_dense=0.0022` | `5.1200` | `-0.0136` | `100.7k` | `197.0` | keep |
| `aux_loss_alpha=0.0002`, `lr_dense=0.0022` | `5.1286` | `-0.0200` | `92.8k` | `200.2` | discard |
| `lr_router=0.0021`, `aux_loss_alpha=0.00015`, `lr_dense=0.0022` | `5.1455` | `-0.0180` | `94.5k` | `216.5` | discard |

<figure>
  <img src="/figures/0011_autoresearch_refinement_gate.svg" alt="Refinement wave around lr_dense 0.0022 showing final validation loss and CORE, with the CORE gate highlighted." />
  <figcaption>The best raw validation-loss point in the refinement wave (`aux_loss_alpha=0.00018`) was vetoed by CORE. The kept winner (`aux_loss_alpha=0.00012`) is the one that clears the full contract.</figcaption>
</figure>

This was the cleanest result in the whole campaign. `0.00018` posted the best raw validation loss in the wave and still lost because CORE fell through the floor. `0.00012` became the final champion because it improved validation loss *and* improved CORE. That is exactly the behavior I wanted.

The agent did not just chase the prettiest scalar. It found a tempting loss improvement, got told “no” by eval, and had to keep searching until it found a point that actually cleared the full contract.

## The final champion

| field | value |
|:--|:--|
| `aux_loss_alpha` | `0.00012` |
| `lr_dense` | `0.0022` |
| `final_valid_loss` | `5.1200` |
| `CORE` | `-0.0136` |
| throughput | `100.7k tokens/s/GPU` |
| mean router CV | `197.0` |

Relative to the seed receipt, that is `0.0787` nats better final validation loss (`5.1987 -> 5.1200`), about `1.51%` lower final validation loss, `+0.00323` better CORE (`-0.0169 -> -0.0136`), and about `3.08%` higher throughput (`97.7k -> 100.7k tokens/s/GPU`).

So the final result is more than a smaller loss number. The winner also cleared the eval gate, improved router health, and ran a bit faster.

## What I think we learned

The first lesson is that bounded config-only autoresearch is already useful. We did not need code mutation to find a real win on the canonical speedrun lane.

The second is that validation-first promotion still needs an eval veto. Twice, the best raw validation point in a local neighborhood was not the right answer under CORE.

The third is that dense LR mattered more than router LR on this slice. The first large improvement came from `lr_dense`, not router-bias tuning.

The fourth is that parallel search changes controller semantics. Keep-or-discard can be local to a wave baseline, but champion selection has to stay global and receipt-backed.

This remains a narrow result: one campaign, one public eval suite, one config-only search surface. But it is the first time the public `nmoe` surface has the loop shape I actually wanted — program, search, receipts, eval gate, cluster fanout, and a real kept winner.

## Receipts

The bundle is `repro/0011.receipts.json`.

It carries the exported campaign artifacts, including `blog_artifacts/0011_autoresearch_manifest.json`, `blog_artifacts/0011_autoresearch_progression.json`, `blog_artifacts/0011_autoresearch_receipts.json`, and `blog_artifacts/0011_autoresearch_run_summaries.json`. The key raw campaign receipts are linked under `campaign_runs/speedrun_super_research/benchmark/...`.

Schema-validation command:

```bash
python3 scripts/repro/verify_post_receipts.py \
  --repo-root . \
  --receipts-dir repro \
  --post 0011
```

Path validation still requires a runtime that has the referenced `campaign_runs/` and `blog_artifacts/` surfaces mounted under the chosen data root.
