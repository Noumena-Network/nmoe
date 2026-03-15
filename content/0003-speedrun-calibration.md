---
title: "The Speedrun Loop"
subtitle: "A small-model speedrun is our fastest honest instrument for architecture research"
date: 2026-03-14
author: "xjdr"
number: "0003"
series: "methodology"
status: "result"
draft: false
receipts:
  - "nmoe/repro/0003.receipts.json"
---

> Post 0003: a speedrun is our smallest honest architecture lab—anchor it to the outside world, score it on more than loss, then use it to compare dense and expert-sparse designs quickly.

Frontier architecture research has a tempo problem. Full runs are too expensive to think with, but toy runs are too fake to trust. The best compromise we have found is the **speedrun**: a small model, a hard training contract, and a scoreboard that turns ideas over in hours instead of weeks.

We did not invent this loop. Keller Jordan's `modded-nanogpt` speedruns showed how much research velocity you can get from a small dense model under a serious contract. We wanted the same thing inside `nmoe` so we could study the question we actually care about: **when does expert sparsity buy something real, and under what contract?**

This post introduces that instrument. The point is not that the current loop is perfect. The point is that it is already useful enough to do honest research.

## The Instrument

Let a speedrun report for model variant `m` be

$$
R(m; C_{\mathrm{train}}, C_{\mathrm{eval}}) = \big(L_T(m),\; Q_T(m),\; S_T(m)\big),
$$

where `L_T(m)` is the validation loss at horizon `T`, `Q_T(m)` is the end-of-run capability score, and `S_T(m)` is the supporting diagnostic surface: throughput, router telemetry, and any other signal that tells us whether an apparent win is real or pathological.

For that instrument to mean anything, three contracts have to be explicit. The anchor contract keeps us tied to a public reference point. The evaluation contract keeps us from optimizing a single smooth scalar and calling it science. The swap contract says what stays fixed when we compare dense and sparse models.

## 1. Anchor It To The Outside World

Our public anchor is Keller Jordan's `modded-nanogpt` June 2024 `AdamW` dense speedrun. It is a good anchor for exactly the reason we wanted one: published logs, a clear recipe neighborhood, and a concrete target.

### The public anchor

The public anchor comes in two pieces. The June 2024 code state tells us what recipe neighborhood we borrowed from. The preserved public run record gives us the exact number we anchor against.

| Surface | Provenance | What it gives us |
|---------|------------|------------------|
| June 2024 code state | `modded-nanogpt` commit `b6b0a0d36e6f1758a8d14d5fcd5f15ca5d19b891` | the small dense recipe neighborhood we adapted into `nmoe` |
| Preserved run record | `records/track_1_short/2024-06-06_AdamW/f66d43d7-e449-4029-8adf-e8537bab49ea.log` | the exact public anchor value: `tel = 3.275959` at step `9536` |
| Public recipe note | `records/track_1_short/2024-06-06_AdamW/README.md` | a remembered recipe note: `lr=0.0018`, `warmup=250`, `warmdown=2000`, `betas=(0.9, 0.95)` |

That distinction matters. The log file is the numerical anchor. The code state is the recipe neighborhood. Our `nmoe` dense calibration lane keeps the same headline shape — `9536` steps, `256 x 2048` tokens/step, `AdamW` at `1.8e-3` / `0.1` / `(0.9, 0.95)` — while using the cleaner `256/2048` WSD decomposition that the rest of this series inherits.

Reference surfaces: [modded-nanogpt June 2024 record](https://github.com/KellerJordan/modded-nanogpt/tree/main/records/track_1_short/2024-06-06_AdamW) · [modded-nanogpt commit `b6b0a0d`](https://github.com/KellerJordan/modded-nanogpt/tree/b6b0a0d36e6f1758a8d14d5fcd5f15ca5d19b891)

### What we matched, and what we did not

| Aspect | modded-nanogpt (June 2024) | nmoe | Match? |
|--------|---------------------------|------|--------|
| Tokenizer | GPT-2 (50257) | GPT-2 (50257 padded to 50304) | ~Yes |
| Embeddings | Tied | Untied | No |
| FFN activation | GELU | SwiGLU | No |
| Attention | SDPA (`scaled_dot_product_attention`) | SDPA (`scaled_dot_product_attention`) | Yes |
| Embedding/logit gains | None | None in these `bf16` calibration runs | Yes |
| Norm eps | `1e-5` | `1e-5` | Yes |
| Data | FineWeb10B (GPT-2 tokenized) | FineWeb10B (GPT-2 tokenized) | ~Yes |
| Optimizer | AdamW | AdamW | Yes |
| μP / init scaling | None | Present in `nmoe` | No |
| Schedule | WSD; preserved public README remembers `250/2000` from memory | WSD with `256/2048` decomposition | Close |

Those mismatches matter. Exact reproduction would require matching the architecture too. Because we intentionally do not do that, the right question becomes: **did we get close enough to know whether our stack lives in the same universe?**

We measure that with the anchor gap

$$
\Delta_{\mathrm{anchor}} = L_T\big(m_{\mathrm{nmoe\ dense}}\big) - L_T\big(m_{\mathrm{public\ anchor}}\big).
$$

At step `9536`, the current dense calibration lands at:

```text
Public anchor log @ step 9536: tel = 3.275959
Dense SDPA @ step 9536:     valid/loss = 3.480672
Gap:                        +0.204713 (~ +0.205)
```

<figure>
  <img src="/figures/dense_calibration_curve.svg" alt="Dense calibration curve comparing nmoe and modded-nanogpt." />
  <figcaption>The gap is real: at step `9536` we are about `+0.205` nats behind the pinned public June 2024 log anchor. That is enough to keep us honest, even though it is not parity.</figcaption>
</figure>

That result is not glamorous. It is exactly the kind of useful annoyance a speedrun is supposed to produce. We now know where the original dense lane sat relative to a public reference, and that was enough to start the research.

The closure pass carried the same idea one step further. Instead of using the public anchor only as a post-hoc comparison, we turned it into the live stop target `target_loss = 3.28`. Under that upgraded loop, the current dense `bf16` and `fp8` lanes both reach the target at step `8320`; `nvfp4` does not and ends at `3.3047` after the full `9536`-step horizon. The anchor has become something better than a historical curiosity: it is now part of the instrument.


## 2. Decide What Counts As A Win

A speedrun with only one number teaches you to worship the wrong god.

Loss matters. It also lies by omission. You can improve the benchmark while quietly making the model worse, making the system much harder to run, or making routing dynamics brittle enough that the apparent win does not survive the next regime.

`0002` argued that the honest MoE observation surface is a small bundle: optimization, capability, and router health. In the speedrun loop, that means the objective is better written as

$$
J(m) = \big(L_T(m),\; Q_T(m)\big),
$$

with diagnostics `S_T(m)` used to reject pathological wins.

For `nmoe`, the capability surface we care about most is `CORE`, inspired in part by the way `nanochat` treats end-of-run evaluation as first-class instead of an afterthought. The historical January calibration lanes that originally shaped this post still matter because they show the transition. Those runs were configured with `eval_tasks = core`, but they still ran with `eval_enabled = false`, so they gave us the loss-side answer without the capability-side answer.

The closure pass finishes that upgrade. We reran the loop as a `3 x 3` matrix — `dense`, `MoE-64`, `MoE-256` crossed with `bf16`, `fp8`, and `nvfp4` — with end-of-run `CORE` enabled on every lane. That is the instrument we actually wanted: the speedrun does not end when the loss target is crossed; it ends when the capability report arrives too.

That changes what the loop can say. In `bf16`, the sparse lanes reach the public loss target earlier than dense, but dense still keeps the best `CORE`. In `fp8`, `MoE-256` reaches the target earliest and posts the best `CORE`. In `nvfp4`, none of the lanes reaches target at all. Without `Q_T(m)`, those three stories collapse into one vague claim about "better loss curves." With it, the speedrun starts acting like a real architecture lab.

## 3. Use The Speedrun To Compare Dense And Sparse Designs

Once the loop is calibrated well enough, the next use is the one we actually built `nmoe` for: dense-vs-MoE architecture research.

For the first swap, we used a simple comparison axis: **active FFN width per token**.

- dense FFN active width per token: `W_active = inter_dim`
- MoE FFN active width per token: `W_active = (K + shared) * moe_inter_dim`

For the closure matrix, we keep that axis fixed and then cross it with precision. The family members are:

| Config | Experts (E) | Active (K) | Shared | Active FFN dim |
|--------|-------------|------------|--------|----------------|
| Dense | `1` | `1` | `0` | `3072` |
| MoE-64 | `64` | `6` | `2` | `8 x 384 = 3072` |
| MoE-256 | `256` | `7` | `1` | `8 x 384 = 3072` |

All nine closure lanes share the same June-style speedrun family: `9536` maximum steps, `256 x 2048` tokens/step, `AdamW` at `1.8e-3` / `0.1` / `(0.9, 0.95)`, `SDPA`, `target_loss = 3.28`, and end-of-run `CORE`.

The completed matrix says:

| Dtype | Model | Stop | Val Loss | CORE |
|-------|-------|------|----------|------|
| `bf16` | Dense | `8320` (`target`) | `3.2725` | `0.060865` |
| `bf16` | MoE-64 | `5760` (`target`) | `3.2769` | `0.050558` |
| `bf16` | MoE-256 | `4864` (`target`) | `3.2778` | `0.051878` |
| `fp8` | Dense | `8320` (`target`) | `3.2758` | `0.057261` |
| `fp8` | MoE-64 | `6272` (`target`) | `3.2684` | `0.060741` |
| `fp8` | MoE-256 | `4864` (`target`) | `3.2796` | `0.070765` |
| `nvfp4` | Dense | `9536` (`full run`) | `3.3047` | `0.049430` |
| `nvfp4` | MoE-64 | `9536` (`full run`) | `3.5950` | `0.030931` |
| `nvfp4` | MoE-256 | `9536` (`full run`) | `3.4573` | `0.049050` |

<figure>
  <img src="/figures/speedrun_closure_matrix.svg" alt="Nine-lane speedrun closure matrix across bf16, fp8, and nvfp4 for dense, MoE-64, and MoE-256, showing stop step, final loss, and CORE." />
  <figcaption>The same June-style speedrun contract tells three different stories. In `bf16`, sparse reaches target earlier but dense still keeps the best `CORE`. In `fp8`, `MoE-256` wins both on stop step and on `CORE`. In `nvfp4`, none of the lanes reaches target, but `MoE-256` is clearly healthier than `MoE-64`.</figcaption>
</figure>

Three things jump out.

First, in `bf16` and `fp8`, expert sparsity is not a toy effect. Both sparse families reach the public loss target sooner than dense, and `MoE-256` is fastest in both precisions.

Second, `CORE` prevents the loss answer from pretending to be the whole answer. In `bf16`, dense still has the strongest capability score even though it reaches the loss target later. In `fp8`, `MoE-256` wins both on stop step and on `CORE`. The "best" architecture depends on what the instrument is actually asked to optimize.

Third, `nvfp4` changes the story rather than just shifting it. None of the three `nvfp4` lanes reaches target. But `MoE-256` is materially healthier than `MoE-64` on both loss and `CORE`, which is exactly the kind of precision-by-architecture interaction the speedrun loop is supposed to expose quickly.

That is exactly what we wanted from the speedrun: not a final scaling law, but a fast honest answer to whether a sparse direction is worth more of our time under a specific contract.

## 4. The Model Answer And The Systems Answer Arrive Together

One of the nicest things about the speedrun is that it gives the model answer and the systems answer almost at the same time.

The historical `bf16` surface that originally motivated this post is still useful here. Before the closure matrix had end-of-run `CORE`, it already told us something important: MoE could win on loss while paying a steep systems tax.

<figure>
  <img src="/figures/speedrun_throughput_overlay.svg" alt="Historical speedrun throughput overlay: dense vs MoE-64 vs the longer MoE-256 slice." />
  <figcaption>Historical `bf16` rank-0 telemetry for the same `524,288` tokens/step family. The sparse lanes win on loss before they win on throughput.</figcaption>
</figure>

Median over steps `1000–7000` on that historical surface:

| Model | tokens/s/GPU | ms/step | TFLOPs |
|-------|--------------:|--------:|-------:|
| Dense SDPA | `~93k` | `~705ms` | `~137` |
| MoE-64 bf16 | `~48k` | `~1367ms` | `~71` |
| Historical MoE-256 iso-tokens/expert slice | `~47k` | `~1404ms` | `~70` |

That divergence is one of the main reasons to use a speedrun in the first place. It tells us quickly when a modeling gain is being bought with a systems tax we have not earned back yet.

## 5. Why Sparse Comparisons Get Messy So Quickly

Even after the closure matrix, the sparse comparison surface gets complicated.

At fixed tokens per step and fixed horizon, the signal available to each routed expert falls roughly as `K / E`. On the completed full-horizon `nvfp4` closure, the difference is already large:

| Config | Experts (E) | Routed K | Total Tokens | Tokens/Expert |
|--------|-------------|----------|-------------:|--------------:|
| MoE-64 | `64` | `6` | `4.9996B` | `468.7M` |
| MoE-256 | `256` | `7` | `4.9996B` | `136.7M` |
| Super-4096 | `4096` | `7` | `~6.3B` | `10.8M` |

<figure>
  <img src="/figures/tokens_per_routed_expert.svg" alt="Tokens per routed expert vs number of experts for MoE-64, MoE-256, and Super-4096." />
  <figcaption>At fixed tokens/step, increasing `E` reduces tokens per routed expert roughly as `K/E`. The dedicated `MoE-256` closure lane removes one old confound, but not the deeper one.</figcaption>
</figure>

That is why sparse research gets confusing so quickly. “More experts” is not one clean change. It moves capacity, systems cost, and signal per expert at the same time.

This is the real handoff to `0004`. `0003` introduces the speedrun loop and closes the first architecture-by-precision matrix. `0004` asks the next question: once sparse comparisons are this confounded, what fairness contract should organize the family at all?

## What The Speedrun Buys Us

After this post, I want the reader to carry five ideas forward:

| Idea | Why it matters |
|------|----------------|
| Speedruns are real research instruments | They let us turn ideas over in hours instead of weeks. |
| Public anchors can become operational stop rules as well as post-hoc comparisons | The Keller Jordan baseline now acts as both an honesty boundary and an operational target. |
| The objective is loss plus capability | The completed `3 x 3` closure matrix shows why `CORE` changes the story. |
| Precision can flip the sparse answer | `bf16`, `fp8`, and `nvfp4` do not tell the same dense-vs-MoE story. |
| Dense-vs-sparse comparisons need a declared contract | Active FFN width per token is a first pass; the harder fairness story starts next. |

## Receipts

- bundle: `repro/0003.receipts.json`

This bundle backs three layers of evidence: the pinned public June 2024 anchor (code commit, record directory, exact log path, and step-`9536` value), the historical January calibration lanes that originally exposed the anchor gap and throughput tax, and the completed March 2026 `3 x 3` closure matrix across `dense` / `MoE-64` / `MoE-256` and `bf16` / `fp8` / `nvfp4`, including exact stop steps, final losses, `CORE` scores, and cluster log paths for every lane.

Schema-validation command:

```bash
python3 scripts/repro/verify_post_receipts.py \
  --repo-root . \
  --receipts-dir repro \
  --post 0003
```

Path validation still requires `--check-paths` plus the referenced artifact mount.
