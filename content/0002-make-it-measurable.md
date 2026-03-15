---
title: "Make It Measurable"
subtitle: "What to track when loss isn't enough"
date: 2026-03-14
author: "xjdr"
number: "0002"
series: "methodology"
status: "result"
receipts:
  - nmoe/repro/0002.receipts.json
---

> Post 0002: what I track for MoE (`bpb` + `CORE` + router health), and why “loss went down” is not a sufficient stopping condition.

Loss is a good training metric right up until it is not.

With dense models, I can usually get away with watching the loss curve and using capability evals as a slower, noisier overlay. With MoE, that stopped being enough. I had runs where loss kept improving while the router was clearly converging to something pathological. I had eval setups that looked fine until distributed execution made them deadlock or lie.

That is what forced this measurement stack.

I want one smooth compression curve, one noisy capability curve, and one set of router-health canaries. If those disagree, the disagreement is the result.

Receipts for this post live in `nmoe/repro/0002.receipts.json`. They cover the exported healthy-vs-collapsed router comparison, the three-panel `bpb`/`CORE`/router-health slice, the miniseries `CORE` summary used for the task-bootstrap variance check, and the distributed lockstep failure contract. Broader multi-seed grids and larger galleries move to later posts.

## The Problem With Loss

For dense models, loss is usually enough to tell me whether training is moving in the right direction. If it goes down, I am probably learning. Capability is noisier, but it tends to move in the same direction.

MoE breaks this. Loss can go down while the router converges to a clearly pathological state. You still get a pretty curve. You also end up training a smaller model wearing a larger model's coat.

So every MoE run now gets three views: a smooth compression metric for learning progress, a capability read for whether the model is becoming more useful, and router-health signals for whether the sparse system is behaving at all.

<figure>
  <img src="/figures/eval_loop.svg" alt="Evaluation loop: train, token-indexed checkpoints, bpb/CORE/router health, decisions." />
  <figcaption>One dial, fixed checkpoint fractions, and multiple signals so we do not confuse “loss went down” with “the system behaved as intended.”</figcaption>
</figure>

### A concrete failure case

This is the simplest reason “watch the loss” fails for MoE: the optimizer can keep making progress even as the router collapses into a much narrower effective expert system.

In a `Super-4096` run (`4096` routed experts, top-`7` routing), validation loss keeps improving smoothly while the router's `max_load` rapidly saturates to the theoretical ceiling of about `100/K %`—equivalently `1/K` in fraction units, meaning one expert appears in almost every token's top-`k`.

<figure>
  <img src="/figures/super4096_loss_router.svg" alt="Validation loss improves while router max_load saturates to 1/K." />
  <figcaption>Loss alone would tell you “great run.” Router health tells you that you are training a hot subnetwork.</figcaption>
</figure>

## What I Actually Track

| Signal | Primary tags/artifacts | Unit | Source | Why it exists |
|--------|-------------------------|------|--------|---------------|
| compression | `valid/loss`, `valid/bpb`, `valid/tokens`, `valid/bytes` | nats/token, bits/byte | `nmoe/metrics.py`, `nmoe/token_bytes.py` | smooth optimization progress, tokenizer-agnostic comparison |
| capability | `eval/CORE`, `eval/core/<task>/centered`, `core_summary.json` | centered accuracy | `nmoe/eval/core/runner.py` | task-level capability movement above random baselines |
| router balance | `router_agg/mean_cv`, `router_agg/min_entropy`, `router/layer_XX/max_load` | CV%, nats, max_load% | `nmoe/metrics.py` | detect collapse hidden by loss curves |
| per-layer health | `router/layer_XX/{cv,entropy,experts_active,bias_range}` | mixed | `nmoe/metrics.py` | localize failure modes to specific layers |
| throughput/runtime | `throughput/*`, `efficiency/*`, `gpu_*` | tokens/s, ms, utilization | `nmoe/metrics.py` | separate optimization quality from systems bottlenecks |

## Bits-Per-Byte: The Smooth Signal

Cross-entropy loss is tokenizer-dependent. A model with `32k` vocab and a model with `200k` vocab report different losses for the same underlying compression quality. That makes it hard to compare across papers, across tokenizers, and even across your own experiments if you change tokenization.

`Bits-per-byte` (`bpb`) fixes that. It measures how many bits you need to encode each byte of the original text:

$$
\text{bpb} = \frac{\text{loss} \times \text{tokens}}{\text{bytes} \times \ln(2)}
$$

A model with `bpb = 0.7` needs `0.7` bits to encode each byte of the original text. That number means the same thing regardless of tokenizer.

We compute `bpb` on a fixed validation set—same data, same order, every time. That gives us a smooth curve that tracks learning progress without the noise of capability benchmarks.

### Bpb data contract and sanity check

For `bpb` checkpoints we log the following tags:

| Tag | Meaning |
|-----|---------|
| `valid/loss` | mean NLL in nats/token under the validation-loss mask |
| `valid/tokens` | evaluated token count under the validation-loss mask |
| `valid/bytes` | decoded byte count for the `bpb` path; EOS is excluded here |
| `valid/bpb` | canonical `bpb` tag from the dedicated `bpb` loss accumulator |

Important current-code nuance: `valid/bpb` is the canonical compression tag. In the current public trainer, the `bpb` path always excludes EOS, while `valid/loss` and `valid/tokens` only exclude EOS when `loss_mask_eos=true`. That means `valid/loss * valid/tokens / (valid/bytes * ln(2))` is only an exact recompute when those masks match.

Sanity query—exact when the validation-loss mask and the `bpb` mask match, for example `loss_mask_eos=true`:

```sql
WITH t AS (
  SELECT
    step,
    max(CASE WHEN tag='valid/loss' THEN value END) AS loss,
    max(CASE WHEN tag='valid/tokens' THEN value END) AS tokens,
    max(CASE WHEN tag='valid/bytes' THEN value END) AS bytes,
    max(CASE WHEN tag='valid/bpb' THEN value END) AS bpb
  FROM read_parquet('<metrics_dir>/<run_id>/step_*.parquet', union_by_name=true)
  WHERE tag IN ('valid/loss','valid/tokens','valid/bytes','valid/bpb')
  GROUP BY step
)
SELECT
  step,
  bpb,
  loss * tokens / (bytes * ln(2)) AS bpb_recomputed,
  abs(bpb - loss * tokens / (bytes * ln(2))) AS abs_err
FROM t
WHERE bpb IS NOT NULL
ORDER BY step;
```

For the current miniseries receipts, treat the exported `valid/bpb` tag as canonical. If we want unconditional public recomputability from logged tags alone, we should also log the `bpb`-specific loss sum or force the two masks to match. Older run layouts may store metrics in `rank_0.duckdb`; the same masking caveat applies there.

## CORE: The Capability Signal

Validation loss tells you the model is compressing better. It says very little about question answering, instruction following, or code reasoning.

For that, we use `CORE` (Capability-Oriented Evaluation): a suite of multiple-choice benchmarks built from tasks such as `MMLU`, `ARC`, `HellaSwag`, and `WinoGrande`. The model sees a question with a small set of options, and we ask whether it assigns the highest probability to the correct one.

The trick is centered scoring. Raw accuracy is misleading because random baselines differ. `MMLU` has `4` choices, so chance is `25%`. `WinoGrande` has `2` choices, so chance is `50%`. A model scoring `35%` on `MMLU` is doing better than one scoring `55%` on `WinoGrande`, even though the raw number is lower.

Centered scoring fixes this:

$$
\text{centered} = \frac{\text{accuracy} - \text{baseline}}{1 - \text{baseline}}
$$

The simplest example is a two-task comparison:

| Task | Choices | Baseline | Accuracy | Centered |
|------|---------|----------|----------|----------|
| A | `2` | `0.50` | `0.60` | `0.20` |
| B | `4` | `0.25` | `0.40` | `0.20` |

Raw accuracy says `60% > 40%`. Centered scoring says both performances land the same distance above random. That is the quantity I actually care about. Once you center every task this way, you can average across tasks and get a single capability number that means something.

`CORE` is noisier than `bpb`; benchmark variance is real. What it gives you is a read on competence beyond raw compression.

To get a feel for that noise, I bootstrap the task-level centered scores from the miniseries `CORE` summary exported in this post's receipts (`22` `CORE` tasks per checkpoint):

| Checkpoint | CORE | Bootstrap 95% CI |
|------------|------|------------------|
| step `863` | `-0.0138` | `[-0.081, 0.031]` |
| step `2156` | `+0.0039` | `[-0.071, 0.053]` |
| step `4312` | `+0.0237` | `[-0.025, 0.059]` |
| step `8624` | `+0.0649` | `[+0.024, +0.112]` |
| delta `863 -> 8624` | `+0.0787` | `[+0.031, +0.137]` |

Takeaway: the long-span improvement across this slice is detectable, but individual checkpoints still carry wide uncertainty bands. Small intermediate deltas are easy to over-read unless you add more tasks or repeat seeds.

<figure>
  <img src="/figures/miniseries_bpb_core_example.svg" alt="Miniseries example: bpb (smooth), CORE (capability), and router CV (health) vs fraction-of-horizon." />
  <figcaption>`bpb` is the smooth physics curve. `CORE` is the capability overlay. Router health is the MoE sanity check that tells you whether the curve means what you think it means.</figcaption>
</figure>

## Router Health: The MoE-Specific Sanity Checks

This is where MoE diverges from dense. You need to watch the routing itself, because unhealthy routing can hide behind good loss.

A quick note on units: some pre-`nmoe` prototypes reported normalized router metrics, for example `CV` as a decimal or entropy normalized to `[0,1]`. In `nmoe`, we log `CV%` and entropy in nats, and only derive normalized views later for intuition.

### Load balance (`CV`)

The coefficient of variation measures how unevenly tokens are distributed across experts:

$$
\text{CV} = \frac{\text{std}(\text{tokens per expert})}{\text{mean}(\text{tokens per expert})}
$$

In `nmoe` we log `CV` as a percentage, so `CV = 87%` means `std = 0.87 x mean`.

In healthy routing, `CV` stays bounded. Experts get roughly comparable traffic. When `CV` explodes into the thousands of percent, you have entered the regime where a few experts get almost everything.

### Entropy

Entropy measures routing diversity. We log raw Shannon entropy, in nats, of the per-expert load distribution:

$$
\mathrm{H} = - \sum_i p_i \log p_i
$$

I usually read it three ways. The maximum is `log(E)`. `H_norm = H / log(E)` puts you back on a `[0,1]` scale, where `1.0` means uniform routing. `E_eff = exp(H)` gives the effective expert count—how many experts you are really using.

### Max load

`nmoe` logs `max_load` as the percentage of routed assignments going to the single most popular expert. Divide by `100` to get the corresponding fraction. In an `E`-expert model with uniform routing, that fraction would be `1/E`.

For top-`K` routing there is an even sharper signature: if one expert appears in almost every token's top-`K`, `max_load` will asymptote to approximately

$$
\mathrm{max\_load} \approx \frac{1}{K}
$$

When `max_load` pins to `1/K` across layers and stays there, you are no longer seeing healthy broad utilization. You are seeing a dominant-expert attractor.

### Dead experts and min entropy

Dead experts count how many experts received zero routed assignments in a step, aggregated across layers. A few dead experts can happen at small per-expert batch. Persistent mass death means you are training a smaller effective model than you think.

Mean entropy can also look healthy while one layer quietly collapses. That is why I track `router_agg/min_entropy` as a canary. It catches the layer that is going bad before the mean admits anything is wrong.

## What Healthy Looks Like

Here is what the metrics look like for a healthy `MoE-64` run versus the collapsed `Super-4096`:

| Metric | MoE-64 bf16 (healthy) | Super-4096 (collapsed) |
|--------|----------------------|------------------------|
| Final loss | `3.43` | `3.47` |
| mean CV% | `87%` | `1733%` |
| mean entropy | `3.91` nats | `4.15` nats |
| min entropy | `3.77` nats | `2.25` nats |
| `E_eff` at min entropy (`exp(H_min)`) | `~44` experts | `~9` experts |
| `L0 max_load` | `16.6%` (`1/K` for `K=6`) | `14.3%` (`1/K` for `K=7`) |

The loss looks similar. The router health does not. `MoE-64` is using most of its capacity. `Super-4096` keeps a high mean entropy, but its worst layer drops to about `9` effective experts while `CV` and `max_load` both scream collapse.

`MoE-64`'s `L0` sits at `1/K`, but only that one layer does. The rest of the network stays healthy. In corrected-stack `Super-4096`, many layers eventually pin there, but the onset is depth-nonuniform rather than simultaneous. `0006` covers that progression layer by layer.

<figure>
  <img src="/figures/router_health_healthy_vs_collapsed.svg" alt="Router health comparison: healthy MoE-64 vs collapsed Super-4096 showing CV, entropy, and max_load over training." />
  <figcaption>Same loss, completely different routing. The healthy run maintains bounded `CV` and high entropy. The collapsed run explodes to `1700%` `CV` and pins `max_load` at `1/K`.</figcaption>
</figure>

This is why I track router health alongside loss. You cannot see the difference from the loss curve alone.

## Where These Numbers Come From

`nmoe` writes metrics as per-step parquet under `<metrics_dir>/{run_id}/step_XXXXXXXX.parquet` from a DuckDB-backed in-memory writer.

| Scope | Tags |
|-------|------|
| per-layer | `router/layer_XX/{cv,entropy,max_load,experts_active,bias_range}` |
| aggregates | `router_agg/{mean_cv,std_cv,mean_entropy,min_entropy,dead_experts_count,experts_active_mean}` |

The implementation is short enough to show directly:

```python
# nmoe/metrics.py (simplified)
l = ffn.last_loads.detach().float()  # per-expert assignment mass for this MoE layer
m = l.mean()
cv = (l.std(unbiased=False) / m * 100.0).item()  # CV%
mx = (l.max() * 100.0).item()                   # max_load%
p = (l / l.sum().clamp_min(1e-9)).clamp_min(1e-12)
entropy = (-p * p.log()).sum().item()           # nats
```

A query pattern I use constantly:

```sql
-- mean CV over time
SELECT step, value
FROM read_parquet('<metrics_dir>/<run_id>/step_*.parquet', union_by_name=true)
WHERE tag = 'router_agg/mean_cv'
ORDER BY step;

-- per-layer max_load at one step
SELECT tag, value
FROM read_parquet('<metrics_dir>/<run_id>/step_00009536.parquet', union_by_name=true)
WHERE tag LIKE 'router/layer_%/max_load'
ORDER BY tag;
```

We keep the public posts free of internal run ids, but internally this is still how I move from story to receipts.

Healthy depends on the regime (`E`, `K`, batch, horizon), but the same invariants keep showing up: all experts stay active at reasonable batch sizes, `CV` stays bounded, entropy stays high, and `max_load` does not pin to `1/K` across layers. When one of those breaks, I start looking for a hidden hot subnetwork.

The same views reappear later in the series: `0003` uses them in the speedruns, `0004` overlays them on the miniseries curves, and `0006` uses them to show collapse propagating layer by layer.

## The Engineering Gotcha

There is a subtlety with MoE eval that took me longer than it should have to make fully explicit.

Under expert parallelism, the model is distributed across GPUs. When you forward a batch, the router on GPU `0` might send tokens to an expert on GPU `3`. If GPU `3` is doing something else, like evaluating a different prompt, you deadlock.

That means evaluation has to run in lockstep: all GPUs process the same prompts at the same time. You cannot shard the eval dataset across GPUs the way you would for dense models.

Once you see the problem, the fix is simple. Everyone evaluates the same prompts in lockstep, and rank `0` accumulates the centered results. It is less efficient than sharding, but it is correct, and correctness is the whole point.

Wrong path for distributed MoE `CORE`—the one that deadlocked or invalidated eval in earlier iterations:

```bash
torchrun --nproc_per_node=8 -m nmoe.eval.core.runner \
  --snapshot <metrics_dir>/eval/<run_id>/step_0000863/eval_snapshot.pt \
  --tasks-file configs/eval/core.toml
```

Current fail-loud guard:

```
RuntimeError: CORE snapshot eval is not supported for MoE under distributed execution (world_size=8). Run eval_tasks=core with eval_mode=inline (live lockstep), or evaluate the snapshot with world_size=1.
```

That message exists for a reason. The alternative is a hang or, worse, an eval that returns numbers that do not correspond to a real distributed forward.

The broader eval tree still has other snapshot-oriented runners. The narrower claim here is about the distributed MoE `CORE` contract: for that case, the correct path is live lockstep `CORE` on the in-memory sharded model (`nmoe.eval.core.runner.run_core_live` via training integration).

Receipts for this post are in `nmoe/repro/0002.receipts.json`.

Post `0003` is where we used this stack to calibrate the setup against a known-good dense baseline before trusting new MoE results.
