---
title: "What Are We Holding Fixed?"
subtitle: "Dense-vs-MoE comparison depends on the fairness contract; a failed `#420` transfer exposed the real problem"
date: 2026-03-14
author: "xjdr"
number: "0004"
series: "methodology"
status: "result"
draft: false
receipts:
  - "nmoe/repro/0004.receipts.json"
  - "nmoe/repro/0003.receipts.json"
---

Dense training lets us pretend that model size has one face. Sparse training does not. The moment we move from dense FFNs to experts, model size splits in two: there is the model we store, and there is the smaller model any one token can actually touch. Most dense-vs-MoE arguments quietly choose one of those objects and call it fairness.

`0004` exists because we did that too.

We started by trying to transfer Karpathy's `#420` loop as literally as we could. The instinct was good. A token-indexed horizon, fixed-fraction checkpoints, and one public family slice are exactly the sort of disciplined contract that prevents research from dissolving into vibes. But the first transfer taught us something more important than the result it produced. It taught us that, in MoE work, the choice of fairness contract is itself part of the experiment.

So the real subject of this post is not whether one small sparse family beat another. It is a simpler and more dangerous question:

**when we compare dense training to expert sparsity, what exactly are we holding fixed?**

## 1. The Useful Failure

The first transfer was deliberately conservative. We kept the dense `#420` shape as intact as we could: token-indexed horizons, fraction-of-horizon checkpoints, and one published family slice under `totalDN = 8`.

$$
T_{\mathrm{total}}(m) = \alpha \cdot P_{\mathrm{total}}(m), \qquad \alpha = 8.
$$

For the published slice, that produced the following headline rows at `20%` of horizon.

| Family | Depth | Routed Experts | Tokens at `20%` | bpb | CORE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d10` | `10` | `64` | `759M` | `1.40` | `+0.001` |
| `d12` | `12` | `256` | `4.5B` | `1.21` | `+0.065` |

<figure>
  <img src="/figures/miniseries_bpb_core_example.svg" alt="The original d10 versus d12 miniseries slice under the token-indexed #420-style contract." />
  <figcaption>The literal transfer looked promising on its face: `d12` separated on both `bpb` and `CORE`. It also changed more than one dial. The deeper model stored more experts and consumed a much larger token budget under the `totalDN` horizon.</figcaption>
</figure>

That slice was useful precisely because it was honest. It told us two true things at once.

First, a dense-style `#420` loop really does transfer to MoE in one limited sense. Token-indexed horizons and fraction-of-horizon checkpoints produce a legible family slice instead of the usual step-count chaos.

Second, a legible slice is not automatically a fair one. In this transfer, depth changed, routed expert count changed, and horizon changed with total stored parameters. The local result may have been interpretable, but the causal dial was not.

That failure is the reason this paper exists.

## 2. Model Size Splits In Two

The right way to say the problem mathematically is that MoE gives us at least two natural parameter objects.

Let `P_total(m)` be the total number of trainable parameters in model `m`. That is the dense notion of model size: everything the model stores, whether a token uses it or not.

Let `P_active(m)` be the parameter mass a token can actually touch. In `nmoe`, the canonical definition is the one encoded in `nmoe.metrics.param_counts`:

$$
P_{\mathrm{active}}(m) = P_{\mathrm{dense}}(m) + L_{\mathrm{moe}}\left(H E + 3 H D_{\mathrm{ff}}^{\mathrm{moe}} K\right) + P_{\mathrm{shared}}(m),
$$

with

$$
P_{\mathrm{shared}}(m) = L_{\mathrm{moe}}\left(3 H D_{\mathrm{ff}}^{\mathrm{moe}} S\right).
$$

Here `H` is width, `E` routed experts, `K` activated experts, `S` shared experts, `D_ff^moe` expert FFN width, and `L_moe` the number of MoE layers.

For dense models, these two objects collapse: `P_total = P_active`. For MoE they do not. Stored capacity and active capacity diverge by design.

That gives us two equally natural horizon contracts:

$$
T_{\mathrm{total}}(m) = \alpha \cdot P_{\mathrm{total}}(m), \qquad
T_{\mathrm{active}}(m) = \beta \cdot P_{\mathrm{active}}(m).
$$

| Contract | What is held fixed | What question it answers |
| --- | --- | --- |
| `totalDN` | tokens per stored parameter | How much data does the whole stored model get? |
| `activeDN` | tokens per active parameter | How much data does the per-token computation actually get? |

Neither contract is silly. They simply answer different questions. Dense training hides that distinction because the two objects coincide. Sparse training forces us to choose.

There is one more local fairness axis worth keeping explicit. In `0003` we matched active FFN width per token:

- dense: `W_active = inter_dim = 3072`
- `MoE-64`: `W_active = (K + shared) * moe_inter_dim = 8 x 384 = 3072`
- `MoE-256`: `W_active = (K + shared) * moe_inter_dim = 8 x 384 = 3072`

That matters. It kills one easy confound. It does **not** settle the global fairness question, because equal active FFN width per token does not imply equal `P_total` or equal `P_active` over the whole model.

## 3. The Active-Like Side Is Already Real

The good news is that one half of the fairness story is already closed.

The completed `0003` closure matrix kept dense, `MoE-64`, and `MoE-256` on the same `9536`-step, `4.9996B`-token June-style speedrun family. Once we compute the exact parameter objects, that turns out to be an almost-perfect active-parameter comparison.

| Model | `P_total` | `P_active` | tokens / `P_total` at `9536` | tokens / `P_active` at `9536` |
| --- | ---: | ---: | ---: | ---: |
| Dense | `190,532,352` | `190,532,352` | `26.24` | `26.24` |
| `MoE-64` | `755,534,592` | `191,073,024` | `6.62` | `26.17` |
| `MoE-256` | `2,836,482,816` | `192,891,648` | `1.76` | `25.92` |

Those active-parameter ratios are close enough that the existing `0003` matrix already behaves like the `activeDN` side of the comparison. The total-parameter ratios are nowhere near equal, so the same matrix is decisively **not** a `totalDN` result.

<figure>
  <img src="/figures/speedrun_closure_matrix.svg" alt="Nine-lane speedrun closure matrix across bf16, fp8, and nvfp4 for dense, MoE-64, and MoE-256, showing stop step, final loss, and CORE." />
  <figcaption>The completed closure matrix from `0003` closes the matched active-parameter side of the story. The three models saw almost the same tokens per active parameter, even though sparse storage grew dramatically across the family. Precision matters too, but the `nvfp4` story belongs to `0005`.</figcaption>
</figure>

If we set the precision pathology aside for the moment and look only at the `bf16` and `fp8` lanes, the active-like record says this:

| Dtype | Dense | `MoE-64` | `MoE-256` | Main read under the active-like contract |
| --- | --- | --- | --- | --- |
| `bf16` | target at `8320`, `CORE = 0.060865` | target at `5760`, `CORE = 0.050558` | target at `4864`, `CORE = 0.051878` | sparse reaches the loss target sooner, but dense still keeps the best `CORE` |
| `fp8` | target at `8320`, `CORE = 0.057261` | target at `6272`, `CORE = 0.060741` | target at `4864`, `CORE = 0.070765` | `MoE-256` wins both on stop step and on `CORE` |

That is already a substantive result. Under an almost-equal active-parameter horizon and a matched active-width swap axis, expert sparsity is not a toy effect. It clearly changes the answer.

But it is still only one contract.

## 4. The Stage-1 `totalDN` Answer

The strongest version of this paper is still the statement

$$
\mathrm{rank}(m \mid T_{\mathrm{active}}) \neq \mathrm{rank}(m \mid T_{\mathrm{total}})
$$

for at least part of the dense-vs-MoE family.

That is the real claim worth earning. If the ranking changes when the horizon contract changes, then fairness is not bookkeeping. It is the result.

If we preserve the dense-derived coefficient from `0003`,

$$
\gamma = \frac{4,999,610,368}{190,532,352} = 26.2402175563,
$$

then the full `totalDN` horizons are:

| Model | full `totalDN` steps |
| --- | ---: |
| Dense | `9536` |
| `MoE-64` | `37813` |
| `MoE-256` | `141963` |

Those full horizons are too expensive for a same-day publish pass, so the first closure pass uses the same `20%` fraction that made the original `d10` / `d12` slice readable at all. Dense needs no new training because `P_total = P_active` there and the dense curve is already fixed. The new work is sparse.

| Model | stage-1 `totalDN20` steps | final valid loss | CORE | Main read |
| --- | ---: | ---: | ---: | --- |
| Dense | existing curve reused | --- | --- | the dense baseline is already fixed; the stage-1 new work is sparse |
| `MoE-64` | `7563` | `3.2625` | `0.051871` | reaches the old speedrun target, but only modestly changes the active-like picture |
| `MoE-256` | `28393` | `2.8554` | `0.145743` | pulls away sharply once stored capacity gets token budget proportional to what the model stores |

The important part is not just the endpoint. Once the contract changes, the schedule changes with it. That means target-hit step alone is not enough. The right comparison surface is fixed fractions of each contract, plus `CORE`.

The stage-1 table already decides the first thing we needed to know. The sparse ordering does **not** flip under `totalDN`; it becomes much stronger. Under the active-like `bf16` contract, `MoE-64` and `MoE-256` are both fast-to-target sparse wins with similar end-of-run `CORE`. Under the staged `totalDN` contract, `MoE-256` keeps training for much longer, ends far lower on loss, and posts a dramatically stronger capability score than `MoE-64`.

## 5. What The Stage-1 Table Already Says

The honest state of the paper now looks like this.

| Contract | Dense | `MoE-64` | `MoE-256` | What the contract says |
| --- | --- | --- | --- | --- |
| active-like (`0003`, `bf16`) | stop `8320`, `CORE = 0.060865` | stop `5760`, `CORE = 0.050558` | stop `4864`, `CORE = 0.051878` | sparse reaches the loss target sooner, but dense still keeps the best `CORE` |
| active-like (`0003`, `fp8`) | stop `8320`, `CORE = 0.057261` | stop `6272`, `CORE = 0.060741` | stop `4864`, `CORE = 0.070765` | `MoE-256` wins both on stop step and on `CORE` |
| `totalDN20` (`0004`, `bf16`) | dense curve already fixed; no new rerun in this stage | `7563`, `3.2625`, `CORE = 0.051871` | `28393`, `2.8554`, `CORE = 0.145743` | the sparse ordering survives and the margin widens dramatically once horizon follows stored capacity |

That is already enough to change the interpretation. The active-like contract asks what happens when per-token computation gets equal data. The staged `totalDN` contract asks what happens when stored capacity gets equal data. On this family, those two questions do not tell the same story.

## 6. Current Answer

Here is the answer that stands today.

1. A dense-style `#420` loop does transfer to MoE in one important sense: token-indexed horizons and fraction-of-horizon schedules produce interpretable slices.
2. The first literal transfer was valuable because it failed honestly. It exposed that depth, sparsity, and token budget were entangled.
3. `0003` closes the matched active-parameter side of the fairness question for dense versus `MoE-64` versus `MoE-256`.
4. The completed `totalDN20` sparse pair shows that the larger sparse model keeps its lead and widens it sharply once stored capacity gets a proportionate token budget.

So the lasting claim of `0004` is still simple: in MoE research, dense-vs-sparse comparison is incomplete until the chosen contract is named. The new result is that even a staged `totalDN` probe already changes the scientific picture materially.

## Limitations

1. The published `totalDN` result is a staged `20%` closure pass; the full `37813` / `141963`-step horizons remain future work.
2. Dense at the same fixed fraction is read from the existing dense curve rather than rerun as a separate stage-1 campaign.
3. The motivating `d10` / `d12` slice remains structurally confounded and should be read as a useful failed transfer rather than a scaling law.
4. The `0003` closure matrix still closes only the matched active-parameter side of the fairness comparison.
5. Precision-dependent pathologies, especially the `nvfp4` lanes, are real but belong in `0005`.

## Receipts

- Fairness-question receipts: `nmoe/repro/0004.receipts.json`
- Active-like closure receipts: `nmoe/repro/0003.receipts.json`
- Canonical total-vs-active parameter definition: `nmoe/metrics.py`
- Verify the receipt bundles from the repo root:

```bash
python3 scripts/repro/verify_post_receipts.py --repo-root . --receipts-dir repro --post 0004
python3 scripts/repro/verify_post_receipts.py --repo-root . --receipts-dir repro --post 0003
```

## References

- Karpathy, `#420` dense-scaling methodology and dense+Muon comparison loop.
- `0003`. *The Speedrun Loop.*
- `0005`. *NVFP4 Dynamics.*
