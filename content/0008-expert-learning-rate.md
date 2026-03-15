---
title: "Do MoE Experts Need Different Learning Rates?"
subtitle: "Why Moonlet's old 15x expert-LR rule overshoots in bf16 AdamW"
date: 2026-03-14
author: "xjdr"
number: "0008"
series: "research"
status: "result"
draft: false
receipts:
  - "nmoe/repro/0008.receipts.json"
---

There is a durable MoE superstition: experts only see a thin slice of tokens, so they must need a much larger learning rate than the dense path. Moonlet inherited that superstition in the loudest possible form: `lr_expert = 15x * lr_dense`.

It sounds plausible because a clean counting argument really does say sparse routing should attenuate raw expert gradients. The trap is that the counting argument lives at the raw gradient, while training happens at the applied optimizer step. If AdamW cancels most of the attenuation before the update, then a large expert-LR multiplier is not a correction. It is an overcorrection.

This post makes that distinction earn its keep on the actual Moonlet contract. The result is narrower than the old superstition and stronger than a tuning anecdote: in Moonlet `bf16` with `AdamW`, `lr_expert = lr_dense` is the right baseline, and `15x` is wrong.

<figure>
  <img src="/figures/0008_expert_lr_sweep.svg" alt="Two-seed bf16 Moonlet sweep over expert learning-rate multipliers from 0.5x to 15x, showing 1x as the best baseline and 15x as the worst overcorrection." />
  <figcaption>The main tuning answer in one picture. On the settled `bf16` lane, `1x` is the best baseline. Smaller undercooks, larger multipliers overcook, and `15x` is the loud wrong answer Moonlet used to carry.</figcaption>
</figure>

The rest of the paper explains why that happens and where the boundary still moves.

## The Moonlet contract

We study the actual `nmoe` Moonlet contract.

| Field | Value |
| --- | --- |
| Config | `configs/moonlet.toml` |
| Architecture | `dim = 2048`, `n_layers = 12` |
| Dense FFN width | `inter_dim = 11264` |
| Routed expert width | `moe_inter_dim = 1408` |
| Routed experts | `E = 64`, `K = 6`, `n_shared_experts = 2` |
| Base LR | `lr_dense = lr_router = 3e-3` |
| Main sweep | `lr_expert = m * 3e-3`, `m in {0.5, 1, 2, 4, 15}` |
| BF16 lane | `dtype = bf16`, `expert_opt = auto -> AdamW` |
| NVFP4 diagnostic lane | `dtype = nvfp4`, `expert_opt = auto -> ExpertAdamW` |
| Update-proof runs | `m in {1, 15}`, seed `42`, metrics every `10` steps |
| NVFP4 grad-health canary | `m in {1, 15}`, seed `42`, `50` steps, extra MoE grad-health tags |
| Schedule | same Moonlet WSD schedule |
| Data | `fineweb_edu` |

The router contract matters because Moonlet multiplies gate logits by `route_scale` before `sigmoid`, chooses top-k on `scores + bias`, renormalizes the selected scores so routed weights sum to `1`, and only then applies `routed_scaling_factor` if it differs from `1`. Moonlet sets `route_scale = 2.446` and `routed_scaling_factor = 1.0`, so the routed weights are still normalized. That changes selection pressure, but it does not multiply post-normalization expert weights by `2.446` and cannot by itself justify a `15x` expert learning-rate multiplier.

This is a result post for `bf16` / `AdamW`, and a diagnostic post for `nvfp4` / `ExpertAdamW`.

## What the math says

Let one MoE block produce

$$
y_i = s(x_i) + \sum_{e \in R_i} g_{i,e} f_e(x_i; \theta_e),
$$

where `R_i` is the selected top-k expert set, `g_{i,e}` is the normalized routed weight, and `s(x_i)` is the always-on shared branch. Training uses mean-reduced cross-entropy,

$$
L = \frac{1}{N} \sum_{i=1}^{N} \ell_i.
$$

For one expert parameter block `\theta_e`,

$$
\nabla_{\theta_e} L = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[e \in R_i] \, g_{i,e} \, J_{i,e}^{\top} \delta_i.
$$

A comparable dense block participates on every token, so its gradient does not carry the routing indicator. Under balanced routing and normalized top-k weights,

$$
\mathbb{E}[\mathbf{1}[e \in R_i] \, g_{i,e}] \approx \Pr(e \in R_i) \, \mathbb{E}[g_{i,e} \mid e \in R_i] \approx \frac{K}{E} \cdot \frac{1}{K} = \frac{1}{E}.
$$

So the naive intuition gets one thing right: sparse routing really does attenuate raw expert gradients. What it still leaves open is the learning-rate multiplier, because training only ever exposes the preconditioned step.

### If the optimizer were SGD

If expert gradients are a scaled version of a shared signal,

$$
g_t^e = c \, h_t, \qquad 0 < c < 1,
$$

then plain SGD gives

$$
\Delta \theta_t^e = -\eta_e g_t^e = -\eta_e c h_t.
$$

To equalize raw step size against a dense path with step `-\eta_d h_t`, SGD would want

$$
\eta_e \approx \frac{\eta_d}{c}.
$$

In a non-adaptive optimizer, larger expert learning rates are therefore the natural correction.

### What AdamW changes

Now take an AdamW-style adaptive optimizer with decoupled weight decay. If the expert gradient is again a scaled version of some base signal,

$$
g_t^e = c \, h_t,
$$

then the moments inherit that scaling,

$$
\hat m_t^e = c \, \hat m_t^h, \qquad \hat v_t^e = c^2 \, \hat v_t^h.
$$

The expert update becomes

$$
\Delta \theta_t^e = -\eta_e \frac{c \, \hat m_t^h}{|c| \sqrt{\hat v_t^h} + \epsilon} - \eta_e \lambda \theta_t^e.
$$

This yields the key split. When `|c| sqrt(v_hat) >> eps`, the amplitude factor `c` largely cancels, so equal learning rates are already the right baseline. When `eps` or quantization noise dominates the denominator, the attenuation survives and a larger expert LR may be necessary. If expert updates are already larger than dense updates, a still larger LR is wrong and a smaller LR would be warranted instead.

That is the whole paper in one sentence: raw-gradient attenuation does not automatically imply a larger expert learning rate. The whole question lives in the gap between the tiny raw signal and the step the optimizer finally applies.

## Predictions

For Moonlet, the derivation makes three falsifiable predictions.

1. `train_signal/expert/grad_to_param` should be much smaller than `train_signal/dense/grad_to_param`.
2. At `lr_expert = lr_dense`, the expert update should stay on the dense-update scale if the adaptive optimizer is canceling most of the attenuation.
3. If we multiply expert LR far above that adaptive baseline, expert update norms should jump above dense update norms and training should either destabilize or lose quality.

## What happened

### BF16 main sweep

The fixed-contract `bf16` Moonlet sweep answers the immediate tuning question.

| Multiplier | Seed 42 | Seed 43 |
| --- | --- | --- |
| `0.5x` | `collapse @110 · 8.4672` | `finish @200 · 9.0694` |
| `1x` | `collapse @200 · 8.4627` | `finish @200 · 8.7487` |
| `2x` | `finish @200 · 9.3468` | `collapse @80 · 8.5722` |
| `4x` | `finish @200 · 9.6952` | `finish @200 · 9.4053` |
| `15x` | `finish @200 · 10.7649` | `collapse @110 · 11.5588` |

The sweep does **not** say that `1x` is magically stable in every seed. One of the `1x` runs still reaches collapse right at the end of the `200`-step window. What it does say is that the baseline is in the right neighborhood and every larger multiplier is worse. `0.5x` undercooks or collapses. `2x`, `4x`, and especially `15x` move the wrong way. `15x` is not a subtle miss; it is the wrong side of the regime boundary.

### BF16 mechanism check

The direct `bf16` update-proof runs test the derivation directly.

| Multiplier | Step | expert/dense `grad_to_param` | expert/dense `optimizer_update_to_pre_param` |
| --- | --- | --- | --- |
| `1x` | `10` | `6.89e-4` | `0.546` |
| `1x` | `50` | `1.07e-3` | `0.361` |
| `1x` | `100` | `1.72e-3` | `0.616` |
| `15x` | `10` | `1.09e-3` | `5.63` |
| `15x` | `50` | `7.41e-4` | `6.12` |
| `15x` | `100` | `1.75e-3` | `12.12` |

At `1x`, raw expert gradients are roughly `580x` to `1450x` smaller than dense by `grad_to_param`, but the applied updates stay on the same order as dense updates at about `0.36x` to `0.62x`. At `15x`, the raw gradient ratios stay tiny, but the optimizer-update ratio jumps to `5.63x`, `6.12x`, and `12.12x`; the run then collapses at step `130`.

<figure>
  <img src="/figures/0008_expert_lr_mechanism.svg" alt="Two-panel bf16 mechanism figure showing tiny raw expert-to-dense gradient ratios for both 1x and 15x, but update ratios near dense scale at 1x and far above dense scale at 15x." />
  <figcaption>The raw-gradient story barely changes between `1x` and `15x`. The applied-update story changes completely. The right baseline is set by the preconditioned step, with routing counts only supplying the tiny raw signal.</figcaption>
</figure>

The measured pattern matches the derivation.

### NVFP4 / ExpertAdamW diagnostic

The current `nvfp4` / `ExpertAdamW` evidence is diagnostic only. It is single-seed and only compares `1x` to `15x`, so it shows direction and mechanism but does not close the Moonlet `nvfp4` tuning question. Within that narrower scope, the lane differs mostly in degree.

| Multiplier | Seed | Steps | Final loss | Stop reason |
| --- | --- | ---: | ---: | --- |
| `1x` | `42` | `200` | `7.9618` | `completed` |
| `15x` | `42` | `200` | `8.1885` | `completed` |

Unlike `bf16`, `15x` does not collapse in the first `200` steps. It still loses to `1x` by the end of the window, and the update ratios show why.

| Multiplier | Step | expert/dense `grad_to_param` | expert/dense `optimizer_update_to_pre_param` |
| --- | --- | --- | --- |
| `1x` | `10` | `9.67e-4` | `0.531` |
| `1x` | `50` | `7.98e-4` | `0.283` |
| `1x` | `100` | `3.06e-3` | `0.217` |
| `1x` | `200` | `2.63e-3` | `0.338` |
| `15x` | `10` | `8.38e-4` | `5.42` |
| `15x` | `50` | `3.91e-4` | `3.91` |
| `15x` | `100` | `2.48e-3` | `3.81` |
| `15x` | `200` | `4.45e-3` | `6.10` |

So `nvfp4` sits closer to the boundary than `bf16`. At `1x`, expert updates are still smaller than dense updates by a larger margin than in `bf16`, yet they stay within dense-scale magnitudes. At `15x`, expert updates again overshoot dense. The right reading is that more attenuation survives the adaptive optimizer in `nvfp4` / `ExpertAdamW` than in `bf16` / `AdamW`, while `15x` still overcorrects for Moonlet.

### NVFP4 grad-health canary

We also ran a short `50`-step `nvfp4` canary with the new per-expert grad-health tags (`zero_frac`, `abs_mean`, `abs_max`). Treat it as extra diagnostic evidence on the expert gradient field.

| Multiplier | Step | Final loss | `w1` zero frac | `w2` zero frac | `w3` zero frac |
| --- | --- | ---: | ---: | ---: | ---: |
| `1x` | `50` | `9.5620` | `0.71%` | `0.71%` | `0.71%` |
| `15x` | `50` | `9.4193` | `13.35%` | `13.35%` | `13.35%` |

Treat this as a warning canary. The `15x` lane can grab a small early loss edge while making the expert gradient field much harsher and much sparser. By `200` steps, that edge is gone and `1x` is better.

## The answer

Here is the scoped answer. Moonlet carried a loud opinion about this question, and on the settled `bf16` lane that opinion is wrong. Experts do not need a larger learning rate than the dense path. `15x` already overshoots, and `0.5x` is not better than `1x`. The clean earned setting there is `lr_expert = lr_dense`.

The reason is that sparse routing attenuates raw expert gradients, but the adaptive optimizer cancels much of that attenuation before the step is applied. The broader lesson is regime-dependent: you have to ask how much of the sparse-routing attenuation survives the optimizer before you start multiplying expert LR by hand. On the current evidence, `bf16` / `AdamW` is the settled proof lane, while the short `nvfp4` / `ExpertAdamW` results point in the same direction without closing the full tuning question.

## What this does not say

This is not a universal MoE expert-LR law.

It does **not** settle:

- non-adaptive controls such as SGD, where the derivation predicts a different answer
- `ExpertMuon` or other adaptive families with materially different update dynamics
- the full `nvfp4` optimum, which still needs a wider sweep and longer-horizon repeats

The paper earns a narrower result than that, and a stronger one: on Moonlet `bf16` with `AdamW`, the old `15x` rule is wrong for mechanistic reasons we can now measure.

## Receipts

Receipt bundle: `nmoe/repro/0008.receipts.json`

```bash
# bf16 main sweep
bash scripts/repro/run_0008_bf16_sweep.sh blog_artifacts/0008_expert_lr_bf16_20260311
python3 scripts/repro/summarize_0008_bf16_sweep.py --kind main --study-root blog_artifacts/0008_expert_lr_bf16_20260311

# bf16 update-proof
bash scripts/repro/run_0008_bf16_updateproof.sh blog_artifacts/0008_expert_lr_bf16_updateproof_20260311
python3 scripts/repro/summarize_0008_bf16_sweep.py --kind updateproof --study-root blog_artifacts/0008_expert_lr_bf16_updateproof_20260311 --steps 10,50,100,130,200

# nvfp4 diagnostics
bash scripts/repro/run_0008_nvfp4_updateproof.sh blog_artifacts/0008_expert_lr_nvfp4_updateproof_b20260311
python3 scripts/repro/summarize_0008_bf16_sweep.py --kind nvfp4_updateproof --study-root blog_artifacts/0008_expert_lr_nvfp4_updateproof_b20260311 --steps 10,50,100,130,200
bash scripts/repro/run_0008_nvfp4_gradhealth.sh blog_artifacts/0008_expert_lr_nvfp4_gradhealth_c20260311
python3 scripts/repro/summarize_0008_bf16_sweep.py --kind nvfp4_gradhealth --study-root blog_artifacts/0008_expert_lr_nvfp4_gradhealth_c20260311 --steps 10,50

# verify receipts
python3 scripts/repro/verify_post_receipts.py --repo-root . --receipts-dir repro --post 0008
```

## References

Key references for this post are Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017); William Fedus, Barret Zoph, and Noam Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2021); Diederik P. Kingma and Jimmy Ba, "Adam: A Method for Stochastic Optimization" (2015); and Ilya Loshchilov and Frank Hutter, "Decoupled Weight Decay Regularization" (2019).
