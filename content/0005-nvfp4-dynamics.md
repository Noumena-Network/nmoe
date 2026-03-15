---
title: "NVFP4 Dynamics"
subtitle: "Why our NVFP4 recipe lagged BF16, and what actually closed almost all of the gap"
date: 2026-03-14
author: "xjdr"
number: "0005"
series: "research"
status: "result"
draft: false
receipts:
  - "nmoe/repro/0005.receipts.json"
---

> Post 0005: the old NVFP4 “fix” was actually one good compensator bundled with one bad one.

I wanted NVFP4 to trail BF16 by a small, boring tax. That is not what this recipe did. The original lane lagged badly enough that the whole story felt wrong.

The first version of this post leaned hard on dither and stiction. Those effects are real. They just were not the main thing closing the gap. Once I corrected the proof surface, the cleaner split showed up: there are two explicit NVFP4-only gains in this tree, `fp4_embed_gain` and `fp4_logits_gain`. One helps. One hurts.

That changes the diagnosis. The right current line is not “turn on the old rescue package.” It is “keep `fp4_embed_gain`, drop `fp4_logits_gain=0.125`, move the expert lane to `AdamW`, and treat the remaining `~+0.046` gap as a bug.”

<figure>
  <img src="/figures/0005_nvfp4_gain_split.svg" alt="Corrected 384-step NVFP4 gain-isolation quartet showing embed-only best, unit next, old pair worse, and logits-only worst." />
  <figcaption>The corrected quartet is the center of gravity. One gain helps, one hurts, and the historical pair should not be defended as a unit.</figcaption>
</figure>

## The question

This post asks one concrete question: why did our NVFP4 MoE speedrun stop tracking BF16, and which parts of that gap were real bugs in the recipe?

All comparisons below hold fixed:

| Aspect | Value |
|--------|-------|
| Model | MoE-64 (`E=64`, `K=6`, `shared=2`) |
| Tokens | `9536` steps × `524k` tokens/step, about `5B` tokens |
| Data | FineWeb deterministic stream |
| Eval | `valid/loss` every `128` steps |
| Primary proof surface | corrected `384`-step gain-isolation quartet + corrected `9536`-step `embed-only + adamw` rerun |

The earlier dither and router experiments still matter. In this version they sit in the supporting mechanism layer rather than carrying the headline explanation.

## One good knob and one bad one

These NVFP4-only gains were originally added because the NVFP4 lane looked like it was squashing useful signal through the expert path. That motivation was reasonable. The pair itself was not.

In the current codebase, there are exactly two explicit NVFP4-only gains on this surface:

| knob | role on this proof surface |
|---|---|
| `fp4_embed_gain` | helpful compensator |
| `fp4_logits_gain` | harmful compensator at `0.125` |

After fixing the blockscaled backward-profile replay path in `nmoe/moe.py`, I reran a corrected `384`-step quartet. The sign of the result survived that correction.

| Arm | `valid/loss @384` | vs corrected unit |
|-----|-------------------:|------------------:|
| `unit` (`1.0 / 1.0`) | `5.1676` | baseline |
| `embed-only` (`10.667 / 1.0`) | `4.9864` | `-0.1812` |
| `logits-only` (`1.0 / 0.125`) | `5.3218` | `+0.1542` |
| `old pair` (`10.667 / 0.125`) | `5.2167` | `+0.0491` |

That table is the cleanest single result in the post. If the historical pair were a coherent fix, it should beat both the unit line and the decomposed arms. It does not. `embed-only` is clearly best. `logits-only` is clearly harmful. The old pair lands in the middle because one knob is rescuing damage created by the other.

The easiest way to say it is still the truest: one knob helped, one knob hurt, and the bad one polluted the story.

## The best current line

The best corrected line we have today keeps only the helpful pieces: `fp4_embed_gain=10.667`, `fp4_logits_gain=1.0`, and `expert_opt=adamw`.

| Run | final `valid/loss` | vs BF16 |
|-----|-------------------:|--------:|
| BF16 baseline | `3.1264` | baseline |
| corrected `embed-only + adamw` | `3.1735` | `+0.0471` |

I care more about the late-window shape than one final checkpoint. From steps `7808` through `9536`, the residual stays in a tight `~+0.046` to `+0.047` band instead of reopening:

- `7808`: `+0.0451`
- `8192`: `+0.0467`
- `8576`: `+0.0465`
- `8960`: `+0.0485`
- `9536`: `+0.0471`

<figure>
  <img src="/figures/0005_nvfp4_residual_gap.svg" alt="Late-window residual gap between corrected NVFP4 embed-only plus AdamW and the BF16 baseline, staying near plus 0.046 from steps 7808 through 9536." />
  <figcaption>Once the harmful knob is gone and the expert lane moves to AdamW, the remaining gap stays tight and remains unacceptable.</figcaption>
</figure>

That is a very different picture from the original NVFP4 failure. We are down to a stubborn residual that we can isolate and attack directly.

## Why the old story broke

The earlier dither work was still useful. It proved that RTN stiction exists and that keeping quantized codes moving matters. I am not walking that back.

What changed is the center of gravity. Once the gain split and optimizer lane were isolated cleanly, they explained much more of the BF16 gap than the original headline did. Dither and router sensitivity are still part of the story. On this recipe they live in the supporting mechanism layer.

That is a narrower claim, but it is a better paper.

## Current conclusion

The honest result is both more limited and more useful. `fp4_embed_gain` helps. `fp4_logits_gain=0.125` hurts. The historical pair is net harmful. `embed-only + adamw` gets this NVFP4 recipe very close to BF16, with an observed residual of about `+0.046`.

That gap is small enough to attack directly and still large enough to be unacceptable. If NVFP4 cannot track BF16 more closely than this on the same recipe, something is still wrong. We treat the remaining gap as a bug we need to remove. A permanent tax of this size would still be unacceptable.

## Receipts

`nmoe/repro/0005.receipts.json` points at the corrected `384`-step gain-isolation quartet and the corrected `9536`-step `embed-only + adamw` rerun. Those two surfaces carry the claims in this post.
