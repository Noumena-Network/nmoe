---
title: "Why Training MoEs is So Hard"
subtitle: "Three failure modes that make frontier MoE training qualitatively different"
date: 2026-03-14
author: "xjdr"
number: "0000"
series: "research"
status: "framing"
---

Recently, I found myself wanting a small, research-focused training repo where I could do quick experiments. I wanted to try new attention architectures, play with mixed precision, swap optimizers, and run weird MoE ablations without dragging an entire production stack behind me.

I tried the three obvious contenders—NeMo, Megatron, and Torchtitan—but none of them fit what I wanted. They were heavy, awkward to stand up, and too broad for the kind of fast research loop I had in mind. Reusing my production stack was no better. That code was built for monitoring, stability, and large infrastructure; folding this work into it would have made both systems worse.

That left me with a simple question: why is training frontier-quality “smallish” MoEs—say, under `20B` total parameters—so difficult, and why doesn't the repo I want already exist?

After enough false starts, my answer reduced to three pressures:

1. FLOPs and FLOP efficiency
2. load balancing and router stability
3. data quality and quantity

<figure>
  <img src="/figures/nmoe_timeline.svg" alt="Timeline of the project: proto MoE experiments, n2, and the nmoe release." />
  <figcaption>Timeline (high level): we started with research prototypes, learned what kept breaking, rebuilt a production-grade stack, then began speedruns and one-dial science loops.</figcaption>
</figure>

One terminology note up front: this story spans the unreleased proto repos (`nanomoe -> n2`) and the released `nmoe` repo (`January 2026`). I tag anecdotes accordingly so I do not accidentally backfill later understanding into earlier decisions.

## Evidence Scope

This post is the preamble for the series. Later posts carry the stronger receipt bundles, the public repro work, and the narrower claims. The proto telemetry here is historical context for failure modes that actually happened in `nanomoe -> n2`, not a substitute for the current `nmoe` proof surfaces. Whenever a figure is only an upper bound or belongs to an earlier measurement regime, I say so directly in the surrounding text or caption.

## What I Mean By “Hard”

Dense training is weirdly forgiving. The dynamics are mostly coupled, and if you have enough parameters, the model will often learn despite your mistakes. This has bitten me in the ass more than once.

MoE training is less forgiving because it introduces **partial activation** and a **learned routing policy**. That creates new attractors: loss can keep going down while the system quietly drifts into behavior you did not mean to train.

So when I say “hard,” I mean things like:

- you have a lot of GPUs and still get terrible utilization
- routing collapses into a hot subnetwork and stays there
- reduced precision does more than hurt accuracy; it changes the optimization regime
- once training is stable, dirty data becomes the limiter and confounds your conclusions

This series is not trying to hand over the one true recipe. It is trying to name the dragons, instrument them, and then turn one dial at a time with plots.

Here is the minimal mental model that makes MoE dynamics feel different from dense training:

<figure>
  <img src="/figures/moe_grad_flow.svg" alt="MoE gradient flow: router selects active experts; inactive experts receive no gradient." />
  <figcaption>In MoE, only selected experts receive gradient signal. That single fact is where most of the weirdness comes from.</figcaption>
</figure>

## FLOPs

[DeepSeek-style ultra-sparse MoEs](https://arxiv.org/abs/2412.19437v2) change the compute economics because the training dynamics are decoupled. Only part of the MLP stack is active for a given token, and the active experts change over time as routing evolves.

That decoupling is also why larger MoEs can look so attractive in the first place. You get large inference-efficiency wins and smaller training-efficiency wins, but you pay for them in two ways. The dynamics become harder to predict, and you need to spend many more FLOPs making sure the routing policy learns something useful while the experts it touches get enough signal.

### The stranded FLOPs problem

Ultra-sparse MoEs occupy a lot of HBM because the experts have to be resident. That means more GPUs. More GPUs, in turn, often means more idle GPUs.

FSDP and related dense-era sharding topologies did not do a good job of converting that footprint into useful expert compute in the research-scale regimes we were testing. Utilization could get ugly very quickly.

The underlying reason is boring: batching. If you have `E` routed experts and route `K` per token, then the per-expert batch is roughly

$$
\mu \;\approx\; \frac{T \cdot K}{E}
$$

where `T` is tokens per replica per step. If you crank `E` without paying for more tokens, you fragment the expert batches. Fragmented batches do not saturate GEMMs.

<figure>
  <img src="/figures/mfu_vs_batch_per_expert.svg" alt="Expert GEMM MFU rises sharply as batch per expert increases." />
  <figcaption>Expert GEMM MFU vs batch/expert (B200 BF16, single-GEMM upper bound). This is why ultra-sparse MoEs can look slow at research scale.</figcaption>
</figure>

Treat that panel as an upper-bound view of expert-GEMM throughput. The end-to-end transport story comes later.

That pressure pushed me toward two ideas: a different expert-parallel topology that keeps GPUs busy, and lower expert precision that at least promised large HBM savings on paper (`FP8` and `NVFP4`).

One representative forward breakdown from our earlier RDEP experiments makes the topology problem concrete:

<figure>
  <img src="/figures/rdep_forward_breakdown.svg" alt="Forward time breakdown where all-to-all of activations dominates at small batch." />
  <figcaption>Historical RDEP forward breakdown from the proto era. At small batch, dispatch (A2A of activations) dominated forward time. This is the regime where more GPUs can mean more overhead.</figcaption>
</figure>

Later transport work lives in `0009`. The point of including this panel here is simpler: topology became its own problem very early.

## Load Balancing / Router Stability

I will leave topology to its dedicated write-up. The other early obsession was mixed precision.

In theory, mixed precision is a gift. In practice, it can still cost more training memory in stacks that keep master weights and gradients in higher precision, then quantize down to a lower-precision representation for the next forward pass. So you buy inference efficiency, which matters a lot, while making training dynamics more fragile.

Every touch to precision also perturbs the rest of optimization. For MoE, one of the first places we saw that perturbation was router stability.

### The DeepSeek approach, and the part we could not borrow

The DeepSeek-V3 tech report describes an elegant aux-loss-free setup with very few knobs and clear intended dynamics. These are tools built for experienced users. One lesson I took from that report is that very large batch sizes appear to be part of the router-stability story in that regime. That is a luxury we do not have when we are doing research on limited hardware.

So we had to work much harder to make small runs stable, efficient, and informative.

### The router learning problem

As I started replicating that setup for mixed-precision experts, the proto-era failure signal kept looking like router-side learning signal that was simply too small for `FP8` or `NVFP4`. The routers would stop learning, and then the experts would starve.

I tried almost everything. Reduced-precision backwards passes. `FP32` master weights and grads. Different optimizer settings. The collapse kept showing up.

A well-timed paper was the [Character.AI Kaiju write-up](https://blog.character.ai/inside-kaiju-building-conversational-models-at-scale/) describing a collection of `INT8` stability interventions. I tried them all, then tried them one at a time.

### The ugly fix that worked

Everything in this subsection is proto-era (`nanomoe -> n2`), before we rebuilt and released `nmoe`. I am not presenting a final recipe here. I am naming the failure mode we kept hitting.

The first fix was ugly and useful. Our working read was that the router gradients were living below the quantization floor. So the interventions that actually moved the needle were mostly gradient-scale hacks:

- rescale embeddings and logits (μP-ish)
- remove gradient clipping as a forcing function
- add a single scalar on the expert output as a temporary crutch

The point was simple: if the router does not receive usable gradient signal, the rest of the system cannot rescue you.

Those wins came with exploding `BF16` grad norms, and the clipping settings we had been using in that regime were suppressing the very router updates we needed. So the practical fix there was to disable global clipping and recenter optimization around router learning signal.

In that proto setup, those changes were enough to get stable mixed-precision router behavior.

### The bungee scalar

Another Kaiju-derived intervention that proved useful in the proto era was a single virtual scalar at the output of the experts. I thought of it as a bungee cord: something that yanked gradients back into a learnable regime when precision changes would otherwise make them disappear.

It felt hacky. It also worked better than letting the router sit below the quantization floor.

Later `nmoe` work replaced this sort of global scalar with a narrower story: the old NVFP4 rescue package was bundling one helpful compensator with one harmful one, and the remaining gap only became small once that split was made explicit. That lands in `0005`.

Practically, that proto recipe meant rescaling gradient flow, removing global clipping, adding a temporary expert-output scalar, and keeping aux-loss-free token-choice routing.

### What stable looked like in that regime

Here is historical router telemetry from a successful proto run (`NVFP4`, step `20k`, `E=64`, `K=6`). This was the shape I cared about in that regime: high entropy, modest CV, and no dead experts.

<figure>
  <img src="/figures/proto_router_health_overlay.svg" alt="Router CV and entropy by layer in a stable proto NVFP4 run (E=64, K=6)." />
  <figcaption>Proto-era NVFP4 router health snapshot. CV is shown as percent (100×CV). Entropy is normalized by log(E), so 1.0 means uniform routing. Historical context only; not a receipt-backed public benchmark.</figcaption>
</figure>

| Layer | CV% | Entropy (norm) | Active | Bias Range |
|-------|------|---------|--------|------------|
| 1 | 20.5 | 0.994 | 64/64 | [−0.57, +0.42] |
| 2 | 28.0 | 0.987 | 64/64 | [−0.52, +0.48] |
| 3 | 36.7 | 0.985 | 64/64 | [−0.52, +0.48] |
| 4 | 11.4 | 0.998 | 64/64 | [−0.47, +0.53] |
| 5 | 17.1 | 0.996 | 64/64 | [−0.41, +0.59] |
| 6 | 9.2 | 0.999 | 64/64 | [−0.40, +0.59] |
| 7 | 8.7 | 0.999 | 64/64 | [−0.38, +0.61] |
| 8 | 12.3 | 0.998 | 64/64 | [−0.42, +0.57] |
| 9 | 7.3 | 0.999 | 64/64 | [−0.27, +0.73] |
| 10 | 8.1 | 0.999 | 64/64 | [−0.31, +0.66] |
| 11 | 6.5 | 1.000 | 64/64 | [−0.37, +0.61] |

That was the bar for the proto regime: the router explores, and every expert gets gradient signal.

## Data

Once the system is stable enough to be meaningfully wrong, data becomes the limiter.

I am keeping this short on purpose. The data pipeline deserves its own post with falsifiable claims and artifacts, and this preamble does not make a standalone quantitative data-mixture claim.

For now, the point is simple: stability only gets you to the starting line. After that, the data has to deserve the run.

## Where We Are Now

The proto repos (`nanomoe -> n2`) did what prototypes are supposed to do: they taught us what breaks.

They also got trashed in the process. “NO SLOP IN THIS HOUSE” is a cute slogan until you have been up all night chasing a `NaN` with nineteen half-finished experiments living in your training loop. As Vik said: live by the slop, die by the slop.

So we distilled what mattered and rebuilt the system with a much smaller surface. That became `nmoe`.

## What’s Next

The rest of the series is the part I actually care about: the speedruns, the one-dial scaling loops, and the weird `MoE`/`FP4` dynamics that only show up once you have a real measurement loop.

The story stays roughly chronological. Each post has one payload, and anything discovered later is explicitly deferred. The standing rule is one dial at a time, with plots.

| Post | Payload |
|------|---------|
| `0001` | what we built: the `nmoe` training system (container-first, TOML, determinism, metrics, NVIZ) |
| `0002` | make it measurable: evaluation as the scientific loop (`bpb`, `CORE`, router health) |
| `0003` | the speedrun loop: public anchoring, fast dense-vs-MoE comparisons, and the first honest swap contract |
| `0004` | `#420`, MoE edition: token-indexed miniseries curves and fairness under sparsity |
| `0005` | `NVFP4` dynamics: one helpful gain, one harmful gain, and the remaining `~+0.046` bug |
| `0006` | extreme sparsity: corrected-stack `Super-4096`, clean falsifiers, and the limits of dashboard telemetry |
| `0007` | the atlas hypothesis: the object `0006` could not name, and what a real receipt would have to measure |
| `0008` | expert learning rate: why Moonlet's old `15x` expert-LR rule overshoots in `bf16` `AdamW` |
| `0009` | `RDEP`: the transport system that keeps sparse expert compute hot across NVLink fabrics |
| `0010` | architecture primitives: reproducing Canon, `mHC`, and Engram with an eval loop that can tell them apart |
| `0011` | let the speedrun search itself: bounded config-only autoresearch on a real public lane |

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437v2)
- [Inside Kaiju: Building Conversational Models at Scale](https://blog.character.ai/inside-kaiju-building-conversational-models-at-scale/)
- [Moonlight: Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [Seed-Coder: Let the Code Model Curate Data for Itself](https://arxiv.org/abs/2506.03524)
- [OLMo-3: The Best Fully Open Model of its Class](https://allenai.org/blog/olmo3)
