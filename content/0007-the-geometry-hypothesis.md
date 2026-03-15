---
title: "The Atlas Hypothesis"
subtitle: "why output-only dashboards cannot name what pretraining built, and what a real receipt would have to measure"
date: 2026-03-14
author: "xjdr"
number: "0007"
series: "research"
status: "hypothesis"
draft: false
receipts:
  - nmoe/repro/0007.receipts.json
---

> Post 0007: the atlas program in plain language, as a bridge from `0006` into the canonical paper surface.

`0006` left me with a harder question than “did the router collapse?” Loss went down, the run stayed numerically stable, the dashboard could convict the run, and I still could not say what object pretraining had actually built or what object the collapse had damaged.

The sharpest claim in the atlas paper is not just that geometry matters. It is that output-only monitoring is too weak near the decision boundary to tell preservation from rewrite. If that is right, then “the router went bad” is not an explanation. It is a symptom report.

My claim is that large-MoE pretraining builds an atlas over routed state space. If that is right, then the question after post-training is not only “did the model still answer?” It is “what part of the atlas was preserved, bent, or rewritten?”

This post accompanies the public whitepaper surface. The formal proofs live in [**Atlas Foundations**](/papers/atlas-foundations.pdf). The job of this essay is to say what object that paper is trying to name, why `0006` forces that object, which parts are already theorem-grade, and what a real receipt would need to measure if someone later claims “post-training preserved the atlas.”

## Why `0006` forced a better object than “the router went bad”

At least three different failure stories tend to get collapsed into one blob:

| Failure story | What it actually means |
|:--|:--|
| boundary sensitivity | the router is near a top-`k` boundary, so small drift causes frequent set changes |
| local incompatibility | adjacent experts are both active, but swapping between them is expensive |
| output damage | behavior degrades even while sparse-routing telemetry still looks superficially fine |

Those are not the same thing. The old 3-state language helped because it forced me to separate boundary sensitivity, local compatibility, and output damage. It was still a symptom dashboard, though. It was not the object itself.

The atlas view is the attempt to name that object directly. That is why the paper starts with observability and evidence contracts rather than with vibes about geometry: if the object is wrong, no dashboard can tell preservation from rewrite.

## Start with RMS normalization

The first geometric fact here is not mystical. It is just what RMS normalization does to motion.

Start with

$$
Z_\epsilon(x) = \frac{x}{\sqrt{\|x\|_2^2 / d + \epsilon}}.
$$

At `\epsilon = 0`, this lands exactly on the sphere of radius `\sqrt{d}`. At finite `\epsilon`, the exact image is not one sphere; it is a ray-preserving radial image. Points on the same residual ray keep the same direction after normalization, but not the same radius.

That immediately gives two different objects:

| Object | Why it matters |
|:--|:--|
| shell-valued normalized states | the implementation-correct object for exact gate magnitudes and exact visible step sizes |
| angular quotient `\hat z = z / \|z\|_2` | the theorem-grade manifold object for the atlas statements |

The practical picture is simple. Radial motion gets heavily discounted by renormalization, while directional motion survives. For an expert update `u_e(z)`, the first-order visible tangent field is

$$
V_e(z) = P_z u_e(z),
\qquad
P_z = I - \frac{z z^\top}{d}.
$$

At `\epsilon = 0`, this is the exact tangent projector. At finite `\epsilon`, the projector interpretation becomes approximate because of an additional shared scalar factor.

The useful takeaway is that routed computation is best read in angular coordinates, while shell-valued states still matter whenever exact amplitudes matter.

## Pretraining builds more than experts

Once the geometry is fixed, the next question is what pretraining organizes on it.

For one routed layer `\ell`, the MoE block has three coupled parts:

| Part | Role |
|:--|:--|
| global coordinate / transport system | moves token states through contextual space |
| chart-local expert update fields | defines the local expert actions on that space |
| router-induced transition map | decides which charts are active, which are adjacent, and where hard boundaries sit |

This is why “the experts contain the knowledge” is too crude. In a large MoE, the router is not plumbing. It is the load-bearing interface that decides which local charts are co-active, which swaps are even possible, and which local substitutions count as semantically nearby.

So the atlas claim is easy to say, even if it takes work to make precise: pretraining learns a structured object on routed state space.

## Overlap is the rule

For expert `e`, define its routed domain

$$
U_{e,\ell} = \{ z : e \in R_\ell(\Gamma_\ell z) \},
$$

where `R_\ell` is the exact routed top-`k` set.

If `k > 1`, these expert domains do not form a partition. They overlap. At a generic regular point, exactly `k` experts are simultaneously active, so the same state belongs to `k` expert domains at once.

What does partition the regular routed region is the active-set stratification:

$$
C_{S,\ell} = \{ z : R_\ell(\Gamma_\ell z) = S \},
\qquad |S| = k.
$$

That yields three different geometric objects:

| Object | Where it matters |
|:--|:--|
| co-active expert cover `\{U_{e,\ell}\}` | local overlap structure |
| active-set cells `\{C_{S,\ell}\}` | exact routed top-`k` membership |
| swap boundaries | where one expert exits and another enters |

That separation is load-bearing. Co-active overlap and swap boundaries are different objects. The first is where local chart compatibility lives. The second is where tearing lives.

## What is already solid

This is the part the paper surface actually proves, not merely motivates.

Under two explicit assumptions — each expert down-projection `W_1^{e,\ell}` has full rank `d`, and exact top-`k` membership descends cleanly to the angular manifold — the layer-local routed structure gives you a real middle ground between metaphor and overclaim.

On the angular regular routed region, each expert induces a smooth immersion into expert-coordinate space. On co-active overlaps, the coordinate relation between two expert-coordinate representations of the same point is

$$
T_{e \to e',\ell} = W_1^{e',\ell}(W_1^{e,\ell})^+.
$$

That relation is linear in expert-coordinate space. Because the expert maps are immersions, the constant-rank theorem gives local intrinsic charts over that regular routed locus. The ordering matters. Weights determine the overlapping immersion cover together with its linear extrinsic overlap relations, and those in turn induce the local intrinsic coordinates.

Independently of the immersion cover, the router induces a disjoint stratification by exact top-`k` set. Hard-routing swap boundaries belong to that stratification. They are deliberately non-classical.

So the accurate statement is this: large-MoE layers determine a canonical overlapping immersion cover with linear extrinsic overlap relations, and that cover in turn yields local intrinsic charts over the angular regular routed set. That is a real theorem. It is stronger than metaphor and weaker than “all routing behavior is smooth.”

## Why adjacency and incompatibility matter

Two local results matter immediately because they connect the geometry to failures you can actually measure.

First, if exactly one expert swap occurs across a swap boundary, the comparison is taken at the same evaluation state, and the one-swap theorem's idealizations hold, then the local discontinuity in the composite routed field is

$$
\Delta V_\ell(z)
=
\rho_b(z)\bigl(v_{e_{k+1},\ell}(z) - v_{e_k,\ell}(z)\bigr).
$$

That is the explicit one-swap theorem. It does not cover multiple simultaneous swaps, higher-order renormalization effects, or broader nonlocal changes under adaptation; those cases pick up extra terms.

Second, under adaptation `\theta_0 \to \theta_1`, the composite field obeys an upper bound with one router-drift term and one content-drift term:

$$
\|\Delta V_\ell(z_0)\|_2
\le
\sum_e |\Delta \rho_{e,\ell}(z_0)|\,\|v^{(\theta_0)}_{e,\ell}(z_0)\|_2
+
\sum_e \rho^{(\theta_1)}_{e,\ell}(z_0)\,\|\Delta v_{e,\ell}(z_0)\|_2.
$$

That is a valid upper bound. It does not yet prove that the bound is tight enough, or already the right control law in practice.

The third local point is conceptual. Local substitutability starts with router adjacency. Parameter-space or output-space similarity can still be interesting, but it does not tell you much about substitution if the router never places those experts in local competition.

## Where the hypothesis begins

I do not want the theorem-grade part to leak into claims it has not earned.

The following still live on the hypothesis or empirical side:

| Still open | Why it matters |
|:--|:--|
| tangent-dominated training regime | whether the angular approximation carries most of the practical load |
| overlap-map condition number as a predictor | whether it really forecasts transition incompatibility |
| collapse as chart degeneracy | whether that phrasing is causally complete |
| perturbation budget as a control law | whether the upper bound is practically tight |
| chart-preserving post-training by default | whether ordinary SFT / RL / continual learning really behaves this way |

The right stance is not “there is no theorem here.” The right stance is that some structure is theorem-grade, while the stronger causal and control claims still have to survive hostile empirical testing.

## Occupancy is how you read the atlas in practice

For a fixed probe family `\mathcal{P}`, define the cell occupancy measure

$$
\mu_{\mathcal{P},\ell}(S)
=
\Pr_{z \sim \mathcal{P}}[R_\ell(\Gamma_\ell z)=S]
$$

and the expert marginal occupancy

$$
\pi_{\mathcal{P},\ell}(e)
=
\Pr_{z \sim \mathcal{P}}[e \in R_\ell(\Gamma_\ell z)].
$$

These are usage properties of the atlas under a data distribution or training run. They are not the atlas object itself.

That distinction matters because the atlas can remain formally well-defined while visited occupancy collapses onto a degraded subset of cells and experts. This is why low-dimensional telemetry like `mean_cv`, `min_entropy`, `dead_experts`, and `experts_active_mean` still matters. Those are occupancy proxies. They do not replace the atlas object, but they do tell you whether the visited region is collapsing onto a narrow support.

Two cautions follow. Rising margins are not health certificates under occupancy degeneration, because the surviving competitions can widen while effective atlas coverage gets worse. And scalar objectives can improve without geometric recovery. A run can lower loss on a narrowed visited region while overlap compatibility remains poor and occupancy remains degraded.

That failure pattern is one of the main reasons this ontology exists at all: scalar improvement without geometric recovery is one of the things this program is trying to explain.

## What chart-preserving post-training means

The normative claim is simple. Ordinary post-training should usually be chart-preserving.

In plain language, policy refinement, style control, calibration, and bounded local behavior changes should normally happen without broadly repurposing the knowledge-bearing atlas in protected layers. If an objective can only be solved by broad chart redefinition, then it is probably misclassified. It should be treated as continued pretraining, chart expansion, or outright knowledge acquisition.

That gives two categories:

| Category | Meaning |
|:--|:--|
| chart-preserving objective | a policy family reaches the target success level without rewriting the protected atlas |
| knowledge-expansive objective | no chart-preserving policy suffices, but a broader chart-expanding or continued-pretraining policy does |

The important point is that this is a classification problem.

## What evidence would actually count

This is where the observability program becomes operational. If someone wants to claim that a tuning run preserved the atlas, the receipt cannot be a vibes-based dashboard. The minimum measurement contract has to fix a probe family, a protected layer set, and the exact extraction objects: canonical pre-gain operational states `z`, exact post-gain gate coordinates `g`, exact fp32 dispatch scores, exact routed top-`k` sets, the `k`-th and `(k+1)`-th competitors, the set margin

$$
M_{\mathrm{set}} = s_{(k)} - s_{(k+1)},
$$

computed post-mask and pre-softmax, together with tangent-visible expert updates on a pre-registered evaluation set.

It also has to report the canonical drift channels — coordinate drift, boundary drift, transition drift, and chart-content drift — alongside occupancy diagnostics, output-damage canaries, and confidence intervals bootstrapped over windows.

Then the pass/fail layer has to make at least three predicates explicit: protected-atlas rewrite, base-domain damage, and knowledge-expansive classification. And any intervention claim has to be judged at matched task success. You do not get credit for “preserving the atlas” by simply failing the task.

## Where the old 3-state language still helps

The older 3-state dashboard still helps as operator intuition. A reasonable informal mapping is state 1 to boundary sensitivity and set-margin behavior, state 2 to co-active overlap compatibility, and state 3 to output-level damage.

That language is still useful. It is just not the canonical public object anymore. State 1 alone does not certify healthy routing, overlap compatibility and swap-boundary behavior are different objects, and output damage sits downstream of the geometry rather than replacing it.

## What would falsify this program

The burden is pretty clean. The program should lose credibility if normalized-coordinate diagnostics are beaten by raw-norm diagnostics, if co-active overlap and swap-boundary diagnostics add no value beyond generic sparse-routing telemetry, if undifferentiated parameter distance predicts forgetting as well as atlas-semantic drift, if chart-preserving policies do not reduce damage at matched task success, or if the tasks that seem knowledge-expansive turn out not to require broader chart redefinition after all.

That is the standard I want this post read against. If those predictions fail, the ontology should be weakened or rejected.

## Why later results need a higher bar

Matched-success intervention receipts are still follow-on work. What is finished here is the foundation and the verification contract. Later results either clear that bar or they do not.

## Further reading

The canonical public paper surface for this program is:

- [**Atlas Foundations**](/papers/atlas-foundations.pdf)
