# HYDRA-vNext Training Specification

## Overview

Three-phase training that uses free labels to shape representations while keeping the oracle rubric as the canonical target.

```
Phase A: L18 Probe Pretrain (surface signals, no judge)
     ↓
Phase B: L24 Judge Train (rubric primary + FW pairwise aux)
     ↓
Phase C: Distill L24 → L18 (gate learns to approximate rubric)
```

## Container-first usage

All HYDRA training + grading runs are expected to happen inside the dataprep/train containers
with a writable `/data` mount.

Examples:
```bash
# Data prep / grading CLI
docker build -f docker/Dockerfile.dataprep -t nmoe-dataprep:local .
docker run --rm --gpus all -v /path/to/data:/data nmoe-dataprep:local python -m nmoe.data.cli --help
```

## Architecture

```
gpt-oss-20B (frozen)
       │
       ├─── Layer 18 ─── mean_pool ─── L18 PROBE
       │                                  │
       │                 ┌────────────────┼────────────────┐
       │                 ↓                ↓                ↓
       │            Surface Reg      Gate-like Cls    Meta Cls
       │            ├─ dclm           ├─ artifacts     └─ domain
       │            ├─ edu_approx     └─ missing
       │            ├─ code
       │            ├─ math
       │            └─ lang_score
       │                 │
       │                 └─ q_gate (scalar) ←── distill ──┐
       │                                                   │
       └─── Layer 24 ─── project ─── transformer ─── L24 JUDGE
                                                        │
                                                   rubric_5dim
                                                        │
                                                   q = agg(y)
```

## Phase A: L18 Probe Pretrain

**Purpose:** Learn strong surface representations and gate-like signals without touching the rubric.

**Steps:** 50k–200k updates (cheap, can run longer)

**Batch Mix per Step:**
- 70% EAI (code + stem + math + med, mixed by token volume)
- 30% FineWeb-Edu Score2

**Loss Weights (per-example normalized):**

| Head | Weight | Source |
|------|--------|--------|
| `fasttext.dclm` | 0.25 | EAI |
| `fasttext.fineweb_edu_approx` | 0.25 | EAI |
| `fasttext.eai_web_code` | 0.25 | EAI |
| `fasttext.eai_general_math` | 0.25 | EAI |
| `language_score` | 0.2 | FW-Edu / EAI |
| `domain` (CE) | 0.2 | task_id |
| `extraction_artifacts` (CE) | 0.5 | EAI taxonomy |
| `missing_content` (CE) | 0.5 | EAI taxonomy |

**Output:** Pretrained L18 probe with strong surface representations.

---

## Phase B: L24 Judge Train

**Purpose:** Learn the 5-dim rubric as the canonical target, with FW pairwise ranking as controlled perturbation.

**Oracle Budget:** 50k docs, stratified by domain (web, code, science, math)

**Max Context:** 4096 tokens

**Step Sampling:**
- 60% Oracle step: batch from oracle-labeled pool only
- 40% Oracle+FW step: batch from oracle pool + matched FW pair set

**Batch Sizes:**
- Oracle batch: B=32–128 docs (depending on packing)
- FW pairs per Oracle+FW step: P=32 pairs (64 docs)

**Loss:**
```python
def judge_loss(batch):
    y_hat = judge.rubric_head(h24)  # 5-dim
    q = agg(y_hat)

    # PRIMARY: Oracle rubric (present in 100% of steps)
    L_oracle = huber(y_hat, batch.oracle_5dim)

    # AUX: FW pairwise ranking (only on Oracle+FW steps)
    if batch.has_fw_pairs:
        q_a = agg(judge.rubric_head(h24_a))
        q_b = agg(judge.rubric_head(h24_b))
        # margin ranking: max(0, margin - (q_a - q_b))
        L_fw = margin_ranking_loss(q_a, q_b, target=1, margin=0.1)
    else:
        L_fw = 0

    return 1.0 * L_oracle + 0.1 * L_fw
```

**Key Guarantee:** Rubric present in 100% of steps. FW is controlled perturbation, never standalone.

---

## Phase C: Distill L24 → L18

**Purpose:** Train L18 gate to approximate L24 rubric scalar cheaply.

**Steps:** 100k+ updates

**Procedure:**
1. Freeze L24
2. Run L24 over huge unlabeled stream to produce `q_judge`
3. Train L18 gate head

**Batch Mix:**
- 80% distill stream (any corpora)
- 20% hard negatives (mined where `q_probe` disagrees with `q_judge` or near thresholds)

**Loss:**
```python
def distill_loss(h18, q_judge):
    q_probe = probe.gate_head(h18)

    # PRIMARY: Distillation
    L_distill = mse(q_probe, stopgrad(q_judge))

    # AUX: Keep Phase A heads active at low weight to prevent drift
    L_aux = 0.1 * phase_a_aux_loss(h18, batch)

    return L_distill + L_aux
```

---

## FineWeb-Edu Pairing Rules

**Source:** `fineweb_edu_score2` parquet (has `int_score`, `language_score`, `token_count`)

### Bucket Keys

| Bucket | Values |
|--------|--------|
| Domain | `web` only (don't mix with other domains) |
| Length | `[0,256)`, `[256,512)`, `[512,1k)`, `[1k,2k)`, `[2k,4k]` |
| Language | `language_score >= 0.9` only |

### Pair Construction

1. Sample two docs from same `(length_bucket, language_bucket)`
2. Require `int_score_a - int_score_b >= 2` (hard gap)
3. If insufficient pairs, relax to `>= 1` but cap relaxed pairs at ≤20%
4. Enforce uniform sampling across length buckets (prevent "shorter is better")

### Loss

```python
# q = agg(rubric_hat)
# margin ranking: max(0, margin - (q_a - q_b))
loss = margin_ranking_loss(q_a, q_b, target=1, margin=0.1)
```

---

## EAI Field Classification

### L18 Labels (use directly)

| Field | Type | Source |
|-------|------|--------|
| `quality_signals.fasttext.dclm` | regression | EAI |
| `quality_signals.fasttext.fineweb_edu_approx` | regression | EAI |
| `quality_signals.fasttext.eai_web_code` | regression | EAI |
| `quality_signals.fasttext.eai_general_math` | regression | EAI |
| `quality_signals.fasttext.english` | regression | EAI |
| `eai_taxonomy.extraction_artifacts` | classification | EAI |
| `eai_taxonomy.missing_content` | classification | EAI |
| `domain` | classification | task_id |

### Sampling Signals (audit-gated, NOT L24 losses)

| Field | Use For |
|-------|---------|
| `eai_taxonomy.reasoning_depth` | Oracle strata / audits |
| `eai_taxonomy.technical_correctness` | Oracle strata / audits |
| `eai_taxonomy.education_level` | Oracle strata / audits |
| `eai_taxonomy.bloom_*` | Oracle strata / audits |

**Rule:** These become L24 aux losses ONLY if 120B audits show strong alignment with rubric.

---

## Minimal Loader Changes

### 1. Raw-source readers must surface required scalars

**EAI parquet:**
```python
fields = [
    "text",
    "quality_signals.fasttext.dclm",
    "quality_signals.fasttext.fineweb_edu_approx",
    "quality_signals.fasttext.eai_web_code",
    "quality_signals.fasttext.eai_general_math",
    "quality_signals.fasttext.english",
    "eai_taxonomy.extraction_artifacts.primary.code",
    "eai_taxonomy.missing_content.primary.code",
]
```

**FineWeb-Edu Score2 parquet:**
```python
fields = ["text", "int_score", "language_score", "token_count"]
```

**DCLM/Dolma3 jsonl.zst:**
```python
fields = ["text", "url", "language_id_whole_page_fasttext"]  # if available
```

### 2. HYDRA training loader feature projection

```python
@dataclass
class HydraTrainingSample:
    text: str
    domain: str  # from task_id

    # L18 signals (optional, masked if missing)
    fasttext_dclm: float | None
    fasttext_edu: float | None
    fasttext_code: float | None
    fasttext_math: float | None
    fasttext_english: float | None
    lang_score: float | None
    extraction_artifacts: int | None  # class index
    missing_content: int | None  # class index

    # L24 signals (optional)
    oracle_5dim: tuple[float, ...] | None  # only for oracle-labeled
    fw_int_score: int | None  # only for FW pairs
    token_count: int | None  # for bucketing
```

**Rule:** Map missing fields → `None`, mask losses accordingly. Do NOT persist giant metadata dicts.

---

## Checkpoint Contract

**Exported `hydra_judge.pt`:**
```python
{
    "projector": state_dict,      # Linear(2880 → 512)
    "encoder": state_dict,        # 1-layer transformer
    "rubric_head": state_dict,    # Linear(512 → 5)
}
# Aux heads NOT exported
```

**Exported `hydra_probe.pt`:**
```python
{
    "gate_head": state_dict,      # Linear(2880 → 1)
    "domain_head": state_dict,    # Linear(2880 → K)
    # Optionally export surface heads for analysis
}
```

---

## Oracle Audit Plan (Pre-training)

Before trusting any free label as L24 aux, validate alignment:

| Signal | Sample Size | Audit Question |
|--------|-------------|----------------|
| FW-Edu `int_score` bins | 1k per bin | Does high score = high rubric? |
| EAI `reasoning_depth` | 1k per level | Does "deep" = high rubric? |
| EAI `technical_correctness` | 1k per level | Does "correct" = high rubric? |
| EAI `education_level` | 1k per level | Does higher = better rubric? |

**Output:** Correlation + calibration curves + disagreement examples → decide if signal belongs in L24 aux or stays sampling-only.

---

## Runtime Inference

```
doc → Tokenize → Backbone layers 0-18
                        ↓
                   L18 Probe → q_gate
                        │
                        ├── < τ_drop (0.25) → DROP (save 50% compute)
                        │
                        └── ≥ τ_drop → continue
                                          ↓
                                   Backbone layers 19-24
                                          ↓
                                   L24 Judge → rubric_5dim → q = agg(y)
                                          │
                        ┌─────────────────┼─────────────────┐
                        ↓                 ↓                 ↓
                   < τ_drop          τ_drop-τ_keep      ≥ τ_keep
                      DROP              BAND               KEEP
                                     (escalate?)
```

---

## Summary

| Phase | Primary Objective | Aux Signals | Data Source |
|-------|-------------------|-------------|-------------|
| A: L18 Pretrain | Surface representations | fasttext, lang, domain, artifacts | EAI + FW-Edu (millions) |
| B: L24 Train | Oracle rubric (5-dim) | FW pairwise ranking (0.1w) | Oracle (50k) + FW pairs |
| C: Distill | L18 gate ≈ L24 q | Phase A aux (0.1w) | L24 predictions (millions) |

**Canonical target:** Oracle 5-dim rubric. Everything else is regularization.
