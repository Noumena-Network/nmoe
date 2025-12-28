# AI Agent Workflow

This document describes the contract-first workflow for AI agents (Claude Code, Codex, etc.) working on nmoe.

## Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Create    │     │   Create    │     │   Agent     │     │   Create    │     │   Merge     │
│   Issue     │ ──▶ │  Worktree   │ ──▶ │  Iterates   │ ──▶ │    PR       │ ──▶ │   Down      │
│ (Contract)  │     │  + Branch   │     │ Until Pass  │     │(Attestation)│     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## Coordinator checklist (no extra context)

Use this when acting as the coordinator/merge captain (human or AI).

1. Create one GitHub Issue per PR using `work_item.yml` (the contract is the source of truth).
2. Ensure the contract includes:
   - strict `scope.allow` / `scope.deny`
   - required `gates` (IDs from `configs/gates/gates.toml`)
   - explicit perf budget (if hot path)
   - `depends_on` and `merge_hotspots` (for merge sequencing)
3. Create a dedicated worktree + branch from `origin/master` (one worktree per issue) and assign an agent.
4. Agent iterates until Tier A passes and commits are clean + issue-scoped.
5. Agent opens a PR with the attestation block filled (Tier A evidence).
6. Maintainer triggers Tier B on the PR merge ref (`refs/pull/<PR>/merge`) with the appropriate bundle.
7. Merge only after Tier B reports pass (and sequencing constraints are satisfied).

## 1. Create Issue (Contract)

Issues are **agent contracts** - structured specs that agents follow without back-and-forth.

### Using the template

Create issue at: https://github.com/Noumena-Network/nmoe/issues/new?template=work_item.yml

### Contract structure

```yaml
nmoe:
  issue: 03a
  slug: attention
  kind: feature          # feature|bugfix|perf|refactor|docs
  risk: hot              # hot|warm|cold
  complexity: M          # S|M|L
  depends_on: [01]       # issue numbers this blocks on
  merge_hotspots: [nmoe/train.py, nmoe/config.py]

scope:
  allow:                 # files agent MAY modify
    - nmoe/attention/**
    - nmoe/model.py
  deny:                  # files agent MUST NOT touch
    - nmoe/csrc/**

contract:
  invariants:
    - B200-only (sm_100a); fail off-target
    - No NCCL all-to-all on MoE path
    - TOML-only config; container-first
  knobs:
    add: []
    change: [attn_global_every, attn_local_window]

gates:
  required:              # gate IDs from configs/gates/gates.toml
    - cpu:import
    - cpu:compileall
    - b200:moonlight_8x20
  perf:
    gate: perf:baseline_delta
    metric: node_tps_p50
    baseline_ref: origin/master
    budget_pct: -10
```

### Key properties

- **Deterministic**: Agent doesn't invent commands - uses gate IDs
- **Scoped**: Explicit allow/deny file lists; no drive-bys
- **Merge-aware**: Declares expected conflict hotspots
- **Public-safe**: No internal paths/hostnames

## 2. Create Worktree + Branch

```bash
# Create isolated worktree for the issue
ISSUE=03a
SLUG=attention
git worktree add ../nmoe-worktrees/issue-${ISSUE}-${SLUG} -b issue/${ISSUE}-${SLUG} origin/master

# Agent works in the worktree
cd ../nmoe-worktrees/issue-${ISSUE}-${SLUG}
```

## 3. Agent Iterates Until Gates Pass

### Agent reads the contract

The agent:
1. Reads the issue contract
2. Understands scope (allow/deny lists)
3. Knows which gates must pass
4. Works within invariants

### Running Tier A gates locally

```bash
# Tier A: CPU/container gates (agent runs these)
python -c "import nmoe"                    # cpu:import
python -m compileall -q nmoe/              # cpu:compileall
```

### Tier B gates (cluster)

Tier B gates require B200 GPUs and are run by the maintainer after PR creation.

Gate definitions are in `configs/gates/gates.toml`.
Tier B gates fail fast if required environment variables (listed as `requires` in the gate registry) are missing.

## 4. Create PR (Attestation)

PRs carry a structured **attestation block** documenting what was done.

### PR template

```yaml
nmoe_pr:
  issue: 03a
  baseline_sha: 99fae30
  pr_sha: e79e342
  risk: hot

gates_passed:
  tier_a:
    - cpu:import
    - cpu:compileall
  tier_b: []              # filled after maintainer runs workflow

perf:
  gate: perf:baseline_delta
  metric: node_tps_p50
  baseline: <number>
  result: <number>
  delta_pct: <number>
  budget_pct: -10
  status: pass|fail
```

### Creating the PR

```bash
# Push branch
git push origin issue/03a-attention

# Create PR via gh CLI
gh pr create \
  --base master \
  --head issue/03a-attention \
  --title "[03a] Global/local attention stack" \
  --body "$(cat <<'EOF'
## PR Attestation
... (fill in template)
EOF
)"
```

## 5. Tier B Gates (Maintainer-Controlled)

### How it works

1. Maintainer triggers workflow via GitHub Actions
2. Workflow uses `environment: b200` (requires approval)
3. ARC spins up runner pod on B200 node (scale 0→1)
4. Gates run against `refs/pull/<PR>/merge` (integration test)
5. Results posted as PR comment
6. Runner scales back to 0

### Triggering gates

```
GitHub → Actions → "Tier B gates" → Run workflow
  - pr_number: <PR number>
  - bundle: hot_path_min | blockscaled_full | frozen_full | full | single
  - single_gate: (if bundle=single) e.g., b200:moonlight_8x20
```

### Gate bundles

| Bundle | Gates |
|--------|-------|
| `hot_path_min` | moonlet_1x10, moonlight_8x20 |
| `blockscaled_full` | hot_path_min + profile_fp8, profile_nvfp4 |
| `frozen_full` | moonlight_frozen_8x20, frozen_profile_fp8, frozen_profile_nvfp4 |
| `zero2_full` | hot_path_min + resume_determinism |
| `full` | all gates |

## 6. Merge Down

### Merge order matters

Respect `depends_on` in issue contracts:
```
01 → 03a → 03b → 08    (core infrastructure chain)
02, 05, 06, 07, 09     (independent features)
11, 12, 13             (post-training)
10                     (final validation - LAST)
```

### Conflict resolution

- Check `merge_hotspots` declared in contracts
- Mechanical resolution first, refactor later
- Owner for key files:
  - `train.py` - trainer semantics
  - `config.py` - public knobs
  - `csrc/*` - ABI/kernels

### Merge command

```bash
git checkout master
git merge --no-ff issue/03a-attention -m "Merge #03a: Global/local attention stack"
```

## Gate Reference

Gates are defined in `configs/gates/gates.toml`.

### Tier A (CPU)

| Gate ID | Command |
|---------|---------|
| `cpu:import` | `python -c "import nmoe"` |
| `cpu:compileall` | `python -m compileall -q nmoe/` |

### Tier B (B200)

| Gate ID | GPUs | Config |
|---------|------|--------|
| `b200:moonlet_1x10` | 1 | moonlet, 10 steps |
| `b200:moonlight_8x20` | 8 | moonlight, 20 steps |
| `b200:moonlight_frozen_8x20` | 8 | moonlight_frozen, 20 steps |
| `b200:profile_fp8` | 8 | moonlight, fp8, 10 steps |
| `b200:profile_nvfp4` | 8 | moonlight, nvfp4, 10 steps |
| `b200:frozen_profile_fp8` | 8 | moonlight_frozen, fp8, 10 steps |
| `b200:frozen_profile_nvfp4` | 8 | moonlight_frozen, nvfp4, 10 steps |
| `zero2:resume_determinism` | 8 | save→resume test |
| `perf:baseline_delta` | - | compare vs baseline |

## Infrastructure

### ARC (Actions Runner Controller)

- **Controller**: `arc-systems` namespace (example; cluster-defined)
- **Runner scale set**: `arc-runners` namespace, name `b200` (example; cluster-defined)
- **Scale**: 0→1 (scale-to-zero when idle)
- **Image**: `ghcr.io/noumena-network/nmoe:runner-latest`

### GitHub Environment

- **Name**: `b200`
- **Protection**: Required reviewer (maintainer)
- **Purpose**: Gate Tier B workflow triggers

### Updating runner image

The runner image is built automatically on push to `master` when the runner/training Dockerfiles change.
It intentionally does **not** rebuild on every code change: Tier‑B gates always run against the PR merge ref
(`refs/pull/<PR>/merge`) and compile the CUDA extension from that checkout.

Manual build:
```bash
docker build -f docker/Dockerfile.runner -t ghcr.io/noumena-network/nmoe:runner-$(git rev-parse --short HEAD) .
docker push ghcr.io/noumena-network/nmoe:runner-$(git rev-parse --short HEAD)
```

## Public Safety

All workflow artifacts must be public-safe:

- ✅ Use env vars for paths (NMOE_DATA_PATH, NMOE_RUN_ROOT)
- ✅ Use gate IDs, not ad-hoc commands
- ✅ Generic PVC/resource names
- ❌ No internal hostnames/IPs
- ❌ No credentials in contracts/attestations
- ❌ No cluster identifiers
