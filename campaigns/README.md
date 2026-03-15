# Campaigns

Phase 1 codifies bounded autoresearch for `community/nmoe` without adding a
second trainer or a second orchestration stack.

Design rules:
- TOML is authoritative.
- Campaigns call canonical `nmoe` runners only.
- Runtime mutation is bounded to an allowlist of config overrides.
- Every run writes a machine-readable receipt.
- Budget stages are explicit and enforceable.

This is the intended synthesis of:
- `simply`: explicit agent-facing research playbooks and named research surfaces
- `autoresearch`: fixed-budget proposal -> run -> score -> decide loops

## Commands

List available campaigns:

```bash
cd /workspace/nmoe
python -m nmoe.cli.main campaign list
```

Inspect a campaign:

```bash
cd /workspace/nmoe
python -m nmoe.cli.main campaign show speedrun_super_smoke
python -m nmoe.cli.main campaign show speedrun_super_benchmark
python -m nmoe.cli.main campaign show speedrun_super_research
```

Run the canonical smoke loop:

```bash
cd /workspace/nmoe
python -m nmoe.cli.main campaign run speedrun_super_smoke --stage smoke
```

Run a bounded candidate:

```bash
cd /workspace/nmoe
python -m nmoe.cli.main campaign run speedrun_super_smoke \
  --stage smoke \
  --candidate relu-squared \
  --set activation=relu_squared \
  --set aux_loss_alpha=0.002
```

Dry-run the resolved command and receipt payload:

```bash
cd /workspace/nmoe
python -m nmoe.cli.main campaign run speedrun_super_smoke --dry-run
```

Run a real bounded benchmark on the full canonical speedrun dataset:

```bash
cd /workspace/nmoe
python -m nmoe.cli.main campaign run speedrun_super_benchmark \
  --stage benchmark \
  --candidate baseline
```

Run a first autoresearch candidate against that same benchmark stage:

```bash
cd /workspace/nmoe
python -m nmoe.cli.main campaign run speedrun_super_benchmark \
  --stage benchmark \
  --candidate lossfree-router \
  --set router_bias_update_rate=1e-4 \
  --set aux_loss_alpha=0.0
```

Run the autonomous config-only loop:

```bash
cd /workspace/nmoe
python -m nmoe.cli.main campaign auto speedrun_super_benchmark --stage benchmark
```

Run the eval-backed autonomous research loop:

```bash
cd /workspace/nmoe
python -m nmoe.cli.main campaign auto speedrun_super_research --stage benchmark
```

Preview the next autonomous proposal without running it:

```bash
cd /workspace/nmoe
python -m nmoe.cli.main campaign auto speedrun_super_benchmark --stage benchmark --dry-run
python -m nmoe.cli.main campaign auto speedrun_super_research --stage benchmark --dry-run
```

## Receipts

Receipts default to:

```text
repro/campaign_runs/<campaign>/<stage>/<timestamp>_<candidate>.json
```

Each receipt records:
- the resolved campaign spec
- the exact command
- the experiment id
- the proposal strategy and reason (for autonomous trials)
- the bounded override set
- the baseline used for comparison
- final metrics
- keep/discard decision

Parallel workers also use:

```text
repro/campaign_runs/<campaign>/<stage>/_claims/<candidate>.json
```

Claims are created atomically before a worker launches training. This is what
makes `campaign auto --max-trials 1` safe to fan out across many cluster pods
without duplicate candidate picks.

## Cluster Parallelism

The intended cluster pattern is:
- one pod = one autoresearch worker
- one worker = one claimed candidate (`--max-trials 1`)
- all workers share the same receipt PVC
- claims prevent duplicate picks

Dry-run the next candidate locally:

```bash
cd /workspace/nmoe
python -m nmoe.cli.main campaign auto speedrun_super_research \
  --stage benchmark \
  --receipt-dir /data/campaign_runs \
  --max-trials 1 \
  --max-no-improve 1 \
  --dry-run
```

Deploy the indexed worker job:

```bash
cd docker
make debug
make push-debug

kubectl apply -f k8s/campaign-worker.yaml
kubectl get jobs,pods -l app=nmoe,role=campaign
```

Scale worker fanout by editing `spec.parallelism` and `spec.completions` in
`k8s/campaign-worker.yaml`.

If `OPENAI_API_KEY` is not present in the worker environment, the LLM-guided
picker falls back automatically to deterministic coordinate search.

## Scope

Phase 1 is intentionally narrow:
- implemented runner: `speedrun`
- mutation tier: `config_only`
- allowed runtime mutation: `--set key=value` for allowlisted config keys
- autonomous mode is TOML-driven and currently uses deterministic coordinate
  descent or an LLM-guided bounded candidate picker over a finite override space
- smoke campaigns may shrink dataset budgets and disable CORE when the objective is
  a fixed-step proxy metric rather than the public leaderboard
- benchmark campaigns can reuse the full canonical speedrun dataset while still
  disabling CORE if the objective is early-loss comparison rather than a
  leaderboard promotion run
- research campaigns can require validation-backed metrics and CORE so the
  controller optimizes on actual research signals instead of train loss alone
- no tracked-file mutation
- no dynamic code patching

Future phases can widen the surface to research harness files, but they should
continue to write the same receipt shape and use the same budgeted loop.
