#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
study_root_arg="${1:-blog_artifacts/0008_expert_lr_nvfp4_decision_d20260312}"

case "$study_root_arg" in
  /data/*)
    study_root="$study_root_arg"
    ;;
  blog_artifacts/*)
    study_root="/data/$study_root_arg"
    ;;
  *)
    study_root="$study_root_arg"
    ;;
esac

torchrun_bin="$repo_root/.venv/bin/torchrun"
if [[ ! -x "$torchrun_bin" ]]; then
  echo "missing torchrun launcher: $torchrun_bin" >&2
  exit 1
fi

export PYTHONPATH="$repo_root:$repo_root/nmoe/csrc:$repo_root/third_party/flash_attn:$repo_root/third_party/quack:$repo_root/triton/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$study_root/checkpoints" "$study_root/metrics"
cd "$repo_root"

master_addr="${NMOE_TORCHRUN_MASTER_ADDR:-127.0.0.1}"
base_port="${NMOE_TORCHRUN_BASE_PORT:-29500}"

next_master_port() {
  local port="$base_port"
  base_port=$((base_port + 1))
  printf '%s' "$port"
}

local_db_root="${NMOE_LOCAL_EXPERIMENTS_DB_ROOT:-$repo_root/tmp/repro_experiments}"
mkdir -p "$local_db_root"
study_slug="$(basename "$study_root")"
local_experiments_db="$local_db_root/${study_slug}.db"
if [[ -f "$study_root/experiments.db" && ! -f "$local_experiments_db" ]]; then
  cp "$study_root/experiments.db" "$local_experiments_db"
fi

sync_experiments_db() {
  if [[ -f "$local_experiments_db" ]]; then
    cp "$local_experiments_db" "$study_root/experiments.db"
  fi
}

steps="${NMOE_0008_STEPS:-600}"
seeds=( ${NMOE_0008_SEEDS:-42 43 44} )
multipliers=( ${NMOE_0008_MULTIPLIERS:-0.5 1 2 4 15} )

run_arm() {
  local seed="$1"
  local multiplier="$2"
  local label="$3"
  local lr_expert="$4"
  local experiment_id="0008_nvfp4_${label}_s${seed}_d"

  echo "[0008:nvfp4:sweep] $experiment_id"
  "$torchrun_bin" --nnodes=1 --master_addr="$master_addr" --master_port="$(next_master_port)" --nproc_per_node=8 -m nmoe.train configs/moonlet.toml \
    --dtype=nvfp4 \
    --steps="$steps" \
    --experiment_id="$experiment_id" \
    --seed="$seed" \
    --lr_dense=0.003 \
    --lr_router=0.003 \
    --lr_expert="$lr_expert" \
    --adam_beta2_expert=0.99 \
    --checkpoint_dir="$study_root/checkpoints/${label}_s${seed}" \
    --metrics_dir="$study_root/metrics" \
    --experiments_db="$local_experiments_db" \
    --log_every=10 \
    --collect_update_stats=false

  sync_experiments_db
}

for seed in "${seeds[@]}"; do
  for multiplier in "${multipliers[@]}"; do
    case "$multiplier" in
      0.5)
        run_arm "$seed" "$multiplier" "m0p5" 0.0015
        ;;
      1)
        run_arm "$seed" "$multiplier" "m1" 0.003
        ;;
      2)
        run_arm "$seed" "$multiplier" "m2" 0.006
        ;;
      4)
        run_arm "$seed" "$multiplier" "m4" 0.012
        ;;
      15)
        run_arm "$seed" "$multiplier" "m15" 0.045
        ;;
      *)
        echo "unsupported multiplier: $multiplier" >&2
        exit 1
        ;;
    esac
  done
done
