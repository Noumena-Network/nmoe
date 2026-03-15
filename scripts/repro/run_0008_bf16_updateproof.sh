#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
study_root_arg="${1:-blog_artifacts/0008_expert_lr_bf16_updateproof_20260311}"

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

run_arm() {
  local tag="$1"
  local experiment_id="$2"
  local lr_expert="$3"

  echo "[0008:updateproof] $experiment_id"
  "$torchrun_bin" --nnodes=1 --master_addr="$master_addr" --master_port="$(next_master_port)" --nproc_per_node=8 -m nmoe.train configs/moonlet.toml \
    --dtype=bf16 \
    --steps=200 \
    --experiment_id="$experiment_id" \
    --seed=42 \
    --lr_dense=0.003 \
    --lr_router=0.003 \
    --lr_expert="$lr_expert" \
    --adam_beta2_expert=0.99 \
    --checkpoint_dir="$study_root/checkpoints/$tag" \
    --metrics_dir="$study_root/metrics" \
    --experiments_db="$local_experiments_db" \
    --log_every=10

  sync_experiments_db
}

run_arm m1_s42 0008_updateproof_m1_s42 0.003
run_arm m15_s42 0008_updateproof_m15_s42 0.045
