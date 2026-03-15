#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
study_root_arg="${1:-blog_artifacts/0008_expert_lr_nvfp4_updateproof_e20260312}"

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
steps="${NMOE_0008_UPDATEPROOF_STEPS:-200}"
seeds=( ${NMOE_0008_UPDATEPROOF_SEEDS:-42} )
multipliers=( ${NMOE_0008_UPDATEPROOF_MULTIPLIERS:-1 2 4 15} )
beta2_expert="${NMOE_0008_UPDATEPROOF_BETA2_EXPERT:-0.99}"
run_suffix="${NMOE_0008_UPDATEPROOF_SUFFIX:-e}"

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
  local seed="$3"
  local lr_expert="$4"

  echo "[0008:nvfp4:updateproof] $experiment_id"
  "$torchrun_bin" --nnodes=1 --master_addr="$master_addr" --master_port="$(next_master_port)" --nproc_per_node=8 -m nmoe.train configs/moonlet.toml \
    --dtype=nvfp4 \
    --steps="$steps" \
    --experiment_id="$experiment_id" \
    --seed="$seed" \
    --lr_dense=0.003 \
    --lr_router=0.003 \
    --lr_expert="$lr_expert" \
    --adam_beta2_expert="$beta2_expert" \
    --checkpoint_dir="$study_root/checkpoints/$tag" \
    --metrics_dir="$study_root/metrics" \
    --experiments_db="$local_experiments_db" \
    --log_every=10

  sync_experiments_db
}

for seed in "${seeds[@]}"; do
  for multiplier in "${multipliers[@]}"; do
    case "$multiplier" in
      0.5)
        run_arm "m0p5_s${seed}" "0008_nvfp4_updateproof_m0p5_s${seed}_${run_suffix}" "$seed" 0.0015
        ;;
      1)
        run_arm "m1_s${seed}" "0008_nvfp4_updateproof_m1_s${seed}_${run_suffix}" "$seed" 0.003
        ;;
      2)
        run_arm "m2_s${seed}" "0008_nvfp4_updateproof_m2_s${seed}_${run_suffix}" "$seed" 0.006
        ;;
      4)
        run_arm "m4_s${seed}" "0008_nvfp4_updateproof_m4_s${seed}_${run_suffix}" "$seed" 0.012
        ;;
      15)
        run_arm "m15_s${seed}" "0008_nvfp4_updateproof_m15_s${seed}_${run_suffix}" "$seed" 0.045
        ;;
      *)
        echo "unsupported multiplier: $multiplier" >&2
        exit 1
        ;;
    esac
  done
done
