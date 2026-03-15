#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
study_root_arg="${1:-blog_artifacts/0005_nvfp4_matrix_20260312}"

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

steps="${NMOE_0005_STEPS:-9536}"
speedrun_data_root="${NMOE_SPEEDRUN_DATA_ROOT:-/data/speedrun}"
seeds=( ${NMOE_0005_SEEDS:-42 43 44} )
arms=( ${NMOE_0005_ARMS:-bf16 nvfp4_rtn nvfp4_dither} )

run_arm() {
  local arm="$1"
  local seed="$2"
  local dtype="$3"
  local dither="$4"
  local forward_ablation="$5"
  local experiment_id="0005_${arm}_s${seed}_e"

  echo "[0005:matrix] $experiment_id"
  "$torchrun_bin" --nnodes=1 --master_addr="$master_addr" --master_port="$(next_master_port)" --nproc_per_node=8 -m nmoe.train configs/speedrun/moe.toml \
    --dtype="$dtype" \
    --steps="$steps" \
    --target_loss=0 \
    --data_path="$speedrun_data_root/train" \
    --validation_data_path="$speedrun_data_root/val" \
    --experiment_id="$experiment_id" \
    --seed="$seed" \
    --checkpoint_dir="$study_root/checkpoints/${arm}_s${seed}" \
    --metrics_dir="$study_root/metrics" \
    --experiments_db="$local_experiments_db" \
    --collect_update_stats=false \
    --nvfp4_resonance_dither="$dither" \
    --blockscaled_forward_ablation="$forward_ablation"

  sync_experiments_db
}

for seed in "${seeds[@]}"; do
  for arm in "${arms[@]}"; do
    case "$arm" in
      bf16)
        run_arm "$arm" "$seed" bf16 true off
        ;;
      nvfp4_rtn)
        run_arm "$arm" "$seed" nvfp4 false off
        ;;
      nvfp4_dither)
        run_arm "$arm" "$seed" nvfp4 true off
        ;;
      nvfp4_w13_bf16)
        run_arm "$arm" "$seed" nvfp4 false w13_bf16
        ;;
      nvfp4_stage1_bf16)
        run_arm "$arm" "$seed" nvfp4 false stage1_bf16
        ;;
      nvfp4_full_bf16)
        run_arm "$arm" "$seed" nvfp4 false full_bf16
        ;;
      nvfp4_w13_fp8)
        run_arm "$arm" "$seed" nvfp4 false w13_fp8
        ;;
      nvfp4_stage1_fp8)
        run_arm "$arm" "$seed" nvfp4 false stage1_fp8
        ;;
      nvfp4_full_fp8)
        run_arm "$arm" "$seed" nvfp4 false full_fp8
        ;;
      *)
        echo "unsupported arm: $arm" >&2
        exit 1
        ;;
    esac
  done
done
