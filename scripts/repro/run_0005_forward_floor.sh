#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
checkpoint_root_arg="${1:?usage: run_0005_forward_floor.sh <checkpoint-root-or-iter-dir> [study-root]}"
study_root_arg="${2:-blog_artifacts/0005_forward_floor_20260312}"
speedrun_data_root="${NMOE_SPEEDRUN_DATA_ROOT:-/data/speedrun}"

case "$checkpoint_root_arg" in
  /data/*)
    checkpoint_root="$checkpoint_root_arg"
    ;;
  blog_artifacts/*)
    checkpoint_root="/data/$checkpoint_root_arg"
    ;;
  *)
    checkpoint_root="$checkpoint_root_arg"
    ;;
esac

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
mkdir -p "$study_root"
cd "$repo_root"

master_addr="${NMOE_TORCHRUN_MASTER_ADDR:-127.0.0.1}"
base_port="${NMOE_TORCHRUN_BASE_PORT:-29500}"

next_master_port() {
  local port="$base_port"
  base_port=$((base_port + 1))
  printf '%s' "$port"
}

run_eval() {
  local label="$1"
  local dtype="$2"
  "$torchrun_bin" --nnodes=1 --master_addr="$master_addr" --master_port="$(next_master_port)" --nproc_per_node=8 scripts/repro/eval_0005_forward_floor.py \
    --checkpoint-root="$checkpoint_root" \
    --dtype="$dtype" \
    --speedrun-data-root="$speedrun_data_root" \
    --out-json="$study_root/${label}.json"
}

run_eval bf16_floor bf16
run_eval nvfp4_floor nvfp4
