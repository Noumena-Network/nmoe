#!/usr/bin/env bash
set -euo pipefail

checkpoint_root_arg="${1:?usage: run_0005_forward_floor_split.sh <checkpoint-root-or-iter-dir> [study-root] [profile] [analyze-steps]}"
study_root_arg="${2:-blog_artifacts/0005_forward_floor_split_20260312}"
profile_arg="${3:-nvfp4}"
analyze_steps_arg="${4:-4}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ "$study_root_arg" = /* ]]; then
  study_root="$study_root_arg"
else
  study_root="/data/${study_root_arg#./}"
fi
mkdir -p "$study_root"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$repo_root:$repo_root/nmoe/csrc:$repo_root/third_party/flash_attn:$repo_root/third_party/quack:$repo_root/triton/python"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0

master_addr="127.0.0.1"
base_port="${NMOE_TORCHRUN_BASE_PORT:-29881}"

echo "[0005:forward-floor-split] checkpoint_root=$checkpoint_root_arg"
echo "[0005:forward-floor-split] study_root=$study_root"
echo "[0005:forward-floor-split] profile=$profile_arg analyze_steps=$analyze_steps_arg"

torchrun --nnodes=1 --master_addr="$master_addr" --master_port="$base_port" --nproc_per_node=8 \
  scripts/repro/eval_0005_forward_floor_split.py \
  --checkpoint-root "$checkpoint_root_arg" \
  --profile "$profile_arg" \
  --analyze-steps "$analyze_steps_arg" \
  --out-json "$study_root/${profile_arg}_split.json"
