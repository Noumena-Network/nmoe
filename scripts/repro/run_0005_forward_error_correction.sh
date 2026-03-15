#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
checkpoint_root="${1:-/data/blog_artifacts/0005_nvfp4_matrix_f20260312/checkpoints/bf16_s42}"
out_json="${2:-/data/blog_artifacts/0005_forward_error_correction_20260312/nvfp4_error_correction.json}"
cases="${NMOE_0005_ERROR_CORR_CASES:-bf16,full_forward@baseline,full_forward@scale75,full_forward@affine,full_forward@scale75_affine,stage1_only@baseline,stage1_only@scale75_affine,w13_only@scale75_affine}"
validation_steps="${NMOE_0005_ERROR_CORR_VALIDATION_STEPS:-20}"
master_addr="127.0.0.1"
base_port="${NMOE_TORCHRUN_BASE_PORT:-29971}"

export PYTHONPATH="$repo_root:$repo_root/nmoe/csrc:$repo_root/third_party/flash_attn:$repo_root/third_party/quack:$repo_root/triton/python${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"

torchrun --nnodes=1 --master_addr="$master_addr" --master_port="$base_port" --nproc_per_node=8 \
  scripts/repro/eval_0005_forward_error_correction.py \
  --checkpoint-root "$checkpoint_root" \
  --profile nvfp4 \
  --cases "$cases" \
  --validation-steps "$validation_steps" \
  --out-json "$out_json"
