#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSpec:
  arm: str
  seed: int
  experiment_id: str
  dtype: str
  dither: bool


RUNS: tuple[RunSpec, ...] = (
  RunSpec("bf16", 42, "0005_bf16_s42_e", "bf16", True),
  RunSpec("bf16", 43, "0005_bf16_s43_e", "bf16", True),
  RunSpec("bf16", 44, "0005_bf16_s44_e", "bf16", True),
  RunSpec("nvfp4_rtn", 42, "0005_nvfp4_rtn_s42_e", "nvfp4", False),
  RunSpec("nvfp4_rtn", 43, "0005_nvfp4_rtn_s43_e", "nvfp4", False),
  RunSpec("nvfp4_rtn", 44, "0005_nvfp4_rtn_s44_e", "nvfp4", False),
  RunSpec("nvfp4_dither", 42, "0005_nvfp4_dither_s42_e", "nvfp4", True),
  RunSpec("nvfp4_dither", 43, "0005_nvfp4_dither_s43_e", "nvfp4", True),
  RunSpec("nvfp4_dither", 44, "0005_nvfp4_dither_s44_e", "nvfp4", True),
)

STEP_KEYS: tuple[str, ...] = (
  "train/loss",
  "valid/loss",
  "router_agg/mean_entropy",
  "router_agg/mean_cv",
  "router_agg/mean_max_load",
  "quant/r0/nvfp4/layer_01/W13_q_nibble_flip_frac",
  "quant/r0/nvfp4/layer_01/W2_q_nibble_flip_frac",
  "quant/r0/nvfp4/layer_11/W13_q_nibble_flip_frac",
  "quant/r0/nvfp4/layer_11/W2_q_nibble_flip_frac",
  "throughput/tokens_per_s_gpu",
  "throughput/ms_per_step",
  "efficiency/tflops",
)


def resolve_study_root(raw: str) -> Path:
  text = raw.strip()
  if text.startswith('/data/'):
    return Path(text)
  if text.startswith('blog_artifacts/'):
    return Path('/data') / text
  return Path(text).resolve()


def load_run(cur: sqlite3.Cursor, spec: RunSpec) -> dict[str, Any]:
  row = cur.execute(
    'select id, status, results_json from runs where experiment_id=? order by started_at desc limit 1',
    (spec.experiment_id,),
  ).fetchone()
  if row is None:
    raise RuntimeError(f'missing run for experiment_id={spec.experiment_id}')
  run_id, status, results_json = row
  results = json.loads(results_json) if results_json else {}
  return {
    'arm': spec.arm,
    'seed': spec.seed,
    'dtype': spec.dtype,
    'dither': spec.dither,
    'experiment_id': spec.experiment_id,
    'run_id': run_id,
    'status': status,
    'final_loss': results.get('final_loss'),
    'steps_completed': results.get('steps_completed'),
    'stop_reason': results.get('stop_reason'),
  }


def format_loss(value: Any) -> str:
  return 'n/a' if value is None else f'{float(value):.4f}'


def format_metric(value: Any) -> str:
  return 'n/a' if value is None else f'{float(value):.6g}'


def parse_steps(raw: str) -> list[int]:
  out: list[int] = []
  for part in raw.split(','):
    text = part.strip()
    if text:
      out.append(int(text))
  return out


def load_step_metrics(study_root: Path, row: dict[str, Any], steps: list[int]) -> list[dict[str, Any]]:
  if not steps:
    return []
  import pyarrow.parquet as pq  # type: ignore

  metrics_root = study_root / 'metrics' / str(row['run_id'])
  max_step = int(row.get('steps_completed') or 0)
  out: list[dict[str, Any]] = []
  for step in steps:
    if step > max_step:
      continue
    path = metrics_root / f'step_{step:08d}.parquet'
    if not path.exists():
      continue
    tags = {item['tag']: item['value'] for item in pq.read_table(path).to_pylist()}
    item = {'step': step}
    for key in STEP_KEYS:
      item[key] = tags.get(key)
    out.append(item)
  return out


def print_table(rows: list[dict[str, Any]]) -> None:
  header = f"{'arm':<13} {'seed':<4} {'dtype':<5} {'dither':<6} {'status':<15} {'steps':>5} {'loss':>8}  stop_reason"
  print(header)
  print('-' * len(header))
  for row in rows:
    print(
      f"{row['arm']:<13} {row['seed']:<4} {row['dtype']:<5} {str(row['dither']):<6} {row['status']:<15} "
      f"{int(row['steps_completed'] or 0):>5} {format_loss(row['final_loss']):>8}  {row['stop_reason'] or 'n/a'}"
    )


def print_step_metrics(rows: list[dict[str, Any]], step_rows: list[dict[str, Any]]) -> None:
  if not step_rows:
    return
  by_arm_seed = {(row['arm'], row['seed']): row for row in rows}
  print('\nstep metrics')
  print('------------')
  for item in step_rows:
    row = by_arm_seed[(item['arm'], item['seed'])]
    print(f"[{row['arm']} seed={row['seed']} step={item['step']}]")
    for key in STEP_KEYS:
      if item.get(key) is not None:
        print(f"  {key}: {format_metric(item[key])}")


def main() -> None:
  ap = argparse.ArgumentParser(description='Summarize the 0005 nvfp4 rerun matrix.')
  ap.add_argument('--study-root', default='blog_artifacts/0005_nvfp4_matrix_20260312')
  ap.add_argument('--steps', default='100,1000,3000,6000,9536')
  args = ap.parse_args()

  study_root = resolve_study_root(args.study_root)
  db_path = study_root / 'experiments.db'
  if not db_path.exists():
    raise RuntimeError(f'missing experiments db: {db_path}')

  with sqlite3.connect(db_path) as conn:
    cur = conn.cursor()
    rows = [load_run(cur, spec) for spec in RUNS]

  print(f'study_root: {study_root}')
  print_table(rows)

  requested_steps = parse_steps(args.steps)
  step_rows: list[dict[str, Any]] = []
  for row in rows:
    for item in load_step_metrics(study_root, row, requested_steps):
      item['arm'] = row['arm']
      item['seed'] = row['seed']
      step_rows.append(item)
  print_step_metrics(rows, step_rows)


if __name__ == '__main__':
  main()
