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
  label: str
  experiment_id: str
  multiplier: str
  seed: int
  beta2_expert: float


@dataclass(frozen=True)
class RunGroup:
  label: str
  default_study_root: str
  rows: tuple[RunSpec, ...]
  supplemental_label: str | None = None
  supplemental_rows: tuple[RunSpec, ...] = ()


MAIN_SWEEP: tuple[RunSpec, ...] = (
  RunSpec("m0p5_s42", "0008_moonlet_bf16_m0p5_s42", "0.5x", 42, 0.99),
  RunSpec("m1_s42", "0008_moonlet_bf16_m1_s42", "1x", 42, 0.99),
  RunSpec("m2_s42", "0008_moonlet_bf16_m2_s42", "2x", 42, 0.99),
  RunSpec("m4_s42", "0008_moonlet_bf16_m4_s42", "4x", 42, 0.99),
  RunSpec("m15_s42", "0008_moonlet_bf16_m15_s42", "15x", 42, 0.99),
  RunSpec("m0p5_s43", "0008_moonlet_bf16_m0p5_s43", "0.5x", 43, 0.99),
  RunSpec("m1_s43", "0008_moonlet_bf16_m1_s43", "1x", 43, 0.99),
  RunSpec("m2_s43", "0008_moonlet_bf16_m2_s43", "2x", 43, 0.99),
  RunSpec("m4_s43", "0008_moonlet_bf16_m4_s43", "4x", 43, 0.99),
  RunSpec("m15_s43", "0008_moonlet_bf16_m15_s43", "15x", 43, 0.99),
)

BETA2_ABLATION: tuple[RunSpec, ...] = (
  RunSpec("m1_b95_s42", "0008_moonlet_bf16_m1_b95_s42", "1x", 42, 0.95),
  RunSpec("m1_b95_s43", "0008_moonlet_bf16_m1_b95_s43", "1x", 43, 0.95),
)

UPDATE_PROOF: tuple[RunSpec, ...] = (
  RunSpec("m1_s42", "0008_updateproof_m1_s42", "1x", 42, 0.99),
  RunSpec("m15_s42", "0008_updateproof_m15_s42", "15x", 42, 0.99),
)

NVFP4_UPDATE_PROOF: tuple[RunSpec, ...] = (
  RunSpec("m1_s42_b", "0008_nvfp4_updateproof_m1_s42_b", "1x", 42, 0.99),
  RunSpec("m15_s42_b", "0008_nvfp4_updateproof_m15_s42_b", "15x", 42, 0.99),
)

NVFP4_UPDATE_PROOF_EXPANDED: tuple[RunSpec, ...] = (
  RunSpec("m1_s42_e", "0008_nvfp4_updateproof_m1_s42_e", "1x", 42, 0.99),
  RunSpec("m2_s42_e", "0008_nvfp4_updateproof_m2_s42_e", "2x", 42, 0.99),
  RunSpec("m4_s42_e", "0008_nvfp4_updateproof_m4_s42_e", "4x", 42, 0.99),
  RunSpec("m15_s42_e", "0008_nvfp4_updateproof_m15_s42_e", "15x", 42, 0.99),
)

NVFP4_MAIN: tuple[RunSpec, ...] = (
  RunSpec("m0p5_s42_d", "0008_nvfp4_m0p5_s42_d", "0.5x", 42, 0.99),
  RunSpec("m1_s42_d", "0008_nvfp4_m1_s42_d", "1x", 42, 0.99),
  RunSpec("m2_s42_d", "0008_nvfp4_m2_s42_d", "2x", 42, 0.99),
  RunSpec("m4_s42_d", "0008_nvfp4_m4_s42_d", "4x", 42, 0.99),
  RunSpec("m15_s42_d", "0008_nvfp4_m15_s42_d", "15x", 42, 0.99),
  RunSpec("m0p5_s43_d", "0008_nvfp4_m0p5_s43_d", "0.5x", 43, 0.99),
  RunSpec("m1_s43_d", "0008_nvfp4_m1_s43_d", "1x", 43, 0.99),
  RunSpec("m2_s43_d", "0008_nvfp4_m2_s43_d", "2x", 43, 0.99),
  RunSpec("m4_s43_d", "0008_nvfp4_m4_s43_d", "4x", 43, 0.99),
  RunSpec("m15_s43_d", "0008_nvfp4_m15_s43_d", "15x", 43, 0.99),
  RunSpec("m0p5_s44_d", "0008_nvfp4_m0p5_s44_d", "0.5x", 44, 0.99),
  RunSpec("m1_s44_d", "0008_nvfp4_m1_s44_d", "1x", 44, 0.99),
  RunSpec("m2_s44_d", "0008_nvfp4_m2_s44_d", "2x", 44, 0.99),
  RunSpec("m4_s44_d", "0008_nvfp4_m4_s44_d", "4x", 44, 0.99),
  RunSpec("m15_s44_d", "0008_nvfp4_m15_s44_d", "15x", 44, 0.99),
)

NVFP4_GRADHEALTH: tuple[RunSpec, ...] = (
  RunSpec("m1_s42_c", "0008_nvfp4_gradhealth_m1_s42_c", "1x", 42, 0.99),
  RunSpec("m15_s42_c", "0008_nvfp4_gradhealth_m15_s42_c", "15x", 42, 0.99),
)

RUN_GROUPS: dict[str, RunGroup] = {
  "main": RunGroup(
    label="main",
    default_study_root="blog_artifacts/0008_expert_lr_bf16_20260311",
    rows=MAIN_SWEEP,
    supplemental_label="beta2_ablation",
    supplemental_rows=BETA2_ABLATION,
  ),
  "updateproof": RunGroup(
    label="updateproof",
    default_study_root="blog_artifacts/0008_expert_lr_bf16_updateproof_20260311",
    rows=UPDATE_PROOF,
  ),
  "nvfp4_updateproof": RunGroup(
    label="nvfp4_updateproof",
    default_study_root="blog_artifacts/0008_expert_lr_nvfp4_updateproof_b20260311",
    rows=NVFP4_UPDATE_PROOF,
  ),
  "nvfp4_updateproof_expanded": RunGroup(
    label="nvfp4_updateproof_expanded",
    default_study_root="blog_artifacts/0008_expert_lr_nvfp4_updateproof_e20260312",
    rows=NVFP4_UPDATE_PROOF_EXPANDED,
  ),
  "nvfp4_main": RunGroup(
    label="nvfp4_main",
    default_study_root="blog_artifacts/0008_expert_lr_nvfp4_decision_d20260312",
    rows=NVFP4_MAIN,
  ),
  "nvfp4_gradhealth": RunGroup(
    label="nvfp4_gradhealth",
    default_study_root="blog_artifacts/0008_expert_lr_nvfp4_gradhealth_c20260311",
    rows=NVFP4_GRADHEALTH,
  ),
}

STEP_KEYS: tuple[str, ...] = (
  "train/loss",
  "train_signal/dense/grad_to_param",
  "train_signal/dense/update_to_pre_param",
  "train_signal/dense/optimizer_update_to_pre_param",
  "train_signal/expert/grad_to_param",
  "train_signal/expert/update_to_pre_param",
  "train_signal/expert/optimizer_update_to_pre_param",
  "train_signal/router/grad_to_param",
  "train_signal/router/update_to_pre_param",
  "router_agg/mean_p90_load",
  "router_agg/mean_p90_importance",
  "router_agg/mean_max_importance",
  "train/moe_grad_zero_frac_w1",
  "train/moe_grad_zero_frac_w2",
  "train/moe_grad_zero_frac_w3",
  "train/moe_grad_abs_mean_w1",
  "train/moe_grad_abs_mean_w2",
  "train/moe_grad_abs_mean_w3",
  "train/moe_grad_abs_max_w1",
  "train/moe_grad_abs_max_w2",
  "train/moe_grad_abs_max_w3",
)

ZERO_FRAC_KEYS: tuple[str, ...] = (
  "train/moe_grad_zero_frac_w1",
  "train/moe_grad_zero_frac_w2",
  "train/moe_grad_zero_frac_w3",
)

ABS_MEAN_KEYS: tuple[str, ...] = (
  "train/moe_grad_abs_mean_w1",
  "train/moe_grad_abs_mean_w2",
  "train/moe_grad_abs_mean_w3",
)

ABS_MAX_KEYS: tuple[str, ...] = (
  "train/moe_grad_abs_max_w1",
  "train/moe_grad_abs_max_w2",
  "train/moe_grad_abs_max_w3",
)


def _resolve_study_root(raw: str) -> Path:
  text = raw.strip()
  if text.startswith("/data/"):
    return Path(text)
  if text.startswith("blog_artifacts/"):
    return Path("/data") / text
  return Path(text).resolve()


def _load_run(cur: sqlite3.Cursor, spec: RunSpec) -> dict[str, Any]:
  row = cur.execute(
    "select id, status, results_json from runs where experiment_id=? order by started_at desc limit 1",
    (spec.experiment_id,),
  ).fetchone()
  if row is None:
    raise RuntimeError(f"missing run for experiment_id={spec.experiment_id}")
  run_id, status, results_json = row
  results = json.loads(results_json) if results_json else {}
  return {
    "label": spec.label,
    "experiment_id": spec.experiment_id,
    "multiplier": spec.multiplier,
    "seed": spec.seed,
    "beta2_expert": spec.beta2_expert,
    "run_id": run_id,
    "status": status,
    "final_loss": results.get("final_loss"),
    "steps_completed": results.get("steps_completed"),
    "stop_reason": results.get("stop_reason"),
    "tokens_seen": results.get("tokens_seen"),
  }


def _format_loss(value: Any) -> str:
  if value is None:
    return "n/a"
  return f"{float(value):.4f}"


def _fmt_metric(value: Any) -> str:
  if value is None:
    return "n/a"
  return f"{float(value):.6g}"


def _print_table(rows: list[dict[str, Any]]) -> None:
  header = (
    f"{'mult':<6} {'seed':<4} {'beta2':<5} {'status':<15} "
    f"{'steps':>5} {'loss':>8}  stop_reason"
  )
  print(header)
  print("-" * len(header))
  for row in rows:
    print(
      f"{row['multiplier']:<6} {row['seed']:<4} {row['beta2_expert']:<5.2f} "
      f"{row['status']:<15} {int(row['steps_completed'] or 0):>5} "
      f"{_format_loss(row['final_loss']):>8}  {row['stop_reason'] or 'n/a'}"
    )


def _parse_steps(raw: str) -> list[int]:
  if not raw.strip():
    return []
  out = []
  for part in raw.split(','):
    text = part.strip()
    if not text:
      continue
    out.append(int(text))
  return out


def _load_step_metrics(study_root: Path, row: dict[str, Any], steps: list[int]) -> list[dict[str, Any]]:
  if not steps:
    return []
  try:
    import pyarrow.parquet as pq  # type: ignore
  except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyarrow is required for --steps") from exc

  metrics_root = study_root / "metrics" / str(row["run_id"])
  max_step = int(row.get("steps_completed") or 0)
  out: list[dict[str, Any]] = []
  for step in steps:
    if step > max_step:
      continue
    path = metrics_root / f"step_{step:08d}.parquet"
    if not path.exists():
      continue
    tags = {item["tag"]: item["value"] for item in pq.read_table(path).to_pylist()}
    dense_grad = tags.get("train_signal/dense/grad_to_param")
    expert_grad = tags.get("train_signal/expert/grad_to_param")
    dense_upd = tags.get("train_signal/dense/update_to_pre_param")
    expert_upd = tags.get("train_signal/expert/update_to_pre_param")
    dense_opt = tags.get("train_signal/dense/optimizer_update_to_pre_param")
    expert_opt = tags.get("train_signal/expert/optimizer_update_to_pre_param")
    item = {"step": step}
    for key in STEP_KEYS:
      item[key] = tags.get(key)
    item["expert_to_dense_grad_ratio"] = (
      (float(expert_grad) / float(dense_grad))
      if dense_grad not in (None, 0.0) and expert_grad is not None
      else None
    )
    item["expert_to_dense_update_ratio"] = (
      (float(expert_upd) / float(dense_upd))
      if dense_upd not in (None, 0.0) and expert_upd is not None
      else None
    )
    item["expert_to_dense_optimizer_update_ratio"] = (
      (float(expert_opt) / float(dense_opt))
      if dense_opt not in (None, 0.0) and expert_opt is not None
      else None
    )
    out.append(item)
  return out


def _attach_step_metrics(study_root: Path, rows: list[dict[str, Any]], steps: list[int]) -> None:
  for row in rows:
    row["step_metrics"] = _load_step_metrics(study_root, row, steps)


def _print_step_metrics(rows: list[dict[str, Any]]) -> None:
  for row in rows:
    metrics = row.get("step_metrics") or []
    if not metrics:
      continue
    print(
      f"\n{row['experiment_id']} ({row['status']}, steps={int(row['steps_completed'] or 0)}, loss={_format_loss(row['final_loss'])})"
    )
    for item in metrics:
      parts = [
        f"step {int(item['step']):>4}: loss={_fmt_metric(item.get('train/loss'))}",
        f"dense grad={_fmt_metric(item.get('train_signal/dense/grad_to_param'))}",
        f"dense upd={_fmt_metric(item.get('train_signal/dense/update_to_pre_param'))}",
        f"dense opt_upd={_fmt_metric(item.get('train_signal/dense/optimizer_update_to_pre_param'))}",
        f"expert grad={_fmt_metric(item.get('train_signal/expert/grad_to_param'))}",
        f"expert upd={_fmt_metric(item.get('train_signal/expert/update_to_pre_param'))}",
        f"expert opt_upd={_fmt_metric(item.get('train_signal/expert/optimizer_update_to_pre_param'))}",
        f"grad_ratio={_fmt_metric(item.get('expert_to_dense_grad_ratio'))}",
        f"upd_ratio={_fmt_metric(item.get('expert_to_dense_update_ratio'))}",
        f"opt_upd_ratio={_fmt_metric(item.get('expert_to_dense_optimizer_update_ratio'))}",
        f"p90_load={_fmt_metric(item.get('router_agg/mean_p90_load'))}",
        f"p90_importance={_fmt_metric(item.get('router_agg/mean_p90_importance'))}",
        f"max_importance={_fmt_metric(item.get('router_agg/mean_max_importance'))}",
      ]
      if any(item.get(key) is not None for key in ZERO_FRAC_KEYS):
        parts.extend(
          [
            f"w1_zero={_fmt_metric(item.get('train/moe_grad_zero_frac_w1'))}",
            f"w2_zero={_fmt_metric(item.get('train/moe_grad_zero_frac_w2'))}",
            f"w3_zero={_fmt_metric(item.get('train/moe_grad_zero_frac_w3'))}",
          ]
        )
      if any(item.get(key) is not None for key in ABS_MEAN_KEYS):
        parts.extend(
          [
            f"w1_abs_mean={_fmt_metric(item.get('train/moe_grad_abs_mean_w1'))}",
            f"w2_abs_mean={_fmt_metric(item.get('train/moe_grad_abs_mean_w2'))}",
            f"w3_abs_mean={_fmt_metric(item.get('train/moe_grad_abs_mean_w3'))}",
          ]
        )
      if any(item.get(key) is not None for key in ABS_MAX_KEYS):
        parts.extend(
          [
            f"w1_abs_max={_fmt_metric(item.get('train/moe_grad_abs_max_w1'))}",
            f"w2_abs_max={_fmt_metric(item.get('train/moe_grad_abs_max_w2'))}",
            f"w3_abs_max={_fmt_metric(item.get('train/moe_grad_abs_max_w3'))}",
          ]
        )
      print("  " + " ".join(parts))


def main() -> int:
  ap = argparse.ArgumentParser(description="Summarize the 0008 Moonlet expert-LR studies.")
  ap.add_argument(
    "--study-root",
    default="",
    help="Study root (logical blog_artifacts/... or filesystem path). Defaults to the selected --kind root.",
  )
  ap.add_argument(
    "--kind",
    default="main",
    choices=sorted(RUN_GROUPS),
    help="Which 0008 run set to summarize",
  )
  ap.add_argument(
    "--steps",
    default="",
    help="Comma-separated steps to include parquet metrics for (for example: 10,50,100)",
  )
  ap.add_argument("--json", action="store_true", help="Emit JSON instead of a text table")
  args = ap.parse_args()

  group = RUN_GROUPS[args.kind]
  study_root = _resolve_study_root(args.study_root or group.default_study_root)
  steps = _parse_steps(args.steps)
  db_path = study_root / "experiments.db"
  if not db_path.exists():
    raise SystemExit(f"missing experiments DB: {db_path}")

  con = sqlite3.connect(str(db_path))
  cur = con.cursor()
  main_rows = [_load_run(cur, spec) for spec in group.rows]
  supplemental_rows = [_load_run(cur, spec) for spec in group.supplemental_rows]
  con.close()
  if steps:
    _attach_step_metrics(study_root, main_rows, steps)
    if supplemental_rows:
      _attach_step_metrics(study_root, supplemental_rows, steps)

  if args.json:
    print(
      json.dumps(
        {
          "kind": args.kind,
          "study_root": str(study_root),
          "rows": main_rows,
          "supplemental_label": group.supplemental_label,
          "supplemental_rows": supplemental_rows,
        },
        indent=2,
      )
    )
    return 0

  print(f"study_root: {study_root}")
  print(f"\n{group.label}")
  _print_table(main_rows)
  if steps:
    _print_step_metrics(main_rows)
  if supplemental_rows:
    print(f"\n{group.supplemental_label}")
    _print_table(supplemental_rows)
    if steps:
      _print_step_metrics(supplemental_rows)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
