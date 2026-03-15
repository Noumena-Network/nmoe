#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_receipt(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"receipt must be a JSON object: {path}")
    return obj


def _request_json(base_url: str, endpoint: str, *, timeout_s: float) -> dict[str, Any] | None:
    url = f"{base_url.rstrip('/')}{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            obj = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    return value


def _receipt_row(obj: dict[str, Any]) -> dict[str, Any]:
    decision = obj.get("decision") or {}
    metrics = obj.get("metrics") or {}
    proposal = obj.get("proposal") or {}
    run = obj.get("run") or {}
    return {
        "receipt_path": obj.get("receipt_path"),
        "candidate_id": obj.get("candidate_id"),
        "status": obj.get("status"),
        "stage": obj.get("stage"),
        "worker_id": (obj.get("worker") or {}).get("id"),
        "run_id": run.get("run_id"),
        "experiment_id": obj.get("experiment_id"),
        "started_at": obj.get("started_at"),
        "ended_at": obj.get("ended_at"),
        "overrides": obj.get("overrides") or {},
        "proposal": {
            "strategy": proposal.get("strategy"),
            "axis": proposal.get("axis"),
            "current_value": proposal.get("current_value"),
            "proposed_value": proposal.get("proposed_value"),
            "reason": proposal.get("reason"),
        },
        "metrics": {
            "final_loss": metrics.get("final_loss"),
            "final_valid_loss": metrics.get("final_valid_loss"),
            "core": metrics.get("core"),
            "tokens_seen": metrics.get("tokens_seen"),
            "steps_completed": metrics.get("steps_completed"),
            "train_time_ms_excl_valid": metrics.get("train_time_ms_excl_valid"),
            "valid_time_ms": metrics.get("valid_time_ms"),
        },
        "decision": {
            "kept": decision.get("kept"),
            "improved": decision.get("improved"),
            "reason": decision.get("reason"),
            "baseline_value": decision.get("baseline_value"),
            "baseline_core_score": decision.get("baseline_core_score"),
            "current_value": decision.get("current_value"),
            "current_core_score": decision.get("current_core_score"),
            "constraint_failures": decision.get("constraint_failures") or [],
        },
    }


def _sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("started_at") or ""), str(row.get("candidate_id") or ""))


def _metric_value(row: dict[str, Any]) -> float | None:
    value = ((row.get("metrics") or {}).get("final_valid_loss"))
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _progression_rows(completed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept_rows = [row for row in completed_rows if (row.get("decision") or {}).get("kept")]
    kept_rows.sort(key=_sort_key)
    if not kept_rows:
        return []

    champion_rows: list[dict[str, Any]] = []
    best_value: float | None = None
    for row in kept_rows:
        current_value = _metric_value(row)
        if current_value is None:
            continue
        if best_value is None or current_value < best_value:
            champion_rows.append(row)
            best_value = current_value
    if not champion_rows:
        return []

    seed_value = _metric_value(champion_rows[0])
    out: list[dict[str, Any]] = []
    prev_value: float | None = None
    prev_candidate: str | None = None
    for rank, row in enumerate(champion_rows, start=1):
        current_value = _metric_value(row)
        delta_prev = None
        delta_seed = None
        if current_value is not None and prev_value is not None:
            delta_prev = current_value - prev_value
        if current_value is not None and seed_value is not None:
            delta_seed = current_value - seed_value
        out.append(
            {
                "rank": rank,
                "candidate_id": row.get("candidate_id"),
                "receipt_path": row.get("receipt_path"),
                "run_id": row.get("run_id"),
                "started_at": row.get("started_at"),
                "overrides": row.get("overrides") or {},
                "final_valid_loss": current_value,
                "core": ((row.get("metrics") or {}).get("core")),
                "delta_vs_previous": delta_prev,
                "delta_vs_seed": delta_seed,
                "previous_candidate": prev_candidate,
            }
        )
        prev_value = current_value
        prev_candidate = str(row.get("candidate_id") or "")
    return out


def _candidate_manifest(all_rows: list[dict[str, Any]], completed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    kept_rows = [row for row in completed_rows if (row.get("decision") or {}).get("kept")]
    failed_rows = [row for row in all_rows if row.get("status") == "failed" and _metric_value(row) is None]
    core_veto_rows = [
        row
        for row in completed_rows
        if "core_drop" in " ".join(str(x) for x in ((row.get("decision") or {}).get("constraint_failures") or []))
    ]
    best_row = None
    for row in kept_rows:
        if best_row is None:
            best_row = row
            continue
        current = _metric_value(row)
        best = _metric_value(best_row)
        if current is not None and (best is None or current < best):
            best_row = row
    return {
        "generated_at_utc": _now_utc(),
        "total_receipts": len(all_rows),
        "completed_receipts": len(completed_rows),
        "kept_receipts": len(kept_rows),
        "failed_receipts": len(failed_rows),
        "core_veto_receipts": len(core_veto_rows),
        "best_kept": None
        if best_row is None
        else {
            "candidate_id": best_row.get("candidate_id"),
            "receipt_path": best_row.get("receipt_path"),
            "run_id": best_row.get("run_id"),
            "final_valid_loss": _metric_value(best_row),
            "core": ((best_row.get("metrics") or {}).get("core")),
            "overrides": best_row.get("overrides") or {},
        },
        "failed_candidates": [
            {
                "candidate_id": row.get("candidate_id"),
                "receipt_path": row.get("receipt_path"),
                "reason": ((row.get("decision") or {}).get("reason")),
            }
            for row in failed_rows
        ],
        "core_veto_candidates": [
            {
                "candidate_id": row.get("candidate_id"),
                "receipt_path": row.get("receipt_path"),
                "final_valid_loss": _metric_value(row),
                "core": ((row.get("metrics") or {}).get("core")),
                "reason": ((row.get("decision") or {}).get("reason")),
            }
            for row in core_veto_rows
        ],
    }


def _run_summary_rows(
    completed_rows: list[dict[str, Any]],
    *,
    nviz_base_url: str | None,
    timeout_s: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in completed_rows:
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            continue
        summary = {}
        router = {}
        if nviz_base_url:
          encoded = urllib.parse.quote(run_id, safe="")
          summary_obj = _request_json(nviz_base_url, f"/api/summary?run={encoded}", timeout_s=timeout_s) or {}
          router_obj = _request_json(nviz_base_url, f"/api/router?run={encoded}", timeout_s=timeout_s) or {}
          summary = summary_obj.get("summary") or {}
          router = router_obj.get("agg") or {}
        out.append(
            {
                "run_id": run_id,
                "candidate_id": row.get("candidate_id"),
                "receipt_path": row.get("receipt_path"),
                "throughput": {
                    "tokens_per_s_gpu": summary.get("throughput/tokens_per_s_gpu"),
                    "ms_per_step": summary.get("throughput/ms_per_step"),
                    "tflops": summary.get("efficiency/tflops"),
                    "bf16_tflops": summary.get("efficiency/bf16_tflops"),
                },
                "gpu": {
                    "mean_utilization_gpu": summary.get("gpu_agg/mean_utilization_gpu"),
                    "max_temperature_c": summary.get("gpu_agg/max_temperature_c"),
                    "total_power_w": summary.get("gpu_agg/total_power_w"),
                    "total_memory_used_gib": summary.get("gpu_agg/total_memory_used_gib"),
                },
                "router": {
                    "mean_cv": router.get("router_agg/mean_cv"),
                    "std_cv": router.get("router_agg/std_cv"),
                    "mean_entropy": router.get("router_agg/mean_entropy"),
                    "min_entropy": router.get("router_agg/min_entropy"),
                    "experts_active_mean": router.get("router_agg/experts_active_mean"),
                    "dead_experts_count": router.get("router_agg/dead_experts_count"),
                },
            }
        )
    return out


def _write_json(path: Path, obj: Any) -> None:
    def _normalize(node: Any) -> Any:
        if isinstance(node, dict):
            return {str(k): _normalize(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_normalize(v) for v in node]
        return _normalize_value(node)

    path.write_text(json.dumps(_normalize(obj), indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export autoresearch campaign receipt summaries for blog artifacts.")
    ap.add_argument("--receipt-dir", required=True, help="Directory containing campaign receipt JSON files")
    ap.add_argument("--output-dir", required=True, help="Output directory for generated JSON artifacts")
    ap.add_argument("--post", required=True, help="Post number prefix, e.g. 0011")
    ap.add_argument("--campaign", required=True, help="Campaign name to embed in exported metadata")
    ap.add_argument("--nviz-base-url", default="", help="Optional NVIZ base URL, e.g. http://127.0.0.1:3001")
    ap.add_argument("--timeout-s", type=float, default=10.0, help="HTTP timeout for NVIZ requests")
    args = ap.parse_args()

    receipt_dir = Path(args.receipt_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    receipts = []
    for path in sorted(receipt_dir.glob("*.json")):
        obj = _load_receipt(path)
        if str(obj.get("campaign_name", "")).strip() != args.campaign:
            continue
        receipts.append(_receipt_row(obj))

    receipts.sort(key=_sort_key)
    completed = [row for row in receipts if row.get("status") == "completed"]
    progression = _progression_rows(completed)
    manifest = _candidate_manifest(receipts, completed)
    run_summaries = _run_summary_rows(
        completed,
        nviz_base_url=str(args.nviz_base_url).strip() or None,
        timeout_s=float(args.timeout_s),
    )

    prefix = f"{args.post}_autoresearch"
    _write_json(output_dir / f"{prefix}_receipts.json", {"campaign": args.campaign, "receipts": receipts})
    _write_json(output_dir / f"{prefix}_progression.json", {"campaign": args.campaign, "progression": progression})
    _write_json(output_dir / f"{prefix}_run_summaries.json", {"campaign": args.campaign, "runs": run_summaries})
    _write_json(output_dir / f"{prefix}_manifest.json", {"campaign": args.campaign, **manifest})
    print(f"wrote autoresearch artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
