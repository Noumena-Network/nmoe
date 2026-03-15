#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BuildConfig:
    data_root: Path
    canonical_data_root: str
    repo_root: Path
    out_dir: Path
    post: str


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing blog artifacts manifest: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"manifest is not an object: {path}")
    return obj


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _manifest_entry(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    entry = manifest.get(key, {})
    if not isinstance(entry, dict):
        raise TypeError(f"manifest entry must be an object: {key}")
    return entry


def _manifest_files_runs(manifest: dict[str, Any], key: str) -> tuple[list[str], dict[str, Any]]:
    entry = _manifest_entry(manifest, key)
    files = [str(x) for x in entry.get("files", [])]
    runs = entry.get("runs", {})
    if not isinstance(runs, dict):
        runs = {}
    return sorted(set(files)), runs


def _blog_artifact_entries(files: list[str], cfg: BuildConfig) -> list[dict[str, str]]:
    return [
        {
            "kind": "blog_artifact_json",
            "path": f"blog_artifacts/{name}",
        }
        for name in files
    ]


def _site_figure_entries(figure_rels: list[str], cfg: BuildConfig) -> list[dict[str, str]]:
    return [
        {
            "kind": "site_figure",
            "path": f"static/figures/{rel}",
            "logical_path": f"/figures/{rel}",
        }
        for rel in figure_rels
    ]


def _paper_artifact_entries(paths: list[str]) -> list[dict[str, str]]:
    return [{"kind": "paper_reference", "path": path} for path in paths]


def _logical_data_path(raw: str, cfg: BuildConfig) -> str:
    s = str(raw)
    for prefix in (str(cfg.canonical_data_root).rstrip("/"), "/data"):
        if prefix and s.startswith(prefix + "/"):
            return s[len(prefix) + 1 :]
    return s.lstrip("/")


def _build_0002(manifest: dict[str, Any], cfg: BuildConfig) -> dict[str, Any]:
    files, runs = _manifest_files_runs(manifest, "0002_measurement")
    files = sorted(
        set(
            files
            + [
                "0004_miniseries_ms_20260112_084552_core_summaries.json",
            ]
        )
    )
    figures = [
        "eval_loop.svg",
        "miniseries_bpb_core_example.svg",
        "router_health_healthy_vs_collapsed.svg",
        "super4096_loss_router.svg",
    ]
    return {
        "post": "0002",
        "title": "Make It Measurable",
        "evidence_tier": "external_repro",
        "status": "complete_for_scope",
        "generated_at_utc": _now_utc(),
        "summary": (
            "Measurement contract receipts: healthy vs collapsed router telemetry and "
            "a three-panel bpb/CORE/router-health example with task-bootstrap CORE variance inputs."
        ),
        "runs": runs,
        "artifacts": _blog_artifact_entries(files, cfg) + _site_figure_entries(figures, cfg),
        "commands": [
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/moe.toml --dtype=bf16 --steps=9536",
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/super.toml --dtype=bf16 --steps=12000",
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe_super.toml --dtype=bf16 --steps=12000",
        ],
        "notes": [
            "0002 three-panel slice references miniseries run ms_20260112_084552_d12 from the manifest and the corresponding Super-256 contract.",
            "Task-bootstrap CORE variance uses task-level centered scores from 0004_miniseries_ms_20260112_084552_core_summaries.json.",
            "The Super-4096 collapse figure in this receipt is a separate stress-test surface from configs/speedrun/small_moe_super.toml.",
            "Lockstep failure handling is documented with the current fail-loud runtime guard in nmoe.eval.core.runner.",
        ],
    }



def _build_0003(manifest: dict[str, Any], cfg: BuildConfig) -> dict[str, Any]:
    files, runs = _manifest_files_runs(manifest, "0003_calibration")
    files.extend(
        [
            "0003_dense_sdpa_throughput.json",
            "0003_moe64_bf16_throughput.json",
            "0003_ultra256_bf16_throughput.json",
            "manifest.json",
        ]
    )
    files = sorted(set(files))
    artifacts = _blog_artifact_entries(files, cfg)
    artifacts.extend(
        _site_figure_entries(
            [
                "dense_calibration_curve.svg",
                "speedrun_dense_moe_ultra_loss.svg",
                "speedrun_throughput_overlay.svg",
                "speedrun_closure_matrix.svg",
                "tokens_per_routed_expert.svg",
            ],
            cfg,
        )
    )
    artifacts.extend(
        [
            {"kind": "config_toml", "path": "configs/speedrun/small_dense_sdpa.toml"},
            {"kind": "config_toml", "path": "configs/speedrun/small_moe.toml"},
            {"kind": "config_toml", "path": "configs/speedrun/small_moe_ultra.toml"},
            {"kind": "metrics_duckdb", "path": "metrics/run_1767924490_732716/rank_0.duckdb"},
            {"kind": "metrics_duckdb", "path": "metrics/run_1767932268_736761/rank_0.duckdb"},
            {"kind": "metrics_duckdb", "path": "metrics/run_1767976973_748979/rank_0.duckdb"},
            {"kind": "cluster_sqlite", "path": "blog_artifacts/0003_closure_20260312_bf16/experiments_bf16.db"},
            {"kind": "cluster_sqlite", "path": "blog_artifacts/0003_closure_20260312_fp8/experiments_fp8.db"},
            {"kind": "cluster_sqlite", "path": "blog_artifacts/0003_closure_20260312_nvfp4/experiments_nvfp4.db"},
            {"kind": "cluster_log", "path": "blog_artifacts/0003_closure_20260312_bf16/logs/dense.log"},
            {"kind": "cluster_log", "path": "blog_artifacts/0003_closure_20260312_bf16/logs/moe64.log"},
            {"kind": "cluster_log", "path": "blog_artifacts/0003_closure_20260312_bf16/logs/moe256.log"},
            {"kind": "cluster_log", "path": "blog_artifacts/0003_closure_20260312_fp8/logs/dense.log"},
            {"kind": "cluster_log", "path": "blog_artifacts/0003_closure_20260312_fp8/logs/moe64.log"},
            {"kind": "cluster_log", "path": "blog_artifacts/0003_closure_20260312_fp8/logs/moe256.log"},
            {"kind": "cluster_log", "path": "blog_artifacts/0003_closure_20260312_nvfp4/logs/dense.log"},
            {"kind": "cluster_log", "path": "blog_artifacts/0003_closure_20260312_nvfp4/logs/moe64.log"},
            {"kind": "cluster_log", "path": "blog_artifacts/0003_closure_20260312_nvfp4/logs/moe256.log"},
        ]
    )
    base = json.loads(r"""{
    "post": "0003",
    "title": "The Speedrun Loop",
    "evidence_tier": "internal_repro",
    "status": "complete",
    "generated_at_utc": "",
    "summary": "Speedrun-loop receipts: the pinned public June 2024 dense anchor, the historical January calibration and throughput surfaces, and the completed March 2026 loss-plus-CORE closure matrix across dense / MoE-64 / MoE-256 and bf16 / fp8 / nvfp4.",
    "runs": {},
    "external_anchor": {
        "repo": "KellerJordan/modded-nanogpt",
        "recipe_commit": "b6b0a0d36e6f1758a8d14d5fcd5f15ca5d19b891",
        "public_record_ref": "HEAD",
        "public_record_dir": "records/track_1_short/2024-06-06_AdamW",
        "public_readme_path": "records/track_1_short/2024-06-06_AdamW/README.md",
        "public_log_path": "records/track_1_short/2024-06-06_AdamW/f66d43d7-e449-4029-8adf-e8537bab49ea.log",
        "anchor_step": 9536,
        "anchor_metric": "tel",
        "anchor_valid_loss": 3.275959,
        "public_recipe_note": "lr=0.0018, warmup=250, warmdown=2000, betas=(0.9, 0.95) IIRC",
        "note": "The code commit gives the June 2024 recipe neighborhood; the preserved public log gives the exact numerical anchor used in the post."
    },
    "comparison_contract": {
        "clean_pair": {
            "models": [
                "dense_sdpa",
                "moe64_bf16"
            ],
            "steps": 9536,
            "tokens_per_step": 524288,
            "batch_size": 256,
            "seq_len": 2048,
            "optimizer": "AdamW",
            "lr": 0.0018,
            "weight_decay": 0.1,
            "betas": [
                0.9,
                0.95
            ],
            "swap_axis": "active_ffn_width_per_token",
            "active_width_dense": 3072,
            "active_width_moe64": 3072
        },
        "exploratory_slice": {
            "model": "ultra256_bf16",
            "published_step": 10752,
            "run_contract_steps": 32800,
            "scope": "longer iso-tokens/expert run; published point is exploratory rather than part of the clean 9536-step pair"
        },
        "closure_matrix_20260312": {
            "models": [
                "dense",
                "moe64",
                "moe256"
            ],
            "dtypes": [
                "bf16",
                "fp8",
                "nvfp4"
            ],
            "steps_max": 9536,
            "tokens_per_step": 524288,
            "batch_size": 256,
            "seq_len": 2048,
            "optimizer": "AdamW",
            "lr": 0.0018,
            "weight_decay": 0.1,
            "betas": [
                0.9,
                0.95
            ],
            "warmup_steps": 256,
            "warmdown_steps": 2048,
            "target_loss": 3.28,
            "eval_enabled": true,
            "eval_tasks": "core",
            "swap_axis": "active_ffn_width_per_token",
            "active_width_dense": 3072,
            "active_width_moe64": 3072,
            "active_width_moe256": 3072,
            "note": "dense uses small_dense_sdpa.toml; MoE-64 uses small_moe.toml with dtype overrides; MoE-256 uses small_moe_ultra.toml overridden down to the fixed 9536-step June-style horizon."
        }
    },
    "measurements": {
        "public_anchor": {
            "step": 9536,
            "valid_loss": 3.275959,
            "source": "external_anchor.public_log_path"
        },
        "dense_sdpa": {
            "step": 9536,
            "valid_loss_exact": 3.4806723594665527,
            "valid_loss_post_rounded": 3.4807,
            "source_artifacts": [
                "blog_artifacts/0003_dense_sdpa_loss.json",
                "metrics/run_1767924490_732716/rank_0.duckdb",
                "static/figures/dense_calibration_curve.svg"
            ]
        },
        "gap_vs_public_anchor": {
            "step": 9536,
            "delta_valid_loss_exact": 0.2047133594665529,
            "delta_valid_loss_post_rounded": 0.205
        },
        "moe64_bf16": {
            "step": 9536,
            "valid_loss_exact": 3.431931734085083,
            "valid_loss_post_rounded": 3.4319,
            "delta_vs_dense_exact": -0.04874062538146973,
            "delta_vs_dense_post_rounded": -0.049,
            "router_mean_cv_exact": 86.99082842740145,
            "router_mean_entropy_exact": 3.9127983179959385,
            "source_artifacts": [
                "blog_artifacts/0003_moe64_bf16_loss.json",
                "blog_artifacts/0003_moe64_bf16_router_cv.json",
                "blog_artifacts/0003_moe64_bf16_router_entropy.json",
                "metrics/run_1767932268_736761/rank_0.duckdb",
                "static/figures/speedrun_dense_moe_ultra_loss.svg"
            ]
        },
        "ultra256_bf16_slice": {
            "step": 10752,
            "valid_loss_exact": 3.2799384593963623,
            "valid_loss_post_rounded": 3.2799,
            "delta_vs_dense_exact": -0.20073390007019043,
            "delta_vs_dense_post_rounded": -0.201,
            "scope": "exploratory_longer_run_slice",
            "source_artifacts": [
                "blog_artifacts/0003_ultra256_bf16_loss.json",
                "metrics/run_1767976973_748979/rank_0.duckdb",
                "static/figures/speedrun_dense_moe_ultra_loss.svg"
            ]
        },
        "throughput_medians_steps_1000_7000": {
            "window": {
                "step_start": 1000,
                "step_end": 7000,
                "count": 61
            },
            "dense_sdpa": {
                "tokens_per_s_gpu_median": 93019.88557814731,
                "tokens_per_s_gpu_avg": 93794.16962578647,
                "ms_per_step_median": 704.5375254191458,
                "tflops_median": 137.4484451738877
            },
            "moe64_bf16": {
                "tokens_per_s_gpu_median": 47929.62507283819,
                "tokens_per_s_gpu_avg": 48056.1098734755,
                "ms_per_step_median": 1367.3380482406353,
                "tflops_median": 70.97746489359179
            },
            "ultra256_bf16": {
                "tokens_per_s_gpu_median": 46671.23102833263,
                "tokens_per_s_gpu_avg": 47345.79646444087,
                "ms_per_step_median": 1404.2055149609225,
                "tflops_median": 69.62321350388423
            },
            "source_artifacts": [
                "blog_artifacts/0003_dense_sdpa_throughput.json",
                "blog_artifacts/0003_moe64_bf16_throughput.json",
                "blog_artifacts/0003_ultra256_bf16_throughput.json",
                "metrics/run_1767924490_732716/rank_0.duckdb",
                "metrics/run_1767932268_736761/rank_0.duckdb",
                "metrics/run_1767976973_748979/rank_0.duckdb",
                "static/figures/speedrun_throughput_overlay.svg"
            ]
        },
        "model_param_counts": {
            "source": "derived_from_configs_and_nmoe.metrics.param_counts",
            "dense": {
                "params_total": 190532352,
                "params_active": 190532352
            },
            "moe64": {
                "params_total": 755534592,
                "params_active": 191073024
            },
            "moe256": {
                "params_total": 2836482816,
                "params_active": 192891648
            },
            "9536_step_contract": {
                "tokens_seen": 4999610368,
                "dense_tokens_per_total_param": 26.24021755633395,
                "dense_tokens_per_active_param": 26.24021755633395,
                "moe64_tokens_per_total_param": 6.617314805717776,
                "moe64_tokens_per_active_param": 26.16596729343528,
                "moe256_tokens_per_total_param": 1.7626085459772767,
                "moe256_tokens_per_active_param": 25.91926776581965
            }
        },
        "tokens_per_routed_expert": {
            "formula": "tokens_seen * K / E",
            "moe64_full_horizon": {
                "tokens_seen": 4999610368,
                "K": 6,
                "E": 64,
                "tokens_per_expert": 468713472.0
            },
            "moe256_full_horizon": {
                "tokens_seen": 4999610368,
                "K": 7,
                "E": 256,
                "tokens_per_expert": 136708916.0
            },
            "super4096_reference": {
                "tokens_seen_approx": 6329000000,
                "K": 7,
                "E": 4096,
                "tokens_per_expert_approx": 10800000.0
            }
        },
        "closure_matrix_20260312": {
            "contract": {
                "models": [
                    "dense",
                    "moe64",
                    "moe256"
                ],
                "dtypes": [
                    "bf16",
                    "fp8",
                    "nvfp4"
                ],
                "steps_max": 9536,
                "tokens_per_step": 524288,
                "batch_size": 256,
                "seq_len": 2048,
                "optimizer": "AdamW",
                "lr": 0.0018,
                "weight_decay": 0.1,
                "betas": [
                    0.9,
                    0.95
                ],
                "warmup_steps": 256,
                "warmdown_steps": 2048,
                "target_loss": 3.28,
                "eval_enabled": true,
                "eval_tasks": "core",
                "swap_axis": "active_ffn_width_per_token",
                "active_width_dense": 3072,
                "active_width_moe64": 3072,
                "active_width_moe256": 3072,
                "note": "dense uses small_dense_sdpa.toml; MoE-64 uses small_moe.toml with dtype overrides; MoE-256 uses small_moe_ultra.toml overridden down to the fixed 9536-step June-style horizon."
            },
            "results": {
                "bf16": {
                    "dense": {
                        "stop_step": 8320,
                        "stop_kind": "target_reached",
                        "final_valid_loss": 3.2725,
                        "core": 0.060865,
                        "tokens_seen": 4362076160,
                        "train_time_ms": 1310950,
                        "step_avg_ms": 157.57
                    },
                    "moe64": {
                        "stop_step": 5760,
                        "stop_kind": "target_reached",
                        "final_valid_loss": 3.2769,
                        "core": 0.050558,
                        "tokens_seen": 3019898880,
                        "train_time_ms": 3115985,
                        "step_avg_ms": 540.97
                    },
                    "moe256": {
                        "stop_step": 4864,
                        "stop_kind": "target_reached",
                        "final_valid_loss": 3.2778,
                        "core": 0.051878,
                        "tokens_seen": 2550136832,
                        "train_time_ms": 3565374,
                        "step_avg_ms": 733.01
                    }
                },
                "fp8": {
                    "dense": {
                        "stop_step": 8320,
                        "stop_kind": "target_reached",
                        "final_valid_loss": 3.2758,
                        "core": 0.057261,
                        "tokens_seen": 4362076160,
                        "train_time_ms": 1304489,
                        "step_avg_ms": 156.79
                    },
                    "moe64": {
                        "stop_step": 6272,
                        "stop_kind": "target_reached",
                        "final_valid_loss": 3.2684,
                        "core": 0.060741,
                        "tokens_seen": 3288334336,
                        "train_time_ms": 3044978,
                        "step_avg_ms": 485.49
                    },
                    "moe256": {
                        "stop_step": 4864,
                        "stop_kind": "target_reached",
                        "final_valid_loss": 3.2796,
                        "core": 0.070765,
                        "tokens_seen": 2550136832,
                        "train_time_ms": 3371195,
                        "step_avg_ms": 693.09
                    }
                },
                "nvfp4": {
                    "dense": {
                        "stop_step": 9536,
                        "stop_kind": "completed_full_horizon",
                        "final_valid_loss": 3.3047,
                        "core": 0.04943,
                        "tokens_seen": 4999610368,
                        "target_reached": false
                    },
                    "moe64": {
                        "stop_step": 9536,
                        "stop_kind": "completed_full_horizon",
                        "final_valid_loss": 3.595,
                        "core": 0.030931,
                        "tokens_seen": 4999610368,
                        "target_reached": false
                    },
                    "moe256": {
                        "stop_step": 9536,
                        "stop_kind": "completed_full_horizon",
                        "final_valid_loss": 3.4573,
                        "core": 0.04905,
                        "tokens_seen": 4999610368,
                        "target_reached": false
                    }
                }
            }
        }
    },
    "artifacts": [],
    "commands": [
        "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_dense_sdpa.toml --dtype=bf16 --eval_enabled=true --eval_tasks=core",
        "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_dense_sdpa.toml --dtype=fp8 --eval_enabled=true --eval_tasks=core",
        "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_dense_sdpa.toml --dtype=nvfp4 --eval_enabled=true --eval_tasks=core",
        "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe.toml --dtype=bf16 --eval_enabled=true --eval_tasks=core",
        "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe.toml --dtype=fp8 --eval_enabled=true --eval_tasks=core",
        "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe.toml --dtype=nvfp4 --eval_enabled=true --eval_tasks=core",
        "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe_ultra.toml --dtype=bf16 --steps=9536 --warmup_steps=256 --hold_tokens=3925721344 --decay_tokens=1073741824 --eval_enabled=true --eval_tasks=core",
        "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe_ultra.toml --dtype=fp8 --steps=9536 --warmup_steps=256 --hold_tokens=3925721344 --decay_tokens=1073741824 --eval_enabled=true --eval_tasks=core",
        "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe_ultra.toml --dtype=nvfp4 --steps=9536 --warmup_steps=256 --hold_tokens=3925721344 --decay_tokens=1073741824 --eval_enabled=true --eval_tasks=core"
    ],
    "provenance_commands": [
        "git clone --filter=blob:none --no-checkout https://github.com/KellerJordan/modded-nanogpt.git /tmp/modded-nanogpt-0003",
        "cd /tmp/modded-nanogpt-0003 && git show HEAD:records/track_1_short/2024-06-06_AdamW/f66d43d7-e449-4029-8adf-e8537bab49ea.log | rg 's:9536 tel:'",
        "kubectl logs -n default nmoe-0003-closure-bf16-ntlwn",
        "kubectl logs -n default nmoe-0003-closure-fp8-mwrjj",
        "kubectl logs -n default nmoe-0003-closure-nvfp4-njq8f"
    ],
    "notes": [
        "This receipt package keeps both layers of the story: the historical January calibration lanes that established the anchor-gap doctrine and the March 2026 closure matrix that upgrades the loop with end-of-run CORE.",
        "The public anchor is split into a recipe neighborhood (June 7 2024 commit b6b0a0d) and an exact numeric log anchor (the current public June 2024 record path at HEAD).",
        "The historical dense lane is still the cleanest statement of the original +0.205 honesty boundary at step 9536.",
        "The closure matrix uses the public anchor operationally via target_loss=3.28 rather than only as a post-hoc comparison.",
        "MoE-256 in the closure matrix is a dedicated fixed-horizon override of small_moe_ultra.toml, not the older longer iso-tokens/expert slice.",
        "bf16 and fp8 close cleanly across all three architectures; nvfp4 does not reach target in any lane, but MoE-256 is materially healthier than MoE-64 on both loss and CORE.",
        "The throughput medians in this post still come from the earlier historical bf16 surface; the new closure matrix is the loss-plus-CORE upgrade.",
        "Validate repo-local paths with verify_post_receipts.py. Full output-path validation still requires --check-paths --check-outputs in a data-mounted environment."
    ],
    "cluster_receipts": {
        "experiments_db_path": "/data/experiments.db",
        "blog_artifact_manifest_path": "blog_artifacts/manifest.json",
        "rank_0_duckdbs": {
            "dense_sdpa": "metrics/run_1767924490_732716/rank_0.duckdb",
            "moe64_bf16": "metrics/run_1767932268_736761/rank_0.duckdb",
            "ultra256_bf16": "metrics/run_1767976973_748979/rank_0.duckdb"
        },
        "eval_outputs_root": "/data/eval/outputs",
        "closure_matrix_roots": {
            "bf16": "blog_artifacts/0003_closure_20260312_bf16",
            "fp8": "blog_artifacts/0003_closure_20260312_fp8",
            "nvfp4": "blog_artifacts/0003_closure_20260312_nvfp4"
        },
        "closure_matrix_logs": {
            "bf16": {
                "dense": "blog_artifacts/0003_closure_20260312_bf16/logs/dense.log",
                "moe64": "blog_artifacts/0003_closure_20260312_bf16/logs/moe64.log",
                "moe256": "blog_artifacts/0003_closure_20260312_bf16/logs/moe256.log"
            },
            "fp8": {
                "dense": "blog_artifacts/0003_closure_20260312_fp8/logs/dense.log",
                "moe64": "blog_artifacts/0003_closure_20260312_fp8/logs/moe64.log",
                "moe256": "blog_artifacts/0003_closure_20260312_fp8/logs/moe256.log"
            },
            "nvfp4": {
                "dense": "blog_artifacts/0003_closure_20260312_nvfp4/logs/dense.log",
                "moe64": "blog_artifacts/0003_closure_20260312_nvfp4/logs/moe64.log",
                "moe256": "blog_artifacts/0003_closure_20260312_nvfp4/logs/moe256.log"
            }
        }
    },
    "run_db_summaries": {
        "dense_sdpa": {
            "experiment_id": "speedrun_small_dense_sdpa_june2024",
            "started_at": "2026-01-09 02:08:10",
            "ended_at": "2026-01-09 03:58:39",
            "status": "completed",
            "steps_completed": 9536,
            "tokens_seen": 4999610368,
            "stop_reason": "completed",
            "target_reached": false,
            "train_time_ms_excl_valid": 6279269.354513381,
            "valid_time_ms": 342495.6297276076,
            "eval_enabled": false,
            "eval_every": 0,
            "eval_tasks": "core"
        },
        "moe64_bf16": {
            "experiment_id": "speedrun_small_moe_june2024",
            "started_at": "2026-01-09 04:17:48",
            "ended_at": "2026-01-09 07:58:02",
            "status": "completed",
            "steps_completed": 9536,
            "tokens_seen": 4999610368,
            "stop_reason": "completed",
            "target_reached": false,
            "train_time_ms_excl_valid": 12619302.91609536,
            "valid_time_ms": 585985.7119314838,
            "eval_enabled": false,
            "eval_every": 0,
            "eval_tasks": "core"
        },
        "ultra256_bf16": {
            "experiment_id": "speedrun_small_moe_ultra_isotpe",
            "started_at": "2026-01-09 16:42:53",
            "ended_at": "2026-01-09 21:02:38",
            "status": "completed_target",
            "steps_completed": 10752,
            "tokens_seen": 5637144576,
            "stop_reason": "target_reached",
            "target_reached": true,
            "val_loss_to_target": 3.2799384593963623,
            "train_time_ms_excl_valid": 14860339.937474346,
            "valid_time_ms": 714863.5901906528,
            "eval_enabled": false,
            "eval_every": 0,
            "eval_tasks": "core"
        },
        "closure_matrix_20260312": {
            "bf16": {
                "dense": {
                    "stop_step": 8320,
                    "stop_kind": "target_reached",
                    "final_valid_loss": 3.2725,
                    "core": 0.060865,
                    "tokens_seen": 4362076160,
                    "train_time_ms": 1310950,
                    "step_avg_ms": 157.57
                },
                "moe64": {
                    "stop_step": 5760,
                    "stop_kind": "target_reached",
                    "final_valid_loss": 3.2769,
                    "core": 0.050558,
                    "tokens_seen": 3019898880,
                    "train_time_ms": 3115985,
                    "step_avg_ms": 540.97
                },
                "moe256": {
                    "stop_step": 4864,
                    "stop_kind": "target_reached",
                    "final_valid_loss": 3.2778,
                    "core": 0.051878,
                    "tokens_seen": 2550136832,
                    "train_time_ms": 3565374,
                    "step_avg_ms": 733.01
                }
            },
            "fp8": {
                "dense": {
                    "stop_step": 8320,
                    "stop_kind": "target_reached",
                    "final_valid_loss": 3.2758,
                    "core": 0.057261,
                    "tokens_seen": 4362076160,
                    "train_time_ms": 1304489,
                    "step_avg_ms": 156.79
                },
                "moe64": {
                    "stop_step": 6272,
                    "stop_kind": "target_reached",
                    "final_valid_loss": 3.2684,
                    "core": 0.060741,
                    "tokens_seen": 3288334336,
                    "train_time_ms": 3044978,
                    "step_avg_ms": 485.49
                },
                "moe256": {
                    "stop_step": 4864,
                    "stop_kind": "target_reached",
                    "final_valid_loss": 3.2796,
                    "core": 0.070765,
                    "tokens_seen": 2550136832,
                    "train_time_ms": 3371195,
                    "step_avg_ms": 693.09
                }
            },
            "nvfp4": {
                "dense": {
                    "stop_step": 9536,
                    "stop_kind": "completed_full_horizon",
                    "final_valid_loss": 3.3047,
                    "core": 0.04943,
                    "tokens_seen": 4999610368,
                    "target_reached": false
                },
                "moe64": {
                    "stop_step": 9536,
                    "stop_kind": "completed_full_horizon",
                    "final_valid_loss": 3.595,
                    "core": 0.030931,
                    "tokens_seen": 4999610368,
                    "target_reached": false
                },
                "moe256": {
                    "stop_step": 9536,
                    "stop_kind": "completed_full_horizon",
                    "final_valid_loss": 3.4573,
                    "core": 0.04905,
                    "tokens_seen": 4999610368,
                    "target_reached": false
                }
            }
        }
    },
    "capability_surface_status": {
        "historical_published_lanes": {
            "dense_sdpa": {
                "eval_enabled": false,
                "eval_tasks": "core",
                "core_metrics_present": false
            },
            "moe64_bf16": {
                "eval_enabled": false,
                "eval_tasks": "core",
                "core_metrics_present": false
            },
            "ultra256_bf16": {
                "eval_enabled": false,
                "eval_tasks": "core",
                "core_metrics_present": false
            }
        },
        "closure_matrix_20260312": {
            "dense_bf16": {
                "eval_enabled": true,
                "eval_tasks": "core",
                "core_metrics_present": true
            },
            "moe64_bf16": {
                "eval_enabled": true,
                "eval_tasks": "core",
                "core_metrics_present": true
            },
            "moe256_bf16": {
                "eval_enabled": true,
                "eval_tasks": "core",
                "core_metrics_present": true
            },
            "dense_fp8": {
                "eval_enabled": true,
                "eval_tasks": "core",
                "core_metrics_present": true
            },
            "moe64_fp8": {
                "eval_enabled": true,
                "eval_tasks": "core",
                "core_metrics_present": true
            },
            "moe256_fp8": {
                "eval_enabled": true,
                "eval_tasks": "core",
                "core_metrics_present": true
            },
            "dense_nvfp4": {
                "eval_enabled": true,
                "eval_tasks": "core",
                "core_metrics_present": true
            },
            "moe64_nvfp4": {
                "eval_enabled": true,
                "eval_tasks": "core",
                "core_metrics_present": true
            },
            "moe256_nvfp4": {
                "eval_enabled": true,
                "eval_tasks": "core",
                "core_metrics_present": true
            }
        },
        "note": "The historical January calibration lanes proved why the speedrun objective had to grow beyond loss. The March 2026 closure matrix is the upgraded eval-enabled surface the post now relies on."
    }
}""")
    base["generated_at_utc"] = _now_utc()
    base["runs"] = runs
    base["artifacts"] = artifacts
    return base

def _build_0004(manifest: dict[str, Any], cfg: BuildConfig) -> dict[str, Any]:
    entry = _manifest_entry(manifest, "0004_miniseries")
    files = [str(x) for x in entry.get("files", [])]
    source = entry.get("source", {})
    if not isinstance(source, dict):
        source = {}

    series_dir = "miniseries/ms_20260112_084552"
    artifacts = _blog_artifact_entries(files, cfg)
    artifacts.extend(
        [
            {"kind": "miniseries_plan", "path": f"{series_dir}/plan.json"},
            {"kind": "miniseries_config", "path": f"{series_dir}/d10.toml"},
            {"kind": "miniseries_config", "path": f"{series_dir}/d12.toml"},
            {
                "kind": "miniseries_results_csv",
                "path": _logical_data_path(
                    str(source.get("results_csv", f"/data/{series_dir}/results.csv")),
                    cfg,
                ),
            },
            {
                "kind": "metrics_duckdb",
                "path": _logical_data_path(
                    str(source.get("metrics_d10", "/data/metrics/ms_20260112_084552_d10/rank_0.duckdb")),
                    cfg,
                ),
            },
            {
                "kind": "metrics_duckdb",
                "path": _logical_data_path(
                    str(source.get("metrics_d12", "/data/metrics/ms_20260112_084552_d12/rank_0.duckdb")),
                    cfg,
                ),
            },
        ]
    )
    return {
        "post": "0004",
        "title": "The #420 Transfer to MoE",
        "evidence_tier": "internal_repro",
        "status": "partial",
        "generated_at_utc": _now_utc(),
        "summary": (
            "Methodology receipts for the published d10/d12 MoE transfer slice: the pinned token-indexed contract, "
            "the generated configs, the exported checkpoint rows and CORE summaries, the source metrics, and the "
            "figure provenance for the current two-point slice."
        ),
        "runs": {
            "d10": "ms_20260112_084552_d10",
            "d12": "ms_20260112_084552_d12",
        },
        "artifacts": artifacts + _site_figure_entries(["miniseries_bpb_core_example.svg"], cfg),
        "commands": [
            "torchrun --nproc_per_node=8 -m nmoe.train /data/miniseries/ms_20260112_084552/d10.toml",
            "torchrun --nproc_per_node=8 -m nmoe.train /data/miniseries/ms_20260112_084552/d12.toml",
            "python3 scripts/repro/verify_post_receipts.py --repo-root . --receipts-dir repro --post 0004",
        ],
        "notes": [
            "plan.json records the pinned slice contract: base_config=configs/speedrun/small_moe_ultra.toml, checkpoints={2%,5%,10%,20%}, target_param_data_ratio=8.0, warmup_ratio=0.01, warmdown_ratio=0.4, tokens_per_step=524288.",
            "The generated d10.toml/d12.toml files are the exact run contracts for the current published slice, but they currently live in the artifact root rather than the repo tree.",
            "0004_miniseries_ms_20260112_084552_results.json backs the published checkpoint rows for tokens, valid/bpb, CORE, and mean CV.",
            "The published min E_eff column is derived from the same exported result rows as exp(router_agg/min_entropy).",
            "This receipt package backs the current methodological claim only: the token-indexed #420 contract transfers cleanly enough to produce an interpretable MoE slice. It does not isolate depth from sparsity, and it does not include the dense control or failure/corrected comparison yet.",
        ],
    }


def _build_0005(_: dict[str, Any], cfg: BuildConfig) -> dict[str, Any]:
    return {
        "post": "0005",
        "title": "NVFP4 Dynamics",
        "evidence_tier": "external_repro",
        "status": "complete_for_scope",
        "generated_at_utc": _now_utc(),
        "summary": (
            "Corrected 0005 receipts: a 384-step gain-isolation quartet proving fp4_embed_gain helps and "
            "fp4_logits_gain=0.125 hurts, plus the corrected 9536-step embed-only+adamw rerun that finishes "
            "at 3.1735 versus the bf16 baseline 3.1264, leaving a late-window residual of about +0.046."
        ),
        "runs": {
            "bf16_baseline": "run_1773277673_591",
            "gain_unit_v4": "run_1773366025_100249",
            "gain_embed_v4": "run_1773366550_103111",
            "gain_logits_v4": "run_1773367081_106017",
            "gain_old_pair_v4": "run_1773367618_108920",
            "embed_only_adamw_v2": "run_1773368195_111994",
        },
        "artifacts": [
            {"kind": "config_contract", "path": "configs/speedrun/moe.toml"},
            {"kind": "training_source", "path": "nmoe/train.py"},
            {"kind": "training_source", "path": "nmoe/model.py"},
            {"kind": "training_source", "path": "nmoe/moe.py"},
            {"kind": "experiments_db", "path": "blog_artifacts/0005_nvfp4_gain_isolation_384_v4_20260313/experiments.db"},
            {"kind": "experiments_db", "path": "blog_artifacts/0005_nvfp4_embed_only_adamw_long_v2_20260313/experiments.db"},
            {"kind": "step_metrics_parquet", "path": "blog_artifacts/0005_nvfp4_gain_isolation_384_v4_20260313/metrics/run_1773366025_100249/step_00000384.parquet"},
            {"kind": "step_metrics_parquet", "path": "blog_artifacts/0005_nvfp4_gain_isolation_384_v4_20260313/metrics/run_1773366550_103111/step_00000384.parquet"},
            {"kind": "step_metrics_parquet", "path": "blog_artifacts/0005_nvfp4_gain_isolation_384_v4_20260313/metrics/run_1773367081_106017/step_00000384.parquet"},
            {"kind": "step_metrics_parquet", "path": "blog_artifacts/0005_nvfp4_gain_isolation_384_v4_20260313/metrics/run_1773367618_108920/step_00000384.parquet"},
            {"kind": "step_metrics_parquet", "path": "blog_artifacts/0005_nvfp4_embed_only_adamw_long_v2_20260313/metrics/run_1773368195_111994/step_00003072.parquet"},
            {"kind": "step_metrics_parquet", "path": "blog_artifacts/0005_nvfp4_embed_only_adamw_long_v2_20260313/metrics/run_1773368195_111994/step_00004096.parquet"},
            {"kind": "step_metrics_parquet", "path": "blog_artifacts/0005_nvfp4_embed_only_adamw_long_v2_20260313/metrics/run_1773368195_111994/step_00009536.parquet"},
            {"kind": "step_metrics_parquet", "path": "blog_artifacts/0005_nvfp4_matrix_f20260312/metrics/run_1773277673_591/step_00009536.parquet"},
        ],
        "commands": [
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/moe.toml --dtype=bf16 --steps=9536",
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/moe.toml --dtype=nvfp4 --steps=384 --fp4_embed_gain=1.0 --fp4_logits_gain=1.0",
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/moe.toml --dtype=nvfp4 --steps=384 --fp4_embed_gain=10.667 --fp4_logits_gain=1.0",
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/moe.toml --dtype=nvfp4 --steps=384 --fp4_embed_gain=1.0 --fp4_logits_gain=0.125",
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/moe.toml --dtype=nvfp4 --steps=384 --fp4_embed_gain=10.667 --fp4_logits_gain=0.125",
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/moe.toml --dtype=nvfp4 --steps=9536 --fp4_embed_gain=10.667 --fp4_logits_gain=1.0 --expert_opt=adamw",
            "python3 scripts/repro/verify_post_receipts.py --repo-root . --receipts-dir repro --post 0005",
        ],
        "notes": [
            "Current code has two explicit NVFP4-only gains on this surface: fp4_embed_gain and fp4_logits_gain. This receipt does not promote a third NVFP4-only gain because none exists in the current tree.",
            "The corrected v4 quartet was rerun after syncing the blockscaled backward-profile replay fix in nmoe/moe.py; that semantic correction does not change the sign of the gain split.",
            "At 384 steps, embed-only beats unit by about 0.181 nats, logits-only is worse than unit by about 0.154 nats, and the historical pair remains worse than unit.",
            "The corrected embed-only+adamw line finishes at 3.173490 versus the bf16 baseline 3.126422. From steps 7808-9536, the residual stays in a tight ~+0.046 to +0.047 band.",
            "This receipt supports the narrowed current claim only: fp4_embed_gain is helpful, fp4_logits_gain=0.125 is harmful, the historical pair is net harmful, and the corrected embed-only+adamw line gets nvfp4 close to bf16 but does not yet close the remaining bug.",
        ],
    }


def _build_0006(_: dict[str, Any], cfg: BuildConfig) -> dict[str, Any]:
    baseline_root = "blog_artifacts/0006_super4096_rerun_r2_clean_20260313"
    aux_pair_root = "blog_artifacts/0006_super4096_auxclean_pair_r1_20260313"
    aux_long_root = "blog_artifacts/0006_super4096_bias0_aux1e4_long_r1_20260313"
    e1024_root = "blog_artifacts/0006_super1024_r1_20260313"
    e2048_root = "blog_artifacts/0006_super2048_long_r1_20260314"
    return {
        "post": "0006",
        "title": "Super-4096",
        "evidence_tier": "external_repro",
        "status": "partial",
        "generated_at_utc": _now_utc(),
        "summary": (
            "Corrected-stack Super-4096 receipts: trusted baseline collapse plus the clean aux-only "
            "falsifier and matched E=1024/E=2048 controls through the 2048-step window. Loss keeps "
            "improving while collapse remains severe; aux materially changes the regime; lowering E helps "
            "but does not by itself remove hard saturation."
        ),
        "runs": {
            "super4096_clean": "run_1773434558_119414",
            "super4096_aux1e4_bias0": "run_1773441395_7407",
            "super1024": "run_1773437178_465",
            "super2048": "run_1773455592_12991",
        },
        "artifacts": [
            {"kind": "config_contract", "path": "configs/speedrun/small_moe_super.toml"},
            {"kind": "training_source", "path": "nmoe/train.py"},
            {"kind": "training_source", "path": "nmoe/model.py"},
            {"kind": "experiments_db", "path": f"{baseline_root}/experiments.db"},
            {"kind": "experiments_db", "path": f"{aux_pair_root}/experiments.db"},
            {"kind": "experiments_db", "path": f"{aux_long_root}/experiments.db"},
            {"kind": "experiments_db", "path": f"{e1024_root}/experiments.db"},
            {"kind": "experiments_db", "path": f"{e2048_root}/experiments.db"},
            {"kind": "step_metrics_parquet", "path": f"{baseline_root}/metrics/run_1773434558_119414/step_00000100.parquet"},
            {"kind": "step_metrics_parquet", "path": f"{baseline_root}/metrics/run_1773434558_119414/step_00000400.parquet"},
            {"kind": "step_metrics_parquet", "path": f"{baseline_root}/metrics/run_1773434558_119414/step_00000700.parquet"},
            {"kind": "step_metrics_parquet", "path": f"{baseline_root}/metrics/run_1773434558_119414/step_00000512.parquet"},
            {"kind": "step_metrics_parquet", "path": f"{baseline_root}/metrics/run_1773434558_119414/step_00002000.parquet"},
            {"kind": "step_metrics_parquet", "path": f"{baseline_root}/metrics/run_1773434558_119414/step_00003600.parquet"},
            {"kind": "step_metrics_parquet", "path": f"{aux_long_root}/metrics/run_1773441395_7407/step_00000512.parquet"},
            {"kind": "step_metrics_parquet", "path": f"{aux_long_root}/metrics/run_1773441395_7407/step_00002000.parquet"},
            {"kind": "step_metrics_parquet", "path": f"{e1024_root}/metrics/run_1773437178_465/step_00000512.parquet"},
            {"kind": "step_metrics_parquet", "path": f"{e1024_root}/metrics/run_1773437178_465/step_00002000.parquet"},
            {"kind": "step_metrics_parquet", "path": f"{e2048_root}/metrics/run_1773455592_12991/step_00000512.parquet"},
            {"kind": "step_metrics_parquet", "path": f"{e2048_root}/metrics/run_1773455592_12991/step_00002000.parquet"},
        ],
        "commands": [
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe_super.toml --dtype=bf16 --steps=12000 --collect_update_stats=false",
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe_super.toml --dtype=bf16 --steps=2048 --router_bias_update_rate=0.0 --aux_loss_alpha=0.0001 --collect_update_stats=false",
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe_super.toml --dtype=bf16 --steps=2048 --n_routed_experts=1024 --collect_update_stats=false",
            "torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe_super.toml --dtype=bf16 --steps=2048 --n_routed_experts=2048 --collect_update_stats=false",
            "python3 scripts/repro/verify_post_receipts.py --repo-root . --receipts-dir repro --post 0006",
        ],
        "notes": [
            "Current 0006 scope uses the corrected-stack rerun and the matched 2048-step falsifier window, not the older January/March 3 baseline.",
            "The trusted baseline and controls were run with output probes, MoE grad-health scans, and update-stat collection disabled in-pod to match the historical overhead surface; that clean toggle is not yet a first-class public CLI flag.",
            "Clean aux-only (bias0, aux=1e-4) materially changes the regime: by step 2000 max_load is about 4.40% instead of about 13.42%, and valid/loss is 3.5728 instead of 3.6494.",
            "Lowering E to 2048 and 1024 lowers CV and improves entropy/loss, but high-index layers still pin near the same 1/K ceiling.",
            "Per-expert load histograms / CCDFs remain missing from this public bundle.",
        ],
    }


def _build_0007(_: dict[str, Any], cfg: BuildConfig) -> dict[str, Any]:
    return {
        "post": "0007",
        "title": "The Geometry Hypothesis",
        "evidence_tier": "paper_canonical",
        "status": "public_reference_surface",
        "generated_at_utc": _now_utc(),
        "summary": (
            "Public paper-PDF mirror for the atlas program: the Atlas Foundations paper served on the "
            "site. The internal paper source tree remains canonical but is not mirrored here."
        ),
        "artifacts": [
            {
                "kind": "paper_pdf",
                "path": "papers/atlas-foundations.pdf",
                "public_asset": "/papers/atlas-foundations.pdf",
            },
        ],
        "commands": [
            "ls papers/atlas-foundations.pdf",
        ],
        "notes": [
            "This public receipt mirrors only the PDFs exposed on the site; it does not enumerate the internal paper source tree, claim ledger, or run notes.",
            "The papers remain the primary truth boundary for the blog post's theorem / ontology claims.",
            "This is a public reference surface, not a matched-success intervention bundle.",
        ],
    }


def _build_0008(_: dict[str, Any], cfg: BuildConfig) -> dict[str, Any]:
    main_root = "blog_artifacts/0008_expert_lr_bf16_20260311"
    proof_root = "blog_artifacts/0008_expert_lr_bf16_updateproof_20260311"
    nvfp4_root = "blog_artifacts/0008_expert_lr_nvfp4_updateproof_b20260311"
    nvfp4_health_root = "blog_artifacts/0008_expert_lr_nvfp4_gradhealth_c20260311"
    main_runs = {
        "m0p5_s42": "run_1773264430_9484",
        "m1_s42": "run_1773261740_2498",
        "m2_s42": "run_1773264178_8727",
        "m4_s42": "run_1773262179_4211",
        "m15_s42": "run_1773261948_3435",
        "m0p5_s43": "run_1773264701_10969",
        "m1_s43": "run_1773262650_5765",
        "m2_s43": "run_1773264590_10264",
        "m4_s43": "run_1773263181_7207",
        "m15_s43": "run_1773262893_6407",
        "m1_b95_s42": "run_1773262430_4988",
        "m1_b95_s43": "run_1773263921_8009",
    }
    proof_runs = {
        "updateproof_m1_s42": "run_1773268882_2350",
        "updateproof_m15_s42": "run_1773269155_3077",
    }
    nvfp4_runs = {
        "nvfp4_updateproof_m1_s42": "run_1773271908_2832",
        "nvfp4_updateproof_m15_s42": "run_1773272179_2822",
    }
    nvfp4_health_runs = {
        "nvfp4_gradhealth_m1_s42": "run_1773272450_4339",
        "nvfp4_gradhealth_m15_s42": "run_1773272599_4329",
    }
    final_steps = {
        "m0p5_s42": 110,
        "m1_s42": 200,
        "m2_s42": 200,
        "m4_s42": 200,
        "m15_s42": 200,
        "m0p5_s43": 200,
        "m1_s43": 200,
        "m2_s43": 80,
        "m4_s43": 200,
        "m15_s43": 110,
        "m1_b95_s42": 200,
        "m1_b95_s43": 200,
    }
    proof_steps = {
        "updateproof_m1_s42": [10, 50, 100, 130, 200],
        "updateproof_m15_s42": [10, 50, 100, 130],
    }
    nvfp4_steps = {
        "nvfp4_updateproof_m1_s42": [10, 50, 100, 130, 200],
        "nvfp4_updateproof_m15_s42": [10, 50, 100, 130, 200],
    }
    nvfp4_health_steps = {
        "nvfp4_gradhealth_m1_s42": [10, 50],
        "nvfp4_gradhealth_m15_s42": [10, 50],
    }
    figures = [
        "0008_expert_lr_sweep.svg",
        "0008_expert_lr_mechanism.svg",
    ]
    artifacts = [
        {"kind": "config_contract", "path": "configs/moonlet.toml"},
        {"kind": "repro_script", "path": "scripts/repro/run_0008_bf16_sweep.sh"},
        {"kind": "repro_script", "path": "scripts/repro/run_0008_bf16_updateproof.sh"},
        {"kind": "repro_script", "path": "scripts/repro/run_0008_nvfp4_updateproof.sh"},
        {"kind": "repro_script", "path": "scripts/repro/run_0008_nvfp4_gradhealth.sh"},
        {"kind": "repro_script", "path": "scripts/repro/summarize_0008_bf16_sweep.py"},
        {"kind": "training_source", "path": "nmoe/train.py"},
        {"kind": "experiments_db", "path": f"{main_root}/experiments.db"},
        {"kind": "experiments_db", "path": f"{proof_root}/experiments.db"},
        {"kind": "experiments_db", "path": f"{nvfp4_root}/experiments.db"},
        {"kind": "experiments_db", "path": f"{nvfp4_health_root}/experiments.db"},
    ]
    for label, run_id in main_runs.items():
        step = final_steps[label]
        artifacts.append(
            {
                "kind": "step_metrics_parquet",
                "path": f"{main_root}/metrics/{run_id}/step_{step:08d}.parquet",
            }
        )
    for label, run_id in proof_runs.items():
        for step in proof_steps[label]:
            artifacts.append(
                {
                    "kind": "step_metrics_parquet",
                    "path": f"{proof_root}/metrics/{run_id}/step_{step:08d}.parquet",
                }
            )
    for label, run_id in nvfp4_runs.items():
        for step in nvfp4_steps[label]:
            artifacts.append(
                {
                    "kind": "step_metrics_parquet",
                    "path": f"{nvfp4_root}/metrics/{run_id}/step_{step:08d}.parquet",
                }
            )
    for label, run_id in nvfp4_health_runs.items():
        for step in nvfp4_health_steps[label]:
            artifacts.append(
                {
                    "kind": "step_metrics_parquet",
                    "path": f"{nvfp4_health_root}/metrics/{run_id}/step_{step:08d}.parquet",
                }
            )
    return {
        "post": "0008",
        "title": "Do MoE Experts Need Different Learning Rates?",
        "evidence_tier": "external_repro",
        "status": "complete_for_scope",
        "generated_at_utc": _now_utc(),
        "summary": (
            "Theory-plus-validation receipts for 0008: the bf16 Moonlet two-seed sweep and bf16 direct "
            "update-proof runs that support the main result, plus a shorter nvfp4/ExpertAdamW diagnostic lane "
            "and grad-health canary that point in the same direction without closing the full nvfp4 tuning "
            "question. The earned result is the bf16 lane: sparse routing attenuates raw expert gradients, but "
            "the adaptive optimizer still prefers lr_expert = lr_dense and makes 15x an overcorrection."
        ),
        "runs": {**main_runs, **proof_runs, **nvfp4_runs, **nvfp4_health_runs},
        "artifacts": artifacts + _site_figure_entries(figures, cfg),
        "commands": [
            "bash scripts/repro/run_0008_bf16_sweep.sh blog_artifacts/0008_expert_lr_bf16_20260311",
            "python3 scripts/repro/summarize_0008_bf16_sweep.py --kind main --study-root blog_artifacts/0008_expert_lr_bf16_20260311",
            "bash scripts/repro/run_0008_bf16_updateproof.sh blog_artifacts/0008_expert_lr_bf16_updateproof_20260311",
            "python3 scripts/repro/summarize_0008_bf16_sweep.py --kind updateproof --study-root blog_artifacts/0008_expert_lr_bf16_updateproof_20260311 --steps 10,50,100,130,200",
            "bash scripts/repro/run_0008_nvfp4_updateproof.sh blog_artifacts/0008_expert_lr_nvfp4_updateproof_b20260311",
            "python3 scripts/repro/summarize_0008_bf16_sweep.py --kind nvfp4_updateproof --study-root blog_artifacts/0008_expert_lr_nvfp4_updateproof_b20260311 --steps 10,50,100,130,200",
            "bash scripts/repro/run_0008_nvfp4_gradhealth.sh blog_artifacts/0008_expert_lr_nvfp4_gradhealth_c20260311",
            "python3 scripts/repro/summarize_0008_bf16_sweep.py --kind nvfp4_gradhealth --study-root blog_artifacts/0008_expert_lr_nvfp4_gradhealth_c20260311 --steps 10,50",
        ],
        "notes": [
            "Primary scope is Moonlet bf16 with expert_opt = auto -> AdamW. The nvfp4/ExpertAdamW surface in this receipt is diagnostic and directional, not a coequal settled proof lane. This receipt does not claim the result transfers unchanged to ExpertMuon.",
            "In the current implementation, route_scale is applied to router logits before sigmoid/topk; the post-normalization scaling knob is routed_scaling_factor, which remains 1.0 in Moonlet.",
            "Across the bf16 two-seed sweep, lr_expert = lr_dense is the best tested multiplier in the 200-step window; 0.5x is not better, and larger multipliers worsen loss or collapse routing earlier.",
            "The beta2_expert ablation flips across the two seeds, so this receipt set does not promote a solved beta2 rule.",
            "The bf16 direct update-proof runs log exact per-group grad norms and post-step optimizer-update ratios: at 1x the expert grad-to-param ratio is about 1e-3 of dense, but the optimizer-update ratio stays in the same order (0.36x to 0.62x); at 15x it rises to 5.63x to 12.12x before routing collapses at step 130.",
            "The nvfp4/ExpertAdamW direct proof is single-seed and only compares 1x to 15x. Within that narrower scope it shows weaker cancellation than bf16 but the same sign: at 1x the expert-to-dense optimizer-update ratio is about 0.22x to 0.53x, while at 15x it rises to 2.38x to 6.10x; this is enough to show directional over-update, not to close the full nvfp4 optimum.",
            "The short nvfp4 grad-health canary uses the new per-expert zero-fraction and abs-mean tags from nmoe/train.py. At step 50, 1x shows about 0.71% exact-zero expert gradients on W1/W2/W3, while 15x shows about 13.35%; that canary is diagnostic only and does not replace the main proof runs.",
        ],
    }


def _build_0009(_: dict[str, Any], cfg: BuildConfig) -> dict[str, Any]:
    return {
        "post": "0009",
        "title": "RDEP",
        "evidence_tier": "paper_canonical",
        "status": "public_reference_surface",
        "generated_at_utc": _now_utc(),
        "summary": (
            "Public paper-PDF mirror for RDEP: the site-served paper PDF that carries the canonical "
            "architecture, analysis, evaluation, and reproducibility appendix. Internal paper sources "
            "and standalone benchmark artifacts are not mirrored here."
        ),
        "artifacts": [
            {
                "kind": "paper_pdf",
                "path": "papers/rdep-nvlink.pdf",
                "public_asset": "/papers/rdep-nvlink.pdf",
            }
        ],
        "commands": [
            "ls papers/rdep-nvlink.pdf",
        ],
        "notes": [
            "This public receipt mirrors only the paper PDF exposed on the site; it does not enumerate the internal paper source tree or standalone benchmark artifacts.",
            "The paper remains the primary truth boundary for the quoted tables, recipes, and reproducibility appendix.",
            "Standalone benchmark JSONs and production training logs are not currently served as public site assets and are therefore not listed as mirrored artifacts.",
        ],
    }


def _build_0010(_: dict[str, Any], cfg: BuildConfig) -> dict[str, Any]:
    structured_dirs = [
        "physicslm4_reval_s0/baseline",
        "physicslm4_reval_s0/engram",
        "physicslm4_reval_s0/mhc",
        "physicslm4_reval_s0/canon_abcd",
        "physicslm4_reval_s0/mhc_canon",
        "physicslm4_reval_s0/full",
        "validation_3seed/baseline_s1",
        "validation_3seed/baseline_s2",
        "validation_3seed/engram_s1",
        "validation_3seed/engram_s2",
        "validation_3seed/mhc_s1",
        "validation_3seed/mhc_s2",
        "validation_3seed/canon_s1",
        "validation_3seed/canon_s2",
        "validation_3seed/mhc_canon_s1",
        "validation_3seed/mhc_canon_s2",
        "validation_3seed/mhc_canon_engram_s1",
        "validation_3seed/mhc_canon_engram_s2",
    ]
    physics_artifacts: list[dict[str, str]] = []
    for d in structured_dirs:
        physics_artifacts.extend(
            [
                {"kind": "physics_run_index", "path": f"physics/{d}/runs.json"},
                {
                    "kind": "physics_summary",
                    "path": f"physics/{d}/analysis/summary.json",
                },
                {
                    "kind": "physics_lano_cfg_dp_valid",
                    "path": f"physics/{d}/analysis/lano_cfg_dp_valid.json",
                },
            ]
        )

    engram_dirs = {
        "engram_repro_ngram_d24_s0_matrix": [
            ("physics_run_index", "runs.json"),
            ("physics_summary", "analysis/summary.json"),
            ("physics_slices_valid", "analysis/slices_valid.json"),
            ("physics_logitlens_valid", "analysis/logitlens_valid.json"),
            ("physics_cka_valid", "analysis/cka_valid.json"),
            ("physics_cka_depth_shift", "analysis/cka_depth_shift.json"),
        ],
        "engram_repro_ngram_d24_s0_matrix_layerce": [
            ("physics_run_index", "runs.json"),
            ("physics_summary", "analysis/summary.json"),
            ("physics_slices_valid", "analysis/slices_valid.json"),
            ("physics_layer_ce_valid", "analysis/layer_ce_valid.json"),
        ],
    }
    for d, entries in engram_dirs.items():
        for kind, rel in entries:
            physics_artifacts.append({"kind": kind, "path": f"physics/{d}/{rel}"})

    mixed_ratio_dirs = [
        "mixed_ratio_3seed/G1L0_s42",
        "mixed_ratio_3seed/G1L0_s43",
        "mixed_ratio_3seed/G1L0_s44",
        "mixed_ratio_3seed/G1L1_s42",
        "mixed_ratio_3seed/G1L1_s43",
        "mixed_ratio_3seed/G1L1_s44",
        "mixed_ratio_3seed/G1L9_s42",
        "mixed_ratio_3seed/G1L9_s43",
        "mixed_ratio_3seed/G1L9_s44",
        "mixed_ratio_3seed/G0L1_s42",
        "mixed_ratio_3seed/G0L1_s43",
        "mixed_ratio_3seed/G0L1_s44",
    ]
    for d in mixed_ratio_dirs:
        physics_artifacts.extend(
            [
                {"kind": "physics_run_index", "path": f"physics/{d}/runs.json"},
                {
                    "kind": "physics_summary",
                    "path": f"physics/{d}/analysis/summary.json",
                },
            ]
        )

    figures = [
        "physicslm4_dpkl.svg",
        "physics_attn_ratio_4k.svg",
        "engram_repro_d24/logitlens_ngram.svg",
        "engram_repro_d24/logitlens_ngram_delta.svg",
        "engram_repro_d24/logitlens_ngram_polysemy.svg",
        "engram_repro_d24/logitlens_ngram_polysemy_delta.svg",
        "engram_repro_d24/logitlens_ngram_scrambled.svg",
        "engram_repro_d24/logitlens_ngram_scrambled_delta.svg",
        "engram_repro_d24/cka_ngram__width_fixed_residual_vanilla_memory_engram_attn_global.svg",
        "engram_repro_d24/cka_ngram__width_fixed_residual_vanilla_memory_ple_ngrammer_attn_global.svg",
        "engram_repro_d24/cka_ngram_polysemy__width_fixed_residual_vanilla_memory_engram_attn_global.svg",
        "engram_repro_d24/cka_ngram_polysemy__width_fixed_residual_vanilla_memory_ple_ngrammer_attn_global.svg",
        "engram_repro_d24/cka_ngram_scrambled__width_fixed_residual_vanilla_memory_engram_attn_global.svg",
        "engram_repro_d24/cka_ngram_scrambled__width_fixed_residual_vanilla_memory_ple_ngrammer_attn_global.svg",
        "engram_repro_d24/mem_gate_heatmap_width_fixed_residual_vanilla_memory_engram_attn_global.svg",
        "engram_repro_d24/slices_layer_ce_ngram_polysemy_width_fixed_residual_vanilla_memory_none_attn_global.svg",
        "engram_repro_d24/slices_layer_ce_ngram_polysemy_width_fixed_residual_vanilla_memory_engram_attn_global.svg",
        "engram_repro_d24/slices_layer_ce_ngram_polysemy_width_fixed_residual_vanilla_memory_ple_ngrammer_attn_global.svg",
    ]
    figure_artifacts = _site_figure_entries(figures, cfg)

    return {
        "post": "0010",
        "title": "Reproducing Canon, mHC, and Engram",
        "evidence_tier": "external_repro",
        "status": "complete_for_scope",
        "generated_at_utc": _now_utc(),
        "summary": (
            "Physics harness receipts for Canon/mHC/Engram claims and figure provenance. "
            "This package captures the multiseed structured-suite run indexes, the seed-0 Engram "
            "slice and layer diagnostics, the 4k mixed-ratio attention sweep, and the rendered figure mapping."
        ),
        "artifacts": physics_artifacts + figure_artifacts,
        "commands": [
            "for seed in 0 1 2; do for spec in 'width=fixed,residual=vanilla,memory=none,attn=global' 'width=fixed,residual=vanilla,memory=engram,attn=global' 'width=fixed,residual=mhc,memory=none,attn=global' 'width=fixed,residual=vanilla,precond=canon,canon_set=ABCD,memory=none,attn=global' 'width=fixed,residual=mhc,precond=canon,canon_set=ABCD,memory=none,attn=global' 'width=fixed,residual=mhc,precond=canon,canon_set=ABCD,memory=engram,attn=global'; do python -m nmoe.research.physics.arch_ablations --output \"./out/physicslm4_reval_s${seed}\" --steps 2000 --seed \"$seed\" --init-seed 0 --dim 256 --n-layers 6 --seq-len 2048 --mlp-type swiglu --lano-cfg-kl --lano-cfg-kl-n 16 --layer-ce --layer-ce-n 256 --tasks \"depo_v2:0.4:n_words_max=30,max_hops=6,n_qa=4,mini_vocab=8,min_tlen=2,max_tlen=4\" \"lano_cfg:0.3:graph_seed=0,depth=6,num_sym=3,deg_min=2,deg_max=2,len_min=2,len_max=3,max_len=1024,token_base=9400\" \"mano:0.3:depth=6,ops=asm\" --variant \"$spec\"; done; done",
            "python -m nmoe.research.physics.arch_ablations --output ./out/engram_repro_matrix_s0 --steps 2000 --seed 0 --init-seed 0 --dim 256 --n-layers 24 --seq-len 256 --slice-metrics --slice-metrics-n 512 --logitlens --logitlens-n 256 --cka --cka-n 128 --tasks \"ngram_polysemy:1.0:n_symbols=512,n_steps=128,table_seed=0\" \"ngram_scrambled:1.0:n_symbols=512,n_steps=128,table_seed=0\" \"ngram:1.0:n_symbols=512,n_steps=128,table_seed=0\" --matrix engram_repro",
            "python -m nmoe.research.physics.viz_logitlens --runs ./out/engram_repro_matrix_s0",
            "python -m nmoe.research.physics.viz_cka --runs ./out/engram_repro_matrix_s0",
            "python -m nmoe.research.physics.arch_ablations --output ./out/engram_repro_matrix_layerce_s0 --steps 2000 --seed 0 --init-seed 0 --dim 256 --n-layers 24 --seq-len 256 --slice-metrics --slice-metrics-n 512 --layer-ce --layer-ce-n 256 --tasks \"ngram:1.0:n_symbols=512,n_steps=128,table_seed=0\" \"ngram_polysemy:1.0:n_symbols=512,n_steps=128,table_seed=0\" \"ngram_scrambled:1.0:n_symbols=512,n_steps=128,table_seed=0\" --matrix engram_repro",
            "python -m nmoe.research.physics.viz_slices --runs ./out/engram_repro_matrix_layerce_s0",
            "for seed in 42 43 44; do for spec in 'width=fixed,residual=mhc,precond=canon,canon_set=ABCD,memory=engram,attn=global' 'width=fixed,residual=mhc,precond=canon,canon_set=ABCD,memory=engram,attn=mixed:G1L1:64' 'width=fixed,residual=mhc,precond=canon,canon_set=ABCD,memory=engram,attn=mixed:G1L9:64' 'width=fixed,residual=mhc,precond=canon,canon_set=ABCD,memory=engram,attn=local:64'; do python -m nmoe.research.physics.arch_ablations --output \"./out/mixed_ratio_4k_s${seed}\" --steps 2000 --seed \"$seed\" --init-seed 0 --dim 256 --n-layers 6 --seq-len 4096 --slice-metrics --slice-metrics-n 512 --tasks \"ngram_polysemy:0.4:n_symbols=512,n_steps=128,table_seed=0\" \"ngram_scrambled:0.4:n_symbols=512,n_steps=128,table_seed=0\" \"depo:0.8:n_entities=6,max_hops=4\" \"mano:0.2:depth=2,ops=asm\" \"mano:0.2:depth=6,ops=asm\" --variant \"$spec\"; done; done",
        ],
        "notes": [
            "Scope is architecture behavior on the physics harness, not large-scale production claims.",
            "Structured-suite table: seed 0 comes from physics/physicslm4_reval_s0/* and seeds 1/2 come from physics/validation_3seed/*; loss/token accuracy are read from analysis/summary.json and DP-KL from analysis/lano_cfg_dp_valid.json.",
            "Seed-0 Engram slice table uses physics/engram_repro_ngram_d24_s0_matrix/analysis/slices_valid.json; the layerwise collision diagnostic uses the matching physics/engram_repro_ngram_d24_s0_matrix_layerce/* export.",
            "The 4k attention figure is the 3-seed mixed-ratio sweep under physics/mixed_ratio_3seed/* (100%, 50%, 10%, 0% global-layer schedules).",
            "The published commands regenerate local ./out/... runs and figures; mapping those outputs to the cited physics/... artifact IDs is still a separate provenance step recorded in this bundle.",
        ],
    }


def _build_post(post: str, manifest: dict[str, Any], cfg: BuildConfig) -> dict[str, Any]:
    if post == "0002":
        return _build_0002(manifest, cfg)
    if post == "0003":
        return _build_0003(manifest, cfg)
    if post == "0004":
        return _build_0004(manifest, cfg)
    if post == "0005":
        return _build_0005(manifest, cfg)
    if post == "0006":
        return _build_0006(manifest, cfg)
    if post == "0007":
        return _build_0007(manifest, cfg)
    if post == "0008":
        return _build_0008(manifest, cfg)
    if post == "0009":
        return _build_0009(manifest, cfg)
    if post == "0010":
        return _build_0010(manifest, cfg)
    raise ValueError(f"unsupported post: {post}")


def _needs_manifest(post: str) -> bool:
    return post in {"0002", "0003", "0004"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build reproducibility receipt bundles for selected research posts.")
    ap.add_argument("--data-root", default="/data", help="Root containing blog_artifacts/ and physics/ (default: /data)")
    ap.add_argument(
        "--canonical-data-root",
        default="/data",
        help="Path prefix emitted into receipts (default: /data)",
    )
    ap.add_argument("--repo-root", default=".", help="Monorepo root (default: cwd)")
    ap.add_argument("--out-dir", default="repro", help="Output directory for *.receipts.json")
    ap.add_argument(
        "--post",
        default="all",
        choices=["all", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010"],
        help="Post to generate",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    if not out_dir.exists() and str(args.out_dir).startswith("nmoe/"):
        # Monorepo compatibility: external path alias points at the internal mirror.
        out_dir = (repo_root / "community" / str(args.out_dir)).resolve()

    cfg = BuildConfig(
        data_root=Path(args.data_root).resolve(),
        canonical_data_root=str(args.canonical_data_root),
        repo_root=repo_root,
        out_dir=out_dir,
        post=str(args.post),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    posts = ["0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010"] if cfg.post == "all" else [cfg.post]
    manifest: dict[str, Any] = {}
    if any(_needs_manifest(post) for post in posts):
        manifest = _load_manifest(cfg.data_root / "blog_artifacts" / "manifest.json")
    for post in posts:
        obj = _build_post(post, manifest, cfg)
        out = cfg.out_dir / f"{post}.receipts.json"
        out.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
