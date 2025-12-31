"""
Deterministic mixture/flow resolver (elegant minimalism).

Reads a canonical mixture TOML and a flow profiles TOML, and produces a
MixturePlan with per-stage, per-source quotas (in sequences) and fixed-point
weights for Smooth Weighted Round-Robin (SWRR).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

from nmoe.config import load_toml


FP_SCALE = 1_000_000  # fixed-point scale for SWRR weights


@dataclass(frozen=True)
class Source:
    id: str
    tokens_b: float
    percent: Optional[float] = None


@dataclass
class HFSource:
    """HuggingFace dataset reference."""
    dataset: str
    subset: Optional[str] = None
    split: str = "train"
    valid_split: Optional[str] = None
    text_field: str = "text"
    data_files: Optional[str] = None  # Glob pattern for specific files within dataset


@dataclass
class SourcePlan:
    id: str
    weight_fp: int
    quota_sequences: int
    target_tokens: int
    paths: List[str]
    hf: Optional[HFSource] = None  # HuggingFace source info
    max_repetition_ratio: float = 1.0
    max_source_fraction: float = 1.0


@dataclass
class StagePlan:
    stage_id: str  # pretrain | mid | long
    total_tokens_b: float
    sources: List[SourcePlan]


@dataclass
class MixturePlan:
    plan_id: str
    plan_hash: str
    mixture_id: str
    flow_mode: str
    sample_temperature: float
    seq_len: int
    stages: List[StagePlan]

    def to_json(self) -> str:
        return json.dumps(
            {
                "plan_id": self.plan_id,
                "mixture_id": self.mixture_id,
                "flow_mode": self.flow_mode,
                "sample_temperature": self.sample_temperature,
                "seq_len": self.seq_len,
                "stages": [
                    {
                        "stage_id": s.stage_id,
                        "total_tokens_b": s.total_tokens_b,
                        "sources": [
                            {
                                "id": sp.id,
                                "weight_fp": sp.weight_fp,
                                "quota_sequences": sp.quota_sequences,
                                "target_tokens": sp.target_tokens,
                                "paths": sp.paths,
                                "max_repetition_ratio": sp.max_repetition_ratio,
                                "max_source_fraction": sp.max_source_fraction,
                            }
                            for sp in s.sources
                        ],
                    }
                    for s in self.stages
                ],
            },
            sort_keys=True,
        )


def _read_toml(path: Path) -> Dict:
    return load_toml(path)


def _largest_remainder_quota(tokens_target: int, seq_len: int, weights: List[float]) -> List[int]:
    """
    Convert a list of relative weights into integer sequence quotas that sum to
    floor(tokens_target/seq_len) using the largest remainder method.
    """
    total_sequences = max(0, tokens_target // seq_len)
    if total_sequences == 0:
        return [0] * len(weights)
    total_w = sum(weights) or 1.0
    exact = [total_sequences * (w / total_w) for w in weights]
    floor_part = [int(x) for x in exact]
    remainders = [x - int(x) for x in exact]
    remaining = total_sequences - sum(floor_part)
    order = sorted(range(len(weights)), key=lambda i: remainders[i], reverse=True)
    for i in range(remaining):
        floor_part[order[i]] += 1
    return floor_part


def _fixed_point_weights(weights: List[float]) -> List[int]:
    total = sum(weights) or 1.0
    return [max(1, int(round((w / total) * FP_SCALE))) for w in weights]


def resolve_plan(
    *,
    mixture_toml: Path,
    flow_profiles_toml: Path,
    flow_section: str,
    seq_len: int,
    active_params_b: Optional[float] = None,
    dataset_root: Optional[Path] = None,
) -> MixturePlan:
    """
    Build a MixturePlan from mixture + flow TOMLs.

    - mixture_toml: configs/mixtures/olmo3_1025.toml
    - flow_profiles_toml: configs/flow_profiles.toml
    - flow_section: e.g., "flow.proxy" | "flow.full_train"
    - seq_len: fixed sequence length
    - dataset_root: optional path prefix to resolve relative source paths
    """
    mix = _read_toml(mixture_toml)
    flow_all = _read_toml(flow_profiles_toml)

    # Locate flow entry
    parts = flow_section.split(".")
    node = flow_all
    for p in parts:
        node = node[p]
    flow_mode: str = node.get("mode", "")
    sample_temperature: float = float(node.get("sample_temperature", 1.0))
    scale: Optional[float] = node.get("scale")
    tokens_b: Optional[float] = node.get("tokens_b")  # Total tokens for flow
    tokens_b_ratio: Optional[float] = node.get("tokens_b_ratio")  # Multiplier for active_params_b
    tokens_b_override: Dict[str, float] = node.get("tokens_b_override", {})
    mixtures_ref = node.get("mixture", {})

    # Resolve tokens_b from ratio if specified
    if tokens_b_ratio is not None and tokens_b is None:
        if active_params_b is None:
            raise ValueError(f"Flow {flow_section} uses tokens_b_ratio but active_params_b not provided")
        tokens_b = tokens_b_ratio * active_params_b

    # If tokens_b is set, compute scale from total mixture tokens
    if tokens_b is not None and scale is None:
        total_mix_tokens_b = 0.0
        for stage_id in ("pretrain", "mid", "long"):
            mix_name = mixtures_ref.get(stage_id)
            if mix_name:
                stage_dict = mix
                for k in f"mixtures.{mix_name}".split("."):
                    stage_dict = stage_dict[k]
                total_mix_tokens_b += float(stage_dict.get("total_tokens_b", 0.0))
        scale = tokens_b / total_mix_tokens_b if total_mix_tokens_b > 0 else 1.0

    stages: List[StagePlan] = []
    for stage_id in ("pretrain", "mid", "long"):
        mix_name = mixtures_ref.get(stage_id)
        if not mix_name:
            continue
        stage_key = f"mixtures.{mix_name}"
        stage_dict = mix
        for k in stage_key.split("."):
            stage_dict = stage_dict[k]
        total_tokens_b = float(stage_dict.get("total_tokens_b", 0.0))
        sources = stage_dict.get("sources", [])

        # Compute target tokens: override > scale > 1.0
        stage_tokens_b = (
            float(tokens_b_override.get(stage_id)) if stage_id in tokens_b_override else total_tokens_b * (scale if scale else 1.0)
        )
        stage_tokens = int(stage_tokens_b * 1_000_000_000)
        weights = [float(s.get("tokens_b", 0.0)) for s in sources]
        quotas = _largest_remainder_quota(stage_tokens, seq_len, weights)
        w_fp = _fixed_point_weights(weights)

        src_plans: List[SourcePlan] = []
        for s, q, wf in zip(sources, quotas, w_fp):
            # Parse HuggingFace source info if present
            hf_source = None
            if "hf_dataset" in s:
                hf_source = HFSource(
                    dataset=str(s["hf_dataset"]),
                    subset=s.get("hf_subset"),
                    split=str(s.get("hf_split", "train")),
                    valid_split=s.get("hf_valid_split"),
                    text_field=str(s.get("text_field", "text")),
                    data_files=s.get("hf_data_files"),
                )
            src_plans.append(
                SourcePlan(
                    id=str(s.get("id")),
                    weight_fp=int(wf),
                    quota_sequences=int(q),
                    target_tokens=int(s.get("tokens_b", 0.0) * 1_000_000_000),
                    paths=[],
                    hf=hf_source,
                )
            )

        stages.append(StagePlan(stage_id=stage_id, total_tokens_b=stage_tokens_b, sources=src_plans))

    plan_proto = {
        "mixture_id": mixture_toml.name,
        "flow_mode": flow_mode,
        "sample_temperature": sample_temperature,
        "seq_len": seq_len,
        "stages": [
            {
                "stage_id": s.stage_id,
                "total_tokens_b": s.total_tokens_b,
                "sources": [
                    {
                        "id": sp.id,
                        "weight_fp": sp.weight_fp,
                        "quota_sequences": sp.quota_sequences,
                        "target_tokens": sp.target_tokens,
                        "paths": sp.paths,
                    }
                    for sp in s.sources
                ],
            }
            for s in stages
        ],
    }
    plan_hash = hashlib.sha256(json.dumps(plan_proto, sort_keys=True).encode()).hexdigest()

    return MixturePlan(
        plan_id=f"{mixture_toml.stem}:{flow_mode}",
        plan_hash=plan_hash,
        mixture_id=mixture_toml.stem,
        flow_mode=flow_mode,
        sample_temperature=sample_temperature,
        seq_len=seq_len,
        stages=stages,
    )


def populate_paths(
    plan: MixturePlan,
    *,
    dataset_root: Path,
    split: str = "train",
) -> None:
    """
    Populate SourcePlan.paths in-place for each stage/source.

    New convention (flow-scoped root):
      {dataset_root}/{stage_id}/{source_id}/{split}/**/*.npy
      where dataset_root typically is /data/flows/<flow>
    """

    def _expand(pattern: str) -> List[str]:
        return sorted(set(glob.glob(str(pattern), recursive=True)))

    for stage in plan.stages:
        for sp in stage.sources:
            base = dataset_root / stage.stage_id / sp.id / split
            files = _expand(str(base / "**/*.npy"))
            sp.paths = files
