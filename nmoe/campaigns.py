from __future__ import annotations

import dataclasses
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nmoe.config import Config, load_toml

CAMPAIGNS_DIRNAME = "campaigns"
DEFAULT_RECEIPT_DIR = "repro/campaign_runs"
CLAIMS_DIRNAME = "_claims"
SUPPORTED_RUNNERS = frozenset({"speedrun"})
SUPPORTED_BASELINES = frozenset({"campaign_receipts", "leaderboard"})
SUPPORTED_DIRECTIONS = frozenset({"min", "max"})
SUPPORTED_MUTATION_TIERS = frozenset({"config_only", "research_surface", "trainer_surface"})
SUPPORTED_SEARCH_STRATEGIES = frozenset({"coordinate_descent", "llm_coordinate_descent"})
KNOWN_CONFIG_KEYS = frozenset(f.name for f in dataclasses.fields(Config))


class CampaignError(ValueError):
  """Raised when a campaign spec or invocation is invalid."""


def _utc_now() -> datetime:
  return datetime.now(timezone.utc)


def _env_truthy(name: str) -> bool:
  value = os.environ.get(name, "").strip().lower()
  return value in {"1", "true", "yes", "on"}


def _utc_from_epoch(ts: float) -> str:
  return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(value: str) -> str:
  value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
  value = value.strip("-._")
  return value or "candidate"


def _require_mapping(obj: Any, *, where: str) -> dict[str, Any]:
  if not isinstance(obj, dict):
    raise CampaignError(f"{where} must be a TOML table")
  return obj


def _require_list_of_str(obj: Any, *, where: str) -> list[str]:
  if obj is None:
    return []
  if not isinstance(obj, list) or any(not isinstance(v, str) for v in obj):
    raise CampaignError(f"{where} must be an array of strings")
  return list(obj)


def _maybe_float(obj: Any, *, where: str) -> float | None:
  if obj is None:
    return None
  if isinstance(obj, (int, float)):
    return float(obj)
  raise CampaignError(f"{where} must be a number")


def _maybe_int(obj: Any, *, where: str) -> int | None:
  if obj is None:
    return None
  if isinstance(obj, int):
    return int(obj)
  raise CampaignError(f"{where} must be an integer")


def _normalize_value(obj: Any) -> str:
  if isinstance(obj, bool):
    return "true" if obj else "false"
  if obj is None:
    return ""
  if isinstance(obj, int):
    return str(obj)
  if isinstance(obj, float):
    return format(obj, ".15g")
  return str(obj).strip()


def _maybe_number(obj: Any) -> float | None:
  if obj is None or isinstance(obj, bool):
    return None
  if isinstance(obj, (int, float)):
    return float(obj)
  try:
    return float(str(obj).strip())
  except Exception:
    return None


def _value_equal(a: Any, b: Any) -> bool:
  lhs = _normalize_value(a)
  rhs = _normalize_value(b)
  if lhs == rhs:
    return True
  lhs_num = _maybe_number(lhs)
  rhs_num = _maybe_number(rhs)
  if lhs_num is not None and rhs_num is not None:
    scale = max(1.0, abs(lhs_num), abs(rhs_num))
    return abs(lhs_num - rhs_num) <= (1e-12 * scale)
  return lhs.lower() == rhs.lower()


@dataclass(frozen=True)
class CampaignBudgetStage:
  name: str
  steps: int | None = None
  max_wall_s: int | None = None

  @classmethod
  def from_dict(cls, name: str, obj: Any) -> "CampaignBudgetStage":
    data = _require_mapping(obj, where=f"budget.{name}")
    steps = _maybe_int(data.get("steps"), where=f"budget.{name}.steps")
    max_wall_s = _maybe_int(data.get("max_wall_s"), where=f"budget.{name}.max_wall_s")
    if steps is not None and steps <= 0:
      raise CampaignError(f"budget.{name}.steps must be > 0")
    if max_wall_s is not None and max_wall_s <= 0:
      raise CampaignError(f"budget.{name}.max_wall_s must be > 0")
    return cls(name=name, steps=steps, max_wall_s=max_wall_s)


@dataclass(frozen=True)
class CampaignObjective:
  primary_metric: str
  direction: str
  min_delta_abs: float = 0.0
  require_target_reached: bool = False
  required_metrics: tuple[str, ...] = ()
  max_final_loss: float | None = None
  min_core_score: float | None = None
  max_core_drop: float | None = None

  @classmethod
  def from_dict(cls, obj: Any) -> "CampaignObjective":
    data = _require_mapping(obj, where="objective")
    primary_metric = str(data.get("primary_metric", "")).strip()
    if not primary_metric:
      raise CampaignError("objective.primary_metric is required")

    direction = str(data.get("direction", "")).strip().lower()
    if direction not in SUPPORTED_DIRECTIONS:
      raise CampaignError(f"objective.direction must be one of {sorted(SUPPORTED_DIRECTIONS)}")

    min_delta_abs = _maybe_float(data.get("min_delta_abs", 0.0), where="objective.min_delta_abs") or 0.0
    if min_delta_abs < 0:
      raise CampaignError("objective.min_delta_abs must be >= 0")

    constraints = _require_mapping(data.get("constraints", {}), where="objective.constraints")
    require_target_reached = bool(constraints.get("require_target_reached", False))
    required_metrics = tuple(
      _require_list_of_str(constraints.get("required_metrics", []), where="objective.constraints.required_metrics")
    )
    max_final_loss = _maybe_float(constraints.get("max_final_loss"), where="objective.constraints.max_final_loss")
    min_core_score = _maybe_float(constraints.get("min_core_score"), where="objective.constraints.min_core_score")
    max_core_drop = _maybe_float(constraints.get("max_core_drop"), where="objective.constraints.max_core_drop")
    if max_core_drop is not None and max_core_drop < 0:
      raise CampaignError("objective.constraints.max_core_drop must be >= 0")

    return cls(
      primary_metric=primary_metric,
      direction=direction,
      min_delta_abs=min_delta_abs,
      require_target_reached=require_target_reached,
      required_metrics=required_metrics,
      max_final_loss=max_final_loss,
      min_core_score=min_core_score,
      max_core_drop=max_core_drop,
    )


@dataclass(frozen=True)
class CampaignMutation:
  tier: str = "config_only"
  allowed_overrides: tuple[str, ...] = ()
  allowed_files: tuple[str, ...] = ()

  @classmethod
  def from_dict(cls, obj: Any) -> "CampaignMutation":
    data = _require_mapping(obj, where="mutation")
    tier = str(data.get("tier", "config_only")).strip()
    if tier not in SUPPORTED_MUTATION_TIERS:
      raise CampaignError(f"mutation.tier must be one of {sorted(SUPPORTED_MUTATION_TIERS)}")

    allowed_overrides = tuple(_require_list_of_str(data.get("allowed_overrides", []), where="mutation.allowed_overrides"))
    unknown_keys = sorted(k for k in allowed_overrides if k not in KNOWN_CONFIG_KEYS)
    if unknown_keys:
      raise CampaignError(f"mutation.allowed_overrides contains unknown config keys: {', '.join(unknown_keys)}")

    allowed_files = tuple(_require_list_of_str(data.get("allowed_files", []), where="mutation.allowed_files"))
    return cls(tier=tier, allowed_overrides=allowed_overrides, allowed_files=allowed_files)


@dataclass(frozen=True)
class CampaignBaseline:
  source: str = "campaign_receipts"
  match: dict[str, str] = field(default_factory=dict)

  @classmethod
  def from_dict(cls, obj: Any) -> "CampaignBaseline":
    data = _require_mapping(obj, where="baseline")
    source = str(data.get("source", "campaign_receipts")).strip().lower()
    if source not in SUPPORTED_BASELINES:
      raise CampaignError(f"baseline.source must be one of {sorted(SUPPORTED_BASELINES)}")

    match = data.get("match", {})
    if match is None:
      match = {}
    if not isinstance(match, dict):
      raise CampaignError("baseline.match must be a TOML inline table")
    match_out = {str(k): str(v) for k, v in match.items()}
    return cls(source=source, match=match_out)


@dataclass(frozen=True)
class CampaignOutputs:
  receipt_dir: str = DEFAULT_RECEIPT_DIR

  @classmethod
  def from_dict(cls, obj: Any) -> "CampaignOutputs":
    data = _require_mapping(obj, where="outputs")
    receipt_dir = str(data.get("receipt_dir", DEFAULT_RECEIPT_DIR)).strip()
    if not receipt_dir:
      raise CampaignError("outputs.receipt_dir must not be empty")
    return cls(receipt_dir=receipt_dir)


@dataclass(frozen=True)
class CampaignSearchAxis:
  key: str
  values: tuple[str, ...]

  @classmethod
  def from_dict(cls, index: int, obj: Any, *, allowed_overrides: tuple[str, ...]) -> "CampaignSearchAxis":
    data = _require_mapping(obj, where=f"search.axis[{index}]")
    key = str(data.get("key", "")).strip()
    if not key:
      raise CampaignError(f"search.axis[{index}].key is required")
    if key not in set(allowed_overrides):
      raise CampaignError(
        f"search.axis[{index}].key={key!r} must be one of mutation.allowed_overrides"
      )

    raw_values = data.get("values")
    if not isinstance(raw_values, list) or not raw_values:
      raise CampaignError(f"search.axis[{index}].values must be a non-empty array")
    values = tuple(_normalize_value(value) for value in raw_values)
    if len(set(values)) != len(values):
      raise CampaignError(f"search.axis[{index}].values must be unique")
    return cls(key=key, values=values)


@dataclass(frozen=True)
class CampaignSearch:
  strategy: str = "coordinate_descent"
  max_trials: int = 4
  max_no_improve: int | None = None
  candidate_prefix: str = "auto"
  seed_overrides: dict[str, str] = field(default_factory=dict)
  llm_model: str | None = None
  llm_api_base: str | None = None
  llm_recent_receipts: int = 8
  llm_candidate_limit: int = 12
  fallback_strategy: str = "coordinate_descent"
  axes: tuple[CampaignSearchAxis, ...] = ()

  @classmethod
  def from_dict(cls, obj: Any, *, allowed_overrides: tuple[str, ...]) -> "CampaignSearch":
    data = _require_mapping(obj, where="search")
    strategy = str(data.get("strategy", "coordinate_descent")).strip().lower()
    if strategy not in SUPPORTED_SEARCH_STRATEGIES:
      raise CampaignError(f"search.strategy must be one of {sorted(SUPPORTED_SEARCH_STRATEGIES)}")

    max_trials = _maybe_int(data.get("max_trials", 4), where="search.max_trials") or 4
    if max_trials <= 0:
      raise CampaignError("search.max_trials must be > 0")

    max_no_improve = _maybe_int(data.get("max_no_improve"), where="search.max_no_improve")
    if max_no_improve is not None and max_no_improve <= 0:
      raise CampaignError("search.max_no_improve must be > 0")

    candidate_prefix = str(data.get("candidate_prefix", "auto")).strip()
    if not candidate_prefix:
      raise CampaignError("search.candidate_prefix must not be empty")

    raw_seed = data.get("seed_overrides", {})
    if raw_seed is None:
      raw_seed = {}
    if not isinstance(raw_seed, dict):
      raise CampaignError("search.seed_overrides must be a TOML inline table")
    seed_overrides = {str(key): _normalize_value(value) for key, value in raw_seed.items()}
    invalid_seed = sorted(key for key in seed_overrides if key not in set(allowed_overrides))
    if invalid_seed:
      raise CampaignError(
        f"search.seed_overrides contains keys outside mutation.allowed_overrides: {', '.join(invalid_seed)}"
      )

    llm_model = None
    llm_api_base = None
    llm_recent_receipts = 8
    llm_candidate_limit = 12
    fallback_strategy = "coordinate_descent"
    if strategy == "llm_coordinate_descent":
      llm_model = str(data.get("llm_model", "")).strip() or None
      llm_api_base = str(data.get("llm_api_base", "")).strip() or None
      llm_recent_receipts = _maybe_int(data.get("llm_recent_receipts", 8), where="search.llm_recent_receipts") or 8
      if llm_recent_receipts <= 0:
        raise CampaignError("search.llm_recent_receipts must be > 0")
      llm_candidate_limit = _maybe_int(data.get("llm_candidate_limit", 12), where="search.llm_candidate_limit") or 12
      if llm_candidate_limit <= 0:
        raise CampaignError("search.llm_candidate_limit must be > 0")
      fallback_strategy = str(data.get("fallback_strategy", "coordinate_descent")).strip().lower() or "coordinate_descent"
      if fallback_strategy not in SUPPORTED_SEARCH_STRATEGIES - {"llm_coordinate_descent"}:
        raise CampaignError("search.fallback_strategy must be coordinate_descent")

    axis_raw = data.get("axis", [])
    if not isinstance(axis_raw, list) or not axis_raw:
      raise CampaignError("search.axis must define at least one search axis")
    axes = tuple(
      CampaignSearchAxis.from_dict(index, axis_obj, allowed_overrides=allowed_overrides)
      for index, axis_obj in enumerate(axis_raw)
    )
    return cls(
      strategy=strategy,
      max_trials=max_trials,
      max_no_improve=max_no_improve,
      candidate_prefix=candidate_prefix,
      seed_overrides=seed_overrides,
      llm_model=llm_model,
      llm_api_base=llm_api_base,
      llm_recent_receipts=llm_recent_receipts,
      llm_candidate_limit=llm_candidate_limit,
      fallback_strategy=fallback_strategy,
      axes=axes,
    )


@dataclass(frozen=True)
class CampaignCandidateProposal:
  candidate_id: str
  overrides: dict[str, str]
  reason: str
  axis: str | None = None
  current_value: str | None = None
  proposed_value: str | None = None


@dataclass(frozen=True)
class SpeedrunRunnerConfig:
  config: str
  dtype: str = "auto"
  activation: str = ""
  eval_enabled: bool = True
  train_tokens: str | None = None
  val_tokens: str | None = None

  @classmethod
  def from_dict(cls, obj: Any) -> "SpeedrunRunnerConfig":
    data = _require_mapping(obj, where="runtime.speedrun")
    config = str(data.get("config", "")).strip()
    if not config:
      raise CampaignError("runtime.speedrun.config is required")
    dtype = str(data.get("dtype", "auto")).strip().lower()
    if dtype not in ("auto", "bf16", "fp8", "nvfp4"):
      raise CampaignError("runtime.speedrun.dtype must be one of auto, bf16, fp8, nvfp4")
    activation = str(data.get("activation", "")).strip()
    eval_enabled = bool(data.get("eval_enabled", True))
    train_tokens = data.get("train_tokens")
    val_tokens = data.get("val_tokens")
    if train_tokens is not None and not isinstance(train_tokens, str):
      raise CampaignError("runtime.speedrun.train_tokens must be a string token budget")
    if val_tokens is not None and not isinstance(val_tokens, str):
      raise CampaignError("runtime.speedrun.val_tokens must be a string token budget")
    return cls(
      config=config,
      dtype=dtype,
      activation=activation,
      eval_enabled=eval_enabled,
      train_tokens=train_tokens,
      val_tokens=val_tokens,
    )


@dataclass(frozen=True)
class CampaignSpec:
  root: Path
  path: Path
  name: str
  kind: str
  runner: str
  description: str = ""
  objective: CampaignObjective = field(default_factory=lambda: CampaignObjective(primary_metric="final_loss", direction="min"))
  mutation: CampaignMutation = field(default_factory=CampaignMutation)
  baseline: CampaignBaseline = field(default_factory=CampaignBaseline)
  outputs: CampaignOutputs = field(default_factory=CampaignOutputs)
  budget: dict[str, CampaignBudgetStage] = field(default_factory=dict)
  search: CampaignSearch | None = None
  speedrun: SpeedrunRunnerConfig | None = None

  @classmethod
  def from_dict(cls, root: Path, path: Path, obj: Any) -> "CampaignSpec":
    data = _require_mapping(obj, where=str(path))
    name = str(data.get("name", "")).strip()
    if not name:
      raise CampaignError(f"{path}: name is required")

    runner = str(data.get("runner", "")).strip().lower()
    if runner not in SUPPORTED_RUNNERS:
      raise CampaignError(f"{path}: runner must be one of {sorted(SUPPORTED_RUNNERS)}")

    kind = str(data.get("kind", runner)).strip().lower() or runner
    description = str(data.get("description", "")).strip()

    objective = CampaignObjective.from_dict(data.get("objective", {}))
    mutation = CampaignMutation.from_dict(data.get("mutation", {}))
    baseline = CampaignBaseline.from_dict(data.get("baseline", {}))
    outputs = CampaignOutputs.from_dict(data.get("outputs", {}))
    search = None
    if data.get("search") is not None:
      search = CampaignSearch.from_dict(data.get("search"), allowed_overrides=mutation.allowed_overrides)

    budget_table = _require_mapping(data.get("budget", {}), where="budget")
    if not budget_table:
      raise CampaignError(f"{path}: budget must define at least one stage")
    budget = {stage: CampaignBudgetStage.from_dict(stage, stage_obj) for stage, stage_obj in budget_table.items()}

    runtime_table = _require_mapping(data.get("runtime", {}), where="runtime")
    speedrun = None
    if runner == "speedrun":
      speedrun = SpeedrunRunnerConfig.from_dict(runtime_table.get("speedrun", {}))

    return cls(
      root=root,
      path=path,
      name=name,
      kind=kind,
      runner=runner,
      description=description,
      objective=objective,
      mutation=mutation,
      baseline=baseline,
      outputs=outputs,
      budget=budget,
      search=search,
      speedrun=speedrun,
    )

  def stage(self, name: str) -> CampaignBudgetStage:
    try:
      return self.budget[name]
    except KeyError as e:
      raise CampaignError(f"campaign {self.name} does not define budget stage {name!r}") from e

  def to_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "kind": self.kind,
      "runner": self.runner,
      "description": self.description,
      "path": str(self.path.relative_to(self.root)),
      "objective": {
        "primary_metric": self.objective.primary_metric,
        "direction": self.objective.direction,
        "min_delta_abs": self.objective.min_delta_abs,
        "constraints": {
          "require_target_reached": self.objective.require_target_reached,
          "required_metrics": list(self.objective.required_metrics),
          "max_final_loss": self.objective.max_final_loss,
          "min_core_score": self.objective.min_core_score,
          "max_core_drop": self.objective.max_core_drop,
        },
      },
      "mutation": {
        "tier": self.mutation.tier,
        "allowed_overrides": list(self.mutation.allowed_overrides),
        "allowed_files": list(self.mutation.allowed_files),
      },
      "baseline": {
        "source": self.baseline.source,
        "match": dict(self.baseline.match),
      },
      "outputs": {
        "receipt_dir": self.outputs.receipt_dir,
      },
      "search": None if self.search is None else {
        "strategy": self.search.strategy,
        "max_trials": self.search.max_trials,
        "max_no_improve": self.search.max_no_improve,
        "candidate_prefix": self.search.candidate_prefix,
        "seed_overrides": dict(self.search.seed_overrides),
        "llm_model": self.search.llm_model,
        "llm_api_base": self.search.llm_api_base,
        "llm_recent_receipts": self.search.llm_recent_receipts,
        "llm_candidate_limit": self.search.llm_candidate_limit,
        "fallback_strategy": self.search.fallback_strategy,
        "axis": [
          {"key": axis.key, "values": list(axis.values)}
          for axis in self.search.axes
        ],
      },
      "budget": {
        stage.name: {"steps": stage.steps, "max_wall_s": stage.max_wall_s}
        for stage in self.budget.values()
      },
      "runner_config": {
        "speedrun": None if self.speedrun is None else {
          "config": self.speedrun.config,
          "dtype": self.speedrun.dtype,
          "activation": self.speedrun.activation,
          "eval_enabled": self.speedrun.eval_enabled,
          "train_tokens": self.speedrun.train_tokens,
          "val_tokens": self.speedrun.val_tokens,
        },
      },
    }


def campaigns_dir(root: Path) -> Path:
  return root / CAMPAIGNS_DIRNAME


def discover_campaign_specs(root: Path) -> list[Path]:
  base = campaigns_dir(root)
  if not base.exists():
    return []
  return sorted(p for p in base.rglob("*.toml") if p.is_file())


def _resolve_campaign_path(root: Path, ref: str) -> Path:
  base = campaigns_dir(root)
  raw = Path(ref)

  candidates: list[Path] = []
  for prefix in (Path(), root, base):
    candidate = prefix / raw
    candidates.append(candidate)
    if candidate.suffix != ".toml":
      candidates.append(candidate.with_suffix(".toml"))

  for candidate in candidates:
    if candidate.exists():
      return candidate.resolve()

  matches: list[Path] = []
  for path in discover_campaign_specs(root):
    rel = path.relative_to(base)
    if path.stem == ref or str(rel) == ref or str(rel.with_suffix("")) == ref:
      matches.append(path)

  if not matches:
    raise CampaignError(f"campaign {ref!r} not found under {base}")
  if len(matches) > 1:
    choices = ", ".join(str(p.relative_to(base)) for p in matches)
    raise CampaignError(f"campaign {ref!r} is ambiguous: {choices}")
  return matches[0].resolve()


def load_campaign(root: Path, ref: str) -> CampaignSpec:
  path = _resolve_campaign_path(root, ref)
  data = load_toml(path)
  return CampaignSpec.from_dict(root.resolve(), path.resolve(), data)


def parse_candidate_overrides(items: list[str]) -> dict[str, str]:
  overrides: dict[str, str] = {}
  for item in items:
    if "=" not in item:
      raise CampaignError(f"override must be key=value, got {item!r}")
    key, value = item.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
      raise CampaignError(f"override must have a key: {item!r}")
    if key not in KNOWN_CONFIG_KEYS:
      raise CampaignError(f"override {key!r} is not a known nmoe config key")
    if key in overrides:
      raise CampaignError(f"override {key!r} specified more than once")
    overrides[key] = value
  return overrides


def validate_candidate_overrides(spec: CampaignSpec, overrides: dict[str, str]) -> None:
  allowed = set(spec.mutation.allowed_overrides)
  if not overrides:
    return
  if not allowed:
    raise CampaignError(f"campaign {spec.name} does not allow runtime overrides")
  disallowed = sorted(k for k in overrides if k not in allowed)
  if disallowed:
    raise CampaignError(
      f"campaign {spec.name} does not allow overrides for: {', '.join(disallowed)} "
      f"(allowed: {', '.join(sorted(allowed))})"
    )


def resolve_receipt_dir(spec: CampaignSpec, override_dir: str | None = None) -> Path:
  if override_dir:
    path = Path(override_dir)
  else:
    path = Path(spec.outputs.receipt_dir)
  if not path.is_absolute():
    path = spec.root / path
  return path


def new_receipt_path(spec: CampaignSpec, *, stage: str, candidate_id: str, receipt_dir: Path) -> Path:
  stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
  name = f"{stamp}_{_slug(candidate_id)}.json"
  return receipt_dir / spec.name / stage / name


def write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any] | None:
  try:
    return json.loads(path.read_text())
  except Exception:
    return None


def _claim_dir(spec: CampaignSpec, *, stage: str, receipt_dir: Path) -> Path:
  return receipt_dir / spec.name / stage / CLAIMS_DIRNAME


def _claim_path(spec: CampaignSpec, *, stage: str, candidate_id: str, receipt_dir: Path) -> Path:
  return _claim_dir(spec, stage=stage, receipt_dir=receipt_dir) / f"{_slug(candidate_id)}.json"


def _claim_overrides(payload: dict[str, Any]) -> dict[str, str]:
  raw = payload.get("overrides", {})
  if not isinstance(raw, dict):
    return {}
  return {str(key): _normalize_value(value) for key, value in raw.items()}


def _claim_ttl_s(spec: CampaignSpec, *, stage: str) -> int:
  stage_cfg = spec.stage(stage)
  base = stage_cfg.max_wall_s or 3600
  return max(base, 3600) + 900


def _claim_active(payload: dict[str, Any], *, now_ts: float) -> bool:
  expires_at_unix = _maybe_number(payload.get("expires_at_unix"))
  if expires_at_unix is None:
    return False
  return now_ts < float(expires_at_unix)


def active_stage_claims(spec: CampaignSpec, *, stage: str, receipt_dir: Path) -> list[dict[str, Any]]:
  claim_dir = _claim_dir(spec, stage=stage, receipt_dir=receipt_dir)
  if not claim_dir.exists():
    return []

  now_ts = _utc_now().timestamp()
  active: list[dict[str, Any]] = []
  for path in sorted(claim_dir.glob("*.json")):
    payload = _load_json(path)
    if payload is None:
      try:
        path.unlink()
      except FileNotFoundError:
        pass
      continue
    if _claim_active(payload, now_ts=now_ts):
      active.append(payload)
      continue
    try:
      path.unlink()
    except FileNotFoundError:
      pass
  return active


def claim_candidate(
  spec: CampaignSpec,
  *,
  stage: str,
  candidate_id: str,
  overrides: dict[str, str],
  receipt_dir: Path,
  proposal: dict[str, Any] | None = None,
  worker_id: str | None = None,
  claim_ttl_s: int | None = None,
) -> dict[str, Any] | None:
  claim_path = _claim_path(spec, stage=stage, candidate_id=candidate_id, receipt_dir=receipt_dir)
  claim_path.parent.mkdir(parents=True, exist_ok=True)
  ttl_s = int(claim_ttl_s or _claim_ttl_s(spec, stage=stage))
  owner = (worker_id or os.environ.get("NMOE_AUTORESEARCH_WORKER_ID", "").strip() or os.environ.get("HOSTNAME", "").strip() or f"pid-{os.getpid()}")

  for _ in range(3):
    now_ts = _utc_now().timestamp()
    payload = {
      "schema_version": 1,
      "campaign_name": spec.name,
      "stage": stage,
      "candidate_id": candidate_id,
      "overrides": dict(overrides),
      "proposal": dict(proposal or {}),
      "owner": owner,
      "claimed_at": _utc_from_epoch(now_ts),
      "claimed_at_unix": now_ts,
      "expires_at": _utc_from_epoch(now_ts + ttl_s),
      "expires_at_unix": now_ts + ttl_s,
      "claim_path": str(claim_path),
    }
    try:
      with claim_path.open("x") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
      return payload
    except FileExistsError:
      existing = _load_json(claim_path)
      if existing is not None and _claim_active(existing, now_ts=now_ts):
        return None
      try:
        claim_path.unlink()
      except FileNotFoundError:
        pass
  return None


def release_candidate_claim(spec: CampaignSpec, *, stage: str, candidate_id: str, receipt_dir: Path) -> None:
  claim_path = _claim_path(spec, stage=stage, candidate_id=candidate_id, receipt_dir=receipt_dir)
  try:
    claim_path.unlink()
  except FileNotFoundError:
    return


def _metric_from_leaderboard_entry(entry: dict[str, Any]) -> dict[str, Any]:
  metrics = dict(entry)
  if "core_score" in metrics and "core" not in metrics:
    metrics["core"] = metrics["core_score"]
  if "val_loss_to_target" in metrics and "final_valid_loss" not in metrics:
    metrics["final_valid_loss"] = metrics["val_loss_to_target"]
  return metrics


def _metric_lookup(metrics: dict[str, Any], key: str) -> Any:
  if key in metrics and metrics.get(key) is not None:
    return metrics.get(key)
  if key == "core":
    return metrics.get("core_score")
  if key == "core_score":
    return metrics.get("core")
  if key == "final_valid_loss":
    return metrics.get("val_loss_to_target")
  if key == "val_loss_to_target":
    return metrics.get("final_valid_loss")
  return None


def _match_entry(entry: dict[str, Any], match: dict[str, str]) -> bool:
  for key, expected in match.items():
    if str(entry.get(key)) != expected:
      return False
  return True


def metrics_meet_objective_shape(spec: CampaignSpec, metrics: dict[str, Any]) -> bool:
  primary_metric = spec.objective.primary_metric
  if _metric_lookup(metrics, primary_metric) is None:
    return False
  for key in spec.objective.required_metrics:
    if _metric_lookup(metrics, key) is None:
      return False
  if spec.objective.min_core_score is not None or spec.objective.max_core_drop is not None:
    if _metric_lookup(metrics, "core") is None:
      return False
  return True


def select_baseline(spec: CampaignSpec, *, stage: str, receipt_dir: Path, leaderboard_path: Path) -> dict[str, Any] | None:
  if spec.baseline.source == "campaign_receipts":
    return select_receipt_baseline(spec, stage=stage, receipt_dir=receipt_dir)
  return select_leaderboard_baseline(spec, leaderboard_path=leaderboard_path)


def select_receipt_baseline(spec: CampaignSpec, *, stage: str, receipt_dir: Path) -> dict[str, Any] | None:
  base = receipt_dir / spec.name / stage
  if not base.exists():
    return None

  primary_metric = spec.objective.primary_metric
  direction = spec.objective.direction
  candidates: list[tuple[bool, float, str, dict[str, Any]]] = []
  for path in sorted(base.glob("*.json")):
    payload = _load_json(path)
    if not payload:
      continue
    if payload.get("status") != "completed":
      continue
    metrics = payload.get("metrics", {})
    if not metrics_meet_objective_shape(spec, metrics):
      continue
    metric_value = _metric_lookup(metrics, primary_metric)
    if metric_value is None:
      continue
    try:
      metric_value = float(metric_value)
    except (TypeError, ValueError):
      continue
    ended_at = str(payload.get("ended_at") or payload.get("started_at") or "")
    kept = bool(payload.get("decision", {}).get("kept", False))
    sort_metric = metric_value if direction == "min" else -metric_value
    candidates.append((kept, sort_metric, ended_at, payload))

  if not candidates:
    return None

  kept_candidates = [item for item in candidates if item[0]]
  pool = kept_candidates if kept_candidates else candidates
  _, _, _, payload = min(pool, key=lambda item: (item[1], item[2]))
  return {
    "source": "campaign_receipts",
    "campaign_name": payload.get("campaign_name"),
    "candidate_id": payload.get("candidate_id"),
    "receipt_path": payload.get("receipt_path"),
    "metrics": dict(payload.get("metrics", {})),
  }


def select_leaderboard_baseline(spec: CampaignSpec, *, leaderboard_path: Path) -> dict[str, Any] | None:
  payload = _load_json(leaderboard_path)
  if not payload:
    return None

  runs = payload.get("runs", [])
  if not isinstance(runs, list):
    return None

  matched = []
  for entry in runs:
    if not isinstance(entry, dict):
      continue
    if spec.baseline.match and not _match_entry(entry, spec.baseline.match):
      continue
    metrics = _metric_from_leaderboard_entry(entry)
    if not metrics_meet_objective_shape(spec, metrics):
      continue
    matched.append(metrics)

  if not matched:
    return None

  reverse = spec.objective.direction == "max"
  best = sorted(matched, key=lambda entry: float(entry[spec.objective.primary_metric]), reverse=reverse)[0]
  return {
    "source": "leaderboard",
    "match": dict(spec.baseline.match),
    "metrics": best,
  }


def stage_receipts(spec: CampaignSpec, *, stage: str, receipt_dir: Path) -> list[dict[str, Any]]:
  base = receipt_dir / spec.name / stage
  if not base.exists():
    return []

  payloads: list[dict[str, Any]] = []
  for path in sorted(base.glob("*.json")):
    payload = _load_json(path)
    if payload is None:
      continue
    payloads.append(payload)
  return payloads


def receipt_overrides(payload: dict[str, Any]) -> dict[str, str]:
  raw = payload.get("overrides", {})
  if not isinstance(raw, dict):
    return {}
  return {str(key): _normalize_value(value) for key, value in raw.items()}


def overrides_signature(overrides: dict[str, str]) -> tuple[tuple[str, str], ...]:
  return tuple(sorted((str(key), _normalize_value(value)) for key, value in overrides.items()))


def resolve_recorded_path(root: Path, value: str | Path | None) -> Path | None:
  if value is None:
    return None
  path = Path(str(value))
  if path.is_absolute():
    return path
  return root / path


def speedrun_base_overrides(spec: CampaignSpec) -> dict[str, str]:
  if spec.speedrun is None:
    raise CampaignError(f"campaign {spec.name} does not define runtime.speedrun")
  config_path = spec.root / "configs" / "speedrun" / f"{spec.speedrun.config}.toml"
  if not config_path.exists():
    raise CampaignError(f"speedrun config not found for campaign {spec.name}: {config_path}")
  data = load_toml(config_path)
  return {
    key: _normalize_value(data[key])
    for key in spec.mutation.allowed_overrides
    if key in data
  }


def _axis_current_index(values: tuple[str, ...], current_value: str) -> int:
  for index, value in enumerate(values):
    if _value_equal(value, current_value):
      return index

  current_num = _maybe_number(current_value)
  numeric = [_maybe_number(value) for value in values]
  if current_num is not None and all(num is not None for num in numeric):
    best_index = 0
    best_distance = float("inf")
    for index, value_num in enumerate(numeric):
      assert value_num is not None
      distance = abs(value_num - current_num)
      if distance < best_distance:
        best_distance = distance
        best_index = index
    return best_index

  return 0


def _coordinate_candidates(
  spec: CampaignSpec,
  *,
  stage: str,
  receipt_dir: Path,
  leaderboard_path: Path | None = None,
) -> list[CampaignCandidateProposal]:
  if spec.search is None:
    raise CampaignError(f"campaign {spec.name} does not define a [search] section")

  receipts = stage_receipts(spec, stage=stage, receipt_dir=receipt_dir)
  tried: set[tuple[tuple[str, str], ...]] = set()
  for payload in receipts:
    status = str(payload.get("status", "")).strip()
    if status in {"planned", "running"}:
      tried.add(overrides_signature(receipt_overrides(payload)))
      continue
    if status in {"completed", "failed"} and metrics_meet_objective_shape(spec, dict(payload.get("metrics", {}))):
      tried.add(overrides_signature(receipt_overrides(payload)))
  for payload in active_stage_claims(spec, stage=stage, receipt_dir=receipt_dir):
    tried.add(overrides_signature(_claim_overrides(payload)))
  search = spec.search

  baseline = select_baseline(
    spec,
    stage=stage,
    receipt_dir=receipt_dir,
    leaderboard_path=leaderboard_path or (spec.root / "LEADERBOARD.json"),
  )
  if baseline is None:
    seed_signature = overrides_signature(search.seed_overrides)
    if seed_signature not in tried:
      candidate_id = f"{search.candidate_prefix}-seed" if search.seed_overrides else f"{search.candidate_prefix}-baseline"
      reason = "no campaign baseline exists; run the configured seed candidate first" if search.seed_overrides else "no campaign baseline exists; run the canonical baseline first"
      return [
        CampaignCandidateProposal(
          candidate_id=candidate_id,
          overrides=dict(search.seed_overrides),
          reason=reason,
        )
      ]
    return []

  current_overrides: dict[str, str] = {}
  baseline_ended_at = ""
  if baseline.get("source") == "campaign_receipts":
    baseline_path = resolve_recorded_path(spec.root, baseline.get("receipt_path"))
    baseline_receipt = _load_json(baseline_path) if baseline_path is not None else None
    if baseline_receipt is not None:
      current_overrides = receipt_overrides(baseline_receipt)
      baseline_ended_at = str(baseline_receipt.get("ended_at") or baseline_receipt.get("started_at") or "")

  axis_failure_counts: dict[str, int] = {}
  if baseline_ended_at:
    for payload in receipts:
      status = str(payload.get("status", "")).strip()
      if status not in {"completed", "failed"}:
        continue
      ended_at = str(payload.get("ended_at") or payload.get("started_at") or "")
      if not ended_at or ended_at <= baseline_ended_at:
        continue
      if bool(payload.get("decision", {}).get("kept", False)):
        continue
      proposal = payload.get("proposal", {})
      if not isinstance(proposal, dict):
        continue
      axis = str(proposal.get("axis", "")).strip()
      if axis:
        axis_failure_counts[axis] = axis_failure_counts.get(axis, 0) + 1

  base_overrides = speedrun_base_overrides(spec)
  campaign_defaults = dict(base_overrides)
  campaign_defaults.update(search.seed_overrides)
  axis_proposals: list[tuple[int, int, list[CampaignCandidateProposal]]] = []
  for axis_index, axis in enumerate(search.axes):
    current_value = current_overrides.get(axis.key, campaign_defaults.get(axis.key, ""))
    current_index = _axis_current_index(axis.values, current_value)
    proposals_for_axis: list[CampaignCandidateProposal] = []
    for distance in range(1, len(axis.values) + 1):
      for sign in (1, -1):
        index = current_index + (sign * distance)
        if index < 0 or index >= len(axis.values):
          continue
        proposed_value = axis.values[index]
        candidate_overrides = dict(current_overrides)
        base_value = base_overrides.get(axis.key)
        if base_value is not None and _value_equal(proposed_value, base_value):
          candidate_overrides.pop(axis.key, None)
        else:
          candidate_overrides[axis.key] = proposed_value
        signature = overrides_signature(candidate_overrides)
        if signature in tried:
          continue
        proposals_for_axis.append(
          CampaignCandidateProposal(
            candidate_id=f"{search.candidate_prefix}-{axis.key}-{_slug(proposed_value)}",
            overrides=candidate_overrides,
            reason=f"coordinate search on {axis.key}: {current_value or '<unset>'} -> {proposed_value}",
            axis=axis.key,
            current_value=current_value or None,
            proposed_value=proposed_value,
          )
        )
    if proposals_for_axis:
      axis_proposals.append((axis_failure_counts.get(axis.key, 0), axis_index, proposals_for_axis))
  axis_proposals.sort(key=lambda item: (item[0], item[1]))
  proposals: list[CampaignCandidateProposal] = []
  max_axis_depth = max((len(items) for _, _, items in axis_proposals), default=0)
  for depth in range(max_axis_depth):
    for _, _, items in axis_proposals:
      if depth < len(items):
        proposals.append(items[depth])
  return proposals


def _receipt_summary(payload: dict[str, Any]) -> dict[str, Any]:
  metrics = dict(payload.get("metrics", {}))
  return {
    "candidate_id": payload.get("candidate_id"),
    "status": payload.get("status"),
    "ended_at": payload.get("ended_at"),
    "overrides": receipt_overrides(payload),
    "metrics": {
      key: metrics.get(key)
      for key in (
        "final_valid_loss",
        "final_loss",
        "core",
        "core_score",
        "train_time_ms_excl_valid",
        "valid_time_ms",
      )
      if key in metrics
    },
    "decision": dict(payload.get("decision", {})),
  }


def _openai_chat_json(
  *,
  model: str,
  system_prompt: str,
  user_prompt: str,
  api_key: str,
  api_base: str,
) -> dict[str, Any] | None:
  url = api_base.rstrip("/") + "/chat/completions"
  body = {
    "model": model,
    "temperature": 0,
    "response_format": {"type": "json_object"},
    "messages": [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
    ],
  }
  request = urllib.request.Request(
    url,
    data=json.dumps(body).encode("utf-8"),
    headers={
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}",
    },
    method="POST",
  )
  try:
    with urllib.request.urlopen(request, timeout=45) as response:
      payload = json.loads(response.read().decode("utf-8"))
  except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
    return None

  try:
    content = payload["choices"][0]["message"]["content"]
  except Exception:
    return None
  try:
    return json.loads(content)
  except Exception:
    return None


def _llm_coordinate_candidate(
  spec: CampaignSpec,
  *,
  stage: str,
  receipt_dir: Path,
  leaderboard_path: Path | None = None,
) -> CampaignCandidateProposal | None:
  if spec.search is None:
    raise CampaignError(f"campaign {spec.name} does not define a [search] section")

  candidates = _coordinate_candidates(spec, stage=stage, receipt_dir=receipt_dir, leaderboard_path=leaderboard_path)
  if not candidates:
    return None

  search = spec.search
  if not _env_truthy("NMOE_AUTORESEARCH_ENABLE_LLM"):
    return None
  api_key = os.environ.get("OPENAI_API_KEY", "").strip()
  if not api_key:
    return None

  model = search.llm_model or os.environ.get("NMOE_AUTORESEARCH_MODEL", "").strip() or "gpt-4o-mini"
  api_base = search.llm_api_base or os.environ.get("OPENAI_API_BASE", "").strip() or "https://api.openai.com/v1"

  receipts = stage_receipts(spec, stage=stage, receipt_dir=receipt_dir)
  recent = sorted(
    (
      payload for payload in receipts
      if str(payload.get("status", "")).strip() in {"completed", "failed"}
    ),
    key=lambda payload: str(payload.get("ended_at") or payload.get("started_at") or ""),
  )[-search.llm_recent_receipts:]

  baseline = select_baseline(
    spec,
    stage=stage,
    receipt_dir=receipt_dir,
    leaderboard_path=leaderboard_path or (spec.root / "LEADERBOARD.json"),
  )
  payload = {
    "campaign": spec.to_dict(),
    "stage": stage,
    "baseline": baseline,
    "recent_receipts": [_receipt_summary(item) for item in recent],
    "candidate_options": [
      {
        "candidate_id": proposal.candidate_id,
        "overrides": proposal.overrides,
        "reason": proposal.reason,
      }
      for proposal in candidates[:search.llm_candidate_limit]
    ],
  }
  system_prompt = (
    "You are selecting the next bounded autoresearch candidate for a sparse-training benchmark. "
    "Choose exactly one candidate_id from candidate_options. Prefer candidates that plausibly improve "
    "validation loss while preserving or improving CORE and routing health. Return strict JSON with keys "
    "'candidate_id' and 'reason'."
  )
  user_prompt = json.dumps(payload, indent=2, sort_keys=True)
  response = _openai_chat_json(
    model=model,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    api_key=api_key,
    api_base=api_base,
  )
  if not isinstance(response, dict):
    return None
  selected_id = str(response.get("candidate_id", "")).strip()
  llm_reason = str(response.get("reason", "")).strip()
  for proposal in candidates:
    if proposal.candidate_id == selected_id:
      return CampaignCandidateProposal(
        candidate_id=proposal.candidate_id,
        overrides=dict(proposal.overrides),
        reason=llm_reason or proposal.reason,
        axis=proposal.axis,
        current_value=proposal.current_value,
        proposed_value=proposal.proposed_value,
      )
  return None


def propose_next_candidate(
  spec: CampaignSpec,
  *,
  stage: str,
  receipt_dir: Path,
  leaderboard_path: Path | None = None,
) -> CampaignCandidateProposal | None:
  if spec.search is None:
    raise CampaignError(f"campaign {spec.name} does not define a [search] section")
  search = spec.search
  if search.strategy == "llm_coordinate_descent":
    llm_choice = _llm_coordinate_candidate(
      spec,
      stage=stage,
      receipt_dir=receipt_dir,
      leaderboard_path=leaderboard_path,
    )
    if llm_choice is not None:
      return llm_choice
  candidates = _coordinate_candidates(
    spec,
    stage=stage,
    receipt_dir=receipt_dir,
    leaderboard_path=leaderboard_path,
  )
  return candidates[0] if candidates else None


def evaluate_metrics(spec: CampaignSpec, metrics: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
  primary_metric = spec.objective.primary_metric
  primary_value = _metric_lookup(metrics, primary_metric)
  if primary_value is None:
    return {
      "primary_metric": primary_metric,
      "direction": spec.objective.direction,
      "current_value": None,
      "baseline_value": None,
      "constraints_pass": False,
      "constraint_failures": [f"missing primary metric {primary_metric!r}"],
      "improved": False,
      "kept": False,
      "reason": f"missing primary metric {primary_metric!r}",
    }

  failures: list[str] = []
  target_reached = bool(metrics.get("target_reached", False))
  final_loss = metrics.get("final_loss")
  core_score = _metric_lookup(metrics, "core")

  for key in spec.objective.required_metrics:
    if _metric_lookup(metrics, key) is None:
      failures.append(f"missing_metric:{key}")

  if spec.objective.require_target_reached and not target_reached:
    failures.append("target_reached=false")
  if spec.objective.max_final_loss is not None:
    if final_loss is None or float(final_loss) > spec.objective.max_final_loss:
      failures.append(f"final_loss>{spec.objective.max_final_loss}")
  if spec.objective.min_core_score is not None:
    if core_score is None or float(core_score) < spec.objective.min_core_score:
      failures.append(f"core_score<{spec.objective.min_core_score}")
  baseline_metrics = baseline.get("metrics", {}) if baseline is not None else {}
  baseline_core_score = _metric_lookup(baseline_metrics, "core")
  if spec.objective.max_core_drop is not None and baseline is not None:
    if core_score is None:
      failures.append("missing_metric:core")
    elif baseline_core_score is None:
      failures.append("baseline_missing_metric:core")
    elif float(core_score) < float(baseline_core_score) - spec.objective.max_core_drop:
      failures.append(f"core_drop>{spec.objective.max_core_drop}")

  current_value = float(primary_value)
  baseline_value = None
  if baseline is not None and _metric_lookup(baseline_metrics, primary_metric) is not None:
    baseline_value = float(_metric_lookup(baseline_metrics, primary_metric))

  if baseline_value is None:
    improved = True
  elif spec.objective.direction == "min":
    improved = current_value <= (baseline_value - spec.objective.min_delta_abs)
  else:
    improved = current_value >= (baseline_value + spec.objective.min_delta_abs)

  constraints_pass = not failures
  kept = constraints_pass and improved
  if not constraints_pass:
    reason = "constraints failed: " + ", ".join(failures)
  elif baseline_value is None:
    reason = "no baseline available"
  elif kept:
    reason = f"{primary_metric} improved from {baseline_value} to {current_value}"
  else:
    reason = f"{primary_metric} did not improve over baseline {baseline_value}"

  return {
    "primary_metric": primary_metric,
    "direction": spec.objective.direction,
    "current_value": current_value,
    "baseline_value": baseline_value,
    "current_core_score": None if core_score is None else float(core_score),
    "baseline_core_score": None if baseline_core_score is None else float(baseline_core_score),
    "constraints_pass": constraints_pass,
    "constraint_failures": failures,
    "improved": improved,
    "kept": kept,
    "reason": reason,
  }
