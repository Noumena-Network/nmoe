from pathlib import Path

import pytest

from nmoe.campaigns import (
  CampaignError,
  claim_candidate,
  evaluate_metrics,
  load_campaign,
  parse_candidate_overrides,
  propose_next_candidate,
  release_candidate_claim,
  select_baseline,
  validate_candidate_overrides,
  write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_campaign_speedrun_super_smoke():
  spec = load_campaign(REPO_ROOT, "speedrun_super_smoke")

  assert spec.name == "speedrun_super_smoke"
  assert spec.runner == "speedrun"
  assert spec.kind == "speedrun"
  assert spec.speedrun is not None
  assert spec.speedrun.config == "super"
  assert spec.speedrun.eval_enabled is False
  assert spec.speedrun.train_tokens == "16M"
  assert spec.speedrun.val_tokens == "1M"
  assert spec.stage("smoke").steps == 32
  assert "activation" in spec.mutation.allowed_overrides


def test_load_campaign_speedrun_super_benchmark():
  spec = load_campaign(REPO_ROOT, "speedrun_super_benchmark")

  assert spec.name == "speedrun_super_benchmark"
  assert spec.runner == "speedrun"
  assert spec.kind == "speedrun"
  assert spec.speedrun is not None
  assert spec.speedrun.config == "super"
  assert spec.speedrun.eval_enabled is False
  assert spec.speedrun.train_tokens is None
  assert spec.speedrun.val_tokens is None
  assert spec.stage("benchmark").steps == 512
  assert spec.objective.max_final_loss == 10.0
  assert "router_bias_update_rate" in spec.mutation.allowed_overrides
  assert spec.search is not None
  assert spec.search.strategy == "coordinate_descent"
  assert spec.search.max_trials == 4
  assert spec.search.max_no_improve == 2
  assert spec.search.axes[0].key == "aux_loss_alpha"
  assert spec.search.axes[0].values[2] == "0.0001"


def test_load_campaign_speedrun_super_research():
  spec = load_campaign(REPO_ROOT, "speedrun_super_research")

  assert spec.name == "speedrun_super_research"
  assert spec.runner == "speedrun"
  assert spec.speedrun is not None
  assert spec.speedrun.eval_enabled is True
  assert spec.objective.primary_metric == "final_valid_loss"
  assert spec.objective.required_metrics == ("core",)
  assert spec.objective.max_core_drop == pytest.approx(0.002)
  assert spec.search is not None
  assert spec.search.strategy == "llm_coordinate_descent"
  assert spec.search.max_trials == 8
  assert spec.search.max_no_improve == 3
  assert spec.search.llm_candidate_limit == 12
  assert spec.search.seed_overrides == {"aux_loss_alpha": "0.0001"}
  assert [axis.key for axis in spec.search.axes] == [
    "aux_loss_alpha",
    "router_bias_update_rate",
    "warmup_steps",
    "lr_dense",
    "lr_router",
  ]
  assert spec.search.axes[0].values[4] == "0.00012"
  assert spec.search.axes[1].values[-1] == "0.003"
  assert spec.search.axes[3].values[-1] == "0.0022"


def test_validate_candidate_overrides_rejects_out_of_scope_keys():
  spec = load_campaign(REPO_ROOT, "speedrun_super_smoke")

  allowed = parse_candidate_overrides([
    "activation=relu_squared",
    "lr_dense=0.0018",
  ])
  validate_candidate_overrides(spec, allowed)

  disallowed = parse_candidate_overrides(["batch_size=64"])
  with pytest.raises(CampaignError):
    validate_candidate_overrides(spec, disallowed)


def test_campaign_receipt_baseline_prefers_kept_candidate(tmp_path: Path):
  spec = load_campaign(REPO_ROOT, "speedrun_super_smoke")

  kept_receipt = tmp_path / spec.name / "smoke" / "20260308T000000Z_kept.json"
  discarded_receipt = tmp_path / spec.name / "smoke" / "20260308T010000Z_discarded.json"

  write_json(
    kept_receipt,
    {
      "campaign_name": spec.name,
      "candidate_id": "kept",
      "stage": "smoke",
      "status": "completed",
      "receipt_path": str(kept_receipt),
      "ended_at": "2026-03-08T00:00:00Z",
      "metrics": {"final_loss": 4.0},
      "decision": {"kept": True},
    },
  )
  write_json(
    discarded_receipt,
    {
      "campaign_name": spec.name,
      "candidate_id": "discarded",
      "stage": "smoke",
      "status": "completed",
      "receipt_path": str(discarded_receipt),
      "ended_at": "2026-03-08T01:00:00Z",
      "metrics": {"final_loss": 3.8},
      "decision": {"kept": False},
    },
  )

  baseline = select_baseline(
    spec,
    stage="smoke",
    receipt_dir=tmp_path,
    leaderboard_path=tmp_path / "LEADERBOARD.json",
  )

  assert baseline is not None
  assert baseline["source"] == "campaign_receipts"
  assert baseline["candidate_id"] == "kept"
  assert baseline["metrics"]["final_loss"] == 4.0


def test_campaign_receipt_baseline_prefers_best_kept_candidate_not_latest(tmp_path: Path):
  spec = load_campaign(REPO_ROOT, "speedrun_super_research")
  stage_dir = tmp_path / spec.name / "benchmark"

  write_json(
    stage_dir / "20260309T190016Z_keep_better.json",
    {
      "campaign_name": spec.name,
      "candidate_id": "keep-better",
      "stage": "benchmark",
      "status": "completed",
      "receipt_path": str(stage_dir / "20260309T190016Z_keep_better.json"),
      "ended_at": "2026-03-09T19:10:16Z",
      "metrics": {
        "final_valid_loss": 5.1270,
        "final_loss": 5.1578,
        "core": -0.0156,
      },
      "decision": {"kept": True},
    },
  )
  write_json(
    stage_dir / "20260309T190047Z_keep_later_but_worse.json",
    {
      "campaign_name": spec.name,
      "candidate_id": "keep-later-but-worse",
      "stage": "benchmark",
      "status": "completed",
      "receipt_path": str(stage_dir / "20260309T190047Z_keep_later_but_worse.json"),
      "ended_at": "2026-03-09T19:20:47Z",
      "metrics": {
        "final_valid_loss": 5.1932,
        "final_loss": 5.2223,
        "core": -0.0167,
      },
      "decision": {"kept": True},
    },
  )

  baseline = select_baseline(
    spec,
    stage="benchmark",
    receipt_dir=tmp_path,
    leaderboard_path=tmp_path / "LEADERBOARD.json",
  )

  assert baseline is not None
  assert baseline["source"] == "campaign_receipts"
  assert baseline["candidate_id"] == "keep-better"
  assert baseline["metrics"]["final_valid_loss"] == pytest.approx(5.1270)


def test_evaluate_metrics_uses_campaign_baseline():
  spec = load_campaign(REPO_ROOT, "speedrun_super_smoke")
  baseline = {
    "source": "campaign_receipts",
    "metrics": {"final_loss": 4.0},
  }

  improved = evaluate_metrics(spec, {"final_loss": 3.9}, baseline)
  assert improved["constraints_pass"] is True
  assert improved["improved"] is True
  assert improved["kept"] is True

  worse = evaluate_metrics(spec, {"final_loss": 4.1}, baseline)
  assert worse["constraints_pass"] is True
  assert worse["improved"] is False
  assert worse["kept"] is False


def test_evaluate_metrics_requires_declared_metrics():
  spec = load_campaign(REPO_ROOT, "speedrun_super_research")
  baseline = {
    "source": "campaign_receipts",
    "metrics": {"final_valid_loss": 5.2000, "final_loss": 5.2200, "core": -0.0168},
  }

  decision = evaluate_metrics(spec, {"final_valid_loss": 5.1000, "final_loss": 5.1500}, baseline)
  assert decision["constraints_pass"] is False
  assert decision["kept"] is False
  assert "missing_metric:core" in decision["constraint_failures"]


def test_evaluate_metrics_rejects_core_regression_beyond_tolerance():
  spec = load_campaign(REPO_ROOT, "speedrun_super_research")
  baseline = {
    "source": "campaign_receipts",
    "metrics": {"final_valid_loss": 5.2000, "final_loss": 5.2200, "core": -0.0160},
  }

  decision = evaluate_metrics(
    spec,
    {"final_valid_loss": 5.1900, "final_loss": 5.1800, "core": -0.0185},
    baseline,
  )

  assert decision["constraints_pass"] is False
  assert decision["kept"] is False
  assert "core_drop>0.002" in decision["constraint_failures"]


def test_evaluate_metrics_allows_small_core_regression_within_tolerance():
  spec = load_campaign(REPO_ROOT, "speedrun_super_research")
  baseline = {
    "source": "campaign_receipts",
    "metrics": {"final_valid_loss": 5.2000, "final_loss": 5.2200, "core": -0.0168},
  }

  decision = evaluate_metrics(
    spec,
    {"final_valid_loss": 5.1949, "final_loss": 5.1900, "core": -0.0182},
    baseline,
  )

  assert decision["constraints_pass"] is True
  assert decision["kept"] is True


def test_propose_next_candidate_uses_baseline_when_none_exists(tmp_path: Path):
  spec = load_campaign(REPO_ROOT, "speedrun_super_benchmark")

  proposal = propose_next_candidate(
    spec,
    stage="benchmark",
    receipt_dir=tmp_path,
    leaderboard_path=tmp_path / "LEADERBOARD.json",
  )

  assert proposal is not None
  assert proposal.candidate_id == "auto-baseline"
  assert proposal.overrides == {}


def test_propose_next_candidate_uses_seed_candidate_for_research_campaign(tmp_path: Path):
  spec = load_campaign(REPO_ROOT, "speedrun_super_research")

  proposal = propose_next_candidate(
    spec,
    stage="benchmark",
    receipt_dir=tmp_path,
    leaderboard_path=tmp_path / "LEADERBOARD.json",
  )

  assert proposal is not None
  assert proposal.candidate_id == "research-seed"
  assert proposal.overrides == {"aux_loss_alpha": "0.0001"}


def test_active_claim_blocks_duplicate_seed_candidate(tmp_path: Path):
  spec = load_campaign(REPO_ROOT, "speedrun_super_research")
  claim = claim_candidate(
    spec,
    stage="benchmark",
    candidate_id="research-seed",
    overrides={"aux_loss_alpha": "0.0001"},
    receipt_dir=tmp_path,
    proposal={"axis": "aux_loss_alpha"},
    worker_id="worker-a",
  )

  assert claim is not None

  proposal = propose_next_candidate(
    spec,
    stage="benchmark",
    receipt_dir=tmp_path,
    leaderboard_path=tmp_path / "LEADERBOARD.json",
  )

  assert proposal is None

  release_candidate_claim(
    spec,
    stage="benchmark",
    candidate_id="research-seed",
    receipt_dir=tmp_path,
  )

  proposal = propose_next_candidate(
    spec,
    stage="benchmark",
    receipt_dir=tmp_path,
    leaderboard_path=tmp_path / "LEADERBOARD.json",
  )
  assert proposal is not None
  assert proposal.candidate_id == "research-seed"


def test_propose_next_candidate_does_not_use_llm_without_explicit_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  spec = load_campaign(REPO_ROOT, "speedrun_super_research")
  monkeypatch.setenv("OPENAI_API_KEY", "test-key")
  monkeypatch.delenv("NMOE_AUTORESEARCH_ENABLE_LLM", raising=False)

  def _fail_llm(*args, **kwargs):
    raise AssertionError("llm proposer should be opt-in only")

  monkeypatch.setattr("nmoe.campaigns._openai_chat_json", _fail_llm)

  proposal = propose_next_candidate(
    spec,
    stage="benchmark",
    receipt_dir=tmp_path,
    leaderboard_path=tmp_path / "LEADERBOARD.json",
  )

  assert proposal is not None
  assert proposal.candidate_id == "research-seed"
  assert proposal.overrides == {"aux_loss_alpha": "0.0001"}


def test_propose_next_candidate_preserves_seed_override_when_returning_from_baseline(tmp_path: Path):
  spec = load_campaign(REPO_ROOT, "speedrun_super_research")
  stage_dir = tmp_path / spec.name / "benchmark"

  write_json(
    stage_dir / "20260309T152511Z_research-seed.json",
    {
      "campaign_name": spec.name,
      "candidate_id": "research-seed",
      "stage": "benchmark",
      "status": "completed",
      "receipt_path": str(stage_dir / "20260309T152511Z_research-seed.json"),
      "ended_at": "2026-03-09T15:25:11Z",
      "metrics": {"final_valid_loss": 5.1986, "final_loss": 5.2233, "core": -0.0168},
      "overrides": {"aux_loss_alpha": "0.0001"},
      "proposal": {"axis": "aux_loss_alpha"},
      "decision": {"kept": True},
    },
  )
  write_json(
    stage_dir / "20260309T154433Z_research-aux_loss_alpha-0.00015.json",
    {
      "campaign_name": spec.name,
      "candidate_id": "research-aux_loss_alpha-0.00015",
      "stage": "benchmark",
      "status": "completed",
      "receipt_path": str(stage_dir / "20260309T154433Z_research-aux_loss_alpha-0.00015.json"),
      "ended_at": "2026-03-09T15:44:33Z",
      "metrics": {"final_valid_loss": 5.1950, "final_loss": 5.2207, "core": -0.0183},
      "overrides": {"aux_loss_alpha": "0.00015"},
      "proposal": {"axis": "aux_loss_alpha"},
      "decision": {"kept": True},
    },
  )

  proposal = propose_next_candidate(
    spec,
    stage="benchmark",
    receipt_dir=tmp_path,
    leaderboard_path=tmp_path / "LEADERBOARD.json",
  )

  assert proposal is not None
  assert proposal.candidate_id == "research-aux_loss_alpha-0.0001"
  assert proposal.overrides == {"aux_loss_alpha": "0.0001"}


def test_propose_next_candidate_round_robins_across_axes_after_local_trials(tmp_path: Path):
  spec = load_campaign(REPO_ROOT, "speedrun_super_research")
  stage_dir = tmp_path / spec.name / "benchmark"

  write_json(
    stage_dir / "20260309T152511Z_research-seed.json",
    {
      "campaign_name": spec.name,
      "candidate_id": "research-seed",
      "stage": "benchmark",
      "status": "completed",
      "receipt_path": str(stage_dir / "20260309T152511Z_research-seed.json"),
      "ended_at": "2026-03-09T15:25:11Z",
      "metrics": {"final_valid_loss": 5.1986, "final_loss": 5.2233, "core": -0.0168},
      "overrides": {"aux_loss_alpha": "0.0001"},
      "decision": {"kept": True},
    },
  )
  write_json(
    stage_dir / "20260309T154433Z_research-aux_loss_alpha-0.00015.json",
    {
      "campaign_name": spec.name,
      "candidate_id": "research-aux_loss_alpha-0.00015",
      "stage": "benchmark",
      "status": "completed",
      "receipt_path": str(stage_dir / "20260309T154433Z_research-aux_loss_alpha-0.00015.json"),
      "ended_at": "2026-03-09T15:44:33Z",
      "metrics": {"final_valid_loss": 5.1950, "final_loss": 5.2207, "core": -0.0183},
      "overrides": {"aux_loss_alpha": "0.00015"},
      "decision": {"kept": True},
    },
  )
  write_json(
    stage_dir / "20260309T161059Z_research-aux_loss_alpha-0.0002.json",
    {
      "campaign_name": spec.name,
      "candidate_id": "research-aux_loss_alpha-0.0002",
      "stage": "benchmark",
      "status": "completed",
      "receipt_path": str(stage_dir / "20260309T161059Z_research-aux_loss_alpha-0.0002.json"),
      "ended_at": "2026-03-09T16:23:51Z",
      "metrics": {"final_valid_loss": 5.2108, "final_loss": 5.2318, "core": -0.0222},
      "overrides": {"aux_loss_alpha": "0.0002"},
      "proposal": {"axis": "aux_loss_alpha"},
      "decision": {"kept": False},
    },
  )
  write_json(
    stage_dir / "20260309T163037Z_research-aux_loss_alpha-0.0005.json",
    {
      "campaign_name": spec.name,
      "candidate_id": "research-aux_loss_alpha-0.0005",
      "stage": "benchmark",
      "status": "completed",
      "receipt_path": str(stage_dir / "20260309T163037Z_research-aux_loss_alpha-0.0005.json"),
      "ended_at": "2026-03-09T16:43:42Z",
      "metrics": {"final_valid_loss": 5.1920, "final_loss": 5.2110, "core": -0.0208},
      "overrides": {"aux_loss_alpha": "0.0005"},
      "proposal": {"axis": "aux_loss_alpha"},
      "decision": {"kept": False},
    },
  )

  proposal = propose_next_candidate(
    spec,
    stage="benchmark",
    receipt_dir=tmp_path,
    leaderboard_path=tmp_path / "LEADERBOARD.json",
  )

  assert proposal is not None
  assert proposal.candidate_id == "research-router_bias_update_rate-0.0015"
  assert proposal.overrides == {
    "aux_loss_alpha": "0.00015",
    "router_bias_update_rate": "0.0015",
  }


def test_propose_next_candidate_walks_from_kept_receipt(tmp_path: Path):
  spec = load_campaign(REPO_ROOT, "speedrun_super_benchmark")
  stage_dir = tmp_path / spec.name / "benchmark"

  write_json(
    stage_dir / "20260309T031135Z_baseline.json",
    {
      "campaign_name": spec.name,
      "candidate_id": "baseline",
      "stage": "benchmark",
      "status": "completed",
      "receipt_path": str(stage_dir / "20260309T031135Z_baseline.json"),
      "ended_at": "2026-03-09T03:11:35Z",
      "metrics": {"final_loss": 5.248080730438232},
      "overrides": {},
      "decision": {"kept": True},
    },
  )
  write_json(
    stage_dir / "20260309T032547Z_low-aux.json",
    {
      "campaign_name": spec.name,
      "candidate_id": "low-aux",
      "stage": "benchmark",
      "status": "completed",
      "receipt_path": str(stage_dir / "20260309T032547Z_low-aux.json"),
      "ended_at": "2026-03-09T03:25:47Z",
      "metrics": {"final_loss": 5.220827102661133},
      "overrides": {"aux_loss_alpha": "0.0001"},
      "decision": {"kept": True},
    },
  )
  write_json(
    stage_dir / "20260309T045033Z_zero-aux.json",
    {
      "campaign_name": spec.name,
      "candidate_id": "zero-aux",
      "stage": "benchmark",
      "status": "completed",
      "receipt_path": str(stage_dir / "20260309T045033Z_zero-aux.json"),
      "ended_at": "2026-03-09T04:50:33Z",
      "metrics": {"final_loss": 5.228492259979248},
      "overrides": {"aux_loss_alpha": "0.0"},
      "decision": {"kept": False},
    },
  )
  write_json(
    stage_dir / "20260309T045837Z_aux-2e-4.json",
    {
      "campaign_name": spec.name,
      "candidate_id": "aux-2e-4",
      "stage": "benchmark",
      "status": "completed",
      "receipt_path": str(stage_dir / "20260309T045837Z_aux-2e-4.json"),
      "ended_at": "2026-03-09T04:58:37Z",
      "metrics": {"final_loss": 5.227102279663086},
      "overrides": {"aux_loss_alpha": "0.0002"},
      "decision": {"kept": False},
    },
  )

  proposal = propose_next_candidate(
    spec,
    stage="benchmark",
    receipt_dir=tmp_path,
    leaderboard_path=tmp_path / "LEADERBOARD.json",
  )

  assert proposal is not None
  assert proposal.candidate_id == "auto-aux_loss_alpha-0.00015"
  assert proposal.overrides == {"aux_loss_alpha": "0.00015"}
