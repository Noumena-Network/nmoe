r"""
nmoe: noumena's moe training library

   _ __   _ __ ___   ___   ___
  | '_ \ | '_ ` _ \ / _ \ / _ \
  | | | || | | | | | (_) |  __/
  |_| |_||_| |_| |_|\___/ \___|

Usage:
  python -m nmoe.train configs/moonlet.toml
  torchrun --nproc_per_node=8 -m nmoe.train configs/moonlight.toml
"""
import os
import sys
import tomllib
import time
import dataclasses
import math
import json
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nmoe import runtime
runtime._maybe_add_repo_third_party_to_sys_path()

from nmoe.config import Config, fingerprint, upgrade_cfg_dict
from nmoe.model import Transformer, MoE
from nmoe.data.loader import build_loader
from nmoe.opt import build_optimizer, update_lr, step
from nmoe.checkpoint import Checkpointer, load_checkpoint, save_checkpoint
from nmoe.metrics import init_metrics, start_metrics, log_training_step, stop_metrics, register_model_timers, cuda_time, collect_router_stats
from nmoe.experiments import ExperimentTracker
from nmoe.eval.hooks import maybe_schedule_eval
from nmoe.token_bytes import loss_nats_to_bpb, token_bytes

try:
  from quack.linear_cross_entropy import chunked_linear_cross_entropy
except Exception as e:  # pragma: no cover
  raise RuntimeError(
    "quack is required for memory-efficient vocab-scale cross-entropy. "
    "Make it importable (e.g. add third_party/quack to PYTHONPATH)."
  ) from e

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# TODO(deepseek-v3-parity): Align our MoE routing/load-balancing with DeepSeek-V3 (arXiv:2412.19437v2):
# - Wire `aux_loss_alpha` to the sequence-wise balance loss and add it to the training loss when enabled.
# - Schedule `router_bias_update_rate` (γ): 1e-3 early training, 0 late (token-based), to freeze a learned policy.
# - Add node-limited routing for multi-node runs (e.g. M<=4 nodes/token) to cap comm and stabilize throughput.
# - DeepSeek uses grad clipping (norm=1.0); this conflicts with nmoe's "no gradient clipping" contract—decide explicitly.

LEADERBOARD_PATH = Path(__file__).parent.parent / "LEADERBOARD.json"

def _collect_moe_grad_health_items(model: Transformer, world: int) -> list[tuple[str, float]]:
  zero_w1: list[torch.Tensor] = []
  zero_w2: list[torch.Tensor] = []
  zero_w3: list[torch.Tensor] = []
  mean_w1: list[torch.Tensor] = []
  mean_w2: list[torch.Tensor] = []
  mean_w3: list[torch.Tensor] = []
  max_w1: list[torch.Tensor] = []
  max_w2: list[torch.Tensor] = []
  max_w3: list[torch.Tensor] = []

  with torch.no_grad():
    for blk in model.blocks:
      ffn = getattr(blk, "ffn", None)
      if not isinstance(ffn, MoE):
        continue
      for name, param, zeros, means, maxes in (
        ("w1", ffn.W1, zero_w1, mean_w1, max_w1),
        ("w2", ffn.W2, zero_w2, mean_w2, max_w2),
        ("w3", ffn.W3, zero_w3, mean_w3, max_w3),
      ):
        del name
        if (not param.requires_grad) or param.grad is None:
          continue
        grad = param.grad.detach().float()
        zeros.append((grad == 0).to(dtype=torch.float32).mean())
        means.append(grad.abs().mean())
        maxes.append(grad.abs().max())

  def _reduce(vals: list[torch.Tensor], *, op: str) -> float | None:
    if not vals:
      return None
    stacked = torch.stack(vals)
    if op == "max":
      acc = stacked.max()
    elif op == "mean":
      acc = stacked.mean()
    else:
      raise ValueError(f"unsupported reduction op={op!r}")
    if world > 1 and dist.is_available() and dist.is_initialized():
      if op == "max":
        dist.all_reduce(acc, op=dist.ReduceOp.MAX)
      else:
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        acc.div_(float(world))
    return float(acc.item())

  items: list[tuple[str, float]] = []
  for name, zeros, means, maxes in (
    ("w1", zero_w1, mean_w1, max_w1),
    ("w2", zero_w2, mean_w2, max_w2),
    ("w3", zero_w3, mean_w3, max_w3),
  ):
    zero_v = _reduce(zeros, op="mean")
    mean_v = _reduce(means, op="mean")
    max_v = _reduce(maxes, op="max")
    if zero_v is not None:
      items.append((f"train/moe_grad_zero_frac_{name}", zero_v))
    if mean_v is not None:
      items.append((f"train/moe_grad_abs_mean_{name}", mean_v))
    if max_v is not None:
      items.append((f"train/moe_grad_abs_max_{name}", max_v))
  return items

def _speedrun_hardware() -> str:
  if not torch.cuda.is_available():
    return "unknown"
  cap: tuple[int, int] | None = None
  try:
    cap = tuple(torch.cuda.get_device_capability())
  except Exception:
    cap = None
  if cap == (10, 0):
    return "B200"
  if cap == (9, 0):
    return "H100"
  try:
    name = str(torch.cuda.get_device_name(0) or "").strip()
  except Exception:
    name = ""
  if "B200" in name or "Blackwell" in name:
    return "B200"
  if "H100" in name:
    return "H100"
  return "unknown"


def _save_speedrun_result(cfg: Config, results: dict, wall_time_s: float, *, world: int, run_id: str):
  """Append speedrun result to LEADERBOARD.json for tracking."""
  from datetime import datetime

  # Extract config name from preset (e.g., "speedrun_moe" -> "moe")
  config_name = cfg.preset.replace("speedrun_", "")

  target_reached = bool(results.get("target_reached", False))
  core_score = results.get("core", None)
  if not target_reached:
    print("[speedrun] Leaderboard: skipped (target_loss not reached)")
    return
  if core_score is None:
    print("[speedrun] Leaderboard: skipped (CORE score missing)")
    return

  final_loss = results.get("val_loss_to_target", None)
  if final_loss is None:
    final_loss = results.get("final_loss", 0.0)

  steps = results.get("step_to_target", None)
  if steps is None:
    steps = results.get("steps_completed", 0)

  tokens = results.get("tokens_to_target", None)
  if tokens is None:
    tokens = results.get("tokens_seen", 0)

  train_time_ms_to_target = results.get("train_time_ms_to_target", None)
  if train_time_ms_to_target is not None:
    wall_time_s = float(train_time_ms_to_target) / 1000.0

  entry = {
    "config": config_name,
    "hardware": _speedrun_hardware(),
    "n_gpus": int(world),
    "dtype": cfg.dtype,
    "final_loss": float(final_loss),
    "core_score": float(core_score),
    "tokens": int(tokens),
    "steps": int(steps),
    "wall_time_s": round(wall_time_s, 1),
    "target_reached": True,
    "date": datetime.now().isoformat(),
    "experiment_id": cfg.experiment_id,
    "run_id": str(run_id),
  }

  # Load existing leaderboard or create new
  if LEADERBOARD_PATH.exists():
    try:
      data = json.loads(LEADERBOARD_PATH.read_text())
    except Exception:
      data = {"runs": []}
  else:
    data = {"runs": []}

  data["runs"].append(entry)

  # Sort by wall time (asc), then steps (asc).
  data["runs"].sort(key=lambda x: (x.get("wall_time_s") or 1e18, x.get("steps") or 1e18))

  LEADERBOARD_PATH.write_text(json.dumps(data, indent=2) + "\n")
  print(f"[speedrun] Result saved to {LEADERBOARD_PATH}")


def train(cfg: Config):
  """Train MoE model. One clear path: forward → loss → backward → step → log → checkpoint."""
  # Default contract: SM100a/B200 only. Explicitly allow a narrow bring-up target:
  # BF16 speedruns/research on H100 (SM90) with SDPA (no MLA, no blockscaled).
  has_moe = bool(getattr(cfg, "n_layers", 0) > getattr(cfg, "n_dense_layers", 0))
  attn = str(getattr(cfg, "attn", "") or "")
  attn_local = str(getattr(cfg, "attn_local", "") or "")

  cap: tuple[int, int] | None = None
  if torch.cuda.is_available():
    try:
      cap = tuple(torch.cuda.get_device_capability())
    except Exception:
      cap = None

  is_sm90 = cap == (9, 0)
  allow_sm90_bf16 = bool(cfg.dtype == "bf16" and is_sm90 and "mla" not in (attn, attn_local))
  require_b200 = not allow_sm90_bf16
  rank, world = runtime.init(cfg.seed, require_b200=require_b200)

  if allow_sm90_bf16 and has_moe:
    # Fail fast with an actionable message if the BF16 RDEP backend isn't built.
    try:
      from nmoe.csrc import rdep as _rdep_ext  # noqa: F401
    except Exception as e:
      raise RuntimeError(
        "BF16 MoE on H100 (SM90) requires building the nmoe CUDA extension for SM90.\n"
        "From the repo root:\n"
        "  cd nmoe/csrc && make clean && make NMOE_CUDA_ARCH=90\n"
      ) from e
  #TODO(EM): these if blocks and raises are ugly. rewrite or move
  if cfg.batch_size <= 0:
    raise ValueError(f"batch_size must be > 0 (got {cfg.batch_size})")
  if world > 1 and (cfg.batch_size % world) != 0:
    raise ValueError(
      f"batch_size ({cfg.batch_size}) must be divisible by world_size ({world}). "
      "Uneven per-rank microbatches break ZeRO-2 AVG semantics and exact token accounting."
    )
  if cfg.seq_len <= 0:
    raise ValueError(f"seq_len must be > 0 (got {cfg.seq_len})")
  if cfg.n_activated_experts is not None and cfg.n_routed_experts is not None:
    if cfg.n_activated_experts <= 0:
      raise ValueError(f"n_activated_experts must be > 0 (got {cfg.n_activated_experts})")
    if cfg.n_activated_experts > cfg.n_routed_experts:
      raise ValueError(
        f"n_activated_experts ({cfg.n_activated_experts}) must be <= n_routed_experts ({cfg.n_routed_experts})"
      )

  # ---------------------------------------------------------------------------
  # NVFP4-only I/O gains
  # ---------------------------------------------------------------------------
  if cfg.dtype == "nvfp4":
    if (cfg.fp4_embed_gain is None) != (cfg.fp4_logits_gain is None):
      raise ValueError(
        "NVFP4 requires both fp4_embed_gain and fp4_logits_gain to be set together "
        "(or leave both unset for unit-gain defaults)."
      )
    if cfg.fp4_embed_gain is None:
      # Keep NVFP4 on the same default embedding/logit contract as bf16/fp8.
      # Non-unit gains remain available for explicit ablations only.
      cfg.fp4_embed_gain = 1.0
      cfg.fp4_logits_gain = 1.0
    if not (cfg.fp4_embed_gain > 0.0 and cfg.fp4_logits_gain > 0.0):
      raise ValueError(
        f"NVFP4 gains must be > 0 (fp4_embed_gain={cfg.fp4_embed_gain}, fp4_logits_gain={cfg.fp4_logits_gain})"
      )
  else:
    if cfg.fp4_embed_gain is not None or cfg.fp4_logits_gain is not None:
      # Speedrun configs pin NVFP4 gains; allow bf16/fp8 overrides without requiring config edits.
      if rank == 0:
        print("[nmoe] ignoring fp4_embed_gain/fp4_logits_gain (NVFP4-only) for dtype!=nvfp4")
      cfg.fp4_embed_gain = None
      cfg.fp4_logits_gain = None
  timers_on = os.getenv('NMOE_TIMERS', '1') not in ('0', 'false', 'False')
  time_ctx = cuda_time if timers_on else (lambda _tag: nullcontext())
  barriers_on = os.getenv("NMOE_DEBUG_BARRIERS", "0") in ("1", "true", "True")
  nvtx_on = os.getenv('NMOE_NVTX', '0') in ('1', 'true', 'True')
  nvtx_ok = bool(nvtx_on and torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx') and hasattr(torch.cuda.nvtx, 'range'))
  nvtx_ctx = (torch.cuda.nvtx.range if nvtx_ok else (lambda _tag: nullcontext()))

  dense_opt = str(os.getenv("NMOE_DENSE_OPT", "")).strip().lower()
  use_ref_dense_opt = dense_opt in ("torch_adamw",)

  exp_tracker: ExperimentTracker | None = None
  run_id = os.getenv("NMOE_RUN", "")
  if rank == 0:
    exp_tracker = ExperimentTracker(cfg)
    run_id = exp_tracker.start_run(run_id=run_id or None)

  # Broadcast run_id so all ranks write metrics under the same run directory.
  if world > 1 and dist.is_available() and dist.is_initialized():
    max_bytes = 256
    if rank == 0:
      run_bytes = run_id.encode("utf-8")
      if len(run_bytes) == 0:
        raise RuntimeError("rank0 run_id is empty")
      if len(run_bytes) > max_bytes:
        raise RuntimeError(f"run_id too long to broadcast ({len(run_bytes)} bytes > {max_bytes})")
      n = torch.tensor([len(run_bytes)], device="cuda", dtype=torch.int32)
      buf = torch.zeros(max_bytes, device="cuda", dtype=torch.uint8)
      buf[:len(run_bytes)] = torch.tensor(list(run_bytes), device="cuda", dtype=torch.uint8)
    else:
      n = torch.zeros(1, device="cuda", dtype=torch.int32)
      buf = torch.zeros(max_bytes, device="cuda", dtype=torch.uint8)
    dist.broadcast(n, src=0)
    dist.broadcast(buf, src=0)
    n_int = int(n.item())
    run_id = bytes(buf[:n_int].tolist()).decode("utf-8")
    if len(run_id) == 0:
      raise RuntimeError("broadcast run_id is empty")

  checkpointer = Checkpointer(
    base=cfg.checkpoint_dir,
    keep_last=getattr(cfg, 'checkpoint_keep_last_n', 5),
    async_io=True,
    async_max_queue=1,
  )

  # Build components
  loader, plan = build_loader(cfg, rank, world)
  model = Transformer(cfg).cuda()
  model.init_weights()
  model.train()
  register_model_timers(model)
  has_moe = bool(cfg.n_layers > cfg.n_dense_layers)
  if use_ref_dense_opt and has_moe:
    raise RuntimeError(
      "NMOE_DENSE_OPT=torch_adamw is only supported for dense-only runs "
      "(set n_dense_layers == n_layers)."
    )

  model_fwd: torch.nn.Module = model
  if use_ref_dense_opt and world > 1:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    model_fwd = torch.nn.parallel.DistributedDataParallel(
      model,
      device_ids=[local_rank],
      output_device=local_rank,
      broadcast_buffers=False,
    )

  if use_ref_dense_opt and rank == 0:
    print("[nmoe] dense optimizer override: DDP + torch.optim.AdamW (bypassing ZeRO-2)")

  optimizer, muon_optimizer, dense_groups = build_optimizer(model, cfg)
  extra_optimizers = {"muon": muon_optimizer} if muon_optimizer is not None else None
  dense_optimizer: torch.optim.Optimizer | None = None
  if use_ref_dense_opt:
    dense_optimizer = torch.optim.AdamW(
      dense_groups,
      betas=(cfg.adam_beta1, cfg.adam_beta2),
      eps=cfg.adam_eps,
    )
  metrics_state = init_metrics(model, cfg.seq_len)
  metrics_ctx = start_metrics(run_id=run_id, metrics_dir=cfg.metrics_dir)
  tok_bytes: torch.Tensor | None = None
  zero2_state = {}
  start_step, tokens_seen, zero2_state = load_checkpoint(
    checkpointer,
    model,
    (dense_optimizer if dense_optimizer is not None else optimizer),
    loader,
    plan,
    cfg,
    rank,
    print,
    extra_optimizers=extra_optimizers,
  )
  last_loss: torch.Tensor | None = None
  log_every = max(1, int(getattr(cfg, 'log_every', 1)))
  config_fingerprint = fingerprint(cfg)
  checkpoint_every = getattr(cfg, 'checkpoint_every', 100)

  try:
    # Speedrun scoring clock: wall time excluding validation time (modded-nanogpt style).
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    wall_t0 = time.perf_counter()
    valid_time_s = 0.0
    target_step: int | None = None
    target_tokens_seen: int | None = None
    target_val_loss: float | None = None
    target_train_time_ms: float | None = None
    last_valid_loss: float | None = None
    last_valid_bpb: float | None = None
    stop_reason: str | None = None
    steps_completed = int(start_step)

    with nvtx_ctx('train/run'):
      for step_num in range(start_step, cfg.steps):
        lr = update_lr(optimizer, muon_optimizer, dense_groups, step_num, tokens_seen, cfg)
        lr_dense = float(lr)
        lr_router = None
        lr_expert = None
        lr_muon = None
        try:
          for g in dense_groups:
            if g.get("name") == "router":
              lr_router = float(g.get("lr"))
              break
        except Exception:
          lr_router = None
        try:
          if optimizer is not None and getattr(optimizer, "param_groups", None):
            lr_expert = float(optimizer.param_groups[0].get("lr", lr_dense))
        except Exception:
          lr_expert = None
        try:
          if muon_optimizer is not None and getattr(muon_optimizer, "param_groups", None):
            lr_muon = float(muon_optimizer.param_groups[0].get("lr", lr_dense))
        except Exception:
          lr_muon = None

        t0 = time.perf_counter()
        inputs, targets = loader.next()
        loader_wait_ms = (time.perf_counter() - t0) * 1000.0

        if barriers_on and world > 1 and dist.is_available() and dist.is_initialized():
          with time_ctx("time_ms/barrier_after_loader"):
            dist.barrier()

        with nvtx_ctx('train/fwd_total'), time_ctx('time_ms/fwd_total'):
          hidden = model_fwd(inputs, return_hidden=True)

        if barriers_on and world > 1 and dist.is_available() and dist.is_initialized():
          with time_ctx("time_ms/barrier_after_fwd"):
            dist.barrier()

        probe_output_path = ((step_num + 1) == 1) or (((step_num + 1) % log_every) == 0) or ((step_num + 1) == int(cfg.steps))
        output_path_items: list[tuple[str, float]] = []

        with nvtx_ctx('train/loss'), time_ctx('time_ms/loss'):
          if not hidden.is_cuda:
            raise RuntimeError("chunked_linear_cross_entropy requires CUDA tensors.")
          logits_gain = float(getattr(model, "fp4_logits_gain", getattr(model, "logits_scale_factor", 1.0)))
          x = (hidden * logits_gain).reshape(-1, hidden.shape[-1])
          if probe_output_path:
            hidden.retain_grad()
            x.retain_grad()
          t = targets.reshape(-1)
          ignore_index = int(cfg.eos_token_id) if getattr(cfg, "loss_mask_eos", True) else -100
          if x.shape[0] % 8 != 0:
            raise ValueError(f"chunked CE requires (batch*seq) % 8 == 0, got {x.shape[0]}")
          ce_loss = chunked_linear_cross_entropy(
            x,
            model.lm_head.weight,
            t,
            chunk_size=8192,
            ignore_index=ignore_index,
            reduction="mean",
            tuned=False,
          )

          # Aux loss: load-balance regularization (Switch-style, differentiable through router weights)
          aux_alpha = float(getattr(cfg, 'aux_loss_alpha', 0.0))
          if aux_alpha > 0.0 and has_moe:
            moe_layers = [blk.ffn for blk in model.blocks if isinstance(getattr(blk, 'ffn', None), MoE)]
            aux_losses = [m.last_aux_loss for m in moe_layers if m.last_aux_loss is not None]
            if aux_losses:
              aux_loss = torch.stack(aux_losses).mean()
              loss = ce_loss + aux_alpha * aux_loss
            else:
              loss = ce_loss
          else:
            loss = ce_loss

        model.zero_grad(set_to_none=True)
        with nvtx_ctx('train/bwd_total'), time_ctx('time_ms/bwd_total'):
          loss.backward()

        if probe_output_path:
          try:
            output_path_items.append(("train/lm_head_logits_gain", float(logits_gain)))
            hidden_grad = getattr(hidden, "grad", None)
            x_grad = getattr(x, "grad", None)
            hidden_grad_norm = None
            x_grad_norm = None
            if hidden_grad is not None:
              hidden_grad_norm = float(hidden_grad.detach().float().norm().item())
              output_path_items.append(("train/lm_head_hidden_grad_norm", hidden_grad_norm))
            if x_grad is not None:
              x_grad_norm = float(x_grad.detach().float().norm().item())
              output_path_items.append(("train/lm_head_scaled_hidden_grad_norm", x_grad_norm))
            if hidden_grad_norm is not None and x_grad_norm is not None and x_grad_norm > 0.0:
              output_path_items.append(("train/lm_head_hidden_to_scaled_hidden_grad_ratio", hidden_grad_norm / x_grad_norm))
          except Exception:
            pass

        moe_grad_items: list[tuple[str, float]] = []
        if ((step_num + 1) % log_every) == 0:
          moe_grad_items = _collect_moe_grad_health_items(model, world)

        if barriers_on and world > 1 and dist.is_available() and dist.is_initialized():
          with time_ctx("time_ms/barrier_after_bwd"):
            dist.barrier()

        s = step_num + 1
        collect_update_stats = bool(getattr(cfg, 'collect_update_stats', True)) and ((s == 1) or ((s % log_every) == 0))
        if bool(getattr(cfg, 'collect_update_stats', True)) and int(getattr(cfg, 'steps', 0)) == int(s):
          collect_update_stats = True

        with nvtx_ctx('train/opt_step'), time_ctx('time_ms/opt_step'):
          if dense_optimizer is not None:
            dense_optimizer.step()
            if muon_optimizer is not None:
              muon_optimizer.step()
            if optimizer is not None:
              optimizer.step()
          else:
            step(
              model,
              optimizer,
              muon_optimizer,
              dense_groups,
              zero2_state,
              cfg,
              world,
              collect_update_stats=collect_update_stats,
            )

        tokens_this_step = int(inputs.numel())
        tokens_seen += cfg.batch_size * cfg.seq_len
        last_loss = loss.detach()

        steps_completed = int(s)
        log_training_step(
          s,
          model=model,
          loss=loss.detach(),
          lr_dense=lr_dense,
          lr_router=lr_router,
          lr_expert=lr_expert,
          lr_muon=lr_muon,
          tokens_seen=tokens_seen,
          tokens_this_step=tokens_this_step,
          state=metrics_state,
          print_fn=print,
          ctx=metrics_ctx,
          loader_wait_ms=loader_wait_ms,
        )
        extra_log_items = []
        extra_log_items.extend(output_path_items)
        extra_log_items.extend(moe_grad_items)
        if extra_log_items and rank == 0 and getattr(metrics_ctx, "writer", None) is not None:
          try:
            metrics_ctx.writer.insert_many(step=s, items=extra_log_items)
            metrics_ctx.writer.flush_parquet(step=s)
          except Exception:
            pass

        # Fail-fast for sweeps: detect obvious numerical failures early on log steps.
        do_log = (s == 1) or ((s % log_every) == 0) or (s == int(cfg.steps))
        if do_log:
          # Check the local (unreduced) loss tensor for finiteness; we only sync at log cadence.
          try:
            finite = bool(torch.isfinite(loss.detach().float()).item())
          except Exception:
            finite = True

          bad = 0
          if not finite:
            bad = 1
          if world > 1 and dist.is_available() and dist.is_initialized():
            t = torch.tensor([bad], device="cuda", dtype=torch.int32)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            bad = int(t.item())
          if bad:
            stop_reason = f"non_finite_loss(step={s})"
            if rank == 0 and getattr(metrics_ctx, "writer", None) is not None:
              try:
                metrics_ctx.writer.insert_many(step=s, items=[
                  ("speedrun/failed", 1.0),
                  ("speedrun/fail_code", 1.0),
                ])
                metrics_ctx.writer.flush_parquet(step=s)
              except Exception:
                pass
            break

          # Router collapse signature: only K experts active everywhere.
          try:
            if cfg.n_routed_experts is not None and cfg.n_activated_experts is not None:
              E = int(cfg.n_routed_experts)
              K = int(cfg.n_activated_experts)
              if E > K and K > 0:
                per, _ = collect_router_stats(model)
                cv_collapse = 100.0 * math.sqrt(float(E - K) / float(K))
                collapse = 0
                for item in per:
                  cv = item.get("cv")
                  ea = item.get("experts_active")
                  if cv is None or ea is None:
                    continue
                  if int(ea) <= (K + 1) and float(cv) >= (0.97 * cv_collapse):
                    collapse = 1
                    break
                if world > 1 and dist.is_available() and dist.is_initialized():
                  t = torch.tensor([collapse], device="cuda", dtype=torch.int32)
                  dist.all_reduce(t, op=dist.ReduceOp.MAX)
                  collapse = int(t.item())
                if collapse:
                  stop_reason = f"router_collapse(step={s})"
                  if rank == 0 and getattr(metrics_ctx, "writer", None) is not None:
                    try:
                      metrics_ctx.writer.insert_many(step=s, items=[
                        ("speedrun/failed", 1.0),
                        ("speedrun/fail_code", 2.0),
                      ])
                      metrics_ctx.writer.flush_parquet(step=s)
                    except Exception:
                      pass
                  break
          except Exception:
            pass

        # Validation: loss-only cross-entropy on a fixed token stream.
        # All ranks participate; we aggregate exact (loss_sum, token_count).
        if getattr(cfg, "validation_enabled", False):
          v_at = getattr(cfg, "validation_at_steps", None) or []
          do_valid = False
          if v_at:
            do_valid = int(s) in v_at or int(s) == int(cfg.steps)
          else:
            v_every = int(getattr(cfg, "validation_every", 0) or 0)
            do_valid = v_every > 0 and ((s % v_every) == 0 or s == cfg.steps)
          if do_valid:
            v_path = getattr(cfg, "validation_data_path", None)
            if not v_path:
              if rank == 0:
                print("[valid] skipped (validation_data_path not set)")
            else:
              v_steps = int(getattr(cfg, "validation_steps", 0) or 0)
              if v_steps <= 0:
                raise ValueError(f"validation_steps must be > 0 when validation is enabled (got {v_steps})")

              # Exclude validation time from speedrun clock.
              if torch.cuda.is_available():
                torch.cuda.synchronize()
              v_t0 = time.perf_counter()

              cfg_v = dataclasses.replace(cfg, data_path=str(v_path), flow_mode=None, steps=v_steps)
              model_fwd.eval()
              with torch.no_grad():
                quiet = (lambda *_a, **_k: None) if rank != 0 else print
                v_loader, _ = build_loader(cfg_v, rank, world, split="valid", print_fn=quiet)
                loss_sum = torch.zeros((), device="cuda", dtype=torch.float32)
                tok_count = torch.zeros((), device="cuda", dtype=torch.float32)
                bpb_enabled = bool(getattr(cfg, "validation_log_bpb", False))
                loss_sum_bpb = torch.zeros((), device="cuda", dtype=torch.float32) if bpb_enabled else None
                byte_count = torch.zeros((), device="cuda", dtype=torch.float32) if bpb_enabled else None
                if bpb_enabled and tok_bytes is None:
                  tok_bytes = token_bytes(cfg.tokenizer, cfg.vocab_size, device=torch.device("cuda"))
                for _ in range(v_steps):
                  v_inp, v_tgt = v_loader.next()
                  v_hidden = model_fwd(v_inp, return_hidden=True)
                  logits_gain = float(getattr(model, "fp4_logits_gain", getattr(model, "logits_scale_factor", 1.0)))
                  x = (v_hidden * logits_gain).reshape(-1, v_hidden.shape[-1])
                  t = v_tgt.reshape(-1)
                  ignore_index = int(cfg.eos_token_id) if getattr(cfg, "loss_mask_eos", True) else -100
                  if x.shape[0] % 8 != 0:
                    raise ValueError(f"chunked CE requires (batch*seq) % 8 == 0, got {x.shape[0]}")
                  loss_sum += chunked_linear_cross_entropy(
                    x,
                    model.lm_head.weight,
                    t,
                    chunk_size=8192,
                    ignore_index=ignore_index,
                    reduction="sum",
                    tuned=False,
                  ).float()
                  tok_count += (t != ignore_index).float().sum()

                  if bpb_enabled:
                    # bpb is defined over raw bytes; exclude special tokens (tiktoken->0 bytes).
                    # For GPT-2 this mainly excludes EOS and any padded IDs.
                    assert loss_sum_bpb is not None and byte_count is not None and tok_bytes is not None
                    ignore_bpb = int(cfg.eos_token_id)
                    loss_sum_bpb += chunked_linear_cross_entropy(
                      x,
                      model.lm_head.weight,
                      t,
                      chunk_size=8192,
                      ignore_index=ignore_bpb,
                      reduction="sum",
                      tuned=False,
                    ).float()
                    tt = t[t != ignore_bpb].to(dtype=torch.long)
                    if tt.numel() > 0:
                      byte_count += tok_bytes.index_select(0, tt).to(dtype=torch.float32).sum()
                v_loader.close()

                if world > 1 and torch.distributed.is_initialized():
                  torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
                  torch.distributed.all_reduce(tok_count, op=torch.distributed.ReduceOp.SUM)
                  if bpb_enabled:
                    assert loss_sum_bpb is not None and byte_count is not None
                    torch.distributed.all_reduce(loss_sum_bpb, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(byte_count, op=torch.distributed.ReduceOp.SUM)
                v_loss = loss_sum / tok_count.clamp(min=1.0)
                v_loss_val = float(v_loss.item())
                v_bpb_val: float | None = None
                v_bytes_val: float | None = None
                last_valid_loss = v_loss_val
                if bpb_enabled:
                  assert loss_sum_bpb is not None and byte_count is not None
                  v_bpb_val = float(loss_nats_to_bpb(loss_sum_bpb, byte_count).item())
                  v_bytes_val = float(byte_count.item())
                  last_valid_bpb = v_bpb_val

                if torch.cuda.is_available():
                  torch.cuda.synchronize()
                valid_time_s += time.perf_counter() - v_t0

                # Track speedrun clock for plotting/debugging (rank0 only).
                if rank == 0 and getattr(metrics_ctx, "writer", None) is not None:
                  try:
                    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
                    train_ms = wall_ms - (valid_time_s * 1000.0)
                    metrics_ctx.writer.insert_many(step=s, items=[
                      ("speedrun/wall_time_ms", float(wall_ms)),
                      ("speedrun/train_time_ms_excl_valid", float(train_ms)),
                      ("speedrun/valid_time_ms", float(valid_time_s * 1000.0)),
                    ])
                    metrics_ctx.writer.flush_parquet(step=s)
                  except Exception as e:
                    print(f"[valid] warning: failed to write speedrun metrics: {e}")

                if rank == 0 and getattr(metrics_ctx, "writer", None) is not None:
                  try:
                    items = [
                      ("valid/loss", float(v_loss_val)),
                      ("valid/tokens", float(tok_count.item())),
                    ]
                    if v_bpb_val is not None:
                      items.append(("valid/bpb", float(v_bpb_val)))
                    if v_bytes_val is not None:
                      items.append(("valid/bytes", float(v_bytes_val)))
                    metrics_ctx.writer.insert_many(step=s, items=items)
                    metrics_ctx.writer.flush_parquet(step=s)
                  except Exception as e:
                    print(f"[valid] warning: failed to write metrics to DuckDB: {e}")
                if rank == 0:
                  if v_bpb_val is None:
                    print(f"[valid] step={s} loss={v_loss_val:.4f} tokens={int(tok_count.item()):,}")
                  else:
                    print(
                      f"[valid] step={s} loss={v_loss_val:.4f} bpb={v_bpb_val:.4f} "
                      f"tokens={int(tok_count.item()):,} bytes={int(v_bytes_val or 0):,}"
                    )

                # Check for target loss (speedrun stopping criterion)
                target_loss = getattr(cfg, "target_loss", None)
                if target_loss is not None and v_loss_val <= target_loss:
                  # Compute speedrun score (time excludes validation).
                  if torch.cuda.is_available():
                    torch.cuda.synchronize()
                  wall_ms = (time.perf_counter() - wall_t0) * 1000.0
                  train_ms = wall_ms - (valid_time_s * 1000.0)
                  target_step = int(s)
                  target_tokens_seen = int(tokens_seen)
                  target_val_loss = float(v_loss_val)
                  target_train_time_ms = float(train_ms)
                  stop_reason = "target_reached"
                  if rank == 0:
                    step_avg = train_ms / max(1, target_step)
                    print(
                      f"[speedrun] target={float(target_loss):.4f} reached: "
                      f"step={target_step} val_loss={target_val_loss:.4f} "
                      f"train_time_ms={train_ms:.0f} step_avg_ms={step_avg:.2f} "
                      f"tokens_seen={target_tokens_seen:,}"
                    )
                    if getattr(metrics_ctx, "writer", None) is not None:
                      try:
                        metrics_ctx.writer.insert_many(step=s, items=[
                          ("speedrun/target_reached", 1.0),
                          ("speedrun/step_to_target", float(target_step)),
                          ("speedrun/tokens_to_target", float(target_tokens_seen)),
                          ("speedrun/train_time_ms_to_target", float(train_ms)),
                        ])
                        metrics_ctx.writer.flush_parquet(step=s)
                      except Exception:
                        pass
                  break
              model.train()

        if stop_reason is not None:
          break

        # Choice-based eval (fast forward-only scoring). All ranks participate.
        eval_every = int(getattr(cfg, "eval_every", 0) or 0)
        if eval_every > 0 and (s % eval_every) == 0 and (not getattr(cfg, "eval_enabled", False)):
          try:
            from nmoe.eval.choices import run_eval, format_results
            eval_results = run_eval(
              model,
              cfg,
              rank=rank,
              world=world,
              max_examples=int(getattr(cfg, "eval_budget_max_examples", 500)),
            )
            if rank == 0:
              print(f"[eval] step={s} {format_results(eval_results)}")
              if getattr(metrics_ctx, "writer", None) is not None:
                items = []
                for tname, r in eval_results.items():
                  items.append((f"eval_choices/{tname}/acc", float(r.get("acc", 0.0))))
                  items.append((f"eval_choices/{tname}/centered_acc", float(r.get("centered_acc", 0.0))))
                  items.append((f"eval_choices/{tname}/n", float(r.get("n", 0.0))))
                metrics_ctx.writer.insert_many(step=s, items=items)
                metrics_ctx.writer.flush_parquet(step=s)
          except Exception as e:
            if rank == 0:
              print(f"[eval] failed: {e}")

        # Opportunistically schedule evaluation (async or inline), if enabled
        try:
          maybe_schedule_eval(s, cfg, model, run_id, print)
        except Exception as e:
          if rank == 0:
            print(f"[eval] scheduling failed: {e}")

        save_checkpoint(
          checkpointer, s, tokens_seen, model,
          (dense_optimizer if dense_optimizer is not None else optimizer),
          loader, plan,
          zero2_state, cfg, rank, config_fingerprint, checkpoint_every, print,
          extra_optimizers=extra_optimizers,
        )

    if rank == 0:
      print(f"[nmoe] Training complete. {tokens_seen:,} tokens.")

    core_score: float | None = None
    want_core = str(getattr(cfg, "eval_tasks", "")).lower() == "core"
    run_core = bool(getattr(cfg, "eval_enabled", False)) and want_core
    # Speedrun contract: if we stop at target loss, run full CORE suite automatically.
    if (not run_core) and want_core and stop_reason == "target_reached" and str(getattr(cfg, "preset", "")).startswith("speedrun_"):
      run_core = True
    if run_core:
      try:
        from nmoe.eval.core.runner import run_core_live
        # For speedruns that hit the target, always run the full suite.
        if stop_reason == "target_reached" and str(getattr(cfg, "preset", "")).startswith("speedrun_"):
          max_per_task = -1
          max_time_s = 0.0
        else:
          max_per_task = int(getattr(cfg, "eval_budget_max_examples", 500))
          max_time_s = float(getattr(cfg, "eval_budget_max_time_s", 0.0))
        summary = run_core_live(
          step=int(steps_completed),
          cfg_dict=dataclasses.asdict(cfg),
          cfg=cfg,
          model=model,
          run_id=str(run_id),
          tasks_file=str(getattr(cfg, "eval_tasks_file", "configs/eval/core.toml")),
          bundle_dir=str(cfg.eval_bundle_dir) if cfg.eval_bundle_dir else str(Path(cfg.data_root) / "eval" / "eval_bundle"),
          max_per_task=max_per_task,
          max_time_s=max_time_s,
        )
        core_score = float(summary.get("CORE", 0.0))
        if rank == 0 and getattr(metrics_ctx, "writer", None) is not None:
          try:
            metrics_ctx.writer.insert_many(step=int(steps_completed), items=[("eval/CORE", float(core_score))])
            metrics_ctx.writer.flush_parquet(step=int(steps_completed))
          except Exception:
            pass
      except Exception as e:
        if rank == 0:
          print(f"[core] failed: {e}")

    final_loss = float(last_loss.item()) if last_loss is not None else 0.0
    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    train_ms = wall_ms - (valid_time_s * 1000.0)
    results = {
      'final_loss': final_loss,
      'tokens_seen': int(tokens_seen),
      'steps_completed': int(steps_completed),
      'train_time_ms_excl_valid': float(train_ms),
      'valid_time_ms': float(valid_time_s * 1000.0),
      'core': float(core_score) if core_score is not None else None,
      'stop_reason': stop_reason or "completed",
      'target_reached': bool(target_step is not None),
      'step_to_target': int(target_step) if target_step is not None else None,
      'tokens_to_target': int(target_tokens_seen) if target_tokens_seen is not None else None,
      'val_loss_to_target': float(target_val_loss) if target_val_loss is not None else None,
      'final_valid_loss': float(last_valid_loss) if last_valid_loss is not None else None,
      'final_valid_bpb': float(last_valid_bpb) if last_valid_bpb is not None else None,
      'train_time_ms_to_target': float(target_train_time_ms) if target_train_time_ms is not None else None,
    }
    if exp_tracker is not None and rank == 0:
      # Map stop_reason to specific status for experiments DB
      if stop_reason is None:
        status = "completed"
      elif stop_reason == "target_reached":
        status = "completed_target"
      elif "non_finite" in stop_reason:
        status = "failed_nan"
      elif "router_collapse" in stop_reason:
        status = "failed_collapse"
      else:
        status = "failed"
      exp_tracker.end_run(run_id, status, results)

    # Save to leaderboard if this is a speedrun
    if rank == 0 and cfg.preset.startswith("speedrun_"):
      _save_speedrun_result(cfg, results, wall_ms / 1000.0, world=world, run_id=str(run_id))

    return results
  except Exception:
    if exp_tracker is not None and rank == 0:
      exp_tracker.end_run(run_id, "failed")
    raise
  finally:
    checkpointer.close()
    try:
      loader.close()
    except Exception:
      pass
    stop_metrics(metrics_ctx)
    if exp_tracker is not None:
      exp_tracker.close()


def main():
  """Entry point. Loads config and starts training.

  Usage:
    python -m nmoe.train <config.toml> [--key=value ...]

  CLI overrides (applied after TOML):
    --dtype=fp8        Override precision (bf16, fp8, nvfp4)
    --steps=2000       Override training steps
    --batch_size=16    Override batch size
    --resume=false     Override resume behavior

  Environment overrides (lowest priority):
    NMOE_DTYPE, NMOE_STEPS, etc.
  """
  if '--help' in sys.argv or '-h' in sys.argv:
    print(__doc__)
    print(main.__doc__)
    sys.exit(0)

  if len(sys.argv) < 2:
    print("Usage: python -m nmoe.train <config.toml> [--key=value ...]", file=sys.stderr)
    sys.exit(1)

  # Load base config from TOML
  with open(sys.argv[1], 'rb') as f:
    cfg_dict = tomllib.load(f)

  # Apply environment variable overrides (NMOE_DTYPE, NMOE_STEPS, etc.)
  for key in ['dtype', 'steps', 'batch_size', 'seq_len', 'resume']:
    env_key = f'NMOE_{key.upper()}'
    if env_key in os.environ:
      val = os.environ[env_key]
      # Parse booleans and ints
      if val.lower() in ('true', 'false'):
        val = val.lower() == 'true'
      elif val.isdigit():
        val = int(val)
      cfg_dict[key] = val

  # Apply CLI overrides (--key=value)
  for arg in sys.argv[2:]:
    if not arg.startswith('--'):
      continue

    # Support boolean flags without '=' for common toggles.
    if '=' not in arg:
      if arg == '--resume':
        cfg_dict['resume'] = True
      elif arg == '--no-resume':
        cfg_dict['resume'] = False
      continue

    key, val = arg[2:].split('=', 1)
    # Parse booleans and ints
    if val.lower() in ('true', 'false'):
      val = val.lower() == 'true'
    elif val.lstrip('-').isdigit():
      val = int(val)
    elif val.replace('.', '', 1).lstrip('-').isdigit():
      val = float(val)
    cfg_dict[key] = val

  cfg_dict = upgrade_cfg_dict(cfg_dict)

  cfg = Config(**cfg_dict)

  try:
    train(cfg)
  finally:
    runtime.finalize()


if __name__ == '__main__':
  main()
