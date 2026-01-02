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
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from nmoe.config import Config, fingerprint
from nmoe.model import Transformer
from nmoe.data.loader import build_loader
from nmoe.opt import build_optimizer, update_lr, step
from nmoe.checkpoint import Checkpointer, load_checkpoint, save_checkpoint
from nmoe.metrics import init_metrics, start_metrics, log_training_step, stop_metrics, register_model_timers, cuda_time
from nmoe.experiments import ExperimentTracker
from nmoe import runtime
from nmoe.eval.hooks import maybe_schedule_eval


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def train(cfg: Config):
  """Train MoE model. One clear path: forward → loss → backward → step → log → checkpoint."""
  rank, world = runtime.init(cfg.seed)
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
  timers_on = os.getenv('NMOE_TIMERS', '1') not in ('0', 'false', 'False')
  time_ctx = cuda_time if timers_on else (lambda _tag: nullcontext())
  nvtx_on = os.getenv('NMOE_NVTX', '0') in ('1', 'true', 'True')
  nvtx_ok = bool(nvtx_on and torch.cuda.is_available() and hasattr(torch.cuda, 'nvtx') and hasattr(torch.cuda.nvtx, 'range'))
  nvtx_ctx = (torch.cuda.nvtx.range if nvtx_ok else (lambda _tag: nullcontext()))

  exp_tracker: ExperimentTracker | None = None
  run_id = os.getenv("NMOE_RUN", "")
  if rank == 0:
    exp_tracker = ExperimentTracker(cfg)
    run_id = exp_tracker.start_run(run_id=run_id or None)

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
  optimizer, dense_groups = build_optimizer(model, cfg)
  metrics_state = init_metrics(model, cfg.seq_len)
  metrics_ctx = start_metrics(run_id=run_id, metrics_dir=cfg.metrics_dir)
  zero2_state = {}
  start_step, tokens_seen, zero2_state = load_checkpoint(checkpointer, model, optimizer, loader, plan, cfg, rank, print)
  last_loss: torch.Tensor | None = None
  config_fingerprint = fingerprint(cfg)
  checkpoint_every = getattr(cfg, 'checkpoint_every', 100)

  try:
    with nvtx_ctx('train/run'):
      for step_num in range(start_step, cfg.steps):
        lr = update_lr(optimizer, dense_groups, step_num, tokens_seen, cfg)

        t0 = time.perf_counter()
        inputs, targets = loader.next()
        loader_wait_ms = (time.perf_counter() - t0) * 1000.0

        with nvtx_ctx('train/fwd_total'), time_ctx('time_ms/fwd_total'):
          logits = model(inputs)

        with nvtx_ctx('train/loss'), time_ctx('time_ms/loss'):
          loss_unreduced = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1), reduction='none')
          mask = (targets != cfg.eos_token_id).reshape(-1).float()
          loss = (loss_unreduced * mask).sum() / mask.sum().clamp(min=1.0)

        model.zero_grad(set_to_none=True)
        with nvtx_ctx('train/bwd_total'), time_ctx('time_ms/bwd_total'):
          loss.backward()

        with nvtx_ctx('train/opt_step'), time_ctx('time_ms/opt_step'):
          step(model, optimizer, dense_groups, zero2_state, cfg, world)

        tokens_this_step = int(inputs.numel())
        tokens_seen += cfg.batch_size * cfg.seq_len
        last_loss = loss.detach()

        s = step_num + 1
        log_training_step(
          s,
          model=model,
          loss=loss.detach(),
          lr=lr,
          tokens_this_step=tokens_this_step,
          state=metrics_state,
          print_fn=print,
          ctx=metrics_ctx,
          loader_wait_ms=loader_wait_ms,
        )

        # Opportunistically schedule evaluation (async or inline), if enabled
        try:
          maybe_schedule_eval(s, cfg, model, run_id, print)
        except Exception as e:
          if rank == 0:
            print(f"[eval] scheduling failed: {e}")

        save_checkpoint(
          checkpointer, s, tokens_seen, model, optimizer, loader, plan,
          zero2_state, cfg, rank, config_fingerprint, checkpoint_every, print
        )

    if rank == 0:
      print(f"[nmoe] Training complete. {tokens_seen:,} tokens.")

    final_loss = float(last_loss.item()) if last_loss is not None else 0.0
    results = {'final_loss': final_loss, 'tokens_seen': tokens_seen, 'steps_completed': cfg.steps}
    if exp_tracker is not None and rank == 0:
      exp_tracker.end_run(run_id, "completed", results)
    return results
  except Exception:
    if exp_tracker is not None and rank == 0:
      exp_tracker.end_run(run_id, "failed")
    raise
  finally:
    checkpointer.close()
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

  cfg = Config(**cfg_dict)

  try:
    train(cfg)
  finally:
    runtime.finalize()


if __name__ == '__main__':
  main()
