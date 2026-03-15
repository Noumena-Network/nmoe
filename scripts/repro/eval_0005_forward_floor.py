#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import tomllib
from pathlib import Path


def _checkpoint_step(name: str) -> int | None:
  if name.isdigit():
    return int(name)
  if name.startswith('iter_'):
    suffix = name[len('iter_'):]
    if suffix.isdigit():
      return int(suffix)
  return None


def _resolve_checkpoint_dir(checkpoint_root: Path) -> Path:
  if checkpoint_root.is_dir() and (checkpoint_root / 'rd.pt').exists():
    return checkpoint_root

  candidates: list[tuple[int, Path]] = []
  for path in checkpoint_root.iterdir():
    if not path.is_dir():
      continue
    step = _checkpoint_step(path.name)
    if step is None:
      continue
    if not (path / 'rd.pt').exists():
      continue
    candidates.append((step, path))

  if not candidates:
    raise RuntimeError(f'no checkpoint directories found under {checkpoint_root}')

  return max(candidates, key=lambda item: item[0])[1]


def _load_cfg(dtype: str, speedrun_data_root: Path):
  from nmoe.config import Config, upgrade_cfg_dict

  with open('configs/speedrun/moe.toml', 'rb') as f:
    cfg_dict = tomllib.load(f)
  cfg_dict['dtype'] = dtype
  cfg_dict['resume'] = False
  cfg_dict['steps'] = int(cfg_dict.get('validation_steps', 20))
  cfg_dict['data_path'] = str(speedrun_data_root / 'train')
  cfg_dict['validation_data_path'] = str(speedrun_data_root / 'val')
  cfg_dict['collect_update_stats'] = False
  return Config(**upgrade_cfg_dict(cfg_dict))


def _load_split_checkpoint(model, checkpoint_root: Path, rank: int) -> Path:
  import torch

  latest = _resolve_checkpoint_dir(checkpoint_root)
  map_location = f'cuda:{torch.cuda.current_device()}'
  rd = torch.load(latest / 'rd.pt', map_location=map_location, weights_only=False)
  dp = torch.load(latest / f'dp_rank_{rank:03d}.pt', map_location=map_location, weights_only=False)
  model.load_state_dict(rd['model_dense'], strict=False)
  model.load_state_dict(dp['model_expert'], strict=False)
  return latest


def main() -> None:
  import torch
  import torch.distributed as dist

  from nmoe import runtime
  runtime._maybe_add_repo_third_party_to_sys_path()
  from nmoe.data.loader import build_loader
  from nmoe.model import Transformer
  from quack.linear_cross_entropy import chunked_linear_cross_entropy

  ap = argparse.ArgumentParser(description='Evaluate a bf16 checkpoint through bf16 or nvfp4 forward paths for 0005.')
  ap.add_argument('--checkpoint-root', required=True)
  ap.add_argument('--dtype', required=True, choices=['bf16', 'nvfp4'])
  ap.add_argument('--speedrun-data-root', default='/data/speedrun')
  ap.add_argument('--out-json', required=True)
  args = ap.parse_args()

  cfg = _load_cfg(args.dtype, Path(args.speedrun_data_root))
  attn = getattr(cfg, 'attn', None)
  attn_local = getattr(cfg, 'attn_local', None)
  cap = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
  is_sm90 = cap == (9, 0)
  allow_sm90_bf16 = bool(cfg.dtype == 'bf16' and is_sm90 and 'mla' not in (attn, attn_local))
  rank, world = runtime.init(cfg.seed, require_b200=not allow_sm90_bf16)

  quiet = (lambda *_a, **_k: None) if rank != 0 else print
  loader_cfg = dataclasses.replace(
    cfg,
    data_path=str(getattr(cfg, 'validation_data_path')),
    flow_mode=None,
    steps=int(cfg.validation_steps),
  )
  v_loader, _ = build_loader(loader_cfg, rank, world, split='valid', print_fn=quiet)
  model = Transformer(cfg).cuda()
  model.eval()
  latest = _load_split_checkpoint(model, Path(args.checkpoint_root), rank)

  loss_sum = torch.zeros((), device='cuda', dtype=torch.float32)
  tok_count = torch.zeros((), device='cuda', dtype=torch.float32)
  ignore_index = int(cfg.eos_token_id) if getattr(cfg, 'loss_mask_eos', True) else -100

  with torch.no_grad():
    for _ in range(int(cfg.validation_steps)):
      inp, tgt = v_loader.next()
      hidden = model(inp, return_hidden=True)
      logits_gain = float(getattr(model, 'fp4_logits_gain', getattr(model, 'logits_scale_factor', 1.0)))
      x = (hidden * logits_gain).reshape(-1, hidden.shape[-1])
      t = tgt.reshape(-1)
      loss_sum += chunked_linear_cross_entropy(
        x,
        model.lm_head.weight,
        t,
        chunk_size=8192,
        ignore_index=ignore_index,
        reduction='sum',
        tuned=False,
      ).float()
      tok_count += (t != ignore_index).float().sum()
  v_loader.close()

  if world > 1 and dist.is_initialized():
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)

  value = float((loss_sum / tok_count.clamp(min=1.0)).item())
  if rank == 0:
    out_path = Path(args.out_json)
    out_path.write_text(json.dumps({'dtype': args.dtype, 'valid_loss': value, 'checkpoint': str(latest)}, indent=2))
    print(json.dumps({'dtype': args.dtype, 'valid_loss': value, 'checkpoint': str(latest)}))


if __name__ == '__main__':
  main()
