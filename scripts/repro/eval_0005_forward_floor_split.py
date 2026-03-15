#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


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


def _load_cfg(speedrun_data_root: Path):
  from nmoe.config import Config, upgrade_cfg_dict

  with open('configs/speedrun/moe.toml', 'rb') as f:
    cfg_dict = tomllib.load(f)
  cfg_dict['dtype'] = 'bf16'
  cfg_dict['resume'] = False
  cfg_dict['steps'] = int(cfg_dict.get('validation_steps', 20))
  cfg_dict['data_path'] = str(speedrun_data_root / 'train')
  cfg_dict['validation_data_path'] = str(speedrun_data_root / 'val')
  cfg_dict['collect_update_stats'] = False
  return Config(**upgrade_cfg_dict(cfg_dict))


def _load_split_checkpoint(model, checkpoint_root: Path, rank: int) -> Path:
  latest = _resolve_checkpoint_dir(checkpoint_root)
  map_location = f'cuda:{torch.cuda.current_device()}'
  rd = torch.load(latest / 'rd.pt', map_location=map_location, weights_only=False)
  dp = torch.load(latest / f'dp_rank_{rank:03d}.pt', map_location=map_location, weights_only=False)
  model.load_state_dict(rd['model_dense'], strict=False)
  model.load_state_dict(dp['model_expert'], strict=False)
  return latest


def _decode_e8m0(scale_bytes: torch.Tensor) -> torch.Tensor:
  scale_i32 = scale_bytes.to(dtype=torch.int32)
  return torch.ldexp(torch.ones_like(scale_i32, dtype=torch.float32), scale_i32 - 127)


def _decode_nvfp4_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
  nib_i32 = nibbles.to(dtype=torch.int32)
  sign = 1.0 - 2.0 * ((nib_i32 >> 3) & 0x1).to(dtype=torch.float32)
  exp = (nib_i32 >> 1) & 0x3
  mant = (nib_i32 & 0x1).to(dtype=torch.float32)
  normal = torch.ldexp(1.0 + 0.5 * mant, exp - 1)
  subnormal = mant * 0.5
  return sign * torch.where(exp == 0, subnormal, normal)


def _dequant_fp8(q: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
  q_rows = q.squeeze(-1).to(dtype=torch.float32)
  scales = _decode_e8m0(scale_bytes.squeeze(-1)).repeat_interleave(32, dim=1)
  return (q_rows * scales[:, : q_rows.shape[1]]).to(dtype=torch.bfloat16)


def _dequant_nvfp4(q: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
  q_u8 = q.squeeze(-1).contiguous()
  lo = q_u8[:, 0::2].to(dtype=torch.int32)
  hi = q_u8[:, 1::2].to(dtype=torch.int32)
  packed = lo | (hi << 8)
  nibbles = torch.stack([
      packed & 0xF,
      (packed >> 4) & 0xF,
      (packed >> 8) & 0xF,
      (packed >> 12) & 0xF,
  ], dim=-1)
  values = _decode_nvfp4_nibbles(nibbles).reshape(q_u8.shape[0], q_u8.shape[1] * 2)
  scales = _decode_e8m0(scale_bytes.squeeze(-1)).repeat_interleave(32, dim=1)
  return (values * scales[:, : values.shape[1]]).to(dtype=torch.bfloat16)


def _quant_dequant_rows(x: torch.Tensor, profile: str) -> torch.Tensor:
  from nmoe.quant import quantize_fp8, quantize_nvfp4

  if profile == 'fp8':
    q, sfa = quantize_fp8(x)
    return _dequant_fp8(q, sfa)
  if profile == 'nvfp4':
    q, sfa = quantize_nvfp4(x)
    return _dequant_nvfp4(q, sfa)
  raise ValueError(f'unsupported profile: {profile}')


def _quant_dequant_w13(W: torch.Tensor, profile: str) -> torch.Tensor:
  E, H, Dff = W.shape
  rows = W.transpose(1, 2).contiguous().view(E * Dff, H)
  dq = _quant_dequant_rows(rows, profile)
  return dq.view(E, Dff, H).transpose(1, 2).contiguous()


def _quant_dequant_w2(W2: torch.Tensor, profile: str) -> torch.Tensor:
  E, Dff, H = W2.shape
  rows = W2.transpose(1, 2).contiguous().view(E * H, Dff)
  dq = _quant_dequant_rows(rows, profile)
  return dq.view(E, H, Dff).transpose(1, 2).contiguous()


def _sum_sq(x: torch.Tensor) -> float:
  return float(x.float().pow(2).sum().item())


def _sum_sq_diff(a: torch.Tensor, b: torch.Tensor) -> float:
  return float((a.float() - b.float()).pow(2).sum().item())


def _layer_name_to_id(name: str) -> int:
  parts = name.split('.')
  if len(parts) >= 3 and parts[0] == 'blocks' and parts[2] == 'ffn':
    return int(parts[1])
  raise ValueError(f'unexpected MoE module name: {name}')


def _collect_local_counts(eid: torch.Tensor, n_local: int, rank: int) -> torch.Tensor:
  flat = eid.reshape(-1).to(dtype=torch.int64)
  dest = torch.div(flat, n_local, rounding_mode='floor')
  local_eid = torch.remainder(flat[dest == rank], n_local)
  return torch.bincount(local_eid, minlength=n_local).to(dtype=torch.int64, device=eid.device)


def _build_valid_mask(offs_pad: torch.Tensor, counts: torch.Tensor, M_pad: int) -> torch.Tensor:
  mask = torch.zeros(M_pad, device=offs_pad.device, dtype=torch.bool)
  start = 0
  for e in range(int(counts.numel())):
    pad_end = int(offs_pad[e].item())
    count = int(counts[e].item())
    if count > 0:
      mask[start:start + count] = True
    start = pad_end
  return mask


def _analyze_moe_batch(module, x: torch.Tensor, profile: str, cache: dict[int, dict[str, torch.Tensor]], rank: int) -> dict[str, float]:
  from nmoe.csrc import rdep as _C
  from nmoe.moe import expert

  X = x.view(-1, x.size(-1)).contiguous().bfloat16()
  if X.numel() == 0:
    return {}
  g, eid = module.router(X)
  gates_fp32 = g.detach().float()
  T, H = X.shape
  K = int(eid.shape[1])
  n_local = int(module.n_local)
  device = X.device
  stream = torch.cuda.current_stream(device)

  offs_pad = torch.empty(n_local, device=device, dtype=torch.int32)
  M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()
  M_recv = _C.dispatch_meta_bf16(
    X.data_ptr(), eid.contiguous().int().data_ptr(), gates_fp32.data_ptr(),
    int(T), int(K), 128,
    offs_pad.data_ptr(), M_host.data_ptr(),
    stream,
  )
  if M_recv <= 0:
    return {}

  M_pad = (int(M_recv) + int(n_local) * (128 - 1) + (128 - 1)) // 128 * 128
  offs_pad[-1] = int(M_pad)
  Xe_pad = torch.empty(int(M_pad), int(H), device=device, dtype=torch.bfloat16)
  _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(M_pad), stream)

  counts = _collect_local_counts(eid, n_local, rank)
  valid_mask = _build_valid_mask(offs_pad, counts, M_pad)
  if not bool(valid_mask.any()):
    return {}

  layer_cache = cache.setdefault(id(module), {})
  if profile not in layer_cache:
    W3 = module.W3 if module.W3 is not None else module.W1
    layer_cache[profile] = {
      'W1_qdq': _quant_dequant_w13(module.W1.detach(), profile),
      'W3_qdq': _quant_dequant_w13(W3.detach(), profile),
      'W2_qdq': _quant_dequant_w2(module.W2.detach(), profile),
      'W1_ref_sq': _sum_sq(module.W1.detach()),
      'W3_ref_sq': _sum_sq(W3.detach()),
      'W2_ref_sq': _sum_sq(module.W2.detach()),
      'W1_err_sq': _sum_sq_diff(_quant_dequant_w13(module.W1.detach(), profile), module.W1.detach()),
      'W3_err_sq': _sum_sq_diff(_quant_dequant_w13(W3.detach(), profile), W3.detach()),
      'W2_err_sq': _sum_sq_diff(_quant_dequant_w2(module.W2.detach(), profile), module.W2.detach()),
    }
  wcache = layer_cache[profile]

  W3 = module.W3 if module.W3 is not None else module.W1
  Y_ref = expert(Xe_pad, module.W1, W3, module.W2, offs_pad, module._activation)[valid_mask]

  Xe_qdq = _quant_dequant_rows(Xe_pad, profile)
  Y_act = expert(Xe_qdq, module.W1, W3, module.W2, offs_pad, module._activation)[valid_mask]
  Y_w13 = expert(Xe_pad, wcache['W1_qdq'], wcache['W3_qdq'], module.W2, offs_pad, module._activation)[valid_mask]
  Y_w2 = expert(Xe_pad, module.W1, W3, wcache['W2_qdq'], offs_pad, module._activation)[valid_mask]
  Y_w = expert(Xe_pad, wcache['W1_qdq'], wcache['W3_qdq'], wcache['W2_qdq'], offs_pad, module._activation)[valid_mask]
  Y_both = expert(Xe_qdq, wcache['W1_qdq'], wcache['W3_qdq'], wcache['W2_qdq'], offs_pad, module._activation)[valid_mask]

  X_valid = Xe_pad[valid_mask]
  X_qdq_valid = Xe_qdq[valid_mask]

  return {
    'rows': float(int(valid_mask.sum().item())),
    'input_ref_sq': _sum_sq(X_valid),
    'input_err_sq': _sum_sq_diff(X_qdq_valid, X_valid),
    'output_ref_sq': _sum_sq(Y_ref),
    'act_only_err_sq': _sum_sq_diff(Y_act, Y_ref),
    'w13_only_err_sq': _sum_sq_diff(Y_w13, Y_ref),
    'w2_only_err_sq': _sum_sq_diff(Y_w2, Y_ref),
    'weight_only_err_sq': _sum_sq_diff(Y_w, Y_ref),
    'both_err_sq': _sum_sq_diff(Y_both, Y_ref),
    'W1_ref_sq': float(wcache['W1_ref_sq']),
    'W3_ref_sq': float(wcache['W3_ref_sq']),
    'W2_ref_sq': float(wcache['W2_ref_sq']),
    'W1_err_sq': float(wcache['W1_err_sq']),
    'W3_err_sq': float(wcache['W3_err_sq']),
    'W2_err_sq': float(wcache['W2_err_sq']),
  }


def main() -> None:
  import torch

  from nmoe import runtime
  runtime._maybe_add_repo_third_party_to_sys_path()
  from nmoe.data.loader import build_loader
  from nmoe.model import MoE, Transformer
  from quack.linear_cross_entropy import chunked_linear_cross_entropy

  ap = argparse.ArgumentParser(description='Split the 0005 forward floor into activation-side vs weight-side NVFP4 error.')
  ap.add_argument('--checkpoint-root', required=True)
  ap.add_argument('--profile', default='nvfp4', choices=['fp8', 'nvfp4'])
  ap.add_argument('--speedrun-data-root', default='/data/speedrun')
  ap.add_argument('--out-json', required=True)
  ap.add_argument('--analyze-steps', type=int, default=4)
  args = ap.parse_args()

  cfg = _load_cfg(Path(args.speedrun_data_root))
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
    steps=int(args.analyze_steps),
  )
  v_loader, _ = build_loader(loader_cfg, rank, world, split='valid', print_fn=quiet)
  model = Transformer(cfg).cuda()
  model.eval()
  latest = _load_split_checkpoint(model, Path(args.checkpoint_root), rank)

  ignore_index = int(cfg.eos_token_id) if getattr(cfg, 'loss_mask_eos', True) else -100
  loss_sum = torch.zeros((), device='cuda', dtype=torch.float32)
  tok_count = torch.zeros((), device='cuda', dtype=torch.float32)

  aggregates: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
  weight_seen: set[int] = set()
  weight_cache: dict[int, dict[str, torch.Tensor]] = {}

  def make_hook(layer_id: int):
    def _hook(module, inputs):
      stats = _analyze_moe_batch(module, inputs[0], args.profile, weight_cache, rank)
      if not stats:
        return None
      agg = aggregates[layer_id]
      for key, value in stats.items():
        if key.startswith('W') and layer_id in weight_seen:
          continue
        agg[key] += float(value)
      if any(key.startswith('W') for key in stats):
        weight_seen.add(layer_id)
      return None
    return _hook

  hooks = []
  for name, module in model.named_modules():
    if isinstance(module, MoE):
      hooks.append(module.register_forward_pre_hook(make_hook(_layer_name_to_id(name))))

  with torch.no_grad():
    for _ in range(int(args.analyze_steps)):
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
  for hook in hooks:
    hook.remove()

  if world > 1 and dist.is_initialized():
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)
    if aggregates:
      keys = sorted({k for layer in aggregates.values() for k in layer.keys()})
      for layer_id in sorted(aggregates.keys()):
        tensor = torch.tensor([aggregates[layer_id].get(k, 0.0) for k in keys], device='cuda', dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        for idx, key in enumerate(keys):
          aggregates[layer_id][key] = float(tensor[idx].item())

  def rel(err_sq: float, ref_sq: float) -> float | None:
    if ref_sq <= 0.0:
      return None
    return math.sqrt(err_sq / ref_sq)

  per_layer: list[dict[str, Any]] = []
  total_rows = 0.0
  total_input_ref_sq = 0.0
  total_input_err_sq = 0.0
  total_output_ref_sq = 0.0
  total_act_err_sq = 0.0
  total_w13_err_sq = 0.0
  total_w2_only_err_sq = 0.0
  total_weight_err_sq = 0.0
  total_both_err_sq = 0.0
  total_W1_ref_sq = 0.0
  total_W1_err_sq = 0.0
  total_W3_ref_sq = 0.0
  total_W3_err_sq = 0.0
  total_W2_ref_sq = 0.0
  total_W2_err_sq = 0.0

  for layer_id in sorted(aggregates.keys()):
    agg = aggregates[layer_id]
    rows = agg.get('rows', 0.0)
    total_rows += rows
    total_input_ref_sq += agg.get('input_ref_sq', 0.0)
    total_input_err_sq += agg.get('input_err_sq', 0.0)
    total_output_ref_sq += agg.get('output_ref_sq', 0.0)
    total_act_err_sq += agg.get('act_only_err_sq', 0.0)
    total_w13_err_sq += agg.get('w13_only_err_sq', 0.0)
    total_w2_only_err_sq += agg.get('w2_only_err_sq', 0.0)
    total_weight_err_sq += agg.get('weight_only_err_sq', 0.0)
    total_both_err_sq += agg.get('both_err_sq', 0.0)
    total_W1_ref_sq += agg.get('W1_ref_sq', 0.0)
    total_W1_err_sq += agg.get('W1_err_sq', 0.0)
    total_W3_ref_sq += agg.get('W3_ref_sq', 0.0)
    total_W3_err_sq += agg.get('W3_err_sq', 0.0)
    total_W2_ref_sq += agg.get('W2_ref_sq', 0.0)
    total_W2_err_sq += agg.get('W2_err_sq', 0.0)
    per_layer.append({
      'layer_id': layer_id,
      'rows': rows,
      'input_rel_rmse': rel(agg.get('input_err_sq', 0.0), agg.get('input_ref_sq', 0.0)),
      'act_only_output_rel_rmse': rel(agg.get('act_only_err_sq', 0.0), agg.get('output_ref_sq', 0.0)),
      'w13_only_output_rel_rmse': rel(agg.get('w13_only_err_sq', 0.0), agg.get('output_ref_sq', 0.0)),
      'w2_only_output_rel_rmse': rel(agg.get('w2_only_err_sq', 0.0), agg.get('output_ref_sq', 0.0)),
      'weight_only_output_rel_rmse': rel(agg.get('weight_only_err_sq', 0.0), agg.get('output_ref_sq', 0.0)),
      'both_output_rel_rmse': rel(agg.get('both_err_sq', 0.0), agg.get('output_ref_sq', 0.0)),
      'W1_rel_rmse': rel(agg.get('W1_err_sq', 0.0), agg.get('W1_ref_sq', 0.0)),
      'W3_rel_rmse': rel(agg.get('W3_err_sq', 0.0), agg.get('W3_ref_sq', 0.0)),
      'W2_rel_rmse': rel(agg.get('W2_err_sq', 0.0), agg.get('W2_ref_sq', 0.0)),
    })

  valid_loss = float((loss_sum / tok_count.clamp(min=1.0)).item())
  out = {
    'profile': args.profile,
    'checkpoint': str(latest),
    'analyze_steps': int(args.analyze_steps),
    'bf16_valid_loss': valid_loss,
    'summary': {
      'rows': total_rows,
      'input_rel_rmse': rel(total_input_err_sq, total_input_ref_sq),
      'act_only_output_rel_rmse': rel(total_act_err_sq, total_output_ref_sq),
      'w13_only_output_rel_rmse': rel(total_w13_err_sq, total_output_ref_sq),
      'w2_only_output_rel_rmse': rel(total_w2_only_err_sq, total_output_ref_sq),
      'weight_only_output_rel_rmse': rel(total_weight_err_sq, total_output_ref_sq),
      'both_output_rel_rmse': rel(total_both_err_sq, total_output_ref_sq),
      'W1_rel_rmse': rel(total_W1_err_sq, total_W1_ref_sq),
      'W3_rel_rmse': rel(total_W3_err_sq, total_W3_ref_sq),
      'W2_rel_rmse': rel(total_W2_err_sq, total_W2_ref_sq),
    },
    'per_layer': per_layer,
  }
  if rank == 0:
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
  main()
