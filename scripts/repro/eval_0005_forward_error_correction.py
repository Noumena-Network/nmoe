#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import tomllib
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


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
    if step is None or not (path / 'rd.pt').exists():
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


def _encode_e8m0_from_scale(scale: torch.Tensor) -> torch.Tensor:
  scale = torch.where(scale > 0, scale, torch.ones_like(scale))
  exp = torch.ceil(torch.log2(scale.float())).to(torch.int32) + 127
  exp = torch.clamp(exp, 0, 254)
  return exp.to(torch.uint8)


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


def _block_view(x: torch.Tensor) -> torch.Tensor:
  if x.ndim != 2 or (x.shape[1] % 32) != 0:
    raise ValueError(f'expected [M, K] with K%32==0, got {tuple(x.shape)}')
  return x.float().view(x.shape[0], x.shape[1] // 32, 32)


def _default_scale_bytes(x: torch.Tensor, profile: str, alpha: float) -> torch.Tensor:
  blocks = _block_view(x)
  amax = blocks.abs().amax(dim=-1)
  dtype_max = 448.0 if profile == 'fp8' else 6.0
  scale = torch.where(amax > 0, (amax / dtype_max) * float(alpha), torch.ones_like(amax))
  return _encode_e8m0_from_scale(scale)


def _quant_dequant_rows_with_scale_bytes(x: torch.Tensor, profile: str, sfa: torch.Tensor) -> torch.Tensor:
  from nmoe.csrc import rdep as _C

  M, K = x.shape
  sfa = sfa.contiguous()
  stream = torch.cuda.current_stream(x.device)
  if profile == 'fp8':
    out_u16 = torch.empty(M, K // 2, dtype=torch.uint16, device=x.device)
    _C.quant_fp8_with_sfa(
      x.data_ptr(), K,
      out_u16.data_ptr(), K // 2,
      sfa.data_ptr(), K // 32,
      M, K, stream,
    )
    q = out_u16.view(torch.uint8).view(M, K, 1).view(torch.float8_e4m3fn)
    return _dequant_fp8(q, sfa.unsqueeze(-1))
  if profile == 'nvfp4':
    out_u16 = torch.empty(M, K // 4, dtype=torch.uint16, device=x.device)
    _C.quant_nvfp4_with_sfa(
      x.data_ptr(), K,
      out_u16.data_ptr(), K // 4,
      sfa.data_ptr(), K // 32,
      M, K, stream,
    )
    q = out_u16.view(torch.uint8).view(M, K // 2, 1)
    return _dequant_nvfp4(q, sfa.unsqueeze(-1))
  raise ValueError(f'unsupported profile: {profile}')


def _quant_dequant_rows_with_scale_alpha(x: torch.Tensor, profile: str, alpha: float) -> torch.Tensor:
  return _quant_dequant_rows_with_scale_bytes(x, profile, _default_scale_bytes(x, profile, alpha))


def _quant_dequant_rows_default(x: torch.Tensor, profile: str) -> torch.Tensor:
  from nmoe.quant import quantize_fp8, quantize_nvfp4

  if profile == 'fp8':
    q, sfa = quantize_fp8(x)
    return _dequant_fp8(q, sfa)
  if profile == 'nvfp4':
    q, sfa = quantize_nvfp4(x)
    return _dequant_nvfp4(q, sfa)
  raise ValueError(f'unsupported profile: {profile}')


def _shift_scale_bytes(scale_bytes: torch.Tensor, delta: int) -> torch.Tensor:
  if delta == 0:
    return scale_bytes
  shifted = scale_bytes.to(dtype=torch.int16) + int(delta)
  shifted = torch.clamp(shifted, 1, 254)
  return shifted.to(dtype=torch.uint8)


def _apply_block_mean_bias(x: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
  x_b = _block_view(x)
  dq_b = _block_view(dq)
  corr = (x_b - dq_b).mean(dim=-1, keepdim=True)
  return (dq_b + corr).reshape_as(dq).to(dtype=torch.bfloat16)


def _apply_block_affine(x: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
  x_b = _block_view(x)
  dq_b = _block_view(dq)
  mean_x = x_b.mean(dim=-1, keepdim=True)
  mean_q = dq_b.mean(dim=-1, keepdim=True)
  x_c = x_b - mean_x
  q_c = dq_b - mean_q
  denom = (q_c * q_c).mean(dim=-1, keepdim=True).clamp_min(1e-12)
  gain = (x_c * q_c).mean(dim=-1, keepdim=True) / denom
  bias = mean_x - gain * mean_q
  return (gain * dq_b + bias).reshape_as(dq).to(dtype=torch.bfloat16)


def _quant_dequant_rows_search(x: torch.Tensor, profile: str, deltas: tuple[int, ...]) -> torch.Tensor:
  base_sfa = _default_scale_bytes(x, profile, 1.0)
  x_blocks = _block_view(x)
  cand_blocks = []
  cand_errs = []
  for delta in deltas:
    dq = _quant_dequant_rows_with_scale_bytes(x, profile, _shift_scale_bytes(base_sfa, delta))
    dq_blocks = _block_view(dq)
    cand_blocks.append(dq_blocks)
    cand_errs.append((dq_blocks - x_blocks).pow(2).mean(dim=-1))

  block_stack = torch.stack(cand_blocks, dim=0)        # [C, M, B, 32]
  err_stack = torch.stack(cand_errs, dim=0)            # [C, M, B]
  best_idx = err_stack.argmin(dim=0)                   # [M, B]
  block_stack = block_stack.permute(1, 2, 0, 3)        # [M, B, C, 32]
  gather_idx = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 32)
  best = block_stack.gather(2, gather_idx).squeeze(2)
  return best.reshape_as(x).to(dtype=torch.bfloat16)


def _quant_dequant_rows_policy(x: torch.Tensor, profile: str, *, scale_alpha: float, correction: str) -> torch.Tensor:
  if correction == 'search1':
    return _quant_dequant_rows_search(x, profile, (-1, 0, 1))
  if correction == 'search2':
    return _quant_dequant_rows_search(x, profile, (-2, -1, 0, 1, 2))

  if abs(scale_alpha - 1.0) < 1e-6:
    dq = _quant_dequant_rows_default(x, profile)
  else:
    dq = _quant_dequant_rows_with_scale_alpha(x, profile, scale_alpha)

  if correction == 'none':
    return dq
  if correction == 'mean':
    return _apply_block_mean_bias(x, dq)
  if correction == 'affine':
    return _apply_block_affine(x, dq)
  raise ValueError(f'unsupported correction: {correction}')


def _quant_dequant_w13(W: torch.Tensor, profile: str, *, scale_alpha: float, correction: str) -> torch.Tensor:
  E, H, Dff = W.shape
  rows = W.transpose(1, 2).contiguous().view(E * Dff, H)
  dq = _quant_dequant_rows_policy(rows, profile, scale_alpha=scale_alpha, correction=correction)
  return dq.view(E, Dff, H).transpose(1, 2).contiguous()


def _quant_dequant_w2(W2: torch.Tensor, profile: str, *, scale_alpha: float, correction: str) -> torch.Tensor:
  E, Dff, H = W2.shape
  rows = W2.transpose(1, 2).contiguous().view(E * H, Dff)
  dq = _quant_dequant_rows_policy(rows, profile, scale_alpha=scale_alpha, correction=correction)
  return dq.view(E, H, Dff).transpose(1, 2).contiguous()


def _expert_selective(
  Xe_pad: torch.Tensor,
  W1: torch.Tensor,
  W3: torch.Tensor,
  W2: torch.Tensor,
  offs_pad: torch.Tensor,
  activation: str,
  profile: str,
  *,
  quant_postact: bool,
  scale_alpha: float,
  correction: str,
) -> torch.Tensor:
  H1 = torch._grouped_mm(Xe_pad, W1, offs=offs_pad)

  if activation == 'swiglu':
    H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)
    A = F.silu(H1) * H3
  elif activation == 'relu_squared':
    A = F.relu(H1) ** 2
  elif activation == 'squared_reglu':
    H3 = torch._grouped_mm(Xe_pad, W3, offs=offs_pad)
    A = F.relu(H1) ** 2 * H3
  else:
    raise ValueError(f'Unknown activation: {activation}')

  if quant_postact:
    A = _quant_dequant_rows_policy(A.contiguous(), profile, scale_alpha=scale_alpha, correction=correction)

  return torch._grouped_mm(A, W2, offs=offs_pad)


_POLICY_MAP = {
  'baseline': (1.0, 'none'),
  'scale75': (0.75, 'none'),
  'scale50': (0.50, 'none'),
  'mean': (1.0, 'mean'),
  'affine': (1.0, 'affine'),
  'scale75_affine': (0.75, 'affine'),
  'search1': (1.0, 'search1'),
  'search2': (1.0, 'search2'),
}


class _CorrectedMoE:
  def __init__(self, profile: str, variant: str, policy: str):
    self.profile = profile
    self.variant = variant
    self.policy = policy
    self.scale_alpha, self.correction = _POLICY_MAP[policy]
    self.weight_cache: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    self.originals: list[tuple[Any, Any]] = []

  def _weights_for(self, module) -> dict[str, torch.Tensor]:
    key = (id(module), self.policy)
    cached = self.weight_cache.get(key)
    if cached is not None:
      return cached
    W3 = module.W3 if module.W3 is not None else module.W1
    cached = {
      'W1_qdq': _quant_dequant_w13(module.W1.detach(), self.profile, scale_alpha=self.scale_alpha, correction=self.correction),
      'W3_qdq': _quant_dequant_w13(W3.detach(), self.profile, scale_alpha=self.scale_alpha, correction=self.correction),
      'W2_qdq': _quant_dequant_w2(module.W2.detach(), self.profile, scale_alpha=self.scale_alpha, correction=self.correction),
    }
    self.weight_cache[key] = cached
    return cached

  def _forward(self, module, x: torch.Tensor) -> torch.Tensor:
    from nmoe.csrc import rdep as _C

    X = x.view(-1, x.size(-1))
    T = X.size(0)
    g, eid = module.router(X)

    E = module.router.n_experts
    with torch.no_grad():
      loads = torch.bincount(eid.reshape(-1), minlength=E).to(torch.float32)
      module.last_loads = loads
    importance = torch.zeros(E, device=g.device, dtype=torch.float32)
    importance.scatter_add_(0, eid.reshape(-1), g.reshape(-1).float())
    module.last_importance = importance
    load_frac = loads / loads.sum().clamp(min=1.0)
    importance_frac = importance / importance.sum().clamp(min=1e-12)
    module.last_aux_loss = E * (importance_frac * load_frac).sum()

    W3 = module.W3 if module.W3 is not None else module.W1
    rdep = module._rdep
    device = X.device
    stream = torch.cuda.current_stream(device)
    gates_fp32 = g.detach().float()
    K = int(eid.shape[1])
    H = int(X.shape[1])
    is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    offs_pad = torch.empty(rdep.n_local, device=device, dtype=torch.int32)
    M_host = torch.zeros(1, device='cpu', dtype=torch.int32).pin_memory()
    align = 128
    M_recv = _C.dispatch_meta_bf16(
      X.contiguous().bfloat16().data_ptr(), eid.contiguous().int().data_ptr(), gates_fp32.data_ptr(),
      int(T), int(K), align,
      offs_pad.data_ptr(), M_host.data_ptr(),
      stream,
    )

    out_f32 = torch.zeros(int(T), int(H), device=device, dtype=torch.float32)
    if M_recv <= 0:
      if is_dist:
        dummy_ye_pad = torch.empty(1, int(H), device=device, dtype=torch.bfloat16)
        _C.return_scatter_from_pad_bf16(dummy_ye_pad.data_ptr(), out_f32.data_ptr(), 0, int(T), int(K), stream)
      out = out_f32.to(dtype=torch.bfloat16)
      if module._shared:
        out = out + module._shared(X)
      return out.view_as(x)

    max_pad = (int(M_recv) + int(rdep.n_local) * (align - 1) + (align - 1)) // align * align
    offs_pad[-1] = int(max_pad)
    Xe_pad = torch.empty(int(max_pad), int(H), device=device, dtype=torch.bfloat16)
    _C.gather_xe_bf16(Xe_pad.data_ptr(), int(M_recv), int(max_pad), stream)

    Xe_use = Xe_pad
    W1_use = module.W1
    W3_use = W3
    W2_use = module.W2

    quant_input = self.variant in ('act_only', 'both', 'stage1_only', 'full_forward')
    quant_w13 = self.variant in ('w13_only', 'weight_only', 'both', 'stage1_only', 'full_forward')
    quant_w2 = self.variant in ('w2_only', 'weight_only', 'both', 'full_forward')
    quant_postact = self.variant in ('postact_only', 'stage1_only', 'full_forward')

    if quant_input:
      Xe_use = _quant_dequant_rows_policy(Xe_pad, self.profile, scale_alpha=self.scale_alpha, correction=self.correction)
    if quant_w13 or quant_w2:
      qweights = self._weights_for(module)
      if quant_w13:
        W1_use = qweights['W1_qdq']
        W3_use = qweights['W3_qdq']
      if quant_w2:
        W2_use = qweights['W2_qdq']

    Ye_pad = _expert_selective(
      Xe_use, W1_use, W3_use, W2_use, offs_pad, module._activation, self.profile,
      quant_postact=quant_postact, scale_alpha=self.scale_alpha, correction=self.correction,
    )
    _C.return_scatter_from_pad_bf16(Ye_pad.data_ptr(), out_f32.data_ptr(), int(M_recv), int(T), int(K), stream)
    out = out_f32.to(dtype=torch.bfloat16)
    if module._shared:
      out = out + module._shared(X)
    return out.view_as(x)

  def patch(self, model) -> None:
    from nmoe.model import MoE

    def bind(module):
      def _patched(x: torch.Tensor, *, _module=module):
        return self._forward(_module, x)
      return _patched

    for module in model.modules():
      if isinstance(module, MoE):
        self.originals.append((module, module.forward))
        module.forward = bind(module)

  def restore(self) -> None:
    for module, forward in reversed(self.originals):
      module.forward = forward
    self.originals.clear()


def _parse_case(case: str) -> tuple[str, str]:
  if '@' in case:
    variant, policy = case.split('@', 1)
  else:
    variant, policy = case, 'baseline'
  variant = variant.strip()
  policy = policy.strip()
  if variant != 'bf16' and policy not in _POLICY_MAP:
    raise ValueError(f'unknown policy {policy!r}')
  return variant, policy


def _evaluate_case(model, loader_cfg, rank: int, world: int, quiet, variant: str, profile: str, policy: str) -> float:
  from nmoe.data.loader import build_loader
  from quack.linear_cross_entropy import chunked_linear_cross_entropy

  v_loader, _ = build_loader(loader_cfg, rank, world, split='valid', print_fn=quiet)
  ignore_index = int(loader_cfg.eos_token_id) if getattr(loader_cfg, 'loss_mask_eos', True) else -100
  loss_sum = torch.zeros((), device='cuda', dtype=torch.float32)
  tok_count = torch.zeros((), device='cuda', dtype=torch.float32)

  patcher = None
  if variant != 'bf16':
    patcher = _CorrectedMoE(profile, variant, policy)
    patcher.patch(model)

  try:
    with torch.no_grad():
      for _ in range(int(loader_cfg.validation_steps)):
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
  finally:
    v_loader.close()
    if patcher is not None:
      patcher.restore()

  if world > 1 and dist.is_initialized():
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)
  return float((loss_sum / tok_count.clamp(min=1.0)).item())


def main() -> None:
  from nmoe import runtime
  runtime._maybe_add_repo_third_party_to_sys_path()
  from nmoe.model import Transformer

  ap = argparse.ArgumentParser(description='Evaluate forward error-correction candidates for 0005 on the real MoE path.')
  ap.add_argument('--checkpoint-root', required=True)
  ap.add_argument('--profile', default='nvfp4', choices=['fp8', 'nvfp4'])
  ap.add_argument('--cases', default='bf16,full_forward@baseline,full_forward@search1,full_forward@search2,full_forward@affine,full_forward@scale75_affine,stage1_only@search1,w13_only@search1')
  ap.add_argument('--speedrun-data-root', default='/data/speedrun')
  ap.add_argument('--validation-steps', type=int, default=None)
  ap.add_argument('--out-json', required=True)
  args = ap.parse_args()

  cfg = _load_cfg(Path(args.speedrun_data_root))
  if args.validation_steps is not None:
    cfg = dataclasses.replace(cfg, validation_steps=int(args.validation_steps))
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
  model = Transformer(cfg).cuda()
  model.eval()
  latest = _load_split_checkpoint(model, Path(args.checkpoint_root), rank)

  case_specs = [_parse_case(item.strip()) for item in args.cases.split(',') if item.strip()]
  results: dict[str, float] = {}
  for variant, policy in case_specs:
    key = variant if variant == 'bf16' else f'{variant}@{policy}'
    results[key] = _evaluate_case(model, loader_cfg, rank, world, quiet, variant, args.profile, policy)

  bf16 = results['bf16']
  out = {
    'profile': args.profile,
    'checkpoint': str(latest),
    'validation_steps': int(loader_cfg.validation_steps),
    'results': results,
    'deltas_vs_bf16': {k: (v - bf16) for k, v in results.items()},
  }
  if rank == 0:
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
  main()
