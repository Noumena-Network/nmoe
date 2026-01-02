"""Optimizer construction, LR scheduling, and training step for MoE.

Two parameter types:
- expert: MoE expert MLPs (sharded via RDEP, optimized with Muon)
- dense: everything else - attention, embeddings, norms, router (replicated, ZeRO-2, AdamW)

Precision support: bf16, fp8, nvfp4 (default: nvfp4)
"""
import math
import torch
import torch.distributed as dist

from nmoe.config import Config
from nmoe import zero2
from nmoe.csrc import rdep as _rdep_ext


class ExpertAdamW(torch.optim.Optimizer):
  """Fused expert AdamW that emits blockscaled weight caches during the update.

  This removes the separate post-step refresh phase (less memory traffic, fewer kernels).
  Contract: experts are BF16 parameters; caches are FP8/NVFP4.
  """

  emits_weight_cache = True

  def __init__(self, moe_modules: list[torch.nn.Module], cfg: Config):
    params: list[torch.nn.Parameter] = []
    for moe in moe_modules:
      for name in ("W1", "W3", "W2"):
        p = getattr(moe, name, None)
        if p is None:
          raise ValueError(f"MoE module missing {name}")
        params.append(p)
    defaults = {
      "lr": cfg.lr_expert,
      "betas": (cfg.adam_beta1, cfg.adam_beta2_expert),  # Higher beta2 for FP8/NVFP4 gradient noise
      "eps": cfg.adam_eps,
      "weight_decay": cfg.weight_decay,
    }
    super().__init__(params, defaults)
    self._moes = moe_modules

  def _init_state(self, p: torch.Tensor) -> dict:
    state = self.state[p]
    if len(state) == 0:
      state["step"] = torch.tensor(0.0, dtype=torch.float32)
      state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
      state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
    return state

  def load_state_dict(self, state_dict: dict) -> None:  # type: ignore[override]
    super().load_state_dict(state_dict)
    # Step tensors may be loaded onto CUDA via map_location; keep them on CPU to
    # avoid per-step device sync when computing bias corrections.
    for st in self.state.values():
      step = st.get("step", None)
      if torch.is_tensor(step) and step.is_cuda:
        st["step"] = step.cpu()

  @torch.no_grad()
  def step(self, closure=None):  # type: ignore[override]
    if closure is not None:
      raise RuntimeError("ExpertAdamW does not support closure")
    if len(self.param_groups) != 1:
      raise RuntimeError("ExpertAdamW expects a single param group")

    group = self.param_groups[0]
    lr = float(group["lr"])
    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    weight_decay = float(group["weight_decay"])

    # Initialize state + bump per-param step (CPU tensors, fast).
    step_t: torch.Tensor | None = None
    for p in group["params"]:
      if p.grad is None:
        raise RuntimeError("ExpertAdamW requires all expert grads to be present")
      st = self._init_state(p)
      st["step"] += 1
      if step_t is None:
        step_t = st["step"]
    assert step_t is not None
    step = int(step_t.item())

    # Bias corrections match the standard AdamW update.
    bias_correction1 = 1.0 - (beta1**step)
    bias_correction2 = 1.0 - (beta2**step)
    step_size = lr / bias_correction1
    inv_bias_correction2_sqrt = 1.0 / math.sqrt(bias_correction2)

    for moe in self._moes:
      profile = getattr(moe, "_dtype", "nvfp4")
      if profile not in ("fp8", "nvfp4"):
        raise ValueError(f"ExpertAdamW only supports fp8/nvfp4 caches. Got {profile}")
      prof_i = 0 if profile == "fp8" else 1

      W1 = moe.W1
      W3 = moe.W3
      W2 = moe.W2
      if not (W1.is_cuda and W3.is_cuda and W2.is_cuda):
        raise RuntimeError("ExpertAdamW requires CUDA tensors")
      if not (W1.dtype == torch.bfloat16 and W3.dtype == torch.bfloat16 and W2.dtype == torch.bfloat16):
        raise RuntimeError("ExpertAdamW requires BF16 expert weights")
      if W1.grad is None or W3.grad is None or W2.grad is None:
        raise RuntimeError("Missing expert grads")
      if not (W1.grad.dtype == torch.bfloat16 and W3.grad.dtype == torch.bfloat16 and W2.grad.dtype == torch.bfloat16):
        raise RuntimeError("ExpertAdamW requires BF16 expert grads")
      if not (W1.grad.is_contiguous() and W3.grad.is_contiguous() and W2.grad.is_contiguous()):
        raise RuntimeError("Expert grads must be contiguous")
      if not (W1.is_contiguous() and W3.is_contiguous() and W2.is_contiguous()):
        raise RuntimeError("Expert weights must be contiguous")

      st1 = self._init_state(W1)
      st3 = self._init_state(W3)
      st2 = self._init_state(W2)

      # Ensure cache exists (initial use or after checkpoint restore).
      if getattr(moe, "_W_cache", None) is None:
        if not hasattr(moe, "refresh_weight_cache"):
          raise RuntimeError("MoE module missing refresh_weight_cache()")
        moe.refresh_weight_cache()
      cache = moe._W_cache
      if cache is None:
        raise RuntimeError("MoE weight cache missing after refresh")

      E, H, Dff = W1.shape
      stream = torch.cuda.current_stream(W1.device)
      _rdep_ext.expert_adamw_step(
        prof_i,
        W1.data_ptr(), W1.grad.data_ptr(), st1["exp_avg"].data_ptr(), st1["exp_avg_sq"].data_ptr(),
        W3.data_ptr(), W3.grad.data_ptr(), st3["exp_avg"].data_ptr(), st3["exp_avg_sq"].data_ptr(),
        W2.data_ptr(), W2.grad.data_ptr(), st2["exp_avg"].data_ptr(), st2["exp_avg_sq"].data_ptr(),
        cache.W13_q.data_ptr(), cache.W13_sf_mma.data_ptr(),
        cache.W2_q.data_ptr(), cache.W2_sf_mma.data_ptr(),
        int(E), int(H), int(Dff),
        float(lr), float(beta1), float(beta2),
        float(weight_decay), float(eps),
        float(step_size), float(inv_bias_correction2_sqrt),
        stream,
      )

    return None


def build_optimizer(model: torch.nn.Module, cfg: Config) -> tuple[torch.optim.Optimizer, list[dict]]:
  """Build optimizer for experts and return dense groups for AdamW/ZeRO-2.

  Returns:
    (expert_optimizer, dense_groups): Expert optimizer and dense param groups for ZeRO-2
  """
  # Collect parameters by type (single source of truth when available)
  expert_params: list[torch.nn.Parameter] = []
  router_params: list[torch.nn.Parameter] = []  # Router gate weights (separate LR, no decay)
  dense_params_decay: list[torch.nn.Parameter] = []
  dense_params_no_decay: list[torch.nn.Parameter] = []

  expert_ids: set[int] = set()
  if hasattr(model, "param_sets"):
    eps, _ = model.param_sets()  # type: ignore[attr-defined]
    expert_ids = {id(p) for p in eps}

  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue

    if id(param) in expert_ids:
      expert_params.append(param)
      continue

    # Router parameters: separate group with lr_router and weight_decay=0
    # Matches old_nmoe behavior for aux-free load balancing stability
    if 'router' in name:
      router_params.append(param)
      continue

    # Dense params: split decay vs no-decay by name pattern
    no_decay = name.endswith('.bias') or 'norm' in name or name.startswith('embedding.') or name.startswith('lm_head.') or 'bungee' in name
    if no_decay:
      dense_params_no_decay.append(param)
    else:
      dense_params_decay.append(param)

  # Build dense param groups (for ZeRO-2 AdamW)
  dense_groups = []
  if dense_params_decay:
    dense_groups.append({
      'name': 'dense_decay',
      'params': dense_params_decay,
      'lr': cfg.lr_dense,
      'weight_decay': cfg.weight_decay,
    })
  if dense_params_no_decay:
    dense_groups.append({
      'name': 'dense_no_decay',
      'params': dense_params_no_decay,
      'lr': cfg.lr_dense,
      'weight_decay': 0.0,
    })
  # Router group: separate LR, no weight decay (critical for aux-free load balancing)
  if router_params:
    dense_groups.append({
      'name': 'router',
      'params': router_params,
      'lr': cfg.lr_router,
      'weight_decay': 0.0,
    })

  # Expert optimizer: AdamW (benchmark proxy for Muon ceiling).
  use_blockscaled = getattr(cfg, "dtype", "nvfp4") in ("fp8", "nvfp4")
  if use_blockscaled and expert_params:
    moes: list[torch.nn.Module] = []
    for blk in getattr(model, "blocks", []):
      ffn = getattr(blk, "ffn", None)
      if ffn is not None and hasattr(ffn, "W1") and hasattr(ffn, "W3") and hasattr(ffn, "W2"):
        moes.append(ffn)
    expert_optimizer = ExpertAdamW(moes, cfg)
  else:
    expert_optimizer = torch.optim.AdamW(
      expert_params,
      lr=cfg.lr_expert,
      betas=(cfg.adam_beta1, cfg.adam_beta2_expert),  # Higher beta2 for expert gradient noise
      eps=cfg.adam_eps,
      weight_decay=cfg.weight_decay,
    )

  return expert_optimizer, dense_groups


def update_lr(optimizer: torch.optim.Optimizer, dense_groups: list[dict], step: int, tokens_seen: int, cfg: Config) -> float:
  """Update learning rate using a DeepSeek-style WSD schedule.

  WSD (Warmup-Sustain-Decay):
  1) Warmup: linear 0 → peak over warmup_steps (by step)
  2) Sustain: hold at peak until hold_tokens (by tokens)
  3) Cosine decay: decay over decay_tokens down to floor (absolute LR)
  4) Floor: hold at floor indefinitely

  Contract:
  - cfg.decay_floor is an *absolute* LR (not a ratio).
  - The same schedule shape is applied to expert, dense, and router peaks,
    with a shared absolute floor.
  """
  peak_expert = float(cfg.lr_expert)
  peak_dense = float(cfg.lr_dense)
  peak_router = float(cfg.lr_router)
  floor = float(cfg.decay_floor)

  if floor <= 0.0:
    raise RuntimeError(f"decay_floor must be > 0 (got {floor})")
  if floor > peak_expert or floor > peak_dense:
    raise RuntimeError(
      f"decay_floor ({floor}) must be <= lr_expert ({peak_expert}) and lr_dense ({peak_dense})"
    )

  # Compute schedule scale factor (0→1 during warmup, 1 during hold, 1→floor_ratio during decay)
  if step < cfg.warmup_steps:
    lr_scale = (step + 1) / max(1, int(cfg.warmup_steps))
  elif tokens_seen < cfg.hold_tokens:
    lr_scale = 1.0
  else:
    t = int(tokens_seen - cfg.hold_tokens)
    if t < cfg.decay_tokens:
      denom = float(max(1, int(cfg.decay_tokens)))
      lr_scale = 0.5 * (1.0 + math.cos(math.pi * float(t) / denom))
    else:
      lr_scale = 0.0

  # Apply schedule to each group's peak LR (floor applied at scale=0)
  floor_expert = floor
  floor_dense = floor
  floor_router = min(floor, peak_router)  # Router floor capped at its peak
  lr_expert = floor_expert + (peak_expert - floor_expert) * lr_scale
  lr_dense = floor_dense + (peak_dense - floor_dense) * lr_scale
  lr_router = floor_router + (peak_router - floor_router) * lr_scale

  for g in optimizer.param_groups:
    g['lr'] = lr_expert
  for g in dense_groups:
    # Router group uses lr_router, others use lr_dense
    if g.get('name') == 'router':
      g['lr'] = lr_router
    else:
      g['lr'] = lr_dense

  # Return the dense LR for logging (backward compatible with older single-LR configs).
  return float(lr_dense)


def step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, dense_groups: list[dict], zero2_state: dict, cfg: Config, world: int) -> None:
  """Optimizer step with ZeRO-2 and post-step hooks.

  Handles:
  - ZeRO-2 AdamW stepping for dense params (if world > 1)
  - Muon stepping for expert params
  - Post-step model updates (quantization rebuild, router bias)
  """
  # ZeRO-2 AdamW step for dense params
  if world > 1:
    zero2.step_dense_adamw(
      dense_groups,
      state=zero2_state,
      pg=dist.group.WORLD,
      betas=(cfg.adam_beta1, cfg.adam_beta2),
      eps=cfg.adam_eps,
    )
  else:
    zero2.step_dense_adamw(dense_groups, state=zero2_state)

  # Muon step for expert params (no ZeRO-2, params already sharded via RDEP)
  optimizer.step()

  # Post-step hooks (quantization rebuild, router bias update)
  with torch.no_grad():
    for blk in model.blocks:
      ffn = getattr(blk, 'ffn', None)
      if ffn is None:
        continue

      # Refresh quantized weight cache (FP8/NVFP4)
      if (not getattr(optimizer, "emits_weight_cache", False)) and hasattr(ffn, 'refresh_weight_cache'):
        try:
          ffn.refresh_weight_cache()
        except Exception:
          pass

      # Update router bias for aux-free load balancing
      if hasattr(ffn, 'router') and hasattr(ffn, 'last_loads'):
        try:
          ffn.router.update_bias(ffn.last_loads, gamma=cfg.router_bias_update_rate)
        except Exception:
          pass
