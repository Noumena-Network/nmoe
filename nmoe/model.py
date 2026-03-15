from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
from importlib import import_module

import torch
from torch import nn
import torch.nn.functional as F
from torch.profiler import record_function
import torch.distributed as dist

from nmoe.attention.rope import RotaryEmbedding
from nmoe.norm import RMSNorm

if TYPE_CHECKING:
  from nmoe.rdep import Rdep


ATTN = {
  "mla": "nmoe.attention.mla.MLA",
  "swa": "nmoe.attention.swa.SWA",
  "nsa": "nmoe.attention.nsa.NSA",
  "dsa": "nmoe.attention.dsa.DSA",
  "kda": "nmoe.attention.kda.KDA",
  "sdpa": "nmoe.attention.sdpa.SDPA",
}


def get_attention(name: str):
  if name not in ATTN:
    raise ValueError(f"Unknown attention '{name}'. Expected one of: {sorted(ATTN.keys())}")
  path = ATTN[name]
  module_path, cls_name = path.rsplit(".", 1)
  return getattr(import_module(module_path), cls_name)


class MLP(nn.Module):
  def __init__(self, dim: int, inter_dim: int, activation: str = "swiglu"):
    super().__init__()
    self.activation = activation
    self.w1 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.bfloat16)
    # relu_squared doesn't use w3 (2-weight architecture)
    if activation != "relu_squared":
      self.w3 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.bfloat16)
    else:
      self.w3 = None
    self.w2 = nn.Linear(inter_dim, dim, bias=False, dtype=torch.bfloat16)

  def init_weights(self, init_std: float = 0.02):
    nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
    if self.w3 is not None:
      nn.init.trunc_normal_(self.w3.weight, mean=0.0, std=0.02)
    nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)

  @record_function("mlp")
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.activation == "swiglu":
      return self.w2(F.silu(self.w1(x)) * self.w3(x))
    elif self.activation == "relu_squared":
      return self.w2(F.relu(self.w1(x)) ** 2)
    elif self.activation == "squared_reglu":
      return self.w2(F.relu(self.w1(x)) ** 2 * self.w3(x))
    else:
      raise ValueError(f"Unknown activation: {self.activation}")


class Router(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.n_experts = config.n_routed_experts
    self.topk = config.n_activated_experts
    self.route_scale = getattr(config, 'route_scale', 1.0)
    # Post-normalization scaling for gradient flow (NVFP4 only, DeepSeek-V3 uses 2.5)
    self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)
    self.gate = nn.Linear(config.dim, self.n_experts, bias=False, dtype=torch.bfloat16)
    self.register_buffer("bias", torch.zeros(self.n_experts, dtype=torch.float32))

  @record_function("router")
  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = self.gate(x).float()
    if self.route_scale != 1.0:
      logits = logits * self.route_scale
    scores = torch.sigmoid(logits)
    scores_for_selection = scores + self.bias
    _, indices = torch.topk(scores_for_selection, k=self.topk, dim=-1)
    weights = torch.gather(scores, 1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    # Post-normalization scaling boosts gradient flow through experts (NVFP4)
    if self.routed_scaling_factor != 1.0:
      weights = weights * self.routed_scaling_factor
    return weights.to(x.dtype), indices

  @torch.no_grad()
  def update_bias(self, expert_loads: torch.Tensor, gamma: float = 0.001):
    expected = 1.0 / self.n_experts
    s = torch.sign(expert_loads - expected)
    self.bias -= gamma * (s - s.mean())
    self.bias.clamp_(-16.0, 16.0)

  def init_weights(self, init_std: float = 0.02):
    nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class MoE(nn.Module):
  def __init__(self, cfg, layer_id: int, *, rdep: Rdep):
    super().__init__()
    self.dim = cfg.dim
    self.moe_inter_dim = getattr(cfg, 'moe_inter_dim', cfg.inter_dim)
    self._rdep = rdep
    self.n_local = rdep.n_local
    self.K = rdep.topk
    self.router = Router(cfg)
    self._activation = getattr(cfg, 'activation', 'swiglu')
    self.W1 = nn.Parameter(torch.empty(self.n_local, self.dim, self.moe_inter_dim, dtype=torch.bfloat16))
    # relu_squared doesn't use W3 (2-weight architecture)
    if self._activation != "relu_squared":
      self.W3 = nn.Parameter(torch.empty(self.n_local, self.dim, self.moe_inter_dim, dtype=torch.bfloat16))
    else:
      self.register_parameter('W3', None)
    self.W2 = nn.Parameter(torch.empty(self.n_local, self.moe_inter_dim, self.dim, dtype=torch.bfloat16))
    self._dtype = getattr(cfg, 'dtype', 'nvfp4')
    self._use_blockscaled = self._dtype in ('fp8', 'nvfp4')
    self._blockscaled_forward_ablation = str(getattr(cfg, 'blockscaled_forward_ablation', 'off') or 'off').strip().lower()
    if self._blockscaled_forward_ablation not in ('off', 'w13_bf16', 'stage1_bf16', 'full_bf16', 'w13_fp8', 'w2_fp8', 'stage1_fp8', 'full_fp8'):
      raise ValueError(
        f"blockscaled_forward_ablation must be one of ('off', 'w13_bf16', 'stage1_bf16', 'full_bf16', 'w13_fp8', 'w2_fp8', 'stage1_fp8', 'full_fp8'), got {self._blockscaled_forward_ablation!r}"
      )
    self._W_cache = None  # QuantizedWeightsFused cache, refreshed after each optimizer step
    n_shared = getattr(cfg, 'n_shared_experts', 0)
    activation = getattr(cfg, 'activation', 'swiglu')
    self._shared = MLP(self.dim, n_shared * self.moe_inter_dim, activation=activation) if n_shared else None
    self.last_loads = None
    self.last_importance = None
    self.last_aux_loss = None
    self.aux_loss_alpha = float(getattr(cfg, 'aux_loss_alpha', 0.0))
    if self._use_blockscaled:
      self.register_load_state_dict_post_hook(self._invalidate_weight_cache_post_load)

  def init_weights(self, init_std: float = 0.02):
    nn.init.trunc_normal_(self.W1, mean=0.0, std=init_std)
    if self.W3 is not None:
      nn.init.trunc_normal_(self.W3, mean=0.0, std=init_std)
    nn.init.trunc_normal_(self.W2, mean=0.0, std=init_std)
    self.router.init_weights(init_std)
    if self._shared:
      self._shared.init_weights(init_std)
    if self._use_blockscaled:
      self.refresh_weight_cache()

  @torch.no_grad()
  def refresh_weight_cache(self):
    """Refresh quantized weight cache. Call after optimizer step."""
    if self._use_blockscaled:
      # Blockscaled kernels are SM100-only; keep import lazy so bf16/dense runs
      # don't depend on blockscaled stack.
      from nmoe.blockscaled.grouped import quantize_weights
      W3 = self.W3 if self.W3 is not None else self.W1  # dummy for relu_squared
      self._W_cache = quantize_weights(self.W1, W3, self.W2, profile=self._dtype)

  @torch.no_grad()
  def invalidate_weight_cache(self) -> bool:
    had_cache = self._W_cache is not None
    self._W_cache = None
    return had_cache

  @torch.no_grad()
  def _invalidate_weight_cache_post_load(self, module, incompatible_keys):
    del incompatible_keys
    module.invalidate_weight_cache()

  @record_function("moe")
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    X = x.view(-1, x.size(-1))
    T = X.size(0)
    g, eid = self.router(X)

    # Load counts (non-differentiable, for metrics/bias updates)
    E = self.router.n_experts
    with torch.no_grad():
      loads = torch.bincount(eid.reshape(-1), minlength=E).to(torch.float32)
      self.last_loads = loads

    # Switch-style aux loss: E * sum((importance/Σimportance) * (load/Σload))
    # Differentiable through g (router weights), provides gradient signal for balance.
    importance = torch.zeros(E, device=g.device, dtype=torch.float32)
    importance.scatter_add_(0, eid.reshape(-1), g.reshape(-1).float())
    self.last_importance = importance
    load_frac = loads / loads.sum().clamp(min=1.0)
    importance_frac = importance / importance.sum().clamp(min=1e-12)
    self.last_aux_loss = E * (importance_frac * load_frac).sum()

    W3 = self.W3 if self.W3 is not None else self.W1  # dummy for relu_squared
    if self._use_blockscaled:
      cache = self._W_cache
      if self._blockscaled_forward_ablation == 'off' and cache is None:
        # Blockscaled kernels are SM100-only; keep import lazy so bf16/dense runs
        # don't depend on blockscaled stack.
        from nmoe.blockscaled.grouped import quantize_weights
        cache = quantize_weights(self.W1, W3, self.W2, profile=self._dtype)
        self._W_cache = cache

      out = self._rdep.moe_blockscaled(
        X.bfloat16(), eid, g, self.W1, W3, self.W2, cache, self._activation, self._blockscaled_forward_ablation
      )
    else:
      out = self._rdep.moe_bf16(X.bfloat16(), eid, g, self.W1, W3, self.W2, self._activation)

    if self._shared:
      out = out + self._shared(X)
    return out.view_as(x)


class TransformerBlock(nn.Module):
  def __init__(self, config: Config, layer_id: int, *, rdep: Rdep | None = None, n_layers: int | None = None):
    super().__init__()
    self.layer_id = layer_id
    self.attn_norm = RMSNorm(config.dim, config.rms_norm_eps)
    self.ffn_norm = RMSNorm(config.dim, config.rms_norm_eps)

    global_every = int(getattr(config, "attn_global_every", 1))
    if global_every < 1:
      raise ValueError(f"attn_global_every must be >= 1, got {global_every}.")
    is_last = n_layers is not None and layer_id == n_layers - 1
    is_global = (global_every == 1) or (((layer_id + 1) % global_every) == 0) or is_last
    attn_name = config.attn if is_global else config.attn_local

    self.attn = get_attention(attn_name)(config)
    if not is_global:
      window = int(getattr(config, "attn_local_window", 0))
      if window <= 0:
        raise ValueError(f"attn_local_window must be > 0 when using local attention, got {window}.")
      if not hasattr(self.attn, "window"):
        raise ValueError(
          f"Local attention '{attn_name}' does not expose a 'window' attribute, "
          f"but attn_local_window={window} was requested."
        )
      self.attn.window = window
    self.is_moe = layer_id >= config.n_dense_layers
    activation = getattr(config, 'activation', 'swiglu')
    if layer_id < config.n_dense_layers:
      self.ffn = MLP(dim=config.dim, inter_dim=config.inter_dim, activation=activation)
    else:
      if rdep is None:
        raise ValueError("MoE layers require an Rdep instance")
      self.ffn = MoE(config, layer_id, rdep=rdep)
    # Depth-dependent initialization std.
    self.init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5

  def init_weights(self):
    self.attn_norm.weight.data.fill_(1.0)
    self.ffn_norm.weight.data.fill_(1.0)
    self.attn.init_weights(self.init_std)
    self.ffn.init_weights(self.init_std)

  @record_function("block")
  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x = x + torch.utils.checkpoint.checkpoint(self.attn, self.attn_norm(x), cos, sin, use_reentrant=False)
    x = x + self.ffn(self.ffn_norm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, config: Config):
    super().__init__()
    self.config = config
    world = dist.get_world_size() if dist.is_initialized() else 1
    has_moe = config.n_layers > config.n_dense_layers
    rdep: Rdep | None = None
    #TODO(EM): these if blocks and raises are ugly. rewrite or move
    if has_moe:
      # SM100-only; keep import lazy so dense/bf16 runs don't require RDEP.
      from nmoe.rdep import Rdep
      if config.n_routed_experts is None or config.n_activated_experts is None:
        raise ValueError("MoE requires n_routed_experts and n_activated_experts")
      if config.n_routed_experts % max(1, world) != 0:
        raise ValueError(f"n_routed_experts ({config.n_routed_experts}) must be divisible by world_size ({world})")
      n_local = config.n_routed_experts // max(1, world)
      capacity = int(config.batch_size * config.seq_len * config.n_activated_experts)
      rdep = Rdep(config.dim, n_local, config.n_activated_experts, profile=config.dtype, capacity=capacity)
    self.embedding = nn.Embedding(config.vocab_size, config.dim, dtype=torch.bfloat16)
    self.rope = RotaryEmbedding(
      head_dim=config.qk_rope_head_dim,
      base=int(config.rope_theta),
      dtype=torch.bfloat16,
      initial_context_length=config.max_position_embeddings,
      max_context_length=config.max_position_embeddings * 2,  # Allow some headroom
      scaling_factor=config.rope_scaling_factor,
      ntk_alpha=config.rope_ntk_alpha,
      ntk_beta=config.rope_ntk_beta,
    )
    self.blocks = nn.ModuleList([
      TransformerBlock(config, layer_id, rdep=rdep, n_layers=config.n_layers)
      for layer_id in range(config.n_layers)
    ])
    self.norm = RMSNorm(config.dim, config.rms_norm_eps)
    self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False, dtype=torch.bfloat16)
    # NVFP4-only I/O gains (set/validated in train.py before model construction).
    self.fp4_embed_gain = float(getattr(config, 'fp4_embed_gain', None) or 1.0)
    self.fp4_logits_gain = float(getattr(config, 'fp4_logits_gain', None) or 1.0)

  def init_weights(self):
    nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    for block in self.blocks:
      block.init_weights()
    self.norm.weight.data.fill_(1.0)
    final_std = self.config.dim ** -0.5
    nn.init.trunc_normal_(self.lm_head.weight, mean=0.0, std=final_std)

  def param_sets(self):
    expert_params: list[torch.nn.Parameter] = []
    for m in self.modules():
      if isinstance(m, MoE):
        expert_params.append(m.W1)
        if m.W3 is not None:
          expert_params.append(m.W3)
        expert_params.append(m.W2)
    expert_ids = {id(p) for p in expert_params}
    dense_params: list[torch.nn.Parameter] = []
    for p in self.parameters():
      if id(p) not in expert_ids:
        dense_params.append(p)
    return expert_params, dense_params

  @record_function("transformer")
  def forward(self, tokens: torch.Tensor, *, return_hidden: bool = False) -> torch.Tensor:
    with record_function("embedding"):
      x = self.embedding(tokens) * self.fp4_embed_gain
    seqlen = tokens.size(1)
    cos = self.rope.cos[:seqlen]
    sin = self.rope.sin[:seqlen]
    for block in self.blocks:
      x = block(x, cos, sin)
    with torch.no_grad():
      moe_layers = [blk.ffn for blk in self.blocks if isinstance(getattr(blk, 'ffn', None), MoE)]
      if moe_layers:
        loads = torch.stack([m.last_loads for m in moe_layers], dim=0)
        importance = torch.stack([m.last_importance for m in moe_layers], dim=0)
        if dist.is_available() and dist.is_initialized():
          dist.all_reduce(loads, op=dist.ReduceOp.SUM)
          dist.all_reduce(importance, op=dist.ReduceOp.SUM)
        loads = loads / loads.sum(dim=-1, keepdim=True).clamp_min(1.0)
        importance = importance / importance.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        for m, l, imp in zip(moe_layers, loads, importance):
          m.last_loads = l
          m.last_importance = imp

    with record_function("norm_f"):
      x = self.norm(x)
    if return_hidden:
      return x
    # Dynamic amax scaling handles range - no clamp needed (TorchTitan/Megatron pattern)
    with record_function("lm_head"):
      logits = self.lm_head(x) * self.fp4_logits_gain

    return logits
