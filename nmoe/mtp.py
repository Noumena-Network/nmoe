"""Multi-Token Prediction (MTP) module.

Extends next-token prediction to predict D additional future tokens sequentially.
Improves sample efficiency during training; can enable speculative decoding at inference.

Architecture (DeepSeek-V3):
  For each prediction depth k (1 to D):
    h'_i^k = M_k [RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))]
    h_{1:T-k}^k = TRM_k(h'_{1:T-k}^k)
    P_{i+k+1}^k = OutHead(h_i^k)

  - Shared: Embedding layer, output head (lm_head) with main model
  - Per-depth: Projection M_k, Transformer block TRM_k (independent weights)

Reference: https://arxiv.org/html/2412.19437v2
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.profiler import record_function

from nmoe.norm import RMSNorm


class MTPBlock(nn.Module):
  """Single MTP prediction block for depth k.

  Applies: proj(concat(RMSNorm(h_prev), RMSNorm(emb))) -> attn -> ffn -> h_k
  """

  def __init__(self, dim: int, inter_dim: int, n_layers: int, depth: int, attn_cls, rms_eps: float):
    super().__init__()
    self.depth = depth
    self.norm_h = RMSNorm(dim, rms_eps)
    self.norm_emb = RMSNorm(dim, rms_eps)
    self.proj = nn.Linear(dim * 2, dim, bias=False, dtype=torch.bfloat16)
    self.attn_norm = RMSNorm(dim, rms_eps)
    self.ffn_norm = RMSNorm(dim, rms_eps)
    self.attn = attn_cls
    self.ffn = _MLP(dim, inter_dim)
    self.init_std = 0.02 / (2 * (n_layers + depth + 1)) ** 0.5

  def init_weights(self):
    nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.02)
    self.norm_h.weight.data.fill_(1.0)
    self.norm_emb.weight.data.fill_(1.0)
    self.attn_norm.weight.data.fill_(1.0)
    self.ffn_norm.weight.data.fill_(1.0)
    self.attn.init_weights(self.init_std)
    self.ffn.init_weights(self.init_std)

  def forward(
    self,
    h_prev: torch.Tensor,
    token_emb: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
  ) -> torch.Tensor:
    h_normed = self.norm_h(h_prev)
    emb_normed = self.norm_emb(token_emb)
    x = self.proj(torch.cat([h_normed, emb_normed], dim=-1))
    x = x + torch.utils.checkpoint.checkpoint(
      self.attn, self.attn_norm(x), cos, sin, use_reentrant=False
    )
    x = x + self.ffn(self.ffn_norm(x))
    return x


class _MLP(nn.Module):
  """Simple MLP for MTP blocks (no MoE)."""

  def __init__(self, dim: int, inter_dim: int):
    super().__init__()
    self.w1 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.bfloat16)
    self.w3 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.bfloat16)
    self.w2 = nn.Linear(inter_dim, dim, bias=False, dtype=torch.bfloat16)

  def init_weights(self, init_std: float = 0.02):
    nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
    nn.init.trunc_normal_(self.w3.weight, mean=0.0, std=0.02)
    nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MTP(nn.Module):
  """Multi-Token Prediction module.

  Predicts D additional future tokens using dedicated transformer blocks.
  Paper indexing (1-indexed k): depth k consumes t_{i+k}, predicts t_{i+k+1}.

  Args:
    config: Model configuration with mtp_depth > 0
    embedding: Shared embedding from main model
    lm_head: Shared output head from main model
    mup_scale: Embedding scale factor
    logits_scale: Output logits scale factor
    attn_cls_fn: Callable that returns attention instance
  """

  def __init__(
    self,
    config,
    embedding: nn.Embedding,
    lm_head: nn.Linear,
    mup_scale: float,
    logits_scale: float,
    attn_cls_fn,
  ):
    super().__init__()
    self.depth = config.mtp_depth
    self.dim = config.dim
    self.ignore_index = config.eos_token_id  # Same masking as main loss
    self.embedding = embedding
    self.lm_head = lm_head
    self.mup_scale = mup_scale
    self.logits_scale = logits_scale
    self.blocks = nn.ModuleList([
      MTPBlock(
        dim=config.dim,
        inter_dim=config.inter_dim,
        n_layers=config.n_layers,
        depth=k,
        attn_cls=attn_cls_fn(),
        rms_eps=config.rms_norm_eps,
      )
      for k in range(self.depth)
    ])
    self.output_norms = nn.ModuleList([
      RMSNorm(config.dim, config.rms_norm_eps) for _ in range(self.depth)
    ])
    self.last_loss: torch.Tensor | None = None

  def init_weights(self):
    for block in self.blocks:
      block.init_weights()
    for norm in self.output_norms:
      norm.weight.data.fill_(1.0)

  @record_function("mtp")
  def forward(
    self,
    h_main: torch.Tensor,
    targets: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
  ) -> None:
    """Compute MTP loss and store in self.last_loss.

    Args:
      h_main: Hidden states from main model (before final norm) [B, T, D]
      targets: Target token IDs [B, T]
      cos, sin: RoPE embeddings from main model [T, head_dim]
    """
    B, T, D = h_main.shape

    if T <= self.depth:
      self.last_loss = None
      return

    total_loss = h_main.new_zeros((), dtype=torch.float32)
    h_prev = h_main

    for k in range(self.depth):
      # Paper indexing (1-indexed k -> 0-indexed): depth k+1 uses t_{i+k+1} to predict t_{i+k+2}
      # 0-indexed: depth k uses targets[:, k : T-1] to predict targets[:, k+1 : T]
      # Sequence length at depth k: T - k - 1
      seq_len_k = T - k - 1
      if seq_len_k <= 0:
        break

      input_tokens = targets[:, k : k + seq_len_k]
      target_tokens = targets[:, k + 1 : k + 1 + seq_len_k]
      h_prev_k = h_prev[:, :seq_len_k]
      cos_k = cos[:seq_len_k]
      sin_k = sin[:seq_len_k]

      with record_function(f"mtp/block_{k}"):
        token_emb = self.embedding(input_tokens) * self.mup_scale
        h_k = self.blocks[k](h_prev_k, token_emb, cos_k, sin_k)
        h_normed = self.output_norms[k](h_k)
        logits_k = self.lm_head(h_normed) * self.logits_scale
        # Paper eq 24: L_k = (1/T) Σ losses, NOT (1/seq_len_k)
        # Deeper depths have fewer positions but same denominator → less weight
        # For batch: average over B sequences, each divided by T
        loss_k = F.cross_entropy(
          logits_k.reshape(-1, logits_k.size(-1)),
          target_tokens.reshape(-1),
          ignore_index=self.ignore_index,
          reduction="sum",
        ) / (B * T)
        total_loss = total_loss + loss_k
        h_prev = h_k

    self.last_loss = total_loss / self.depth
