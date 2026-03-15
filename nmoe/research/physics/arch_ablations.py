"""
Architecture ablations (physics harness).

This runner exists to compare an emerging "next architecture" across four axes:
  - width: MatFormer-style nested width (dynamic within a forward pass)
  - residual law: vanilla vs AltUp vs mHC
  - memory: none vs Engram vs PLE+Ngrammer
  - local mixing: global vs local sliding attention (SWA-style proxy)
  - preconditioner: Canon (depthwise causal conv residual), stackable with residual law

It is intentionally micro-scale and mechanism-isolating (PhysicsLM4 style).

Run:
  python -m nmoe.research.physics.arch_ablations --output /tmp/arch --steps 2000 --matrix stage1
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nmoe.research.physics.data.generators import (
  BOS,
  ANSWER_START,
  EOS,
  MANO_OP_BASE,
  NGRAM_MODE_A,
  NGRAM_MODE_B,
  NGRAM_SYM_MAX,
  NGRAM_SYM_MIN,
  SyntheticMix,
)


# ----------------------------- data -----------------------------------------

# Task-tag tokens (optional; used to make stack-gating learnable without inference).
#
# Reserved range: 9000.. (unused by existing synthetic tasks).
_TASK_TAGS: dict[str, int] = {
  "depo": 9000,
  "brevo": 9001,
  "mano": 9002,
  "ngram": 9003,
  "ngram_polysemy": 9004,
  "ngram_mixed": 9005,
  "ngram_scrambled": 9006,
}


def _tag_token_for_task(task: str) -> int:
  tok = _TASK_TAGS.get(str(task))
  if tok is None:
    raise KeyError(f"Unknown task {task!r} for --tag-task (known={sorted(_TASK_TAGS)}).")
  return int(tok)


@dataclass(frozen=True)
class TaskSpec:
  name: str
  weight: float
  kwargs: dict


def _parse_task_spec(spec: str) -> TaskSpec:
  parts = spec.split(":")
  name = parts[0]
  weight = float(parts[1]) if len(parts) > 1 and parts[1] else 1.0
  kwargs: dict = {}
  if len(parts) > 2 and parts[2]:
    for kv in parts[2].split(","):
      k, v = kv.split("=", 1)
      v = v.strip()
      if v.lower() in ("true", "false"):
        kwargs[k] = (v.lower() == "true")
        continue
      try:
        kwargs[k] = int(v)
        continue
      except ValueError:
        pass
      try:
        kwargs[k] = float(v)
        continue
      except ValueError:
        pass
      kwargs[k] = v
  return TaskSpec(name=name, weight=weight, kwargs=kwargs)


@dataclass(frozen=True)
class Split:
  tokens: torch.Tensor  # [N, seq_len+1]
  labels: torch.Tensor  # [N, seq_len+1] uint8 (1=supervise token under loss_mode=answer_only)
  tasks: list[str]


def _pad(tokens: list[int], *, target_len: int, pad_token: int) -> list[int]:
  if len(tokens) > target_len:
    raise ValueError(f"Refusing to truncate: len={len(tokens)} > target_len={target_len}.")
  return tokens + [pad_token] * (target_len - len(tokens))


def _build_split(*, tasks: tuple[str, ...], n: int, seq_len: int, seed: int, tag_task: bool) -> Split:
  mix = SyntheticMix(seed=seed)
  for spec in tasks:
    ts = _parse_task_spec(spec)
    mix.add(ts.name, weight=ts.weight, **ts.kwargs)
  samples = mix.generate(int(n))
  doc_len = int(seq_len) + 1

  toks = torch.full((len(samples), doc_len), EOS, dtype=torch.long)
  labels = torch.zeros((len(samples), doc_len), dtype=torch.uint8)
  task_names: list[str] = []

  for i, s in enumerate(samples):
    tt = list(s.tokens)
    ll = list(s.labels)
    if bool(tag_task):
      if not tt or int(tt[0]) != int(BOS):
        raise ValueError(f"Expected sample to start with BOS={int(BOS)} (task={s.task!r}).")
      tag_tok = _tag_token_for_task(s.task)
      tt = [tt[0], tag_tok] + tt[1:]
      ll = [0, 0] + ll[1:]
    tt = _pad(tt, target_len=doc_len, pad_token=EOS)
    ll = _pad(ll, target_len=doc_len, pad_token=0)
    toks[i] = torch.tensor(tt, dtype=torch.long)
    labels[i] = torch.tensor(ll, dtype=torch.uint8)
    task_names.append(s.task)

  return Split(tokens=toks, labels=labels, tasks=task_names)


def _answer_mask_from_input(x_in: torch.Tensor, *, eos_token_id: int) -> torch.Tensor:
  # Legacy (single-ANSWER_START datasets). Kept for debugging/analysis only.
  started = (x_in == int(ANSWER_START)).cumsum(dim=1) > 0
  return started & (x_in != int(eos_token_id))


def _mask_for_loss_mode(*, loss_mode: str, y: torch.Tensor, y_labels: torch.Tensor) -> torch.Tensor:
  if loss_mode == "full":
    # Pad tokens are EOS in this harness.
    return y != int(EOS)
  if loss_mode == "answer_only":
    # Trust generator-provided supervision mask (enables multi-QA samples).
    return y_labels.to(dtype=torch.bool)
  raise ValueError(f"Unknown loss_mode={loss_mode!r}")


def _loss_and_metrics(
  logits: torch.Tensor,  # [B,T,V]
  targets: torch.Tensor,  # [B,T]
  *,
  answer_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
  ignore_index = -100
  masked = targets.clone()
  masked[~answer_mask] = ignore_index
  loss = F.cross_entropy(logits.view(-1, logits.size(-1)), masked.reshape(-1), ignore_index=ignore_index)

  with torch.no_grad():
    pred = logits.argmax(dim=-1)
    correct = (pred == targets) & answer_mask
    tok_acc = float(correct.sum().item() / max(1, answer_mask.sum().item()))
    per_ex_ok = ((pred == targets) | ~answer_mask).all(dim=-1)
    per_ex_has = answer_mask.any(dim=-1)
    em = float((per_ex_ok & per_ex_has).sum().item() / max(1, per_ex_has.sum().item()))
  return loss, {"answer_token_acc": tok_acc, "answer_exact_match": em}


def _logitlens_kl_to_final(
  *,
  final_logits: torch.Tensor,  # [B,T,V]
  layer_logits: torch.Tensor,  # [B,T,V]
  token_mask: torch.Tensor,  # [B,T] bool
) -> torch.Tensor:
  """
  LogitLens metric used by Engram: layer-wise KL divergence to the final output distribution.

  We compute KL(p_final || p_layer) averaged over masked tokens, where p_* are
  softmaxes over vocab at each position.
  """
  if final_logits.shape != layer_logits.shape:
    raise ValueError(f"shape mismatch: final={tuple(final_logits.shape)} layer={tuple(layer_logits.shape)}")
  if token_mask.shape != final_logits.shape[:2]:
    raise ValueError(f"mask shape mismatch: mask={tuple(token_mask.shape)} logits={tuple(final_logits.shape)}")

  log_p_final = F.log_softmax(final_logits, dim=-1, dtype=torch.float32)
  log_p_layer = F.log_softmax(layer_logits, dim=-1, dtype=torch.float32)
  p_final = log_p_final.exp()
  kl = (p_final * (log_p_final - log_p_layer)).sum(dim=-1)  # [B,T]

  mask = token_mask.to(dtype=torch.bool)
  denom = mask.sum().clamp_min(1)
  return kl[mask].sum() / denom


def _linear_cka(*, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """
  Linear CKA (centered) between two feature matrices.

  This is the representation similarity metric used by Engram for layer alignment
  heatmaps. We avoid Gram matrices (N×N) and compute the linear-form directly:

    CKA(X, Y) = ||XᵀY||_F² / (||XᵀX||_F · ||YᵀY||_F)

  where X and Y are centered across samples (rows).
  """
  if x.ndim != 2 or y.ndim != 2:
    raise ValueError(f"expected 2D features, got x={tuple(x.shape)} y={tuple(y.shape)}")
  if x.shape[0] != y.shape[0]:
    raise ValueError(f"sample mismatch: x={tuple(x.shape)} y={tuple(y.shape)}")
  if x.shape[1] != y.shape[1]:
    raise ValueError(f"dim mismatch: x={tuple(x.shape)} y={tuple(y.shape)}")

  x = x.to(dtype=torch.float32)
  y = y.to(dtype=torch.float32)
  x = x - x.mean(dim=0, keepdim=True)
  y = y - y.mean(dim=0, keepdim=True)

  xty = x.T @ y
  num = (xty * xty).sum()

  xtx = x.T @ x
  yty = y.T @ y
  denom = (xtx * xtx).sum().sqrt() * (yty * yty).sum().sqrt()
  if float(denom.item()) <= 0:
    return torch.tensor(0.0, dtype=torch.float32)
  return (num / denom).clamp(min=0.0, max=1.0)


def _collect_layer_features(
  *,
  model: AblationTransformer,
  split: Split,
  batch_size: int,
  loss_mode: str,
  device: torch.device,
) -> list[torch.Tensor]:
  """
  Collect per-layer hidden states as feature matrices for similarity analysis.

  Returns:
    features[l]: [N, dim] float32 (masked tokens, concatenated across the split)
  """
  model.eval()
  n_layers = len(model.blocks)
  chunks: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

  n = int(split.tokens.size(0))
  n_batches_total = math.ceil(n / int(batch_size))

  with torch.no_grad():
    for b in range(n_batches_total):
      start = b * int(batch_size)
      end = min(n, start + int(batch_size))
      batch = split.tokens[start:end].to(device, non_blocking=True)
      batch_labels = split.labels[start:end].to(device, non_blocking=True)
      x_in = batch[:, :-1]
      y = batch[:, 1:]
      y_labels = batch_labels[:, 1:]

      _, _, _ = model(x_in, collect_stats=False, collect_hiddens=True)
      hiddens = model._last_hiddens
      if hiddens is None or len(hiddens) != n_layers:
        raise RuntimeError("collect_hiddens=True did not produce per-layer hidden states")

      mask = _mask_for_loss_mode(loss_mode=loss_mode, y=y, y_labels=y_labels)

      if int(mask.sum().item()) == 0:
        continue

      for l, h in enumerate(hiddens):
        feats = h[mask].detach().to(dtype=torch.float32, device="cpu")
        chunks[l].append(feats)

  out: list[torch.Tensor] = []
  for l in range(n_layers):
    if not chunks[l]:
      raise RuntimeError("no features collected; check masks / data generation")
    out.append(torch.cat(chunks[l], dim=0))
  return out


def _cka_matrix(*, baseline: list[torch.Tensor], variant: list[torch.Tensor]) -> list[list[float]]:
  if len(baseline) != len(variant):
    raise ValueError(f"layer mismatch: baseline={len(baseline)} variant={len(variant)}")
  n_layers = len(baseline)
  mat: list[list[float]] = []
  for i in range(n_layers):
    row: list[float] = []
    for j in range(n_layers):
      row.append(float(_linear_cka(x=variant[i], y=baseline[j]).item()))
    mat.append(row)
  return mat


def _iter_minibatches(*, tokens: torch.Tensor, labels: torch.Tensor, batch_size: int, rng: np.random.Generator):
  n = int(tokens.size(0))
  while True:
    idx = rng.integers(0, n, size=int(batch_size), endpoint=False)
    yield tokens[idx], labels[idx]


# ----------------------------- model ----------------------------------------

class _RMSNormF32(nn.Module):
  def __init__(self, dim: int, eps: float):
    super().__init__()
    self.dim = int(dim)
    self.eps = float(eps)
    self.weight = nn.Parameter(torch.ones(self.dim, dtype=torch.float32))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x.float(), (self.dim,), self.weight, self.eps).to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
  def __init__(self, *, dim: int, n_heads: int, window: int | None, canon_qkv_kernel: int | None):
    super().__init__()
    if dim % n_heads != 0:
      raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
    self.dim = int(dim)
    self.n_heads = int(n_heads)
    self.head_dim = self.dim // self.n_heads
    self.window = int(window) if window is not None else None
    self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
    self.out = nn.Linear(self.dim, self.dim, bias=False)
    self.canonB = CanonConv(dim=3 * self.dim, kernel_size=int(canon_qkv_kernel)) if canon_qkv_kernel is not None else None

  def forward(
    self,
    x: torch.Tensor,
    *,
    canonB_alpha: torch.Tensor | None = None,
    collect_stats: bool = False,
  ) -> torch.Tensor:
    del collect_stats
    B, T, C = x.shape
    qkv = self.qkv(x)
    if self.canonB is not None:
      a = canonB_alpha
      if a is None:
        qkv = qkv + self.canonB(qkv)
      else:
        qkv = qkv + a * self.canonB(qkv)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,D]
    k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    # Use SDPA (Flash Attention) to avoid cuBLAS batched matmul bug on cu128
    if self.window is None:
      # Pure causal attention - use is_causal=True
      y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
      # Windowed attention - construct mask
      w = int(self.window)
      if w <= 0:
        raise ValueError(f"window must be > 0, got {w}")
      # Causal + window mask: attend only to positions in [i-w+1, i]
      idx = torch.arange(T, device=x.device)
      causal = idx[None, :] <= idx[:, None]  # [T, T] lower triangular
      in_window = idx[None, :] >= (idx[:, None] - (w - 1))  # within window
      mask = causal & in_window  # True = attend, False = mask out
      # SDPA expects attn_mask where True = KEEP (opposite of masked_fill convention)
      y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.out(y)

  def stats(self) -> dict[str, float]:
    return {}


def _inv_softplus(y: float) -> float:
  # For large y, inv_softplus(y) ~= y. Avoid overflow in exp(y).
  y = float(y)
  if y <= 0.0:
    return float("-inf")
  if y < 20.0:
    return float(math.log(math.expm1(y)))
  return y


class ConvergedSelfAttention(nn.Module):
  """
  Two-path attention with learned per-token local/global mixing.

  - Global path: standard causal attention.
  - Local path: causal attention with a *soft* learnable window (distance penalty).

  This is a physics-harness proxy for a “converged NSA”: one local primitive and one
  global primitive, with a learned router between them (biased to local at init).
  """

  def __init__(self, *, dim: int, n_heads: int, window_init: int, eps: float, canon_qkv_kernel: int | None):
    super().__init__()
    if dim % n_heads != 0:
      raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
    self.dim = int(dim)
    self.n_heads = int(n_heads)
    self.head_dim = self.dim // self.n_heads
    self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
    self.out = nn.Linear(self.dim, self.dim, bias=False)
    self.canonB = CanonConv(dim=3 * self.dim, kernel_size=int(canon_qkv_kernel)) if canon_qkv_kernel is not None else None

    # Per-head learnable soft window size (tokens). Used in a differentiable penalty.
    w0 = float(window_init)
    self.window_logit = nn.Parameter(torch.full((self.n_heads,), _inv_softplus(w0), dtype=torch.float32))
    self.window_tau = 2.0  # softness: higher = smoother penalty

    # Local-vs-global gate (per token).
    gate_hidden = max(8, int(self.dim) // 4)
    self.gate_norm = _RMSNormF32(self.dim, float(eps))
    self.gate_mlp = nn.Sequential(
      nn.Linear(self.dim, gate_hidden, bias=True),
      nn.GELU(),
      nn.Linear(gate_hidden, 1, bias=True),  # local fraction
    )
    nn.init.normal_(self.gate_mlp[0].weight, mean=0.0, std=0.02)
    nn.init.zeros_(self.gate_mlp[0].bias)
    nn.init.normal_(self.gate_mlp[-1].weight, mean=0.0, std=0.02)
    # Bias toward local at init (similar to NSA's SWA bias).
    self.gate_mlp[-1].bias.data.fill_(1.4)  # sigmoid -> ~0.8

    self._last_local_gate_mean: torch.Tensor | None = None
    self._last_window_frac_mean: torch.Tensor | None = None
    self._last_window_mean: torch.Tensor | None = None

  def forward(
    self,
    x: torch.Tensor,
    *,
    canonB_alpha: torch.Tensor | None = None,
    collect_stats: bool = False,
  ) -> torch.Tensor:
    B, T, C = x.shape
    qkv = self.qkv(x)
    if self.canonB is not None:
      a = canonB_alpha
      if a is None:
        qkv = qkv + self.canonB(qkv)
      else:
        qkv = qkv + a * self.canonB(qkv)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,D]
    k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    # Compute attention scores - loop over batch to avoid cu128 strided batched gemm bug
    # att[b,h] = q[b,h] @ k[b,h].T / sqrt(d)
    att_list = []
    for b in range(B):
      att_b = []
      for h in range(self.n_heads):
        att_b.append((q[b, h] @ k[b, h].T) / math.sqrt(self.head_dim))
      att_list.append(torch.stack(att_b, dim=0))
    att = torch.stack(att_list, dim=0)  # [B,H,T,T]

    # Causal mask.
    mask = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1)

    # Global path - use SDPA for the p @ v part
    att_g = att.masked_fill(mask, float("-inf"))
    p_g = att_g.softmax(dim=-1, dtype=torch.float32).to(dtype=q.dtype)
    # Loop for p @ v to avoid cu128 bug
    y_g_list = []
    for b in range(B):
      y_g_b = []
      for h in range(self.n_heads):
        y_g_b.append(p_g[b, h] @ v[b, h])
      y_g_list.append(torch.stack(y_g_b, dim=0))
    y_g = torch.stack(y_g_list, dim=0)  # [B,H,T,D]

    # Local path: apply a soft distance penalty controlled by a learnable window.
    idx = torch.arange(T, device=x.device)
    dist = (idx[:, None] - idx[None, :]).clamp(min=0).to(dtype=q.dtype)  # [T,T], 0 for future
    w = F.softplus(self.window_logit).to(dtype=q.dtype).clamp(min=1.0, max=float(T))  # [H]
    # sigmoid((w - dist)/tau) in (0,1); take log to add into attention logits.
    # clamp_min avoids -inf from log(0).
    penalty = torch.sigmoid((w.view(1, self.n_heads, 1, 1) - dist.view(1, 1, T, T)) / float(self.window_tau)).clamp_min(1e-12)
    att_l = (att + penalty.log()).masked_fill(mask, float("-inf"))
    p_l = att_l.softmax(dim=-1, dtype=torch.float32).to(dtype=q.dtype)
    # Loop for p @ v
    y_l_list = []
    for b in range(B):
      y_l_b = []
      for h in range(self.n_heads):
        y_l_b.append(p_l[b, h] @ v[b, h])
      y_l_list.append(torch.stack(y_l_b, dim=0))
    y_l = torch.stack(y_l_list, dim=0)  # [B,H,T,D]

    # Per-token gate: local fraction.
    g_local = torch.sigmoid(self.gate_mlp(self.gate_norm(x))).to(dtype=q.dtype)  # [B,T,1]
    if collect_stats:
      self._last_local_gate_mean = g_local.detach().float().mean()
      self._last_window_mean = w.detach().float().mean()
      self._last_window_frac_mean = (w.detach().float() / max(1.0, float(T))).mean()

    g = g_local.view(B, 1, T, 1)
    y = g * y_l + (1.0 - g) * y_g
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.out(y)

  def stats(self) -> dict[str, float]:
    if self._last_local_gate_mean is None:
      return {}
    out = {
      "local_gate_mean": float(self._last_local_gate_mean.item()),
    }
    if self._last_window_frac_mean is not None:
      out["window_frac_mean"] = float(self._last_window_frac_mean.item())
    if self._last_window_mean is not None:
      out["window_mean"] = float(self._last_window_mean.item())
    return out


class CanonConv(nn.Module):
  """Short depthwise causal conv in the residual stream (Canon A/C proxy)."""

  def __init__(self, *, dim: int, kernel_size: int):
    super().__init__()
    k = int(kernel_size)
    if k <= 1:
      raise ValueError(f"kernel_size must be > 1, got {k}")
    self.dim = int(dim)
    self.kernel_size = k
    self.conv = nn.Conv1d(
      in_channels=self.dim,
      out_channels=self.dim,
      kernel_size=self.kernel_size,
      groups=self.dim,
      bias=False,
      padding=self.kernel_size - 1,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B,T,C] -> Conv1d wants [B,C,T]
    y = self.conv(x.transpose(1, 2))
    y = y[..., : x.size(1)]  # causal: trim to original length
    return y.transpose(1, 2)


class MatFormerMLP(nn.Module):
  """Nested-width MLP with a per-token width gate (compute proxy)."""

  def __init__(self, *, dim: int, hidden_small: int, hidden_large: int):
    super().__init__()
    self.dim = int(dim)
    self.hidden_small = int(hidden_small)
    self.hidden_large = int(hidden_large)
    if self.hidden_small <= 0 or self.hidden_large <= 0 or self.hidden_small >= self.hidden_large:
      raise ValueError(f"Require 0 < hidden_small < hidden_large, got {hidden_small}, {hidden_large}")

    self.w1 = nn.Parameter(torch.empty((self.hidden_large, self.dim)))
    self.w2 = nn.Parameter(torch.empty((self.dim, self.hidden_large)))
    self.gate = nn.Linear(self.dim, 1, bias=True)
    self._last_p_mean: torch.Tensor | None = None

    nn.init.normal_(self.w1, mean=0.0, std=0.02)
    nn.init.normal_(self.w2, mean=0.0, std=0.02)
    nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
    nn.init.zeros_(self.gate.bias)

  def forward(self, x: torch.Tensor, *, collect_stats: bool) -> torch.Tensor:
    # Large path.
    hL = F.linear(x, self.w1)  # [B,T,Hl]
    oL = F.linear(F.silu(hL), self.w2)  # [B,T,C]

    # Small path (weight-tied slice).
    w1s = self.w1[: self.hidden_small, :]
    w2s = self.w2[:, : self.hidden_small]
    hS = F.linear(x, w1s)
    oS = F.linear(F.silu(hS), w2s)

    p = torch.sigmoid(self.gate(x))  # [B,T,1]
    if collect_stats:
      self._last_p_mean = p.detach().float().mean()
    return (1.0 - p) * oS + p * oL

  def stats(self) -> dict[str, float]:
    if self._last_p_mean is None:
      return {}
    return {"p_large_mean": float(self._last_p_mean.item())}


class StandardMLP(nn.Module):
  def __init__(self, *, dim: int, hidden: int, canon_hidden_kernel: int | None):
    super().__init__()
    self.hidden = int(hidden)
    self.fc1 = nn.Linear(dim, self.hidden, bias=False)
    self.fc2 = nn.Linear(int(hidden), dim, bias=False)
    self.canonD = CanonConv(dim=self.hidden, kernel_size=int(canon_hidden_kernel)) if canon_hidden_kernel is not None else None
    nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
    nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)

  def forward(self, x: torch.Tensor, *, canonD_alpha: torch.Tensor | None = None) -> torch.Tensor:
    h = self.fc1(x)
    if self.canonD is not None:
      a = canonD_alpha
      if a is None:
        h = h + self.canonD(h)
      else:
        h = h + a * self.canonD(h)
    return self.fc2(F.silu(h))


class SwiGLUMLP(nn.Module):
  """
  Gated MLP (SwiGLU) with optional Canon-D applied to concatenated (gate, up) streams.

  This matches PhysicsLM4's Canon-D semantics for Llama-family MLPs:
    x1 = gate_proj(x), x3 = up_proj(x)
    if canonD: [x1,x3] = canonD([x1,x3])
    out = down_proj(silu(x1) * x3)
  """

  def __init__(self, *, dim: int, hidden: int, canon_hidden_kernel: int | None):
    super().__init__()
    self.hidden = int(hidden)
    self.gate_proj = nn.Linear(dim, self.hidden, bias=False)
    self.up_proj = nn.Linear(dim, self.hidden, bias=False)
    self.down_proj = nn.Linear(int(hidden), dim, bias=False)
    self.canonD = (
      CanonConv(dim=2 * self.hidden, kernel_size=int(canon_hidden_kernel)) if canon_hidden_kernel is not None else None
    )
    nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
    nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
    nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)

  def forward(self, x: torch.Tensor, *, canonD_alpha: torch.Tensor | None = None) -> torch.Tensor:
    x1 = self.gate_proj(x)
    x3 = self.up_proj(x)
    if self.canonD is not None:
      cat = torch.cat([x1, x3], dim=-1)
      a = canonD_alpha
      if a is None:
        cat = cat + self.canonD(cat)
      else:
        cat = cat + a * self.canonD(cat)
      x1, x3 = cat.chunk(2, dim=-1)
    return self.down_proj(F.silu(x1) * x3)


def _masked_mean_pool(x: torch.Tensor, *, input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
  # x: [B,T,C], input_ids: [B,T]
  if x.dim() != 3 or input_ids.dim() != 2:
    raise ValueError(f"expected x [B,T,C] and input_ids [B,T], got x={tuple(x.shape)} ids={tuple(input_ids.shape)}")
  if x.shape[:2] != input_ids.shape:
    raise ValueError(f"shape mismatch: x={tuple(x.shape)} ids={tuple(input_ids.shape)}")
  mask = (input_ids != int(pad_token_id)).to(dtype=x.dtype)  # [B,T]
  denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
  pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B,C]
  return pooled


class StackGateMLP(nn.Module):
  """
  Prompt-conditioned stack gate.

  Outputs:
    [αA, αB, αC, αD] in [0,1] per sample.

  αA..αD: Canon placements

  Note: Engram and mHC are fixed-on in the stack-gate experiment:
    - Engram already has an internal content gate (q·k → sigmoid).
    - mHC consistently saturates to "on" in our earlier runs.
  """

  def __init__(self, *, dim: int, hidden: int, eps: float):
    super().__init__()
    self.dim = int(dim)
    self.hidden = int(hidden)
    self.eps = float(eps)
    self.rms = _RMSNormF32(self.dim, self.eps)
    self.fc1 = nn.Linear(self.dim, self.hidden, bias=True)
    self.fc2 = nn.Linear(self.hidden, 4, bias=True)

    # Match our NSA/MoE gate conventions: trunc_normal_ init + asymmetric final bias.
    nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=0.02)
    nn.init.zeros_(self.fc1.bias)
    nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=0.02)
    nn.init.zeros_(self.fc2.bias)

    # Asymmetric init (NSA-style): start on a reasonable default path and learn to deviate.
    #
    # Default: Canon-B/C/D on (~0.8), Canon-A off (~0.1).
    # sigmoid([−2.2, 1.4, 1.4, 1.4]) ≈ [0.10, 0.80, 0.80, 0.80].
    self.fc2.bias.data.copy_(torch.tensor([-2.2, 1.4, 1.4, 1.4], dtype=self.fc2.bias.dtype))

  def forward(self, pooled: torch.Tensor) -> torch.Tensor:
    if pooled.dim() != 2 or pooled.size(-1) != self.dim:
      raise ValueError(f"expected pooled [B,{self.dim}], got {tuple(pooled.shape)}")
    h = self.rms(pooled).to(dtype=pooled.dtype)
    h = F.gelu(self.fc1(h))
    return self.fc2(h).sigmoid()


def _bigram_ids(input_ids: torch.Tensor, *, table_size: int, multiplier: int) -> torch.Tensor:
  if input_ids.dim() != 2:
    raise ValueError(f"expected input_ids [B,T], got {tuple(input_ids.shape)}")
  B, T = input_ids.shape
  prev = torch.empty_like(input_ids)
  prev[:, 0] = 0
  prev[:, 1:] = input_ids[:, :-1]
  cur = input_ids

  is_prev = (prev >= NGRAM_SYM_MIN) & (prev <= NGRAM_SYM_MAX)
  is_cur = (cur >= NGRAM_SYM_MIN) & (cur <= NGRAM_SYM_MAX)
  ok = is_prev & is_cur

  a = (prev - NGRAM_SYM_MIN).clamp(min=0).to(torch.int64)
  b = (cur - NGRAM_SYM_MIN).clamp(min=0).to(torch.int64)
  h = (a * int(multiplier) + b) % int(table_size)

  out = torch.zeros((B, T), device=input_ids.device, dtype=torch.int64)
  out[ok] = h[ok]
  return out


class MemoryModule(nn.Module):
  def forward(  # pragma: no cover
    self,
    *,
    x: torch.Tensor,
    bigram_ids: torch.Tensor,
    layer_id: int,
    collect_stats: bool,
  ) -> torch.Tensor:
    raise NotImplementedError

  def stats(self) -> dict[str, float]:
    return {}


class EngramMemory(MemoryModule):
  def __init__(self, *, n_layers: int, dim: int, eps: float, table_size: int, mem_dim: int, multiplier: int):
    super().__init__()
    self.n_layers = int(n_layers)
    self.dim = int(dim)
    self.table_size = int(table_size)
    self.mem_dim = int(mem_dim)
    self.multiplier = int(multiplier)

    self.embed = nn.Embedding(self.table_size, self.mem_dim)
    self.k_proj = nn.Linear(self.mem_dim, self.dim, bias=False)
    self.v_proj = nn.Linear(self.mem_dim, self.dim, bias=False)
    self.q_proj = nn.ModuleList([nn.Linear(self.dim, self.dim, bias=False) for _ in range(self.n_layers)])
    self.rms = _RMSNormF32(self.dim, eps)
    self._last_gate_mean: dict[int, torch.Tensor] = {}

  def forward(self, *, x: torch.Tensor, bigram_ids: torch.Tensor, layer_id: int, collect_stats: bool) -> torch.Tensor:
    layer_id = int(layer_id)
    mem = self.embed(bigram_ids.clamp(min=0, max=self.table_size - 1))
    k = self.k_proj(mem)
    v = self.v_proj(mem)
    q = self.q_proj[layer_id](self.rms(x))

    gate_logits = (q * k).sum(dim=-1) / math.sqrt(self.dim)
    gate = gate_logits.sigmoid()
    gate = gate * (bigram_ids != 0).to(dtype=gate.dtype)
    if collect_stats:
      self._last_gate_mean[layer_id] = gate.detach().float().mean()
    return gate.unsqueeze(-1) * v

  def stats(self) -> dict[str, float]:
    return {f"mem.layer{layer}.gate_mean": float(v.item()) for layer, v in self._last_gate_mean.items()}


class ScalarGateMemory(MemoryModule):
  """
  Ablation: per-layer learned scalar gate (no content dependence).

  If this matches Engram, the q-k content gating isn't doing work.
  """
  def __init__(self, *, n_layers: int, dim: int, eps: float, table_size: int, mem_dim: int, multiplier: int):
    super().__init__()
    self.n_layers = int(n_layers)
    self.dim = int(dim)
    self.table_size = int(table_size)
    self.mem_dim = int(mem_dim)
    self.multiplier = int(multiplier)

    self.embed = nn.Embedding(self.table_size, self.mem_dim)
    self.v_proj = nn.Linear(self.mem_dim, self.dim, bias=False)
    # Learned scalar gate per layer (initialized to ~0.25 to match observed Engram values)
    self.gate_logit = nn.Parameter(torch.full((self.n_layers,), -1.1))  # sigmoid(-1.1) ≈ 0.25
    self._last_gate: dict[int, torch.Tensor] = {}

  def forward(self, *, x: torch.Tensor, bigram_ids: torch.Tensor, layer_id: int, collect_stats: bool) -> torch.Tensor:
    layer_id = int(layer_id)
    mem = self.embed(bigram_ids.clamp(min=0, max=self.table_size - 1))
    v = self.v_proj(mem)

    gate = self.gate_logit[layer_id].sigmoid()
    gate_mask = (bigram_ids != 0).to(dtype=v.dtype)
    if collect_stats:
      self._last_gate[layer_id] = gate.detach()
    return (gate * gate_mask).unsqueeze(-1) * v

  def stats(self) -> dict[str, float]:
    return {f"mem.layer{layer}.gate": float(v.item()) for layer, v in self._last_gate.items()}


class PlacementMemory(MemoryModule):
  """
  Ablation: PLE-style memory but only active in specified layers.

  If this matches Engram (with active_layers={0,1}), the gate is just learning placement.
  """
  def __init__(self, *, n_layers: int, dim: int, table_size: int, mem_dim: int, multiplier: int, active_layers: tuple[int, ...] = (0, 1)):
    super().__init__()
    self.n_layers = int(n_layers)
    self.dim = int(dim)
    self.table_size = int(table_size)
    self.mem_dim = int(mem_dim)
    self.multiplier = int(multiplier)
    self.active_layers = set(int(x) for x in active_layers)

    self.embed = nn.Embedding(self.table_size, self.mem_dim)
    self.proj = nn.ModuleList([nn.Linear(self.mem_dim, self.dim, bias=False) for _ in range(self.n_layers)])
    self._last_delta_rms: dict[int, torch.Tensor] = {}

  def forward(self, *, x: torch.Tensor, bigram_ids: torch.Tensor, layer_id: int, collect_stats: bool) -> torch.Tensor:
    layer_id = int(layer_id)
    if layer_id not in self.active_layers:
      return torch.zeros_like(x)

    mem = self.embed(bigram_ids.clamp(min=0, max=self.table_size - 1))
    delta = self.proj[layer_id](mem)
    delta = delta * (bigram_ids != 0).to(dtype=delta.dtype).unsqueeze(-1)
    if collect_stats:
      self._last_delta_rms[layer_id] = delta.detach().float().pow(2).mean().sqrt()
    return delta

  def stats(self) -> dict[str, float]:
    return {f"mem.layer{layer}.delta_rms": float(v.item()) for layer, v in self._last_delta_rms.items()}


class PleNgrammerMemory(MemoryModule):
  def __init__(self, *, n_layers: int, dim: int, table_size: int, mem_dim: int, multiplier: int):
    super().__init__()
    self.n_layers = int(n_layers)
    self.dim = int(dim)
    self.table_size = int(table_size)
    self.mem_dim = int(mem_dim)
    self.multiplier = int(multiplier)

    self.embed = nn.Embedding(self.table_size, self.mem_dim)
    self.proj = nn.ModuleList([nn.Linear(self.mem_dim, self.dim, bias=False) for _ in range(self.n_layers)])
    self._last_delta_rms: dict[int, torch.Tensor] = {}

  def forward(self, *, x: torch.Tensor, bigram_ids: torch.Tensor, layer_id: int, collect_stats: bool) -> torch.Tensor:
    layer_id = int(layer_id)
    mem = self.embed(bigram_ids.clamp(min=0, max=self.table_size - 1))
    delta = self.proj[layer_id](mem)
    delta = delta * (bigram_ids != 0).to(dtype=delta.dtype).unsqueeze(-1)
    if collect_stats:
      self._last_delta_rms[layer_id] = delta.detach().float().pow(2).mean().sqrt()
    return delta

  def stats(self) -> dict[str, float]:
    return {f"mem.layer{layer}.delta_rms": float(v.item()) for layer, v in self._last_delta_rms.items()}


class AltUp(nn.Module):
  """AltUp (Gemma-style) multi-stream predict/correct wrapper (PyTorch)."""

  def __init__(self, *, d_model: int, num_inputs: int, active_idx: int, coef_clip: float | None, eps: float):
    super().__init__()
    self.d_model = int(d_model)
    self.num_inputs = int(num_inputs)
    self.active_idx = int(active_idx)
    self.coef_clip = float(coef_clip) if coef_clip is not None else None

    # Single "modality" in physics harness (we just need data-dependent coefs).
    self.correction_coefs = nn.Parameter(torch.empty((self.num_inputs,)))
    self.prediction_coefs = nn.Parameter(torch.empty((self.num_inputs, self.num_inputs)))
    self.correct_output_scale = nn.Parameter(torch.ones((self.d_model,)))
    self.router_norm = _RMSNormF32(self.d_model, eps)
    self.modality_router = nn.Linear(self.d_model, 1, bias=False)

    nn.init.normal_(self.correction_coefs, mean=0.0, std=1e-2)
    nn.init.normal_(self.prediction_coefs, mean=0.0, std=1e-4)
    nn.init.normal_(self.modality_router.weight, mean=0.0, std=0.02)

    self._last_coef_rms: torch.Tensor | None = None

  def _coefs(self, x_active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Produce a single scalar "modality" per token and use it to scale the coef tensors.
    m = torch.tanh(self.modality_router(self.router_norm(x_active)) * (self.d_model ** -1.0))  # [B,T,1]
    pred = self.prediction_coefs.float()
    corr = self.correction_coefs.float()
    if self.coef_clip is not None:
      pred = pred.clamp(-self.coef_clip, self.coef_clip)
      corr = corr.clamp(-self.coef_clip, self.coef_clip)
    # Scale coefs by m (physics proxy for Gemma's modality mixture).
    pred = pred.view(1, 1, self.num_inputs, self.num_inputs)  # [1,1,N,N]
    pred = m.unsqueeze(-1) * pred  # [B,T,1,1] * [1,1,N,N] -> [B,T,N,N]
    corr = corr.view(1, 1, self.num_inputs) * m  # [1,1,N] * [B,T,1] -> [B,T,N]
    return pred, (corr + 1.0)

  def predict(self, xs: list[torch.Tensor], *, collect_stats: bool) -> list[torch.Tensor]:
    pred, _ = self._coefs(xs[self.active_idx])  # pred: [B,T,N,N]
    out: list[torch.Tensor] = []
    for i in range(self.num_inputs):
      mix = 0.0
      for j in range(self.num_inputs):
        mix = mix + pred[..., i, j].unsqueeze(-1) * xs[j]
      out.append(xs[i] + mix)
    if collect_stats:
      self._last_coef_rms = pred.detach().float().pow(2).mean().sqrt()
    return out

  def correct(self, predictions: list[torch.Tensor], activated: torch.Tensor) -> list[torch.Tensor]:
    _, corr = self._coefs(activated)  # corr: [B,T,N]
    activated = activated * self.correct_output_scale
    innovation = activated - predictions[self.active_idx]
    out: list[torch.Tensor] = []
    for i in range(self.num_inputs):
      out.append(predictions[i] + corr[..., i].unsqueeze(-1) * innovation)
    return out

  def stats(self) -> dict[str, float]:
    if self._last_coef_rms is None:
      return {}
    return {"coef_rms": float(self._last_coef_rms.item())}


def _sinkhorn_knopp(log_M: torch.Tensor, *, iters: int) -> torch.Tensor:
  log_M = log_M - log_M.amax(dim=(-2, -1), keepdim=True)
  M = torch.exp(log_M)
  for _ in range(int(iters)):
    M = M / (M.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    M = M / (M.sum(dim=-2, keepdim=True).clamp_min(1e-12))
  return M


class MHCMaps(nn.Module):
  """mHC coefficient maps (physics proxy; doubly-stochastic H_res via Sinkhorn)."""

  def __init__(self, *, n: int, dim: int, eps: float, sinkhorn_iters: int = 20, alpha_init: float = 0.01):
    super().__init__()
    self.n = int(n)
    self.dim = int(dim)
    self.eps = float(eps)
    self.sinkhorn_iters = int(sinkhorn_iters)

    nC = self.n * self.dim
    out_dim = self.n * self.n + 2 * self.n
    self.rms = _RMSNormF32(nC, self.eps)

    self.alpha_pre = nn.Parameter(torch.tensor(float(alpha_init)))
    self.alpha_post = nn.Parameter(torch.tensor(float(alpha_init)))
    self.alpha_res = nn.Parameter(torch.tensor(float(alpha_init)))

    self.phi = nn.Parameter(torch.empty((nC, out_dim)))
    self.b = nn.Parameter(torch.empty((out_dim,)))

    # Init to near-identity residual, uniform pre, ones post.
    b_pre = math.log((1.0 / self.n) / (1.0 - 1.0 / self.n))
    b_post = 0.0
    b_res = torch.full((self.n, self.n), math.log(1e-6), dtype=torch.float32)
    b_res.fill_(math.log(1e-6))
    b_res.diagonal().fill_(0.0)
    b_full = torch.cat([
      torch.full((self.n,), b_pre, dtype=torch.float32),
      torch.full((self.n,), b_post, dtype=torch.float32),
      b_res.reshape(-1),
    ])
    self.b.data.copy_(b_full)
    nn.init.normal_(self.phi, mean=0.0, std=0.02)

  def forward(self, x_stream: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x_stream: [B,T,n,C]
    B, T, n, C = x_stream.shape
    if n != self.n or C != self.dim:
      raise ValueError(f"Unexpected shape {tuple(x_stream.shape)} for n={self.n}, dim={self.dim}")
    x_vec = x_stream.reshape(B, T, n * C)
    x_prime = self.rms(x_vec).float()
    tt = x_prime @ self.phi  # [B,T,out_dim]

    off_pre = 0
    off_post = off_pre + n
    off_res = off_post + n

    pre_raw = (self.alpha_pre * tt[..., off_pre:off_post]) + self.b[off_pre:off_post]
    post_raw = (self.alpha_post * tt[..., off_post:off_res]) + self.b[off_post:off_res]
    res_raw = (self.alpha_res * tt[..., off_res:]).reshape(B, T, n, n) + self.b[off_res:].reshape(n, n)

    H_pre = pre_raw.sigmoid()
    H_post = 2.0 * post_raw.sigmoid()
    H_res = _sinkhorn_knopp(res_raw, iters=self.sinkhorn_iters)
    return H_pre.to(dtype=x_stream.dtype), H_post.to(dtype=x_stream.dtype), H_res.to(dtype=x_stream.dtype)


class AblationBlock(nn.Module):
  def __init__(
    self,
    *,
    dim: int,
    n_heads: int,
    hidden: int,
    width_kind: str,
    hidden_small: int,
    mlp_type: str,
    canon_kernel: int | None,
    canon_set: str,
    attn_kind: str,
    attn_window: int | None,
    eps: float,
  ):
    super().__init__()
    self.dim = int(dim)
    self.norm1 = _RMSNormF32(self.dim, eps)
    self.norm2 = _RMSNormF32(self.dim, eps)

    canon_set = str(canon_set)
    self.canonA = CanonConv(dim=self.dim, kernel_size=int(canon_kernel)) if canon_kernel is not None and "A" in canon_set else None
    self.canonC = CanonConv(dim=self.dim, kernel_size=int(canon_kernel)) if canon_kernel is not None and "C" in canon_set else None
    canon_qkv_kernel = int(canon_kernel) if canon_kernel is not None and "B" in canon_set else None

    attn_kind = str(attn_kind)
    if attn_kind == "converged":
      if attn_window is None:
        raise ValueError("attn=converged requires attn_window")
      self.attn = ConvergedSelfAttention(
        dim=self.dim,
        n_heads=int(n_heads),
        window_init=int(attn_window),
        eps=float(eps),
        canon_qkv_kernel=canon_qkv_kernel,
      )
    elif attn_kind in ("global", "local"):
      self.attn = CausalSelfAttention(dim=self.dim, n_heads=int(n_heads), window=attn_window, canon_qkv_kernel=canon_qkv_kernel)
    else:
      raise ValueError(f"Unknown attn_kind={attn_kind!r}. Use one of: global, local, converged.")

    if width_kind == "fixed":
      canon_hidden_kernel = int(canon_kernel) if canon_kernel is not None and "D" in canon_set else None
      mlp_type = str(mlp_type)
      if mlp_type == "silu":
        self.mlp = StandardMLP(dim=self.dim, hidden=int(hidden), canon_hidden_kernel=canon_hidden_kernel)
      elif mlp_type == "swiglu":
        self.mlp = SwiGLUMLP(dim=self.dim, hidden=int(hidden), canon_hidden_kernel=canon_hidden_kernel)
      else:
        raise ValueError(f"Unknown mlp_type={mlp_type!r}. Use one of: silu, swiglu.")
    elif width_kind == "matformer":
      if canon_kernel is not None and "D" in canon_set:
        raise ValueError("canon_set includes 'D' but width=matformer is not supported (mlp shape mismatch)")
      self.mlp = MatFormerMLP(dim=self.dim, hidden_small=int(hidden_small), hidden_large=int(hidden))
    else:
      raise ValueError(f"Unknown width_kind={width_kind!r}")

  def forward(
    self,
    x: torch.Tensor,
    *,
    collect_stats: bool,
    canon_alphaA: torch.Tensor | None = None,
    canon_alphaB: torch.Tensor | None = None,
    canon_alphaC: torch.Tensor | None = None,
    canon_alphaD: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, dict[str, float]]:
    stats: dict[str, float] = {}

    x1 = self.norm1(x)
    if self.canonA is not None:
      a = canon_alphaA
      if a is None:
        x1 = x1 + self.canonA(x1)
      else:
        x1 = x1 + a * self.canonA(x1)
    x = x + self.attn(x1, canonB_alpha=canon_alphaB, collect_stats=collect_stats)

    if isinstance(self.mlp, MatFormerMLP):
      x2 = self.norm2(x)
      if self.canonC is not None:
        a = canon_alphaC
        if a is None:
          x2 = x2 + self.canonC(x2)
        else:
          x2 = x2 + a * self.canonC(x2)
      x = x + self.mlp(x2, collect_stats=collect_stats)
      if collect_stats:
        stats.update(self.mlp.stats())
    else:
      x2 = self.norm2(x)
      if self.canonC is not None:
        a = canon_alphaC
        if a is None:
          x2 = x2 + self.canonC(x2)
        else:
          x2 = x2 + a * self.canonC(x2)
      x = x + self.mlp(x2, canonD_alpha=canon_alphaD)

    return x, stats


@dataclass(frozen=True)
class Variant:
  width: str  # fixed | matformer
  residual: str  # vanilla | altup | mhc
  memory: str  # none | engram | ple_ngrammer
  attn: str  # global | local | converged | mixed
  precond: str = "none"  # none | canon
  canon_set: str = ""  # subset of ABCD (A/C residual-stream, B=QKV, D=MLP-hidden)
  attn_window: int | None = None
  attn_global_every: int | None = None  # for mixed: place global layer every N (rest local)

  def key(self) -> str:
    # Build attention specifier
    if self.attn == "mixed" and self.attn_global_every is not None:
      a = f"mixed:G1L{self.attn_global_every - 1}:{self.attn_window}"
    elif self.attn_window is None:
      a = self.attn
    else:
      a = f"{self.attn}:{self.attn_window}"
    if self.precond == "none":
      return f"width={self.width},residual={self.residual},memory={self.memory},attn={a}"
    if self.precond == "canon" and self.residual == "vanilla":
      # Back-compat shorthand: "canon" historically meant vanilla + CanonConv.
      if not self.canon_set:
        return f"width={self.width},residual=canon,memory={self.memory},attn={a}"
      return f"width={self.width},residual=canon,canon_set={self.canon_set},memory={self.memory},attn={a}"
    if not self.canon_set:
      return f"width={self.width},residual={self.residual},precond={self.precond},memory={self.memory},attn={a}"
    return f"width={self.width},residual={self.residual},precond={self.precond},canon_set={self.canon_set},memory={self.memory},attn={a}"


@dataclass(frozen=True)
class ModelCfg:
  vocab_size: int = 10240
  dim: int = 256
  hidden: int = 512
  hidden_small: int = 256
  mlp_type: str = "silu"  # silu | swiglu (required for Canon-D faithfulness)
  n_layers: int = 6
  n_heads: int = 4
  eps: float = 1e-6
  # Memory
  mem_table_size: int = 4096
  mem_dim: int = 128
  mem_multiplier: int = 1_000_003
  # Residual params
  altup_inputs: int = 4
  altup_active_idx: int = 0
  altup_coef_clip: float | None = 120.0
  mhc_streams: int = 4
  mhc_sinkhorn_iters: int = 20
  # Canon
  canon_kernel: int = 4


class AblationTransformer(nn.Module):
  def __init__(self, *, variant: Variant, cfg: ModelCfg, stack_gate: bool, gate_cond: str, gate_budget: float):
    super().__init__()
    self.variant = variant
    self.cfg = cfg
    self.embed = nn.Embedding(int(cfg.vocab_size), int(cfg.dim))
    self.unembed = nn.Linear(int(cfg.dim), int(cfg.vocab_size), bias=False)

    canon_kernel: int | None = None
    canon_set = ""
    if variant.precond == "canon":
      canon_kernel = int(cfg.canon_kernel)
      canon_set = (variant.canon_set or "AC").upper()
      bad = sorted(set(canon_set) - set("ABCD"))
      if bad:
        raise ValueError(f"canon_set must be subset of ABCD, got {variant.canon_set!r} (bad={bad})")

    attn_window = None
    if variant.attn in ("local", "converged", "mixed"):
      if variant.attn_window is None:
        raise ValueError(f"attn={variant.attn} requires attn_window")
      attn_window = int(variant.attn_window)
    elif variant.attn != "global":
      raise ValueError(f"Unknown attn={variant.attn!r}. Use one of: global, local, converged, mixed.")

    # For mixed attention: determine per-layer attention kind
    def get_attn_kind_for_layer(layer_idx: int) -> str:
      if variant.attn == "mixed":
        if variant.attn_global_every is None:
          raise ValueError("attn=mixed requires attn_global_every")
        # Layer 0, N, 2N, ... are global; rest are local
        if layer_idx % variant.attn_global_every == 0:
          return "global"
        return "local"
      return str(variant.attn)

    def get_attn_window_for_layer(layer_idx: int) -> int | None:
      if variant.attn == "mixed":
        if layer_idx % variant.attn_global_every == 0:
          return None  # global layers have no window
        return attn_window
      return attn_window

    self.blocks = nn.ModuleList([
      AblationBlock(
        dim=int(cfg.dim),
        n_heads=int(cfg.n_heads),
        hidden=int(cfg.hidden),
        width_kind=variant.width,
        hidden_small=int(cfg.hidden_small),
        mlp_type=str(cfg.mlp_type),
        canon_kernel=canon_kernel,
        canon_set=canon_set,
        attn_kind=get_attn_kind_for_layer(i),
        attn_window=get_attn_window_for_layer(i),
        eps=float(cfg.eps),
      )
      for i in range(int(cfg.n_layers))
    ])

    # Memory module.
    if variant.memory == "none":
      self.memory = None
    elif variant.memory == "engram":
      self.memory = EngramMemory(
        n_layers=int(cfg.n_layers),
        dim=int(cfg.dim),
        eps=float(cfg.eps),
        table_size=int(cfg.mem_table_size),
        mem_dim=int(cfg.mem_dim),
        multiplier=int(cfg.mem_multiplier),
      )
    elif variant.memory == "ple_ngrammer":
      self.memory = PleNgrammerMemory(
        n_layers=int(cfg.n_layers),
        dim=int(cfg.dim),
        table_size=int(cfg.mem_table_size),
        mem_dim=int(cfg.mem_dim),
        multiplier=int(cfg.mem_multiplier),
      )
    else:
      raise ValueError(f"Unknown memory={variant.memory!r}")

    # Residual law modules.
    if variant.residual == "altup":
      self.altup = nn.ModuleList([
        AltUp(
          d_model=int(cfg.dim),
          num_inputs=int(cfg.altup_inputs),
          active_idx=int(cfg.altup_active_idx),
          coef_clip=cfg.altup_coef_clip,
          eps=float(cfg.eps),
        )
        for _ in range(int(cfg.n_layers))
      ])
    else:
      self.altup = None

    if variant.residual == "mhc":
      self.mhc = nn.ModuleList([
        MHCMaps(
          n=int(cfg.mhc_streams),
          dim=int(cfg.dim),
          eps=float(cfg.eps),
          sinkhorn_iters=int(cfg.mhc_sinkhorn_iters),
        )
        for _ in range(int(cfg.n_layers))
      ])
    else:
      self.mhc = None

    nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
    nn.init.normal_(self.unembed.weight, mean=0.0, std=0.02)

    self.gate_cond = str(gate_cond)
    if self.gate_cond not in ("embed_pool", "layer1_pool"):
      raise ValueError(f"Unknown gate_cond={self.gate_cond!r}. Use one of: embed_pool, layer1_pool.")

    self.gate_budget = float(gate_budget)
    if self.gate_budget <= 0:
      raise ValueError(f"gate_budget must be > 0 (got {self.gate_budget})")

    gate_hidden = max(8, int(cfg.dim) // 4)
    self.stack_gate = StackGateMLP(dim=int(cfg.dim), hidden=gate_hidden, eps=float(cfg.eps)) if bool(stack_gate) else None
    if self.stack_gate is not None and self.variant.residual == "altup":
      raise ValueError("stack-gate is not supported for residual=altup in this harness (yet)")

    self._last_stats: dict[str, float] = {}
    self._last_hiddens: list[torch.Tensor] | None = None

  def forward(
    self,
    input_ids: torch.Tensor,
    *,
    collect_stats: bool,
    collect_hiddens: bool = False,
  ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    # input_ids: [B,T]
    x0 = self.embed(input_ids)
    B, T, C = x0.shape

    bigram_ids = None
    if self.memory is not None:
      bigram_ids = _bigram_ids(
        input_ids,
        table_size=int(self.cfg.mem_table_size),
        multiplier=int(self.cfg.mem_multiplier),
      )

    stats: dict[str, float] = {}
    hiddens: list[torch.Tensor] | None = [] if collect_hiddens else None

    # Stack gates (prompt-level).
    one = torch.ones((B, 1, 1), device=x0.device, dtype=x0.dtype)
    zero = torch.zeros((B, 1, 1), device=x0.device, dtype=x0.dtype)
    alphaA = one
    alphaB = one
    alphaC = one
    alphaD = one
    alphaE = one  # memory fixed-on (Engram has internal gate)
    gate_penalty = x0.new_zeros(())  # scalar

    start_layer = 0
    x_start = x0

    if self.stack_gate is not None:
      if self.gate_cond == "embed_pool":
        pooled = _masked_mean_pool(x0, input_ids=input_ids, pad_token_id=int(EOS))
      else:
        # layer1_pool: run block0 once in a "minimal path" (no memory, no Canon) and pool that.
        blk0 = self.blocks[0]
        x_probe, _ = blk0(
          x0,
          collect_stats=False,
          canon_alphaA=zero,
          canon_alphaB=zero,
          canon_alphaC=zero,
          canon_alphaD=zero,
        )
        pooled = _masked_mean_pool(x_probe, input_ids=input_ids, pad_token_id=int(EOS))
        x_start = x_probe
        start_layer = 1
        if hiddens is not None:
          hiddens.append(x_probe)

      gates = self.stack_gate(pooled)  # [B,4]
      alphaA = gates[:, 0].view(B, 1, 1)
      alphaB = gates[:, 1].view(B, 1, 1)
      alphaC = gates[:, 2].view(B, 1, 1)
      alphaD = gates[:, 3].view(B, 1, 1)

      # Soft budget constraint (router-style): penalize only when exceeding budget.
      # This avoids L1-style collapse (α→0) and leaves the interior unconstrained.
      #
      # Budget applies only to Canon placement gates; memory is fixed-on.
      budget_used = alphaA + alphaB + alphaC + alphaD  # [B,1,1]
      budget_over = (budget_used - float(self.gate_budget)).clamp_min(0.0)
      gate_penalty = (budget_over * budget_over).mean()

      if collect_stats:
        stats.update({
          "stack.alphaA_mean": float(alphaA.mean().item()),
          "stack.alphaB_mean": float(alphaB.mean().item()),
          "stack.alphaC_mean": float(alphaC.mean().item()),
          "stack.alphaD_mean": float(alphaD.mean().item()),
          "stack.budget_used_mean": float(budget_used.mean().item()),
          "stack.budget_over_mean": float(budget_over.mean().item()),
          "stack.penalty_mean": float(gate_penalty.detach().float().item()),
        })

    if self.variant.residual == "vanilla":
      x = x_start
      for l in range(int(start_layer), len(self.blocks)):
        blk = self.blocks[l]
        if self.memory is not None and bigram_ids is not None:
          x = x + alphaE * self.memory(x=x, bigram_ids=bigram_ids, layer_id=l, collect_stats=collect_stats)
        x, blk_stats = blk(
          x,
          collect_stats=collect_stats,
          canon_alphaA=alphaA,
          canon_alphaB=alphaB,
          canon_alphaC=alphaC,
          canon_alphaD=alphaD,
        )
        if hiddens is not None:
          hiddens.append(x)
        if collect_stats:
          for k, v in blk.attn.stats().items():
            stats[f"attn.layer{l}.{k}"] = float(v)
          stats.update({f"block{l}.{k}": v for k, v in blk_stats.items()})
      if collect_stats and self.memory is not None:
        stats.update(self.memory.stats())
      self._last_stats = stats
      self._last_hiddens = hiddens
      return self.unembed(x), stats, gate_penalty

    if self.variant.residual == "altup":
      if self.altup is None:
        raise RuntimeError("altup modules missing")
      xs = [x0] * int(self.cfg.altup_inputs)
      for l, blk in enumerate(self.blocks):
        preds = self.altup[l].predict(xs, collect_stats=collect_stats)
        x = preds[int(self.cfg.altup_active_idx)]
        if collect_stats:
          for k, v in self.altup[l].stats().items():
            stats[f"altup.layer{l}.{k}"] = float(v)
        if self.memory is not None and bigram_ids is not None:
          x = x + self.memory(x=x, bigram_ids=bigram_ids, layer_id=l, collect_stats=collect_stats)
        x, blk_stats = blk(x, collect_stats=collect_stats)
        xs = self.altup[l].correct(preds, x)
        if hiddens is not None:
          hiddens.append(xs[int(self.cfg.altup_active_idx)])
        if collect_stats:
          for k, v in blk.attn.stats().items():
            stats[f"attn.layer{l}.{k}"] = float(v)
          stats.update({f"block{l}.{k}": v for k, v in blk_stats.items()})
      x = xs[int(self.cfg.altup_active_idx)]
      if collect_stats and self.memory is not None:
        stats.update(self.memory.stats())
      self._last_stats = stats
      self._last_hiddens = hiddens
      return self.unembed(x), stats, gate_penalty

    if self.variant.residual == "mhc":
      if self.mhc is None:
        raise RuntimeError("mhc modules missing")
      n = int(self.cfg.mhc_streams)
      if int(start_layer) == 0:
        x_stream = x0.unsqueeze(2).expand(-1, -1, n, -1).contiguous()
      else:
        # layer1_pool: we already computed x_probe as x_start and appended it to hiddens.
        x_stream = x_start.unsqueeze(2).expand(-1, -1, n, -1).contiguous()

      for l in range(int(start_layer), len(self.blocks)):
        blk = self.blocks[l]
        H_pre, H_post, H_res = self.mhc[l](x_stream)

        x_in = torch.einsum("btn,btnc->btc", H_pre, x_stream)
        if self.memory is not None and bigram_ids is not None:
          x_in = x_in + alphaE * self.memory(x=x_in, bigram_ids=bigram_ids, layer_id=l, collect_stats=collect_stats)
        y, blk_stats = blk(
          x_in,
          collect_stats=collect_stats,
          canon_alphaA=alphaA,
          canon_alphaB=alphaB,
          canon_alphaC=alphaC,
          canon_alphaD=alphaD,
        )
        y_stream = y.unsqueeze(2) * H_post.unsqueeze(-1)
        x_res = torch.einsum("btij,btjc->btic", H_res, x_stream)
        x_stream = x_res + y_stream
        if hiddens is not None:
          hiddens.append(x_stream.mean(dim=2))
        if collect_stats:
          for k, v in blk.attn.stats().items():
            stats[f"attn.layer{l}.{k}"] = float(v)
          stats.update({f"block{l}.{k}": v for k, v in blk_stats.items()})
      x = x_stream.mean(dim=2)
      if collect_stats and self.memory is not None:
        stats.update(self.memory.stats())
      self._last_stats = stats
      self._last_hiddens = hiddens
      return self.unembed(x), stats, gate_penalty

    raise ValueError(f"Unknown residual={self.variant.residual!r}")


# ----------------------------- runner ---------------------------------------

@dataclass(frozen=True)
class RunCfg:
  output: Path
  steps: int = 2000
  seed: int = 42
  # Data
  tasks: tuple[str, ...] = (
    "ngram:1.0:n_symbols=512,n_steps=128,table_seed=0",
    "depo:1.0:n_entities=50,max_hops=4",
    "mano:1.0:depth=3,ops=asm",
  )
  n_train: int = 20000
  n_valid: int = 2000
  seq_len: int = 256
  # Optim
  lr: float = 3e-4
  weight_decay: float = 0.1
  batch_size: int = 32
  log_every: int = 50
  eval_every: int = 200
  # Loss
  loss_mode: str = "answer_only"  # answer_only | full
  # Diagnostics
  logitlens: bool = False
  logitlens_n: int = 256  # examples per task
  cka: bool = False
  cka_n: int = 256  # examples per task
  cka_baseline: str | None = None  # variant key to use as CKA baseline (default: vanilla)
  layer_ce: bool = False  # per-layer CE to ground-truth
  layer_ce_n: int = 256  # examples per task
  lano_cfg_kl: bool = False  # DP-computable next-token KL for lano_cfg (PhysicsLM4-style)
  lano_cfg_kl_n: int = 16  # examples per task for DP-KL (DP is expensive)
  slice_metrics: bool = False  # per-task slice accuracy (hops/depth/mode)
  slice_metrics_n: int = 512  # examples per task for slice metrics
  tag_task: bool = False  # prepend a task-tag token after BOS
  # Prompt-conditioned stack gates (Canon + memory + mHC)
  stack_gate: bool = False
  gate_cond: str = "layer1_pool"  # embed_pool | layer1_pool
  gate_lambda: float = 0.0  # budget penalty weight
  gate_budget: float = 3.0  # soft budget for Σ α_i (penalize only when exceeded)
  # Seeds (split init vs data for cleaner CKA)
  init_seed: int | None = None  # if None, uses seed; else separate init seed
  # Model
  model: ModelCfg = ModelCfg()


def _cfg_to_json(cfg: RunCfg) -> dict:
  d = asdict(cfg)
  d["output"] = str(cfg.output)
  return d


def _evaluate(model: AblationTransformer, split: Split, *, batch_size: int, loss_mode: str, device: torch.device) -> dict[str, float]:
  model.eval()
  n = int(split.tokens.size(0))
  n_batches = math.ceil(n / int(batch_size))
  tot_loss = 0.0
  tot_acc = 0.0
  tot_em = 0.0
  tot_batches = 0

  with torch.no_grad():
    for b in range(n_batches):
      start = b * int(batch_size)
      end = min(n, start + int(batch_size))
      batch = split.tokens[start:end].to(device, non_blocking=True)
      batch_labels = split.labels[start:end].to(device, non_blocking=True)
      x_in = batch[:, :-1]
      y = batch[:, 1:]
      y_labels = batch_labels[:, 1:]

      logits, _, _ = model(x_in, collect_stats=False)

      mask = _mask_for_loss_mode(loss_mode=loss_mode, y=y, y_labels=y_labels)

      loss, m = _loss_and_metrics(logits, y, answer_mask=mask)
      loss_val = float(loss.item())
      # Clamp to prevent inf/nan from corrupting the average
      if not math.isfinite(loss_val) or loss_val > 100.0:
        loss_val = 100.0
      tot_loss += loss_val
      tot_acc += float(m["answer_token_acc"])
      tot_em += float(m["answer_exact_match"])
      tot_batches += 1

  return {
    "loss": tot_loss / max(1, tot_batches),
    "answer_token_acc": tot_acc / max(1, tot_batches),
    "answer_exact_match": tot_em / max(1, tot_batches),
  }


def _slice_key(task: str, input_ids: torch.Tensor) -> str:
  """
  Derive a "difficulty slice" key from the token sequence.

  This avoids carrying generator metadata through the Split, while enabling
  paper-style curves (e.g., Depo hops, Mano depth).
  """
  # input_ids: [T] (includes BOS...EOS padding)
  if task == "depo":
    # Hop token is encoded as 8000 + k and appears before ANSWER_START.
    hop_tokens = input_ids[(input_ids >= 8001) & (input_ids < 9000)]
    if hop_tokens.numel() == 0:
      return "hops=?"
    k = int(hop_tokens[0].item()) - 8000
    return f"hops={k}"
  if task == "mano":
    # Depth equals number of operator tokens (4000..4003) in the prefix expression.
    ops = ((input_ids >= MANO_OP_BASE) & (input_ids < MANO_OP_BASE + 4)).sum().item()
    return f"depth={int(ops)}"
  if task == "ngram_polysemy":
    # Mode token is at position 1: 5990 (A) / 5991 (B).
    mode = input_ids[(input_ids == int(NGRAM_MODE_A)) | (input_ids == int(NGRAM_MODE_B))]
    if mode.numel() == 0:
      return "mode=?"
    return "mode=B" if int(mode[0].item()) == int(NGRAM_MODE_B) else "mode=A"
  return "all"


def _evaluate_gate_means(
  *,
  model: AblationTransformer,
  split: Split,
  batch_size: int,
  loss_mode: str,
  device: torch.device,
) -> list[float] | None:
  """
  Estimate per-layer memory gate means for a split (averaged over batches).

  NOTE: This is slice-level (distribution-level) gating, not per-token attribution.
  """
  if model.memory is None:
    return None
  model.eval()
  n_layers = len(model.blocks)
  sums = [0.0 for _ in range(n_layers)]
  n_batches = 0

  n = int(split.tokens.size(0))
  n_batches_total = math.ceil(n / int(batch_size))

  with torch.no_grad():
    for b in range(n_batches_total):
      start = b * int(batch_size)
      end = min(n, start + int(batch_size))
      batch = split.tokens[start:end].to(device, non_blocking=True)
      x_in = batch[:, :-1]

      _, stats, _ = model(x_in, collect_stats=True)
      got_any = False
      for l in range(n_layers):
        k = f"mem.layer{l}.gate_mean"
        if k in stats:
          sums[l] += float(stats[k])
          got_any = True
      if got_any:
        n_batches += 1

  if n_batches == 0:
    return None
  return [s / n_batches for s in sums]


def _evaluate_stack_gates(
  *,
  model: AblationTransformer,
  split: Split,
  batch_size: int,
  device: torch.device,
) -> dict[str, float] | None:
  """
  Estimate prompt-level stack gate means for a split (averaged over batches).

  Returns (if enabled):
    {
      "alphaA": ...,
      "alphaB": ...,
      "alphaC": ...,
      "alphaD": ...,
      "budget_used": ...,
      "budget_over": ...,
      "penalty": ...,
    }
  """
  if model.stack_gate is None:
    return None

  keys = [
    ("alphaA", "stack.alphaA_mean"),
    ("alphaB", "stack.alphaB_mean"),
    ("alphaC", "stack.alphaC_mean"),
    ("alphaD", "stack.alphaD_mean"),
    ("budget_used", "stack.budget_used_mean"),
    ("budget_over", "stack.budget_over_mean"),
    ("penalty", "stack.penalty_mean"),
  ]

  model.eval()
  sums = {short: 0.0 for short, _ in keys}
  n_batches = 0

  n = int(split.tokens.size(0))
  n_batches_total = math.ceil(n / int(batch_size))

  with torch.no_grad():
    for b in range(n_batches_total):
      start = b * int(batch_size)
      end = min(n, start + int(batch_size))
      batch = split.tokens[start:end].to(device, non_blocking=True)
      x_in = batch[:, :-1]

      _, stats, _ = model(x_in, collect_stats=True)
      got = False
      for short, full in keys:
        if full in stats:
          sums[short] += float(stats[full])
          got = True
      if got:
        n_batches += 1

  if n_batches == 0:
    return None
  return {k: float(v / n_batches) for k, v in sums.items()}


def _evaluate_attn_means(
  *,
  model: AblationTransformer,
  split: Split,
  batch_size: int,
  device: torch.device,
) -> dict[str, list[float]] | None:
  """
  Estimate per-layer converged-attention stats for a split (averaged over batches).

  Returns (if present in stats):
    {
      "attn_local_gate_mean_by_layer": [...],
      "attn_window_frac_mean_by_layer": [...],
    }
  """
  model.eval()
  n_layers = len(model.blocks)
  local_sum = [0.0 for _ in range(n_layers)]
  win_frac_sum = [0.0 for _ in range(n_layers)]
  n_batches = 0

  n = int(split.tokens.size(0))
  n_batches_total = math.ceil(n / int(batch_size))

  with torch.no_grad():
    for b in range(n_batches_total):
      start = b * int(batch_size)
      end = min(n, start + int(batch_size))
      batch = split.tokens[start:end].to(device, non_blocking=True)
      x_in = batch[:, :-1]

      _, stats, _ = model(x_in, collect_stats=True)
      got_any = False
      for l in range(n_layers):
        k = f"attn.layer{l}.local_gate_mean"
        if k in stats:
          local_sum[l] += float(stats[k])
          got_any = True
        k = f"attn.layer{l}.window_frac_mean"
        if k in stats:
          win_frac_sum[l] += float(stats[k])
          got_any = True
      if got_any:
        n_batches += 1

  if n_batches == 0:
    return None

  return {
    "attn_local_gate_mean_by_layer": [float(x / n_batches) for x in local_sum],
    "attn_window_frac_mean_by_layer": [float(x / n_batches) for x in win_frac_sum],
  }


def _evaluate_slices(
  *,
  model: AblationTransformer,
  split: Split,
  batch_size: int,
  loss_mode: str,
  device: torch.device,
  include_layer_ce: bool,
) -> dict[str, dict]:
  """
  Per-task slice evaluation (difficulty curves + optional gate stats).

  Returns:
    {
      slice_key: {
        "n": int,
        "loss": ...,
        "answer_token_acc": ...,
        "answer_exact_match": ...,
        "gate_mean_by_layer": [...],  # memory only (if present)
        "attn_local_gate_mean_by_layer": [...],  # converged attn only (if present)
        "attn_window_frac_mean_by_layer": [...],  # converged attn only (if present)
        "stack_gate": {...},  # prompt-gate only (if present)
      }
    }
  """
  # Group indices by slice key.
  buckets: dict[str, list[int]] = {}
  for i in range(int(split.tokens.size(0))):
    task = split.tasks[i]
    key = _slice_key(task, split.tokens[i, :-1])
    buckets.setdefault(key, []).append(i)

  out: dict[str, dict] = {}
  for key, idx in sorted(buckets.items(), key=lambda kv: kv[0]):
    sub = Split(tokens=split.tokens[idx], labels=split.labels[idx], tasks=[split.tasks[i] for i in idx])
    metrics = _evaluate(model, sub, batch_size=min(int(batch_size), int(sub.tokens.size(0))), loss_mode=loss_mode, device=device)

    layer_ce = None
    if bool(include_layer_ce):
      layer_ce = _evaluate_layer_ce(
        model=model,
        split=sub,
        batch_size=min(int(batch_size), int(sub.tokens.size(0))),
        loss_mode=loss_mode,
        device=device,
      )

    gate_means = _evaluate_gate_means(
      model=model,
      split=sub,
      batch_size=min(int(batch_size), int(sub.tokens.size(0))),
      loss_mode=loss_mode,
      device=device,
    )
    attn_means = _evaluate_attn_means(
      model=model,
      split=sub,
      batch_size=min(int(batch_size), int(sub.tokens.size(0))),
      device=device,
    )
    stack_gates = _evaluate_stack_gates(
      model=model,
      split=sub,
      batch_size=min(int(batch_size), int(sub.tokens.size(0))),
      device=device,
    )
    rec: dict[str, object] = {"n": int(sub.tokens.size(0)), **metrics}
    if layer_ce is not None:
      rec["layer_ce"] = layer_ce
    if gate_means is not None:
      rec["gate_mean_by_layer"] = [float(x) for x in gate_means]
    if attn_means is not None:
      for k, v in attn_means.items():
        rec[k] = [float(x) for x in v]
    if stack_gates is not None:
      rec["stack_gate"] = {k: float(v) for k, v in stack_gates.items()}
    out[key] = rec
  return out


def _evaluate_logitlens(
  *,
  model: AblationTransformer,
  split: Split,
  batch_size: int,
  loss_mode: str,
  device: torch.device,
) -> list[float]:
  """
  Compute Engram-style LogitLens KL curves: KL(p_final || p_layer) by layer.
  """
  model.eval()
  n_layers = len(model.blocks)
  kl_sum = torch.zeros((n_layers,), dtype=torch.float64)
  n_batches = 0

  n = int(split.tokens.size(0))
  n_batches_total = math.ceil(n / int(batch_size))

  with torch.no_grad():
    for b in range(n_batches_total):
      start = b * int(batch_size)
      end = min(n, start + int(batch_size))
      batch = split.tokens[start:end].to(device, non_blocking=True)
      batch_labels = split.labels[start:end].to(device, non_blocking=True)
      x_in = batch[:, :-1]
      y = batch[:, 1:]
      y_labels = batch_labels[:, 1:]

      final_logits, _, _ = model(x_in, collect_stats=False, collect_hiddens=True)
      hiddens = model._last_hiddens
      if hiddens is None or len(hiddens) != n_layers:
        raise RuntimeError("collect_hiddens=True did not produce per-layer hidden states")

      mask = _mask_for_loss_mode(loss_mode=loss_mode, y=y, y_labels=y_labels)

      for l, h in enumerate(hiddens):
        layer_logits = model.unembed(h)
        kl = _logitlens_kl_to_final(final_logits=final_logits, layer_logits=layer_logits, token_mask=mask)
        kl_sum[l] += float(kl.item())
      n_batches += 1

  out = (kl_sum / max(1, n_batches)).tolist()
  return [float(x) for x in out]


def _evaluate_layer_ce(
  *,
  model: AblationTransformer,
  split: Split,
  batch_size: int,
  loss_mode: str,
  device: torch.device,
) -> dict:
  """
  Compute per-layer CE to ground-truth labels (not KL to final).

  This is the decisive metric for "are late layers doing useful work?":
  CE_l = CE(unembed(h_l), y) for each layer l.

  Returns:
    {
      "ce_by_layer": [ce_0, ce_1, ..., ce_L-1],
      "frac_late": fraction of CE reduction in last third of layers,
      "layer_contributions": [(ce_{l-1} - ce_l) for l in 1..L-1],
    }
  """
  model.eval()
  n_layers = len(model.blocks)
  ce_sum = torch.zeros((n_layers,), dtype=torch.float64)
  count = 0

  n = int(split.tokens.size(0))
  n_batches_total = math.ceil(n / int(batch_size))

  with torch.no_grad():
    for b in range(n_batches_total):
      start = b * int(batch_size)
      end = min(n, start + int(batch_size))
      batch = split.tokens[start:end].to(device, non_blocking=True)
      batch_labels = split.labels[start:end].to(device, non_blocking=True)
      x_in = batch[:, :-1]
      y = batch[:, 1:]
      y_labels = batch_labels[:, 1:]

      _, _, _ = model(x_in, collect_stats=False, collect_hiddens=True)
      hiddens = model._last_hiddens
      if hiddens is None or len(hiddens) != n_layers:
        raise RuntimeError("collect_hiddens=True did not produce per-layer hidden states")

      mask = _mask_for_loss_mode(loss_mode=loss_mode, y=y, y_labels=y_labels)

      denom = mask.sum().float().clamp(min=1.0)

      for l, h in enumerate(hiddens):
        layer_logits = model.unembed(h)
        # CE to ground-truth labels
        ce = torch.nn.functional.cross_entropy(
          layer_logits.reshape(-1, layer_logits.size(-1)),
          y.reshape(-1),
          reduction="none",
        ).reshape_as(y)
        ce_masked = (ce * mask).sum() / denom
        ce_sum[l] += float(ce_masked.item())
      count += 1

  ce_by_layer = (ce_sum / max(1, count)).tolist()

  # Compute layer contributions: how much CE drops from layer l-1 to l
  layer_contributions = []
  for l in range(1, n_layers):
    layer_contributions.append(ce_by_layer[l - 1] - ce_by_layer[l])

  # Prefix-min envelope to make "late contribution" well-defined even if CE is non-monotone.
  ce_env = []
  best = float("inf")
  for ce in ce_by_layer:
    best = min(best, float(ce))
    ce_env.append(best)
  env_contrib = [ce_env[l - 1] - ce_env[l] for l in range(1, n_layers)]

  # Compute frac_late: fraction of total envelope reduction in last third of layers.
  total_reduction = ce_env[0] - ce_env[-1]
  late_start = (2 * n_layers) // 3  # last third starts here
  late_reduction = sum(env_contrib[late_start - 1:]) if late_start > 0 else 0.0
  frac_late = late_reduction / total_reduction if total_reduction > 1e-8 else 0.0

  # Raw fraction (may be ill-conditioned if CE increases mid-depth).
  total_reduction_raw = ce_by_layer[0] - ce_by_layer[-1]
  late_reduction_raw = sum(layer_contributions[late_start - 1:]) if late_start > 0 else 0.0
  frac_late_raw = late_reduction_raw / total_reduction_raw if total_reduction_raw > 1e-8 else 0.0

  return {
    "ce_by_layer": [float(x) for x in ce_by_layer],
    "ce_env_by_layer": [float(x) for x in ce_env],
    "frac_late": float(frac_late),
    "frac_late_raw": float(frac_late_raw),
    "layer_contributions": [float(x) for x in layer_contributions],
    "env_contributions": [float(x) for x in env_contrib],
  }


def _evaluate_lano_cfg_kl(
  *,
  model: AblationTransformer,
  split: Split,
  batch_size: int,
  device: torch.device,
  task_kwargs: dict,
) -> dict[str, float]:
  """
  DP-computable next-token KL on lano_cfg.

  We compare model logits vs the exact next-token distribution from a layered CFG
  DP (PhysicsLM4-style). This is the clean signal for "did this mechanism improve
  global structure modeling?" without relying on brittle exact-match.
  """
  try:
    from nmoe.research.physics.generators.lano_cfg import build_layered_cfg, dp_next_token_distribution
  except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import lano_cfg DP utilities") from e

  # Reconstruct the fixed graph used by the generator from task kwargs.
  graph_seed = int(task_kwargs.get("graph_seed", 0))
  depth = int(task_kwargs.get("depth", 6))
  num_sym = int(task_kwargs.get("num_sym", 3))
  deg_min = int(task_kwargs.get("deg_min", 2))
  deg_max = int(task_kwargs.get("deg_max", 2))
  len_min = int(task_kwargs.get("len_min", 2))
  len_max = int(task_kwargs.get("len_max", 3))
  disallow_duplicate_sym = bool(task_kwargs.get("disallow_duplicate_sym", True))
  disallow_duplicate_seq = bool(task_kwargs.get("disallow_duplicate_seq", True))
  token_base = int(task_kwargs.get("token_base", 9400))

  graph = build_layered_cfg(
    graph_seed=graph_seed,
    depth=depth,
    num_sym=num_sym,
    deg_min=deg_min,
    deg_max=deg_max,
    len_min=len_min,
    len_max=len_max,
    disallow_duplicate_sym=disallow_duplicate_sym,
    disallow_duplicate_seq=disallow_duplicate_seq,
  )

  model.eval()
  n = int(split.tokens.size(0))
  n_batches = math.ceil(n / int(batch_size))

  kl_sum = 0.0
  n_ex = 0
  len_sum = 0.0

  with torch.no_grad():
    for b in range(n_batches):
      start = b * int(batch_size)
      end = min(n, start + int(batch_size))
      # DP is per-example anyway; keep batch size small.
      for i in range(start, end):
        seq = split.tokens[i]
        eos_pos = (seq == int(EOS)).nonzero(as_tuple=False)
        if eos_pos.numel() == 0:
          continue
        eos_pos = int(eos_pos[0].item())
        if eos_pos < 2:
          continue

        term_toks = seq[1:eos_pos]  # exclude BOS, exclude EOS
        # Map terminal token IDs back to 1..num_sym.
        term_ids: list[int] = []
        ok = True
        for t in term_toks.tolist():
          tid = int(t) - int(token_base) + 1
          if tid < 1 or tid > int(num_sym):
            ok = False
            break
          term_ids.append(tid)
        if not ok or not term_ids:
          continue

        # DP target distribution (len = L+1, each row = [EOS] + terminals).
        dp_probs, _ = dp_next_token_distribution(graph=graph, seq_term_ids=term_ids)

        # Model logits for positions 0..L (predict first terminal .. EOS).
        x_in = seq[:eos_pos].unsqueeze(0).to(device, non_blocking=True)  # [1, L+1]
        logits, _, _ = model(x_in, collect_stats=False)
        logits = logits[0]  # [L+1, V]

        # Extract model log-probs for EOS + terminals.
        idxs = [int(EOS)] + [int(token_base) + j for j in range(int(num_sym))]
        log_p = F.log_softmax(logits.to(dtype=torch.float32), dim=-1)
        log_p_sel = log_p[:, idxs].detach().cpu().to(dtype=torch.float64)  # [L+1, 1+T]

        target = torch.tensor(dp_probs, dtype=torch.float64)
        if target.shape != log_p_sel.shape:
          raise RuntimeError(f"shape mismatch: dp={tuple(target.shape)} model={tuple(log_p_sel.shape)}")

        eps = 1e-12
        target = (target + eps) / (target + eps).sum(dim=-1, keepdim=True)
        kl = (target * (target.log() - log_p_sel)).sum(dim=-1).mean()

        kl_sum += float(kl.item())
        len_sum += float(len(term_ids))
        n_ex += 1

  return {
    "dp_kl": float(kl_sum / max(1, n_ex)),
    "n": float(n_ex),
    "avg_len": float(len_sum / max(1, n_ex)),
  }


def _compute_depth_shift(cka_matrix: list[list[float]]) -> dict:
  """
  Compute depth shift metrics from a CKA matrix.

  Args:
    cka_matrix: rows=variant layers, cols=baseline layers

  Returns:
    {
      "argmax_by_layer": [argmax_j for each variant layer i],
      "depth_shift": mean(argmax_j - i) over layers (positive = aligns to deeper baseline),
      "early_shift": mean shift for first third of layers,
      "mid_shift": mean shift for middle third,
      "late_shift": mean shift for last third,
    }
  """
  n_layers = len(cka_matrix)
  argmax_by_layer = []
  shifts = []

  for i, row in enumerate(cka_matrix):
    j_max = max(range(len(row)), key=lambda j: row[j])
    argmax_by_layer.append(j_max)
    shifts.append(j_max - i)

  # Split into thirds
  third = max(1, n_layers // 3)
  early_shifts = shifts[:third]
  mid_shifts = shifts[third:2 * third]
  late_shifts = shifts[2 * third:]

  return {
    "argmax_by_layer": argmax_by_layer,
    "depth_shift": sum(shifts) / len(shifts) if shifts else 0.0,
    "early_shift": sum(early_shifts) / len(early_shifts) if early_shifts else 0.0,
    "mid_shift": sum(mid_shifts) / len(mid_shifts) if mid_shifts else 0.0,
    "late_shift": sum(late_shifts) / len(late_shifts) if late_shifts else 0.0,
  }


ProgressCallback = Callable[[int, int, float, dict], None]  # (step, total_steps, loss, metrics) -> None


def _train_variant(
  *,
  cfg: RunCfg,
  variant: Variant,
  train: Split,
  valid: Split,
  device: torch.device,
  return_model: bool = False,
  progress_callback: ProgressCallback | None = None,
) -> tuple[dict, AblationTransformer | None]:
  run_dir = cfg.output / "runs" / variant.key()
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "config.json").write_text(json.dumps(_cfg_to_json(cfg), indent=2), encoding="utf-8")
  (run_dir / "variant.json").write_text(json.dumps(asdict(variant), indent=2), encoding="utf-8")

  # Use init_seed for model initialization (separate from data seed for cleaner CKA)
  init_seed = cfg.init_seed if cfg.init_seed is not None else int(cfg.seed)
  torch.manual_seed(init_seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(init_seed)

  model = AblationTransformer(
    variant=variant,
    cfg=cfg.model,
    stack_gate=bool(cfg.stack_gate),
    gate_cond=str(cfg.gate_cond),
    gate_budget=float(cfg.gate_budget),
  ).to(device)
  model.train()
  opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay), betas=(0.9, 0.95), eps=1e-8)

  # Data seed for minibatch sampling
  rng = np.random.default_rng(int(cfg.seed))
  it = _iter_minibatches(tokens=train.tokens, labels=train.labels, batch_size=int(cfg.batch_size), rng=rng)

  log_path = run_dir / "train.jsonl"
  f = log_path.open("w", encoding="utf-8")

  def log(rec: dict) -> None:
    f.write(json.dumps(rec) + "\n")
    f.flush()

  for step in range(1, int(cfg.steps) + 1):
    batch, batch_labels = next(it)
    batch = batch.to(device, non_blocking=True)
    batch_labels = batch_labels.to(device, non_blocking=True)
    x_in = batch[:, :-1]
    y = batch[:, 1:]
    y_labels = batch_labels[:, 1:]

    collect = (step == 1) or (step % int(cfg.log_every) == 0)

    opt.zero_grad(set_to_none=True)
    logits, stats, gate_penalty = model(x_in, collect_stats=collect)

    mask = _mask_for_loss_mode(loss_mode=str(cfg.loss_mode), y=y, y_labels=y_labels)

    loss, m = _loss_and_metrics(logits, y, answer_mask=mask)
    if bool(cfg.stack_gate) and float(cfg.gate_lambda) > 0:
      loss = loss + float(cfg.gate_lambda) * gate_penalty
    loss.backward()
    opt.step()

    # Call progress callback every step
    if progress_callback is not None:
      progress_callback(step, int(cfg.steps), float(loss.item()), m)

    if collect:
      rec = {
        "step": int(step),
        "loss": float(loss.item()),
        **m,
        "stats": stats,
      }
      if bool(cfg.stack_gate):
        rec["gate_penalty"] = float(gate_penalty.detach().float().item())
      log(rec)

    if step % int(cfg.eval_every) == 0:
      v = _evaluate(model, valid, batch_size=int(cfg.batch_size), loss_mode=cfg.loss_mode, device=device)
      log({"step": int(step), "valid": v})
      model.train()

  f.close()

  final_train = _evaluate(model, train, batch_size=int(cfg.batch_size), loss_mode=cfg.loss_mode, device=device)
  final_valid = _evaluate(model, valid, batch_size=int(cfg.batch_size), loss_mode=cfg.loss_mode, device=device)
  out = {
    "run_dir": str(run_dir),
    "train_log": str(log_path),
    "final": {"train": final_train, "valid": final_valid},
  }

  if bool(cfg.logitlens):
    logitlens: dict[str, list[float]] = {}
    for task in sorted(set(valid.tasks)):
      idx = [i for i, t in enumerate(valid.tasks) if t == task]
      if not idx:
        continue
      idx = idx[: int(cfg.logitlens_n)]
      sub = Split(tokens=valid.tokens[idx], labels=valid.labels[idx], tasks=[task] * len(idx))
      logitlens[task] = _evaluate_logitlens(
        model=model,
        split=sub,
        batch_size=min(int(cfg.batch_size), int(sub.tokens.size(0))),
        loss_mode=cfg.loss_mode,
        device=device,
      )
    out["logitlens_valid"] = logitlens

  if return_model:
    return out, model
  return out, None


def _stage1_matrix() -> list[Variant]:
  return [
    Variant(width="fixed", residual="vanilla", memory="none", attn="global"),
    Variant(width="matformer", residual="vanilla", memory="none", attn="global"),
    Variant(width="fixed", residual="altup", memory="none", attn="global"),
    Variant(width="fixed", residual="mhc", memory="none", attn="global"),
    Variant(width="fixed", residual="vanilla", precond="canon", memory="none", attn="global"),
    Variant(width="fixed", residual="vanilla", memory="engram", attn="global"),
    Variant(width="fixed", residual="vanilla", memory="ple_ngrammer", attn="global"),
    Variant(width="fixed", residual="vanilla", memory="none", attn="local", attn_window=64),
  ]


def _attn_compare_matrix() -> list[Variant]:
  # Attention-only compare: baseline vs sliding window vs converged local/global router.
  return [
    Variant(width="fixed", residual="vanilla", memory="none", attn="global"),
    Variant(width="fixed", residual="vanilla", memory="none", attn="local", attn_window=64),
    Variant(width="fixed", residual="vanilla", memory="none", attn="converged", attn_window=64),
  ]


def _engram_repro_matrix() -> list[Variant]:
  # Paper-faithful axes: baseline vs memory (Engram vs PLE+Ngrammer), no extra knobs.
  return [
    Variant(width="fixed", residual="vanilla", memory="none", attn="global"),
    Variant(width="fixed", residual="vanilla", memory="engram", attn="global"),
    Variant(width="fixed", residual="vanilla", memory="ple_ngrammer", attn="global"),
  ]


def _canon_repro_matrix() -> list[Variant]:
  # Canon positions roughly match PhysicsLM4's A/B/C/D placements.
  return [
    Variant(width="fixed", residual="vanilla", memory="none", attn="global"),
    Variant(width="fixed", residual="vanilla", precond="canon", canon_set="A", memory="none", attn="global"),
    Variant(width="fixed", residual="vanilla", precond="canon", canon_set="C", memory="none", attn="global"),
    Variant(width="fixed", residual="vanilla", precond="canon", canon_set="AC", memory="none", attn="global"),
    Variant(width="fixed", residual="vanilla", precond="canon", canon_set="ABCD", memory="none", attn="global"),
  ]


def _stack_interactions_matrix() -> list[Variant]:
  # Core stack probes: Canon × Engram × mHC (with Canon "on everything" = ABCD).
  return [
    Variant(width="fixed", residual="vanilla", precond="canon", canon_set="ABCD", memory="engram", attn="global"),
    Variant(width="fixed", residual="mhc", precond="canon", canon_set="ABCD", memory="engram", attn="global"),
    Variant(width="fixed", residual="vanilla", precond="canon", canon_set="ABCD", memory="ple_ngrammer", attn="global"),
    Variant(width="fixed", residual="mhc", precond="canon", canon_set="ABCD", memory="ple_ngrammer", attn="global"),
  ]


def _canon_attn_cross_matrix() -> list[Variant]:
  """Canon placement × attention type cross to test Canon-A redundancy with local attention.

  Fixed: residual=mhc, memory=engram
  Varying: canon_set ∈ {B, BC, BCD, ABCD}, attn ∈ {global, local:64, converged:64}

  If BCD+local ≈ ABCD+global, Canon-A is redundant with local attention.
  """
  variants = []
  for canon_set in ["B", "BC", "BCD", "ABCD"]:
    for attn, attn_window in [("global", None), ("local", 64), ("converged", 64)]:
      variants.append(Variant(
        width="fixed",
        residual="mhc",
        precond="canon",
        canon_set=canon_set,
        memory="engram",
        attn=attn,
        attn_window=attn_window,
      ))
  return variants


def _parse_variant(s: str) -> Variant:
  fields = {}
  for part in s.split(","):
    k, v = part.split("=", 1)
    fields[k.strip()] = v.strip()
  width = fields.get("width", "fixed")
  residual_raw = fields.get("residual", "vanilla")
  precond = fields.get("precond", "none")
  canon_set = fields.get("canon_set", "")
  residual = residual_raw
  if residual_raw == "canon":
    residual = "vanilla"
    if precond == "none":
      precond = "canon"
  if canon_set and precond == "none":
    precond = "canon"
  memory = fields.get("memory", "none")
  attn_raw = fields.get("attn", "global")
  attn = attn_raw
  attn_window = None
  attn_global_every = None
  if ":" in attn_raw:
    parts = attn_raw.split(":")
    attn = parts[0]
    if attn == "mixed":
      # Parse mixed:G1L4:64 -> 1 global every 5 layers, window=64
      if len(parts) != 3:
        raise ValueError(f"attn=mixed requires format 'mixed:GxLy:W' (e.g., mixed:G1L4:64), got {attn_raw!r}")
      ratio_str, window_str = parts[1], parts[2]
      import re
      m = re.match(r"G(\d+)L(\d+)", ratio_str)
      if not m:
        raise ValueError(f"Invalid mixed ratio '{ratio_str}', expected GxLy (e.g., G1L4)")
      n_global, n_local = int(m.group(1)), int(m.group(2))
      if n_global != 1:
        raise ValueError(f"Only G1Ly patterns supported (1 global layer), got G{n_global}")
      attn_global_every = n_global + n_local  # e.g., G1L4 -> every 5 layers
      attn_window = int(window_str)
    else:
      attn_window = int(parts[1])
  if attn not in ("global", "local", "converged", "mixed"):
    raise ValueError(f"Unknown attn={attn_raw!r}. Use one of: global, local:W, converged:W, mixed:G1Ly:W.")
  if attn == "global" and attn_window is not None:
    raise ValueError("attn=global must not specify a window (no ':W')")
  if attn in ("local", "converged") and attn_window is None:
    raise ValueError(f"attn={attn} requires a window (use '{attn}:W')")
  if attn == "mixed" and (attn_window is None or attn_global_every is None):
    raise ValueError("attn=mixed requires format 'mixed:G1Ly:W' (e.g., mixed:G1L4:64)")
  if residual not in ("vanilla", "altup", "mhc"):
    raise ValueError(f"Unknown residual={residual_raw!r}. Use one of: vanilla, altup, mhc, canon (shorthand).")
  if precond not in ("none", "canon"):
    raise ValueError(f"Unknown precond={precond!r}. Use one of: none, canon.")
  canon_set = str(canon_set).upper()
  bad = sorted(set(canon_set) - set("ABCD"))
  if bad:
    raise ValueError(f"canon_set must be subset of ABCD, got {canon_set!r} (bad={bad})")
  return Variant(width=width, residual=residual, precond=precond, canon_set=canon_set, memory=memory, attn=attn, attn_window=attn_window, attn_global_every=attn_global_every)


def run_single_variant(
  *,
  output: Path,
  variant_spec: str,
  steps: int = 2000,
  seed: int = 42,
  dim: int = 256,
  n_layers: int = 6,
  seq_len: int = 2048,
  mlp_type: str = 'swiglu',
  tasks: list[str] | None = None,
  slice_metrics: bool = True,
  layer_ce: bool = False,
  progress_callback: ProgressCallback | None = None,
) -> dict:
  """Run a single physics variant with optional progress callback.

  This is the public API for running physics experiments from lab.py.
  Unlike main(), this function can be called directly from Python with
  a progress callback for notebook-native progress display.

  Args:
    output: Output directory for this run
    variant_spec: Variant specification string (e.g., "width=fixed,residual=vanilla,memory=none,attn=global")
    steps: Number of training steps
    seed: Random seed for data and init
    dim: Model dimension
    n_layers: Number of layers
    seq_len: Sequence length
    mlp_type: MLP type (silu or swiglu)
    tasks: Task specs (e.g., ['depo_v2:1.0'])
    slice_metrics: Enable per-slice metrics
    layer_ce: Enable per-layer CE analysis
    progress_callback: Optional callback(step, total_steps, loss, metrics) for progress updates

  Returns:
    Dictionary with training results and metrics
  """
  if tasks is None:
    tasks = ['depo_v2:1.0']

  # Build model config
  model_cfg = ModelCfg(
    dim=dim,
    hidden=dim * 4,
    hidden_small=dim,
    mlp_type=mlp_type,
    n_layers=n_layers,
    n_heads=max(4, dim // 64),
  )

  cfg = RunCfg(
    output=output,
    steps=steps,
    seed=seed,
    n_train=20000,
    n_valid=2000,
    seq_len=seq_len,
    batch_size=32,
    eval_every=200,
    log_every=50,
    loss_mode='answer_only',
    slice_metrics=slice_metrics,
    slice_metrics_n=512,
    layer_ce=layer_ce,
    layer_ce_n=256,
    tasks=tuple(tasks),
    model=model_cfg,
  )
  output.mkdir(parents=True, exist_ok=True)

  # Build data splits
  train_split = _build_split(
    tasks=cfg.tasks,
    n=cfg.n_train,
    seq_len=cfg.seq_len,
    seed=cfg.seed,
    tag_task=False,
  )
  valid_split = _build_split(
    tasks=cfg.tasks,
    n=cfg.n_valid,
    seq_len=cfg.seq_len,
    seed=cfg.seed + 1_000_000,
    tag_task=False,
  )

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  variant = _parse_variant(variant_spec)

  # Train with callback
  result, model = _train_variant(
    cfg=cfg,
    variant=variant,
    train=train_split,
    valid=valid_split,
    device=device,
    return_model=slice_metrics or layer_ce,
    progress_callback=progress_callback,
  )

  # Run slice metrics if requested
  if slice_metrics and model is not None:
    slices = {}
    for task in sorted(set(valid_split.tasks)):
      idx = [i for i, t in enumerate(valid_split.tasks) if t == task]
      if not idx:
        continue
      idx = idx[:cfg.slice_metrics_n]
      sub = Split(tokens=valid_split.tokens[idx], labels=valid_split.labels[idx], tasks=[task] * len(idx))
      slices[task] = _evaluate_slices(
        model=model,
        split=sub,
        batch_size=min(cfg.batch_size, sub.tokens.size(0)),
        loss_mode=cfg.loss_mode,
        device=device,
        include_layer_ce=layer_ce,
      )
    result["slices_valid"] = slices

  # Run layer CE if requested
  if layer_ce and model is not None:
    layer_ce_results = {}
    for task in sorted(set(valid_split.tasks)):
      idx = [i for i, t in enumerate(valid_split.tasks) if t == task]
      if not idx:
        continue
      idx = idx[:cfg.layer_ce_n]
      sub = Split(tokens=valid_split.tokens[idx], labels=valid_split.labels[idx], tasks=[task] * len(idx))
      layer_ce_results[task] = _evaluate_layer_ce(
        model=model,
        split=sub,
        batch_size=min(cfg.batch_size, sub.tokens.size(0)),
        loss_mode=cfg.loss_mode,
        device=device,
      )
    result["layer_ce_valid"] = layer_ce_results

  # Save summary
  summary = {variant.key(): result["final"]}
  (output / "analysis").mkdir(parents=True, exist_ok=True)
  (output / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2))

  if slice_metrics and "slices_valid" in result:
    (output / "analysis" / "slices_valid.json").write_text(json.dumps(result["slices_valid"], indent=2))

  if layer_ce and "layer_ce_valid" in result:
    (output / "analysis" / "layer_ce_valid.json").write_text(json.dumps(result["layer_ce_valid"], indent=2))

  return result


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description="Architecture ablations (physics harness).")
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--steps", type=int, default=2000)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--seq-len", type=int, default=256)
  p.add_argument("--n-train", type=int, default=20000)
  p.add_argument("--n-valid", type=int, default=2000)
  p.add_argument("--batch-size", type=int, default=32)
  p.add_argument("--eval-every", type=int, default=200)
  p.add_argument("--log-every", type=int, default=50)
  p.add_argument("--loss-mode", type=str, default="answer_only", choices=["answer_only", "full"])
  p.add_argument("--logitlens", action="store_true", help="Compute LogitLens KL-to-final curves on a valid subset.")
  p.add_argument("--logitlens-n", type=int, default=256, help="Examples per task for LogitLens evaluation.")
  p.add_argument("--cka", action="store_true", help="Compute Engram-style layer alignment via linear CKA.")
  p.add_argument("--cka-n", type=int, default=256, help="Examples per task for CKA evaluation.")
  p.add_argument("--cka-baseline", type=str, default=None, help="Variant key to use as CKA baseline (default: vanilla baseline).")
  p.add_argument("--layer-ce", action="store_true", help="Compute per-layer CE to ground-truth labels.")
  p.add_argument("--layer-ce-n", type=int, default=256, help="Examples per task for layer-CE evaluation.")
  p.add_argument("--lano-cfg-kl", action="store_true", help="Compute DP next-token KL for lano_cfg (PhysicsLM4-style).")
  p.add_argument("--lano-cfg-kl-n", type=int, default=16, help="Examples per task for lano_cfg DP-KL (DP is expensive).")
  p.add_argument("--slice-metrics", action="store_true", help="Compute per-task slice metrics (hops/depth/mode).")
  p.add_argument("--slice-metrics-n", type=int, default=512, help="Examples per task for slice metrics.")
  p.add_argument("--tag-task", action="store_true", help="Prepend a task-tag token after BOS for each sample (sanity for stack gating).")
  p.add_argument("--stack-gate", action="store_true", help="Enable prompt-conditioned stack gates (Canon + memory + mHC).")
  p.add_argument("--gate-cond", type=str, default="layer1_pool", choices=["embed_pool", "layer1_pool"], help="Gate conditioning signal.")
  p.add_argument("--gate-lambda", type=float, default=0.0, help="Budget penalty weight for gates (λ).")
  p.add_argument("--gate-budget", type=float, default=3.0, help="Soft budget for Σ α_i in stack-gate (penalize only when exceeded).")
  p.add_argument("--init-seed", type=int, default=None, help="Separate seed for model init (default: same as --seed).")
  p.add_argument("--dim", type=int, default=256, help="Model dimension (default: 256).")
  p.add_argument("--n-layers", type=int, default=6, help="Number of layers (default: 6).")
  p.add_argument("--mlp-type", type=str, default="silu", choices=["silu", "swiglu"], help="MLP type (silu or swiglu). Use swiglu for Canon-D faithfulness.")
  p.add_argument("--tasks", type=str, nargs="*", default=None, help="Task specs like 'ngram:1.0:n_steps=128'.")
  p.add_argument("--matrix", type=str, default=None, choices=["stage1", "attn_compare", "engram_repro", "canon_repro", "stack_interactions", "canon_attn_cross"], help="Run a built-in matrix.")
  p.add_argument("--variant", type=str, default=None, help="Single variant spec: width=...,residual=...,precond=...,memory=...,attn=global|local:W|converged:W")
  args = p.parse_args(argv)

  if args.matrix is None and args.variant is None:
    raise SystemExit("Provide either --matrix stage1 or --variant ...")

  # Build model config with custom dim/n_layers
  model_cfg = ModelCfg(
    dim=int(args.dim),
    hidden=int(args.dim) * 4,
    hidden_small=int(args.dim),
    mlp_type=str(args.mlp_type),
    n_layers=int(args.n_layers),
    n_heads=max(4, int(args.dim) // 64),
  )

  cfg = RunCfg(
    output=args.output,
    steps=int(args.steps),
    seed=int(args.seed),
    n_train=int(args.n_train),
    n_valid=int(args.n_valid),
    seq_len=int(args.seq_len),
    batch_size=int(args.batch_size),
    eval_every=int(args.eval_every),
    log_every=int(args.log_every),
    loss_mode=str(args.loss_mode),
    logitlens=bool(args.logitlens),
    logitlens_n=int(args.logitlens_n),
    cka=bool(args.cka),
    cka_n=int(args.cka_n),
    cka_baseline=args.cka_baseline,
    layer_ce=bool(args.layer_ce),
    layer_ce_n=int(args.layer_ce_n),
    lano_cfg_kl=bool(args.lano_cfg_kl),
    lano_cfg_kl_n=int(args.lano_cfg_kl_n),
    slice_metrics=bool(args.slice_metrics),
    slice_metrics_n=int(args.slice_metrics_n),
    tag_task=bool(args.tag_task),
    stack_gate=bool(args.stack_gate),
    gate_cond=str(args.gate_cond),
    gate_lambda=float(args.gate_lambda),
    gate_budget=float(args.gate_budget),
    init_seed=args.init_seed,
    tasks=tuple(args.tasks) if args.tasks else RunCfg.tasks,
    model=model_cfg,
  )
  cfg.output.mkdir(parents=True, exist_ok=True)

  # Record task kwargs (for evaluators that need generator parameters but Split only carries task names).
  task_kwargs_by_name: dict[str, dict] = {}
  for spec in cfg.tasks:
    ts = _parse_task_spec(spec)
    if ts.name in task_kwargs_by_name and task_kwargs_by_name[ts.name] != ts.kwargs:
      raise ValueError(f"Duplicate task {ts.name!r} with different kwargs is not supported in this harness.")
    task_kwargs_by_name[ts.name] = dict(ts.kwargs)

  train = _build_split(tasks=cfg.tasks, n=int(cfg.n_train), seq_len=int(cfg.seq_len), seed=int(cfg.seed), tag_task=bool(cfg.tag_task))
  valid = _build_split(tasks=cfg.tasks, n=int(cfg.n_valid), seq_len=int(cfg.seq_len), seed=int(cfg.seed) + 1_000_000, tag_task=bool(cfg.tag_task))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  variants = []
  if args.matrix == "stage1":
    variants = _stage1_matrix()
  if args.matrix == "attn_compare":
    variants = _attn_compare_matrix()
  if args.matrix == "engram_repro":
    variants = _engram_repro_matrix()
  if args.matrix == "canon_repro":
    variants = _canon_repro_matrix()
  if args.matrix == "stack_interactions":
    variants = _stack_interactions_matrix()
  if args.matrix == "canon_attn_cross":
    variants = _canon_attn_cross_matrix()
  if args.variant is not None:
    variants = [_parse_variant(str(args.variant))]

  # CKA baseline: use --cka-baseline if provided, else default vanilla
  default_baseline = Variant(width="fixed", residual="vanilla", memory="none", attn="global")
  if cfg.cka_baseline is not None:
    baseline_key = cfg.cka_baseline
    # Check if baseline is in variants list
    baseline_in_variants = any(v.key() == baseline_key for v in variants)
    if not baseline_in_variants:
      # Try to parse it as a variant spec
      try:
        baseline = _parse_variant(baseline_key)
        variants = [baseline] + variants
      except Exception:
        raise SystemExit(f"CKA baseline '{baseline_key}' not found in variants and not parseable")
    else:
      # Move baseline to front
      variants = [v for v in variants if v.key() == baseline_key] + [v for v in variants if v.key() != baseline_key]
  else:
    baseline_key = default_baseline.key()
    if bool(cfg.cka):
      if all(v.key() != baseline_key for v in variants):
        variants = [default_baseline] + variants
      else:
        variants = [default_baseline] + [v for v in variants if v.key() != baseline_key]

  # Track whether we need to return model for post-train analyses.
  # Note: slice_metrics also needs a model, so include it here.
  need_model = bool(cfg.cka) or bool(cfg.layer_ce) or bool(cfg.slice_metrics) or bool(cfg.lano_cfg_kl)

  runs: dict[str, dict] = {}
  baseline_feats: dict[str, list[torch.Tensor]] | None = None

  for v in variants:
    meta, model = _train_variant(cfg=cfg, variant=v, train=train, valid=valid, device=device, return_model=need_model)

    if need_model:
      if model is None:
        raise RuntimeError("return_model=True did not return a model")

      # Layer CE evaluation
      if bool(cfg.layer_ce):
        layer_ce_results: dict[str, dict] = {}
        for task in sorted(set(valid.tasks)):
          idx = [i for i, t in enumerate(valid.tasks) if t == task]
          if not idx:
            continue
          idx = idx[: int(cfg.layer_ce_n)]
          sub = Split(tokens=valid.tokens[idx], labels=valid.labels[idx], tasks=[task] * len(idx))
          layer_ce_results[task] = _evaluate_layer_ce(
            model=model,
            split=sub,
            batch_size=min(int(cfg.batch_size), int(sub.tokens.size(0))),
            loss_mode=cfg.loss_mode,
            device=device,
          )
        if layer_ce_results:
          meta["layer_ce_valid"] = layer_ce_results

      # Lano-cfg DP KL evaluation (PhysicsLM4-style)
      if bool(cfg.lano_cfg_kl) and "lano_cfg" in task_kwargs_by_name:
        idx = [i for i, t in enumerate(valid.tasks) if t == "lano_cfg"][: int(cfg.lano_cfg_kl_n)]
        if idx:
          sub = Split(tokens=valid.tokens[idx], labels=valid.labels[idx], tasks=["lano_cfg"] * len(idx))
          meta["lano_cfg_dp_valid"] = _evaluate_lano_cfg_kl(
            model=model,
            split=sub,
            batch_size=1,
            device=device,
            task_kwargs=task_kwargs_by_name["lano_cfg"],
          )

      # Slice metrics (difficulty curves)
      if bool(cfg.slice_metrics):
        slice_results: dict[str, dict] = {}
        for task in sorted(set(valid.tasks)):
          idx = [i for i, t in enumerate(valid.tasks) if t == task]
          if not idx:
            continue
          idx = idx[: int(cfg.slice_metrics_n)]
          sub = Split(tokens=valid.tokens[idx], labels=valid.labels[idx], tasks=[task] * len(idx))
          slice_results[task] = _evaluate_slices(
            model=model,
            split=sub,
            batch_size=min(int(cfg.batch_size), int(sub.tokens.size(0))),
            loss_mode=cfg.loss_mode,
            device=device,
            include_layer_ce=bool(cfg.layer_ce),
          )
        if slice_results:
          meta["slices_valid"] = slice_results

      # CKA evaluation
      if bool(cfg.cka):
        if baseline_feats is None:
          if v.key() != baseline_key:
            raise RuntimeError(f"CKA requires baseline '{baseline_key}' to be evaluated first")
          baseline_feats = {}
          for task in sorted(set(valid.tasks)):
            idx = [i for i, t in enumerate(valid.tasks) if t == task]
            if not idx:
              continue
            idx = idx[: int(cfg.cka_n)]
            sub = Split(tokens=valid.tokens[idx], labels=valid.labels[idx], tasks=[task] * len(idx))
            baseline_feats[task] = _collect_layer_features(
              model=model,
              split=sub,
              batch_size=min(int(cfg.batch_size), int(sub.tokens.size(0))),
              loss_mode=cfg.loss_mode,
              device=device,
            )
        else:
          cka_task: dict[str, list[list[float]]] = {}
          cka_depth_shift: dict[str, dict] = {}
          for task, base in baseline_feats.items():
            idx = [i for i, t in enumerate(valid.tasks) if t == task]
            if not idx:
              continue
            idx = idx[: int(cfg.cka_n)]
            sub = Split(tokens=valid.tokens[idx], labels=valid.labels[idx], tasks=[task] * len(idx))
            feats = _collect_layer_features(
              model=model,
              split=sub,
              batch_size=min(int(cfg.batch_size), int(sub.tokens.size(0))),
              loss_mode=cfg.loss_mode,
              device=device,
            )
            mat = _cka_matrix(baseline=base, variant=feats)
            cka_task[task] = mat
            cka_depth_shift[task] = _compute_depth_shift(mat)
          if cka_task:
            meta["cka_valid"] = cka_task
            meta["cka_depth_shift"] = cka_depth_shift

      del model
      if device.type == "cuda":
        torch.cuda.empty_cache()

    runs[v.key()] = meta

  (cfg.output / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")

  # Summary table.
  summary = {}
  logitlens_valid = {}
  cka_valid_out: dict[str, dict[str, list[list[float]]]] = {}
  cka_depth_shift_out: dict[str, dict[str, dict]] = {}
  layer_ce_valid_out: dict[str, dict[str, dict]] = {}
  slices_valid_out: dict[str, dict[str, dict]] = {}
  lano_cfg_dp_out: dict[str, dict] = {}
  for k, meta in runs.items():
    summary[k] = meta.get("final", {})
    if "logitlens_valid" in meta:
      logitlens_valid[k] = meta["logitlens_valid"]
    if "cka_valid" in meta:
      cka_valid_out[k] = meta["cka_valid"]
    if "cka_depth_shift" in meta:
      cka_depth_shift_out[k] = meta["cka_depth_shift"]
    if "layer_ce_valid" in meta:
      layer_ce_valid_out[k] = meta["layer_ce_valid"]
    if "slices_valid" in meta:
      slices_valid_out[k] = meta["slices_valid"]
    if "lano_cfg_dp_valid" in meta:
      lano_cfg_dp_out[k] = meta["lano_cfg_dp_valid"]

  out_dir = cfg.output / "analysis"
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

  if logitlens_valid:
    (out_dir / "logitlens_valid.json").write_text(json.dumps(logitlens_valid, indent=2), encoding="utf-8")

  if cka_valid_out:
    payload = {"_meta": {"baseline": baseline_key}, **cka_valid_out}
    (out_dir / "cka_valid.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

  if cka_depth_shift_out:
    payload = {"_meta": {"baseline": baseline_key}, **cka_depth_shift_out}
    (out_dir / "cka_depth_shift.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

  if layer_ce_valid_out:
    (out_dir / "layer_ce_valid.json").write_text(json.dumps(layer_ce_valid_out, indent=2), encoding="utf-8")
  if slices_valid_out:
    (out_dir / "slices_valid.json").write_text(json.dumps(slices_valid_out, indent=2), encoding="utf-8")
  if lano_cfg_dp_out:
    (out_dir / "lano_cfg_dp_valid.json").write_text(json.dumps(lano_cfg_dp_out, indent=2), encoding="utf-8")

  print(json.dumps(summary, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
