"""Chunked linear cross-entropy for large vocab without materializing logits.

This implements a memory-bounded `linear -> cross_entropy` via:
  - Token chunking (for locality)
  - Vocab blocking inside each token chunk (to avoid [N, V] logits/probs)

The algorithm is a 2-pass streaming softmax:
  1) Over vocab blocks: compute per-token `logsumexp` and gather target logits.
  2) Over vocab blocks: recompute logits, form `p = exp(logits - logsumexp)`, and
     accumulate grads for hidden and weight (subtracting 1 at target positions).

No fp32 allocation of shape (chunk_tokens, vocab_size) is ever created; only
per-vocab-block temporaries exist.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


_DEFAULT_CHUNK_TOKENS = 2048
_DEFAULT_CHUNK_VOCAB = 4096


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise ValueError(msg)


def _streaming_logsumexp_and_target(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int,
    chunk_vocab: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return (logsumexp, target_logit, valid_mask) for a token chunk.

  hidden: [T, D] (bf16/fp16/fp32)
  weight: [V, D] (same dtype as hidden recommended)
  target: [T] (int64)
  """
  T, _ = hidden.shape
  V = weight.shape[0]
  device = hidden.device

  valid = target != ignore_index
  safe_target = torch.where(valid, target, torch.zeros_like(target))
  if safe_target.device.type == "cpu":
    _require(int(safe_target.min()) >= 0 and int(safe_target.max()) < V, "target out of range for vocab size")
  m = torch.full((T,), float("-inf"), device=device, dtype=torch.float32)
  s = torch.zeros((T,), device=device, dtype=torch.float32)
  hidden_f = hidden.float()
  tlogit = (hidden_f * weight[safe_target].float()).sum(dim=-1)
  tlogit = tlogit * valid.to(torch.float32)

  for v0 in range(0, V, chunk_vocab):
    v1 = min(v0 + chunk_vocab, V)
    w_block = weight[v0:v1]  # [Bv, D]
    logits_f = (hidden @ w_block.t()).float()  # [T, Bv] fp32 block

    bmax = logits_f.max(dim=-1).values  # [T]
    bsum = torch.exp(logits_f - bmax[:, None]).sum(dim=-1)  # [T]

    m_new = torch.maximum(m, bmax)
    s = s * torch.exp(m - m_new) + bsum * torch.exp(bmax - m_new)
    m = m_new

  logz = m + torch.log(s)
  return logz, tlogit, valid


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
  @staticmethod
  def forward(  # type: ignore[override]
      ctx,
      hidden: torch.Tensor,
      weight: torch.Tensor,
      target: torch.Tensor,
      ignore_index: int,
      chunk_tokens: int,
      chunk_vocab: int,
  ) -> torch.Tensor:
    hidden = hidden.contiguous()
    weight = weight.contiguous()
    target = target.contiguous()
    _require(hidden.dim() == 2, f"hidden must be [N, D] (got {tuple(hidden.shape)})")
    _require(weight.dim() == 2, f"weight must be [V, D] (got {tuple(weight.shape)})")
    _require(target.dim() == 1, f"target must be [N] (got {tuple(target.shape)})")
    _require(hidden.shape[0] == target.shape[0], "hidden/target token count mismatch")
    _require(hidden.shape[1] == weight.shape[1], "hidden/weight dim mismatch")
    _require(target.dtype == torch.int64, f"target must be int64 (got {target.dtype})")
    _require(chunk_tokens > 0, "chunk_tokens must be > 0")
    _require(chunk_vocab > 0, "chunk_vocab must be > 0")

    device = hidden.device
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_count = torch.zeros((), device=device, dtype=torch.float32)

    N = hidden.shape[0]
    for t0 in range(0, N, chunk_tokens):
      t1 = min(t0 + chunk_tokens, N)
      logz, tlogit, valid = _streaming_logsumexp_and_target(
        hidden[t0:t1],
        weight,
        target[t0:t1],
        ignore_index=ignore_index,
        chunk_vocab=chunk_vocab,
      )
      loss_vec = (logz - tlogit)  # [T]
      total_loss = total_loss + loss_vec[valid].sum()
      total_count = total_count + valid.sum(dtype=torch.float32)

    ctx.ignore_index = int(ignore_index)
    ctx.chunk_tokens = int(chunk_tokens)
    ctx.chunk_vocab = int(chunk_vocab)
    ctx.save_for_backward(hidden, weight, target, total_count)
    return total_loss / total_count.clamp_min(1.0)

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
    hidden, weight, target, total_count = ctx.saved_tensors
    ignore_index = int(ctx.ignore_index)
    chunk_tokens = int(ctx.chunk_tokens)
    chunk_vocab = int(ctx.chunk_vocab)

    scale = (grad_output / total_count.clamp_min(1.0)).to(torch.float32)
    N, D = hidden.shape
    V = weight.shape[0]

    dx = torch.zeros_like(hidden)
    dW = torch.zeros_like(weight)

    for t0 in range(0, N, chunk_tokens):
      t1 = min(t0 + chunk_tokens, N)
      h = hidden[t0:t1]
      t = target[t0:t1]
      logz, _tlogit, valid = _streaming_logsumexp_and_target(
        h, weight, t, ignore_index=ignore_index, chunk_vocab=chunk_vocab
      )
      h_f = h.float()

      # Accumulate dX for this token chunk in fp32, then cast once.
      dx_chunk = torch.zeros((t1 - t0, D), device=h.device, dtype=torch.float32)
      valid_f = valid.to(torch.float32)

      for v0 in range(0, V, chunk_vocab):
        v1 = min(v0 + chunk_vocab, V)
        w_block = weight[v0:v1]
        logits_f = (h @ w_block.t()).float()
        probs = torch.exp(logits_f - logz[:, None])  # [T, Bv] fp32
        probs = probs * valid_f[:, None]

        offset = t - v0
        in_block = valid & (offset >= 0) & (offset < (v1 - v0))
        idx = torch.nonzero(in_block, as_tuple=False).squeeze(1)
        probs[idx, offset[idx].to(torch.int64)] -= 1.0

        dx_chunk.add_(probs @ w_block.float())
        dW_block = probs.t().matmul(h_f)
        dW[v0:v1].add_(dW_block.to(dW.dtype))

      dx[t0:t1] = (dx_chunk * scale).to(dx.dtype)

    dW.mul_(scale.to(dW.dtype))
    return dx, dW, None, None, None, None


def chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = -100,
    chunk_tokens: int = _DEFAULT_CHUNK_TOKENS,
    chunk_vocab: int = _DEFAULT_CHUNK_VOCAB,
) -> torch.Tensor:
  """Cross-entropy loss for large vocab without materializing `[N, V]` logits.

  Args:
    hidden: `[B, S, D]` or `[N, D]` hidden states. Typically `model(..., return_hidden=True)`.
    weight: `[V, D]` output weight (e.g. `model.lm_head.weight`).
    target: `[B, S]` or `[N]` target token ids (int64).
    ignore_index: Token id to ignore in the mean reduction.
    chunk_tokens: Number of tokens per chunk.
    chunk_vocab: Vocab block size.
  """
  if hidden.dim() == 3:
    B, S, D = hidden.shape
    hidden = hidden.view(B * S, D)
  if target.dim() == 2:
    target = target.reshape(-1)

  return _ChunkedLinearCrossEntropy.apply(
    hidden,
    weight,
    target,
    int(ignore_index),
    int(chunk_tokens),
    int(chunk_vocab),
  )


@dataclass(frozen=True)
class _MemcheckArgs:
  dim: int = 4096
  vocab: int = 201_088
  tokens: int = 8192
  dtype: str = "bf16"
  chunk_tokens: int = _DEFAULT_CHUNK_TOKENS
  chunk_vocab: int = _DEFAULT_CHUNK_VOCAB


def _memcheck(args: _MemcheckArgs) -> None:
  if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for memcheck.")
  device = torch.device("cuda")
  dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

  def run_chunked() -> int:
    torch.manual_seed(0)
    hidden = torch.randn((args.tokens, args.dim), device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn((args.vocab, args.dim), device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(0, args.vocab, (args.tokens,), device=device, dtype=torch.int64)
    torch.cuda.reset_peak_memory_stats()
    loss = chunked_cross_entropy(
      hidden,
      weight,
      target,
      ignore_index=-100,
      chunk_tokens=args.chunk_tokens,
      chunk_vocab=args.chunk_vocab,
    )
    loss.backward()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())

  def run_dense() -> int:
    torch.manual_seed(0)
    hidden = torch.randn((args.tokens, args.dim), device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn((args.vocab, args.dim), device=device, dtype=dtype, requires_grad=True)
    target = torch.randint(0, args.vocab, (args.tokens,), device=device, dtype=torch.int64)
    torch.cuda.reset_peak_memory_stats()
    logits = hidden @ weight.t()
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())

  # Separate runs to avoid comparing against leftover allocations from the first path.
  torch.cuda.empty_cache()
  peak_chunked = run_chunked()
  torch.cuda.empty_cache()
  peak_dense = run_dense()

  print(f"chunked peak alloc: {peak_chunked / 1e9:.2f} GB")
  print(f"dense   peak alloc: {peak_dense / 1e9:.2f} GB")


def _main() -> None:
  p = argparse.ArgumentParser(description="nmoe.loss memcheck (CUDA-only)")
  p.add_argument("--dim", type=int, default=_MemcheckArgs.dim)
  p.add_argument("--vocab", type=int, default=_MemcheckArgs.vocab)
  p.add_argument("--tokens", type=int, default=_MemcheckArgs.tokens)
  p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=_MemcheckArgs.dtype)
  p.add_argument("--chunk_tokens", type=int, default=_MemcheckArgs.chunk_tokens)
  p.add_argument("--chunk_vocab", type=int, default=_MemcheckArgs.chunk_vocab)
  ns = p.parse_args()
  _memcheck(_MemcheckArgs(**vars(ns)))


if __name__ == "__main__":
  _main()
