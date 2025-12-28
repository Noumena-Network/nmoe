import math

import torch
from torch import nn


def rotate_pe(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  """Apply rotary position embedding to tensor x.

  Note: cos/sin are expected to be pre-sliced to seqlen and pre-cast to x.dtype
  (done in Transformer.forward and RotaryEmbedding.__init__).
  """
  # cos/sin: [seqlen, head_dim//2] -> [seqlen, 1, head_dim//2] for broadcasting
  cos = cos[:x.size(1), :].unsqueeze(-2)
  sin = sin[:x.size(1), :].unsqueeze(-2)
  half = x.size(-1) // 2
  x1 = x[..., :half]
  x2 = x[..., half:]
  out = torch.empty_like(x)
  out[..., :half] = x1 * cos - x2 * sin
  out[..., half:] = x2 * cos + x1 * sin
  return out


class RotaryEmbedding(nn.Module):
  def __init__(
    self,
    head_dim: int,
    base: int,
    dtype: torch.dtype,
    initial_context_length: int = 4096,
    max_context_length: int = 131072,
    scaling_factor: float = 1.0,
    ntk_alpha: float = 1.0,
    ntk_beta: float = 32.0,
    device: torch.device | None = None,
  ) -> None:
    super().__init__()
    self.head_dim = head_dim
    self.base = base
    self.dtype = dtype
    self.initial_context_length = initial_context_length
    self.max_context_length = max_context_length
    self.scaling_factor = scaling_factor
    self.ntk_alpha = ntk_alpha
    self.ntk_beta = ntk_beta
    self.device = device
    cos, sin = self._compute_cos_sin(0, self.max_context_length)
    # Register as buffers so they move with model.cuda() and pre-cast to target dtype
    self.register_buffer('cos', cos.to(dtype), persistent=False)
    self.register_buffer('sin', sin.to(dtype), persistent=False)

  def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
    freq = self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device) / self.head_dim)
    if self.scaling_factor > 1.0:
      concentration = (0.1 * math.log(self.scaling_factor) + 1.0)
      d_half = self.head_dim / 2
      low  = (d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base))
      high = (d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base))
      assert 0 < low < high < d_half - 1
      interpolation = 1.0 / (self.scaling_factor * freq)
      extrapolation = 1.0 / freq
      ramp = (torch.arange(d_half, dtype=torch.float32, device=freq.device) - low) / (high - low)
      mask = 1 - ramp.clamp(0, 1)
      inv_freq = interpolation * (1 - mask) + extrapolation * mask
    else:
      concentration = 1.0
      inv_freq = 1.0 / freq
    return concentration, inv_freq

  def _compute_cos_sin(self, start: int, num_tokens: int):
    concentration, inv_freq = self._compute_concentration_and_inv_freq()
    t = torch.arange(start, start + num_tokens, dtype=torch.float32, device=self.device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos() * concentration
    sin = freqs.sin() * concentration
    return cos, sin
