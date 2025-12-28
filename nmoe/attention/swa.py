import torch
import torch.nn as nn
from torch.profiler import record_function

from nmoe.attention.rope import rotate_pe
from nmoe.config import Config
import nmoe.triton.swa as swa_k


class SWA(nn.Module):
  """Sliding-Window Attention wrapper around Triton kernel (nmoe.triton.swa).

  Fused QKV + OUT + learned sinks. Applies RoPE to q,k and calls the kernel with
  layout expected by SWA: q[B,T,Hkv,G,D], k[B,T,Hkv,D], v[B,T,Hkv,D].
  """

  def __init__(self, config: Config):
    super().__init__()
    self.dim = config.dim
    self.n_heads = config.n_heads
    self.head_dim = int(config.v_head_dim)
    if self.head_dim not in (16, 32, 64, 128, 256):
      raise RuntimeError(f"SWA requires head_dim in {{16,32,64,128,256}}, got {self.head_dim}.")

    self.rope_dim = int(getattr(config, "qk_rope_head_dim", 0))
    if self.rope_dim < 0 or self.rope_dim > self.head_dim:
      raise RuntimeError(
        f"SWA requires 0 <= qk_rope_head_dim <= v_head_dim. Got qk_rope_head_dim={self.rope_dim}, v_head_dim={self.head_dim}."
      )
    if self.rope_dim % 2 != 0:
      raise RuntimeError(f"SWA requires rope_dim to be even, got {self.rope_dim}.")
    self.nope_dim = self.head_dim - self.rope_dim

    swa_opts = getattr(config, "attn_swa", {}) or {}
    self.n_kv_heads = int(swa_opts.get("kv_heads", self.n_heads))
    if self.n_heads % self.n_kv_heads != 0:
      raise RuntimeError(f"SWA requires H % KV == 0 (H={self.n_heads}, KV={self.n_kv_heads}).")
    self.repeat_kv = self.n_heads // self.n_kv_heads
    out_feats = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
    self.qkv = nn.Linear(self.dim, out_feats, bias=True, dtype=torch.bfloat16)
    self.out = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=True, dtype=torch.bfloat16)
    self.sinks = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.bfloat16))
    self.window = 0
    self.softmax_scale = self.head_dim ** -0.5
    # Cache start_q to avoid per-forward allocation
    self.register_buffer('start_q', torch.zeros(1, dtype=torch.int32), persistent=False)

  def init_weights(self, init_std: float = 0.02):
    nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=0.02)
    if self.qkv.bias is not None:
      nn.init.zeros_(self.qkv.bias)
    nn.init.trunc_normal_(self.out.weight, mean=0.0, std=init_std)
    if self.out.bias is not None:
      nn.init.zeros_(self.out.bias)
    with torch.no_grad():
      self.sinks.zero_()

  @record_function("attn")
  def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, T, _ = x.size()
    qkv = self.qkv(x)
    H, KV, D = self.n_heads, self.n_kv_heads, self.head_dim
    q, k, v = torch.split(qkv, [H * D, KV * D, KV * D], dim=-1)
    # split() returns views that share storage with qkv; make q/k materialize
    # their own storage before any in-place slice writes (autograd restriction).
    q = q.view(B, T, H, D).contiguous()
    k = k.view(B, T, KV, D).contiguous()
    v = v.view(B, T, KV, D)

    if self.rope_dim:
      q_rope = rotate_pe(q[..., self.nope_dim:], cos, sin)
      q[..., self.nope_dim:].copy_(q_rope)
      k_rope = rotate_pe(k[..., self.nope_dim:], cos, sin)
      k[..., self.nope_dim:].copy_(k_rope)

    G = self.repeat_kv
    assert H == KV * G, f"Heads must satisfy H = KV*G (got H={H}, KV={KV}, G={G})"
    q = q.view(B, T, KV, G, D)

    # Use cached start_q buffer (already on correct device via register_buffer)
    bandwidth = int(self.window) if self.window and self.window > 0 else 0

    with record_function("attn.kernel[swa]"):
      o = swa_k.attention(q, k, v, self.sinks, self.softmax_scale, bandwidth, self.start_q)
    return self.out(o)
