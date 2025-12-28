import math

import torch

from nmoe.loss import chunked_cross_entropy


def _dense_ce(hidden, weight, target, ignore_index):
  logits = hidden @ weight.t()
  return torch.nn.functional.cross_entropy(logits, target, ignore_index=ignore_index)


@torch.no_grad()
def _supports_bf16(device):
  if device.type == "cuda":
    return True
  # CPU bf16 matmul is backend-dependent; keep tests robust.
  return False


def test_chunked_ce_forward_parity_fp32():
  torch.manual_seed(0)
  device = torch.device("cpu")
  N, D, V = 32, 16, 23
  ignore_index = -100

  hidden = torch.randn((N, D), device=device, dtype=torch.float32, requires_grad=True)
  weight = torch.randn((V, D), device=device, dtype=torch.float32, requires_grad=True)
  target = torch.randint(0, V, (N,), device=device, dtype=torch.int64)
  target[::7] = ignore_index

  dense = _dense_ce(hidden, weight, target, ignore_index)
  chunked = chunked_cross_entropy(hidden, weight, target, ignore_index=ignore_index, chunk_tokens=8, chunk_vocab=5)

  assert torch.allclose(dense, chunked, rtol=1e-6, atol=1e-6)


def test_chunked_ce_grad_parity_fp32():
  torch.manual_seed(0)
  device = torch.device("cpu")
  N, D, V = 32, 16, 23
  ignore_index = -100

  hidden0 = torch.randn((N, D), device=device, dtype=torch.float32, requires_grad=True)
  weight0 = torch.randn((V, D), device=device, dtype=torch.float32, requires_grad=True)
  target = torch.randint(0, V, (N,), device=device, dtype=torch.int64)
  target[::5] = ignore_index

  hidden1 = hidden0.detach().clone().requires_grad_(True)
  weight1 = weight0.detach().clone().requires_grad_(True)

  dense = _dense_ce(hidden0, weight0, target, ignore_index)
  chunked = chunked_cross_entropy(hidden1, weight1, target, ignore_index=ignore_index, chunk_tokens=8, chunk_vocab=7)

  dense.backward()
  chunked.backward()

  assert torch.allclose(hidden0.grad, hidden1.grad, rtol=1e-6, atol=1e-6)
  assert torch.allclose(weight0.grad, weight1.grad, rtol=1e-6, atol=1e-6)


def test_chunked_ce_bf16_tolerance_if_supported():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if not _supports_bf16(device):
    return

  torch.manual_seed(0)
  N, D, V = 64, 32, 101
  ignore_index = -100

  hidden0 = torch.randn((N, D), device=device, dtype=torch.bfloat16, requires_grad=True)
  weight0 = torch.randn((V, D), device=device, dtype=torch.bfloat16, requires_grad=True)
  target = torch.randint(0, V, (N,), device=device, dtype=torch.int64)
  target[::9] = ignore_index

  hidden1 = hidden0.detach().clone().requires_grad_(True)
  weight1 = weight0.detach().clone().requires_grad_(True)

  dense = _dense_ce(hidden0, weight0, target, ignore_index)
  chunked = chunked_cross_entropy(hidden1, weight1, target, ignore_index=ignore_index, chunk_tokens=16, chunk_vocab=13)

  dense.backward()
  chunked.backward()

  # Practical bf16 tolerances. (CUDA kernels differ; exp/log reductions also vary slightly.)
  assert torch.allclose(dense.float(), chunked.float(), rtol=5e-3, atol=5e-3)
  assert torch.allclose(hidden0.grad.float(), hidden1.grad.float(), rtol=7e-3, atol=7e-3)
  assert torch.allclose(weight0.grad.float(), weight1.grad.float(), rtol=7e-3, atol=7e-3)


def test_chunked_ce_all_ignored_is_zero_loss_and_zero_grads():
  torch.manual_seed(0)
  device = torch.device("cpu")
  N, D, V = 16, 8, 11
  ignore_index = -100

  hidden = torch.randn((N, D), device=device, dtype=torch.float32, requires_grad=True)
  weight = torch.randn((V, D), device=device, dtype=torch.float32, requires_grad=True)
  target = torch.full((N,), ignore_index, device=device, dtype=torch.int64)

  loss = chunked_cross_entropy(hidden, weight, target, ignore_index=ignore_index, chunk_tokens=8, chunk_vocab=4)
  loss.backward()

  assert math.isfinite(float(loss.detach()))
  assert float(loss.detach()) == 0.0
  assert torch.equal(hidden.grad, torch.zeros_like(hidden))
  assert torch.equal(weight.grad, torch.zeros_like(weight))
