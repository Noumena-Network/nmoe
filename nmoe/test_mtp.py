"""Unit tests for Multi-Token Prediction (MTP) module."""
import torch
import torch.nn.functional as F


def test_mtp_disabled():
  """MTP should be None when mtp_depth=0."""
  from nmoe.config import Config
  from nmoe.model import Transformer

  cfg = Config(
    dim=64,
    inter_dim=128,
    n_layers=2,
    n_heads=4,
    n_dense_layers=2,
    mtp_depth=0,
    batch_size=2,
    seq_len=32,
  )

  # Mock distributed
  import torch.distributed as dist
  if not dist.is_initialized():
    import os
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')

  model = Transformer(cfg)
  assert model.mtp is None, "MTP should be None when mtp_depth=0"
  print("✓ test_mtp_disabled passed")


def test_mtp_enabled():
  """MTP should be created when mtp_depth>0."""
  from nmoe.config import Config
  from nmoe.model import Transformer

  cfg = Config(
    dim=64,
    inter_dim=128,
    n_layers=2,
    n_heads=4,
    n_dense_layers=2,
    mtp_depth=1,
    batch_size=2,
    seq_len=32,
  )

  model = Transformer(cfg)
  assert model.mtp is not None, "MTP should be created when mtp_depth>0"
  assert model.mtp.depth == 1
  assert len(model.mtp.blocks) == 1
  print("✓ test_mtp_enabled passed")


def test_mtp_last_loss_cleared():
  """last_loss should be None when targets not provided or not training."""
  from nmoe.config import Config
  from nmoe.model import Transformer

  cfg = Config(
    dim=64,
    inter_dim=128,
    n_layers=2,
    n_heads=4,
    n_dense_layers=2,
    mtp_depth=1,
    batch_size=2,
    seq_len=32,
  )

  model = Transformer(cfg).cuda()
  model.init_weights()

  tokens = torch.randint(0, cfg.vocab_size, (2, 32), device='cuda')
  targets = torch.randint(0, cfg.vocab_size, (2, 32), device='cuda')

  # Training mode, with targets - should compute loss
  model.train()
  _ = model(tokens, targets=targets)
  assert model.mtp.last_loss is not None, "last_loss should be set when targets provided"

  # Training mode, without targets - should clear loss
  _ = model(tokens)
  assert model.mtp.last_loss is None, "last_loss should be None when targets not provided"

  # Eval mode - should clear loss
  model.eval()
  _ = model(tokens, targets=targets)
  assert model.mtp.last_loss is None, "last_loss should be None in eval mode"

  print("✓ test_mtp_last_loss_cleared passed")


def test_mtp_ignore_index():
  """MTP should use ignore_index matching main loss (eos_token_id)."""
  from nmoe.config import Config
  from nmoe.model import Transformer

  cfg = Config(
    dim=64,
    inter_dim=128,
    n_layers=2,
    n_heads=4,
    n_dense_layers=2,
    mtp_depth=1,
    batch_size=2,
    seq_len=32,
    eos_token_id=199999,
  )

  model = Transformer(cfg).cuda()
  model.init_weights()
  model.train()

  # Create targets with some padding tokens
  tokens = torch.randint(0, 1000, (2, 32), device='cuda')
  targets = torch.randint(0, 1000, (2, 32), device='cuda')
  targets[:, -5:] = cfg.eos_token_id  # Last 5 positions are padding

  # Forward pass
  _ = model(tokens, targets=targets)
  loss_with_padding = model.mtp.last_loss.item()

  # Verify ignore_index is set correctly
  assert model.mtp.ignore_index == cfg.eos_token_id
  print(f"✓ test_mtp_ignore_index passed (loss={loss_with_padding:.4f})")


def test_mtp_gradient_flow():
  """Gradients should flow through all MTP depths."""
  from nmoe.config import Config
  from nmoe.model import Transformer

  cfg = Config(
    dim=64,
    inter_dim=128,
    n_layers=2,
    n_heads=4,
    n_dense_layers=2,
    mtp_depth=2,
    batch_size=2,
    seq_len=32,
  )

  model = Transformer(cfg).cuda()
  model.init_weights()
  model.train()

  tokens = torch.randint(0, cfg.vocab_size, (2, 32), device='cuda')
  targets = torch.randint(0, cfg.vocab_size, (2, 32), device='cuda')

  # Forward pass
  logits = model(tokens, targets=targets)
  mtp_loss = model.mtp.last_loss

  # Backward
  mtp_loss.backward()

  # Check gradients exist for MTP blocks
  for k, block in enumerate(model.mtp.blocks):
    assert block.proj.weight.grad is not None, f"MTP block {k} proj has no gradient"
    assert block.proj.weight.grad.abs().sum() > 0, f"MTP block {k} proj gradient is zero"

  print("✓ test_mtp_gradient_flow passed")


def test_mtp_shapes():
  """Verify MTP internal shapes are correct."""
  from nmoe.config import Config
  from nmoe.model import Transformer

  cfg = Config(
    dim=64,
    inter_dim=128,
    n_layers=2,
    n_heads=4,
    n_dense_layers=2,
    mtp_depth=2,
    batch_size=2,
    seq_len=32,
  )

  model = Transformer(cfg).cuda()
  model.init_weights()
  model.train()

  # Verify block shapes
  for k, block in enumerate(model.mtp.blocks):
    # Projection: 2*dim -> dim
    assert block.proj.weight.shape == (cfg.dim, cfg.dim * 2)
    # MLP shapes
    assert block.ffn.w1.weight.shape == (cfg.inter_dim, cfg.dim)
    assert block.ffn.w2.weight.shape == (cfg.dim, cfg.inter_dim)

  print("✓ test_mtp_shapes passed")


if __name__ == '__main__':
  test_mtp_disabled()
  test_mtp_enabled()
  test_mtp_last_loss_cleared()
  test_mtp_ignore_index()
  test_mtp_gradient_flow()
  test_mtp_shapes()
  print("\n✅ All MTP tests passed!")
