"""HYDRA-vNext training entrypoint.

Fix-forward only: this file implements Phase A/B/C from HYDRA_TRAINING.md and
removes the legacy single-source oracle training path.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hydra_dataset import (
  DOMAIN_MAP,
  EAIParquetSource,
  FineWebEduScore2Source,
  HydraPair,
  HydraSample,
  OracleLabelSource,
)
from .model import Transformer, pool_hidden


EOS_TOKEN_ID = 199999


def _setup_distributed() -> tuple[torch.device, int, int]:
  """Initialize distributed training and return (device, rank, world_size)."""
  local_rank = int(os.environ.get("LOCAL_RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))
  rank = int(os.environ.get("RANK", 0))

  if world_size > 1:
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

  device = torch.device(f"cuda:{local_rank}")
  return device, rank, world_size


def _tokenize_batch(
  enc: tiktoken.Encoding,
  texts: list[str],
  *,
  max_ctx: int = 4096,
  pad_id: int = EOS_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  ids = [enc.encode_ordinary(t)[:max_ctx] for t in texts]
  max_len = max((len(x) for x in ids), default=0)
  bsz = len(ids)
  input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long)
  mask = torch.zeros((bsz, max_len), dtype=torch.bool)
  for i, row in enumerate(ids):
    n = len(row)
    if n == 0:
      continue
    input_ids[i, :n] = torch.tensor(row, dtype=torch.long)
    mask[i, :n] = True
  positions = torch.arange(max_len, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
  return input_ids, positions, mask


def _pad_token_batch(
  token_rows: list[torch.Tensor],
  *,
  pad_id: int = EOS_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  max_len = max((int(t.numel()) for t in token_rows), default=0)
  bsz = len(token_rows)
  input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long)
  mask = torch.zeros((bsz, max_len), dtype=torch.bool)
  for i, t in enumerate(token_rows):
    n = int(t.numel())
    if n == 0:
      continue
    input_ids[i, :n] = t[:n].to(dtype=torch.long)
    mask[i, :n] = True
  positions = torch.arange(max_len, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
  return input_ids, positions, mask


class ProbeHead(nn.Module):
  def __init__(
    self,
    hidden_dim: int,
    *,
    num_domains: int = 4,
    num_artifact_classes: int = 8,
    num_missing_classes: int = 8,
  ):
    super().__init__()
    self.gate = nn.Linear(hidden_dim, 1)
    self.domain = nn.Linear(hidden_dim, num_domains)

    self.fasttext_dclm = nn.Linear(hidden_dim, 1)
    self.fasttext_edu = nn.Linear(hidden_dim, 1)
    self.fasttext_code = nn.Linear(hidden_dim, 1)
    self.fasttext_math = nn.Linear(hidden_dim, 1)
    self.lang_score = nn.Linear(hidden_dim, 1)

    self.extraction_artifacts = nn.Linear(hidden_dim, num_artifact_classes)
    self.missing_content = nn.Linear(hidden_dim, num_missing_classes)

  def forward(self, h18_pooled: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
      "gate": self.gate(h18_pooled).squeeze(-1),
      "domain": self.domain(h18_pooled),
      "fasttext_dclm": self.fasttext_dclm(h18_pooled).squeeze(-1),
      "fasttext_edu": self.fasttext_edu(h18_pooled).squeeze(-1),
      "fasttext_code": self.fasttext_code(h18_pooled).squeeze(-1),
      "fasttext_math": self.fasttext_math(h18_pooled).squeeze(-1),
      "lang_score": self.lang_score(h18_pooled).squeeze(-1),
      "extraction_artifacts": self.extraction_artifacts(h18_pooled),
      "missing_content": self.missing_content(h18_pooled),
    }


class JudgeEncoder(nn.Module):
  def __init__(self, hidden_dim: int, *, mid_dim: int = 512, nhead: int = 8):
    super().__init__()
    self.q_token = nn.Parameter(torch.empty(hidden_dim))
    nn.init.normal_(self.q_token, mean=0.0, std=0.02)
    self.projector = nn.Linear(hidden_dim, mid_dim)
    self.encoder = nn.TransformerEncoderLayer(d_model=mid_dim, nhead=nhead, batch_first=True)
    self.rubric_head = nn.Linear(mid_dim, 5)

  def forward(self, h24_seq: torch.Tensor) -> torch.Tensor:
    bsz, _, h = h24_seq.shape
    q = self.q_token.unsqueeze(0).unsqueeze(1).expand(bsz, 1, h)
    seq = torch.cat([q, h24_seq], dim=1)
    z = self.projector(seq)
    z = self.encoder(z)
    return self.rubric_head(z[:, 0, :])

  @staticmethod
  def aggregate(rubric: torch.Tensor) -> torch.Tensor:
    r = torch.clamp(rubric / 4.0, 0.0, 1.0)
    return r.mean(dim=-1)

  def export(self) -> dict[str, dict[str, torch.Tensor]]:
    return {
      "projector": self.projector.state_dict(),
      "encoder": self.encoder.state_dict() | {"q_token": self.q_token.detach().cpu()},
      "rubric_head": self.rubric_head.state_dict(),
    }


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, *, weight: float) -> torch.Tensor:
  mask = torch.isfinite(target)
  if not mask.any():
    return pred.new_tensor(0.0)
  return float(weight) * F.mse_loss(pred[mask], target[mask])


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, *, weight: float) -> torch.Tensor:
  mask = target >= 0
  if not mask.any():
    return logits.new_tensor(0.0)
  return float(weight) * F.cross_entropy(logits[mask], target[mask])


@dataclass(frozen=True)
class PhaseAConfig:
  checkpoint: str
  out_dir: str
  max_steps: int = 50_000
  batch_size: int = 32
  lr: float = 1e-3
  max_ctx: int = 4096

  eai_code_glob: str = ""
  eai_stem_glob: str = ""
  eai_math_glob: str = ""
  eai_med_glob: str = ""
  fw_score2_glob: str = ""


@dataclass(frozen=True)
class PhaseBConfig:
  checkpoint: str
  out_dir: str
  oracle_labels_jsonl: str
  oracle_data_root: str

  max_steps: int = 10_000
  batch_size: int = 64
  lr: float = 1e-4
  max_ctx: int = 4096

  oracle_only_prob: float = 0.6
  fw_pairs_per_step: int = 32
  fw_score2_glob: str = ""


@dataclass(frozen=True)
class PhaseCConfig:
  checkpoint: str
  out_dir: str
  probe_path: str
  judge_export_path: str
  distill_source_glob: str

  max_steps: int = 100_000
  batch_size: int = 64
  lr: float = 1e-4
  max_ctx: int = 4096


def _cycle(it_fn) -> Iterator[HydraSample]:
  it = iter(it_fn())
  while True:
    try:
      yield next(it)
    except StopIteration:
      it = iter(it_fn())


class FineWebPairSampler:
  def __init__(
    self,
    src: FineWebEduScore2Source,
    *,
    min_gap: int = 2,
    relaxed_gap: int = 1,
    max_relaxed_frac: float = 0.2,
    reservoir_per_score: int = 256,
  ):
    self._it = iter(src)
    self._min_gap = int(min_gap)
    self._relaxed_gap = int(relaxed_gap)
    self._max_relaxed_frac = float(max_relaxed_frac)
    self._reservoir_cap = int(reservoir_per_score)

    # bucket -> score -> samples
    self._buf: list[list[list[HydraSample]]] = [
      [[ ] for _ in range(6)] for _ in range(len(FineWebEduScore2Source.LENGTH_BUCKETS))
    ]
    self._relaxed = 0
    self._total = 0

  def _add(self, s: HydraSample) -> None:
    if s.fw_int_score is None:
      return
    score = int(s.fw_int_score)
    if score < 0 or score > 5:
      return
    b = FineWebEduScore2Source.length_bucket(s.token_count)
    bucket = self._buf[b][score]
    if len(bucket) < self._reservoir_cap:
      bucket.append(s)
      return
    # reservoir replace
    j = random.randrange(len(bucket) + 1)
    if j < len(bucket):
      bucket[j] = s

  def _fill(self, min_per_score: int = 8) -> None:
    need = True
    while need:
      need = False
      for b in range(len(self._buf)):
        for score in range(6):
          if len(self._buf[b][score]) < min_per_score:
            need = True
            break
        if need:
          break
      if not need:
        return
      s = next(self._it)
      self._add(s)

  def next_pairs(self, n: int) -> list[HydraPair]:
    self._fill()
    out: list[HydraPair] = []
    buckets = len(self._buf)
    per_bucket = max(1, int(n // buckets))
    remainder = int(n - per_bucket * buckets)

    for b in range(buckets):
      want = per_bucket + (1 if b < remainder else 0)
      for _ in range(want):
        out.append(self._sample_from_bucket(b))
    random.shuffle(out)
    return out

  def _sample_from_bucket(self, b: int) -> HydraPair:
    # Prefer hard gaps; allow relaxed under global cap.
    for _ in range(64):
      score_a = random.randint(2, 5)
      score_b = random.randint(0, score_a - self._min_gap)
      a_bucket = self._buf[b][score_a]
      b_bucket = self._buf[b][score_b]
      if a_bucket and b_bucket:
        self._total += 1
        return HydraPair(a=random.choice(a_bucket), b=random.choice(b_bucket))

    allow_relaxed = self._total == 0 or (self._relaxed / max(self._total, 1)) < self._max_relaxed_frac
    if allow_relaxed:
      for _ in range(64):
        score_a = random.randint(1, 5)
        score_b = random.randint(0, score_a - self._relaxed_gap)
        a_bucket = self._buf[b][score_a]
        b_bucket = self._buf[b][score_b]
        if a_bucket and b_bucket:
          self._total += 1
          self._relaxed += 1
          return HydraPair(a=random.choice(a_bucket), b=random.choice(b_bucket))

    # Fallback: any two different scores.
    scores = [s for s in range(6) if self._buf[b][s]]
    if len(scores) < 2:
      # keep filling until we can.
      self._fill(min_per_score=1)
      return self._sample_from_bucket(b)
    scores.sort()
    score_a, score_b = scores[-1], scores[0]
    self._total += 1
    return HydraPair(a=random.choice(self._buf[b][score_a]), b=random.choice(self._buf[b][score_b]))


def train_phase_a(cfg: PhaseAConfig) -> None:
  device, rank, world_size = _setup_distributed()
  enc = tiktoken.get_encoding("o200k_harmony")

  model = Transformer.from_checkpoint(cfg.checkpoint, device=device)
  for p in model.parameters():
    p.requires_grad_(False)
  model.eval()

  probe = ProbeHead(model.config.hidden_size).to(device)
  probe.train()
  opt = torch.optim.AdamW(probe.parameters(), lr=cfg.lr)

  def eai_sources() -> list[Iterator[HydraSample]]:
    out: list[Iterator[HydraSample]] = []
    if cfg.eai_code_glob:
      out.append(_cycle(lambda: EAIParquetSource(cfg.eai_code_glob, domain="code", max_ctx=cfg.max_ctx)))
    if cfg.eai_stem_glob:
      out.append(_cycle(lambda: EAIParquetSource(cfg.eai_stem_glob, domain="science", max_ctx=cfg.max_ctx)))
    if cfg.eai_math_glob:
      out.append(_cycle(lambda: EAIParquetSource(cfg.eai_math_glob, domain="math", max_ctx=cfg.max_ctx)))
    if cfg.eai_med_glob:
      out.append(_cycle(lambda: EAIParquetSource(cfg.eai_med_glob, domain="science", max_ctx=cfg.max_ctx)))
    return out

  eai_iters = eai_sources()
  fw_iter = _cycle(lambda: FineWebEduScore2Source(cfg.fw_score2_glob, max_ctx=cfg.max_ctx)) if cfg.fw_score2_glob else None
  if not eai_iters and fw_iter is None:
    raise ValueError("Phase A requires at least one source (EAI or FW score2)")

  log_every = int(os.environ.get("HYDRA_LOG_EVERY", "50"))
  early_stop_loss = float(os.environ.get("HYDRA_EARLY_STOP_LOSS", "5.0"))
  early_stop_patience = int(os.environ.get("HYDRA_EARLY_STOP_PATIENCE", "1000"))
  loss_history: deque[float] = deque(maxlen=early_stop_patience)
  eai_i = 0

  for step in range(1, cfg.max_steps + 1):
    t0 = time.perf_counter()
    batch: list[HydraSample] = []
    for _ in range(cfg.batch_size):
      if fw_iter is not None and random.random() >= 0.7:
        batch.append(next(fw_iter))
      elif eai_iters:
        batch.append(next(eai_iters[eai_i % len(eai_iters)]))
        eai_i += 1
      else:
        batch.append(next(fw_iter))  # type: ignore[arg-type]

    input_ids, positions, mask = _tokenize_batch(enc, [s.text for s in batch], max_ctx=cfg.max_ctx)
    input_ids = input_ids.to(device)
    positions = positions.to(device)
    mask = mask.to(device)

    with torch.no_grad():
      with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        _, h = model(
          input_ids,
          positions,
          return_hidden_states=True,
          up_to_layer=18,
          no_logits=True,
        )
      h18 = h.get(18)
      if h18 is None:
        raise RuntimeError("layer 18 hidden state not captured")
      pooled = pool_hidden(h18.float(), mask=mask)

    preds = probe(pooled)

    domain = torch.tensor([DOMAIN_MAP.get(s.domain, 0) for s in batch], dtype=torch.long, device=device)
    fasttext_dclm = torch.tensor([s.fasttext_dclm if s.fasttext_dclm is not None else float("nan") for s in batch], device=device)
    fasttext_edu = torch.tensor([s.fasttext_edu if s.fasttext_edu is not None else float("nan") for s in batch], device=device)
    fasttext_code = torch.tensor([s.fasttext_code if s.fasttext_code is not None else float("nan") for s in batch], device=device)
    fasttext_math = torch.tensor([s.fasttext_math if s.fasttext_math is not None else float("nan") for s in batch], device=device)
    # Use EAI fasttext.english as language score when available.
    lang = torch.tensor(
      [
        (s.lang_score if s.lang_score is not None else (s.fasttext_english if s.fasttext_english is not None else float("nan")))
        for s in batch
      ],
      device=device,
    )
    artifacts = torch.tensor([s.extraction_artifacts if s.extraction_artifacts is not None else -1 for s in batch], device=device)
    missing = torch.tensor([s.missing_content if s.missing_content is not None else -1 for s in batch], device=device)

    loss = 0.0 * pooled.sum()
    loss = loss + _masked_mse(preds["fasttext_dclm"], fasttext_dclm, weight=0.25)
    loss = loss + _masked_mse(preds["fasttext_edu"], fasttext_edu, weight=0.25)
    loss = loss + _masked_mse(preds["fasttext_code"], fasttext_code, weight=0.25)
    loss = loss + _masked_mse(preds["fasttext_math"], fasttext_math, weight=0.25)
    loss = loss + _masked_mse(preds["lang_score"], lang, weight=0.2)
    loss = loss + 0.2 * F.cross_entropy(preds["domain"], domain)
    loss = loss + _masked_ce(preds["extraction_artifacts"], artifacts, weight=0.5)
    loss = loss + _masked_ce(preds["missing_content"], missing, weight=0.5)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
    opt.step()

    if rank == 0 and (step == 1 or (step % log_every == 0)):
      dt = (time.perf_counter() - t0) * 1000.0
      print(f"[hydra][phase_a] step={step}/{cfg.max_steps} loss={float(loss.item()):.4f} ms/step={dt:.1f}")

    # Early stopping: if loss below threshold for patience steps, stop
    loss_val = float(loss.item())
    loss_history.append(loss_val)
    if len(loss_history) == early_stop_patience and max(loss_history) < early_stop_loss:
      if rank == 0:
        print(f"[hydra][phase_a] early stop: loss < {early_stop_loss} for {early_stop_patience} steps")
      break

    # Checkpoint every 5k steps
    if rank == 0 and step % 5000 == 0:
      out = Path(cfg.out_dir)
      out.mkdir(parents=True, exist_ok=True)
      ckpt_path = out / f"hydra_probe_phase_a_step{step}.pt"
      torch.save(probe.state_dict(), ckpt_path)
      print(f"[hydra][phase_a] checkpoint saved: {ckpt_path}")

  if rank == 0:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), out / "hydra_probe_phase_a.pt")


def train_phase_b(cfg: PhaseBConfig) -> None:
  device, rank, world_size = _setup_distributed()
  enc = tiktoken.get_encoding("o200k_harmony")

  model = Transformer.from_checkpoint(cfg.checkpoint, device=device)
  for p in model.parameters():
    p.requires_grad_(False)
  model.eval()

  judge = JudgeEncoder(model.config.hidden_size).to(device)
  judge.train()
  opt = torch.optim.AdamW(judge.parameters(), lr=cfg.lr, weight_decay=1e-4)

  oracle_it = _cycle(lambda: OracleLabelSource(cfg.oracle_labels_jsonl, data_root=cfg.oracle_data_root, max_ctx=cfg.max_ctx))
  fw_sampler = None
  if cfg.fw_score2_glob:
    fw_sampler = FineWebPairSampler(FineWebEduScore2Source(cfg.fw_score2_glob, max_ctx=cfg.max_ctx), reservoir_per_score=256)

  log_every = int(os.environ.get("HYDRA_LOG_EVERY", "50"))
  warmup_steps = int(os.environ.get("HYDRA_WARMUP_STEPS", "500"))

  for step in range(1, cfg.max_steps + 1):
    t0 = time.perf_counter()
    oracle_batch = [next(oracle_it) for _ in range(cfg.batch_size)]
    input_ids, positions, _ = _pad_token_batch([s.input_ids or torch.empty(0) for s in oracle_batch], pad_id=EOS_TOKEN_ID)
    targets = torch.tensor([s.oracle_5dim for s in oracle_batch], dtype=torch.float32)

    input_ids = input_ids.to(device)
    positions = positions.to(device)
    targets = targets.to(device)

    with torch.no_grad():
      with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        _, h = model(
          input_ids,
          positions,
          return_hidden_states=True,
          up_to_layer=24,
          no_logits=True,
        )
      h24 = h.get(24)
      if h24 is None:
        raise RuntimeError("layer 24 hidden state not captured")
      h24f = torch.nan_to_num(h24, nan=0.0).float()

    rubric = judge(h24f)
    rubric = torch.clamp(rubric, 0.0, 4.0)
    targets = torch.clamp(targets, 0.0, 4.0)
    oracle_loss = F.huber_loss(rubric, targets, delta=0.5)
    loss = oracle_loss

    pair_loss = None
    if fw_sampler is not None and random.random() >= cfg.oracle_only_prob:
      pairs = fw_sampler.next_pairs(cfg.fw_pairs_per_step)
      ids_a, pos_a, _ = _tokenize_batch(enc, [p.a.text for p in pairs], max_ctx=cfg.max_ctx)
      ids_b, pos_b, _ = _tokenize_batch(enc, [p.b.text for p in pairs], max_ctx=cfg.max_ctx)

      with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
          _, h_a = model(ids_a.to(device), pos_a.to(device), return_hidden_states=True, up_to_layer=24, no_logits=True)
          _, h_b = model(ids_b.to(device), pos_b.to(device), return_hidden_states=True, up_to_layer=24, no_logits=True)
        h24_a = torch.nan_to_num(h_a.get(24), nan=0.0).float()
        h24_b = torch.nan_to_num(h_b.get(24), nan=0.0).float()

      q_a = JudgeEncoder.aggregate(judge(h24_a))
      q_b = JudgeEncoder.aggregate(judge(h24_b))
      pair_loss = F.margin_ranking_loss(q_a, q_b, target=torch.ones_like(q_a), margin=0.1)
      loss = loss + 0.1 * pair_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(judge.parameters(), 1.0)

    if warmup_steps > 0:
      lr_scale = min(1.0, step / float(warmup_steps))
      for g in opt.param_groups:
        g["lr"] = cfg.lr * lr_scale

    opt.step()

    if rank == 0 and (step == 1 or (step % log_every == 0)):
      dt = (time.perf_counter() - t0) * 1000.0
      pair_str = f" pair={float(pair_loss.item()):.4f}" if pair_loss is not None else ""
      print(
        f"[hydra][phase_b] step={step}/{cfg.max_steps} loss={float(loss.item()):.4f} "
        f"oracle={float(oracle_loss.item()):.4f}{pair_str} ms/step={dt:.1f}"
      )

    # Checkpoint every 2k steps (Phase B is shorter)
    if rank == 0 and step % 2000 == 0:
      out = Path(cfg.out_dir)
      out.mkdir(parents=True, exist_ok=True)
      ckpt_path = out / f"hydra_judge_step{step}.pt"
      torch.save(judge.export(), ckpt_path)
      print(f"[hydra][phase_b] checkpoint saved: {ckpt_path}")

  if rank == 0:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(judge.export(), out / "hydra_judge.pt")


def train_phase_c(cfg: PhaseCConfig) -> None:
  device, rank, world_size = _setup_distributed()
  enc = tiktoken.get_encoding("o200k_harmony")

  model = Transformer.from_checkpoint(cfg.checkpoint, device=device)
  for p in model.parameters():
    p.requires_grad_(False)
  model.eval()

  probe = ProbeHead(model.config.hidden_size).to(device)
  probe.load_state_dict(torch.load(cfg.probe_path, map_location=device))
  probe.train()

  export = torch.load(cfg.judge_export_path, map_location="cpu")
  judge = JudgeEncoder(model.config.hidden_size).to(device)
  judge.projector.load_state_dict(export["projector"])
  # q_token was stashed in encoder dict
  q_token = export["encoder"].pop("q_token")
  judge.encoder.load_state_dict(export["encoder"])
  judge.rubric_head.load_state_dict(export["rubric_head"])
  judge.q_token.data.copy_(q_token.to(device=device, dtype=judge.q_token.dtype))
  judge.eval()
  for p in judge.parameters():
    p.requires_grad_(False)

  opt = torch.optim.AdamW(probe.parameters(), lr=cfg.lr)
  log_every = int(os.environ.get("HYDRA_LOG_EVERY", "50"))

  distill_it = _cycle(lambda: EAIParquetSource(cfg.distill_source_glob, domain="web", max_ctx=cfg.max_ctx))
  hard: deque[HydraSample] = deque(maxlen=4096)

  for step in range(1, cfg.max_steps + 1):
    t0 = time.perf_counter()
    use_hard = (len(hard) >= cfg.batch_size) and (random.random() < 0.2)
    batch = random.sample(list(hard), k=cfg.batch_size) if use_hard else [next(distill_it) for _ in range(cfg.batch_size)]

    input_ids, positions, mask = _tokenize_batch(enc, [s.text for s in batch], max_ctx=cfg.max_ctx)
    input_ids = input_ids.to(device)
    positions = positions.to(device)
    mask = mask.to(device)

    with torch.no_grad():
      with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        _, h = model(input_ids, positions, return_hidden_states=True, up_to_layer=24, no_logits=True)
      h18 = h.get(18)
      h24 = h.get(24)
      if h18 is None or h24 is None:
        raise RuntimeError("missing hidden states (need 18 and 24)")
      pooled = pool_hidden(h18.float(), mask=mask)
      q_judge = JudgeEncoder.aggregate(judge(torch.nan_to_num(h24, nan=0.0).float()))

    preds = probe(pooled)
    q_probe = preds["gate"]

    loss = F.mse_loss(q_probe, q_judge.detach())

    # Keep Phase A heads alive at low weight to avoid drift.
    domain = torch.tensor([DOMAIN_MAP.get(s.domain, 0) for s in batch], dtype=torch.long, device=device)
    fasttext_dclm = torch.tensor([s.fasttext_dclm if s.fasttext_dclm is not None else float("nan") for s in batch], device=device)
    fasttext_edu = torch.tensor([s.fasttext_edu if s.fasttext_edu is not None else float("nan") for s in batch], device=device)
    fasttext_code = torch.tensor([s.fasttext_code if s.fasttext_code is not None else float("nan") for s in batch], device=device)
    fasttext_math = torch.tensor([s.fasttext_math if s.fasttext_math is not None else float("nan") for s in batch], device=device)
    lang = torch.tensor(
      [
        (s.lang_score if s.lang_score is not None else (s.fasttext_english if s.fasttext_english is not None else float("nan")))
        for s in batch
      ],
      device=device,
    )
    artifacts = torch.tensor([s.extraction_artifacts if s.extraction_artifacts is not None else -1 for s in batch], device=device)
    missing = torch.tensor([s.missing_content if s.missing_content is not None else -1 for s in batch], device=device)

    aux = 0.0 * loss
    aux = aux + _masked_mse(preds["fasttext_dclm"], fasttext_dclm, weight=0.25)
    aux = aux + _masked_mse(preds["fasttext_edu"], fasttext_edu, weight=0.25)
    aux = aux + _masked_mse(preds["fasttext_code"], fasttext_code, weight=0.25)
    aux = aux + _masked_mse(preds["fasttext_math"], fasttext_math, weight=0.25)
    aux = aux + _masked_mse(preds["lang_score"], lang, weight=0.2)
    aux = aux + 0.2 * F.cross_entropy(preds["domain"], domain)
    aux = aux + _masked_ce(preds["extraction_artifacts"], artifacts, weight=0.5)
    aux = aux + _masked_ce(preds["missing_content"], missing, weight=0.5)
    loss = loss + 0.1 * aux

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
    opt.step()

    with torch.no_grad():
      err = (q_probe.detach() - q_judge.detach()).abs().cpu()
      topk = torch.topk(err, k=min(8, len(batch))).indices.tolist()
      for i in topk:
        hard.append(batch[i])

    if rank == 0 and (step == 1 or (step % log_every == 0)):
      dt = (time.perf_counter() - t0) * 1000.0
      print(f"[hydra][phase_c] step={step}/{cfg.max_steps} loss={float(loss.item()):.4f} ms/step={dt:.1f}")

    # Checkpoint every 10k steps (Phase C is longest)
    if rank == 0 and step % 10000 == 0:
      out = Path(cfg.out_dir)
      out.mkdir(parents=True, exist_ok=True)
      ckpt_path = out / f"hydra_probe_step{step}.pt"
      torch.save(probe.state_dict(), ckpt_path)
      print(f"[hydra][phase_c] checkpoint saved: {ckpt_path}")

  if rank == 0:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), out / "hydra_probe.pt")


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser("nmoe.data.train")
  sp = p.add_subparsers(dest="cmd", required=True)

  a = sp.add_parser("phase-a")
  a.add_argument("--checkpoint", required=True)
  a.add_argument("--out", required=True)
  a.add_argument("--max-steps", type=int, default=50_000)
  a.add_argument("--batch-size", type=int, default=32)
  a.add_argument("--lr", type=float, default=1e-3)
  a.add_argument("--max-ctx", type=int, default=4096)
  a.add_argument("--eai-code", default="")
  a.add_argument("--eai-stem", default="")
  a.add_argument("--eai-math", default="")
  a.add_argument("--eai-med", default="")
  a.add_argument("--fw-score2", default="")

  b = sp.add_parser("phase-b")
  b.add_argument("--checkpoint", required=True)
  b.add_argument("--out", required=True)
  b.add_argument("--oracle-labels", required=True)
  b.add_argument("--oracle-data-root", required=True)
  b.add_argument("--max-steps", type=int, default=10_000)
  b.add_argument("--batch-size", type=int, default=64)
  b.add_argument("--lr", type=float, default=1e-4)
  b.add_argument("--max-ctx", type=int, default=4096)
  b.add_argument("--oracle-only-prob", type=float, default=0.6)
  b.add_argument("--fw-score2", default="")
  b.add_argument("--fw-pairs-per-step", type=int, default=32)

  c = sp.add_parser("phase-c")
  c.add_argument("--checkpoint", required=True)
  c.add_argument("--out", required=True)
  c.add_argument("--probe", required=True)
  c.add_argument("--judge", required=True)
  c.add_argument("--distill", required=True)
  c.add_argument("--max-steps", type=int, default=100_000)
  c.add_argument("--batch-size", type=int, default=64)
  c.add_argument("--lr", type=float, default=1e-4)
  c.add_argument("--max-ctx", type=int, default=4096)

  return p


def main(argv: list[str] | None = None) -> int:
  args = _build_argparser().parse_args(argv)

  if args.cmd == "phase-a":
    train_phase_a(
      PhaseAConfig(
        checkpoint=args.checkpoint,
        out_dir=args.out,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_ctx=args.max_ctx,
        eai_code_glob=args.eai_code,
        eai_stem_glob=args.eai_stem,
        eai_math_glob=args.eai_math,
        eai_med_glob=args.eai_med,
        fw_score2_glob=args.fw_score2,
      )
    )
    return 0

  if args.cmd == "phase-b":
    train_phase_b(
      PhaseBConfig(
        checkpoint=args.checkpoint,
        out_dir=args.out,
        oracle_labels_jsonl=args.oracle_labels,
        oracle_data_root=args.oracle_data_root,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_ctx=args.max_ctx,
        oracle_only_prob=args.oracle_only_prob,
        fw_pairs_per_step=args.fw_pairs_per_step,
        fw_score2_glob=args.fw_score2,
      )
    )
    return 0

  if args.cmd == "phase-c":
    train_phase_c(
      PhaseCConfig(
        checkpoint=args.checkpoint,
        out_dir=args.out,
        probe_path=args.probe,
        judge_export_path=args.judge,
        distill_source_glob=args.distill,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_ctx=args.max_ctx,
      )
    )
    return 0

  raise SystemExit(2)


if __name__ == "__main__":
  raise SystemExit(main())

