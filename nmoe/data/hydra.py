from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .model import BatchedGenerator, Transformer, pool_hidden
from .train import ProbeHead, JudgeEncoder
from .score import build_prompt, right_trim, grade_prompts, compute_aggregated
from .docid import parse_doc_id, shard_path

try:
  from openai_harmony import HarmonyEncodingName, load_harmony_encoding
except Exception:  # pragma: no cover
  HarmonyEncodingName = None  # type: ignore
  load_harmony_encoding = None  # type: ignore


def _load_enc():
    if load_harmony_encoding is None:
        raise RuntimeError("openai_harmony not available")
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def cmd_oracle_label(args: argparse.Namespace) -> int:
    """Oracle labeling with per-record streaming writes and resume.

    - Appends to scores.jsonl and audit_snippets.jsonl as each record finishes and flushes immediately.
    - Prints a compact heartbeat every 100 new records.
    - Resumes by skipping doc_ids already present in scores.jsonl.
    - Writes summary.json at the end (and summary.partial.json periodically).
    """
    enc = _load_enc()
    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    # Resume: load already processed doc_ids (if any)
    scores_path = out_dir / "scores.jsonl"
    audit_path = out_dir / "audit_snippets.jsonl"
    summary_path = out_dir / "summary.json"
    partial_summary_path = out_dir / "summary.partial.json"
    processed: set[str] = set()
    per_source: Dict[str, Tuple[int, int]] = {}
    ok_total = 0
    already = 0
    if scores_path.exists():
        with scores_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    j = json.loads(line)
                    did = j.get("doc_id")
                    src = j.get("source", "unknown")
                    if did:
                        processed.add(str(did))
                    ok = bool(j.get("ok", j.get("scores") is not None))
                    c_all, c_ok = per_source.get(src, (0, 0))
                    per_source[src] = (c_all + 1, c_ok + (1 if ok else 0))
                    if ok:
                        ok_total += 1
                    already += 1
                except Exception:
                    # Ignore malformed trailing line
                    pass

    # Input iterator (don’t materialize full list if large)
    def iter_samples():
        if getattr(args, "input_docids", None):
            import numpy as np
            with open(args.input_docids, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    j = json.loads(line)
                    doc_id = j.get("doc_id")
                    if not doc_id:
                        continue
                    if doc_id in processed:
                        continue  # resume skip
                    d = parse_doc_id(doc_id)
                    arr = np.load(shard_path(args.data_root, d), mmap_mode="r")
                    toks = arr[int(d.start) : int(d.end)].tolist()
                    text = enc.decode(toks)
                    src = j.get("source") or d.source
                    yield {"doc_id": doc_id, "source": src, "text": text}
        else:
            with open(args.input_jsonl, "r", encoding="utf-8") as f:
                idx = already
                for line in f:
                    if not line.strip():
                        continue
                    j = json.loads(line)
                    text = j.get("text", "")
                    if not text:
                        continue
                    doc_id = j.get("doc_id") or j.get("id") or f"row:{idx}"
                    idx += 1
                    if doc_id in processed:
                        continue
                    src = j.get("source") or j.get("src") or "unknown"
                    yield {"doc_id": doc_id, "source": src, "text": text}

    # Streaming write handles (append mode)
    scores_f = scores_path.open("a", encoding="utf-8")
    audit_f = audit_path.open("a", encoding="utf-8")

    # Generator once; submit chunks
    device = torch.device("cuda")
    gen = BatchedGenerator(
        args.checkpoint, max_seq_len=args.max_ctx, max_batch=args.max_batch, device=device
    )

    CHUNK = max(args.max_batch * 2, 64)
    buf: List[Dict[str, Any]] = []
    wrote = 0
    total_seen = already

    def process_chunk(chunk: List[Dict[str, Any]]):
        nonlocal wrote, ok_total, total_seen, per_source
        if not chunk:
            return
        # Build prompts for chunk
        prompts: List[List[int]] = []
        snippets: List[str] = []
        for s in chunk:
            toks = build_prompt(enc, s["text"], is_code=False)
            toks = right_trim(toks, max_ctx=args.max_ctx, max_new=args.max_new)
            prompts.append(toks)
            snippets.append(s["text"][:256].replace("\n", " "))

        def on_finish(idx: int, r: Dict[str, Any]) -> None:
            nonlocal wrote, ok_total, total_seen, per_source
            try:
                s = chunk[idx]; snip = snippets[idx]
                row: Dict[str, Any] = {
                    "doc_id": s["doc_id"],
                    "source": s["source"],
                    "ok": bool(r.get("ok")),
                    "snippet": snip,
                }
                if row["ok"] and r.get("scores"):
                    row["scores"] = r["scores"]
                    row["aggregated"] = compute_aggregated(r["scores"])  # [0,1]
                    ok_total += 1
                else:
                    row["raw"] = r.get("final_text", "")
                scores_f.write(json.dumps(row, ensure_ascii=False) + "\n"); scores_f.flush()
                audit_f.write(json.dumps({k: row[k] for k in ("doc_id", "source", "ok")} | {"scores": row.get("scores"), "snippet": snip}, ensure_ascii=False) + "\n"); audit_f.flush()
                c_all, c_ok = per_source.get(s["source"], (0, 0))
                per_source[s["source"]] = (c_all + 1, c_ok + (1 if row["ok"] else 0))
                wrote += 1; total_seen += 1
                if wrote % 100 == 0:
                    partial = {
                        "total_seen": total_seen,
                        "parsed": ok_total,
                        "parse_rate": (ok_total / max(1, total_seen)),
                    }
                    with partial_summary_path.open("w", encoding="utf-8") as pf:
                        json.dump(partial, pf)
                    print(f"wrote {wrote} (this run), total_seen={total_seen}, ok={ok_total}", flush=True)
            except Exception:
                pass

        # Stream results as each sequence finishes
        _ = grade_prompts(gen, enc, prompts, max_new=args.max_new, on_finish=on_finish)

    # Drain iterator
    for s in iter_samples():
        buf.append(s)
        if len(buf) >= CHUNK:
            process_chunk(buf)
            buf.clear()
    process_chunk(buf)

    # Close appenders
    scores_f.close(); audit_f.close()

    # Final summary
    total = total_seen
    summary = {
        "total": total,
        "parsed": ok_total,
        "parse_rate": (ok_total / max(1, total)),
        "per_source": {k: {"count": v[0], "parsed": v[1], "parse_rate": (v[1] / max(1, v[0]))} for k, v in per_source.items()},
        "scorer_model": "gpt-oss-120b",
        "scorer_version": "gpt-oss-120b-v1",
        "resumed_from": already,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    # Expect labels.jsonl with scores and aggregated or compute aggregated
    dims = ["helpfulness", "correctness", "coherence", "complexity", "density"]
    X: List[List[float]] = []
    y: List[float] = []
    with open(args.labels, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            s = j.get("scores")
            if not s:
                continue
            xrow = [float(s.get(k, 0.0)) / 4.0 for k in dims]
            X.append(xrow)
            if "aggregated" in j:
                y.append(float(j["aggregated"]))
            else:
                y.append(sum(xrow) / len(xrow))
    if not X:
        print("No valid rows in labels file")
        return 1
    Xn = np.asarray(X, dtype=np.float64)
    yn = np.asarray(y, dtype=np.float64)
    # Solve least squares w to minimize ||Xw - y||
    w, *_ = np.linalg.lstsq(Xn, yn, rcond=None)
    w = np.clip(w, 0.0, 1.0)
    w = w / max(1e-8, w.sum())

    # Simple τ suggestion at target keep rate
    target_keep = float(args.target_keep)
    scores = (Xn @ w)
    taus = np.quantile(scores, [1 - target_keep, target_keep])
    out = {
        "weights": {k: float(v) for k, v in zip(dims, w)},
        "tau_drop": float(taus[0]),
        "tau_keep": float(taus[1]),
        "count": int(len(scores)),
    }
    out_path = Path(args.out)
    _ensure_dir(out_path)
    with (out_path / "calibration_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0


def _tokenize_batch(texts: List[str], enc_name: str = "o200k_harmony", max_ctx: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
    import tiktoken

    enc = tiktoken.get_encoding(enc_name)
    ids = [enc.encode_ordinary(t)[:max_ctx] for t in texts]
    max_len = max(len(x) for x in ids)
    input_ids = torch.zeros(len(ids), max_len, dtype=torch.long)
    for i, t in enumerate(ids):
        input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    positions = torch.arange(max_len, dtype=torch.long).unsqueeze(0).repeat(len(ids), 1)
    return input_ids, positions


def _load_calibration(cal_dir: str) -> Tuple[Dict[str, float], float, float]:
    p = Path(cal_dir) / "calibration_summary.json"
    with p.open("r", encoding="utf-8") as f:
        j = json.load(f)
    w = j.get("weights") or {}
    return w, float(j.get("tau_drop", 0.3)), float(j.get("tau_keep", 0.55))


def _aggregated_with_w(scores: Dict[str, float], w: Dict[str, float]) -> float:
    dims = ["helpfulness", "correctness", "coherence", "complexity", "density"]
    if not w:
        return compute_aggregated(scores)
    s = 0.0
    tot = 0.0
    for k in dims:
        if k in scores and k in w:
            s += (float(scores[k]) / 4.0) * float(w[k])
            tot += float(w[k])
    if tot <= 0:
        return compute_aggregated(scores)
    return s / tot


class _LegacyJudgeEncoderV0(nn.Module):
  """Legacy HYDRA judge head (pre-export-format).

  Checkpoints with keys like:
    q0, proj.*, enc0.*, enc1.*, shared_head.*, score_embed.*
  """

  def __init__(self, hidden_dim: int, *, mid_dim: int = 512, nhead: int = 8):
    super().__init__()
    self.q0 = nn.Parameter(torch.empty(hidden_dim))
    nn.init.normal_(self.q0, mean=0.0, std=0.02)
    self.proj = nn.Linear(hidden_dim, mid_dim)
    self.enc0 = nn.TransformerEncoderLayer(d_model=mid_dim, nhead=nhead, batch_first=True)
    self.enc1 = nn.TransformerEncoderLayer(d_model=mid_dim, nhead=nhead, batch_first=True)
    self.shared_head = nn.Linear(mid_dim, 5)
    self.score_embed = nn.Sequential(
      nn.Linear(5, mid_dim),
      nn.ReLU(),
      nn.Linear(mid_dim, mid_dim),
    )

  def forward(self, h24_seq: torch.Tensor) -> torch.Tensor:
    bsz, _, h = h24_seq.shape
    q = self.q0.unsqueeze(0).unsqueeze(1).expand(bsz, 1, h)
    z = self.proj(torch.cat([q, h24_seq], dim=1))
    z = self.enc0(z)
    s0 = self.shared_head(z[:, 0, :])
    z0 = z[:, 0, :] + self.score_embed(s0)
    z = torch.cat([z0.unsqueeze(1), z[:, 1:, :]], dim=1)
    z = self.enc1(z)
    return self.shared_head(z[:, 0, :])


def load_local_hydra_judge(*, model: Transformer, heads_dir: str | Path, device: torch.device) -> nn.Module:
  """Load HYDRA judge head weights on top of a frozen backbone."""
  ckpt_dir = Path(heads_dir)
  # Try new export format first, fall back to legacy
  judge_path = ckpt_dir / "hydra_judge.pt"
  if not judge_path.exists():
    judge_path = ckpt_dir / "hydra_judge_phase_b.pt"
  ckpt = torch.load(judge_path, map_location=device)

  # New format from JudgeEncoder.export() has nested dicts.
  if isinstance(ckpt, dict) and "projector" in ckpt and "encoder" in ckpt and "rubric_head" in ckpt:
    judge = JudgeEncoder(model.config.hidden_size).to(device)
    judge.projector.load_state_dict(ckpt["projector"])
    q_token = ckpt["encoder"].pop("q_token", None)
    judge.encoder.load_state_dict(ckpt["encoder"])
    judge.rubric_head.load_state_dict(ckpt["rubric_head"])
    if q_token is not None:
      judge.q_token.data.copy_(q_token.to(device=device, dtype=judge.q_token.dtype))

    judge = judge.to(device=device, dtype=torch.bfloat16)
    judge.eval()
    return judge

  # Legacy v0 flat state_dict (enc0/enc1 + score_embed refinement).
  if isinstance(ckpt, dict) and "q0" in ckpt and "proj.weight" in ckpt and "enc0.self_attn.in_proj_weight" in ckpt:
    legacy = _LegacyJudgeEncoderV0(model.config.hidden_size).to(device)
    legacy.load_state_dict(ckpt)
    legacy = legacy.to(device=device, dtype=torch.bfloat16)
    legacy.eval()
    return legacy

  # Legacy flat state_dict for the current JudgeEncoder module.
  judge = JudgeEncoder(model.config.hidden_size).to(device)
  judge.load_state_dict(ckpt)
  judge = judge.to(device=device, dtype=torch.bfloat16)
  judge.eval()
  return judge


def load_local_hydra_probe(*, model: Transformer, heads_dir: str | Path, device: torch.device) -> ProbeHead:
  """Load HYDRA probe head weights for early exit decisions."""
  ckpt_dir = Path(heads_dir)
  probe = ProbeHead(model.config.hidden_size).to(device)
  # Try phase-c (distilled) first, then phase-a, then legacy
  probe_path = ckpt_dir / "hydra_probe_phase_c.pt"
  if not probe_path.exists():
    probe_path = ckpt_dir / "hydra_probe_phase_a.pt"
  if not probe_path.exists():
    probe_path = ckpt_dir / "hydra_probe.pt"
  if probe_path.exists():
    probe.load_state_dict(torch.load(probe_path, map_location=device))
  else:
    print(f"[hydra] Warning: No probe checkpoint found in {ckpt_dir}, using random init")
  probe = probe.to(device=device, dtype=torch.bfloat16)
  probe.eval()
  return probe


@torch.no_grad()
def grade_texts_with_local_hydra_judge(
  *,
  model: Transformer,
  judge: nn.Module,
  probe: ProbeHead | None = None,
  ln24: nn.LayerNorm | None = None,
  texts: List[str],
  doc_ids: List[str],
  max_ctx: int,
  batch_size: int,
  w: Dict[str, float],
  tau_drop: float,
  tau_keep: float,
  device: torch.device,
  enc_name: str = "o200k_harmony",
  use_early_exit: bool = True,
) -> List[Dict[str, Any]]:
  """Grade texts with HYDRA probe (L18 early exit) + judge (L24 final).

  Flow from HYDRA_TRAINING.md:
  1. Run probe at L18 → q_gate
  2. If q_gate < τ_drop → DROP (save compute, skip L24)
  3. If q_gate ≥ τ_drop → continue to L24 → judge → final decision

  Returns list of rows:
    {doc_id, q_probe, scores{5}, aggregated, decision, early_exit}
  """
  if len(texts) != len(doc_ids):
    raise ValueError("texts/doc_ids length mismatch")
  if ln24 is None:
    ln24 = nn.LayerNorm(model.config.hidden_size).to(device)
  judge_dtype = next(judge.parameters()).dtype
  probe_dtype = next(probe.parameters()).dtype if probe is not None else None
  out: List[Dict[str, Any]] = []

  for i in range(0, len(texts), batch_size):
    chunk_texts = texts[i : i + batch_size]
    chunk_ids = doc_ids[i : i + batch_size]
    input_ids, positions = _tokenize_batch(chunk_texts, enc_name=enc_name, max_ctx=max_ctx)
    input_ids = input_ids.to(device)
    positions = positions.to(device)

    # Always run to L18 first
    _, h = model(
      input_ids,
      positions,
      return_hidden_states=True,
      up_to_layer=18,
      no_logits=True,
    )
    h18 = h.get(18)
    if h18 is None:
      raise RuntimeError("Missing hidden state at layer 18")
    h18_pooled = pool_hidden(h18).float()

    # Probe scoring
    if probe is not None:
      probe_out = probe(h18_pooled.to(dtype=probe_dtype))
      q_probe = probe_out["gate"]
    else:
      q_probe = torch.zeros(len(chunk_ids), device=device)

    # Determine which samples need L24
    need_judge = torch.ones(len(chunk_ids), dtype=torch.bool, device=device)
    if use_early_exit and probe is not None:
      need_judge = q_probe >= tau_drop

    # Process early exits
    for j, did in enumerate(chunk_ids):
      q_val = float(q_probe[j].item())
      if not need_judge[j]:
        out.append({
          "doc_id": did,
          "q_probe": q_val,
          "scores": None,
          "aggregated": q_val,
          "decision": "drop",
          "early_exit": True,
        })

    # Continue to L24 for remaining samples
    judge_indices = need_judge.nonzero(as_tuple=True)[0].tolist()
    if judge_indices:
      judge_input_ids = input_ids[judge_indices]
      judge_positions = positions[judge_indices]

      _, h_full = model(
        judge_input_ids,
        judge_positions,
        return_hidden_states=True,
        up_to_layer=model.config.num_hidden_layers,
        no_logits=True,
      )
      h24 = h_full.get(24)
      if h24 is None:
        raise RuntimeError("Missing hidden state at layer 24")
      h24f = torch.nan_to_num(h24, nan=0.0, posinf=1e4, neginf=-1e4).float()
      h24f = ln24(h24f).to(dtype=judge_dtype)

      sj = judge(h24f)
      sj = torch.clamp(sj, 0.0, 4.0)

      for k, orig_j in enumerate(judge_indices):
        did = chunk_ids[orig_j]
        q_val = float(q_probe[orig_j].item())
        final_scores = {
          dim: float(sj[k, idx].item())
          for idx, dim in enumerate(["helpfulness", "correctness", "coherence", "complexity", "density"])
        }
        agg_final = _aggregated_with_w(final_scores, w)
        if agg_final >= tau_keep:
          decision = "keep"
        elif agg_final < tau_drop:
          decision = "drop"
        else:
          decision = "band"
        out.append({
          "doc_id": did,
          "q_probe": q_val,
          "scores": final_scores,
          "aggregated": agg_final,
          "decision": decision,
          "early_exit": False,
        })

  return out


def cmd_grade(args: argparse.Namespace) -> int:
  """Grade corpus with HYDRA probe + judge (L18 early exit + L24 final)."""
  def iter_rows() -> Iterator[Dict[str, Any]]:
    if getattr(args, "input_docids", None):
      if not getattr(args, "data_root", None):
        raise ValueError("--data-root is required when using --input-docids")
      with open(args.input_docids, "r", encoding="utf-8") as f:
        for line in f:
          if not line.strip():
            continue
          j = json.loads(line)
          doc_id = j.get("doc_id") or j.get("id")
          if not doc_id:
            continue
          src = j.get("source")
          if not src:
            try:
              src = parse_doc_id(doc_id).source
            except Exception:
              src = "unknown"
          yield {"doc_id": doc_id, "source": src}
    else:
      with open(args.input_jsonl, "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
          if not line.strip():
            continue
          j = json.loads(line)
          text = j.get("text")
          if text is None:
            continue
          did = j.get("doc_id") or j.get("id") or f"row:{idx}"
          idx += 1
          yield {"doc_id": did, "source": j.get("source", "unknown"), "text": text}

  # Load backbone, probe, and judge heads
  device = torch.device("cuda")
  print("[hydra] Loading backbone...")
  model = Transformer.from_checkpoint(args.checkpoint, device=device)
  for p in model.parameters():
    p.requires_grad_(False)
  model.eval()

  print("[hydra] Loading heads...")
  judge = load_local_hydra_judge(model=model, heads_dir=args.heads_dir, device=device)
  probe = load_local_hydra_probe(model=model, heads_dir=args.heads_dir, device=device)
  ln24 = nn.LayerNorm(model.config.hidden_size).to(device)
  judge_dtype = next(judge.parameters()).dtype
  probe_dtype = next(probe.parameters()).dtype

  # Calibration
  w, tau_drop, tau_keep = _load_calibration(args.calibration)
  use_early_exit = not getattr(args, "no_early_exit", False)
  print(f"[hydra] tau_drop={tau_drop:.3f} tau_keep={tau_keep:.3f} early_exit={use_early_exit}")

  # Output setup
  out_dir = Path(args.out)
  _ensure_dir(out_dir)
  out_path = out_dir / "quality_scores.jsonl"
  out_tmp_path = out_dir / "quality_scores.jsonl.tmp"
  summary_path = out_dir / "summary.json"

  # Stats
  kept, dropped, band, early_exits = 0, 0, 0, 0
  per_source: Dict[str, Dict[str, int]] = {}
  total = 0
  t_start = time.perf_counter()

  with out_tmp_path.open("w", encoding="utf-8") as outf:
    B = int(args.max_batch)
    chunk: List[Dict[str, Any]] = []

    for rec in iter_rows():
      chunk.append(rec)
      if len(chunk) < B:
        continue

      # Process chunk
      total += len(chunk)

      # Build tensors
      if getattr(args, "input_docids", None):
        toks_list: List[List[int]] = []
        for r in chunk:
          d = parse_doc_id(r["doc_id"])
          arr = np.load(shard_path(args.data_root, d), mmap_mode="r")
          toks = arr[int(d.start) : int(d.end)].tolist()
          toks_list.append(toks[: args.max_ctx])
        max_len = max(len(t) for t in toks_list) if toks_list else 1
        input_ids = torch.zeros(len(toks_list), max_len, dtype=torch.long)
        positions = torch.zeros(len(toks_list), max_len, dtype=torch.long)
        for j2, t in enumerate(toks_list):
          n = len(t)
          if n:
            input_ids[j2, :n] = torch.tensor(t, dtype=torch.long)
            positions[j2, :n] = torch.arange(n, dtype=torch.long)
      else:
        texts = [r["text"] for r in chunk]
        input_ids, positions = _tokenize_batch(texts, max_ctx=args.max_ctx)

      input_ids = input_ids.to(device)
      positions = positions.to(device)

      with torch.no_grad():
        # L18 probe pass
        _, h18_dict = model(input_ids, positions, return_hidden_states=True, up_to_layer=18, no_logits=True)
        h18 = h18_dict.get(18)
        if h18 is None:
          raise RuntimeError("Missing hidden state at layer 18")
        h18_pooled = pool_hidden(h18).float()

        # Probe scoring
        probe_out = probe(h18_pooled.to(dtype=probe_dtype))
        q_probe = probe_out["gate"]

        # Early exit mask
        if use_early_exit:
          need_judge = q_probe >= tau_drop
        else:
          need_judge = torch.ones(len(chunk), dtype=torch.bool, device=device)

        judge_indices = need_judge.nonzero(as_tuple=True)[0].tolist()

        # L24 judge pass for non-early-exit samples
        rubric_scores = {}
        if judge_indices:
          judge_input_ids = input_ids[judge_indices]
          judge_positions = positions[judge_indices]
          _, h24_dict = model(judge_input_ids, judge_positions, return_hidden_states=True, up_to_layer=model.config.num_hidden_layers, no_logits=True)
          h24 = h24_dict.get(24)
          if h24 is None:
            raise RuntimeError("Missing hidden state at layer 24")
          h24f = torch.nan_to_num(h24, nan=0.0, posinf=1e4, neginf=-1e4).float()
          h24f = ln24(h24f).to(dtype=judge_dtype)
          sj = judge(h24f)
          sj = torch.clamp(sj, 0.0, 4.0)

          for k, orig_j in enumerate(judge_indices):
            rubric_scores[orig_j] = {
              dim: float(sj[k, idx].item())
              for idx, dim in enumerate(["helpfulness", "correctness", "coherence", "complexity", "density"])
            }

      # Write results
      for j, rec in enumerate(chunk):
        q_val = float(q_probe[j].item())
        src = rec["source"]
        if src not in per_source:
          per_source[src] = {"kept": 0, "dropped": 0, "band": 0, "early_exit": 0}

        if not need_judge[j]:
          # Early exit → drop
          decision = "drop"
          dropped += 1
          early_exits += 1
          per_source[src]["dropped"] += 1
          per_source[src]["early_exit"] += 1
          out_row = {
            "doc_id": rec["doc_id"],
            "source": src,
            "q_probe": q_val,
            "scores": None,
            "aggregated": q_val,
            "decision": decision,
            "early_exit": True,
          }
        else:
          final_scores = rubric_scores.get(j, {})
          agg_final = _aggregated_with_w(final_scores, w) if final_scores else q_val
          if agg_final >= tau_keep:
            decision = "keep"
            kept += 1
            per_source[src]["kept"] += 1
          elif agg_final < tau_drop:
            decision = "drop"
            dropped += 1
            per_source[src]["dropped"] += 1
          else:
            decision = "band"
            band += 1
            per_source[src]["band"] += 1
          out_row = {
            "doc_id": rec["doc_id"],
            "source": src,
            "q_probe": q_val,
            "scores": final_scores,
            "aggregated": agg_final,
            "decision": decision,
            "early_exit": False,
          }
        outf.write(json.dumps(out_row, ensure_ascii=False) + "\n")

      # Progress
      if total % (B * 10) == 0:
        elapsed = time.perf_counter() - t_start
        rate = total / elapsed if elapsed > 0 else 0
        print(f"[hydra] processed {total} docs | kept={kept} dropped={dropped} band={band} early_exit={early_exits} | {rate:.1f} docs/s")

      chunk = []

    # Final chunk
    if chunk:
      total += len(chunk)
      if getattr(args, "input_docids", None):
        toks_list = []
        for r in chunk:
          d = parse_doc_id(r["doc_id"])
          arr = np.load(shard_path(args.data_root, d), mmap_mode="r")
          toks = arr[int(d.start) : int(d.end)].tolist()
          toks_list.append(toks[: args.max_ctx])
        max_len = max(len(t) for t in toks_list) if toks_list else 1
        input_ids = torch.zeros(len(toks_list), max_len, dtype=torch.long)
        positions = torch.zeros(len(toks_list), max_len, dtype=torch.long)
        for j2, t in enumerate(toks_list):
          n = len(t)
          if n:
            input_ids[j2, :n] = torch.tensor(t, dtype=torch.long)
            positions[j2, :n] = torch.arange(n, dtype=torch.long)
      else:
        texts = [r["text"] for r in chunk]
        input_ids, positions = _tokenize_batch(texts, max_ctx=args.max_ctx)

      input_ids = input_ids.to(device)
      positions = positions.to(device)

      with torch.no_grad():
        _, h18_dict = model(input_ids, positions, return_hidden_states=True, up_to_layer=18, no_logits=True)
        h18 = h18_dict.get(18)
        h18_pooled = pool_hidden(h18).float()
        probe_out = probe(h18_pooled.to(dtype=probe_dtype))
        q_probe = probe_out["gate"]

        if use_early_exit:
          need_judge = q_probe >= tau_drop
        else:
          need_judge = torch.ones(len(chunk), dtype=torch.bool, device=device)

        judge_indices = need_judge.nonzero(as_tuple=True)[0].tolist()
        rubric_scores = {}
        if judge_indices:
          judge_input_ids = input_ids[judge_indices]
          judge_positions = positions[judge_indices]
          _, h24_dict = model(judge_input_ids, judge_positions, return_hidden_states=True, up_to_layer=model.config.num_hidden_layers, no_logits=True)
          h24 = h24_dict.get(24)
          h24f = torch.nan_to_num(h24, nan=0.0).float()
          h24f = ln24(h24f).to(dtype=judge_dtype)
          sj = judge(h24f)
          sj = torch.clamp(sj, 0.0, 4.0)
          for k, orig_j in enumerate(judge_indices):
            rubric_scores[orig_j] = {
              dim: float(sj[k, idx].item())
              for idx, dim in enumerate(["helpfulness", "correctness", "coherence", "complexity", "density"])
            }

      for j, rec in enumerate(chunk):
        q_val = float(q_probe[j].item())
        src = rec["source"]
        if src not in per_source:
          per_source[src] = {"kept": 0, "dropped": 0, "band": 0, "early_exit": 0}
        if not need_judge[j]:
          dropped += 1
          early_exits += 1
          per_source[src]["dropped"] += 1
          per_source[src]["early_exit"] += 1
          out_row = {"doc_id": rec["doc_id"], "source": src, "q_probe": q_val, "scores": None, "aggregated": q_val, "decision": "drop", "early_exit": True}
        else:
          final_scores = rubric_scores.get(j, {})
          agg_final = _aggregated_with_w(final_scores, w) if final_scores else q_val
          if agg_final >= tau_keep:
            decision = "keep"
            kept += 1
            per_source[src]["kept"] += 1
          elif agg_final < tau_drop:
            decision = "drop"
            dropped += 1
            per_source[src]["dropped"] += 1
          else:
            decision = "band"
            band += 1
            per_source[src]["band"] += 1
          out_row = {"doc_id": rec["doc_id"], "source": src, "q_probe": q_val, "scores": final_scores, "aggregated": agg_final, "decision": decision, "early_exit": False}
        outf.write(json.dumps(out_row, ensure_ascii=False) + "\n")

  out_tmp_path.replace(out_path)
  elapsed = time.perf_counter() - t_start
  summary = {
    "total": total,
    "kept": kept,
    "dropped": dropped,
    "band": band,
    "early_exits": early_exits,
    "early_exit_rate": early_exits / max(1, total),
    "keep_rate": kept / max(1, total),
    "per_source": per_source,
    "tau_drop": tau_drop,
    "tau_keep": tau_keep,
    "elapsed_sec": elapsed,
    "docs_per_sec": total / elapsed if elapsed > 0 else 0,
  }
  with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
  print(json.dumps(summary, indent=2))
  return 0


def build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser("nmoe.data.hydra")
  sp = p.add_subparsers(dest="cmd", required=True)

  # Oracle labeling (120B backbone)
  p_label = sp.add_parser("oracle-label", help="Label data with 120B oracle")
  g_label = p_label.add_mutually_exclusive_group(required=True)
  g_label.add_argument("--input-jsonl", dest="input_jsonl")
  g_label.add_argument("--input-docids", dest="input_docids")
  p_label.add_argument("--data-root", help="Data root for shard paths when using --input-docids")
  p_label.add_argument("--checkpoint", required=True)
  p_label.add_argument("--out", required=True)
  p_label.add_argument("--max-new", dest="max_new", type=int, default=2048)
  p_label.add_argument("--max-ctx", dest="max_ctx", type=int, default=4096)
  p_label.add_argument("--max-batch", dest="max_batch", type=int, default=32)
  p_label.set_defaults(func=cmd_oracle_label)

  # Calibration
  p_cal = sp.add_parser("calibrate", help="Fit aggregation weights and thresholds from labels.jsonl")
  p_cal.add_argument("--labels", required=True)
  p_cal.add_argument("--out", required=True)
  p_cal.add_argument("--target-keep", default=0.5)
  p_cal.set_defaults(func=cmd_calibrate)

  # Grading (20B backbone + trained heads)
  p_grade = sp.add_parser("grade", help="Grade a corpus with HYDRA (probe early exit + judge final)")
  g_grade = p_grade.add_mutually_exclusive_group(required=True)
  g_grade.add_argument("--input-jsonl")
  g_grade.add_argument("--input-docids")
  p_grade.add_argument("--data-root", help="Data root for shard paths when using --input-docids")
  p_grade.add_argument("--checkpoint", required=True, help="20B backbone checkpoint dir")
  p_grade.add_argument("--heads-dir", required=True, help="Dir containing hydra_probe*.pt and hydra_judge*.pt")
  p_grade.add_argument("--calibration", required=True, help="Dir containing calibration_summary.json")
  p_grade.add_argument("--out", required=True)
  p_grade.add_argument("--max-ctx", dest="max_ctx", type=int, default=4096)
  p_grade.add_argument("--max-batch", dest="max_batch", type=int, default=16)
  p_grade.add_argument("--no-early-exit", dest="no_early_exit", action="store_true",
                       help="Disable probe early exit (always run full L24)")
  p_grade.set_defaults(func=cmd_grade)

  return p


def main(argv: List[str] | None = None) -> int:
  ap = build_argparser()
  args = ap.parse_args(argv)
  return args.func(args)


if __name__ == "__main__":
  raise SystemExit(main())
