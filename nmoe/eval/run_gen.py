from __future__ import annotations

"""
Generation-based eval runner (GSM8K / HumanEval / GPQA).

Issue [07] intent:
- No dependency on `nmoe.serve.*` (standalone, model-direct generation).
- Distributed-safe: all ranks participate and remain in lockstep by construction.
- Prompt-length semantics preserved (explicit length tracking; no "real-token padding" bugs).
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import tiktoken

from nmoe.metrics import start_metrics, stop_metrics

from nmoe.eval.tasks.gsm8k import GSM8K
from nmoe.eval.tasks.gpqa import GPQA
from nmoe.eval.tasks.humaneval import HumanEval


TASKS: dict[str, type] = {
    "GSM8K": GSM8K,
    "HumanEval": HumanEval,
    "GPQA": GPQA,
}


def _init_dist() -> tuple[int, int]:
    if "RANK" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return int(rank), int(world)


def _print0(rank: int, *a: Any, **kw: Any) -> None:
    if rank == 0:
        print(*a, **kw)


def _load_config(path: str):
    import tomllib
    from nmoe.config import Config

    with open(path, "rb") as f:
        cfg_dict = tomllib.load(f)
    return Config(**cfg_dict)


def _load_snapshot(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = obj.get("config", {})
    model_state = obj.get("model_state", {})
    if not isinstance(cfg_dict, dict) or not isinstance(model_state, dict):
        raise ValueError("snapshot must contain dict fields 'config' and 'model_state'")
    return cfg_dict, model_state


@torch.no_grad()
def _forward_hidden(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "embedding") or not hasattr(model, "blocks") or not hasattr(model, "norm"):
        raise TypeError("model must be an nmoe.model.Transformer-like module")
    if not hasattr(model, "rope"):
        raise TypeError("model must have rope buffers (RotaryEmbedding)")

    x = model.embedding(tokens) * float(getattr(model, "mup_scale_factor", 1.0))
    seqlen = int(tokens.size(1))
    cos = model.rope.cos[:seqlen].to(tokens.device)
    sin = model.rope.sin[:seqlen].to(tokens.device)
    for block in model.blocks:
        x = block(x, cos, sin)
    x = model.norm(x)
    return x


@torch.no_grad()
def _argmax_vocab_chunked(
    h: torch.Tensor,  # [D] fp32
    lm_head_weight: torch.Tensor,  # [V,D]
    *,
    logits_scale: float,
    chunk_size: int,
) -> int:
    V = int(lm_head_weight.size(0))
    best_val = None
    best_idx = 0
    for start in range(0, V, chunk_size):
        end = min(V, start + chunk_size)
        w = lm_head_weight[start:end].to(dtype=torch.float32)  # [C,D]
        logits = (w @ h) * logits_scale  # [C]
        val, idx = torch.max(logits, dim=0)
        if best_val is None or float(val) > float(best_val):
            best_val = val
            best_idx = start + int(idx.item())
    return int(best_idx)


@torch.no_grad()
def _topk_vocab_chunked(
    h: torch.Tensor,  # [D] fp32
    lm_head_weight: torch.Tensor,  # [V,D]
    *,
    logits_scale: float,
    k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    V = int(lm_head_weight.size(0))
    k = max(1, int(k))
    best_vals = None
    best_idx = None
    for start in range(0, V, chunk_size):
        end = min(V, start + chunk_size)
        w = lm_head_weight[start:end].to(dtype=torch.float32)  # [C,D]
        logits = (w @ h) * logits_scale  # [C]
        kk = min(k, int(logits.numel()))
        vals, idx = torch.topk(logits, k=kk)
        idx = idx + start
        if best_vals is None:
            best_vals, best_idx = vals, idx
        else:
            cat_vals = torch.cat([best_vals, vals], dim=0)
            cat_idx = torch.cat([best_idx, idx], dim=0)
            kk = min(k, int(cat_vals.numel()))
            vals2, sel = torch.topk(cat_vals, k=kk)
            best_vals = vals2
            best_idx = cat_idx.index_select(0, sel)
    assert best_vals is not None and best_idx is not None
    return best_vals, best_idx


@torch.no_grad()
def _generate_lockstep(
    model: torch.nn.Module,
    enc: tiktoken.Encoding,
    prompt: str,
    *,
    pad_to: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    pad_token_id: int,
) -> str:
    """Generate `max_new_tokens` tokens with explicit prompt length handling.

    All ranks must call this with identical `pad_to` and `max_new_tokens` to
    preserve distributed lockstep.
    """
    device = next(model.parameters()).device
    prompt_ids = enc.encode(prompt)
    if not prompt_ids:
        prompt_ids = [pad_token_id]
    if len(prompt_ids) >= pad_to:
        prompt_ids = prompt_ids[: pad_to - 1]

    tokens = torch.full((1, pad_to), pad_token_id, device=device, dtype=torch.long)
    plen = len(prompt_ids)
    tokens[0, :plen] = torch.tensor(prompt_ids, device=device, dtype=torch.long)
    cur_len = plen

    generated: list[int] = []
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise TypeError("model must expose lm_head.weight")
    W = lm_head.weight.detach()
    logits_scale = float(getattr(model, "logits_scale_factor", 1.0))
    vocab_chunk = 4096
    for _ in range(int(max_new_tokens)):
        hidden = _forward_hidden(model, tokens)  # [1,T,D]
        h = hidden[0, cur_len - 1].to(dtype=torch.float32)  # [D]
        if temperature <= 0.0 or int(top_k) <= 1:
            next_id = _argmax_vocab_chunked(h, W, logits_scale=logits_scale, chunk_size=vocab_chunk)
        else:
            vals, idx = _topk_vocab_chunked(
                h, W, logits_scale=logits_scale, k=int(top_k), chunk_size=vocab_chunk
            )
            scaled = vals / float(temperature)
            probs = torch.softmax(scaled, dim=-1)
            choice = int(torch.multinomial(probs, num_samples=1).item())
            next_id = int(idx[choice].item())
        if cur_len < pad_to:
            tokens[0, cur_len] = int(next_id)
        generated.append(int(next_id))
        cur_len = min(cur_len + 1, pad_to)

    return enc.decode(generated)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Run generation-based evaluations")
    p.add_argument("--list-tasks", action="store_true", help="List tasks and exit")
    p.add_argument("--task", type=str, default="", help="Task name (use --list-tasks)")
    p.add_argument("--config", type=str, default="", help="Training config TOML (for model architecture + tokenizer)")
    p.add_argument("--checkpoint", type=str, default="", help="Optional eval snapshot (.pt) with config+model_state")
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="", help="Optional output directory for summary.json")
    args = p.parse_args(argv)

    if args.list_tasks:
        for name in sorted(TASKS.keys()):
            print(name)
        return

    if not args.task:
        raise SystemExit("--task is required (or use --list-tasks)")
    if args.task not in TASKS:
        raise SystemExit(f"unknown task '{args.task}' (use --list-tasks)")

    rank, world = _init_dist()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for nmoe.eval.run_gen (B200-targeted); run in the training container")

    # Load cfg/model
    cfg = None
    state = None
    if args.checkpoint:
        cfg_dict, state = _load_snapshot(args.checkpoint)
        from nmoe.config import Config
        cfg = Config(**cfg_dict)
    elif args.config:
        cfg = _load_config(args.config)
    else:
        raise SystemExit("must pass --checkpoint (eval snapshot) or --config (training config TOML)")

    from nmoe.model import Transformer

    model = Transformer(cfg).cuda()
    model.init_weights()
    if state is not None:
        model.load_state_dict(state, strict=False)
    model.eval()

    enc = tiktoken.get_encoding(str(getattr(cfg, "tokenizer", "o200k_harmony")))
    pad_token_id = int(getattr(cfg, "eos_token_id", 0))

    task = TASKS[args.task]()
    max_problems = args.max_problems
    n = len(task) if max_problems is None else min(int(max_problems), len(task))

    # Shared padding length across all ranks for lockstep.
    prompts = [task.format_prompt(task[i]) for i in range(n)]
    max_prompt = max(len(enc.encode(p)) for p in prompts) if prompts else 1
    pad_to = int(((max_prompt + int(args.max_new_tokens) + 127) // 128) * 128)

    my_idx = list(range(rank, n, world))
    max_per_rank = (n + world - 1) // world

    metrics_ctx = start_metrics(run_id=os.getenv("NMOE_RUN", "eval"), metrics_dir=getattr(cfg, "metrics_dir", None))
    passed = 0
    total = 0
    dummy_ex = task[0] if n > 0 else {}

    try:
        for it in range(max_per_rank):
            if it < len(my_idx):
                ex = task[my_idx[it]]
                prompt = task.format_prompt(ex)
                real = True
            else:
                ex = dummy_ex
                prompt = task.format_prompt(ex) if n > 0 else ""
                real = False

            completion = _generate_lockstep(
                model,
                enc,
                prompt,
                pad_to=pad_to,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                pad_token_id=pad_token_id,
            )

            if real:
                ok = bool(task.evaluate(ex, completion))
                passed += int(ok)
                total += 1
                if rank == 0:
                    _print0(rank, f"\r{passed}/{total} ({100.0*passed/max(1,total):.2f}%)", end="", flush=True)

        if rank == 0:
            print()

        if world > 1:
            device = next(model.parameters()).device
            t = torch.tensor([passed, total], device=device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            passed, total = int(t[0].item()), int(t[1].item())

        acc = (passed / total) if total > 0 else 0.0
        _print0(rank, f"[eval/{args.task}] {passed}/{total} ({100.0*acc:.2f}%)")
        if metrics_ctx.writer is not None and rank == 0:
            metrics_ctx.writer.insert_many(
                step=0,
                items=[
                    (f"eval_gen/{args.task}/acc", float(acc)),
                    (f"eval_gen/{args.task}/passed", float(passed)),
                    (f"eval_gen/{args.task}/total", float(total)),
                ],
            )

        if rank == 0:
            base = Path(args.out_dir) if args.out_dir else Path(str(getattr(cfg, "metrics_dir", "/data/metrics")))
            out = base / "eval_gen" / str(os.getenv("NMOE_RUN", "eval")) / args.task
            out.mkdir(parents=True, exist_ok=True)
            (out / "summary.json").write_text(
                json.dumps(
                    {
                        "task": args.task,
                        "acc": acc,
                        "passed": passed,
                        "total": total,
                        "config": asdict(cfg),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        stop_metrics(metrics_ctx)


if __name__ == "__main__":
    main()
