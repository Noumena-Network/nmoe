from __future__ import annotations

"""
Choice-based evaluation (MMLU, ARC, HellaSwag, ...).

Design constraints (Issue [07]):
- Must not materialize (seq, vocab) log_softmax/logits for vocab≈200k.
- Must keep all ranks in lockstep by construction (no barriers in inner loops).
"""

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.distributed as dist
import tiktoken

from nmoe.eval.adapters import iter_choices


BASELINES: dict[str, float] = {
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "HellaSwag": 0.25,
    "MMLU": 0.25,
    "MMLU-pro": 0.10,  # 10 choices
    "OpenBookQA": 0.25,
    "WinoGrande": 0.5,
    "BoolQ": 0.5,
    "COPA": 0.5,
}

DEFAULT_TASKS: list[tuple[str, str]] = [
    ("MMLU", "hf:cais/mmlu:all:dev"),
    ("MMLU-pro", "hf:TIGER-Lab/MMLU-Pro:test"),
]

FULL_TASKS: list[tuple[str, str]] = DEFAULT_TASKS + [
    ("ARC-Easy", "hf:allenai/ai2_arc:ARC-Easy:test"),
    ("ARC-Challenge", "hf:allenai/ai2_arc:ARC-Challenge:test"),
    ("HellaSwag", "hf:Rowan/hellaswag:validation"),
    ("OpenBookQA", "hf:allenai/openbookqa:main:test"),
    ("WinoGrande", "hf:allenai/winogrande:winogrande_debiased:validation"),
    ("BoolQ", "hf:google/boolq:validation"),
    ("COPA", "hf:super_glue:copa:validation"),
]


@dataclass(frozen=True)
class ChoiceEvalStats:
    correct: int
    total: int


def _centered_acc(acc: float, baseline: float) -> float:
    if baseline >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (acc - baseline) / (1.0 - baseline)))


def _round_up(x: int, multiple: int) -> int:
    return int(((x + multiple - 1) // multiple) * multiple)


@torch.no_grad()
def _forward_hidden(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Forward model up to final norm (no lm_head), returning hidden [B,T,D]."""
    # This replicates nmoe.model.Transformer.forward() up to norm_f, without lm_head.
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
def _logsumexp_vocab_chunked(
    h: torch.Tensor,  # [N,D], float32
    lm_head_weight: torch.Tensor,  # [V,D], bf16/fp16/fp32
    *,
    logits_scale: float,
    chunk_size: int,
) -> torch.Tensor:
    """Compute logsumexp over vocab for each row of h without materializing (N,V)."""
    if h.dim() != 2 or lm_head_weight.dim() != 2:
        raise ValueError("expected h=[N,D], lm_head_weight=[V,D]")
    if h.size(1) != lm_head_weight.size(1):
        raise ValueError("hidden dim mismatch with lm_head weight")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    V = int(lm_head_weight.size(0))
    out = None
    for start in range(0, V, chunk_size):
        end = min(V, start + chunk_size)
        w = lm_head_weight[start:end].to(dtype=torch.float32)  # [C,D]
        logits = (h @ w.t()) * logits_scale  # [N,C] fp32
        lse = torch.logsumexp(logits, dim=-1)  # [N]
        out = lse if out is None else torch.logaddexp(out, lse)
    assert out is not None
    return out


@torch.no_grad()
def _target_logits(
    h: torch.Tensor,  # [N,D], float32
    lm_head_weight: torch.Tensor,  # [V,D], bf16/fp16/fp32
    target_ids: torch.Tensor,  # [N], int64
    *,
    logits_scale: float,
) -> torch.Tensor:
    w = lm_head_weight.index_select(0, target_ids).to(dtype=torch.float32)  # [N,D]
    return (h * w).sum(dim=-1) * logits_scale  # [N]


@torch.no_grad()
def _score_option_logprob(
    model: torch.nn.Module,
    enc: tiktoken.Encoding,
    prompt: str,
    option: str,
    *,
    pad_to: int,
    pad_token_id: int,
    vocab_chunk_size: int,
    compute_score: bool = True,
) -> float:
    """Return sum log p(option_tokens | prompt) for a single option."""
    prompt_ids = enc.encode(prompt)
    opt_ids = enc.encode(" " + option) if option else [pad_token_id]
    if len(prompt_ids) < 1 or len(opt_ids) < 1:
        return float("-inf")

    # Full sequence used for causal next-token prediction.
    full = prompt_ids + opt_ids
    if len(full) > pad_to:
        full = full[:pad_to]
        opt_ids = opt_ids[: max(0, pad_to - len(prompt_ids))]
    if len(full) < pad_to:
        full = full + [pad_token_id] * (pad_to - len(full))

    device = next(model.parameters()).device
    tokens = torch.tensor(full, device=device, dtype=torch.long).unsqueeze(0)  # [1,T]
    hidden = _forward_hidden(model, tokens)[0]  # [T,D]

    if not compute_score:
        return float("-inf")

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise TypeError("model must expose lm_head.weight")

    logits_scale = float(getattr(model, "logits_scale_factor", 1.0))
    W = lm_head.weight.detach()  # [V,D]

    # Positions predicting option tokens: pos = prompt_len + j - 1
    prompt_len = len(prompt_ids)
    opt_len = len(opt_ids)
    if opt_len <= 0 or prompt_len <= 0:
        return float("-inf")
    pos0 = prompt_len - 1
    pos = torch.arange(pos0, pos0 + opt_len, device=device, dtype=torch.long)
    # If truncated, guard
    pos = pos[pos < pad_to]
    if pos.numel() == 0:
        return float("-inf")

    target = torch.tensor(opt_ids[: pos.numel()], device=device, dtype=torch.long)
    h = hidden.index_select(0, pos).to(dtype=torch.float32)  # [N,D]

    lse = _logsumexp_vocab_chunked(h, W, logits_scale=logits_scale, chunk_size=vocab_chunk_size)
    tgt = _target_logits(h, W, target, logits_scale=logits_scale)
    return float((tgt - lse).sum().item())


def _all_reduce_sum_i64(x: int, device: torch.device) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return int(x)
    t = torch.tensor([int(x)], device=device, dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


@torch.no_grad()
def eval_task(
    model: torch.nn.Module,
    enc: tiktoken.Encoding,
    task_name: str,
    source: str,
    *,
    max_examples: int,
    rank: int,
    world: int,
    vocab_chunk_size: int,
) -> dict[str, float]:
    model_was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    examples = list(iter_choices(task_name, source, max_examples))
    if not examples:
        raise RuntimeError(f"no examples produced for task='{task_name}' source='{source}'")

    # Compute padding targets from the shared example list (deterministic across ranks).
    max_options = 1
    max_seq = 2
    for ex in examples:
        prompt_ids = enc.encode(ex["prompt"])
        max_options = max(max_options, len(ex["options"]))
        for opt in ex["options"]:
            opt_ids = enc.encode(" " + str(opt))
            max_seq = max(max_seq, len(prompt_ids) + len(opt_ids))
    pad_to = _round_up(max_seq, 128)

    max_per_rank = (len(examples) + world - 1) // world
    my = examples[rank::world]
    num_real = len(my)

    pad_token_id = int(getattr(getattr(model, "config", None), "eos_token_id", 0))

    correct = 0
    total = 0
    dummy = examples[0]

    for i in range(max_per_rank):
        ex = my[i] if i < num_real else dummy
        prompt = str(ex["prompt"])
        options = [str(o) for o in ex["options"]]
        label = int(ex["label"])

        # Ensure fixed number of forwards per-example across ranks by padding options.
        best_idx = 0
        best_score = -math.inf
        for j in range(max_options):
            is_real_opt = j < len(options)
            opt = options[j] if is_real_opt else ""
            score = _score_option_logprob(
                model,
                enc,
                prompt,
                opt,
                pad_to=pad_to,
                pad_token_id=pad_token_id,
                vocab_chunk_size=vocab_chunk_size,
                compute_score=is_real_opt,
            )
            if score > best_score:
                best_score = score
                best_idx = j

        if i < num_real:
            correct += int(best_idx == label)
            total += 1

    correct = _all_reduce_sum_i64(correct, device)
    total = _all_reduce_sum_i64(total, device)
    acc = (correct / total) if total > 0 else 0.0
    centered = _centered_acc(acc, BASELINES.get(task_name, 0.25))

    if model_was_training:
        model.train()
    return {"acc": float(acc), "centered_acc": float(centered), "n": float(total)}


def run_eval(
    model: torch.nn.Module,
    cfg,
    rank: int,
    world: int,
    *,
    max_examples: int = 500,
    tasks: Optional[list[tuple[str, str]]] = None,
    vocab_chunk_size: int = 4096,
) -> dict[str, dict[str, float]]:
    """Run choice-based eval; all ranks must call together when distributed."""
    if world > 1 and not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("world>1 requires torch.distributed to be initialized (run under torchrun)")

    tok_name = str(getattr(cfg, "tokenizer", "o200k_harmony"))
    enc = tiktoken.get_encoding(tok_name)
    chosen = DEFAULT_TASKS if tasks is None else tasks

    out: dict[str, dict[str, float]] = {}
    for task_name, source in chosen:
        try:
            out[task_name] = eval_task(
                model,
                enc,
                task_name,
                source,
                max_examples=int(max_examples),
                rank=int(rank),
                world=int(world),
                vocab_chunk_size=int(vocab_chunk_size),
            )
        except Exception as e:
            out[task_name] = {"acc": 0.0, "centered_acc": 0.0, "n": 0.0, "error": str(e)}
            if rank == 0:
                print(f"[eval/choices] task '{task_name}' failed: {e}")
    return out


def format_results(results: dict[str, dict[str, float]]) -> str:
    parts = []
    for name, r in results.items():
        if "error" in r:
            parts.append(f"{name}: ERROR")
            continue
        acc = float(r.get("acc", 0.0))
        centered = float(r.get("centered_acc", 0.0))
        n = int(r.get("n", 0.0))
        parts.append(f"{name}: acc={acc:.3f} centered={centered:.3f} n={n}")
    return " | ".join(parts)


def _parse_tasks_arg(tasks: str) -> list[tuple[str, str]]:
    if tasks.lower() in ("default", "defaults"):
        return list(DEFAULT_TASKS)
    if tasks.lower() in ("full", "all"):
        return list(FULL_TASKS)
    out: list[tuple[str, str]] = []
    for item in tasks.split(","):
        item = item.strip()
        if not item:
            continue
        matches = [t for t in FULL_TASKS if t[0].lower() == item.lower()]
        if not matches:
            raise ValueError(f"unknown task '{item}' (use --list-tasks)")
        out.append(matches[0])
    if not out:
        raise ValueError("no tasks selected")
    return out


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Choice-based evaluation (fast forward-only scoring)")
    p.add_argument("--list-tasks", action="store_true", help="Print available task names and exit")
    p.add_argument("--tasks", type=str, default="default", help="default|full|comma-separated task names")
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--vocab-chunk-size", type=int, default=4096)
    p.add_argument("--snapshot", type=str, default="", help="Path to eval snapshot (.pt) with config+model_state")
    args = p.parse_args(argv)

    if args.list_tasks:
        for name, src in FULL_TASKS:
            print(f"{name}\t{src}")
        return

    if not args.snapshot:
        raise SystemExit("--snapshot is required for CLI mode")

    import torch.distributed as dist  # local import to keep module import light
    if "RANK" in __import__("os").environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1

    obj = torch.load(args.snapshot, map_location="cpu")
    cfg_dict = obj.get("config", {})
    from nmoe.config import Config
    from nmoe.model import Transformer

    cfg = Config(**cfg_dict)
    model = Transformer(cfg).cuda()
    model.load_state_dict(obj.get("model_state", {}), strict=False)

    tasks = _parse_tasks_arg(args.tasks)
    res = run_eval(
        model,
        cfg,
        rank,
        world,
        max_examples=args.max_examples,
        tasks=tasks,
        vocab_chunk_size=args.vocab_chunk_size,
    )
    if rank == 0:
        print(format_results(res))


if __name__ == "__main__":
    main()
