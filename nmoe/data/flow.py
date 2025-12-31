"""
TOML-driven dataflow runner (single orchestrator path).

Design:
- One explicit entrypoint: `python -m nmoe.data.flow run --config ...`
- K8s Indexed-compatible: shard index is derived from an env var (default: JOB_COMPLETION_INDEX)
- Deterministic re-run: writes a `flow_spec.json` and refuses to reuse an output dir with mismatched spec

This is intentionally small and delegates heavy work to existing modules:
- HYDRA: `nmoe.data.hydra.cmd_grade`
- Rephrase: `nmoe.data.cli.cmd_rephrase`
- Tokenize/pack: `nmoe.data.cli.cmd_prep`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from nmoe.config import load_toml


def _read_toml(path: Path) -> Dict[str, Any]:
    obj = load_toml(path)
    if not isinstance(obj, dict):
        raise ValueError("invalid TOML root (expected table)")
    return obj


def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def _set_nested(obj: Dict[str, Any], keys: list[str], value: Any) -> None:
    cur: Dict[str, Any] = obj
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def apply_env_overrides(cfg: Dict[str, Any], *, prefix: str = "NMOE_DATAFLOW__") -> Dict[str, Any]:
    """Apply env overrides into a TOML dict.

    Format:
      NMOE_DATAFLOW__SECTION__KEY=value
      NMOE_DATAFLOW__section__nested__key=value
    """
    out = json.loads(json.dumps(cfg))  # deep copy with basic types
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        path = k[len(prefix) :].strip("_")
        if not path:
            continue
        keys = [p.lower() for p in path.split("__") if p]
        _set_nested(out, keys, _parse_scalar(v))
    return out


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class FlowSpec:
    cfg_path: str
    cfg: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {"config_path": self.cfg_path, "config": self.cfg}


def _require_table(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    v = cfg.get(name)
    if not isinstance(v, dict):
        raise ValueError(f"missing or invalid [{name}] table")
    return v


def _resolve_shard(run_cfg: Dict[str, Any]) -> tuple[int, int]:
    num_shards = int(run_cfg.get("num_shards", 1))
    if num_shards <= 0:
        raise ValueError(f"run.num_shards must be > 0 (got {num_shards})")
    env = str(run_cfg.get("shard_index_env", "JOB_COMPLETION_INDEX"))
    shard_index = int(os.environ.get(env, "0"))
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index out of range: {shard_index} (num_shards={num_shards}, env={env})")
    return shard_index, num_shards


def _stage_paths(output_dir: Path) -> Dict[str, Path]:
    return {
        "spec": output_dir / "flow_spec.json",
        "raw": output_dir / "raw_docs.jsonl",
        "grades_dir": output_dir / "hydra_grades",
        "summary": output_dir / "hydra_grades" / "summary.json",
        "scores": output_dir / "hydra_grades" / "quality_scores.jsonl",
        "kept": output_dir / "kept_docs.jsonl",
        "rephrased": output_dir / "rephrased_docs.jsonl",
        "shards_dir": output_dir / "training_shards",
    }


def _ensure_flow_spec(paths: Dict[str, Path], spec: FlowSpec) -> None:
    spec_path = paths["spec"]
    if spec_path.exists():
        existing = json.loads(spec_path.read_text(encoding="utf-8"))
        if existing != spec.to_json():
            raise RuntimeError(
                "Output directory contains a different flow_spec.json; "
                "use a new output directory or delete the old one."
            )
        return
    _atomic_write_json(spec_path, spec.to_json())


def _write_raw_docs(*, src_cfg: Dict[str, Any], run_cfg: Dict[str, Any], out_path: Path, shard_index: int, num_shards: int) -> int:
    from .sources import HuggingFaceSource

    if out_path.exists():
        # Already materialized for this shard.
        n = 0
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    max_docs_global = int(run_cfg.get("max_docs_global", 0))
    if max_docs_global <= 0:
        raise ValueError("run.max_docs_global must be > 0 for this flow")
    max_docs_shard = (max_docs_global + num_shards - 1) // num_shards

    ds = HuggingFaceSource(
        dataset=str(src_cfg["dataset"]),
        split=str(src_cfg.get("split", "train")),
        subset=str(src_cfg.get("subset")) if src_cfg.get("subset") else None,
        text_field=str(src_cfg.get("text_field", "text")),
        id_field=str(src_cfg.get("id_field")) if src_cfg.get("id_field") else None,
        streaming=bool(src_cfg.get("streaming", True)),
    ).shard(num_shards=num_shards, index=shard_index)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for doc in ds:
            if count >= max_docs_shard:
                break
            if not doc.text or not doc.text.strip():
                continue
            row = {"id": doc.doc_id, "text": doc.text}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp_path.replace(out_path)
    return count


def _run_hydra_grade(*, model_cfg: Dict[str, Any], hydra_cfg: Dict[str, Any], in_path: Path, out_dir: Path) -> None:
    from .hydra import cmd_grade
    import argparse as _ap

    summary = out_dir / "summary.json"
    scores = out_dir / "quality_scores.jsonl"
    if summary.exists() and scores.exists():
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    args = _ap.Namespace(
        input_jsonl=str(in_path),
        input_docids=None,
        checkpoint=str(model_cfg["checkpoint"]),
        heads_dir=str(hydra_cfg["heads_dir"]),
        calibration=str(hydra_cfg["calibration_dir"]),
        out=str(out_dir),
        max_ctx=int(hydra_cfg.get("max_ctx", 4096)),
        max_batch=int(hydra_cfg.get("max_batch", 16)),
    )
    rc = cmd_grade(args)
    if rc != 0:
        raise RuntimeError(f"HYDRA grade failed (rc={rc})")


def _select_kept(*, scores_path: Path, raw_path: Path, out_path: Path) -> int:
    if out_path.exists():
        n = 0
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    keep: set[str] = set()
    with scores_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            if j.get("decision") == "keep":
                keep.add(str(j["doc_id"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    kept = 0
    with raw_path.open("r", encoding="utf-8") as in_f, tmp_path.open("w", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            j = json.loads(line)
            did = str(j.get("id", ""))
            if did and did in keep:
                out_f.write(json.dumps({"id": did, "text": j["text"]}, ensure_ascii=False) + "\n")
                kept += 1
    tmp_path.replace(out_path)
    return kept


def _run_rephrase(*, rephrase_cfg: Dict[str, Any], hydra_cfg: Dict[str, Any], model_cfg: Dict[str, Any], in_path: Path, out_path: Path) -> int:
    if not bool(rephrase_cfg.get("enabled", True)):
        return 0
    if out_path.exists():
        n = 0
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    from .cli import cmd_rephrase
    import argparse as _ap

    args = _ap.Namespace(
        input=str(in_path),
        output=str(out_path),
        checkpoint=str(model_cfg["checkpoint"]),
        quality_scores=None,
        quality_threshold=0.0,
        num_versions=int(rephrase_cfg.get("num_versions", 10)),
        chunk_size=int(rephrase_cfg.get("chunk_size", 2048)),
        no_style_diversity=not bool(rephrase_cfg.get("style_diversity", True)),
        verify_fidelity=False,
        fidelity_threshold=0.85,
        max_ctx=int(rephrase_cfg.get("max_ctx", 8192)),
        max_batch=int(rephrase_cfg.get("max_batch", 32)),
        batch_size=int(rephrase_cfg.get("batch_size", 8)),
        max_new=int(rephrase_cfg.get("max_new", 4096)),
        temperature=float(rephrase_cfg.get("temperature", 0.8)),
        top_p=float(rephrase_cfg.get("top_p", 0.9)),
        max_docs=None,
        hydra_filter=bool(rephrase_cfg.get("hydra_filter", False)),
        hydra_heads_dir=str(hydra_cfg["heads_dir"]),
        hydra_calibration=str(hydra_cfg["calibration_dir"]),
        hydra_max_ctx=int(rephrase_cfg.get("hydra_max_ctx", 4096)),
        hydra_batch_size=int(rephrase_cfg.get("hydra_batch_size", 16)),
        hydra_tau_keep=None,
    )
    rc = cmd_rephrase(args)
    if rc != 0:
        raise RuntimeError(f"rephrase failed (rc={rc})")

    # Count rows written
    n = 0
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _run_prep(*, prep_cfg: Dict[str, Any], run_cfg: Dict[str, Any], in_path: Path, out_dir: Path) -> None:
    if not bool(prep_cfg.get("enabled", True)):
        return

    manifest = out_dir / "manifest.json"
    if manifest.exists():
        return

    from .cli import cmd_prep
    import argparse as _ap

    args = _ap.Namespace(
        source="jsonl",
        paths=[str(in_path)],
        output=str(out_dir),
        name=str(run_cfg["name"]),
        version="v1",
        tokenizer=str(prep_cfg.get("tokenizer", "o200k_harmony")),
        vocab_size=int(prep_cfg.get("vocab_size", 201088)),
        eos_token_id=int(prep_cfg.get("eos_token_id", 199999)),
        num_shards=int(prep_cfg.get("num_shards", 1)),
        tokens_per_shard=int(prep_cfg.get("tokens_per_shard", 500_000_000)),
        workers=int(prep_cfg.get("workers", 8)),
        batch_size=int(prep_cfg.get("batch_size", 1000)),
        text_field="text",
        min_tokens=int(prep_cfg.get("min_tokens", 10)),
        max_tokens=int(prep_cfg.get("max_tokens", 1_000_000)),
        parallel=bool(prep_cfg.get("parallel", False)),
        dataset=None,
        split=None,
        subset=None,
    )
    rc = cmd_prep(args)
    if rc != 0:
        raise RuntimeError(f"prep failed (rc={rc})")


def cmd_run(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    cfg = _read_toml(cfg_path)
    cfg = apply_env_overrides(cfg)

    run_cfg = _require_table(cfg, "run")
    src_cfg = _require_table(cfg, "source")
    model_cfg = _require_table(cfg, "model")
    hydra_cfg = _require_table(cfg, "hydra")
    rephrase_cfg = _require_table(cfg, "rephrase")
    prep_cfg = _require_table(cfg, "prep")

    shard_index, num_shards = _resolve_shard(run_cfg)
    output_root = Path(str(run_cfg["output_root"]))
    output_dir = output_root / f"shard_{shard_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _stage_paths(output_dir)

    spec = FlowSpec(cfg_path=str(cfg_path), cfg=cfg)
    _ensure_flow_spec(paths, spec)

    # Stage 1: materialize raw docs for this shard
    n_raw = _write_raw_docs(
        src_cfg=src_cfg,
        run_cfg=run_cfg,
        out_path=paths["raw"],
        shard_index=shard_index,
        num_shards=num_shards,
    )
    if args.verbose:
        print(f"[raw] {n_raw} docs → {paths['raw']}")

    # Stage 2: HYDRA grade
    _run_hydra_grade(model_cfg=model_cfg, hydra_cfg=hydra_cfg, in_path=paths["raw"], out_dir=paths["grades_dir"])
    if args.verbose:
        print(f"[hydra] grades → {paths['grades_dir']}")

    # Stage 3: select kept
    n_kept = _select_kept(scores_path=paths["scores"], raw_path=paths["raw"], out_path=paths["kept"])
    if args.verbose:
        print(f"[keep] {n_kept} docs → {paths['kept']}")

    # Stage 4: rephrase
    n_reph = 0
    if n_kept:
        n_reph = _run_rephrase(
            rephrase_cfg=rephrase_cfg,
            hydra_cfg=hydra_cfg,
            model_cfg=model_cfg,
            in_path=paths["kept"],
            out_path=paths["rephrased"],
        )
    if args.verbose and bool(rephrase_cfg.get("enabled", True)):
        print(f"[rephrase] {n_reph} rows → {paths['rephrased']}")

    # Stage 5: prep
    if bool(prep_cfg.get("enabled", True)) and n_reph:
        _run_prep(prep_cfg=prep_cfg, run_cfg=run_cfg, in_path=paths["rephrased"], out_dir=paths["shards_dir"])
        if args.verbose:
            print(f"[prep] shards → {paths['shards_dir']}")

    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("nmoe.data.flow")
    sp = p.add_subparsers(dest="cmd", required=True)
    run = sp.add_parser("run", help="Run a TOML-defined dataflow (Indexed-shard compatible)")
    run.add_argument("--config", required=True, help="Path to dataflow TOML")
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
