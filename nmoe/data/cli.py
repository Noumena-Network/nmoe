"""
Command-line interface for nmoe data preprocessing.

Usage:
    # From HuggingFace
    python -m nmoe.data.cli \
        --source huggingface \
        --dataset HuggingFaceFW/fineweb-edu \
        --split train \
        --output /data/fineweb_edu \
        --name fineweb_edu

    # From JSONL files
    python -m nmoe.data.cli \
        --source jsonl \
        --paths /data/raw/*.jsonl.gz \
        --output /data/processed \
        --name my_dataset

    # Verify manifest
    python -m nmoe.data.cli verify /data/fineweb_edu/manifest.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Any, Dict

from nmoe.config import load_toml

# arXiv processing (lazy import to avoid dependency issues)
def _get_arxiv():
    from .arxiv import (
        ArxivS3Source,
        Ar5ivHTMLSource,
        ArxivMultiFormatSource,
        download_s3_tars,
        download_ar5iv_dataset,
    )
    return ArxivS3Source, Ar5ivHTMLSource, ArxivMultiFormatSource, download_s3_tars, download_ar5iv_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

def _read_toml(path: Path) -> Dict[str, Any]:
    obj = load_toml(path)
    if not isinstance(obj, dict):
        raise ValueError("invalid TOML root (expected table)")
    return obj


def _parse_tokens(s: str) -> int:
    """Parse token count string like '100M', '1B', '1.5T'."""
    s = s.strip().upper()
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000, 'T': 1_000_000_000_000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def _autodetect_flow_name(*, mixture_toml: Path, flow_profiles_toml: Path) -> str:
    """Infer a flow name from a mixture TOML and a flow-profiles TOML.

    This is used to keep `prep-mixture --config <mixture>` ergonomic for the
    common case where there is exactly one matching flow entry.
    """
    mix = _read_toml(mixture_toml)
    mixtures = mix.get("mixtures")
    if not isinstance(mixtures, dict) or not mixtures:
        raise ValueError(f"{mixture_toml} is missing a [mixtures] table")
    mixture_ids = set(mixtures.keys())

    fp = _read_toml(flow_profiles_toml)
    flows = fp.get("flow")
    if not isinstance(flows, dict) or not flows:
        raise ValueError(f"{flow_profiles_toml} is missing a [flow] table")

    candidates: list[str] = []
    for flow_name, flow_cfg in flows.items():
        if not isinstance(flow_cfg, dict):
            continue
        ref = flow_cfg.get("mixture")
        if not isinstance(ref, dict):
            continue
        if any(m in mixture_ids for m in ref.values()):
            candidates.append(str(flow_name))

    if not candidates:
        raise ValueError(
            f"Could not infer flow for {mixture_toml} from {flow_profiles_toml}; "
            "pass --flow explicitly."
        )
    if len(candidates) == 1:
        return candidates[0]

    # Prefer a flow whose name matches the mixture file stem.
    for c in candidates:
        if c == mixture_toml.stem:
            return c

    raise ValueError(
        "Multiple flows match this mixture; pass --flow explicitly. "
        f"Candidates: {', '.join(sorted(candidates))}"
    )


def cmd_prep(args: argparse.Namespace) -> int:
    """Run preprocessing pipeline."""
    from .sources import (
        ArrowSource,
        HfFileSystemSource,
        HuggingFaceSource,
        HydraFilteredSource,
        HydraScoringSource,
        JSONLSource,
        JSONLZstSource,
        create_source,
    )
    try:
        from .prep import PrepConfig, PrepPipeline, ParallelPrepPipeline
    except Exception as e:
        log.error(f"prep requires numpy (and other deps); run in the container image. ({e})")
        return 1

    # Validate HYDRA args if enabled
    if args.hydra_filter or args.hydra_audit_dir:
        if not args.checkpoint:
            log.error("--checkpoint required with --hydra-filter or --hydra-audit-dir")
            return 1
        if not args.heads_dir:
            log.error("--heads-dir required with --hydra-filter or --hydra-audit-dir")
            return 1
        if not args.calibration:
            log.error("--calibration required with --hydra-filter or --hydra-audit-dir")
            return 1

    # Create base source
    if args.source == "huggingface" or args.source == "hf":
        base_source = HuggingFaceSource(
            dataset=args.dataset,
            split=args.split,
            text_field=args.text_field,
            subset=args.subset,
            streaming=True,
        )
        # HF sharding via IterableDataset.shard()
        if args.num_workers > 1:
            base_source = base_source.shard(num_shards=args.num_workers, index=args.worker_index)
    elif args.source == "jsonl":
        if not args.paths:
            log.error("--paths required for jsonl source")
            return 1
        import glob as glob_mod
        paths = []
        for pattern in args.paths:
            paths.extend(sorted(glob_mod.glob(pattern)))
        if not paths:
            log.error(f"No files found matching: {args.paths}")
            return 1
        # File-level sharding for K8s indexed jobs
        if args.num_workers > 1:
            paths = [p for i, p in enumerate(paths) if i % args.num_workers == args.worker_index]
            log.info(f"Worker {args.worker_index + 1}/{args.num_workers}: processing {len(paths)} files")
        base_source = JSONLSource(paths=paths, text_field=args.text_field)
    elif args.source == "jsonl_zst":
        if not args.paths:
            log.error("--paths required for jsonl_zst source")
            return 1
        import glob as glob_mod
        paths = []
        for pattern in args.paths:
            paths.extend(sorted(glob_mod.glob(pattern)))
        if not paths:
            log.error(f"No files found matching: {args.paths}")
            return 1
        # File-level sharding for K8s indexed jobs
        if args.num_workers > 1:
            paths = [p for i, p in enumerate(paths) if i % args.num_workers == args.worker_index]
            log.info(f"Worker {args.worker_index + 1}/{args.num_workers}: processing {len(paths)} files")
        base_source = JSONLZstSource(paths=paths, text_field=args.text_field, id_field=args.id_field)
    elif args.source == "arrow" or args.source == "parquet":
        if not args.paths:
            log.error("--paths required for arrow/parquet source")
            return 1
        import glob as glob_mod
        paths = []
        for pattern in args.paths:
            paths.extend(sorted(glob_mod.glob(pattern)))
        if not paths:
            log.error(f"No files found matching: {args.paths}")
            return 1
        # File-level sharding for K8s indexed jobs
        if args.num_workers > 1:
            total_files = len(paths)
            paths = [p for i, p in enumerate(paths) if i % args.num_workers == args.worker_index]
            log.info(f"Worker {args.worker_index + 1}/{args.num_workers}: processing {len(paths)}/{total_files} files")
        base_source = ArrowSource(paths=paths, text_field=args.text_field, id_field=args.id_field)
    else:
        log.error(f"Unknown source type: {args.source}")
        return 1

    # Wrap with HYDRA if requested
    source = base_source
    hydra_model = None
    if args.hydra_filter or args.hydra_audit_dir:
        log.info("Loading HYDRA model and judge...")
        try:
            import torch
        except Exception as e:
            log.error(f"HYDRA filtering requires torch; run in the container image with deps installed. ({e})")
            return 1
        from .model import Transformer
        from .hydra import load_local_hydra_judge, _load_calibration

        if not torch.cuda.is_available():
            log.error("HYDRA filtering requires CUDA; run in the container on a GPU node.")
            return 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hydra_model = Transformer.from_checkpoint(args.checkpoint, device=device)
        for p in hydra_model.parameters():
            p.requires_grad = False
        hydra_model.eval()

        hydra_judge = load_local_hydra_judge(
            model=hydra_model,
            heads_dir=args.heads_dir,
            device=device,
        )
        hydra_w, hydra_tau_drop, hydra_tau_keep = _load_calibration(args.calibration)
        log.info(f"  HYDRA loaded: tau_drop={hydra_tau_drop:.2f}, tau_keep={hydra_tau_keep:.2f}")

        # Determine audit path (worker-specific if parallel)
        audit_path = None
        if args.hydra_audit_dir:
            audit_dir = Path(args.hydra_audit_dir)
            audit_dir.mkdir(parents=True, exist_ok=True)
            if args.num_workers > 1:
                audit_path = audit_dir / f"hydra_scores_worker_{args.worker_index:03d}.jsonl"
            else:
                audit_path = audit_dir / "hydra_scores.jsonl"
            log.info(f"  HYDRA audit: {audit_path}")

        source = HydraFilteredSource(
            source=base_source,
            model=hydra_model,
            judge=hydra_judge,
            w=hydra_w,
            tau_drop=hydra_tau_drop,
            tau_keep=hydra_tau_keep,
            batch_size=args.hydra_batch_size,
            max_ctx=args.hydra_max_ctx,
            audit_path=audit_path,
            device=device,
        )

    # Worker-specific output dir
    output_dir = Path(args.output)
    if args.num_workers > 1:
        output_dir = output_dir / f"worker_{args.worker_index:03d}"

    # Parse max_tokens_total if provided
    max_tokens_total = None
    if args.max_tokens_total:
        max_tokens_total = _parse_tokens(args.max_tokens_total)
        if args.num_workers > 1:
            max_tokens_total = max_tokens_total // args.num_workers
        log.info(f"Token limit: {max_tokens_total:,}")

    # Create config
    config = PrepConfig(
        output_dir=output_dir,
        dataset_name=args.name,
        version=args.version,
        tokenizer=args.tokenizer,
        vocab_size=args.vocab_size,
        eos_token_id=args.eos_token_id,
        num_shards=args.num_shards,
        tokens_per_shard=args.tokens_per_shard,
        num_workers=args.workers,
        batch_size=args.batch_size,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_tokens_total=max_tokens_total,
    )

    # Progress callback
    last_report = [0]
    token_limit = max_tokens_total  # capture for closure

    def progress(docs: int, tokens: int):
        if docs - last_report[0] >= 10000:
            if token_limit:
                log.info(f"Progress: {docs:,} docs, {tokens:,}/{token_limit:,} tokens")
            else:
                log.info(f"Progress: {docs:,} docs, {tokens:,} tokens")
            last_report[0] = docs

    # Run pipeline
    if args.parallel:
        pipeline = ParallelPrepPipeline(source, config)
    else:
        pipeline = PrepPipeline(source, config)

    try:
        manifest = pipeline.run(progress_callback=progress)
        log.info(f"Complete: {manifest.total_documents:,} docs, {manifest.total_tokens:,} tokens")
        log.info(f"Shards: {manifest.num_shards}")
        # Log HYDRA stats if enabled
        if (args.hydra_filter or args.hydra_audit_dir) and hasattr(source, 'stats'):
            stats = source.stats
            total = stats.get('total', 0)
            if total > 0:
                log.info(f"HYDRA: keep={stats['keep']:,} band={stats['band']:,} drop={stats['drop']:,} ({stats['keep']/total*100:.1f}% kept)")
        return 0
    except KeyboardInterrupt:
        log.warning("Interrupted")
        return 130
    except Exception as e:
        log.error(f"Failed: {e}")
        raise


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify manifest integrity."""
    from .sinks import verify_manifest

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error(f"Manifest not found: {manifest_path}")
        return 1

    log.info(f"Verifying: {manifest_path}")
    is_valid, errors = verify_manifest(manifest_path, check_checksums=args.checksums)

    if is_valid:
        log.info("Manifest verified successfully")
        return 0
    else:
        log.error(f"Verification failed with {len(errors)} errors:")
        for error in errors:
            log.error(f"  - {error}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show manifest info."""
    from .sinks import load_manifest

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error(f"Manifest not found: {manifest_path}")
        return 1

    manifest = load_manifest(manifest_path)

    print(f"Dataset:    {manifest.dataset}")
    print(f"Version:    {manifest.version}")
    print(f"Tokenizer:  {manifest.tokenizer}")
    print(f"Vocab Size: {manifest.vocab_size:,}")
    print(f"EOS Token:  {manifest.eos_token_id}")
    print(f"Dtype:      {manifest.dtype}")
    print(f"Created:    {manifest.created_at}")
    print()
    print(f"Total Tokens:    {manifest.total_tokens:,}")
    print(f"Total Documents: {manifest.total_documents:,}")
    print(f"Num Shards:      {manifest.num_shards}")
    print()
    print(f"Avg tokens/doc: {manifest.total_tokens / max(1, manifest.total_documents):.1f}")
    print(f"Avg tokens/shard: {manifest.total_tokens / max(1, manifest.num_shards):,.0f}")

    return 0


def cmd_regenerate_index(args: argparse.Namespace) -> int:
    """Regenerate index file from shard."""
    from .index import regenerate_index_from_shard

    shard_path = Path(args.shard)
    if not shard_path.exists():
        log.error(f"Shard not found: {shard_path}")
        return 1

    log.info(f"Regenerating index for: {shard_path}")
    idx_path = regenerate_index_from_shard(
        shard_path,
        eos_token_id=args.eos_token_id,
    )
    log.info(f"Wrote index: {idx_path}")
    return 0


def cmd_dedup(args: argparse.Namespace) -> int:
    """Near-duplicate deduplication for JSONL input (keep-first).

    Expects JSONL with a text field (default: 'text'). Writes JSONL of kept rows.
    """
    import json
    import gzip
    from .dedup import LSHIndex, minhash_signature, jaccard_from_signature

    in_path = Path(args.input)
    out_path = Path(args.output)
    text_field = args.text_field

    if not in_path.exists():
        log.error(f"Input file not found: {in_path}")
        return 1

    total = kept = dropped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outf = open(out_path, "w", encoding="utf-8")

    # LSH index for signatures
    idx = LSHIndex(args.perms)
    opener = gzip.open if str(in_path).endswith(".gz") else open
    with opener(in_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                row = {text_field: line}
            text = str(row.get(text_field, ""))
            if not text:
                continue

            sig = minhash_signature(
                text, shingle=args.shingle, num_perm=args.perms, seed=args.seed
            )
            is_dup = False
            for cand in idx.candidates(sig):
                if jaccard_from_signature(sig, idx.get(cand)) >= args.jaccard:
                    is_dup = True
                    break
            if is_dup:
                dropped += 1
                continue
            idx.add(sig)
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    outf.close()
    log.info(f"Dedup complete: total={total:,} kept={kept:,} dropped={dropped:,}")
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    """Cluster embeddings and write labels/centroids + summary JSON."""
    try:
        import numpy as np
        import torch
    except Exception as e:
        log.error(f"Missing dependency for clustering: {e}")
        return 1
    from .diversity import cluster_embeddings, coverage_metrics

    emb_path = Path(args.emb)
    if not emb_path.exists():
        log.error(f"Embeddings file not found: {emb_path}")
        return 1
    arr = np.load(emb_path)
    if arr.ndim != 2:
        log.error(f"Expected 2D embeddings, got shape {arr.shape}")
        return 1

    x = torch.from_numpy(arr).to(torch.float32)
    labels, centroids = cluster_embeddings(
        x, args.k, seed=args.seed, iters=args.iters, batch_size=args.batch
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.npy"
    cents_path = out_dir / "centroids.npy"
    np.save(labels_path, labels.numpy())
    np.save(cents_path, centroids.detach().cpu().numpy())

    metrics = coverage_metrics(labels, args.k)
    (out_dir / "cluster_summary.json").write_text(json.dumps(metrics, indent=2))
    log.info(
        f"Clustered N={arr.shape[0]:,} → k={args.k}; coverage={metrics['cluster_coverage']:.3f} entropy={metrics['entropy']:.3f}"
    )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect shard data for quality issues."""
    from .inspect import inspect_shards

    return inspect_shards(
        manifest_path=Path(args.manifest) if args.manifest else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        eos_token_id=args.eos_token_id,
        num_shards=args.num_shards,
        check_all=args.all,
        verbose=args.verbose,
        sample_size=args.sample if args.sample else None,
        compute_stats=args.stats,
    )


def cmd_rephrase(args: argparse.Namespace) -> int:
    """K2-style knowledge rephrasing with style diversity and fidelity verification."""
    try:
        import torch
        import tiktoken
    except Exception as e:
        log.error(f"Rephrase requires torch+tiktoken; run in the container image with deps installed. ({e})")
        return 1
    from .model import BatchedGenerator
    from .rephrase import get_encoding, rephrase_batch

    log = logging.getLogger(__name__)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(output_path.name + ".tmp")

    # Load quality scores if provided (for filtering)
    quality_filter = {}
    if args.quality_scores:
        log.info(f"Loading quality scores from {args.quality_scores}")
        with open(args.quality_scores, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                doc_id = j.get('doc_id')
                agg_score = j.get('aggregated', 0.0)
                if doc_id and agg_score >= args.quality_threshold:
                    quality_filter[doc_id] = agg_score
        log.info(f"Loaded {len(quality_filter)} docs above quality threshold {args.quality_threshold}")

    # Initialize BatchedGenerator
    log.info(f"Loading model from {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = BatchedGenerator(
        args.checkpoint,
        max_seq_len=args.max_ctx,
        max_batch=args.max_batch,
        device=device
    )
    enc = get_encoding()

    # Optional: HYDRA-judge gate on the *generated* rephrases (quality gate for synthetic).
    judge = None
    w = tau_drop = tau_keep = None
    if args.hydra_filter:
        from .hydra import _load_calibration, load_local_hydra_judge

        if not args.hydra_heads_dir or not args.hydra_calibration:
            log.error("--hydra-heads-dir and --hydra-calibration are required when --hydra-filter is set")
            return 1

        w, tau_drop, tau_keep = _load_calibration(args.hydra_calibration)
        if args.hydra_tau_keep is not None:
            tau_keep = float(args.hydra_tau_keep)

        judge = load_local_hydra_judge(model=gen.model, heads_dir=args.hydra_heads_dir, device=device)

    total_docs = 0
    total_written = 0
    total_empty = 0

    def flush_batch(batch_ids: List[str], batch_docs: List[str], out_f) -> None:
        nonlocal total_written, total_empty

        rephrased_batch = rephrase_batch(
            gen, enc, batch_docs,
            num_versions=args.num_versions,
            chunk_size=args.chunk_size,
            use_style_diversity=not args.no_style_diversity,
            max_new=args.max_new,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # Fidelity verification if requested (batch-local).
        if args.verify_fidelity:
            model = gen.model
            originals_flat: List[str] = []
            rephrased_flat: List[str] = []
            doc_version_map: List[tuple[int, int]] = []

            for doc_idx, versions in enumerate(rephrased_batch):
                for version_idx, reph in enumerate(versions):
                    originals_flat.append(batch_docs[doc_idx])
                    rephrased_flat.append(reph)
                    doc_version_map.append((doc_idx, version_idx))

            fidelity_scores = compute_fidelity(model, enc, originals_flat, rephrased_flat, device=device)
            fidelity_nested = [[0.0] * len(versions) for versions in rephrased_batch]
            for (doc_idx, version_idx), score in zip(doc_version_map, fidelity_scores):
                fidelity_nested[doc_idx][version_idx] = score

            rephrased_batch = filter_by_fidelity(rephrased_batch, fidelity_nested, threshold=args.fidelity_threshold)

        # Optional HYDRA gating (batch-local).
        hydra_lookup: dict[str, dict] = {}
        if args.hydra_filter and judge is not None:
            from .hydra import grade_texts_with_local_hydra_judge

            cand_ids: List[str] = []
            cand_texts: List[str] = []
            for doc_idx, versions in enumerate(rephrased_batch):
                for version_idx, text in enumerate(versions):
                    if not text:
                        continue
                    cid = f"{batch_ids[doc_idx]}_rephrase_{version_idx}"
                    cand_ids.append(cid)
                    cand_texts.append(text)

            if cand_texts:
                graded = grade_texts_with_local_hydra_judge(
                    model=gen.model,
                    judge=judge,
                    texts=cand_texts,
                    doc_ids=cand_ids,
                    max_ctx=args.hydra_max_ctx,
                    batch_size=args.hydra_batch_size,
                    w=w,
                    tau_drop=tau_drop,
                    tau_keep=tau_keep,
                    device=device,
                )
                for g in graded:
                    hydra_lookup[str(g["doc_id"])] = g

        for doc_idx, versions in enumerate(rephrased_batch):
            parent_id = batch_ids[doc_idx]
            for version_idx, reph_text in enumerate(versions):
                if not reph_text:
                    total_empty += 1
                    continue

                cid = f"{parent_id}_rephrase_{version_idx}"
                if args.hydra_filter:
                    g = hydra_lookup.get(cid)
                    if g is None or g.get("decision") != "keep":
                        continue

                record = {
                    "doc_id": cid,
                    "parent_doc_id": parent_id,
                    "text": reph_text,
                    "is_synthetic": True,
                    "synthetic_kind": "rephrase",
                    "version": version_idx,
                }
                if quality_filter:
                    record["parent_quality_score"] = quality_filter.get(parent_id, 0.0)
                if args.hydra_filter:
                    g = hydra_lookup.get(cid)
                    if g is not None:
                        record["hydra_scores"] = g.get("scores")
                        record["hydra_aggregated"] = g.get("aggregated")
                        record["hydra_decision"] = g.get("decision")

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1

    batch_ids: List[str] = []
    batch_docs: List[str] = []

    log.info(f"Streaming documents from {args.input}")
    with open(args.input, "r") as in_f, open(tmp_output_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            j = json.loads(line)
            doc_id = j.get("doc_id") or j.get("id") or str(total_docs)

            if quality_filter and doc_id not in quality_filter:
                continue

            text = j.get("text", "")
            if not text:
                continue

            batch_ids.append(str(doc_id))
            batch_docs.append(text)
            total_docs += 1

            if len(batch_docs) >= args.batch_size:
                log.info(f"Processing batch (docs={len(batch_docs)})")
                flush_batch(batch_ids, batch_docs, out_f)
                batch_ids = []
                batch_docs = []

            if args.max_docs and total_docs >= args.max_docs:
                break

        if batch_docs:
            log.info(f"Processing batch (docs={len(batch_docs)})")
            flush_batch(batch_ids, batch_docs, out_f)

    tmp_output_path.replace(output_path)

    log.info(f"Complete: {total_docs} input docs → {total_written} rephrase rows (dropped_empty={total_empty})")
    log.info(f"Output written to {output_path}")
    return 0


def cmd_prep_mixture(args: argparse.Namespace) -> int:
    """Preprocess all sources in a mixture stage."""
    from .mixture import resolve_plan
    from .sources import DataSource, Document, HuggingFaceSource, HfFileSystemSource, HydraFilteredSource
    try:
        from .prep import PrepConfig, PrepPipeline, ParallelPrepPipeline
    except Exception as e:
        log.error(f"prep-mixture requires numpy (and other deps); run in the container image. ({e})")
        return 1

    # Core config
    mixture_path = Path(args.config)
    flow_path = Path(args.flow_profiles)
    flow = args.flow

    if not mixture_path.exists():
        log.error(f"Mixture file not found: {mixture_path}")
        return 1
    if not flow_path.exists():
        log.error(f"Flow profiles file not found: {flow_path}")
        return 1

    if not flow:
        try:
            flow = _autodetect_flow_name(mixture_toml=mixture_path, flow_profiles_toml=flow_path)
        except Exception as e:
            log.error(str(e))
            return 1

    # Optional config-side defaults (used by fixture configs and K8s jobs).
    mix_cfg = _read_toml(mixture_path)
    hydra_cfg = mix_cfg.get("hydra") if isinstance(mix_cfg.get("hydra"), dict) else {}

    if getattr(args, "hydra_filter", False):
        args.checkpoint = args.checkpoint or hydra_cfg.get("checkpoint")
        args.heads_dir = args.heads_dir or hydra_cfg.get("heads_dir")
        args.calibration = args.calibration or hydra_cfg.get("calibration")

        if not args.checkpoint:
            log.error("--checkpoint is required with --hydra-filter (or set [hydra].checkpoint in the config)")
            return 1
        if not args.heads_dir:
            log.error("--heads-dir is required with --hydra-filter (or set [hydra].heads_dir in the config)")
            return 1
        if not args.calibration:
            log.error("--calibration is required with --hydra-filter (or set [hydra].calibration in the config)")
            return 1

    # Resolve default output root: /data/flows/<flow>
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = Path("/data/flows") / flow

    # Resolve the plan to get source definitions
    plan = resolve_plan(
        mixture_toml=mixture_path,
        flow_profiles_toml=flow_path,
        flow_section=f"flow.{flow}",
        seq_len=4096,
    )

    # Filter to requested stage
    stages = [s for s in plan.stages if s.stage_id == args.stage] if args.stage else plan.stages
    if not stages:
        log.error(f"No stages found (requested: {args.stage})")
        return 1

    # Collect sources with HF definitions
    sources_to_prep = []
    for stage in stages:
        for sp in stage.sources:
            if sp.hf is None:
                log.warning(f"Source '{sp.id}' has no HuggingFace definition, skipping")
                continue
            sources_to_prep.append((stage.stage_id, sp))

    if not sources_to_prep:
        log.error("No sources with HuggingFace definitions found")
        return 1

    log.info(f"Preparing {len(sources_to_prep)} sources")

    # Get token limit: explicit --max-tokens overrides flow's tokens_b
    if args.max_tokens:
        max_tokens_total = _parse_tokens(args.max_tokens)
        log.info(f"Using explicit token limit: {max_tokens_total:,}")
    else:
        # Sum tokens_b from all stages being prepped
        flow_tokens_b = sum(s.total_tokens_b for s in stages)
        max_tokens_total = int(flow_tokens_b * 1_000_000_000) if flow_tokens_b > 0 else None
        if max_tokens_total:
            log.info(f"Using flow token limit: {max_tokens_total:,} ({flow_tokens_b}B from flow definition)")

    # Calculate per-source quotas based on mixture weights (using target_tokens from SourcePlan)
    source_quotas = {}
    if max_tokens_total:
        total_target = sum(sp.target_tokens for _, sp in sources_to_prep)
        for stage_id, sp in sources_to_prep:
            # Proportional allocation based on target_tokens weights
            quota = int(max_tokens_total * sp.target_tokens / total_target) if total_target > 0 else 0
            source_quotas[sp.id] = quota
        log.info(f"Token limit: {max_tokens_total:,} (proportional by source weight)")
        for sid, quota in source_quotas.items():
            log.info(f"  {sid}: {quota:,} tokens")

    splits = args.splits.split(",") if args.splits else ["train"]
    tokens_so_far = 0

    # Worker partitioning for parallel indexed jobs
    worker_index = args.worker_index
    num_workers = args.num_workers
    if num_workers > 1:
        log.info(f"Worker {worker_index + 1}/{num_workers} (partitioning sources across workers)")

    import hashlib

    class _LimitAndHashSource(DataSource):
        def __init__(
            self,
            *,
            source: DataSource,
            limit_docs: int | None,
            hasher: "hashlib._Hash" | None,
        ):
            self._source = source
            self._limit_docs = limit_docs
            self._hasher = hasher

        @property
        def name(self) -> str:
            return self._source.name

        def estimate_size(self) -> int | None:
            return None

        def __iter__(self):
            n = 0
            for doc in self._source:
                if self._limit_docs is not None and n >= self._limit_docs:
                    break
                if self._hasher is not None:
                    self._hasher.update((doc.doc_id + "\n").encode("utf-8"))
                yield doc
                n += 1

    # Load HYDRA model and judge if filtering enabled
    hydra_model = None
    hydra_judge = None
    hydra_w = None
    hydra_tau_drop = 0.3
    hydra_tau_keep = 0.5
    if args.hydra_filter:
        log.info("Loading HYDRA model and judge for quality filtering...")
        try:
            import torch
        except Exception as e:
            log.error(f"HYDRA filtering requires torch; run in the container image with deps installed. ({e})")
            return 1
        from .model import Transformer
        from .hydra import load_local_hydra_judge, _load_calibration

        if not torch.cuda.is_available():
            log.error("HYDRA filtering requires CUDA; run in the container on a GPU node.")
            return 1

        device = torch.device("cuda")
        hydra_model = Transformer.from_checkpoint(args.checkpoint, device=device)
        for p in hydra_model.parameters():
            p.requires_grad = False
        hydra_model.eval()

        hydra_judge = load_local_hydra_judge(
            model=hydra_model,
            heads_dir=args.heads_dir,
            device=device,
        )
        hydra_w, hydra_tau_drop, hydra_tau_keep = _load_calibration(args.calibration)
        log.info(f"  HYDRA loaded: tau_drop={hydra_tau_drop:.2f}, tau_keep={hydra_tau_keep:.2f}")

    # Build list of all (stage_id, sp, split) work items
    # With file-level partitioning, ALL workers process ALL sources,
    # but each worker only processes their partition of files within each source
    work_items = []
    for stage_id, sp in sources_to_prep:
        hf = sp.hf
        for split in splits:
            # Determine which HF split to use
            if split == "valid" and hf.valid_split:
                hf_split = hf.valid_split
            elif split == "valid":
                continue  # Skip if no valid_split defined
            else:
                hf_split = hf.split
            work_items.append((stage_id, sp, split, hf_split))

    if num_workers > 1:
        log.info(f"Worker {worker_index + 1}/{num_workers}: file-level partitioning across {len(work_items)} sources")

    for stage_id, sp, split, hf_split in work_items:
        hf = sp.hf

        # With parallel workers, each writes to worker-specific subdirectory
        if num_workers > 1:
            output_dir = output_root / stage_id / sp.id / split / f"worker_{worker_index:03d}"
        else:
            output_dir = output_root / stage_id / sp.id / split
        manifest_path = output_dir / "manifest.json"

        # Skip if already processed (unless --force)
        if manifest_path.exists() and not args.force:
            log.info(f"Skipping {sp.id}/{split} (already exists, use --force to reprocess)")
            if getattr(args, "limit", None):
                try:
                    base_source = HuggingFaceSource(
                        dataset=hf.dataset,
                        split=hf_split,
                        subset=hf.subset,
                        text_field=hf.text_field,
                        streaming=True,
                        data_files=hf.data_files,
                    )
                    if num_workers > 1:
                        base_source = base_source.shard(num_shards=num_workers, index=worker_index)

                    h = hashlib.sha256()
                    for _ in _LimitAndHashSource(source=base_source, limit_docs=int(args.limit), hasher=h):
                        pass
                    log.info(f"sample_ids_sha256 {stage_id}/{sp.id}/{split} {h.hexdigest()}")
                except Exception as e:
                    log.warning(f"Failed to compute sample-id hash for skipped output: {e}")
            continue

        log.info(f"Processing: {sp.id}/{split}")
        log.info(f"  HF: {hf.dataset} / {hf.subset or 'default'} / {hf_split}")
        if hf.data_files:
            log.info(f"  Data files: {hf.data_files}")
        log.info(f"  Output: {output_dir}")

        if args.dry_run:
            continue

        # Create source - try HuggingFaceSource first, fall back to HfFileSystemSource
        # for datasets with pyarrow schema issues (e.g., heterogeneous metadata types)
        base_source = HuggingFaceSource(
            dataset=hf.dataset,
            split=hf_split,
            subset=hf.subset,
            text_field=hf.text_field,
            streaming=True,
            data_files=hf.data_files,
        )
        # Apply sharding if multiple workers (for HuggingFaceSource)
        if num_workers > 1:
            base_source = base_source.shard(num_shards=num_workers, index=worker_index)
        use_hf_filesystem_fallback = False

        # Optional doc limit + sample-id hash for determinism checks.
        sample_id_hasher = hashlib.sha256() if getattr(args, "limit", None) else None
        if getattr(args, "limit", None):
            base_source = _LimitAndHashSource(source=base_source, limit_docs=int(args.limit), hasher=sample_id_hasher)

        # Wrap with HYDRA filter if enabled
        if args.hydra_filter and hydra_model is not None:
            source = HydraFilteredSource(
                source=base_source,
                model=hydra_model,
                judge=hydra_judge,
                w=hydra_w,
                tau_drop=hydra_tau_drop,
                tau_keep=hydra_tau_keep,
                batch_size=args.hydra_batch_size,
                max_ctx=args.hydra_max_ctx,
            )
            log.info(f"  HYDRA filtering enabled (batch={args.hydra_batch_size}, ctx={args.hydra_max_ctx})")
        else:
            source = base_source

        # Use per-source quota (proportional to mixture weight)
        # With file-level partitioning, divide quota by number of workers
        source_token_limit = source_quotas.get(sp.id) if source_quotas else None
        if source_token_limit and num_workers > 1:
            source_token_limit = source_token_limit // num_workers
            log.info(f"  Worker {worker_index}: token limit = {source_token_limit:,} (1/{num_workers} of source quota)")

        # Create config
        config = PrepConfig(
            output_dir=output_dir,
            dataset_name=sp.id,
            version="v1",
            tokenizer=args.tokenizer,
            vocab_size=args.vocab_size,
            eos_token_id=args.eos_token_id,
            num_shards=args.num_shards,
            num_workers=args.workers,
            batch_size=args.batch_size,
            max_tokens_total=source_token_limit,
        )

        # Progress callback
        last_report = [0]
        limit_for_log = source_token_limit  # capture for closure
        def progress(docs: int, tokens: int):
            if docs - last_report[0] >= 10000:
                if limit_for_log:
                    log.info(f"  Progress: {docs:,} docs, {tokens:,}/{limit_for_log:,} tokens")
                else:
                    log.info(f"  Progress: {docs:,} docs, {tokens:,} tokens")
                last_report[0] = docs

        # Run pipeline
        if args.parallel:
            pipeline = ParallelPrepPipeline(source, config)
        else:
            pipeline = PrepPipeline(source, config)

        try:
            manifest = pipeline.run(progress_callback=progress)
            tokens_so_far += manifest.total_tokens
            log.info(f"  Complete: {manifest.total_documents:,} docs, {manifest.total_tokens:,} tokens")
            if sample_id_hasher is not None:
                log.info(f"sample_ids_sha256 {stage_id}/{sp.id}/{split} {sample_id_hasher.hexdigest()}")
            # Log HYDRA filter stats if enabled
            if args.hydra_filter and hasattr(source, 'stats'):
                stats = source.stats
                kept = int(stats.get("keep", 0))
                total = int(stats.get("total", 0))
                keep_rate = kept / total * 100 if total > 0 else 0.0
                log.info(f"  HYDRA: {kept:,}/{total:,} kept ({keep_rate:.1f}%)")
        except KeyboardInterrupt:
            log.warning("Interrupted")
            return 130
        except Exception as e:
            # Check if this is a pyarrow schema/compression error - if so, retry with HfFileSystemSource
            error_str = str(e)
            is_fallback_error = (
                "ArrowTypeError" in type(e).__name__ or
                "ArrowInvalid" in type(e).__name__ or
                "changed from" in error_str or
                "Expected bytes" in error_str or
                "zstd not supported" in error_str or
                "Compression type" in error_str
            )

            if is_fallback_error and hf.data_files and not use_hf_filesystem_fallback:
                log.warning(f"  Fallback to HfFileSystemSource: {e}")
                use_hf_filesystem_fallback = True

                # Create fallback source with worker partitioning
                base_source = HfFileSystemSource(
                    repo_id=hf.dataset,
                    data_files=hf.data_files,
                    text_field=hf.text_field,
                    worker_index=worker_index,
                    num_workers=num_workers,
                )

                # Wrap with HYDRA filter if enabled
                if args.hydra_filter and hydra_model is not None:
                    source = HydraFilteredSource(
                        source=base_source,
                        model=hydra_model,
                        judge=hydra_judge,
                        w=hydra_w,
                        tau_drop=hydra_tau_drop,
                        tau_keep=hydra_tau_keep,
                        batch_size=args.hydra_batch_size,
                        max_ctx=args.hydra_max_ctx,
                    )
                else:
                    source = base_source

                # Recreate pipeline with fallback source
                if args.parallel:
                    pipeline = ParallelPrepPipeline(source, config)
                else:
                    pipeline = PrepPipeline(source, config)

                try:
                    manifest = pipeline.run(progress_callback=progress)
                    tokens_so_far += manifest.total_tokens
                    log.info(f"  Complete (fallback): {manifest.total_documents:,} docs, {manifest.total_tokens:,} tokens")
                except Exception as e2:
                    log.error(f"  Failed (fallback): {e2}")
                    if not args.continue_on_error:
                        return 1
            else:
                log.error(f"  Failed: {e}")
                if not args.continue_on_error:
                    return 1
            continue

    log.info(f"Done: {tokens_so_far:,} tokens total")
    return 0


def cmd_merge_manifests(args: argparse.Namespace) -> int:
    """Merge worker manifests into a single top-level manifest.

    Used after parallel prep-mixture to consolidate outputs.
    """
    import glob
    from .sinks import load_manifest, ManifestInfo, ShardInfo

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        log.error(f"Source directory not found: {source_dir}")
        return 1

    # Find all worker manifests
    pattern = str(source_dir / "**/worker_*/manifest.json")
    worker_manifests = sorted(glob.glob(pattern, recursive=True))

    if not worker_manifests:
        log.error(f"No worker manifests found matching: {pattern}")
        return 1

    log.info(f"Found {len(worker_manifests)} worker manifests")

    # Aggregate stats
    total_docs = 0
    total_tokens = 0
    all_shards = []
    last_manifest = None

    for manifest_path in worker_manifests:
        try:
            manifest = load_manifest(Path(manifest_path))
            last_manifest = manifest
            total_docs += manifest.total_documents
            total_tokens += manifest.total_tokens
            # Adjust shard paths to be relative to source_dir
            worker_dir = Path(manifest_path).parent
            for shard in manifest.shards:
                # Convert absolute path to relative from source_dir
                shard_path = worker_dir / shard.path if not Path(shard.path).is_absolute() else Path(shard.path)
                try:
                    rel_path = shard_path.relative_to(source_dir)
                except ValueError:
                    rel_path = shard_path
                all_shards.append(ShardInfo(
                    path=str(rel_path),
                    num_tokens=shard.num_tokens,
                    num_documents=shard.num_documents,
                    checksum=shard.checksum,
                ))
            log.info(f"  {manifest_path}: {manifest.total_documents:,} docs, {manifest.total_tokens:,} tokens")
        except Exception as e:
            log.warning(f"  Failed to load {manifest_path}: {e}")
            continue

    if last_manifest is None:
        log.error("No readable worker manifests found; nothing to merge.")
        return 1

    # Write merged manifest
    merged = ManifestInfo(
        dataset=args.dataset_name or "merged",
        version="v1",
        tokenizer=last_manifest.tokenizer,
        vocab_size=last_manifest.vocab_size,
        eos_token_id=last_manifest.eos_token_id,
        total_documents=total_docs,
        total_tokens=total_tokens,
        shards=all_shards,
    )

    output_path = source_dir / "manifest.json"
    import json
    with open(output_path, "w") as f:
        json.dump({
            "dataset": merged.dataset,
            "version": merged.version,
            "tokenizer": merged.tokenizer,
            "vocab_size": merged.vocab_size,
            "eos_token_id": merged.eos_token_id,
            "total_documents": merged.total_documents,
            "total_tokens": merged.total_tokens,
            "shards": [
                {
                    "path": s.path,
                    "num_tokens": s.num_tokens,
                    "num_documents": s.num_documents,
                    "checksum": s.checksum,
                }
                for s in merged.shards
            ],
        }, f, indent=2)

    log.info(f"Merged manifest written to {output_path}")
    log.info(f"  Total: {total_docs:,} documents, {total_tokens:,} tokens, {len(all_shards)} shards")
    return 0


def cmd_flow(args: argparse.Namespace) -> int:
    """Run a preset flow profile (validation/bench helper).

    This is intentionally a thin wrapper around `prep-mixture` to avoid
    maintaining a second parallel stack.
    """
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log.error(f"Flow profiles file not found: {cfg_path}")
        return 1

    cfg = _read_toml(cfg_path)
    profiles = cfg.get("profiles")
    if not isinstance(profiles, dict):
        log.error(f"{cfg_path} is missing a [profiles] table")
        return 1

    prof = profiles.get(args.profile)
    if not isinstance(prof, dict):
        log.error(f"Unknown profile '{args.profile}'. Available: {', '.join(sorted(profiles.keys()))}")
        return 1

    mixture = prof.get("mixture")
    if not isinstance(mixture, str) or not mixture:
        log.error(f"[profiles.{args.profile}] is missing 'mixture' (path to mixture TOML)")
        return 1

    argv: list[str] = ["prep-mixture", "--config", mixture, "--flow-profiles", str(cfg_path)]

    if isinstance(prof.get("flow"), str) and prof["flow"]:
        argv += ["--flow", str(prof["flow"])]
    if isinstance(prof.get("stage"), str) and prof["stage"]:
        argv += ["--stage", str(prof["stage"])]
    if isinstance(prof.get("splits"), str) and prof["splits"]:
        argv += ["--splits", str(prof["splits"])]
    if prof.get("seed") is not None:
        argv += ["--seed", str(int(prof["seed"]))]
    if prof.get("limit") is not None:
        argv += ["--limit", str(int(prof["limit"]))]
    if isinstance(prof.get("output_root"), str) and prof["output_root"]:
        argv += ["--output-root", str(prof["output_root"])]

    for k, flag in (
        ("tokenizer", "--tokenizer"),
        ("num_shards", "--num-shards"),
        ("workers", "--workers"),
        ("batch_size", "--batch-size"),
    ):
        v = prof.get(k)
        if v is None:
            continue
        argv += [flag, str(v)]

    argv += ["--num-workers", str(int(args.num_workers)), "--worker-index", str(int(args.worker_index))]
    return main(argv)


def cmd_hydra_passthrough(args: argparse.Namespace) -> int:
    """Run nmoe.data.hydra subcommands via the main CLI."""
    from . import hydra as hydra_mod
    argv = [str(args.hydra_subcommand)] + list(getattr(args, "args", []) or [])
    return hydra_mod.main(argv)


def cmd_arxiv_download(args: argparse.Namespace) -> int:
    """Download arXiv data from S3 and/or ar5iv."""
    if getattr(args, "dry_run", False):
        log.info("arxiv-download --dry-run: no files will be downloaded.")
        log.info("Set --output to a directory and omit --dry-run to run the download.")
        return 0

    if not getattr(args, "output", None):
        log.error("--output is required unless --dry-run is set")
        return 1

    _, _, _, download_s3_tars, download_ar5iv_dataset = _get_arxiv()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    year_months = args.year_months if args.year_months else None

    if args.source in ("s3", "both"):
        log.info("Downloading arXiv S3 source tars...")
        s3_output = output_dir / "s3_src"
        try:
            downloaded = download_s3_tars(
                s3_output,
                year_months=year_months,
                max_files=args.max_files,
                max_workers=args.workers,
                aws_profile=args.aws_profile,
                skip_existing=not args.force,
            )
            log.info(f"Downloaded {len(downloaded)} S3 tar files")
        except Exception as e:
            log.error(f"S3 download failed: {e}")
            if args.source == "s3":
                return 1

    if args.source in ("ar5iv", "both"):
        log.info("Downloading ar5iv HTML dataset...")
        ar5iv_output = output_dir / "ar5iv"
        quality_levels = args.quality.split(",") if args.quality else ["no_problem", "warning"]
        try:
            download_ar5iv_dataset(
                ar5iv_output,
                quality_levels=quality_levels,
                max_workers=args.workers,
            )
            log.info(f"ar5iv dataset downloaded to {ar5iv_output}")
        except Exception as e:
            log.error(f"ar5iv download failed: {e}")
            if args.source == "ar5iv":
                return 1

    log.info("arXiv download complete")
    return 0


def cmd_arxiv_prep(args: argparse.Namespace) -> int:
    """Preprocess arXiv data to tokenized shards."""
    ArxivS3Source, Ar5ivHTMLSource, ArxivMultiFormatSource, _, _ = _get_arxiv()
    try:
        from .prep import PrepConfig, PrepPipeline
    except Exception as e:
        log.error(f"arxiv-prep requires numpy (and other deps); run in the container image. ({e})")
        return 1

    output_dir = Path(args.output)

    # Create source based on what's available
    if args.s3_dir and args.ar5iv_dir:
        log.info("Using ArxivMultiFormatSource (both LaTeX and HTML)")
        source = ArxivMultiFormatSource(
            s3_tar_dir=args.s3_dir,
            ar5iv_dir=args.ar5iv_dir,
            emit_both_formats=args.emit_both_formats,
            quality_levels=args.quality.split(",") if args.quality else ["no_problem", "warning"],
            worker_index=args.worker_index,
            num_workers=args.num_workers,
            max_docs=args.max_docs,
        )
    elif args.s3_dir:
        log.info("Using ArxivS3Source (LaTeX only)")
        year_months = args.year_months if args.year_months else None
        source = ArxivS3Source(
            tar_dir=args.s3_dir,
            year_month_filter=year_months,
            worker_index=args.worker_index,
            num_workers=args.num_workers,
            max_docs=args.max_docs,
        )
    elif args.ar5iv_dir:
        log.info("Using Ar5ivHTMLSource (HTML only)")
        source = Ar5ivHTMLSource(
            html_dir=args.ar5iv_dir,
            quality_levels=args.quality.split(",") if args.quality else ["no_problem", "warning"],
            worker_index=args.worker_index,
            num_workers=args.num_workers,
            max_docs=args.max_docs,
        )
    else:
        log.error("Either --s3-dir or --ar5iv-dir (or both) required")
        return 1

    # With parallel workers, each writes to worker-specific subdirectory
    if args.num_workers > 1:
        output_dir = output_dir / f"worker_{args.worker_index:03d}"

    log.info(f"Output: {output_dir}")
    log.info(f"Worker: {args.worker_index + 1}/{args.num_workers}")

    # Create config
    config = PrepConfig(
        output_dir=output_dir,
        dataset_name="arxiv",
        version="v1",
        tokenizer=args.tokenizer,
        vocab_size=args.vocab_size,
        eos_token_id=args.eos_token_id,
        num_shards=args.num_shards,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )

    # Progress callback
    last_report = [0]
    def progress(docs: int, tokens: int):
        if docs - last_report[0] >= 1000:
            log.info(f"Progress: {docs:,} docs, {tokens:,} tokens")
            last_report[0] = docs

    # Run pipeline
    pipeline = PrepPipeline(source, config)

    try:
        manifest = pipeline.run(progress_callback=progress)
        log.info(f"Complete: {manifest.total_documents:,} docs, {manifest.total_tokens:,} tokens")
        log.info(f"Shards: {manifest.num_shards}")
        return 0
    except KeyboardInterrupt:
        log.warning("Interrupted")
        return 130
    except Exception as e:
        log.error(f"Failed: {e}")
        return 1


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="nmoe data preprocessing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # prep command
    prep_parser = subparsers.add_parser("prep", help="Preprocess dataset")
    prep_parser.add_argument("--source", required=True, choices=["huggingface", "hf", "jsonl", "jsonl_zst", "arrow", "parquet"])
    prep_parser.add_argument("--dataset", help="HuggingFace dataset name")
    prep_parser.add_argument("--split", default="train", help="Dataset split")
    prep_parser.add_argument("--subset", help="Dataset subset/config")
    prep_parser.add_argument("--paths", nargs="+", help="File paths (for jsonl/arrow/parquet)")
    prep_parser.add_argument("--text-field", default="text", help="Text field name")
    prep_parser.add_argument("--id-field", help="Document ID field name (optional)")
    prep_parser.add_argument("--output", "-o", required=True, help="Output directory")
    prep_parser.add_argument("--name", required=True, help="Dataset name")
    prep_parser.add_argument("--version", default="v1", help="Dataset version")
    prep_parser.add_argument("--tokenizer", default="o200k_harmony", help="Tokenizer name")
    prep_parser.add_argument("--vocab-size", type=int, default=201088, help="Vocabulary size")
    prep_parser.add_argument("--eos-token-id", type=int, default=199999, help="EOS token ID")
    prep_parser.add_argument("--num-shards", type=int, default=64, help="Number of output shard buckets")
    prep_parser.add_argument("--tokens-per-shard", type=int, default=500_000_000, help="Max tokens per shard")
    prep_parser.add_argument("--workers", "-j", type=int, default=4, help="Tokenization workers")
    prep_parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    prep_parser.add_argument("--min-tokens", type=int, default=10, help="Min tokens per doc")
    prep_parser.add_argument("--max-tokens", type=int, default=1_000_000, help="Max tokens per doc")
    prep_parser.add_argument("--max-tokens-total", help="Early stop after N tokens (e.g., '100M', '1B', '10B')")
    prep_parser.add_argument("--parallel", action="store_true", help="Use parallel pipeline")
    # File-level sharding for K8s indexed jobs
    prep_parser.add_argument("--worker-index", type=int, default=0, help="Worker index for file partitioning (0-based)")
    prep_parser.add_argument("--num-workers", type=int, default=1, help="Total workers for file partitioning")
    # HYDRA quality filtering
    prep_parser.add_argument("--hydra-filter", action="store_true", help="Filter with HYDRA (only emit 'keep' docs)")
    prep_parser.add_argument("--hydra-audit-dir", help="Directory to write HYDRA scores for ALL docs (audit trail)")
    prep_parser.add_argument("--checkpoint", help="Path to model checkpoint (required for HYDRA)")
    prep_parser.add_argument("--heads-dir", help="Directory with hydra_judge.pt (required for HYDRA)")
    prep_parser.add_argument("--calibration", help="Directory with calibration_summary.json (required for HYDRA)")
    prep_parser.add_argument("--hydra-batch-size", type=int, default=32, help="HYDRA batch size")
    prep_parser.add_argument("--hydra-max-ctx", type=int, default=4096, help="HYDRA max context")
    prep_parser.set_defaults(func=cmd_prep)

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify manifest")
    verify_parser.add_argument("manifest", help="Path to manifest.json")
    verify_parser.add_argument("--checksums", action="store_true", help="Verify checksums (slow)")
    verify_parser.set_defaults(func=cmd_verify)

    # info command
    info_parser = subparsers.add_parser("info", help="Show manifest info")
    info_parser.add_argument("manifest", help="Path to manifest.json")
    info_parser.set_defaults(func=cmd_info)

    # regenerate-index command
    regen_parser = subparsers.add_parser("regenerate-index", help="Regenerate index from shard")
    regen_parser.add_argument("shard", help="Path to .npy shard")
    regen_parser.add_argument("--eos-token-id", type=int, required=True, help="EOS token ID")
    regen_parser.set_defaults(func=cmd_regenerate_index)

    # prep-mixture command
    pm_parser = subparsers.add_parser("prep-mixture", help="Preprocess sources for a mixture and flow")
    pm_parser.add_argument("--config", required=True, help="Mixture TOML (e.g., configs/mixtures/fineweb_edu_proxy.toml)")
    pm_parser.add_argument("--flow-profiles", default="configs/flow_profiles.toml", help="Flow profiles TOML (default: configs/flow_profiles.toml)")
    pm_parser.add_argument("--flow", help="Flow name (auto-detected if unique)")
    pm_parser.add_argument("--stage", help="Stage to process (pretrain, mid, long). If not set, all stages.")
    pm_parser.add_argument("--splits", default="train", help="Comma-separated splits to process (train,valid)")
    pm_parser.add_argument("--seed", type=int, required=True, help="Determinism seed (logged; used in sample-id hashing)")
    pm_parser.add_argument("--limit", type=int, help="Limit docs processed per source (also logs sample-id SHA256)")
    pm_parser.add_argument("--output-root", help="Output root (defaults to /data/flows/<flow>)")
    pm_parser.add_argument("--tokenizer", default="o200k_harmony", help="Tokenizer name")
    pm_parser.add_argument("--vocab-size", type=int, default=201088, help="Vocabulary size")
    pm_parser.add_argument("--eos-token-id", type=int, default=199999, help="EOS token ID")
    pm_parser.add_argument("--num-shards", type=int, default=64, help="Number of shards per source")
    pm_parser.add_argument("--workers", "-j", type=int, default=8, help="Number of workers")
    pm_parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    pm_parser.add_argument("--parallel", action="store_true", help="Use parallel pipeline")
    pm_parser.add_argument("--force", action="store_true", help="Reprocess even if output exists")
    pm_parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    pm_parser.add_argument("--continue-on-error", action="store_true", help="Continue if a source fails")
    pm_parser.add_argument("--max-tokens", help="Override token limit (e.g., '100M', '1B')")
    # HYDRA quality filtering
    pm_parser.add_argument("--hydra-filter", action="store_true", help="Filter documents with HYDRA quality judge")
    pm_parser.add_argument("--checkpoint", help="Path to gpt-oss-20b checkpoint (required with --hydra-filter)")
    pm_parser.add_argument("--heads-dir", help="Directory containing hydra_judge.pt (required with --hydra-filter)")
    pm_parser.add_argument("--calibration", help="Directory containing calibration_summary.json (required with --hydra-filter)")
    pm_parser.add_argument("--hydra-batch-size", type=int, default=32, help="Batch size for HYDRA grading (default: 32)")
    pm_parser.add_argument("--hydra-max-ctx", type=int, default=4096, help="Max context for HYDRA grading (default: 4096)")
    # Parallel worker partitioning (for indexed K8s jobs)
    pm_parser.add_argument("--worker-index", type=int, default=0, help="Worker index for sharded processing (0-based)")
    pm_parser.add_argument("--num-workers", type=int, default=1, help="Total number of parallel workers")
    pm_parser.set_defaults(func=cmd_prep_mixture)

    # flow command (validation/bench helper)
    flow_parser = subparsers.add_parser("flow", help="Run a flow profile from flow_profiles.toml (validation/bench)")
    flow_parser.add_argument("--config", required=True, help="Flow profiles TOML (e.g., configs/flow_profiles.toml)")
    flow_parser.add_argument("--profile", required=True, help="Profile name (e.g., mini)")
    flow_parser.add_argument("--num-workers", type=int, default=1, help="Total workers for sharded processing")
    flow_parser.add_argument("--worker-index", type=int, default=0, help="Worker index (0-based)")
    flow_parser.set_defaults(func=cmd_flow)

    # hydra passthrough commands
    for name, subcmd, help_txt in (
        ("hydra-oracle-label", "oracle-label", "HYDRA oracle labeling (wrapper for nmoe.data.hydra oracle-label)"),
        ("hydra-calibrate", "calibrate", "HYDRA calibration (wrapper for nmoe.data.hydra calibrate)"),
        ("hydra-grade", "grade", "HYDRA grading (wrapper for nmoe.data.hydra grade)"),
    ):
        hp = subparsers.add_parser(name, help=help_txt)
        hp.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to nmoe.data.hydra")
        hp.set_defaults(func=cmd_hydra_passthrough, hydra_subcommand=subcmd)

    # dedup command (MinHash near-dedup)
    dedup_parser = subparsers.add_parser("dedup", help="Near-deduplicate JSONL (keep-first) using MinHash LSH")
    dedup_parser.add_argument("--in", dest="input", required=True, help="Input JSONL(.gz) with a text field")
    dedup_parser.add_argument("--out", dest="output", required=True, help="Output JSONL of kept rows")
    dedup_parser.add_argument("--text-field", default="text", help="Text field name (default: text)")
    dedup_parser.add_argument("--shingle", type=int, default=13, help="Character shingle size (default 13)")
    dedup_parser.add_argument("--perms", type=int, default=128, help="Number of MinHash permutations (default 128)")
    dedup_parser.add_argument("--jaccard", type=float, default=0.82, help="Jaccard threshold to drop near-dupes")
    dedup_parser.add_argument("--seed", type=int, default=42)
    dedup_parser.set_defaults(func=cmd_dedup)

    # cluster command
    cl_parser = subparsers.add_parser("cluster", help="Cluster embeddings with MiniBatch K-Means")
    cl_parser.add_argument("--emb", required=True, help="Path to embeddings .npy [N,H]")
    cl_parser.add_argument("--k", type=int, required=True, help="Number of clusters")
    cl_parser.add_argument("--seed", type=int, default=42)
    cl_parser.add_argument("--iters", type=int, default=50)
    cl_parser.add_argument("--batch", type=int, default=4096)
    cl_parser.add_argument("--out", required=True, help="Output directory for labels/centroids/summary")
    cl_parser.set_defaults(func=cmd_cluster)

    # inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect shard data for quality issues")
    inspect_parser.add_argument("--manifest", help="Path to manifest.json")
    inspect_parser.add_argument("--data-dir", help="Data directory (if no manifest)")
    inspect_parser.add_argument("--eos-token-id", type=int, default=199999, help="EOS token ID (default: 199999)")
    inspect_parser.add_argument("--num-shards", type=int, default=1, help="Number of shards to inspect (default: 1)")
    inspect_parser.add_argument("--all", action="store_true", help="Inspect all shards")
    inspect_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    inspect_parser.add_argument("--sample", type=int, help="Sample N random tokens per shard")
    inspect_parser.add_argument("--stats", action="store_true", help="Compute token distribution stats")
    inspect_parser.set_defaults(func=cmd_inspect)

    # merge-manifests command (for parallel prep-mixture)
    merge_parser = subparsers.add_parser("merge-manifests", help="Merge worker manifests after parallel prep-mixture")
    merge_parser.add_argument("source_dir", help="Directory containing worker_NNN subdirectories")
    merge_parser.add_argument("--dataset-name", help="Dataset name for merged manifest")
    merge_parser.set_defaults(func=cmd_merge_manifests)

    # rephrase command - K2-style knowledge rephrasing
    rephrase_parser = subparsers.add_parser("rephrase", help="K2-style knowledge rephrasing with style diversity")
    rephrase_parser.add_argument("--input", required=True, help="Input JSONL file with documents")
    rephrase_parser.add_argument("--output", required=True, help="Output JSONL file for rephrased documents")
    rephrase_parser.add_argument("--checkpoint", required=True, help="Model checkpoint directory")
    rephrase_parser.add_argument("--quality-scores", help="Optional quality_scores.jsonl from HYDRA grading")
    rephrase_parser.add_argument("--quality-threshold", type=float, default=0.6, help="Minimum quality score to rephrase (default: 0.6)")
    rephrase_parser.add_argument("--num-versions", type=int, default=10, help="Number of rephrased versions per doc (default: 10 for K2)")
    rephrase_parser.add_argument("--chunk-size", type=int, default=2048, help="Max tokens per chunk for long docs (default: 2048)")
    rephrase_parser.add_argument("--no-style-diversity", action="store_true", help="Disable style-diverse prompts")
    rephrase_parser.add_argument("--verify-fidelity", action="store_true", help="Verify semantic fidelity and filter low-scoring rephrasings")
    rephrase_parser.add_argument("--fidelity-threshold", type=float, default=0.85, help="Minimum fidelity score (default: 0.85)")
    rephrase_parser.add_argument("--max-ctx", type=int, default=8192, help="Max context length (default: 8192)")
    rephrase_parser.add_argument("--max-batch", type=int, default=32, help="Max batch size for generator (default: 32)")
    rephrase_parser.add_argument("--batch-size", type=int, default=8, help="Documents per rephrasing batch (default: 8)")
    rephrase_parser.add_argument("--max-new", type=int, default=4096, help="Max new tokens per chunk (default: 4096)")
    rephrase_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    rephrase_parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold (default: 0.9)")
    rephrase_parser.add_argument("--max-docs", type=int, help="Max documents to process (for testing)")
    rephrase_parser.add_argument("--hydra-filter", action="store_true", help="Gate generated rephrases with local HYDRA judge head")
    rephrase_parser.add_argument("--hydra-heads-dir", help="Dir containing hydra_judge.pt (required with --hydra-filter)")
    rephrase_parser.add_argument("--hydra-calibration", help="Dir containing calibration_summary.json (required with --hydra-filter)")
    rephrase_parser.add_argument("--hydra-max-ctx", type=int, default=4096, help="Max context for HYDRA grading (default: 4096)")
    rephrase_parser.add_argument("--hydra-batch-size", type=int, default=16, help="Batch size for HYDRA grading (default: 16)")
    rephrase_parser.add_argument("--hydra-tau-keep", type=float, help="Override tau_keep for HYDRA gating")
    rephrase_parser.set_defaults(func=cmd_rephrase)

    # arxiv-download command
    arxiv_dl_parser = subparsers.add_parser("arxiv-download", help="Download arXiv data from S3 and/or ar5iv")
    arxiv_dl_parser.add_argument("--source", choices=["s3", "ar5iv", "both"], default="both",
                                  help="Which sources to download (default: both)")
    arxiv_dl_parser.add_argument("--dry-run", action="store_true", help="Print what would be done and exit")
    arxiv_dl_parser.add_argument("--output", "-o", help="Output directory (required unless --dry-run)")
    arxiv_dl_parser.add_argument("--year-months", nargs="+", help="Filter by YYMM (e.g., 2301 2302)")
    arxiv_dl_parser.add_argument("--max-files", type=int, help="Max tar files to download (for testing)")
    arxiv_dl_parser.add_argument("--quality", default="no_problem,warning",
                                  help="ar5iv quality levels (default: no_problem,warning)")
    arxiv_dl_parser.add_argument("--workers", "-j", type=int, default=4, help="Parallel download workers")
    arxiv_dl_parser.add_argument("--aws-profile", help="AWS profile for S3 requester-pays")
    arxiv_dl_parser.add_argument("--force", action="store_true", help="Re-download existing files")
    arxiv_dl_parser.set_defaults(func=cmd_arxiv_download)

    # arxiv-prep command
    arxiv_prep_parser = subparsers.add_parser("arxiv-prep", help="Preprocess arXiv to tokenized shards")
    arxiv_prep_parser.add_argument("--s3-dir", help="Directory with S3 LaTeX tars")
    arxiv_prep_parser.add_argument("--ar5iv-dir", help="Directory with ar5iv HTML")
    arxiv_prep_parser.add_argument("--output", "-o", required=True, help="Output directory")
    arxiv_prep_parser.add_argument("--emit-both-formats", action="store_true", default=True,
                                    help="Emit both LaTeX and HTML as separate docs (default: True)")
    arxiv_prep_parser.add_argument("--year-months", nargs="+", help="Filter S3 by YYMM")
    arxiv_prep_parser.add_argument("--quality", default="no_problem,warning",
                                    help="ar5iv quality levels (default: no_problem,warning)")
    arxiv_prep_parser.add_argument("--tokenizer", default="o200k_harmony", help="Tokenizer name")
    arxiv_prep_parser.add_argument("--vocab-size", type=int, default=201088, help="Vocabulary size")
    arxiv_prep_parser.add_argument("--eos-token-id", type=int, default=199999, help="EOS token ID")
    arxiv_prep_parser.add_argument("--num-shards", type=int, default=64, help="Number of shards")
    arxiv_prep_parser.add_argument("--workers", "-j", type=int, default=8, help="Tokenization workers")
    arxiv_prep_parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    arxiv_prep_parser.add_argument("--max-docs", type=int, help="Max documents (for testing)")
    arxiv_prep_parser.add_argument("--worker-index", type=int, default=0,
                                    help="Worker index for K8s indexed jobs (0-based)")
    arxiv_prep_parser.add_argument("--num-workers", type=int, default=1,
                                    help="Total number of parallel workers")
    arxiv_prep_parser.set_defaults(func=cmd_arxiv_prep)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
