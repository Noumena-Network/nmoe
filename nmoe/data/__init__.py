"""
nmoe.data - Data loading, preprocessing, inference, and mixture utilities.

Primary exports:
- build_loader: Main entry point for creating data loaders
- DeterministicLoader: Production loader with SWRR mixing
- MixturePlan: Data structure for mixture configuration

Preprocessing:
- PrepPipeline: Local preprocessing pipeline
- ParallelPrepPipeline: Multiprocess preprocessing
- PrepConfig: Pipeline configuration
- HuggingFaceSource: Stream from HuggingFace Hub

Index/Shard utilities:
- IndexReader, IndexWriter: Document boundary indices
- ShardedWriter: Write .npy + .idx shards
- ManifestInfo: Dataset manifest
"""
from __future__ import annotations

import importlib
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Loader
    "build_loader": (".loader", "build_loader"),
    "DeterministicLoader": (".loader", "DeterministicLoader"),
    # Mixture
    "MixturePlan": (".mixture", "MixturePlan"),
    "StagePlan": (".mixture", "StagePlan"),
    "SourcePlan": (".mixture", "SourcePlan"),
    "HFSource": (".mixture", "HFSource"),
    "resolve_plan": (".mixture", "resolve_plan"),
    "populate_paths": (".mixture", "populate_paths"),
    # Dataset
    "NumpyFSLDataset": (".dataset", "NumpyFSLDataset"),
    "Cursor": (".dataset", "Cursor"),
    # Index
    "IndexReader": (".index", "IndexReader"),
    "IndexWriter": (".index", "IndexWriter"),
    "regenerate_index_from_shard": (".index", "regenerate_index_from_shard"),
    # Sources
    "DataSource": (".sources", "DataSource"),
    "Document": (".sources", "Document"),
    "HuggingFaceSource": (".sources", "HuggingFaceSource"),
    "HfFileSystemSource": (".sources", "HfFileSystemSource"),
    "JSONLSource": (".sources", "JSONLSource"),
    "TextFileSource": (".sources", "TextFileSource"),
    "ArrowSource": (".sources", "ArrowSource"),
    "HydraScore": (".sources", "HydraScore"),
    "HydraScoringSource": (".sources", "HydraScoringSource"),
    "HydraFilteredSource": (".sources", "HydraFilteredSource"),
    "create_source": (".sources", "create_source"),
    # Sinks
    "ShardWriter": (".sinks", "ShardWriter"),
    "ShardedWriter": (".sinks", "ShardedWriter"),
    "ShardInfo": (".sinks", "ShardInfo"),
    "ManifestInfo": (".sinks", "ManifestInfo"),
    "load_manifest": (".sinks", "load_manifest"),
    "verify_manifest": (".sinks", "verify_manifest"),
    # Preprocessing
    "PrepConfig": (".prep", "PrepConfig"),
    "PrepPipeline": (".prep", "PrepPipeline"),
    "ParallelPrepPipeline": (".prep", "ParallelPrepPipeline"),
    # Transforms
    "normalize_text": (".transforms", "normalize_text"),
    "tokenize": (".transforms", "tokenize"),
    "pack_document": (".transforms", "pack_document"),
    "get_shard_id": (".transforms", "get_shard_id"),
    "make_file_name": (".transforms", "make_file_name"),
    # Scoring
    "build_prompt": (".score", "build_prompt"),
    "right_trim": (".score", "right_trim"),
    "grade_prompts": (".score", "grade_prompts"),
    "compute_aggregated": (".score", "compute_aggregated"),
    "compute_aggregated_code": (".score", "compute_aggregated_code"),
    # Rephrasing
    "format_prompt": (".rephrase", "format_prompt"),
    "create_parser": (".rephrase", "create_parser"),
    "stop_tokens": (".rephrase", "stop_tokens"),
    "sample_top_p": (".rephrase", "sample_top_p"),
    # Dedup
    "minhash_signature": (".dedup", "minhash_signature"),
    "jaccard_from_signature": (".dedup", "jaccard_from_signature"),
    "stream_near_dedup": (".dedup", "stream_near_dedup"),
    "LSHIndex": (".dedup", "LSHIndex"),
    "exact_hash": (".dedup", "exact_hash"),
    "dedup_exact": (".dedup", "dedup_exact"),
    "dedup_line_level": (".dedup", "dedup_line_level"),
    "dedup_paragraph_level": (".dedup", "dedup_paragraph_level"),
    # Diversity
    "cluster_embeddings": (".diversity", "cluster_embeddings"),
    "coverage_metrics": (".diversity", "coverage_metrics"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    module = importlib.import_module(module_name, __name__)
    return getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_EXPORTS.keys()))

__all__ = [
    # Loader
    "build_loader",
    "DeterministicLoader",
    # Mixture
    "MixturePlan",
    "resolve_plan",
    "populate_paths",
    # Dataset
    "NumpyFSLDataset",
    "Cursor",
    # Index
    "IndexReader",
    "IndexWriter",
    "regenerate_index_from_shard",
    # Sources
    "DataSource",
    "Document",
    "HuggingFaceSource",
    "HfFileSystemSource",
    "JSONLSource",
    "TextFileSource",
    "ArrowSource",
    "HydraScore",
    "HydraScoringSource",
    "HydraFilteredSource",
    "create_source",
    # Sinks
    "ShardWriter",
    "ShardedWriter",
    "ShardInfo",
    "ManifestInfo",
    "load_manifest",
    "verify_manifest",
    # Preprocessing
    "PrepConfig",
    "PrepPipeline",
    "ParallelPrepPipeline",
    # Transforms
    "normalize_text",
    "tokenize",
    "pack_document",
    "get_shard_id",
    "make_file_name",
    # Scoring
    "build_prompt",
    "right_trim",
    "grade_prompts",
    "compute_aggregated",
    "compute_aggregated_code",
    # Rephrasing
    "format_prompt",
    "create_parser",
    "stop_tokens",
    "sample_top_p",
    # Dedup
    "minhash_signature",
    "jaccard_from_signature",
    "stream_near_dedup",
    "LSHIndex",
    "exact_hash",
    "dedup_exact",
    "dedup_line_level",
    "dedup_paragraph_level",
    # Diversity
    "cluster_embeddings",
    "coverage_metrics",
]
