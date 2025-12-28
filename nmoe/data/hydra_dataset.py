"""HYDRA-vNext training sources.

This module is intentionally small:
- It defines the minimal sample types and data sources needed to implement
  HYDRA_TRAINING.md without duplicating the main dataprep pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

try:
  import pyarrow.parquet as pq
except ImportError:
  pq = None  # type: ignore


DOMAIN_MAP: dict[str, int] = {"web": 0, "code": 1, "science": 2, "math": 3}


@dataclass(frozen=True)
class HydraSample:
  text: str
  domain: str
  input_ids: torch.Tensor | None = None

  fasttext_dclm: float | None = None
  fasttext_edu: float | None = None
  fasttext_code: float | None = None
  fasttext_math: float | None = None
  fasttext_english: float | None = None
  lang_score: float | None = None
  extraction_artifacts: int | None = None
  missing_content: int | None = None

  oracle_5dim: tuple[float, float, float, float, float] | None = None

  fw_int_score: int | None = None
  token_count: int | None = None


@dataclass(frozen=True)
class HydraPair:
  a: HydraSample
  b: HydraSample


def _extract_nested(row: dict[str, Any], path: str) -> Any:
  val: Any = row
  for key in path.split("."):
    if not isinstance(val, dict):
      return None
    val = val.get(key)
  return val


def _extract_tax_primary_code(row: dict[str, Any], field: str) -> int | None:
  code = _extract_nested(row, f"eai_taxonomy.{field}.primary.code")
  if code is None:
    return None
  try:
    return int(code)
  except (TypeError, ValueError):
    return None


class EAIParquetSource:
  def __init__(self, glob_pattern: str, *, domain: str, max_ctx: int = 4096):
    if pq is None:
      raise ImportError("pyarrow is required for EAI parquet reading")
    self._paths = sorted(Path("/").glob(glob_pattern.lstrip("/")))
    self._domain = domain
    self._max_chars = int(max_ctx) * 4

  def __iter__(self) -> Iterator[HydraSample]:
    for path in self._paths:
      pf = pq.ParquetFile(path)
      for batch in pf.iter_batches(batch_size=1024):
        cols = batch.to_pydict()
        n = len(cols.get("text", []))
        for i in range(n):
          text = cols.get("text", [None])[i]
          if not text:
            continue
          row = {k: (v[i] if i < len(v) else None) for k, v in cols.items()}
          yield HydraSample(
            text=text[: self._max_chars],
            domain=self._domain,
            fasttext_dclm=_extract_nested(row, "quality_signals.fasttext.dclm"),
            fasttext_edu=_extract_nested(row, "quality_signals.fasttext.fineweb_edu_approx"),
            fasttext_code=_extract_nested(row, "quality_signals.fasttext.eai_web_code"),
            fasttext_math=_extract_nested(row, "quality_signals.fasttext.eai_general_math"),
            fasttext_english=_extract_nested(row, "quality_signals.fasttext.english"),
            extraction_artifacts=_extract_tax_primary_code(row, "extraction_artifacts"),
            missing_content=_extract_tax_primary_code(row, "missing_content"),
          )


class FineWebEduScore2Source:
  LENGTH_BUCKETS: list[tuple[int, int]] = [
    (0, 256),
    (256, 512),
    (512, 1024),
    (1024, 2048),
    (2048, 4096),
  ]

  def __init__(self, glob_pattern: str, *, max_ctx: int = 4096, min_lang_score: float = 0.9):
    if pq is None:
      raise ImportError("pyarrow is required for FineWeb parquet reading")
    self._paths = sorted(Path("/").glob(glob_pattern.lstrip("/")))
    self._max_chars = int(max_ctx) * 4
    self._min_lang_score = float(min_lang_score)

  def __iter__(self) -> Iterator[HydraSample]:
    for path in self._paths:
      pf = pq.ParquetFile(path)
      for batch in pf.iter_batches(batch_size=1024):
        cols = batch.to_pydict()
        n = len(cols.get("text", []))
        for i in range(n):
          text = cols.get("text", [None])[i]
          if not text:
            continue
          lang_score = cols.get("language_score", [None])[i]
          if lang_score is None or float(lang_score) < self._min_lang_score:
            continue
          yield HydraSample(
            text=text[: self._max_chars],
            domain="web",
            lang_score=float(lang_score),
            fw_int_score=cols.get("int_score", [None])[i],
            token_count=cols.get("token_count", [None])[i],
          )

  @staticmethod
  def length_bucket(token_count: int | None) -> int:
    if token_count is None:
      return 2
    t = int(token_count)
    for i, (lo, hi) in enumerate(FineWebEduScore2Source.LENGTH_BUCKETS):
      if lo <= t < hi:
        return i
    return len(FineWebEduScore2Source.LENGTH_BUCKETS) - 1


class OracleLabelSource:
  DIMS: list[str] = ["helpfulness", "correctness", "coherence", "complexity", "density"]

  def __init__(self, labels_jsonl: str, *, data_root: str, max_ctx: int = 4096):
    from .docid import parse_doc_id, shard_path

    self._labels_jsonl = labels_jsonl
    self._data_root = data_root
    self._max_ctx = int(max_ctx)
    self._parse_doc_id = parse_doc_id
    self._shard_path = shard_path

  def __iter__(self) -> Iterator[HydraSample]:
    with open(self._labels_jsonl, "r", encoding="utf-8") as f:
      for line in f:
        if not line.strip():
          continue
        j = json.loads(line)
        scores = j.get("scores")
        doc_id = j.get("doc_id")
        if not scores or not doc_id:
          continue

        if isinstance(scores, dict):
          vals = tuple(float(scores.get(k, 0.0)) for k in self.DIMS)
        elif isinstance(scores, list) and len(scores) == 5:
          vals = tuple(float(scores[i]) for i in range(5))
        else:
          continue

        doc = self._parse_doc_id(doc_id)
        path = self._shard_path(self._data_root, doc)
        arr = np.load(path, mmap_mode="r")
        toks = arr[int(doc.start) : int(doc.end)].astype(np.int64)
        if len(toks) > self._max_ctx:
          toks = toks[: self._max_ctx]

        source = doc.source.lower()
        if "code" in source or "stack" in source:
          domain = "code"
        elif "math" in source:
          domain = "math"
        elif "arxiv" in source or "stem" in source:
          domain = "science"
        else:
          domain = "web"

        yield HydraSample(
          text="",
          domain=domain,
          input_ids=torch.from_numpy(toks.copy()).long(),
          oracle_5dim=vals,  # type: ignore[arg-type]
        )

