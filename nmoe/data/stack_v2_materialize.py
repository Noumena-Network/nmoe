"""Materialize Stack v2 file contents from Software Heritage S3.

Input: The Stack v2 "ids" parquet files under /data/datasets/stack_v2.
Output: jsonl.zst shards under /data/datasets/stack_v2_content/<run_id>/worker_<idx>/.

This is quality-first: select top-N rows by stars/forks (with per-repo caps),
then download content bytes and emit a normal text+metadata corpus.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import boto3

try:
  import pyarrow.parquet as pq
except Exception as e:  # pragma: no cover
  pq = None  # type: ignore
  _pyarrow_err = e


def _now_ms() -> int:
  return int(time.time() * 1000)


def _as_bool(v: str) -> bool:
  s = v.strip().lower()
  if s in ("1", "true", "t", "yes", "y"):
    return True
  if s in ("0", "false", "f", "no", "n"):
    return False
  raise ValueError(f"invalid bool: {v!r}")


def _safe_int(x: Any) -> int:
  if x is None:
    return 0
  try:
    return int(x)
  except Exception:
    return 0


def _safe_str(x: Any) -> str:
  if x is None:
    return ""
  return str(x)


def _is_probably_binary(buf: bytes) -> bool:
  if not buf:
    return True
  if b"\x00" in buf:
    return True
  # If decoding produced lots of replacement bytes, it was likely binary.
  # We can't see replacement chars here; use a cheap heuristic: low printable ratio.
  sample = buf[:4096]
  printable = sum(1 for b in sample if 9 <= b <= 13 or 32 <= b <= 126)
  return printable / max(1, len(sample)) < 0.6


def _decode_text(raw: bytes, enc: str) -> str | None:
  if _is_probably_binary(raw):
    return None
  enc = (enc or "").strip()
  candidates = [enc] if enc else []
  candidates += ["utf-8", "latin-1"]
  for name in candidates:
    try:
      return raw.decode(name, errors="strict")
    except Exception:
      continue
  try:
    return raw.decode("utf-8", errors="replace")
  except Exception:
    return None


def _skip_path(path: str) -> bool:
  p = path.lower()
  bad_dirs = (
    "/node_modules/",
    "/vendor/",
    "/third_party/",
    "/third-party/",
    "/dist/",
    "/build/",
    "/target/",
    "/pods/",
    "/carthage/",
    "/.git/",
  )
  for d in bad_dirs:
    if d in p:
      return True
  bad_files = (
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "cargo.lock",
    "go.sum",
    "poetry.lock",
  )
  base = p.rsplit("/", 1)[-1]
  if base in bad_files:
    return True
  bad_ext = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".mp4",
    ".mp3",
    ".class",
    ".jar",
    ".so",
    ".dll",
    ".exe",
    ".dylib",
    ".o",
    ".a",
    ".ttf",
    ".woff",
    ".woff2",
  )
  for ext in bad_ext:
    if base.endswith(ext):
      return True
  return False


@dataclass(frozen=True)
class FileRef:
  blob_id: str
  src_encoding: str
  path: str
  repo_name: str
  language: str
  is_vendor: bool
  is_generated: bool
  length_bytes: int
  star_events_count: int
  fork_events_count: int
  detected_licenses: list[str] | None
  license_type: str
  snapshot_id: str
  revision_id: str
  branch_name: str

  def score(self) -> float:
    return math.log1p(max(0, self.star_events_count)) + 0.5 * math.log1p(max(0, self.fork_events_count))


class _ZstdJsonlWriter:
  def __init__(self, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    self._out_path = out_path
    self._fh = None
    self._writer = None
    self._proc = None

    try:
      import zstandard as zstd  # type: ignore
    except Exception as e:  # pragma: no cover
      raise RuntimeError(
        "Missing optional dependency `zstandard` (required for writing .jsonl.zst). "
        "Run materialization inside the container image with deps installed."
      ) from e

    self._fh = out_path.open("wb")
    self._writer = zstd.ZstdCompressor(level=3).stream_writer(self._fh)

    self._rows = 0
    self._bytes = 0

  @property
  def rows(self) -> int:
    return self._rows

  @property
  def bytes_written(self) -> int:
    return self._bytes

  def write(self, obj: dict[str, Any]) -> None:
    line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    assert self._writer is not None
    self._writer.write(line)
    self._rows += 1
    self._bytes += len(line)

  def close(self) -> None:
    if self._writer is None:
      return
    try:
      self._writer.close()
    except Exception:
      pass

    if self._fh is not None:
      try:
        self._fh.close()
      except Exception:
        pass

    if self._proc is not None:
      rc = self._proc.wait()
      if rc != 0:
        err = b""
        try:
          err = self._proc.stderr.read() if self._proc.stderr is not None else b""
        except Exception:
          pass
        raise RuntimeError(f"zstd failed (rc={rc}) writing {self._out_path}: {err[:2000]!r}")


def _iter_parquet_paths(input_root: Path) -> list[Path]:
  # HuggingFace hf download layout: /data/datasets/stack_v2/data/.../*.parquet
  data_dir = input_root / "data"
  if not data_dir.exists():
    raise FileNotFoundError(f"missing {data_dir}")
  paths = sorted(data_dir.glob("**/*.parquet"))
  if not paths:
    raise FileNotFoundError(f"no parquet files under {data_dir}")
  return paths


def _assign_paths(paths: list[Path], *, worker_index: int, num_workers: int) -> list[Path]:
  if num_workers <= 0:
    raise ValueError("num_workers must be > 0")
  if worker_index < 0 or worker_index >= num_workers:
    raise ValueError(f"worker_index out of range: {worker_index}/{num_workers}")
  return [p for i, p in enumerate(paths) if (i % num_workers) == worker_index]


def _flatten_row(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
  # Some variants store a list of file dicts under `files`. If present, yield one row per file.
  files = row.get("files")
  if isinstance(files, list):
    for f in files:
      if isinstance(f, dict):
        yield row | f
    return
  yield row


def _select_top_n(
  parquet_paths: list[Path],
  *,
  top_n: int,
  min_bytes: int,
  max_bytes: int,
  exclude_vendor: bool,
  exclude_generated: bool,
  max_files_per_repo: int,
) -> list[FileRef]:
  if pq is None:  # pragma: no cover
    raise RuntimeError(f"pyarrow not available: {_pyarrow_err}")

  # Keep top-N via a tiny heap: store as (score, blob_id, FileRef)
  import heapq

  heap: list[tuple[float, str, FileRef]] = []
  per_repo = defaultdict(int)

  cols = [
    "blob_id",
    "src_encoding",
    "path",
    "repo_name",
    "language",
    "is_vendor",
    "is_generated",
    "length_bytes",
    "star_events_count",
    "fork_events_count",
    "detected_licenses",
    "license_type",
    "snapshot_id",
    "revision_id",
    "branch_name",
    # grouped datasets may have `files` and store some of these inside file dicts;
    # reading extra columns is cheap relative to S3 fetch.
    "files",
  ]

  scanned = 0
  kept = 0
  for path in parquet_paths:
    pf = pq.ParquetFile(path)
    read_cols = [c for c in cols if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=2048, columns=read_cols):
      d = batch.to_pydict()
      n = 0
      for v in d.values():
        if isinstance(v, list):
          n = max(n, len(v))
      for i in range(n):
        row = {k: (v[i] if isinstance(v, list) and i < len(v) else None) for k, v in d.items()}
        for r in _flatten_row(row):
          scanned += 1
          blob_id = _safe_str(r.get("blob_id"))
          if not blob_id:
            continue
          repo = _safe_str(r.get("repo_name"))
          if max_files_per_repo > 0 and repo:
            if per_repo[repo] >= max_files_per_repo:
              continue

          path_in_repo = _safe_str(r.get("path"))
          if path_in_repo and _skip_path(path_in_repo):
            continue

          is_vendor = bool(r.get("is_vendor")) if r.get("is_vendor") is not None else False
          is_generated = bool(r.get("is_generated")) if r.get("is_generated") is not None else False
          if exclude_vendor and is_vendor:
            continue
          if exclude_generated and is_generated:
            continue

          length_bytes = _safe_int(r.get("length_bytes"))
          if length_bytes < min_bytes or length_bytes > max_bytes:
            continue

          ref = FileRef(
            blob_id=blob_id,
            src_encoding=_safe_str(r.get("src_encoding")),
            path=path_in_repo,
            repo_name=repo,
            language=_safe_str(r.get("language") or r.get("gha_language")),
            is_vendor=is_vendor,
            is_generated=is_generated,
            length_bytes=length_bytes,
            star_events_count=_safe_int(r.get("star_events_count")),
            fork_events_count=_safe_int(r.get("fork_events_count")),
            detected_licenses=r.get("detected_licenses") if isinstance(r.get("detected_licenses"), list) else None,
            license_type=_safe_str(r.get("license_type")),
            snapshot_id=_safe_str(r.get("snapshot_id")),
            revision_id=_safe_str(r.get("revision_id")),
            branch_name=_safe_str(r.get("branch_name")),
          )

          score = ref.score()
          key = (score, ref.blob_id)
          if len(heap) < top_n:
            heapq.heappush(heap, (key[0], key[1], ref))
            if max_files_per_repo > 0 and repo:
              per_repo[repo] += 1
            kept += 1
          else:
            if key > (heap[0][0], heap[0][1]):
              _, _, dropped = heapq.heapreplace(heap, (key[0], key[1], ref))
              if max_files_per_repo > 0 and dropped.repo_name:
                per_repo[dropped.repo_name] -= 1
              if max_files_per_repo > 0 and repo:
                per_repo[repo] += 1

    if scanned > 0 and (scanned % 1_000_000) == 0:
      print(f"[stack_v2] scanned={scanned:,} heap={len(heap):,}", flush=True)

  out = [t[2] for t in heap]
  out.sort(key=lambda r: (r.score(), r.blob_id), reverse=True)
  print(f"[stack_v2] selected={len(out):,} scanned={scanned:,}", flush=True)
  return out


def _s3_client() -> Any:
  # boto3 will pick up AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY from env.
  return boto3.client("s3")


def _fetch_blob(s3: Any, blob_id: str) -> bytes | None:
  try:
    # Bucket: softwareheritage, key: content/<blob_id>
    obj = s3.get_object(Bucket="softwareheritage", Key=f"content/{blob_id}")
    body = obj["Body"]
    raw = body.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
      return gz.read()
  except Exception:
    return None


def _materialize(
  refs: list[FileRef],
  *,
  out_dir: Path,
  worker_index: int,
  max_rows_per_shard: int = 50_000,
  max_uncompressed_bytes_per_shard: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
  s3 = _s3_client()

  out_dir.mkdir(parents=True, exist_ok=True)
  shard = 0
  writer = _ZstdJsonlWriter(out_dir / f"part_{shard:05d}.jsonl.zst")
  wrote = 0
  fetched = 0
  skipped = 0
  failures = 0

  def rotate() -> None:
    nonlocal shard, writer
    writer.close()
    shard += 1
    writer = _ZstdJsonlWriter(out_dir / f"part_{shard:05d}.jsonl.zst")

  t0 = _now_ms()
  for i, ref in enumerate(refs, start=1):
    raw = _fetch_blob(s3, ref.blob_id)
    fetched += 1
    if raw is None:
      failures += 1
      continue

    text = _decode_text(raw, ref.src_encoding)
    if not text:
      skipped += 1
      continue

    row = {
      "id": ref.blob_id,
      "text": text,
      "metadata": {
        "repo_name": ref.repo_name,
        "path": ref.path,
        "language": ref.language,
        "is_vendor": ref.is_vendor,
        "is_generated": ref.is_generated,
        "length_bytes": ref.length_bytes,
        "star_events_count": ref.star_events_count,
        "fork_events_count": ref.fork_events_count,
        "detected_licenses": ref.detected_licenses,
        "license_type": ref.license_type,
        "snapshot_id": ref.snapshot_id,
        "revision_id": ref.revision_id,
        "branch_name": ref.branch_name,
      },
    }
    writer.write(row)
    wrote += 1

    if writer.rows >= max_rows_per_shard or writer.bytes_written >= max_uncompressed_bytes_per_shard:
      rotate()

    if i == 1 or (i % 1000) == 0:
      dt = (_now_ms() - t0) / 1000.0
      rps = wrote / max(dt, 1e-6)
      print(
        f"[stack_v2] worker={worker_index} i={i}/{len(refs)} wrote={wrote} failed={failures} "
        f"skipped={skipped} rows/s={rps:.1f}",
        flush=True,
      )

  writer.close()

  dt = (_now_ms() - t0) / 1000.0
  return {
    "worker_index": worker_index,
    "selected": len(refs),
    "fetched": fetched,
    "wrote": wrote,
    "skipped": skipped,
    "failures": failures,
    "seconds": dt,
  }


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--input-root", required=True, help="Path to /data/datasets/stack_v2 (IDs parquet tree)")
  p.add_argument("--output-root", required=True, help="Output root (run dir) for stack_v2_content")
  p.add_argument("--worker-index", type=int, required=True)
  p.add_argument("--num-workers", type=int, required=True)
  p.add_argument("--top-n", type=int, required=True, help="Top-N files (by stars/forks) per worker")
  p.add_argument("--min-bytes", type=int, default=256)
  p.add_argument("--max-bytes", type=int, default=256_000)
  p.add_argument("--exclude-vendor", type=_as_bool, default=True)
  p.add_argument("--exclude-generated", type=_as_bool, default=True)
  p.add_argument("--max-files-per-repo", type=int, default=200)
  args = p.parse_args(argv)

  input_root = Path(args.input_root)
  output_root = Path(args.output_root)
  worker_index = int(args.worker_index)
  num_workers = int(args.num_workers)

  if pq is None:  # pragma: no cover
    raise RuntimeError(f"pyarrow not available: {_pyarrow_err}")

  all_paths = _iter_parquet_paths(input_root)
  my_paths = _assign_paths(all_paths, worker_index=worker_index, num_workers=num_workers)
  if not my_paths:
    print(f"[stack_v2] worker={worker_index} no parquet paths assigned", flush=True)
    return 0

  print(f"[stack_v2] worker={worker_index} paths={len(my_paths)}/{len(all_paths)}", flush=True)

  refs = _select_top_n(
    my_paths,
    top_n=int(args.top_n),
    min_bytes=int(args.min_bytes),
    max_bytes=int(args.max_bytes),
    exclude_vendor=bool(args.exclude_vendor),
    exclude_generated=bool(args.exclude_generated),
    max_files_per_repo=int(args.max_files_per_repo),
  )

  out_dir = output_root / f"worker_{worker_index:03d}"
  metrics = _materialize(refs, out_dir=out_dir, worker_index=worker_index)
  (out_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
  print(json.dumps(metrics, indent=2), flush=True)
  return 0


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
