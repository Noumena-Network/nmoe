"""
arXiv data sources for multi-format processing.

Provides sources for:
- ArxivS3Source: Stream from downloaded S3 LaTeX tars
- Ar5ivHTMLSource: Stream from ar5iv HTML dataset or API
- ArxivMultiFormatSource: Emit both formats as equivalent documents
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator, Set, Tuple

from ..sources import DataSource, Document
from .latex import latex_to_text, find_main_tex, extract_tex_metadata
from .html import html_to_text, extract_html_metadata

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 2_000_000  # hard cap to avoid pathological expansions / OOM


@dataclass
class ArxivPaper:
    """Represents a processed arXiv paper."""
    arxiv_id: str
    latex_text: Optional[str] = None
    html_text: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    categories: Optional[List[str]] = None


class ArxivS3Source(DataSource):
    """
    Stream documents from arXiv S3 bulk LaTeX source tars.

    The S3 bucket contains tar files like arXiv_src_YYMM_NNN.tar,
    each containing nested .tar.gz files per paper.

    Args:
        tar_dir: Directory containing downloaded arXiv_src_*.tar files
        year_month_filter: Optional list like ["2301", "2302"] to filter by date
        worker_index: Index of this worker (for K8s indexed jobs)
        num_workers: Total number of parallel workers
        max_docs: Optional limit for testing
    """

    def __init__(
        self,
        tar_dir: str | Path,
        *,
        year_month_filter: Optional[List[str]] = None,
        worker_index: int = 0,
        num_workers: int = 1,
        max_docs: Optional[int] = None,
    ):
        self.tar_dir = Path(tar_dir)
        self.year_month_filter = set(year_month_filter) if year_month_filter else None
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.max_docs = max_docs

        # Stats tracking
        self._stats = {
            "processed": 0,
            "success": 0,
            "parse_errors": 0,
            "empty_docs": 0,
        }

    @property
    def name(self) -> str:
        return "arxiv_s3_latex"

    def estimate_size(self) -> int | None:
        # Rough estimate: ~2.3M papers total, distributed across workers
        return 2_300_000 // self.num_workers

    def _get_tar_files(self) -> List[Path]:
        """Get list of tar files assigned to this worker."""
        all_tars = sorted(self.tar_dir.glob("arXiv_src_*.tar"))

        if self.year_month_filter:
            # Filter by year_month in filename (e.g., arXiv_src_2301_001.tar)
            filtered = []
            for tar in all_tars:
                match = re.search(r'arXiv_src_(\d{4})_', tar.name)
                if match and match.group(1) in self.year_month_filter:
                    filtered.append(tar)
            all_tars = filtered

        # Partition across workers
        worker_tars = [t for i, t in enumerate(all_tars) if i % self.num_workers == self.worker_index]
        return worker_tars

    def __iter__(self) -> Iterator[Document]:
        """Yield documents from S3 tar files."""
        tar_files = self._get_tar_files()
        logger.info(f"ArxivS3Source worker {self.worker_index}/{self.num_workers}: processing {len(tar_files)} tar files")

        doc_count = 0

        for tar_path in tar_files:
            try:
                for doc in self._process_tar(tar_path):
                    yield doc
                    doc_count += 1

                    if self.max_docs and doc_count >= self.max_docs:
                        logger.info(f"Reached max_docs limit: {self.max_docs}")
                        return

            except Exception as e:
                logger.error(f"Failed to process tar {tar_path}: {e}")
                continue

        logger.info(f"ArxivS3Source stats: {self._stats}")

    def _process_tar(self, tar_path: Path) -> Iterator[Document]:
        """Process a single arXiv tar file containing nested paper .tar.gz files."""
        with tarfile.open(tar_path, 'r') as outer_tar:
            for member in outer_tar:
                if not member.name.endswith('.tar.gz') and not member.name.endswith('.gz'):
                    continue

                # Extract arxiv_id from path (e.g., "2301/2301.12345.tar.gz")
                arxiv_id = self._extract_arxiv_id(member.name)
                if not arxiv_id:
                    continue

                try:
                    # Extract nested tar.gz
                    f = outer_tar.extractfile(member)
                    if f is None:
                        continue

                    try:
                        doc = self._process_paper_archive(arxiv_id, f)
                        if doc:
                            yield doc
                    finally:
                        try:
                            f.close()
                        except Exception:
                            pass

                except Exception as e:
                    logger.debug(f"Failed to process {arxiv_id}: {e}")
                    self._stats["parse_errors"] += 1

    def _extract_arxiv_id(self, path: str) -> Optional[str]:
        """Extract arXiv ID from file path."""
        # Pattern: YYMM/YYMM.NNNNN.tar.gz or YYMM/YYMM.NNNNNvN.tar.gz
        match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', path)
        if match:
            return match.group(1)
        return None

    # Extensions we actually need for LaTeX processing
    _TEX_EXTENSIONS = frozenset({'.tex', '.bib', '.sty', '.cls', '.bst', '.def', '.cfg'})

    def _is_text_file(self, name: str) -> bool:
        """Check if file is a text file we need (skip binary figures)."""
        name_lower = name.lower()
        # Check extension
        for ext in self._TEX_EXTENSIONS:
            if name_lower.endswith(ext):
                return True
        # Also accept extensionless files (might be tex)
        if '.' not in name.split('/')[-1]:
            return True
        return False

    _MAX_FILE_BYTES = 2 * 1024 * 1024       # 2MB per file - skip pathologically large tex
    _MAX_TOTAL_TEX_BYTES = 8 * 1024 * 1024  # 8MB total per paper - skip monster papers

    def _process_paper_archive(self, arxiv_id: str, archive_f) -> Optional[Document]:
        """Process a paper's .tar.gz archive and extract text.

        Important: This function must be streaming and memory-bounded.
        Many arXiv paper archives are large; reading the full nested .tar.gz
        into memory will OOM at scale.
        """
        self._stats["processed"] += 1

        # Read only text files from the archive (skip binary figures to save memory)
        files: Dict[str, bytes] = {}

        try:
            buf = io.BufferedReader(archive_f)
            is_gz = (buf.peek(2)[:2] == b'\x1f\x8b')

            def read_fileobj_limited(fileobj, limit: int) -> bytes:
                if limit <= 0:
                    return b""
                chunks: list[bytes] = []
                total = 0
                while True:
                    # Keep reads small to avoid large transient allocations.
                    chunk = fileobj.read(min(1024 * 1024, limit - total))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= limit:
                        break
                return b"".join(chunks)

            def reject() -> Optional[Document]:
                self._stats["parse_errors"] += 1
                return None

            total_bytes = 0

            if is_gz:
                with gzip.GzipFile(fileobj=buf, mode='rb') as gz:
                    # Peek at the first tar header without decompressing the full stream.
                    header = gz.read(512)
                    is_tar = False
                    if len(header) == 512:
                        try:
                            tarfile.TarInfo.frombuf(header, encoding='utf-8', errors='surrogateescape')
                            is_tar = True
                        except Exception:
                            is_tar = False

                    if is_tar:
                        # Chain the already-read header back onto the remaining gz stream.
                        # CRITICAL: Cap total decompressed bytes to prevent gzip bombs.
                        _MAX_DECOMP = self._MAX_TOTAL_TEX_BYTES * 2  # 16MB cap on decompression
                        class _Chain(io.RawIOBase):
                            def __init__(self, prefix: bytes, rest, limit: int):
                                self._p = io.BytesIO(prefix)
                                self._r = rest
                                self._limit = limit
                                self._read = len(prefix)
                            def readable(self):
                                return True
                            def read(self, n=-1):
                                if self._read >= self._limit:
                                    return b""  # Hit limit, return EOF
                                b = self._p.read(n)
                                if b:
                                    self._read += len(b)
                                    return b
                                remaining = self._limit - self._read
                                if n == -1 or n > remaining:
                                    n = remaining
                                b = self._r.read(n)
                                self._read += len(b) if b else 0
                                return b
                        stream = _Chain(header, gz, _MAX_DECOMP)
                        with tarfile.open(fileobj=stream, mode='r|*') as inner_tar:
                            for m in inner_tar:
                                if not m.isfile() or not self._is_text_file(m.name):
                                    continue
                                if m.size is not None and m.size > self._MAX_FILE_BYTES:
                                    return reject()
                                f = inner_tar.extractfile(m)
                                if not f:
                                    continue
                                data = read_fileobj_limited(f, self._MAX_FILE_BYTES)
                                total_bytes += len(data)
                                if total_bytes > self._MAX_TOTAL_TEX_BYTES:
                                    return reject()
                                files[m.name] = data
                    else:
                        # Single file gzip: read up to cap.
                        rest = read_fileobj_limited(gz, self._MAX_TOTAL_TEX_BYTES - len(header))
                        content = header + rest
                        if len(content) >= self._MAX_TOTAL_TEX_BYTES:
                            return reject()
                        files['main.tex'] = content
            else:
                # Raw tar stream.
                with tarfile.open(fileobj=buf, mode='r|*') as tar:
                    for m in tar:
                        if not m.isfile() or not self._is_text_file(m.name):
                            continue
                        if m.size is not None and m.size > self._MAX_FILE_BYTES:
                            return reject()
                        f = tar.extractfile(m)
                        if not f:
                            continue
                        data = read_fileobj_limited(f, self._MAX_FILE_BYTES)
                        total_bytes += len(data)
                        if total_bytes > self._MAX_TOTAL_TEX_BYTES:
                            return reject()
                        files[m.name] = data
        except Exception as e:
            logger.debug(f"Archive extraction failed for {arxiv_id}: {e}")
            return None

        if not files:
            return None

        # Find main tex file
        main_tex = find_main_tex(files)
        if not main_tex:
            self._stats["parse_errors"] += 1
            return None

        # Get main content
        try:
            content = files[main_tex].decode('utf-8', errors='ignore')
        except Exception:
            self._stats["parse_errors"] += 1
            return None

        # Create include resolver for multi-file documents
        def resolve_include(filename: str) -> Optional[str]:
            # Try exact match
            if filename in files:
                return files[filename].decode('utf-8', errors='ignore')
            # Try with .tex extension
            if not filename.endswith('.tex'):
                tex_name = filename + '.tex'
                if tex_name in files:
                    return files[tex_name].decode('utf-8', errors='ignore')
            # Try in subdirectories
            for path, data in files.items():
                if path.endswith('/' + filename) or path.endswith('/' + filename + '.tex'):
                    return data.decode('utf-8', errors='ignore')
            return None

        # Convert to text
        try:
            text = latex_to_text(
                content,
                preserve_math=True,
                expand_includes=resolve_include,
            )
        except Exception as e:
            logger.debug(f"LaTeX conversion failed for {arxiv_id}: {e}")
            self._stats["parse_errors"] += 1
            return None

        if len(text) > _MAX_TEXT_CHARS:
            self._stats["parse_errors"] += 1
            return None

        if not text or len(text.strip()) < 100:
            self._stats["empty_docs"] += 1
            return None

        # Extract metadata
        try:
            metadata = extract_tex_metadata(content)
        except Exception:
            metadata = None

        self._stats["success"] += 1

        return Document(
            doc_id=f"arxiv:{arxiv_id}:latex",
            text=text,
            metadata={
                "arxiv_id": arxiv_id,
                "format": "latex",
                "title": metadata.title if metadata else None,
                "authors": metadata.authors if metadata else None,
                "abstract": metadata.abstract if metadata else None,
            }
        )


class Ar5ivHTMLSource(DataSource):
    """
    Stream documents from ar5iv HTML dataset.

    The ar5iv dataset contains HTML5+MathML conversions organized by quality:
    - no_problem: Clean conversions (~72%)
    - warning: Minor issues (~23%)
    - error: Conversion errors (~5%)

    Args:
        html_dir: Directory containing ar5iv HTML files
        quality_levels: Which quality tiers to include
        worker_index: Index of this worker (for K8s indexed jobs)
        num_workers: Total number of parallel workers
        max_docs: Optional limit for testing
    """

    def __init__(
        self,
        html_dir: str | Path,
        *,
        quality_levels: List[str] = ["no_problem", "warning"],
        worker_index: int = 0,
        num_workers: int = 1,
        max_docs: Optional[int] = None,
    ):
        self.html_dir = Path(html_dir)
        self.quality_levels = quality_levels
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.max_docs = max_docs

        self._stats = {
            "processed": 0,
            "success": 0,
            "parse_errors": 0,
            "empty_docs": 0,
        }

    @property
    def name(self) -> str:
        return "arxiv_ar5iv_html"

    def estimate_size(self) -> int | None:
        # ~1.7M papers with HTML, distributed across workers
        return 1_700_000 // self.num_workers

    def _get_html_files(self) -> List[Path]:
        """Get list of HTML files assigned to this worker."""
        all_files = []

        for quality in self.quality_levels:
            quality_dir = self.html_dir / quality
            if quality_dir.exists():
                # ar5iv structure: quality/YYMM/arxiv_id.html
                all_files.extend(quality_dir.glob("**/*.html"))

        all_files = sorted(all_files)

        # Partition across workers
        worker_files = [f for i, f in enumerate(all_files) if i % self.num_workers == self.worker_index]
        return worker_files

    def __iter__(self) -> Iterator[Document]:
        """Yield documents from ar5iv HTML files."""
        html_files = self._get_html_files()
        logger.info(f"Ar5ivHTMLSource worker {self.worker_index}/{self.num_workers}: processing {len(html_files)} HTML files")

        doc_count = 0

        for html_path in html_files:
            try:
                doc = self._process_html_file(html_path)
                if doc:
                    yield doc
                    doc_count += 1

                    if self.max_docs and doc_count >= self.max_docs:
                        logger.info(f"Reached max_docs limit: {self.max_docs}")
                        return

            except Exception as e:
                logger.debug(f"Failed to process {html_path}: {e}")
                self._stats["parse_errors"] += 1

        logger.info(f"Ar5ivHTMLSource stats: {self._stats}")

    def _process_html_file(self, html_path: Path) -> Optional[Document]:
        """Process a single ar5iv HTML file."""
        self._stats["processed"] += 1

        # Extract arxiv_id from filename
        arxiv_id = html_path.stem
        # Normalize: remove version suffix if present
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)

        # Read HTML
        try:
            html_content = html_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.debug(f"Failed to read {html_path}: {e}")
            return None

        # Convert to text
        try:
            text = html_to_text(html_content, preserve_math=True)
        except Exception as e:
            logger.debug(f"HTML conversion failed for {arxiv_id}: {e}")
            self._stats["parse_errors"] += 1
            return None

        if len(text) > _MAX_TEXT_CHARS:
            self._stats["parse_errors"] += 1
            return None

        if not text or len(text.strip()) < 100:
            self._stats["empty_docs"] += 1
            return None

        # Extract metadata
        try:
            metadata = extract_html_metadata(html_content)
        except Exception:
            metadata = None

        self._stats["success"] += 1

        return Document(
            doc_id=f"arxiv:{arxiv_id}:html",
            text=text,
            metadata={
                "arxiv_id": arxiv_id,
                "format": "html",
                "title": metadata.title if metadata else None,
                "authors": metadata.authors if metadata else None,
                "abstract": metadata.abstract if metadata else None,
                "categories": metadata.categories if metadata else None,
            }
        )


class ArxivMultiFormatSource(DataSource):
    """
    Unified arXiv source that emits BOTH formats as separate documents.

    For papers with both LaTeX and HTML available, yields two documents:
    - arxiv:{id}:latex - LaTeX source processed to text
    - arxiv:{id}:html - ar5iv HTML processed to text

    This allows the model to learn that both formats represent equivalent content.

    Args:
        s3_tar_dir: Directory with downloaded S3 LaTeX tars
        ar5iv_dir: Directory with ar5iv HTML dataset
        emit_both_formats: If True, emit both formats when available
        worker_index: Index of this worker (for K8s indexed jobs)
        num_workers: Total number of parallel workers
        max_docs: Optional limit for testing
    """

    def __init__(
        self,
        s3_tar_dir: str | Path,
        ar5iv_dir: str | Path,
        *,
        emit_both_formats: bool = True,
        quality_levels: List[str] = ["no_problem", "warning"],
        worker_index: int = 0,
        num_workers: int = 1,
        max_docs: Optional[int] = None,
    ):
        self.s3_tar_dir = Path(s3_tar_dir)
        self.ar5iv_dir = Path(ar5iv_dir)
        self.emit_both_formats = emit_both_formats
        self.quality_levels = quality_levels
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.max_docs = max_docs

        self._stats = {
            "latex_only": 0,
            "html_only": 0,
            "both_formats": 0,
            "total_docs": 0,
        }

    @property
    def name(self) -> str:
        return "arxiv_multiformat"

    def estimate_size(self) -> int | None:
        # ~4M documents total (papers × 1.75 for dual format)
        return 4_000_000 // self.num_workers

    def __iter__(self) -> Iterator[Document]:
        """
        Yield documents from both sources.

        Strategy:
        1. Build index of available HTML files by arxiv_id
        2. Iterate through LaTeX tars
        3. For each paper: emit LaTeX doc, then HTML doc if available
        4. Track papers seen to avoid duplicates
        """
        # First, build HTML availability index
        html_index = self._build_html_index()
        logger.info(f"ArxivMultiFormatSource: {len(html_index)} papers have HTML available")

        # Create LaTeX source
        latex_source = ArxivS3Source(
            self.s3_tar_dir,
            worker_index=self.worker_index,
            num_workers=self.num_workers,
        )

        # Create HTML source (we'll use it to look up specific papers)
        seen_arxiv_ids: Set[str] = set()
        doc_count = 0

        # Iterate through LaTeX documents
        for latex_doc in latex_source:
            arxiv_id = latex_doc.metadata.get("arxiv_id") if latex_doc.metadata else None

            if not arxiv_id:
                continue

            if arxiv_id in seen_arxiv_ids:
                continue
            seen_arxiv_ids.add(arxiv_id)

            # Emit LaTeX document
            yield latex_doc
            doc_count += 1

            if self.max_docs and doc_count >= self.max_docs:
                return

            # Check if HTML is available
            if self.emit_both_formats and arxiv_id in html_index:
                html_path = html_index[arxiv_id]
                html_doc = self._load_html_doc(arxiv_id, html_path)

                if html_doc:
                    # Mark as having paired format
                    if latex_doc.metadata:
                        latex_doc.metadata["has_paired_format"] = True
                    if html_doc.metadata:
                        html_doc.metadata["has_paired_format"] = True

                    yield html_doc
                    doc_count += 1
                    self._stats["both_formats"] += 1

                    if self.max_docs and doc_count >= self.max_docs:
                        return
                else:
                    self._stats["latex_only"] += 1
            else:
                self._stats["latex_only"] += 1

        self._stats["total_docs"] = doc_count
        logger.info(f"ArxivMultiFormatSource stats: {self._stats}")

    def _build_html_index(self) -> Dict[str, Path]:
        """Build index mapping arxiv_id to HTML file path."""
        index = {}

        for quality in self.quality_levels:
            quality_dir = self.ar5iv_dir / quality
            if not quality_dir.exists():
                continue

            for html_path in quality_dir.glob("**/*.html"):
                arxiv_id = html_path.stem
                # Remove version suffix
                arxiv_id = re.sub(r'v\d+$', '', arxiv_id)

                # Only keep first occurrence (prefer higher quality)
                if arxiv_id not in index:
                    index[arxiv_id] = html_path

        return index

    def _load_html_doc(self, arxiv_id: str, html_path: Path) -> Optional[Document]:
        """Load and process a single HTML document."""
        try:
            html_content = html_path.read_text(encoding='utf-8', errors='ignore')
            text = html_to_text(html_content, preserve_math=True)

            if not text or len(text.strip()) < 100:
                return None

            metadata = extract_html_metadata(html_content)

            return Document(
                doc_id=f"arxiv:{arxiv_id}:html",
                text=text,
                metadata={
                    "arxiv_id": arxiv_id,
                    "format": "html",
                    "title": metadata.title if metadata else None,
                    "authors": metadata.authors if metadata else None,
                    "abstract": metadata.abstract if metadata else None,
                    "categories": metadata.categories if metadata else None,
                }
            )
        except Exception as e:
            logger.debug(f"Failed to load HTML for {arxiv_id}: {e}")
            return None


def create_arxiv_source(
    source_type: str,
    **kwargs,
) -> DataSource:
    """
    Factory function for arXiv sources.

    Args:
        source_type: One of "s3", "ar5iv", "multiformat"
        **kwargs: Source-specific arguments

    Returns:
        Configured DataSource instance
    """
    sources = {
        "s3": ArxivS3Source,
        "latex": ArxivS3Source,
        "ar5iv": Ar5ivHTMLSource,
        "html": Ar5ivHTMLSource,
        "multiformat": ArxivMultiFormatSource,
        "combined": ArxivMultiFormatSource,
    }

    if source_type not in sources:
        raise ValueError(f"Unknown arXiv source type: {source_type}. Available: {list(sources.keys())}")

    return sources[source_type](**kwargs)
