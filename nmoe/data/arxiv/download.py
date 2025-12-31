"""
Download utilities for arXiv data.

Supports:
- S3 bulk access (requester-pays bucket)
- ar5iv HTML dataset download
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# S3 bucket configuration
ARXIV_S3_BUCKET = "arxiv"
ARXIV_S3_REGION = "us-east-1"
ARXIV_SRC_PREFIX = "src/"
ARXIV_PDF_PREFIX = "pdf/"

# ar5iv dataset URLs
AR5IV_BASE_URL = "https://sigmathling.kwarc.info/resources/ar5iv-dataset-2024"


@dataclass
class S3TarInfo:
    """Information about an arXiv S3 tar file."""
    key: str
    size_bytes: int
    year_month: str
    sequence: int

    @property
    def filename(self) -> str:
        return os.path.basename(self.key)


def parse_s3_manifest(manifest_path: Path) -> List[S3TarInfo]:
    """
    Parse arXiv S3 manifest XML for file inventory.

    Args:
        manifest_path: Path to downloaded manifest XML

    Returns:
        List of S3TarInfo objects
    """
    tree = ET.parse(manifest_path)
    root = tree.getroot()

    files = []
    for file_elem in root.findall('.//file'):
        # Get filename from nested element
        filename_elem = file_elem.find('filename')
        size_elem = file_elem.find('size')
        yymm_elem = file_elem.find('yymm')
        seq_elem = file_elem.find('seq_num')

        if filename_elem is None or filename_elem.text is None:
            continue

        filename = filename_elem.text
        size = int(size_elem.text) if size_elem is not None and size_elem.text else 0
        yymm = yymm_elem.text if yymm_elem is not None and yymm_elem.text else ""
        seq = int(seq_elem.text) if seq_elem is not None and seq_elem.text else 0

        # Key is the filename (e.g., "src/arXiv_src_0001_001.tar")
        files.append(S3TarInfo(
            key=filename,
            size_bytes=size,
            year_month=yymm,
            sequence=seq,
        ))

    return sorted(files, key=lambda f: (f.year_month, f.sequence))


def download_s3_manifest(
    output_path: Path,
    manifest_type: str = "src",
    aws_profile: Optional[str] = None,
) -> Path:
    """
    Download arXiv S3 manifest file.

    Args:
        output_path: Directory to save manifest
        manifest_type: "src" for source files, "pdf" for PDFs
        aws_profile: AWS profile name (optional)

    Returns:
        Path to downloaded manifest
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_key = f"{manifest_type}/arXiv_{manifest_type}_manifest.xml"
    local_path = output_path / f"arXiv_{manifest_type}_manifest.xml"

    cmd = [
        "aws", "s3", "cp",
        f"s3://{ARXIV_S3_BUCKET}/{manifest_key}",
        str(local_path),
        "--request-payer", "requester",
        "--region", ARXIV_S3_REGION,
    ]

    if aws_profile:
        cmd.extend(["--profile", aws_profile])

    logger.info(f"Downloading manifest: {manifest_key}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to download manifest: {result.stderr}")

    logger.info(f"Manifest saved to: {local_path}")
    return local_path


def download_s3_tars(
    output_dir: Path,
    *,
    year_months: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    max_workers: int = 4,
    aws_profile: Optional[str] = None,
    skip_existing: bool = True,
) -> List[Path]:
    """
    Download arXiv S3 source tars (requester-pays).

    Args:
        output_dir: Directory to save tar files
        year_months: Optional list of YYMM to filter (e.g., ["2301", "2302"])
        max_files: Maximum number of tar files to download
        max_workers: Number of parallel downloads
        aws_profile: AWS profile name (optional)
        skip_existing: Skip already downloaded files

    Returns:
        List of downloaded file paths

    Note:
        Requires AWS credentials with s3:GetObject permission.
        Typical cost: ~$0.09/GB for data transfer out.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download manifest first
    manifest_path = download_s3_manifest(output_dir, "src", aws_profile)
    files = parse_s3_manifest(manifest_path)

    # Filter by year_month if specified
    if year_months:
        year_months_set = set(year_months)
        files = [f for f in files if f.year_month in year_months_set]

    # Apply max_files limit
    if max_files:
        files = files[:max_files]

    logger.info(f"Downloading {len(files)} tar files to {output_dir}")

    # Calculate total size
    total_size_gb = sum(f.size_bytes for f in files) / (1024**3)
    logger.info(f"Total size: {total_size_gb:.2f} GB (estimated cost: ${total_size_gb * 0.09:.2f})")

    downloaded = []

    def download_file(file_info: S3TarInfo) -> Optional[Path]:
        local_path = output_dir / file_info.filename

        if skip_existing and local_path.exists():
            expected_size = file_info.size_bytes
            actual_size = local_path.stat().st_size
            if actual_size == expected_size:
                logger.debug(f"Skipping existing: {file_info.filename}")
                return local_path

        cmd = [
            "aws", "s3", "cp",
            f"s3://{ARXIV_S3_BUCKET}/{file_info.key}",
            str(local_path),
            "--request-payer", "requester",
            "--region", ARXIV_S3_REGION,
        ]

        if aws_profile:
            cmd.extend(["--profile", aws_profile])

        logger.info(f"Downloading: {file_info.filename} ({file_info.size_bytes / (1024**2):.1f} MB)")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Failed to download {file_info.filename}: {result.stderr}")
            return None

        return local_path

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, f): f for f in files}

        for future in as_completed(futures):
            file_info = futures[future]
            try:
                path = future.result()
                if path:
                    downloaded.append(path)
            except Exception as e:
                logger.error(f"Download failed for {file_info.filename}: {e}")

    logger.info(f"Downloaded {len(downloaded)}/{len(files)} files")
    return downloaded


def download_ar5iv_dataset(
    output_dir: Path,
    *,
    quality_levels: List[str] = ["no_problem", "warning"],
    max_workers: int = 4,
) -> Path:
    """
    Download ar5iv HTML dataset bundles.

    Args:
        output_dir: Directory to save HTML files
        quality_levels: Which quality tiers to download
        max_workers: Number of parallel downloads

    Returns:
        Path to output directory

    Note:
        The ar5iv dataset may require license agreement.
        Total size: ~236GB compressed for no_problem + warning tiers.
    """
    # ar5iv dataset requires manual download - no verified automatic source.
    # Total size: ~236GB compressed for no_problem + warning tiers.
    raise NotImplementedError(
        "ar5iv dataset requires manual download.\n"
        "1. Visit: https://sigmathling.kwarc.info/resources/ar5iv-dataset-2024/\n"
        "2. Download the desired quality tiers (no_problem, warning, error)\n"
        f"3. Extract to: {output_dir}\n"
        "4. Re-run with --ar5iv-dir pointing to the extracted directory"
    )


def estimate_download_cost(
    manifest_path: Path,
    year_months: Optional[List[str]] = None,
) -> dict:
    """
    Estimate AWS costs for downloading arXiv data.

    Args:
        manifest_path: Path to S3 manifest XML
        year_months: Optional filter

    Returns:
        Dict with size and cost estimates
    """
    files = parse_s3_manifest(manifest_path)

    if year_months:
        year_months_set = set(year_months)
        files = [f for f in files if f.year_month in year_months_set]

    total_bytes = sum(f.size_bytes for f in files)
    total_gb = total_bytes / (1024**3)

    # AWS S3 data transfer pricing (as of 2024)
    # First 100TB: $0.09/GB
    cost_per_gb = 0.09

    return {
        "num_files": len(files),
        "total_size_gb": total_gb,
        "total_size_tb": total_gb / 1024,
        "estimated_cost_usd": total_gb * cost_per_gb,
        "year_months": sorted(set(f.year_month for f in files)),
    }
