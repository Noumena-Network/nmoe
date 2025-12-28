"""
arXiv processing pipeline for LaTeX source and ar5iv HTML.

Supports:
- S3 bulk access (LaTeX .tar.gz source files)
- ar5iv HTML5+MathML experimental format
- Multi-format emission (both LaTeX and HTML as equivalent documents)

The pipeline emits both formats for papers where available, allowing
the model to learn that LaTeX and HTML are equivalent representations.
"""
from __future__ import annotations

from .latex import (
    latex_to_text,
    find_main_tex,
    extract_tex_metadata,
    clean_latex_commands,
)
from .html import (
    html_to_text,
    mathml_to_latex,
    extract_html_metadata,
)
from .sources import (
    ArxivS3Source,
    Ar5ivHTMLSource,
    ArxivMultiFormatSource,
    create_arxiv_source,
)
from .download import (
    download_s3_tars,
    download_ar5iv_dataset,
    download_s3_manifest,
    parse_s3_manifest,
    estimate_download_cost,
)
from .metadata import (
    ArxivMetadata,
    ArxivMetadataClient,
)

__all__ = [
    # LaTeX processing
    "latex_to_text",
    "find_main_tex",
    "extract_tex_metadata",
    "clean_latex_commands",
    # HTML processing
    "html_to_text",
    "mathml_to_latex",
    "extract_html_metadata",
    # Data sources
    "ArxivS3Source",
    "Ar5ivHTMLSource",
    "ArxivMultiFormatSource",
    "create_arxiv_source",
    # Download utilities
    "download_s3_tars",
    "download_ar5iv_dataset",
    "download_s3_manifest",
    "parse_s3_manifest",
    "estimate_download_cost",
    # Metadata
    "ArxivMetadata",
    "ArxivMetadataClient",
]
