"""
LaTeX processing utilities for arXiv papers.

Converts LaTeX source to clean text while preserving mathematical notation.
Uses pylatexenc for robust parsing with math preservation.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List

logger = logging.getLogger(__name__)

# Lazy import for pylatexenc
_pylatexenc = None


def _get_pylatexenc():
    global _pylatexenc
    if _pylatexenc is None:
        try:
            from pylatexenc import latex2text, latexwalker
            _pylatexenc = (latex2text, latexwalker)
        except ImportError:
            raise ImportError(
                "pylatexenc is required for LaTeX processing. "
                "Install with: pip install pylatexenc>=2.10"
            )
    return _pylatexenc


@dataclass
class TexMetadata:
    """Metadata extracted from LaTeX preamble."""
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    date: Optional[str] = None
    keywords: Optional[List[str]] = None


# Patterns for metadata extraction
TITLE_PATTERN = re.compile(r'\\title\s*(?:\[[^\]]*\])?\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', re.DOTALL)
AUTHOR_PATTERN = re.compile(r'\\author\s*(?:\[[^\]]*\])?\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', re.DOTALL)
ABSTRACT_PATTERN = re.compile(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', re.DOTALL)
DATE_PATTERN = re.compile(r'\\date\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', re.DOTALL)

# Patterns for cleaning
COMMENT_PATTERN = re.compile(r'(?<!\\)%.*$', re.MULTILINE)
CITE_PATTERN = re.compile(r'\\cite\w*\s*(?:\[[^\]]*\])?\s*\{[^}]*\}')
REF_PATTERN = re.compile(r'\\(?:ref|eqref|pageref)\s*\{[^}]*\}')
LABEL_PATTERN = re.compile(r'\\label\s*\{[^}]*\}')
FOOTNOTE_PATTERN = re.compile(r'\\footnote\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}')

# Input/include patterns
INPUT_PATTERN = re.compile(r'\\(?:input|include)\s*\{([^}]+)\}')

# Environment patterns for removal
REMOVE_ENVS = ['figure', 'table', 'tikzpicture', 'pgfpicture', 'thebibliography']


def latex_to_text(
    latex_content: str,
    *,
    expand_includes: Optional[Callable[[str], Optional[str]]] = None,
    **_,  # Accept but ignore legacy args for API compat
) -> str:
    r"""
    Process LaTeX for training - returns raw LaTeX with includes expanded.

    Model learns complete paper structure including preamble, packages,
    and all LaTeX formatting. Only \input{}/\include{} are expanded
    to make multi-file papers self-contained.

    Args:
        latex_content: Raw LaTeX source
        expand_includes: Callback to resolve \\input{} and \\include{}

    Returns:
        Raw LaTeX with includes expanded
    """
    if expand_includes:
        return _expand_includes(latex_content, expand_includes)
    return latex_content


def _expand_includes(text: str, resolver: Callable[[str], Optional[str]], depth: int = 0) -> str:
    """Recursively expand \\input{} and \\include{} commands."""
    if depth > 10:  # Prevent infinite recursion
        return text

    def replace_include(match):
        filename = match.group(1).strip()
        # Add .tex extension if not present
        if not filename.endswith('.tex'):
            filename = filename + '.tex'

        content = resolver(filename)
        if content is None:
            logger.debug(f"Could not resolve include: {filename}")
            return f"% [Could not include: {filename}]"

        # Recursively expand
        return _expand_includes(content, resolver, depth + 1)

    return INPUT_PATTERN.sub(replace_include, text)


def _remove_environment(text: str, env_name: str) -> str:
    """Remove a LaTeX environment and its contents."""
    pattern = re.compile(
        rf'\\begin\{{{env_name}\}}.*?\\end\{{{env_name}\}}',
        re.DOTALL
    )
    return pattern.sub('', text)


def _convert_with_math_preserved(text: str) -> str:
    """
    Convert LaTeX to text, preserving math environments verbatim.

    Strategy: Extract math, replace with placeholders, convert, restore.
    """
    latex2text, latexwalker = _get_pylatexenc()

    # Math delimiters to preserve
    math_patterns = [
        (r'\$\$(.+?)\$\$', 'displaymath'),      # $$...$$
        (r'\\\[(.+?)\\\]', 'displaymath'),       # \[...\]
        (r'\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}', 'equation'),
        (r'\\begin\{align\*?\}(.+?)\\end\{align\*?\}', 'align'),
        (r'\\begin\{eqnarray\*?\}(.+?)\\end\{eqnarray\*?\}', 'eqnarray'),
        (r'\\begin\{gather\*?\}(.+?)\\end\{gather\*?\}', 'gather'),
        (r'\\begin\{multline\*?\}(.+?)\\end\{multline\*?\}', 'multline'),
        (r'\$(.+?)\$', 'inlinemath'),            # $...$
        (r'\\\((.+?)\\\)', 'inlinemath'),        # \(...\)
    ]

    # Extract and placeholder math
    math_blocks = []
    placeholder_text = text

    for pattern, math_type in math_patterns:
        regex = re.compile(pattern, re.DOTALL)

        def replace_math(match):
            idx = len(math_blocks)
            full_match = match.group(0)
            math_blocks.append(full_match)
            return f"<<MATH_{idx}>>"

        placeholder_text = regex.sub(replace_math, placeholder_text)

    # Convert non-math LaTeX to text
    try:
        l2t = latex2text.LatexNodes2Text(
            math_mode='verbatim',  # Keep any remaining math
            keep_braced_groups=False,
            strict_latex_spaces=False,
        )
        converted = l2t.latex_to_text(placeholder_text)
    except Exception as e:
        logger.warning(f"pylatexenc conversion failed: {e}")
        converted = placeholder_text

    # Restore math blocks
    for idx, math_block in enumerate(math_blocks):
        converted = converted.replace(f"<<MATH_{idx}>>", math_block)

    return converted


def _convert_without_math(text: str) -> str:
    """Convert LaTeX to plain text, converting math to text representation."""
    latex2text, _ = _get_pylatexenc()

    try:
        l2t = latex2text.LatexNodes2Text(
            math_mode='text',  # Convert math to text
            keep_braced_groups=False,
            strict_latex_spaces=False,
        )
        return l2t.latex_to_text(text)
    except Exception as e:
        logger.warning(f"pylatexenc conversion failed: {e}")
        return text


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace in converted text."""
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Strip lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    # Strip overall
    return text.strip()


def find_main_tex(files: Dict[str, bytes]) -> Optional[str]:
    """
    Find the main .tex file in a multi-file document.

    Args:
        files: Dict mapping filename to content bytes

    Returns:
        Filename of main tex file, or None if not found
    """
    tex_files = [f for f in files.keys() if f.endswith('.tex')]

    if not tex_files:
        return None

    if len(tex_files) == 1:
        return tex_files[0]

    # Heuristic 1: Look for common main file names
    common_names = ['main.tex', 'paper.tex', 'article.tex', 'manuscript.tex', 'document.tex']
    for name in common_names:
        if name in tex_files:
            return name

    # Heuristic 2: Look for \documentclass
    for filename in tex_files:
        try:
            content = files[filename].decode('utf-8', errors='ignore')
            if '\\documentclass' in content:
                return filename
        except Exception:
            continue

    # Heuristic 3: Look for \begin{document}
    for filename in tex_files:
        try:
            content = files[filename].decode('utf-8', errors='ignore')
            if '\\begin{document}' in content:
                return filename
        except Exception:
            continue

    # Fallback: return first tex file alphabetically
    return sorted(tex_files)[0]


def extract_tex_metadata(latex_content: str) -> TexMetadata:
    """
    Extract metadata from LaTeX preamble.

    Args:
        latex_content: Raw LaTeX source

    Returns:
        TexMetadata with extracted fields
    """
    metadata = TexMetadata()

    # Extract title
    title_match = TITLE_PATTERN.search(latex_content)
    if title_match:
        metadata.title = _clean_tex_field(title_match.group(1))

    # Extract authors
    author_match = AUTHOR_PATTERN.search(latex_content)
    if author_match:
        raw_authors = author_match.group(1)
        # Split by common author separators
        authors = re.split(r'\\and\s*|,\s*(?=[A-Z])', raw_authors)
        metadata.authors = [_clean_tex_field(a) for a in authors if a.strip()]

    # Extract abstract
    abstract_match = ABSTRACT_PATTERN.search(latex_content)
    if abstract_match:
        metadata.abstract = _clean_tex_field(abstract_match.group(1))

    # Extract date
    date_match = DATE_PATTERN.search(latex_content)
    if date_match:
        metadata.date = _clean_tex_field(date_match.group(1))

    return metadata


def _clean_tex_field(text: str) -> str:
    """Clean a single metadata field."""
    # Remove comments
    text = COMMENT_PATTERN.sub('', text)
    # Remove common formatting commands
    text = re.sub(r'\\(?:textbf|textit|emph|textrm|textsc)\s*\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:bf|it|em|rm|sc)\b\s*', '', text)
    # Remove footnotes
    text = FOOTNOTE_PATTERN.sub('', text)
    # Remove affiliations and thanks
    text = re.sub(r'\\(?:thanks|footnote|inst)\s*\{[^}]*\}', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()


def clean_latex_commands(text: str) -> str:
    """
    Remove/replace common LaTeX commands for cleaner output.

    This is a lighter-weight alternative to full pylatexenc conversion.
    """
    # Handle common text formatting
    text = re.sub(r'\\(?:textbf|textit|emph|textrm|textsc|texttt)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1', text)

    # Handle emphasis
    text = re.sub(r'\{\\(?:bf|it|em|rm|sc|tt)\s+([^}]*)\}', r'\1', text)

    # Handle sections
    text = re.sub(r'\\(?:section|subsection|subsubsection|paragraph)\*?\s*\{([^}]*)\}', r'\n\n\1\n\n', text)

    # Handle itemize/enumerate
    text = re.sub(r'\\item\s*', '\n- ', text)
    text = re.sub(r'\\begin\{(?:itemize|enumerate)\}', '', text)
    text = re.sub(r'\\end\{(?:itemize|enumerate)\}', '', text)

    # Remove common commands with no content replacement
    text = re.sub(r'\\(?:vspace|hspace|smallskip|medskip|bigskip|noindent|newpage|clearpage)\*?\s*(?:\{[^}]*\})?', '', text)

    return text
