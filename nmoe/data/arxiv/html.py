"""
HTML/MathML processing utilities for ar5iv papers.

Extracts text from ar5iv HTML5+MathML documents while preserving
mathematical notation as LaTeX.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Lazy import for lxml
_lxml = None


def _get_lxml():
    global _lxml
    if _lxml is None:
        try:
            from lxml import html, etree
            _lxml = (html, etree)
        except ImportError:
            raise ImportError(
                "lxml is required for HTML processing. "
                "Install with: pip install lxml>=5.0"
            )
    return _lxml


@dataclass
class HtmlMetadata:
    """Metadata extracted from ar5iv HTML."""
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    arxiv_id: Optional[str] = None
    categories: Optional[List[str]] = None


# MathML operator mappings to LaTeX
MATHML_OPERATORS = {
    # Greek letters
    '\u03b1': r'\alpha', '\u03b2': r'\beta', '\u03b3': r'\gamma',
    '\u03b4': r'\delta', '\u03b5': r'\epsilon', '\u03b6': r'\zeta',
    '\u03b7': r'\eta', '\u03b8': r'\theta', '\u03b9': r'\iota',
    '\u03ba': r'\kappa', '\u03bb': r'\lambda', '\u03bc': r'\mu',
    '\u03bd': r'\nu', '\u03be': r'\xi', '\u03c0': r'\pi',
    '\u03c1': r'\rho', '\u03c3': r'\sigma', '\u03c4': r'\tau',
    '\u03c5': r'\upsilon', '\u03c6': r'\phi', '\u03c7': r'\chi',
    '\u03c8': r'\psi', '\u03c9': r'\omega',
    '\u0393': r'\Gamma', '\u0394': r'\Delta', '\u0398': r'\Theta',
    '\u039b': r'\Lambda', '\u039e': r'\Xi', '\u03a0': r'\Pi',
    '\u03a3': r'\Sigma', '\u03a6': r'\Phi', '\u03a8': r'\Psi',
    '\u03a9': r'\Omega',
    # Operators
    '\u2211': r'\sum', '\u220f': r'\prod', '\u222b': r'\int',
    '\u222c': r'\iint', '\u222d': r'\iiint',
    '\u221e': r'\infty', '\u2202': r'\partial',
    '\u2207': r'\nabla', '\u221a': r'\sqrt',
    '\u00d7': r'\times', '\u00f7': r'\div',
    '\u00b1': r'\pm', '\u2213': r'\mp',
    '\u2264': r'\leq', '\u2265': r'\geq',
    '\u2260': r'\neq', '\u2248': r'\approx',
    '\u2261': r'\equiv', '\u221d': r'\propto',
    '\u2208': r'\in', '\u2209': r'\notin',
    '\u2282': r'\subset', '\u2283': r'\supset',
    '\u2286': r'\subseteq', '\u2287': r'\supseteq',
    '\u222a': r'\cup', '\u2229': r'\cap',
    '\u2205': r'\emptyset', '\u2200': r'\forall',
    '\u2203': r'\exists', '\u00ac': r'\neg',
    '\u2227': r'\land', '\u2228': r'\lor',
    '\u21d2': r'\Rightarrow', '\u21d0': r'\Leftarrow',
    '\u21d4': r'\Leftrightarrow',
    '\u2192': r'\rightarrow', '\u2190': r'\leftarrow',
    '\u2194': r'\leftrightarrow',
    '\u22c5': r'\cdot', '\u2026': r'\ldots',
    '\u22ef': r'\cdots', '\u22ee': r'\vdots',
    # Brackets
    '\u27e8': r'\langle', '\u27e9': r'\rangle',
    '\u230a': r'\lfloor', '\u230b': r'\rfloor',
    '\u2308': r'\lceil', '\u2309': r'\rceil',
}


def html_to_text(
    html_content: str,
    *,
    preserve_math: bool = True,
    include_abstract: bool = True,
    include_title: bool = True,
) -> str:
    """
    Extract text from ar5iv HTML while preserving math as LaTeX.

    Args:
        html_content: Raw HTML string
        preserve_math: Convert MathML to LaTeX notation
        include_abstract: Include abstract in output
        include_title: Include title in output

    Returns:
        Clean text with math preserved as LaTeX notation
    """
    html_parser, etree = _get_lxml()

    try:
        doc = html_parser.fromstring(html_content)
    except Exception as e:
        logger.warning(f"Failed to parse HTML: {e}")
        return ""

    parts = []

    # Extract title if requested
    if include_title:
        title_elem = doc.find('.//h1[@class="ltx_title ltx_title_document"]')
        if title_elem is not None:
            title_text = _extract_element_text(title_elem, preserve_math)
            if title_text:
                parts.append(title_text)
                parts.append("")  # Blank line after title

    # Extract abstract if requested
    if include_abstract:
        abstract_elem = doc.find('.//div[@class="ltx_abstract"]')
        if abstract_elem is not None:
            abstract_text = _extract_element_text(abstract_elem, preserve_math)
            if abstract_text:
                # Remove "Abstract" header if present
                abstract_text = re.sub(r'^Abstract\.?\s*', '', abstract_text)
                parts.append(abstract_text)
                parts.append("")  # Blank line after abstract

    # Extract main article content
    article = doc.find('.//article[@class="ltx_document"]')
    if article is not None:
        # Process sections
        for section in article.findall('.//section'):
            section_text = _extract_section(section, preserve_math)
            if section_text:
                parts.append(section_text)

    # If no structured content found, try to extract all text
    if not parts:
        body = doc.find('.//body')
        if body is not None:
            parts.append(_extract_element_text(body, preserve_math))

    text = '\n\n'.join(parts)
    return _normalize_whitespace(text)


def _extract_section(section_elem, preserve_math: bool) -> str:
    """Extract text from a section element."""
    html_parser, etree = _get_lxml()

    parts = []

    # Get section heading
    heading = section_elem.find('.//h2') or section_elem.find('.//h3') or section_elem.find('.//h4')
    if heading is not None:
        heading_text = _extract_element_text(heading, preserve_math)
        if heading_text:
            # Remove section numbers
            heading_text = re.sub(r'^\d+(\.\d+)*\.?\s*', '', heading_text)
            parts.append(heading_text)

    # Get paragraphs
    for para in section_elem.findall('.//p[@class="ltx_p"]'):
        para_text = _extract_element_text(para, preserve_math)
        if para_text:
            parts.append(para_text)

    # Get equations (display math)
    for eq in section_elem.findall('.//table[@class="ltx_equation"]'):
        eq_text = _extract_element_text(eq, preserve_math)
        if eq_text and preserve_math:
            parts.append(f"\\[{eq_text}\\]")

    return '\n\n'.join(parts)


def _extract_element_text(elem, preserve_math: bool) -> str:
    """Extract text from an element, handling MathML."""
    html_parser, etree = _get_lxml()

    parts = []

    def process_node(node):
        if isinstance(node, str):
            parts.append(node)
            return

        tag = etree.QName(node.tag).localname if isinstance(node.tag, str) else node.tag

        # Handle MathML
        if tag == 'math' or (isinstance(node.tag, str) and 'math' in node.tag):
            if preserve_math:
                latex = mathml_to_latex(node)
                # Determine if inline or display
                display = node.get('display', 'inline')
                if display == 'block':
                    parts.append(f"\\[{latex}\\]")
                else:
                    parts.append(f"${latex}$")
            else:
                parts.append(_mathml_to_text(node))
            return

        # Handle text content
        if node.text:
            parts.append(node.text)

        # Process children
        for child in node:
            process_node(child)
            if child.tail:
                parts.append(child.tail)

    process_node(elem)
    return ''.join(parts)


def mathml_to_latex(mathml_elem) -> str:
    """
    Convert MathML element to LaTeX string.

    Handles Presentation MathML (what ar5iv uses) with common elements:
    - mi (identifier), mn (number), mo (operator)
    - mfrac, msqrt, mroot, msup, msub, msubsup
    - mrow, mfenced, mtable, mtr, mtd
    """
    html_parser, etree = _get_lxml()

    def process(elem) -> str:
        if elem is None:
            return ''

        tag = etree.QName(elem.tag).localname if isinstance(elem.tag, str) else str(elem.tag)
        # Strip namespace if present
        if '}' in tag:
            tag = tag.split('}')[1]

        # Text content
        text = (elem.text or '').strip()

        # Apply operator mappings
        if text in MATHML_OPERATORS:
            text = MATHML_OPERATORS[text]

        if tag == 'math':
            return ''.join(process(child) for child in elem)

        elif tag == 'mrow':
            return ''.join(process(child) for child in elem)

        elif tag == 'mi':  # Identifier
            return text

        elif tag == 'mn':  # Number
            return text

        elif tag == 'mo':  # Operator
            return text

        elif tag == 'mtext':  # Text
            return f'\\text{{{text}}}'

        elif tag == 'mspace':
            return ' '

        elif tag == 'mfrac':  # Fraction
            children = list(elem)
            if len(children) >= 2:
                num = process(children[0])
                den = process(children[1])
                return f'\\frac{{{num}}}{{{den}}}'
            return ''

        elif tag == 'msqrt':  # Square root
            content = ''.join(process(child) for child in elem)
            return f'\\sqrt{{{content}}}'

        elif tag == 'mroot':  # nth root
            children = list(elem)
            if len(children) >= 2:
                base = process(children[0])
                index = process(children[1])
                return f'\\sqrt[{index}]{{{base}}}'
            return ''

        elif tag == 'msup':  # Superscript
            children = list(elem)
            if len(children) >= 2:
                base = process(children[0])
                sup = process(children[1])
                return f'{{{base}}}^{{{sup}}}'
            return ''

        elif tag == 'msub':  # Subscript
            children = list(elem)
            if len(children) >= 2:
                base = process(children[0])
                sub = process(children[1])
                return f'{{{base}}}_{{{sub}}}'
            return ''

        elif tag == 'msubsup':  # Both sub and superscript
            children = list(elem)
            if len(children) >= 3:
                base = process(children[0])
                sub = process(children[1])
                sup = process(children[2])
                return f'{{{base}}}_{{{sub}}}^{{{sup}}}'
            return ''

        elif tag == 'munder':  # Under
            children = list(elem)
            if len(children) >= 2:
                base = process(children[0])
                under = process(children[1])
                return f'\\underset{{{under}}}{{{base}}}'
            return ''

        elif tag == 'mover':  # Over
            children = list(elem)
            if len(children) >= 2:
                base = process(children[0])
                over = process(children[1])
                return f'\\overset{{{over}}}{{{base}}}'
            return ''

        elif tag == 'munderover':  # Under and over
            children = list(elem)
            if len(children) >= 3:
                base = process(children[0])
                under = process(children[1])
                over = process(children[2])
                return f'\\underset{{{under}}}{{\\overset{{{over}}}{{{base}}}}}'
            return ''

        elif tag == 'mfenced':  # Fenced (parentheses, brackets, etc.)
            open_delim = elem.get('open', '(')
            close_delim = elem.get('close', ')')
            content = ''.join(process(child) for child in elem)
            return f'{open_delim}{content}{close_delim}'

        elif tag == 'mtable':  # Matrix/table
            rows = []
            for row in elem:
                row_tag = etree.QName(row.tag).localname if isinstance(row.tag, str) else str(row.tag)
                if '}' in row_tag:
                    row_tag = row_tag.split('}')[1]
                if row_tag == 'mtr':
                    cells = []
                    for cell in row:
                        cells.append(process(cell))
                    rows.append(' & '.join(cells))
            return '\\begin{matrix}' + ' \\\\ '.join(rows) + '\\end{matrix}'

        elif tag == 'mtr':  # Table row
            cells = [process(child) for child in elem]
            return ' & '.join(cells)

        elif tag == 'mtd':  # Table cell
            return ''.join(process(child) for child in elem)

        elif tag == 'semantics':
            # Prefer presentation MathML over content MathML
            children = list(elem)
            if children:
                return process(children[0])
            return ''

        elif tag == 'annotation':
            # Skip annotations
            return ''

        else:
            # Unknown tag: process children
            return ''.join(process(child) for child in elem)

    result = process(mathml_elem)
    return result.strip()


def _mathml_to_text(mathml_elem) -> str:
    """Convert MathML to plain text (no LaTeX)."""
    html_parser, etree = _get_lxml()

    def extract_text(elem):
        parts = []
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            parts.append(extract_text(child))
            if child.tail:
                parts.append(child.tail)
        return ''.join(parts)

    return extract_text(mathml_elem)


def extract_html_metadata(html_content: str) -> HtmlMetadata:
    """
    Extract metadata from ar5iv HTML.

    Args:
        html_content: Raw HTML string

    Returns:
        HtmlMetadata with extracted fields
    """
    html_parser, etree = _get_lxml()
    metadata = HtmlMetadata()

    try:
        doc = html_parser.fromstring(html_content)
    except Exception as e:
        logger.warning(f"Failed to parse HTML for metadata: {e}")
        return metadata

    # Extract title
    title_elem = doc.find('.//h1[@class="ltx_title ltx_title_document"]')
    if title_elem is not None:
        metadata.title = _get_text_content(title_elem)

    # Extract authors
    author_elems = doc.findall('.//span[@class="ltx_personname"]')
    if author_elems:
        metadata.authors = [_get_text_content(a) for a in author_elems]

    # Extract abstract
    abstract_elem = doc.find('.//div[@class="ltx_abstract"]')
    if abstract_elem is not None:
        abstract_text = _get_text_content(abstract_elem)
        # Remove "Abstract" header
        abstract_text = re.sub(r'^Abstract\.?\s*', '', abstract_text)
        metadata.abstract = abstract_text

    # Extract arxiv ID from URL or meta
    canonical = doc.find('.//link[@rel="canonical"]')
    if canonical is not None:
        href = canonical.get('href', '')
        match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', href)
        if match:
            metadata.arxiv_id = match.group(1)

    # Try meta tags
    if not metadata.arxiv_id:
        for meta in doc.findall('.//meta'):
            if meta.get('name') == 'citation_arxiv_id':
                metadata.arxiv_id = meta.get('content')
                break

    # Extract categories from keywords or classification
    keywords_elem = doc.find('.//div[@class="ltx_keywords"]')
    if keywords_elem is not None:
        keywords_text = _get_text_content(keywords_elem)
        # Parse categories (e.g., "cs.LG, stat.ML")
        categories = re.findall(r'[a-z-]+\.[A-Z]{2}', keywords_text, re.IGNORECASE)
        if categories:
            metadata.categories = categories

    return metadata


def _get_text_content(elem) -> str:
    """Get all text content from an element."""
    html_parser, etree = _get_lxml()

    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_get_text_content(child))
        if child.tail:
            parts.append(child.tail)
    return ' '.join(''.join(parts).split())


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace in converted text."""
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Strip lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()
