"""
arXiv metadata client for enriching papers with API data.

Uses the arXiv API to fetch:
- Title, authors, abstract
- Categories and primary category
- Submission and update dates
- DOI and license information
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from urllib.request import urlopen
from urllib.parse import quote

logger = logging.getLogger(__name__)

# arXiv API configuration
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_API_DELAY = 3.0  # Seconds between requests (respect rate limits)


@dataclass
class ArxivMetadata:
    """Metadata for an arXiv paper."""
    arxiv_id: str
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    abstract: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    primary_category: Optional[str] = None
    submitted: Optional[datetime] = None
    updated: Optional[datetime] = None
    doi: Optional[str] = None
    license: Optional[str] = None
    journal_ref: Optional[str] = None
    comment: Optional[str] = None


class ArxivMetadataClient:
    """
    Client for fetching metadata from the arXiv API.

    Respects rate limits (3 second delay between requests).
    Supports batch queries for efficiency.
    """

    def __init__(self, delay: float = ARXIV_API_DELAY):
        self.delay = delay
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self):
        """Wait if needed to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()

    def get_metadata(self, arxiv_id: str) -> Optional[ArxivMetadata]:
        """
        Fetch metadata for a single paper.

        Args:
            arxiv_id: arXiv identifier (e.g., "2301.12345" or "2301.12345v2")

        Returns:
            ArxivMetadata or None if not found
        """
        results = self.bulk_metadata([arxiv_id])
        return results.get(arxiv_id)

    def bulk_metadata(
        self,
        arxiv_ids: List[str],
        batch_size: int = 100,
    ) -> Dict[str, ArxivMetadata]:
        """
        Fetch metadata for multiple papers efficiently.

        Args:
            arxiv_ids: List of arXiv identifiers
            batch_size: Number of IDs per API request (max 100)

        Returns:
            Dict mapping arxiv_id to ArxivMetadata
        """
        results = {}

        # Process in batches
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i:i + batch_size]

            try:
                batch_results = self._fetch_batch(batch)
                results.update(batch_results)
            except Exception as e:
                logger.error(f"Failed to fetch batch {i}-{i+len(batch)}: {e}")

        return results

    def _fetch_batch(self, arxiv_ids: List[str]) -> Dict[str, ArxivMetadata]:
        """Fetch metadata for a batch of IDs."""
        self._wait_for_rate_limit()

        # Build query
        id_list = ",".join(arxiv_ids)
        url = f"{ARXIV_API_BASE}?id_list={quote(id_list)}&max_results={len(arxiv_ids)}"

        logger.debug(f"Fetching metadata for {len(arxiv_ids)} papers")

        try:
            with urlopen(url, timeout=30) as response:
                xml_data = response.read().decode('utf-8')
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return {}

        return self._parse_response(xml_data)

    def _parse_response(self, xml_data: str) -> Dict[str, ArxivMetadata]:
        """Parse arXiv API XML response."""
        # Define namespaces
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom',
        }

        results = {}

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            return {}

        for entry in root.findall('atom:entry', ns):
            try:
                metadata = self._parse_entry(entry, ns)
                if metadata:
                    results[metadata.arxiv_id] = metadata
            except Exception as e:
                logger.debug(f"Failed to parse entry: {e}")

        return results

    def _parse_entry(self, entry, ns: dict) -> Optional[ArxivMetadata]:
        """Parse a single entry element."""
        # Extract ID from <id> element
        id_elem = entry.find('atom:id', ns)
        if id_elem is None or id_elem.text is None:
            return None

        # Parse arxiv_id from URL (e.g., "http://arxiv.org/abs/2301.12345v1")
        id_match = re.search(r'arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)', id_elem.text)
        if not id_match:
            # Try older format (e.g., "cs/0001001")
            id_match = re.search(r'arxiv\.org/abs/([a-z-]+/\d+(?:v\d+)?)', id_elem.text)
            if not id_match:
                return None

        arxiv_id = id_match.group(1)
        # Remove version for consistency
        arxiv_id_base = re.sub(r'v\d+$', '', arxiv_id)

        metadata = ArxivMetadata(arxiv_id=arxiv_id_base)

        # Title
        title_elem = entry.find('atom:title', ns)
        if title_elem is not None and title_elem.text:
            metadata.title = ' '.join(title_elem.text.split())

        # Authors
        for author in entry.findall('atom:author', ns):
            name_elem = author.find('atom:name', ns)
            if name_elem is not None and name_elem.text:
                metadata.authors.append(name_elem.text.strip())

        # Abstract/Summary
        summary_elem = entry.find('atom:summary', ns)
        if summary_elem is not None and summary_elem.text:
            metadata.abstract = ' '.join(summary_elem.text.split())

        # Categories
        for category in entry.findall('atom:category', ns):
            term = category.get('term')
            if term:
                metadata.categories.append(term)

        # Primary category
        primary = entry.find('arxiv:primary_category', ns)
        if primary is not None:
            metadata.primary_category = primary.get('term')

        # Dates
        published = entry.find('atom:published', ns)
        if published is not None and published.text:
            try:
                metadata.submitted = datetime.fromisoformat(published.text.replace('Z', '+00:00'))
            except ValueError:
                pass

        updated = entry.find('atom:updated', ns)
        if updated is not None and updated.text:
            try:
                metadata.updated = datetime.fromisoformat(updated.text.replace('Z', '+00:00'))
            except ValueError:
                pass

        # DOI
        doi_elem = entry.find('arxiv:doi', ns)
        if doi_elem is not None and doi_elem.text:
            metadata.doi = doi_elem.text.strip()

        # Journal reference
        journal_ref = entry.find('arxiv:journal_ref', ns)
        if journal_ref is not None and journal_ref.text:
            metadata.journal_ref = journal_ref.text.strip()

        # Comment
        comment = entry.find('arxiv:comment', ns)
        if comment is not None and comment.text:
            metadata.comment = comment.text.strip()

        return metadata

    def list_ids_by_category(
        self,
        category: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 1000,
    ) -> List[str]:
        """
        List paper IDs by category and optional date range.

        Args:
            category: arXiv category (e.g., "cs.LG", "math.AG")
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            max_results: Maximum number of results

        Returns:
            List of arXiv IDs
        """
        self._wait_for_rate_limit()

        # Build query
        query = f"cat:{category}"
        if start_date or end_date:
            date_query = f"submittedDate:[{start_date or '*'} TO {end_date or '*'}]"
            query = f"{query} AND {date_query}"

        url = (
            f"{ARXIV_API_BASE}?"
            f"search_query={quote(query)}&"
            f"max_results={max_results}&"
            f"sortBy=submittedDate&"
            f"sortOrder=descending"
        )

        logger.debug(f"Listing IDs for category {category}")

        try:
            with urlopen(url, timeout=60) as response:
                xml_data = response.read().decode('utf-8')
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return []

        # Parse just the IDs
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        ids = []

        try:
            root = ET.fromstring(xml_data)
            for entry in root.findall('atom:entry', ns):
                id_elem = entry.find('atom:id', ns)
                if id_elem is not None and id_elem.text:
                    match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', id_elem.text)
                    if match:
                        ids.append(match.group(1))
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")

        return ids
