"""
pubmed_search.py – Retrieve glioblastoma surgical papers from PubMed.

Uses the NCBI E-utilities API (no API key required for low-volume use;
set NCBI_API_KEY env var for higher throughput).

Usage example
-------------
>>> from src.pubmed_search import PubMedSearcher
>>> searcher = PubMedSearcher()
>>> pmids = searcher.search(max_results=200)
>>> records = searcher.fetch_details(pmids)
>>> searcher.save_results(records, "outputs/pubmed_raw.json")
"""

import os
import time
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests

from src.utils import cache_get, cache_set, output_dir, save_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default surgical GBM search queries
# ---------------------------------------------------------------------------

DEFAULT_QUERIES: list[str] = [
    (
        '(glioblastoma OR "high-grade glioma" OR "glioblastoma multiforme")'
        ' AND (surgery OR resection OR craniotomy OR "extent of resection"'
        ' OR "awake craniotomy" OR "awake mapping" OR "fluorescence-guided"'
        ' OR "5-ALA" OR "intraoperative MRI" OR iMRI OR "recurrent glioblastoma")'
        ' AND humans[MeSH] AND English[lang]'
    )
]

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedSearcher:
    """Thin wrapper around NCBI E-utilities for PubMed searches."""

    def __init__(
        self,
        queries: list[str] | None = None,
        api_key: str | None = None,
        delay: float = 0.35,
    ) -> None:
        """
        Parameters
        ----------
        queries:  list of PubMed query strings; defaults to DEFAULT_QUERIES
        api_key:  NCBI API key (optional; raises rate-limit if omitted)
        delay:    seconds to sleep between API calls (keep ≥ 0.34 without key)
        """
        self.queries = queries or DEFAULT_QUERIES
        self.api_key = api_key or os.environ.get("NCBI_API_KEY", "")
        self.delay = delay

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict, response_type: str = "json") -> Any:
        """Perform a GET request with retries and rate-limit delay.

        Parameters
        ----------
        url:           E-utilities endpoint URL
        params:        query parameters (modified in-place)
        response_type: ``"json"`` (default) returns a parsed dict;
                       ``"text"`` returns the raw response text.
        """
        if self.api_key:
            params["api_key"] = self.api_key
        if response_type == "json":
            params["retmode"] = "json"
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                time.sleep(self.delay)
                if response_type == "json":
                    return resp.json()
                return resp.text
            except Exception as exc:
                logger.warning("PubMed request failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"PubMed request failed after 3 attempts: {url}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, max_results: int = 500) -> list[str]:
        """
        Run all configured queries and return a deduplicated list of PMIDs.

        Parameters
        ----------
        max_results:  maximum total PMIDs to collect across all queries
        """
        pmids: set[str] = set()
        per_query = max_results // len(self.queries) + 1

        for query in self.queries:
            cache_key = f"esearch::{query}::{per_query}"
            cached = cache_get("pubmed_search", cache_key)
            if cached is not None:
                logger.info("PubMed search cache hit: %d PMIDs", len(cached))
                pmids.update(cached)
                continue

            logger.info("PubMed search: %s", query[:80])
            data = self._get(
                f"{EUTILS_BASE}/esearch.fcgi",
                {"db": "pubmed", "term": query, "retmax": per_query},
            )
            ids = data.get("esearchresult", {}).get("idlist", [])
            cache_set("pubmed_search", cache_key, ids)
            pmids.update(ids)

        result = list(pmids)[:max_results]
        logger.info("Total unique PMIDs collected: %d", len(result))
        return result

    def fetch_details(self, pmids: list[str], batch_size: int = 200) -> list[dict]:
        """
        Fetch PubMed XML-derived JSON records for *pmids* in batches.

        Returns a list of dicts, one per article.
        """
        records: list[dict] = []

        for start in range(0, len(pmids), batch_size):
            batch = pmids[start : start + batch_size]
            cache_key = "efetch::" + ",".join(sorted(batch))
            cached = cache_get("pubmed_fetch", cache_key)
            if cached is not None:
                records.extend(cached)
                continue

            logger.info(
                "Fetching PubMed records %d–%d / %d",
                start + 1,
                start + len(batch),
                len(pmids),
            )
            data = self._get(
                f"{EUTILS_BASE}/efetch.fcgi",
                {"db": "pubmed", "id": ",".join(batch), "rettype": "abstract", "retmode": "xml"},
                response_type="text",
            )
            # efetch returns PubMed XML; parse it into records
            batch_records = self._parse_efetch(data, batch)
            cache_set("pubmed_fetch", cache_key, batch_records)
            records.extend(batch_records)

        return records

    def _parse_efetch(self, xml_text: str, pmids: list[str]) -> list[dict]:
        """
        Parse the PubMed efetch XML response into a flat list of dicts.

        Malformed records are skipped with a warning rather than raising.
        """
        articles = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("Failed to parse efetch XML response: %s", exc)
            return [{"pmid": p, "title": "", "abstract": "", "journal": "", "year": "", "doi": ""} for p in pmids]

        for art in root.findall("PubmedArticle"):
            try:
                medline = art.find("MedlineCitation")
                if medline is None:
                    raise ValueError("Missing MedlineCitation")

                pmid_el = medline.find("PMID")
                pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

                article_el = medline.find("Article")
                if article_el is None:
                    raise ValueError(f"Missing Article for PMID {pmid!r}")

                title = self._extract_text(article_el.find("ArticleTitle"))

                abstract_el = article_el.find("Abstract")
                abstract_text = ""
                if abstract_el is not None:
                    parts = []
                    for at in abstract_el.findall("AbstractText"):
                        parts.append(self._extract_text(at))
                    abstract_text = " ".join(p for p in parts if p)

                journal_el = article_el.find("Journal")
                journal = ""
                year = ""
                if journal_el is not None:
                    title_el = journal_el.find("Title")
                    journal = title_el.text.strip() if title_el is not None and title_el.text else ""
                    issue = journal_el.find("JournalIssue")
                    if issue is not None:
                        pub_date = issue.find("PubDate")
                        if pub_date is not None:
                            year_el = pub_date.find("Year")
                            if year_el is not None and year_el.text:
                                year = year_el.text.strip()
                            else:
                                medline_date = pub_date.find("MedlineDate")
                                if medline_date is not None and medline_date.text:
                                    year = medline_date.text.strip()[:4]

                doi = ""
                for loc_id in article_el.findall("ELocationID"):
                    if loc_id.get("EIdType") == "doi" and loc_id.text:
                        doi = loc_id.text.strip()
                        break

                articles.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract_text,
                        "journal": journal,
                        "year": year,
                        "doi": doi,
                    }
                )
            except Exception as exc:
                logger.warning("Skipping malformed PubMed record: %s", exc)

        # Fall back: if parsing yields nothing, return stub records for each PMID
        if not articles:
            articles = [{"pmid": p, "title": "", "abstract": "", "journal": "", "year": "", "doi": ""} for p in pmids]

        return articles

    @staticmethod
    def _extract_text(el: Any) -> str:
        """Return all text content of an XML element (including tail of children)."""
        if el is None:
            return ""
        return (ET.tostring(el, encoding="unicode", method="text") or "").strip()

    def save_results(self, records: list[dict], path: str | None = None) -> Path:
        """Save *records* as JSON to *path* (default: outputs/pubmed_raw.json)."""
        dest = Path(path) if path else output_dir() / "pubmed_raw.json"
        save_json(records, dest)
        logger.info("Saved %d PubMed records to %s", len(records), dest)
        return dest
