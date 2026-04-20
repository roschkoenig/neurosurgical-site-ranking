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

    def _get(self, url: str, params: dict) -> dict:
        """Perform a GET request with retries and rate-limit delay."""
        if self.api_key:
            params["api_key"] = self.api_key
        params["retmode"] = "json"
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                time.sleep(self.delay)
                return resp.json()
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
                {"db": "pubmed", "id": ",".join(batch), "rettype": "abstract"},
            )
            # efetch JSON returns PubmedArticleSet structure
            batch_records = self._parse_efetch(data, batch)
            cache_set("pubmed_fetch", cache_key, batch_records)
            records.extend(batch_records)

        return records

    def _parse_efetch(self, data: Any, pmids: list[str]) -> list[dict]:
        """
        Parse the PubMed efetch JSON response into a flat list of dicts.

        The efetch endpoint with retmode=json returns the raw XML converted
        to JSON by NCBI. The structure varies; we extract the fields we need
        defensively.
        """
        articles = []

        # Top-level key differs between single and multi-article responses
        pubmed_set = data.get("PubmedArticleSet", {})
        article_list = pubmed_set.get("PubmedArticle", [])
        if isinstance(article_list, dict):
            article_list = [article_list]

        for art in article_list:
            medline = art.get("MedlineCitation", {})
            pmid = str(medline.get("PMID", {}).get("#text", medline.get("PMID", "")))
            article = medline.get("Article", {})
            title = self._extract_text(article.get("ArticleTitle", ""))
            abstract_obj = article.get("Abstract", {})
            abstract_text = self._extract_text(
                abstract_obj.get("AbstractText", "")
            )
            journal = article.get("Journal", {}).get("Title", "")
            pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = pub_date.get("Year", pub_date.get("MedlineDate", "")[:4] if isinstance(pub_date.get("MedlineDate", ""), str) else "")

            # DOI
            id_list = article.get("ELocationID", [])
            if isinstance(id_list, dict):
                id_list = [id_list]
            doi = next(
                (i.get("#text", "") for i in id_list if i.get("@EIdType") == "doi"),
                "",
            )

            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract_text,
                    "journal": journal,
                    "year": str(year),
                    "doi": doi,
                }
            )

        # Fall back: if parsing yields nothing, return stub records for each PMID
        if not articles:
            articles = [{"pmid": p, "title": "", "abstract": "", "journal": "", "year": "", "doi": ""} for p in pmids]

        return articles

    @staticmethod
    def _extract_text(obj: Any) -> str:
        """Recursively extract plain text from a PubMed JSON text node."""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return obj.get("#text", "")
        if isinstance(obj, list):
            parts = []
            for item in obj:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("#text", ""))
            return " ".join(parts)
        return ""

    def save_results(self, records: list[dict], path: str | None = None) -> Path:
        """Save *records* as JSON to *path* (default: outputs/pubmed_raw.json)."""
        dest = Path(path) if path else output_dir() / "pubmed_raw.json"
        save_json(records, dest)
        logger.info("Saved %d PubMed records to %s", len(records), dest)
        return dest
