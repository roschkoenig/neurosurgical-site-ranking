"""
openalex_client.py – Retrieve and cache OpenAlex metadata for works and authors.

OpenAlex is free and does not require an API key, but requests should include
a polite-pool email via the ``mailto`` parameter.  Set OPENALEX_EMAIL in the
environment to enable this.

Usage example
-------------
>>> from src.openalex_client import OpenAlexClient
>>> client = OpenAlexClient()
>>> work = client.get_work_by_pmid("12345678")
>>> author = client.get_author(work["authorships"][0]["author"]["id"])
"""

import logging
import os
import time
from typing import Any

import requests

from src.utils import cache_get, cache_set

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"


class OpenAlexClient:
    """Thin wrapper around the OpenAlex REST API with local JSON caching."""

    def __init__(
        self,
        email: str | None = None,
        delay: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        email:  polite-pool email (recommended by OpenAlex)
        delay:  seconds to sleep between requests
        """
        self.email = email or os.environ.get("OPENALEX_EMAIL", "")
        self.delay = delay

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _params(self, extra: dict | None = None) -> dict:
        p: dict[str, Any] = {}
        if self.email:
            p["mailto"] = self.email
        if extra:
            p.update(extra)
        return p

    def _get(self, url: str, params: dict | None = None) -> dict:
        """HTTP GET with retry logic."""
        full_params = self._params(params)
        for attempt in range(3):
            try:
                resp = requests.get(url, params=full_params, timeout=30)
                resp.raise_for_status()
                time.sleep(self.delay)
                return resp.json()
            except Exception as exc:
                logger.warning("OpenAlex request failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAlex request failed after 3 attempts: {url}")

    # ------------------------------------------------------------------
    # Work lookup
    # ------------------------------------------------------------------

    def get_work_by_pmid(self, pmid: str) -> dict | None:
        """Return OpenAlex work record for *pmid*, or None if not found."""
        key = f"pmid:{pmid}"
        cached = cache_get("openalex_works", key)
        if cached is not None:
            return cached or None  # empty dict == not found sentinel

        url = f"{OPENALEX_BASE}/works/pmid:{pmid}"
        try:
            data = self._get(url)
            cache_set("openalex_works", key, data)
            return data
        except Exception:
            cache_set("openalex_works", key, {})
            return None

    def get_work_by_doi(self, doi: str) -> dict | None:
        """Return OpenAlex work record for *doi*, or None."""
        # Normalise DOI: strip leading https://doi.org/ if present
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
        key = f"doi:{clean_doi}"
        cached = cache_get("openalex_works", key)
        if cached is not None:
            return cached or None

        url = f"{OPENALEX_BASE}/works/doi:{clean_doi}"
        try:
            data = self._get(url)
            cache_set("openalex_works", key, data)
            return data
        except Exception:
            cache_set("openalex_works", key, {})
            return None

    def search_work_by_title(self, title: str) -> dict | None:
        """
        Search OpenAlex for a work by title and return the best match, or None.
        Uses the OpenAlex search endpoint.
        """
        if not title:
            return None
        key = f"title_search:{title}"
        cached = cache_get("openalex_works", key)
        if cached is not None:
            return cached or None

        data = self._get(
            f"{OPENALEX_BASE}/works",
            {"search": title, "per-page": 1},
        )
        results = data.get("results", [])
        result = results[0] if results else {}
        cache_set("openalex_works", key, result)
        return result or None

    def enrich_paper(self, paper: dict) -> dict:
        """
        Given a paper dict (with pmid / doi / title keys), fetch the matching
        OpenAlex work and merge key fields back into *paper*.

        Adds:
            openalex_id, cited_by_count, authorships, concepts,
            referenced_works_count, publication_year
        """
        work: dict | None = None

        if paper.get("pmid"):
            work = self.get_work_by_pmid(paper["pmid"])
        if not work and paper.get("doi"):
            work = self.get_work_by_doi(paper["doi"])
        if not work and paper.get("title"):
            work = self.search_work_by_title(paper["title"])

        if not work:
            paper.setdefault("openalex_id", "")
            paper.setdefault("cited_by_count", 0)
            paper.setdefault("authorships", [])
            paper.setdefault("concepts", [])
            paper.setdefault("referenced_works_count", 0)
            return paper

        paper["openalex_id"] = work.get("id", "")
        paper["cited_by_count"] = work.get("cited_by_count", 0)
        paper["authorships"] = work.get("authorships", [])
        paper["concepts"] = [
            {"display_name": c.get("display_name", ""), "score": c.get("score", 0)}
            for c in work.get("concepts", [])
        ]
        paper["referenced_works_count"] = len(work.get("referenced_works", []))
        if not paper.get("year") or paper["year"] == "":
            paper["year"] = str(work.get("publication_year", ""))

        return paper

    # ------------------------------------------------------------------
    # Author lookup
    # ------------------------------------------------------------------

    def get_author(self, openalex_author_id: str) -> dict | None:
        """
        Return OpenAlex author record for *openalex_author_id*.

        *openalex_author_id* can be the full URL
        (``https://openalex.org/A12345678``) or just the short ID
        (``A12345678``).
        """
        # Normalise to short ID
        short_id = openalex_author_id.split("/")[-1]
        cached = cache_get("openalex_authors", short_id)
        if cached is not None:
            return cached or None

        url = f"{OPENALEX_BASE}/authors/{short_id}"
        try:
            data = self._get(url)
            cache_set("openalex_authors", short_id, data)
            return data
        except Exception:
            cache_set("openalex_authors", short_id, {})
            return None

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def enrich_papers(self, papers: list[dict]) -> list[dict]:
        """
        Enrich a list of paper dicts with OpenAlex metadata.

        Enrichment is skipped for papers that already have ``openalex_id``
        populated.
        """
        enriched = []
        for i, paper in enumerate(papers):
            if paper.get("openalex_id"):
                enriched.append(paper)
                continue
            logger.info(
                "Enriching paper %d/%d (pmid=%s)",
                i + 1,
                len(papers),
                paper.get("pmid", "?"),
            )
            enriched.append(self.enrich_paper(paper))
        return enriched
