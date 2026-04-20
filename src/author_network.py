"""
author_network.py – Build an author co-authorship graph and compute metrics.

Graph nodes  = OpenAlex author IDs
Graph edges  = co-authorship (weighted by paper weight)

Author metrics computed per node:
    weighted_citation_score   – sum(paper_weight * cited_by_count) per paper
    core_surgical_count       – number of core_surgical papers
    recency_score             – recency-weighted activity (recent papers count more)
    degree_centrality         – fraction of co-authors relative to all authors
    pagerank                  – PageRank over the co-authorship graph

Usage example
-------------
>>> from src.author_network import AuthorNetwork
>>> net = AuthorNetwork()
>>> net.build(papers)        # papers enriched with authorships + label
>>> metrics = net.compute_metrics()
>>> net.save_authors_csv(metrics)
"""

import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from src.utils import output_dir

logger = logging.getLogger(__name__)

CURRENT_YEAR: int = datetime.now().year  # used for recency decay


def _recency_weight(year_str: str, half_life: float = 5.0) -> float:
    """Exponential decay: papers from ``half_life`` years ago contribute 0.5."""
    try:
        year = int(year_str)
    except (ValueError, TypeError):
        return 0.5  # unknown year gets neutral weight
    age = max(0, CURRENT_YEAR - year)
    return math.exp(-math.log(2) * age / half_life)


class AuthorNetwork:
    """Build and analyse the surgical GBM author co-authorship network."""

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        # author_id -> list of paper-level contribution records
        self._contributions: dict[str, list[dict]] = defaultdict(list)
        # author_id -> display_name (for labelling)
        self._names: dict[str, str] = {}
        # author_id -> list of raw authorship affiliation strings
        self._affiliations: dict[str, list[str]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build(self, papers: list[dict]) -> None:
        """
        Populate the co-authorship graph from *papers*.

        Each paper must have:
            authorships  – list of OpenAlex authorship dicts
            label        – classification label (core_surgical / surgery_adjacent / …)
            paper_weight – numeric weight from classifier
            cited_by_count – int
            year         – string publication year
        """
        for paper in papers:
            label = paper.get("label", "non_core")
            if label == "non_core":
                continue  # exclude non-core papers from network

            weight = float(paper.get("paper_weight", 1.0))
            citations = int(paper.get("cited_by_count", 0))
            year = str(paper.get("year", ""))

            authorships = paper.get("authorships", [])
            author_ids = []

            for auth in authorships:
                # OpenAlex authorship dict structure:
                # {"author": {"id": "...", "display_name": "..."}, "institutions": [...]}
                author_obj = auth.get("author") or {}
                author_id = author_obj.get("id", "")
                if not author_id:
                    continue
                # Use short ID (strip URL prefix)
                short_id = author_id.split("/")[-1]

                display_name = author_obj.get("display_name", short_id)
                self._names[short_id] = display_name

                # Collect affiliations for site mapping
                for inst in auth.get("institutions", []):
                    raw_name = inst.get("display_name", "")
                    country = inst.get("country_code", "")
                    if raw_name:
                        self._affiliations[short_id].append(
                            f"{raw_name}|{country}"
                        )

                # Record this author's contribution to the paper
                self._contributions[short_id].append(
                    {
                        "paper_weight": weight,
                        "cited_by_count": citations,
                        "recency_weight": _recency_weight(year),
                        "label": label,
                    }
                )

                # Add node
                if short_id not in self.graph:
                    self.graph.add_node(short_id, display_name=display_name)

                author_ids.append(short_id)

            # Add co-authorship edges (weight proportional to paper weight)
            for i, a in enumerate(author_ids):
                for b in author_ids[i + 1 :]:
                    if self.graph.has_edge(a, b):
                        self.graph[a][b]["weight"] += weight
                    else:
                        self.graph.add_edge(a, b, weight=weight)

        logger.info(
            "Built author graph: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute_metrics(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per author and computed metrics.

        Columns:
            author_id, display_name, weighted_citation_score,
            core_surgical_count, recency_score, degree_centrality, pagerank,
            affiliations (pipe-separated unique strings)
        """
        if self.graph.number_of_nodes() == 0:
            logger.warning("Author graph is empty – no metrics to compute.")
            return pd.DataFrame()

        # Degree centrality
        degree_centrality = nx.degree_centrality(self.graph)

        # PageRank (scipy is required by NetworkX; fall back to degree centrality)
        try:
            pagerank = nx.pagerank(self.graph, weight="weight", max_iter=200)
        except (nx.exception.PowerIterationFailedConvergence, ModuleNotFoundError):
            logger.warning(
                "PageRank unavailable (scipy missing or convergence failure); "
                "using degree centrality as fallback."
            )
            pagerank = degree_centrality

        rows = []
        for author_id, contribs in self._contributions.items():
            weighted_citation = sum(
                c["paper_weight"] * c["cited_by_count"] for c in contribs
            )
            core_count = sum(1 for c in contribs if c["label"] == "core_surgical")
            recency = sum(
                c["paper_weight"] * c["recency_weight"] for c in contribs
            )
            # Unique affiliations (deduplicated, US-filtered later by site matcher)
            aff_list = list(dict.fromkeys(self._affiliations.get(author_id, [])))

            rows.append(
                {
                    "author_id": author_id,
                    "display_name": self._names.get(author_id, author_id),
                    "weighted_citation_score": round(weighted_citation, 2),
                    "core_surgical_count": core_count,
                    "recency_score": round(recency, 4),
                    "degree_centrality": round(
                        degree_centrality.get(author_id, 0.0), 6
                    ),
                    "pagerank": round(pagerank.get(author_id, 0.0), 8),
                    "affiliations": "; ".join(aff_list),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            # Composite author score: normalise and combine
            df = self._add_composite_score(df)
            df.sort_values("author_kol_score", ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def _add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a composite KOL score (0–100 scale) to *df*.

        Weights:
            weighted_citation_score  40 %
            core_surgical_count      30 %
            recency_score            15 %
            pagerank                 15 %
        """
        def _norm(col: str) -> pd.Series:
            mn, mx = df[col].min(), df[col].max()
            if mx == mn:
                return pd.Series(0.0, index=df.index)
            return (df[col] - mn) / (mx - mn)

        df["author_kol_score"] = (
            0.40 * _norm("weighted_citation_score")
            + 0.30 * _norm("core_surgical_count")
            + 0.15 * _norm("recency_score")
            + 0.15 * _norm("pagerank")
        ) * 100
        df["author_kol_score"] = df["author_kol_score"].round(2)
        return df

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save_authors_csv(
        self, df: pd.DataFrame, path: str | None = None
    ) -> Path:
        """Save author metrics DataFrame as CSV."""
        dest = Path(path) if path else output_dir() / "authors.csv"
        df.to_csv(dest, index=False)
        logger.info("Saved %d author rows to %s", len(df), dest)
        return dest
