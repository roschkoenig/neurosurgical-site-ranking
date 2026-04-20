"""
site_scorer.py – Compute site-level KOL and cluster-depth scores.

Input:  authors_df   – author metrics + canonical_site column
Output: site_scores  – one row per canonical site with aggregated metrics
        kol_candidates – top-centile KOL authors affiliated with canonical sites

Site metrics computed:
    top_author_score    – highest author_kol_score among affiliated authors
    top3_sum            – sum of top-3 author_kol_scores
    authors_above_50    – count of authors with author_kol_score ≥ 50
    mean_top5           – mean of top-5 author_kol_scores
    cluster_depth       – number of unique authors affiliated with the site
    site_kol_score      – composite weighted score (0–100)
    top_authors         – semicolon-separated list of up to 20 top author names

KOL candidate output (kol_candidates):
    All author columns from authors_df plus kol_centile_rank, filtered to
    authors with a confirmed canonical-site affiliation whose author_kol_score
    is in the top ``top_centile`` percent (default 30 %).

Usage example
-------------
>>> from src.site_scorer import SiteScorer
>>> scorer = SiteScorer()
>>> site_scores = scorer.compute(authors_df)
>>> scorer.save(site_scores)
>>> candidates = scorer.kol_candidates(authors_df)
>>> scorer.save_kol_candidates(candidates)
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import output_dir

logger = logging.getLogger(__name__)


class SiteScorer:
    """Aggregate author-level KOL scores to canonical site-level metrics."""

    def __init__(
        self,
        top_author_weight: float = 0.30,
        top3_sum_weight: float = 0.25,
        above_threshold_weight: float = 0.20,
        mean_top5_weight: float = 0.15,
        cluster_depth_weight: float = 0.10,
        score_threshold: float = 50.0,
    ) -> None:
        """
        Parameters
        ----------
        *_weight:        fractional weights summing to 1.0 for the composite score
        score_threshold: author_kol_score cut-off for the ``authors_above_N`` metric
        """
        self.weights = {
            "top_author_score": top_author_weight,
            "top3_sum": top3_sum_weight,
            "authors_above_threshold": above_threshold_weight,
            "mean_top5": mean_top5_weight,
            "cluster_depth": cluster_depth_weight,
        }
        self.score_threshold = score_threshold

    # ------------------------------------------------------------------

    def compute(
        self, authors_df: pd.DataFrame, top_n_authors: int = 20
    ) -> pd.DataFrame:
        """
        Compute site-level metrics from *authors_df*.

        *authors_df* must have columns:
            author_id, author_kol_score, canonical_site, site_confidence

        Optional column:
            display_name – used to populate the ``top_authors`` list

        Only authors with site_confidence ≥ 0.65 are included in site
        aggregation (low-confidence matches are excluded to avoid noise).

        Parameters
        ----------
        authors_df    : author metrics DataFrame (see above)
        top_n_authors : maximum number of author names to include in
                        the ``top_authors`` column (default 20)

        Returns a DataFrame sorted by site_kol_score descending.
        """
        if authors_df.empty or "canonical_site" not in authors_df.columns:
            logger.warning("authors_df is empty or missing canonical_site column.")
            return pd.DataFrame()

        # Filter to confidently matched, non-empty sites
        df = authors_df[
            (authors_df["canonical_site"] != "")
            & (authors_df["site_confidence"] >= 0.65)
        ].copy()

        if df.empty:
            logger.warning("No confidently matched authors found for site scoring.")
            return pd.DataFrame()

        has_names = "display_name" in df.columns

        rows = []
        for site, group in df.groupby("canonical_site"):
            group_sorted = group.sort_values("author_kol_score", ascending=False)
            scores = group_sorted["author_kol_score"].tolist()
            top_score = scores[0] if scores else 0.0
            top3 = sum(scores[:3])
            mean_top5 = sum(scores[:5]) / min(len(scores), 5) if scores else 0.0
            above_thresh = int(sum(1 for s in scores if s >= self.score_threshold))
            depth = len(scores)

            if has_names:
                top_names = (
                    group_sorted["display_name"]
                    .dropna()
                    .head(top_n_authors)
                    .tolist()
                )
            else:
                top_names = []

            rows.append(
                {
                    "canonical_site": site,
                    "top_author_score": round(top_score, 2),
                    "top3_sum": round(top3, 2),
                    "authors_above_threshold": above_thresh,
                    "mean_top5": round(mean_top5, 2),
                    "cluster_depth": depth,
                    "top_authors": "; ".join(top_names),
                }
            )

        site_df = pd.DataFrame(rows)
        site_df = self._add_composite_score(site_df)
        site_df.sort_values("site_kol_score", ascending=False, inplace=True)
        site_df.reset_index(drop=True, inplace=True)
        return site_df

    def _add_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise each metric and combine with configured weights."""

        def _norm(col: str) -> pd.Series:
            mn, mx = df[col].min(), df[col].max()
            if mx == mn:
                return pd.Series(0.0, index=df.index)
            return (df[col] - mn) / (mx - mn)

        composite = sum(
            w * _norm(col) for col, w in self.weights.items()
        )
        df["site_kol_score"] = (composite * 100).round(2)
        return df

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, site_df: pd.DataFrame, path: str | None = None) -> Path:
        """Save site scores to CSV."""
        dest = Path(path) if path else output_dir() / "site_scores.csv"
        site_df.to_csv(dest, index=False)
        logger.info("Saved %d site rows to %s", len(site_df), dest)
        return dest

    # ------------------------------------------------------------------
    # KOL candidate list
    # ------------------------------------------------------------------

    def kol_candidates(
        self,
        authors_df: pd.DataFrame,
        top_centile: float = 30.0,
    ) -> pd.DataFrame:
        """
        Return site-affiliated authors whose KOL score is in the top
        *top_centile* percent.

        Only authors with a confirmed canonical-site match
        (``site_confidence`` ≥ 0.65) are considered.  Among those, the
        score threshold is the (100 − *top_centile*)th percentile of
        ``author_kol_score``, so with the default ``top_centile=30`` the
        function returns authors at or above the 70th percentile.

        A ``kol_centile_rank`` column (1 = highest score) is added to the
        returned DataFrame.

        Parameters
        ----------
        authors_df  : author metrics DataFrame; must contain
                      ``author_kol_score``, ``canonical_site``, and
                      ``site_confidence`` columns.
        top_centile : percentage of top scorers to retain (default 30 %).

        Returns a DataFrame sorted by ``author_kol_score`` descending.
        """
        required = {"author_kol_score", "canonical_site", "site_confidence"}
        if authors_df.empty or not required.issubset(authors_df.columns):
            missing = required - set(authors_df.columns)
            logger.warning(
                "authors_df is empty or missing columns: %s", missing or "(empty)"
            )
            return pd.DataFrame()

        # Filter to confidently site-matched authors
        df = authors_df[
            (authors_df["canonical_site"] != "")
            & (authors_df["site_confidence"] >= 0.65)
        ].copy()

        if df.empty:
            logger.warning("No site-affiliated authors found for KOL candidate list.")
            return pd.DataFrame()

        # Determine the score threshold for the requested centile
        percentile_cutoff = 100.0 - top_centile
        threshold = df["author_kol_score"].quantile(percentile_cutoff / 100.0)

        candidates = df[df["author_kol_score"] >= threshold].copy()
        candidates.sort_values("author_kol_score", ascending=False, inplace=True)
        candidates.reset_index(drop=True, inplace=True)
        candidates.insert(0, "kol_centile_rank", range(1, len(candidates) + 1))

        logger.info(
            "KOL candidates (top %.0f %%): %d authors (score ≥ %.2f)",
            top_centile,
            len(candidates),
            threshold,
        )
        return candidates

    def save_kol_candidates(
        self, candidates_df: pd.DataFrame, path: str | None = None
    ) -> Path:
        """Save KOL candidates to CSV."""
        dest = Path(path) if path else output_dir() / "kol_candidates.csv"
        candidates_df.to_csv(dest, index=False)
        logger.info("Saved %d KOL candidate rows to %s", len(candidates_df), dest)
        return dest
