"""
site_scorer.py – Compute candidate-site-level composite KOL scores.

Composite KOL score formula (weights must sum to 1.0):
    composite_kol_score =
        0.45 * site_neurosurgical_kol_strength   (surgical relevance of top KOLs)
      + 0.25 * site_network_engagement            (co-authorship network centrality)
      + 0.20 * site_depth                         (breadth of site's author pool)
      + 0.10 * site_confidence                    (affiliation matching confidence)
    * 100   (scaled to 0–100)

All four components are min-max normalised across candidate sites before
combination.  A single scored site receives 0.5 for each equal-valued
component (neutral midpoint).

Only candidate sites (from longlist_csv) appear in site_scores.csv.
Sites with no matched authors receive NaN component scores and an
evidence_status of "no_candidate_authors_matched" or "alias_issue_likely".

Author deduplication
--------------------
Before site aggregation, authors are deduplicated by:
    1. OpenAlex author_id (keep row with highest author_kol_score)
    2. Normalised display_name + canonical_site (name collision within a site)

Possible duplicates (same site, similar name, different author_id) are
written to outputs/possible_duplicate_authors.csv for human review.
"""

import logging
import math
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from src.utils import normalise_text, output_dir

logger = logging.getLogger(__name__)

# Composite score weights (must sum to 1.0)
_W_NEURO   = 0.45
_W_NETWORK = 0.25
_W_DEPTH   = 0.20
_W_CONF    = 0.10

# Minimum site_confidence for an author to contribute to site scoring
_CONF_THRESHOLD = 0.65

# rapidfuzz threshold above which two different author_ids at the same site
# are flagged as possible duplicates
_NAME_SIM_THRESHOLD = 92.0


class SiteScorer:
    """
    Aggregate author-level metrics to candidate-site composite KOL scores.

    Only sites listed in *longlist_csv* are emitted in site_scores.csv and
    kol_candidates.csv.  All other sites still participate in the author
    network and appear in authors.csv / match_audit.csv for context.
    """

    def __init__(
        self,
        longlist_csv: str | Path | None = None,
        aliases_csv: str | Path | None = None,
    ) -> None:
        """
        Parameters
        ----------
        longlist_csv : path to data/site_longlist.csv; loads candidate sites
                       with country and region metadata.
        aliases_csv  : path to data/site_aliases.csv; used to detect
                       alias-coverage gaps when assigning evidence_status.
        """
        self._candidate_meta: dict[str, dict] = {}   # site -> {country, region}
        self._sites_in_aliases: set[str] = set()     # sites present in aliases.csv

        if longlist_csv:
            self._load_longlist(longlist_csv)
        if aliases_csv:
            self._load_aliases_sites(aliases_csv)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_longlist(self, path: str | Path) -> None:
        try:
            df = pd.read_csv(path)
            site_col = next(
                (c for c in df.columns if normalise_text(c) == "site"),
                df.columns[0],
            )
            region_col = next(
                (c for c in df.columns
                 if "region" in normalise_text(c) or "state" in normalise_text(c)),
                None,
            )
            country_col = next(
                (c for c in df.columns if normalise_text(c) == "country"),
                None,
            )
            for _, row in df.iterrows():
                site = str(row[site_col]).strip()
                if not site or site.lower() == "nan":
                    continue
                self._candidate_meta[site] = {
                    "country": str(row[country_col]).strip() if country_col else "",
                    "region": str(row[region_col]).strip() if region_col else "",
                }
            logger.info(
                "Loaded %d candidate sites from %s", len(self._candidate_meta), path
            )
        except Exception as exc:
            logger.warning("Could not load longlist CSV '%s': %s", path, exc)

    def _load_aliases_sites(self, path: str | Path) -> None:
        """Record which canonical sites appear in site_aliases.csv."""
        try:
            df = pd.read_csv(path)
            if "canonical_site" in df.columns:
                self._sites_in_aliases = set(
                    df["canonical_site"].dropna().str.strip()
                )
        except Exception as exc:
            logger.warning("Could not load aliases CSV '%s': %s", path, exc)

    @property
    def candidate_sites(self) -> set[str]:
        """Return the set of candidate site names from the longlist."""
        return set(self._candidate_meta.keys())

    # ------------------------------------------------------------------
    # Author deduplication
    # ------------------------------------------------------------------

    def _deduplicate(
        self, authors_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Deduplicate *authors_df* and identify possible duplicate pairs.

        Steps
        -----
        1. Sort by author_kol_score desc; drop duplicate author_id rows.
        2. Within each (canonical_site, normalised display_name), keep the
           highest-scoring row (same-name dedup within a site).

        Possible-duplicate detection
        ----------------------------
        Within each canonical_site, flag pairs with *different* author_ids
        whose normalised display_names have a rapidfuzz token_sort_ratio
        above _NAME_SIM_THRESHOLD.

        Returns
        -------
        (deduped_df, possible_duplicates_df)
        """
        if authors_df.empty:
            _empty_dups = pd.DataFrame(
                columns=[
                    "canonical_site", "author_id_1", "display_name_1",
                    "author_id_2", "display_name_2", "name_similarity",
                ]
            )
            return authors_df, _empty_dups

        df = authors_df.copy()

        # Step 1: deduplicate by author_id
        df = (
            df.sort_values("author_kol_score", ascending=False)
            .drop_duplicates(subset=["author_id"], keep="first")
        )

        # Step 2: within-site name dedup
        if "display_name" in df.columns:
            df["_name_key"] = df["display_name"].fillna("").str.lower().str.strip()
            df = (
                df.sort_values("author_kol_score", ascending=False)
                .drop_duplicates(subset=["_name_key", "canonical_site"], keep="first")
                .drop(columns=["_name_key"])
            )

        # Possible duplicate detection (fuzzy pairs within same site)
        dup_rows: list[dict] = []
        if "display_name" in df.columns:
            for site, grp in df[df["canonical_site"] != ""].groupby("canonical_site"):
                entries = (
                    grp[["author_id", "display_name"]].dropna().values.tolist()
                )
                for i in range(len(entries)):
                    for j in range(i + 1, len(entries)):
                        id_a, name_a = entries[i]
                        id_b, name_b = entries[j]
                        if id_a == id_b:
                            continue
                        score = fuzz.token_sort_ratio(
                            normalise_text(str(name_a)),
                            normalise_text(str(name_b)),
                        )
                        if score >= _NAME_SIM_THRESHOLD:
                            dup_rows.append(
                                {
                                    "canonical_site": site,
                                    "author_id_1": id_a,
                                    "display_name_1": name_a,
                                    "author_id_2": id_b,
                                    "display_name_2": name_b,
                                    "name_similarity": round(score, 1),
                                }
                            )

        dup_df = (
            pd.DataFrame(dup_rows)
            if dup_rows
            else pd.DataFrame(
                columns=[
                    "canonical_site", "author_id_1", "display_name_1",
                    "author_id_2", "display_name_2", "name_similarity",
                ]
            )
        )
        return df.reset_index(drop=True), dup_df

    # ------------------------------------------------------------------
    # Main scoring
    # ------------------------------------------------------------------

    def compute(
        self,
        authors_df: pd.DataFrame,
        paper_label_sets: dict | None = None,
        top_n_authors: int = 20,
    ) -> pd.DataFrame:
        """
        Compute candidate-site-level composite KOL scores.

        Parameters
        ----------
        authors_df       : author metrics DataFrame with columns
                           author_id, display_name, author_kol_score,
                           core_surgical_count, [surgery_adjacent_count],
                           pagerank, degree_centrality, centrifugal_score,
                           canonical_site, site_confidence.
        paper_label_sets : optional dict from AuthorNetwork.paper_label_sets()
                           {author_id: {label: frozenset_of_paper_ids}}.
                           When supplied, n_core_surgical_papers and
                           n_surgery_adjacent_papers are unique paper counts
                           across all site-affiliated authors; otherwise they
                           are the sum of per-author counts.
        top_n_authors    : max author names in top_kol_names column.

        Returns
        -------
        DataFrame with one row per candidate site sorted by composite_kol_score
        descending.  All candidate sites from the longlist appear; unscored
        sites have NaN component scores and a descriptive evidence_status.
        """
        if authors_df.empty or "canonical_site" not in authors_df.columns:
            logger.warning("authors_df is empty or missing canonical_site column.")
            return self._empty_site_df()

        # Filter to confidently matched authors
        matched = authors_df[
            (authors_df["canonical_site"] != "")
            & (authors_df["site_confidence"] >= _CONF_THRESHOLD)
        ].copy()

        # Deduplicate
        matched, _ = self._deduplicate(matched)

        # Restrict to candidate sites
        if self._candidate_meta:
            scored_df = matched[
                matched["canonical_site"].isin(self._candidate_meta)
            ].copy()
        else:
            scored_df = matched.copy()

        has_names       = "display_name" in scored_df.columns
        has_sadj        = "surgery_adjacent_count" in scored_df.columns
        has_pagerank    = "pagerank" in scored_df.columns
        has_degree      = "degree_centrality" in scored_df.columns
        has_centrifugal = "centrifugal_score" in scored_df.columns

        site_rows: list[dict] = []

        for site, group in scored_df.groupby("canonical_site"):
            g = group.sort_values("author_kol_score", ascending=False)
            n_authors = len(g)
            scores     = g["author_kol_score"].tolist()
            core_cnts  = g["core_surgical_count"].tolist()
            sadj_cnts  = (
                g["surgery_adjacent_count"].tolist()
                if has_sadj
                else [0] * n_authors
            )

            # ── Component 1: site_neurosurgical_kol_strength ───────────────
            # Author KOL score weighted by surgical specificity
            #   adj_kol_i = kol_i * (0.5 + 0.5 * core_i / max(1, core_i + sadj_i))
            adj_kols = [
                kol * (0.5 + 0.5 * cs / max(1, cs + sa))
                for kol, cs, sa in zip(scores, core_cnts, sadj_cnts)
            ]
            top5_adj = adj_kols[:5]
            neuro_raw = sum(top5_adj) / len(top5_adj) if top5_adj else 0.0

            # ── Component 2: site_network_engagement ───────────────────────
            pr_vals   = g["pagerank"].tolist()          if has_pagerank    else [0.0] * n_authors
            deg_vals  = g["degree_centrality"].tolist() if has_degree      else [0.0] * n_authors
            cent_vals = g["centrifugal_score"].tolist() if has_centrifugal else [0.0] * n_authors
            top5_net  = [
                0.5 * pr + 0.3 * dg + 0.2 * ct
                for pr, dg, ct in zip(pr_vals[:5], deg_vals[:5], cent_vals[:5])
            ]
            network_raw = sum(top5_net) / len(top5_net) if top5_net else 0.0

            # ── Component 3: site_depth ────────────────────────────────────
            depth_raw = math.log1p(n_authors)

            # ── Component 4: site_confidence ──────────────────────────────
            confidence_raw = float(g["site_confidence"].mean())

            # ── Paper counts (unique across all site authors) ──────────────
            if paper_label_sets:
                core_papers: set = set()
                sadj_papers: set = set()
                for aid in g["author_id"]:
                    amap = paper_label_sets.get(aid, {})
                    core_papers |= amap.get("core_surgical", set())
                    sadj_papers |= amap.get("surgery_adjacent", set())
                n_core = len(core_papers)
                n_sadj = len(sadj_papers)
            else:
                n_core = int(g["core_surgical_count"].sum())
                n_sadj = int(g["surgery_adjacent_count"].sum()) if has_sadj else 0

            # ── Top KOL names ──────────────────────────────────────────────
            top_names = (
                g["display_name"].dropna().head(top_n_authors).tolist()
                if has_names
                else []
            )

            site_rows.append(
                {
                    "canonical_site":         site,
                    "_neuro_raw":             neuro_raw,
                    "_network_raw":           network_raw,
                    "_depth_raw":             depth_raw,
                    "_confidence_raw":        confidence_raw,
                    "n_kol_candidates":       n_authors,
                    "n_core_surgical_papers": n_core,
                    "n_surgery_adjacent_papers": n_sadj,
                    "top_kol_names":          "; ".join(top_names),
                    "evidence_status":        "scored",
                }
            )

        site_df = pd.DataFrame(site_rows) if site_rows else pd.DataFrame()

        if not site_df.empty:
            site_df = self._apply_composite(site_df)

        site_df = self._merge_metadata_and_missing(site_df)

        site_df.sort_values(
            "composite_kol_score", ascending=False, na_position="last", inplace=True
        )
        site_df.reset_index(drop=True, inplace=True)
        return site_df

    # ------------------------------------------------------------------
    # Composite formula helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_composite(df: pd.DataFrame) -> pd.DataFrame:
        """
        Min-max normalise the four raw components and compute composite_kol_score.

        When all values of a component are equal (e.g. only one site scored),
        the normalised value is set to 0.5 (neutral midpoint) to avoid
        collapsing all scores to zero.
        """

        def _norm(col: str) -> pd.Series:
            mn, mx = df[col].min(), df[col].max()
            if mx == mn:
                return pd.Series(
                    0.5 if mx > 0 else 0.0, index=df.index, dtype=float
                )
            return (df[col] - mn) / (mx - mn)

        neuro_n   = _norm("_neuro_raw")
        network_n = _norm("_network_raw")
        depth_n   = _norm("_depth_raw")
        conf_n    = _norm("_confidence_raw")

        composite = (
            _W_NEURO   * neuro_n
            + _W_NETWORK * network_n
            + _W_DEPTH   * depth_n
            + _W_CONF    * conf_n
        ) * 100

        df["site_neurosurgical_kol_strength"] = neuro_n.round(4)
        df["site_network_engagement"]          = network_n.round(4)
        df["site_depth"]                       = depth_n.round(4)
        df["site_confidence"]                  = conf_n.round(4)
        df["composite_kol_score"]              = composite.round(2)

        df.drop(
            columns=["_neuro_raw", "_network_raw", "_depth_raw", "_confidence_raw"],
            inplace=True,
            errors="ignore",
        )
        return df

    # ------------------------------------------------------------------
    # Metadata merge + missing sites
    # ------------------------------------------------------------------

    def _merge_metadata_and_missing(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach country/region from the longlist and add rows for candidate
        sites that have no scored authors.

        evidence_status for missing sites
        ----------------------------------
        "alias_issue_likely"           – site is in aliases.csv but no author
                                         matched (possibly country filter or
                                         alias gap)
        "no_candidate_authors_matched" – site is NOT in aliases.csv at all
        """
        scored_sites = (
            set(scored_df["canonical_site"]) if not scored_df.empty else set()
        )

        # Build rows for unscored candidate sites
        missing_rows: list[dict] = []
        for site, meta in self._candidate_meta.items():
            if site in scored_sites:
                continue
            in_aliases = site in self._sites_in_aliases
            status = (
                "alias_issue_likely"
                if in_aliases
                else "no_candidate_authors_matched"
            )
            missing_rows.append(
                {
                    "canonical_site":                 site,
                    "country":                        meta.get("country", ""),
                    "region":                         meta.get("region", ""),
                    "composite_kol_score":            float("nan"),
                    "site_neurosurgical_kol_strength": float("nan"),
                    "site_network_engagement":         float("nan"),
                    "site_depth":                     float("nan"),
                    "site_confidence":                float("nan"),
                    "n_kol_candidates":               0,
                    "n_core_surgical_papers":         0,
                    "n_surgery_adjacent_papers":      0,
                    "top_kol_names":                  "",
                    "evidence_status":                status,
                }
            )

        missing_df = pd.DataFrame(missing_rows)

        if scored_df.empty:
            combined = missing_df if not missing_df.empty else self._empty_site_df()
            return combined

        # Merge metadata into scored sites
        if self._candidate_meta:
            meta_df = pd.DataFrame(
                [
                    {
                        "canonical_site": s,
                        "country": m.get("country", ""),
                        "region":  m.get("region", ""),
                    }
                    for s, m in self._candidate_meta.items()
                ]
            )
            scored_df = scored_df.merge(meta_df, on="canonical_site", how="left")
        else:
            if "country" not in scored_df.columns:
                scored_df["country"] = ""
            if "region" not in scored_df.columns:
                scored_df["region"] = ""

        scored_df["country"] = scored_df.get("country", pd.Series()).fillna("")
        scored_df["region"]  = scored_df.get("region", pd.Series()).fillna("")

        # Enforce final column order
        col_order = [
            "canonical_site", "country", "region",
            "composite_kol_score",
            "site_neurosurgical_kol_strength", "site_network_engagement",
            "site_depth", "site_confidence",
            "n_kol_candidates", "n_core_surgical_papers", "n_surgery_adjacent_papers",
            "top_kol_names", "evidence_status",
        ]
        for c in col_order:
            if c not in scored_df.columns:
                scored_df[c] = float("nan")
        scored_df = scored_df[col_order]

        combined = (
            pd.concat([scored_df, missing_df], ignore_index=True)
            if not missing_df.empty
            else scored_df
        )
        return combined

    def _empty_site_df(self) -> pd.DataFrame:
        cols = [
            "canonical_site", "country", "region",
            "composite_kol_score",
            "site_neurosurgical_kol_strength", "site_network_engagement",
            "site_depth", "site_confidence",
            "n_kol_candidates", "n_core_surgical_papers", "n_surgery_adjacent_papers",
            "top_kol_names", "evidence_status",
        ]
        return pd.DataFrame(columns=cols)

    # ------------------------------------------------------------------
    # KOL candidate list
    # ------------------------------------------------------------------

    def kol_candidates(
        self,
        authors_df: pd.DataFrame,
        top_centile: float = 30.0,
    ) -> pd.DataFrame:
        """
        Return candidate-site-affiliated authors in the top *top_centile* percent.

        Only confidently matched (site_confidence ≥ threshold) authors at
        candidate sites are considered.  Authors are deduplicated before the
        centile threshold is applied.

        A kol_centile_rank column (1 = highest) is prepended.
        """
        required = {"author_kol_score", "canonical_site", "site_confidence"}
        if authors_df.empty or not required.issubset(authors_df.columns):
            logger.warning("authors_df is missing required columns for kol_candidates.")
            return pd.DataFrame()

        df = authors_df[
            (authors_df["canonical_site"] != "")
            & (authors_df["site_confidence"] >= _CONF_THRESHOLD)
        ].copy()

        if df.empty:
            return pd.DataFrame()

        if self._candidate_meta:
            df = df[df["canonical_site"].isin(self._candidate_meta)]
            if df.empty:
                logger.warning("No candidate-site-affiliated authors for KOL list.")
                return pd.DataFrame()

        df, _ = self._deduplicate(df)

        threshold = df["author_kol_score"].quantile((100.0 - top_centile) / 100.0)
        candidates = df[df["author_kol_score"] >= threshold].copy()
        candidates.sort_values("author_kol_score", ascending=False, inplace=True)
        candidates.reset_index(drop=True, inplace=True)
        candidates.insert(0, "kol_centile_rank", range(1, len(candidates) + 1))

        logger.info(
            "KOL candidates (top %.0f %%, candidate sites): %d authors (score ≥ %.2f)",
            top_centile, len(candidates), threshold,
        )
        return candidates

    # ------------------------------------------------------------------
    # Missing-candidate audit
    # ------------------------------------------------------------------

    def missing_candidates(self, site_df: pd.DataFrame) -> pd.DataFrame:
        """Return candidate sites with evidence_status != 'scored'."""
        if site_df.empty or "evidence_status" not in site_df.columns:
            cols = ["canonical_site", "country", "region", "evidence_status"]
            return pd.DataFrame(columns=cols)
        cols = [c for c in ["canonical_site", "country", "region", "evidence_status"]
                if c in site_df.columns]
        missing = site_df[site_df["evidence_status"] != "scored"][cols].copy()
        return missing.reset_index(drop=True)

    def save_missing_candidates(
        self, site_df: pd.DataFrame, path: str | None = None
    ) -> Path:
        dest = Path(path) if path else output_dir() / "missing_candidate_sites.csv"
        df = self.missing_candidates(site_df)
        df.to_csv(dest, index=False)
        logger.info("Saved %d missing candidate rows to %s", len(df), dest)
        return dest

    # ------------------------------------------------------------------
    # Possible duplicate authors
    # ------------------------------------------------------------------

    def possible_duplicate_authors(self, authors_df: pd.DataFrame) -> pd.DataFrame:
        """Return the possible-duplicate-authors DataFrame."""
        _, dups = self._deduplicate(authors_df)
        return dups

    def save_possible_duplicates(
        self, authors_df: pd.DataFrame, path: str | None = None
    ) -> Path:
        dest = (
            Path(path) if path else output_dir() / "possible_duplicate_authors.csv"
        )
        df = self.possible_duplicate_authors(authors_df)
        df.to_csv(dest, index=False)
        logger.info("Saved %d possible duplicate rows to %s", len(df), dest)
        return dest

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, site_df: pd.DataFrame, path: str | None = None) -> Path:
        dest = Path(path) if path else output_dir() / "site_scores.csv"
        site_df.to_csv(dest, index=False)
        logger.info("Saved %d site rows to %s", len(site_df), dest)
        return dest

    def save_kol_candidates(
        self, candidates_df: pd.DataFrame, path: str | None = None
    ) -> Path:
        dest = Path(path) if path else output_dir() / "kol_candidates.csv"
        candidates_df.to_csv(dest, index=False)
        logger.info("Saved %d KOL candidate rows to %s", len(candidates_df), dest)
        return dest
