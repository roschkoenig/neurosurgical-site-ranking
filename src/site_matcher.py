"""
site_matcher.py – Map raw affiliation strings to canonical site names.

Matching pipeline (in order):
    1. Exact alias match       (match_method = "exact_alias")
    2. Fuzzy match             (match_method = "fuzzy")
    3. Optional LLM step       (match_method = "llm")   [only for unresolved]
    4. Unresolved bucket       (match_method = "unresolved")

The function ``match_affiliation`` is deterministic for steps 1–2.  The LLM
step (step 3) is only triggered when ``use_llm=True`` and the deterministic
confidence is below ``llm_threshold``.

Input CSV (data/site_aliases.csv):
    canonical_site, alias

Usage example
-------------
>>> from src.site_matcher import SiteMatcher
>>> matcher = SiteMatcher("data/site_aliases.csv")
>>> result = matcher.match_affiliation("Stanford University Medical Center, CA")
>>> print(result)
{'canonical_site': 'Stanford University', 'confidence': 0.95,
 'match_method': 'fuzzy'}
"""

import csv
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from rapidfuzz import fuzz, process

from src.utils import normalise_text, output_dir

logger = logging.getLogger(__name__)

FUZZY_HIGH_THRESHOLD = 88   # score ≥ this → high-confidence fuzzy match
FUZZY_LOW_THRESHOLD = 70    # score ≥ this → low-confidence (may trigger LLM)


class SiteMatcher:
    """
    Deterministic + optional LLM affiliation-to-site mapper.

    Parameters
    ----------
    aliases_csv:   path to CSV with columns canonical_site, alias
    use_llm:       enable LLM adjudication for low-confidence matches
    llm_threshold: fuzzy score below which LLM is invoked (0–100)
    llm_provider:  "openai" (API) or "ollama" (local Llama via Ollama)
    openai_model:  model name used for OpenAI LLM step
    ollama_model:  model name served by local Ollama (e.g., llama3.1:8b)
    ollama_url:    base URL for local Ollama API
    """

    def __init__(
        self,
        aliases_csv: str | Path,
        use_llm: bool = False,
        llm_threshold: float = FUZZY_HIGH_THRESHOLD,
        llm_provider: str = "openai",
        openai_model: str = "gpt-4o-mini",
        ollama_model: str = "llama3.1:8b",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self.use_llm = use_llm
        self.llm_threshold = llm_threshold
        self.llm_provider = llm_provider.strip().lower()
        self.openai_model = openai_model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url.rstrip("/")
        # If prerequisites are missing, disable LLM once and avoid repeated warnings.
        self._llm_disabled: bool = False

        if self.llm_provider not in {"openai", "ollama"}:
            logger.warning(
                "Unknown llm_provider '%s'; defaulting to openai.", self.llm_provider
            )
            self.llm_provider = "openai"

        # Load alias table
        self._alias_map: dict[str, str] = {}          # normalised alias -> canonical_site
        self._canonical_sites: list[str] = []
        self._canonical_aliases: dict[str, list[str]] = {}  # canonical -> [original aliases]
        self._load_aliases(aliases_csv)

        # Audit log (populated during matching)
        self._audit: list[dict] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_aliases(self, path: str | Path) -> None:
        """Load canonical_site / alias pairs from CSV."""
        canonical_set: set[str] = set()
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    canonical = (row.get("canonical_site") or "").strip()
                    alias = (row.get("alias") or "").strip()
                    if not canonical:
                        continue
                    canonical_set.add(canonical)
                    if alias:
                        self._alias_map[normalise_text(alias)] = canonical
                        # Track original (non-normalised) alias for targeted search
                        self._canonical_aliases.setdefault(canonical, []).append(alias)
                    # Also register the canonical name itself as an alias
                    self._alias_map[normalise_text(canonical)] = canonical
        except FileNotFoundError:
            logger.warning("Aliases CSV not found: %s – site matching will be limited.", path)

        self._canonical_sites = sorted(canonical_set)
        logger.info(
            "Loaded %d canonical sites, %d aliases",
            len(self._canonical_sites),
            len(self._alias_map),
        )

    # ------------------------------------------------------------------
    # Core matching logic
    # ------------------------------------------------------------------

    def match_affiliation(
        self, affiliation: str, author_id: str = ""
    ) -> dict[str, Any]:
        """
        Map a single raw affiliation string to a canonical site.

        Returns
        -------
        dict with keys:
            raw_affiliation, canonical_site, confidence, match_method
        """
        result: dict[str, Any] = {
            "raw_affiliation": affiliation,
            "canonical_site": "",
            "confidence": 0.0,
            "match_method": "unresolved",
            "author_id": author_id,
        }

        if not affiliation:
            self._audit.append(result)
            return result

        norm = normalise_text(affiliation)

        # ---- Step 1: Exact alias match ----
        if norm in self._alias_map:
            result["canonical_site"] = self._alias_map[norm]
            result["confidence"] = 1.0
            result["match_method"] = "exact_alias"
            self._audit.append(result)
            return result

        # Partial exact match: any alias that is a substring of the normalised affiliation
        for alias_norm, canonical in self._alias_map.items():
            if alias_norm and alias_norm in norm:
                result["canonical_site"] = canonical
                result["confidence"] = 0.95
                result["match_method"] = "exact_alias"
                self._audit.append(result)
                return result

        # ---- Step 2: Fuzzy match ----
        if self._canonical_sites:
            best = process.extractOne(
                norm,
                [normalise_text(s) for s in self._canonical_sites],
                scorer=fuzz.token_sort_ratio,
            )
            if best is not None:
                score, idx = best[1], best[2]
                canonical = self._canonical_sites[idx]
                if score >= FUZZY_HIGH_THRESHOLD:
                    result["canonical_site"] = canonical
                    result["confidence"] = round(score / 100, 3)
                    result["match_method"] = "fuzzy"
                    self._audit.append(result)
                    return result
                elif score >= FUZZY_LOW_THRESHOLD:
                    result["canonical_site"] = canonical
                    result["confidence"] = round(score / 100, 3)
                    result["match_method"] = "fuzzy"
                    # Low confidence – may try LLM below
                    if not self.use_llm or score >= self.llm_threshold:
                        self._audit.append(result)
                        return result

        # ---- Step 3: LLM adjudication (optional) ----
        if self.use_llm and result["confidence"] < self.llm_threshold / 100:
            llm_result = self._llm_match(affiliation)
            if llm_result:
                result.update(llm_result)
                result["match_method"] = "llm"
                self._audit.append(result)
                return result

        # ---- Step 4: Unresolved ----
        self._audit.append(result)
        return result

    def _llm_match(self, affiliation: str) -> dict | None:
        """
        Use an LLM to adjudicate an affiliation string against the site list.

        Returns dict with canonical_site, confidence, rationale on success,
        or None on failure / no confident match.

        Provider-specific requirements:
            openai: ``openai`` package and OPENAI_API_KEY env var
            ollama: local Ollama server and pulled local model
        """
        if self._llm_disabled:
            return None

        if self.llm_provider == "ollama":
            return self._llm_match_ollama(affiliation)
        return self._llm_match_openai(affiliation)

    def _llm_match_openai(self, affiliation: str) -> dict | None:
        """OpenAI-backed LLM affiliation adjudication."""

        try:
            import openai  # optional dependency
        except ImportError:
            logger.warning("openai package not installed; LLM step skipped.")
            self._llm_disabled = True
            return None

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set; LLM step skipped.")
            self._llm_disabled = True
            return None

        site_list = "\n".join(f"- {s}" for s in self._canonical_sites)
        prompt = (
            "You are a research assistant helping map institution affiliations "
            "to a list of known US neurosurgical trial sites.\n\n"
            f"Affiliation text: \"{affiliation}\"\n\n"
            f"Candidate sites:\n{site_list}\n\n"
            "Instructions:\n"
            "1. Return the BEST matching canonical site from the list above, "
            "or 'UNRESOLVED' if no good match exists.\n"
            "2. Provide a confidence between 0.0 and 1.0.\n"
            "3. Provide a short rationale (≤20 words).\n"
            "4. NEVER invent a site not in the list.\n"
            "5. Respond ONLY with valid JSON in this format:\n"
            '{"best_match": "...", "confidence": 0.0, "rationale": "..."}'
        )

        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=120,
            )
            content = (response.choices[0].message.content or "").strip()
            data = self._parse_llm_json(content)
            if data:
                return data
        except Exception as exc:
            logger.warning("LLM match failed: %s", exc)

        return None

    def _llm_match_ollama(self, affiliation: str) -> dict | None:
        """Local Ollama-backed LLM affiliation adjudication."""
        site_list = "\n".join(f"- {s}" for s in self._canonical_sites)
        prompt = (
            "You are a research assistant helping map institution affiliations "
            "to a list of known US neurosurgical trial sites.\n\n"
            f"Affiliation text: \"{affiliation}\"\n\n"
            f"Candidate sites:\n{site_list}\n\n"
            "Instructions:\n"
            "1. Return the BEST matching canonical site from the list above, "
            "or 'UNRESOLVED' if no good match exists.\n"
            "2. Provide a confidence between 0.0 and 1.0.\n"
            "3. Provide a short rationale (≤20 words).\n"
            "4. NEVER invent a site not in the list.\n"
            "5. Respond ONLY with valid JSON in this format:\n"
            '{"best_match": "...", "confidence": 0.0, "rationale": "..."}'
        )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
                timeout=90,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                logger.warning("Ollama error; LLM step skipped: %s", payload["error"])
                self._llm_disabled = True
                return None

            content = (payload.get("response") or "").strip()
            data = self._parse_llm_json(content)
            if data:
                return data
            logger.warning("Ollama returned unparsable/invalid JSON; LLM match skipped.")
        except requests.RequestException as exc:
            logger.warning("Ollama unavailable; LLM step skipped: %s", exc)
            self._llm_disabled = True
        except Exception as exc:
            logger.warning("Ollama LLM match failed: %s", exc)

        return None

    def _parse_llm_json(self, content: str) -> dict | None:
        """Parse and validate model JSON output for site mapping."""
        if not content:
            return None

        import json

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Some local models wrap JSON in prose/code fences; extract first JSON object.
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        best = str(data.get("best_match", "")).strip()
        if not best or best.upper() == "UNRESOLVED":
            return None

        # Accept exact canonical names and robustly recover near-matches.
        if best in self._canonical_sites:
            canonical = best
        else:
            best_norm = normalise_text(best)
            exact_norm = next(
                (s for s in self._canonical_sites if normalise_text(s) == best_norm),
                "",
            )
            if exact_norm:
                canonical = exact_norm
            else:
                fuzzy = process.extractOne(
                    best_norm,
                    [normalise_text(s) for s in self._canonical_sites],
                    scorer=fuzz.token_sort_ratio,
                )
                if not fuzzy or fuzzy[1] < 90:
                    return None
                canonical = self._canonical_sites[fuzzy[2]]

        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        confidence = max(0.0, min(1.0, confidence))

        if canonical:
            return {
                "canonical_site": canonical,
                "confidence": confidence,
                "rationale": data.get("rationale", ""),
            }
        return None

    # ------------------------------------------------------------------
    # Batch matching
    # ------------------------------------------------------------------

    def match_authors(self, authors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match each author's affiliations to canonical sites.

        *authors_df* must have columns: author_id, affiliations
        (affiliations is a semicolon-separated string of
        ``DisplayName|CountryCode`` pairs).

        Returns the DataFrame with added columns:
            canonical_site, site_confidence, match_method
        """
        canonical_sites = []
        confidences = []
        methods = []

        for _, row in authors_df.iterrows():
            author_id = row.get("author_id", "")
            aff_string = row.get("affiliations", "")
            # Try each affiliation and take the best high-confidence match
            best_result: dict[str, Any] = {
                "canonical_site": "",
                "confidence": 0.0,
                "match_method": "unresolved",
            }
            for aff_entry in aff_string.split("; "):
                # Format: "Institution Name|CountryCode"
                parts = aff_entry.split("|")
                aff_name = parts[0].strip()
                country = parts[1].strip().upper() if len(parts) > 1 else ""
                # Country filter removed: allow all countries so that UK/AUS/EU
                # affiliations can be matched to international candidate sites.
                result = self.match_affiliation(aff_name, author_id=author_id)
                if result["confidence"] > best_result["confidence"]:
                    best_result = result
                if best_result["confidence"] >= 0.95:
                    break  # good enough

            canonical_sites.append(best_result["canonical_site"])
            confidences.append(best_result["confidence"])
            methods.append(best_result["match_method"])

        authors_df = authors_df.copy()
        authors_df["canonical_site"] = canonical_sites
        authors_df["site_confidence"] = confidences
        authors_df["match_method"] = methods
        return authors_df

    # ------------------------------------------------------------------
    # Audit / output
    # ------------------------------------------------------------------

    def save_audit(self, path: str | None = None) -> Path:
        """Save the full audit log to CSV."""
        dest = Path(path) if path else output_dir() / "match_audit.csv"
        pd.DataFrame(self._audit).to_csv(dest, index=False)
        logger.info("Saved %d audit rows to %s", len(self._audit), dest)
        return dest

    def save_unresolved(self, authors_df: pd.DataFrame, path: str | None = None) -> Path:
        """Save rows with unresolved site matches to CSV."""
        dest = Path(path) if path else output_dir() / "unresolved_affiliations.csv"
        unresolved = authors_df[authors_df["match_method"] == "unresolved"].copy()
        unresolved.to_csv(dest, index=False)
        logger.info("Saved %d unresolved rows to %s", len(unresolved), dest)
        return dest

    def aliases_for_sites(self, site_names: list[str]) -> dict[str, list[str]]:
        """
        Return a dict mapping each canonical site name to its list of known aliases.

        Used by the targeted enrichment step to build alias-aware PubMed queries.
        Sites with no aliases in site_aliases.csv fall back to the canonical name.

        Parameters
        ----------
        site_names : canonical site names (from site_longlist.csv)

        Returns
        -------
        dict  {canonical_site: [alias1, alias2, ...]}
        """
        return {
            s: (self._canonical_aliases.get(s) or [s])
            for s in site_names
        }
