# Neurosurgical Site KOL Ranking

Ranks candidate neurosurgical sites by literature-based **KOL strength** using PubMed and OpenAlex. Each site gets a single `composite_kol_score` (0–100) derived from its affiliated authors' publication record.

---

## Quick start

```bash
git clone https://github.com/roschkoenig/neurosurgical-site-ranking.git
cd neurosurgical-site-ranking
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt ipykernel
# Open notebooks/pipeline_colab.ipynb and select the .venv kernel
```

No API keys required. All API responses are cached in `cache/` so re-runs are fast.

---

## How the composite score is calculated

The pipeline works in five stages:

**1. Papers** — PubMed is searched for GBM surgical papers. Each paper is classified:

| Label | Weight | Meaning |
|---|---|---|
| `core_surgical` | 3× | GBM surgery / operative technique |
| `surgery_adjacent` | 1× | Adjacent treatment (radiosurgery, chemo, trials) |
| `non_core` | 0× | Excluded (biology, preclinical, paediatric) |

**2. Authors** — A co-authorship network is built from `core_surgical` and `surgery_adjacent` papers. Each author receives a `author_kol_score` (0–100):

$$\text{author\_kol\_score} = 0.35 \cdot \text{citations} + 0.25 \cdot \text{core\_surgical\_papers} + 0.15 \cdot \text{recency} + 0.15 \cdot \text{PageRank} + 0.10 \cdot \text{edge\_authorship}$$

All five inputs are min-max normalised across the full author pool before combining.

**3. Site matching** — Each author's affiliation string is matched to a canonical site name from `data/site_aliases.csv` using exact alias lookup, then fuzzy matching, then optionally an LLM. Authors with confidence < 0.65 are excluded from scoring.

**4. Site components** — For each candidate site, four components are computed from its matched authors:

| Component | What it measures |
|---|---|
| `site_neurosurgical_kol_strength` | Mean adjusted KOL score of top-5 authors, weighted by surgical specificity (`core_surgical` vs `surgery_adjacent` mix) |
| `site_network_engagement` | Mean network centrality of top-5 authors (PageRank 50 %, degree 30 %, edge-authorship 20 %) |
| `site_depth` | log(1 + number of matched authors) — rewards breadth of surgical expertise |
| `site_confidence` | Mean affiliation-match confidence across all site authors |

**5. Composite score** — The four components are min-max normalised across all candidate sites and combined:

$$\boxed{\text{composite\_kol\_score} = (0.45 \cdot \text{neuro\_strength} + 0.25 \cdot \text{network} + 0.20 \cdot \text{depth} + 0.10 \cdot \text{confidence}) \times 100}$$

Sites with no matched authors are not represented as zeros. They receive a descriptive `evidence_status` (`alias_issue_likely` or `no_candidate_authors_matched`) and can be recovered via the automated targeted-enrichment step.

---

## Key files

| Path | Purpose |
|---|---|
| `data/site_longlist.csv` | Candidate sites with country and region |
| `data/site_aliases.csv` | Canonical site names and all known affiliation aliases |
| `notebooks/pipeline_colab.ipynb` | End-to-end pipeline (runs locally or in Colab) |
| `outputs/site_scores.csv` | One row per candidate site with composite score and components |
| `outputs/kol_candidates.csv` | Top-30-centile authors per candidate site |
| `outputs/missing_candidate_sites.csv` | Sites not matched after the first pass |
| `outputs/possible_duplicate_authors.csv` | Flagged near-duplicate author records |
| `outputs/match_audit.csv` | Full affiliation-matching audit trail |

---

## Adding or fixing sites

- **New site**: add a row to `data/site_longlist.csv` and at least one alias to `data/site_aliases.csv`.
- **Unmatched site**: check `outputs/unresolved_affiliations.csv` to see what affiliation strings are appearing, then add the missing alias to `data/site_aliases.csv`.
- **Ambiguous alias**: if a generic alias (e.g. a hospital group name) maps to multiple campuses, make each alias specific enough to distinguish them.
