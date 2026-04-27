# AGENTS.md

## Project purpose

This repository ranks neurosurgical/neuro-oncology trial sites by candidate-site-constrained KOL strength. The goal is to support site selection for neuro-oncology/neurosurgical device and trial readiness, not to produce a generic global institution ranking.

## Core principles

- `data/site_longlist.csv` is the authoritative candidate-site universe.
- Final site scores must only be produced for candidate sites in `site_longlist.csv`.
- Non-candidate institutions may be retained for audit/network context, but must not appear in the main `outputs/site_scores.csv`.
- Missing candidate sites should not be treated as true zero-scoring sites unless they have been explicitly searched and matched with confidence.
- UK/AUS sites are particularly vulnerable to aliasing and hospital/university mapping failures; surface these failure modes clearly.
- Prefer transparent, inspectable notebook workflows over hidden automation.

## Main outputs

Primary outputs:

- `outputs/site_scores.csv`
- `outputs/kol_candidates.csv`
- `outputs/missing_candidate_sites.csv`

Secondary audit outputs:

- `outputs/candidate_authors.csv`
- `outputs/noncandidate_matched_authors.csv`
- `outputs/unresolved_affiliations.csv`
- `outputs/possible_duplicate_authors.csv`

## Scoring intent

The final score should be a single composite KOL score per candidate site.

The score should weight strongly toward:

- neurosurgical contribution;
- core GBM/neuro-oncology surgical papers;
- deep author-network engagement;
- cross-institutional co-authorship / bridging behaviour.

Suggested composite:

```text
Composite KOL score =
  0.45 * site_neurosurgical_kol_strength
+ 0.25 * site_network_engagement
+ 0.20 * site_depth
+ 0.10 * site_confidence