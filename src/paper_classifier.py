"""
paper_classifier.py – Rule-based surgical relevance classifier.

Papers are labelled as one of:
    core_surgical       – directly about GBM surgery / operative technique
    surgery_adjacent    – GBM treatment adjacent to surgery (chemo, RT, trials)
    non_core            – disease biology, preclinical, paediatric-only, etc.

Classification is intentionally conservative and transparent so it can be
swapped for a learned model later.  The rules operate on:
    - title text
    - abstract text (if available)
    - journal name
    - MeSH / concept keywords

Usage example
-------------
>>> from src.paper_classifier import PaperClassifier
>>> clf = PaperClassifier()
>>> label = clf.classify({"title": "Extent of resection in glioblastoma", ...})
>>> print(label)   # 'core_surgical'
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword rule sets
# ---------------------------------------------------------------------------

# Terms that strongly indicate core surgical content
CORE_SURGICAL_TITLE_TERMS: list[str] = [
    r"\bextent of resection\b",
    r"\bgross[\s-]total resection\b",
    r"\bsubtotal resection\b",
    r"\bsupratotal resection\b",
    r"\bcraniotomy\b",
    r"\bawake craniotomy\b",
    r"\bawake mapping\b",
    r"\bsurgical resection\b",
    r"\bintraoperative mri\b",
    r"\bimri\b",
    r"\b5-ala\b",
    r"\bfluorescence.guided\b",
    r"\b5.aminolevulinic\b",
    r"\bsurgical outcome\b",
    r"\bneurosurgical\b",
    r"\bneurosurgeon\b",
    r"\bresection extent\b",
    r"\btumor resection\b",
    r"\btumour resection\b",
    r"\brecurrent glioblastoma.{0,30}(surgery|resection|re.?resection)\b",
    r"\b(surgery|resection|re.?resection).{0,30}recurrent glioblastoma\b",
    r"\bintraoperative ultrasound\b",
    r"\bneuro.?navigation\b",
    r"\bfunctional mapping\b",
    r"\bcortical mapping\b",
    r"\bsubcortical mapping\b",
    r"\beloquent cortex\b",
    r"\bstereotactic biopsy\b",
]

# Terms in title that are surgery-adjacent (treatment / multimodal)
SURGERY_ADJACENT_TITLE_TERMS: list[str] = [
    r"\bglioblastoma\b",
    r"\bhigh.grade glioma\b",
    r"\bgbm\b",
    r"\btemozolomide\b",
    r"\bbevacizumab\b",
    r"\bstereotactic radiosurgery\b",
    r"\bchemoradiation\b",
    r"\bimmunotherapy\b",
    r"\bclinical trial\b",
    r"\brandomized.{0,20}trial\b",
    r"\bstudy\b",
    r"\bsurvival\b",
    r"\bprognosis\b",
    r"\boutcomes\b",
]

# Journals strongly associated with surgical content
SURGICAL_JOURNALS: set[str] = {
    "journal of neurosurgery",
    "neurosurgery",
    "world neurosurgery",
    "acta neurochirurgica",
    "operative neurosurgery",
    "journal of neuro-oncology",
    "neuro-oncology",
    "neurosurgical focus",
    "childs nervous system",
    "journal of neurosurgery spine",
    "clinical neurology and neurosurgery",
    "surgical neurology international",
}

# Terms that down-grade to non_core even if other terms match
NON_CORE_TERMS: list[str] = [
    r"\bin vitro\b",
    r"\bcell line\b",
    r"\bmouse model\b",
    r"\bmouse brain\b",
    r"\bmurine\b",
    r"\bzebrafish\b",
    r"\bnanoparticle\b",
    r"\bgene therapy\b",
    r"\bmedulloblastoma\b",
    r"\bependymoma\b",
    r"\bpaediatric\b",
    r"\bpediatric\b",
    r"\bdiffuse intrinsic pontine\b",
    r"\bdipg\b",
    r"\bcase report\b",  # single case reports are lower value for KOL ranking
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class PaperClassifier:
    """
    Rule-based three-class classifier for surgical GBM relevance.

    Assign weights used downstream:
        core_surgical    -> weight 3.0
        surgery_adjacent -> weight 1.0
        non_core         -> weight 0.0
    """

    CLASS_WEIGHTS: dict[str, float] = {
        "core_surgical": 3.0,
        "surgery_adjacent": 1.0,
        "non_core": 0.0,
    }

    def __init__(self) -> None:
        # Pre-compile regex patterns for speed
        self._core_re = [
            re.compile(p, re.IGNORECASE) for p in CORE_SURGICAL_TITLE_TERMS
        ]
        self._adjacent_re = [
            re.compile(p, re.IGNORECASE) for p in SURGERY_ADJACENT_TITLE_TERMS
        ]
        self._non_core_re = [
            re.compile(p, re.IGNORECASE) for p in NON_CORE_TERMS
        ]

    def classify(self, paper: dict[str, Any]) -> str:
        """
        Return a class label for *paper*.

        Parameters
        ----------
        paper:  dict with at least one of: title, abstract, journal, concepts
        """
        title = (paper.get("title") or "").lower()
        abstract = (paper.get("abstract") or "").lower()
        journal = (paper.get("journal") or "").lower()
        combined = f"{title} {abstract}"

        # Check non-core exclusions first
        for pattern in self._non_core_re:
            if pattern.search(combined):
                # Only downgrade if the non_core signal is in the title
                if pattern.search(title):
                    return "non_core"

        # Check core surgical signals in title (higher confidence)
        for pattern in self._core_re:
            if pattern.search(title):
                return "core_surgical"

        # Check core surgical signals in abstract
        core_hits = sum(1 for p in self._core_re if p.search(combined))
        if core_hits >= 2:
            return "core_surgical"
        if core_hits >= 1:
            # A surgical journal + 1 core term = core_surgical
            if self._is_surgical_journal(journal):
                return "core_surgical"
            return "surgery_adjacent"

        # Adjacent terms in title
        for pattern in self._adjacent_re:
            if pattern.search(title):
                return "surgery_adjacent"

        return "non_core"

    def _is_surgical_journal(self, journal: str) -> bool:
        for sj in SURGICAL_JOURNALS:
            if sj in journal:
                return True
        return False

    def classify_all(self, papers: list[dict]) -> list[dict]:
        """
        Classify a list of papers in-place, adding a ``label`` and
        ``paper_weight`` field to each.
        """
        for paper in papers:
            label = self.classify(paper)
            paper["label"] = label
            paper["paper_weight"] = self.CLASS_WEIGHTS[label]
        return papers

    @staticmethod
    def weight_for(label: str) -> float:
        return PaperClassifier.CLASS_WEIGHTS.get(label, 0.0)
