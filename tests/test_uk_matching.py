"""
Tests for UK affiliation matching (issue #10).

Verifies that affiliations containing country tags (|GB) and semicolon-
separated institution lists are correctly matched to canonical sites.
"""

import sys
from pathlib import Path

import pytest

# Ensure the repo root is on the path when running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.site_matcher import SiteMatcher

ALIASES_CSV = Path(__file__).resolve().parent.parent / "data" / "site_aliases.csv"


@pytest.fixture(scope="module")
def matcher():
    return SiteMatcher(ALIASES_CSV)


def test_kings_college_london_with_country_tag(matcher):
    """'King's College London|GB' should resolve to King's College Hospital, London."""
    result = matcher.match_affiliation("King's College London|GB")
    assert result["canonical_site"] == "King's College Hospital, London"
    assert result["match_method"] == "exact_alias"


def test_nhnn_ucl_semicolon_separated(matcher):
    """Multi-institution string with semicolons should match via NHNN alias."""
    affiliation = "National Hospital for Neurology and Neurosurgery|GB; University College London|GB"
    result = matcher.match_affiliation(affiliation)
    assert result["canonical_site"] == "University College London Hospital (including NHNN), London"
    assert result["match_method"] == "exact_alias"


def test_oxford_semicolon_separated(matcher):
    """'University of Oxford|GB; Oxford University Hospitals NHS Trust|GB' → John Radcliffe."""
    affiliation = "University of Oxford|GB; Oxford University Hospitals NHS Trust|GB"
    result = matcher.match_affiliation(affiliation)
    assert result["canonical_site"] == "John Radcliffe Hospital, Oxford"
    assert result["match_method"] == "exact_alias"
