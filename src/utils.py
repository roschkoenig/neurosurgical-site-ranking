"""
utils.py – Shared helpers for caching, text normalisation, and I/O.

All modules in this project import from here so that common logic
(HTTP caching, slug generation, directory setup) lives in one place.
"""

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """Return the repository root (parent of this file's directory)."""
    return Path(__file__).resolve().parent.parent


def cache_dir() -> Path:
    return ensure_dir(project_root() / "cache")


def output_dir() -> Path:
    return ensure_dir(project_root() / "outputs")


# ---------------------------------------------------------------------------
# Local JSON cache (keyed by an arbitrary string)
# ---------------------------------------------------------------------------

def _cache_path(namespace: str, key: str) -> Path:
    """Return the file path for a cache entry."""
    bucket = ensure_dir(cache_dir() / namespace)
    hashed = hashlib.sha256(key.encode()).hexdigest()
    return bucket / f"{hashed}.json"


def cache_get(namespace: str, key: str) -> Any | None:
    """Return the cached value for *key* in *namespace*, or None."""
    path = _cache_path(namespace, key)
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def cache_set(namespace: str, key: str, value: Any) -> None:
    """Store *value* under *key* in *namespace*."""
    path = _cache_path(namespace, key)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """
    Lower-case, strip accents, remove apostrophes, collapse whitespace,
    remove punctuation.  Used for affiliation matching.

    Normalisation steps (in order):
    1. Replace curly apostrophes (\\u2018, \\u2019) with straight apostrophes.
    2. Remove all apostrophes entirely.
    3. Unicode-normalise (NFKD) and strip combining characters (accents).
    4. Lower-case.
    5. Replace any remaining non-alphanumeric characters with a space.
    6. Collapse runs of whitespace.
    """
    if not text:
        return ""
    # 1+2. Normalise apostrophes: curly → straight, then remove entirely
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("'", "")
    # 3. Unicode normalise and strip combining characters (accents)
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    # 4. Lower-case
    lower = ascii_text.lower()
    # 5. Replace remaining punctuation / separators with space
    cleaned = re.sub(r"[^a-z0-9 ]", " ", lower)
    # 6. Collapse runs of whitespace
    return re.sub(r"\s+", " ", cleaned).strip()


def slugify(text: str) -> str:
    """Return a URL-safe slug for *text*."""
    return re.sub(r"[-\s]+", "-", normalise_text(text))


# ---------------------------------------------------------------------------
# JSON / CSV I/O helpers
# ---------------------------------------------------------------------------

def save_json(data: Any, path: str | Path) -> None:
    """Write *data* as pretty JSON to *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load JSON from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
