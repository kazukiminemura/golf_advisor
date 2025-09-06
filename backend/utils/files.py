from __future__ import annotations

from pathlib import Path


def safe_filename(name: str) -> str:
    """Return a filesystem-safe filename (basename, no path separators)."""
    base = Path(name).name
    return "".join(c for c in base if c.isalnum() or c in {"_", "-", ".", " "}).strip()


