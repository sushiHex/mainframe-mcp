"""Shared test helpers for the v2 package's GPU-free suite."""

from pathlib import Path


def write_md(path: Path, text: str) -> Path:
    """Write a markdown file with parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
