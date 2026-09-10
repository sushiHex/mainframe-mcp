"""Canonical path identity.

Every path that is stored, compared, hashed, or placed in a SQL predicate
goes through `canonical` first. `doc_id` is 128 bits of the canonical path
so it is safe inside filter strings; `doc_path` never is.
"""

import hashlib
import os


def canonical(path) -> str:
    """Absolute, symlink-resolved, case-folded (Windows), forward-slash path."""
    p = os.path.realpath(str(path))
    p = os.path.normcase(p)
    return p.replace("\\", "/")


def doc_id(canonical_path: str) -> str:
    """32-hex (128-bit) identity of a canonical path."""
    return hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()[:32]


def chunk_key(doc_id_: str, chunk_index: int) -> str:
    """Deterministic merge key: one row per (document, chunk position)."""
    return f"{doc_id_}#{chunk_index}"
