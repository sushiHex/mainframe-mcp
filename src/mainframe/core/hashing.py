"""Hashes: `file_hash` (raw bytes — change detection) and `content_hash`
(whitespace/case-normalized chunk text — diagnostic only; no dedup decision)."""

import hashlib


def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
