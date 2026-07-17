"""Deduplication at ingest time.

Two layers:
1. Exact dedup: SHA-256 content hash
2. Near-dupe: cosine similarity > 0.90 against existing chunks
"""

import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)

NEAR_DUPE_THRESHOLD = 0.90


def content_hash(text: str) -> str:
    """SHA-256 hash of normalized text."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def check_near_dupes(
    new_embedding: list[float],
    existing_embeddings: list[list[float]],
    existing_ids: list[str],
    threshold: float = NEAR_DUPE_THRESHOLD,
) -> list[tuple[str, float]]:
    """Check if a new embedding is a near-duplicate of existing ones.

    Returns list of (chunk_id, similarity) for chunks above threshold.
    """
    if len(existing_embeddings) == 0:
        return []

    new_vec = np.asarray(new_embedding)
    # asarray avoids a copy when the caller already passes an ndarray (so a
    # batch caller can build the donor matrix once instead of per-call).
    existing_mat = np.asarray(existing_embeddings)

    # Cosine similarity (vectors should already be normalized)
    similarities = existing_mat @ new_vec

    dupes = []
    for i, sim in enumerate(similarities):
        if sim >= threshold:
            dupes.append((existing_ids[i], float(sim)))

    if dupes:
        logger.info(f"Found {len(dupes)} near-duplicates above {threshold}")

    return sorted(dupes, key=lambda x: x[1], reverse=True)


class DonorCache:
    """Batch-scoped exact-dedup donor index.

    ingest_file's per-file donor lookup was a full projected table scan; an
    88-file sync therefore scanned 37K rows 88 times (measured 2026-07-16).
    A batch builds this once from Store.get_hash_doc_map and maintains it
    incrementally: remove_doc BEFORE the file's chunks are deleted from the
    table (so a mid-ingest crash leaves the cache conservative, not stale),
    add_doc after a successful insert (so intra-batch dedup keeps working).
    Session-lane files are never donors (is_donor=False), mirroring
    get_existing_hashes(exclude_source_type="session").
    """

    def __init__(self, hash_docs: dict[str, set[str]]):
        # Takes OWNERSHIP of hash_docs (the caller builds it fresh from a scan
        # and never reads it again) — no defensive copy of 37K entries.
        self.hash_docs = hash_docs
        self.doc_hashes: dict[str, set[str]] = {}
        for h, docs in self.hash_docs.items():
            for d in docs:
                self.doc_hashes.setdefault(d, set()).add(h)

    def is_donor(self, content_hash: str, doc_path: str) -> bool:
        """True when the hash exists in at least one OTHER document — the same
        semantics as `content_hash in get_existing_hashes(exclude_doc_path=
        doc_path)`: a hash shared by this doc AND another still counts (the
        other copy survives). Point lookup, O(1) — the consumer checks one
        chunk at a time, so materializing the full donor set per file
        (O(all-hashes), the pattern this cache exists to kill) is unnecessary."""
        docs = self.hash_docs.get(content_hash)
        return bool(docs) and bool(docs - {doc_path})

    def remove_doc(self, doc_path: str) -> None:
        for h in self.doc_hashes.pop(doc_path, set()):
            docs = self.hash_docs.get(h)
            if docs is not None:
                docs.discard(doc_path)
                if not docs:
                    del self.hash_docs[h]

    def add_doc(self, doc_path: str, hashes: set[str], is_donor: bool) -> None:
        if not is_donor:
            return
        for h in hashes:
            self.hash_docs.setdefault(h, set()).add(doc_path)
        self.doc_hashes.setdefault(doc_path, set()).update(hashes)
