"""Turn one on-disk document into index rows.

Pure with respect to the store: the caller batches DocRows into
`Store.upsert_batch`. A file that fails ANY stage returns DocFailure and the
caller leaves it out of the batch (its previous rows survive) — with ONE
exception, stated at the embed: a device-level failure is not a fact about this
document, so it propagates instead of being reported as one."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mainframe.core.chunker import chunk_markdown
from mainframe.core.classify import CAPTURE_LANE, classify_source_type
from mainframe.core.device import is_device_failure
from mainframe.core.hashing import content_hash, file_hash
from mainframe.core.paths import chunk_key, doc_id
from mainframe.core.secrets import scrub_secrets
from mainframe.core.store import DocRows
from mainframe.memory.frontmatter import split as split_frontmatter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaneFile:
    path: str      # canonical
    lane: str      # "knowledge" | "capture"
    project: str


@dataclass(frozen=True)
class DocFailure:
    doc_path: str
    error: str


class _Unchanged:
    def __repr__(self):
        return "UNCHANGED"


UNCHANGED = _Unchanged()


def prepare_document(lf: LaneFile, chunk_cfg: dict, embedder, known_hash: str | None = None,
                     contextualizer=None, now: str | None = None):
    try:
        text = Path(lf.path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return DocFailure(lf.path, f"read: {e}")
    fh = file_hash(text)
    if known_hash is not None and known_hash == fh:
        return UNCHANGED
    did = doc_id(lf.path)
    if lf.lane == CAPTURE_LANE:
        # Choke-point scrub: every capture-lane chunk is scrubbed before it can
        # be embedded or searched, regardless of which writer produced the file.
        text = scrub_secrets(text)
    # Frontmatter (`---\nkey: value\n...\n---`) is metadata, not content, but
    # stripping it before chunking is a retrieval-affecting change that must be
    # measured against v1 eval parity before it ships — opt in via
    # chunker.strip_frontmatter (default False keeps this byte-identical to
    # chunking `text` outright). When on, `split` is a no-op ({}, text
    # unchanged) for files with no fence, and chunk offsets are shifted back to
    # absolute positions in `text` so line-span citation (search.py's
    # file_hash + slice re-check) still lines up against the on-disk file.
    if chunk_cfg.get("strip_frontmatter"):
        _, body = split_frontmatter(text)
    else:
        body = text
    body_offset = len(text) - len(body)
    try:
        chunks = chunk_markdown(body, max_tokens=chunk_cfg.get("chunk_size", 256),
                                overlap_ratio=chunk_cfg.get("overlap_ratio", 0.35))
    except Exception as e:
        return DocFailure(lf.path, f"chunk: {e}")
    if not chunks:
        return DocRows(doc_id=did, doc_path=lf.path, rows=[])
    if body_offset:
        for c in chunks:
            c.char_start += body_offset
            c.char_end += body_offset
    if contextualizer is not None and getattr(contextualizer, "enabled", False):
        try:
            contexts = contextualizer.contextualize(body, [c.text for c in chunks])
        except Exception as e:
            logger.warning(f"contextualize failed for {lf.path} (indexing plain chunks): {e}")
            contexts = []
        for c, cx in zip(chunks, contexts):
            if cx:
                c.text = f"{scrub_secrets(cx)}\n\n{c.text}"
    try:
        embeddings = embedder.embed([c.text for c in chunks])
    except Exception as e:
        # A DocFailure says "THIS document is bad", and the caller acts on that
        # by leaving it out of the batch and carrying on. A dead CUDA context is
        # not that: swallowing it here meant the fault never reached
        # `Models.invoke`, nothing reloaded, and the job SUCCEEDED with every
        # file in the batch marked failed — so the self-healing this branch
        # advertises could not fire on the index path at all. Let the device
        # speak for itself; the registry decides what to do about it.
        if is_device_failure(e):
            raise
        return DocFailure(lf.path, f"embed: {e}")
    try:
        authored_at = datetime.fromtimestamp(Path(lf.path).stat().st_mtime).isoformat()
    except OSError:
        authored_at = ""
    indexed_at = now or datetime.now().isoformat()
    source_type = classify_source_type(lf.lane, lf.path)
    rows = [{
        "chunk_key": chunk_key(did, c.chunk_index), "doc_id": did, "doc_path": lf.path,
        "project": lf.project, "lane": lf.lane, "source_type": source_type,
        "file_hash": fh, "content_hash": content_hash(c.text),
        "text": c.text, "heading": c.heading, "chunk_index": c.chunk_index,
        "char_start": c.char_start, "char_end": c.char_end, "token_count": c.token_count,
        "vector": e, "authored_at": authored_at, "indexed_at": indexed_at,
    } for c, e in zip(chunks, embeddings)]
    return DocRows(doc_id=did, doc_path=lf.path, rows=rows)
