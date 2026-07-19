"""Provenance tracking for Mainframe files.

Stores per-file metadata: content hash, timestamps, retrieval counts, source info.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class Manifest:
    def __init__(self, mainframe_dir: Path):
        self.path = mainframe_dir / ".manifest.json"
        self.data: dict = {}
        self._load()

    @property
    def _bak(self) -> Path:
        return self.path.with_name(self.path.name + ".bak")

    @property
    def _tmp(self) -> Path:
        return self.path.with_name(self.path.name + ".tmp")

    def _load(self):
        """Load the manifest with fallbacks: main -> .tmp -> .bak -> empty.
        .tmp is always either absent or COMPLETE (it is fully written before any
        rename), so it recovers the in-flight generation if a crash landed
        between _save()'s two os.replace calls (main already rotated away, tmp
        not yet promoted). .bak holds the previous generation. A bad manifest
        must never brick server startup."""
        for label, candidate in (("", self.path), (".tmp", self._tmp), (".bak", self._bak)):
            if candidate.exists():
                try:
                    self.data = json.loads(candidate.read_text(encoding="utf-8"))
                    if label:
                        logger.warning(f"Manifest missing/corrupt — recovered from {label}")
                    return
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Manifest load failed from {candidate.name}: {e}")
        self.data = {"files": {}, "version": 1}

    def _save(self):
        """Atomic write: full content to a temp file, rotate the previous
        manifest to .bak, then os.replace (atomic on NTFS). A crash at any
        point leaves the old file, the complete .tmp, or the complete new one
        (all of which _load can read) — never a truncated manifest."""
        tmp = self._tmp
        tmp.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if self.path.exists():
            try:
                os.replace(self.path, self._bak)
            except OSError:
                pass
        try:
            os.replace(tmp, self.path)
        except OSError:
            # Final promotion failed with main already rotated away — restore
            # the previous generation so the manifest never vanishes, then
            # surface the failure loudly.
            try:
                os.replace(self._bak, self.path)
            except OSError:
                pass
            raise

    def record_ingest(self, file_path: str, content_hash: str, chunk_count: int):
        """Record that a file was ingested."""
        existing = self.data["files"].get(file_path, {})
        now = datetime.now().isoformat()
        self.data["files"][file_path] = {
            "content_hash": content_hash,
            "chunk_count": chunk_count,
            # Preserve the original first-seen timestamp across re-ingests.
            "ingested_at": existing.get("ingested_at", now),
            "updated_at": now,
            "retrieval_count": existing.get("retrieval_count", 0),
        }
        self._save()

    def remove(self, file_path: str):
        """Drop a file's manifest entry so it can be cleanly re-ingested later.

        Without this, deleting a file leaves a ghost entry: re-creating an
        identical file makes needs_reindex() return False, so ingest_file
        returns "unchanged" and the file is silently never re-indexed.
        """
        self.data["files"].pop(file_path, None)
        self._save()

    def remove_many(self, file_paths: list[str]):
        """Drop several manifest entries with a single save (batch pruning)."""
        for fp in file_paths:
            self.data["files"].pop(fp, None)
        self._save()

    def get_hash(self, file_path: str) -> str | None:
        """Get stored content hash for a file."""
        return self.data["files"].get(file_path, {}).get("content_hash")

    def bump_retrieval(self, file_path: str):
        """Increment retrieval counter (in memory only — call save() to persist)."""
        if file_path in self.data["files"]:
            self.data["files"][file_path]["retrieval_count"] = (
                self.data["files"][file_path].get("retrieval_count", 0) + 1
            )

    def save(self):
        """Persist manifest to disk. Call after batched operations."""
        self._save()

    def get_stats(self) -> dict:
        """Return summary statistics."""
        files = self.data["files"]
        return {
            "total_files": len(files),
            "total_chunks": sum(f.get("chunk_count", 0) for f in files.values()),
            "total_retrievals": sum(f.get("retrieval_count", 0) for f in files.values()),
            "top_retrieved": sorted(
                [(k, v.get("retrieval_count", 0)) for k, v in files.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }

    def needs_reindex(self, file_path: str, current_hash: str) -> bool:
        """Check if a file has changed since last ingest."""
        stored_hash = self.get_hash(file_path)
        if stored_hash is None:
            return True  # Never ingested
        return stored_hash != current_hash
