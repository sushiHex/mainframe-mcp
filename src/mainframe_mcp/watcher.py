"""File change detection for incremental indexing.

`diff_files` compares an explicit file list against the manifest using the
stored SHA-256 hashes, so only files that actually changed are re-indexed and
files deleted from disk can be pruned.
"""

import logging
from pathlib import Path

from mainframe_mcp.dedup import content_hash
from mainframe_mcp.manifest import Manifest

logger = logging.getLogger(__name__)


def _norm(p) -> str:
    """Separator-normalized path key (Windows clients may store either slash)."""
    return str(p).replace("\\", "/")


def diff_files(files: list[str], manifest: Manifest) -> dict:
    """Diff an explicit file list against the manifest.

    Returns {"new", "modified", "deleted", "unchanged"}. Comparison is on
    separator-normalized keys (a backslash disk path must not mismatch a
    forward-slash manifest key). Deletions are decided by FILE EXISTENCE of the
    manifest's own keys — NOT by absence from `files` — so a file that simply
    isn't in this scan subset (e.g. a manually-ingested doc) is never pruned.
    """
    manifest_files = manifest.data.get("files", {})
    manifest_norm = {_norm(k): k for k in manifest_files}
    result = {"new": [], "modified": [], "deleted": [], "unchanged": []}

    for path_str in files:
        key = _norm(path_str)
        try:
            current_hash = content_hash(Path(path_str).read_text(encoding="utf-8"))
        except OSError:
            continue  # vanished mid-scan; the deletion pass below handles it
        if key not in manifest_norm:
            result["new"].append(path_str)
        elif manifest.needs_reindex(manifest_norm[key], current_hash):
            result["modified"].append(path_str)
        else:
            result["unchanged"].append(path_str)

    for real_key in manifest_files:  # original (un-normalized) manifest keys
        if not Path(real_key).exists():
            result["deleted"].append(real_key)

    logger.info(
        f"Diff: {len(result['new'])} new, {len(result['modified'])} modified, "
        f"{len(result['deleted'])} deleted, {len(result['unchanged'])} unchanged"
    )
    return result
