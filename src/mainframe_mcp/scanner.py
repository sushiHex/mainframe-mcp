"""Project scanner — discovers and ingests knowledge from all projects.

Recursively finds research/, docs/, and root agent-context files (CLAUDE.md,
AGENTS.md, HERMES.md, .hermes.md) across all repos. No hardcoded paths —
discovers by directory name convention.
"""

import logging
import os
from pathlib import Path

from mainframe_mcp.store import ROOT_CONTEXT_FILES

logger = logging.getLogger(__name__)

DEFAULT_REPOS_DIR = Path.home() / "repos"

# Directory names that contain ingestible knowledge
KNOWLEDGE_DIRS = {"research", "docs", "machines", "playbooks"}

# Skip these directories entirely
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "extracted", "raw",
             ".lancedb", ".models", ".claude", "eval", "build", "dist"}

# Skip these filenames (boilerplate, not knowledge)
SKIP_FILES = {"README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE.md",
              "TODO.md", "NEWS.md", "SCORECARD.md"}


def scan_projects(
    repos_dir: Path | None = None,
    config: dict | None = None,
) -> list[dict]:
    """Scan all projects for knowledge files.

    Discovers:
      - Any CLAUDE.md at project root (one level deep)
      - Any .md file inside a directory named research/, docs/, machines/, playbooks/

    Returns list of:
        {"path": str, "project": str, "source_type": str}
    """
    if repos_dir is None:
        if config and "paths" in config:
            repos_dir = Path(config["paths"]["repos_dir"])
        else:
            repos_dir = DEFAULT_REPOS_DIR

    if not repos_dir.exists():
        logger.warning(f"Repos directory not found: {repos_dir}")
        return []

    # Same classifier as ingest_file so the source_type field can't disagree.
    from mainframe_mcp.store import classify_source_type

    files = []

    for project_dir in sorted(repos_dir.iterdir()):
        if not project_dir.is_dir() or project_dir.name.startswith("."):
            continue

        project = project_dir.name

        # Root agent-context files (CLAUDE.md / AGENTS.md / HERMES.md / .hermes.md)
        root_context = set()
        for name in ROOT_CONTEXT_FILES:
            ctx = project_dir / name
            if ctx.exists():
                root_context.add(ctx)
                files.append({
                    "path": str(ctx),
                    "project": project,
                    "source_type": classify_source_type(str(ctx)),
                })

        # Walk the tree, pruning SKIP_DIRS in place so we never descend into
        # node_modules/.venv/build/etc. (rglob would stat all of them first).
        for dirpath, dirnames, filenames in os.walk(project_dir):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for fn in sorted(filenames):
                if not fn.endswith(".md") or fn in SKIP_FILES:
                    continue
                md_file = Path(dirpath) / fn
                if md_file in root_context:
                    continue
                # Require a knowledge dir (research/docs/...) anywhere in the path.
                rel_parts = md_file.relative_to(project_dir).parts
                if not any(part in KNOWLEDGE_DIRS for part in rel_parts):
                    continue
                files.append({
                    "path": str(md_file),
                    "project": project,
                    "source_type": classify_source_type(str(md_file)),
                })

    projects = set(f["project"] for f in files)
    logger.info(f"Scanned {repos_dir}: {len(files)} files across {len(projects)} projects")
    return files
