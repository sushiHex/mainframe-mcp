"""source_type from (lane, canonical_path). The lane is EXPLICIT — decided by
which scanner produced the file — so no lane is ever inferred from a path
substring. source_type only distinguishes knowledge sub-kinds."""

KNOWLEDGE_LANE = "knowledge"
CAPTURE_LANE = "capture"

# Root-level agent-context files (project conventions/evidence). SOUL.md and
# other identity files are deliberately NOT here.
ROOT_CONTEXT_FILES = frozenset({"CLAUDE.md", "AGENTS.md", "HERMES.md", ".hermes.md"})
KNOWLEDGE_DIRS = frozenset({"research", "docs", "machines", "playbooks"})

_ROOT_LOWER = frozenset(n.lower() for n in ROOT_CONTEXT_FILES)
_DOCS_DIRS = ("docs", "machines", "playbooks")


def classify_source_type(lane: str, canonical_path: str) -> str:
    if lane == CAPTURE_LANE:
        return "session"
    p = canonical_path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    if name.lower() in _ROOT_LOWER:
        return "project"
    if "/research/" in p:
        return "research"
    if any(f"/{d}/" in p for d in _DOCS_DIRS):
        return "docs"
    return "library"
