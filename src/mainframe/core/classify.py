"""source_type from (lane, canonical_path). The lane is EXPLICIT — decided by
which scanner produced the file — so no lane is ever inferred from a path
substring. source_type distinguishes knowledge sub-kinds; it also carries a
capture's published kind."""

from mainframe.memory.frontmatter import split as split_frontmatter

KNOWLEDGE_LANE = "knowledge"
CAPTURE_LANE = "capture"

# Root-level agent-context files (project conventions/evidence). SOUL.md and
# other identity files are deliberately NOT here.
ROOT_CONTEXT_FILES = frozenset({"CLAUDE.md", "AGENTS.md", "HERMES.md", ".hermes.md"})
KNOWLEDGE_DIRS = frozenset({"research", "docs", "machines", "playbooks"})

_ROOT_LOWER = frozenset(n.lower() for n in ROOT_CONTEXT_FILES)
_DOCS_DIRS = ("docs", "machines", "playbooks")


# Capture kinds an author declared fit for default search (`write_capture`'s
# `type:`). Positive on purpose: a new kind stays opt-in until listed here.
PUBLISHED_KINDS = frozenset({"decision", "note"})

# The frontmatter must close within this many characters to declare a kind;
# `write_capture`'s block is well under 1 KB.
KIND_HEAD = 4096


def declared_kind(text: str) -> str:
    """A capture's published kind from its frontmatter, else "session". Read
    from the file as written: the scrubber can span the closing fence."""
    head = text.replace("\r\n", "\n")[:KIND_HEAD]
    kind = split_frontmatter(head)[0].get("type", "")
    return kind if kind in PUBLISHED_KINDS else "session"


def declared_kind_of(path) -> str:
    """`declared_kind` of a file, reading only its head: KIND_HEAD characters
    with newlines translated, as `Path.read_text` yields on the index path."""
    try:
        with open(path, encoding="utf-8", errors="replace", newline=None) as f:
            return declared_kind(f.read(KIND_HEAD))
    except OSError:
        return "session"


def classify_source_type(lane: str, canonical_path: str, kind: str = "session") -> str:
    if lane == CAPTURE_LANE:
        return kind if kind in PUBLISHED_KINDS else "session"
    p = canonical_path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    if name.lower() in _ROOT_LOWER:
        return "project"
    if "/research/" in p:
        return "research"
    if any(f"/{d}/" in p for d in _DOCS_DIRS):
        return "docs"
    return "library"
