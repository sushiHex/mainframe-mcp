"""The three lanes and receipts.

knowledge: <repos_dir>/<repo>/{research,docs,machines,playbooks}/**/*.md + root
           context files (git-tracked; the scanner mirrors v1's rules exactly
           so the eval corpus is identical — except research/sessions/, the
           v1 raw lane, which remains separate from v2 captures).
capture:   <mainframe_dir>/captures/<repo>/*.md (immutable; quarantine/ and
           overflow/ subdirs are neither indexed nor pending).
note:      promoted <repo>/research/memory/<slug>.md (+ <slug>.sources.json);
           drafts <mainframe_dir>/drafts/<repo>/<slug>.md (+ .sources.json).
Lane membership is decided HERE, by which enumerator produced the file — never
by a path substring.
"""

import json
import logging
import os
import re
from fnmatch import fnmatchcase
from pathlib import Path

from mainframe.core.classify import (CAPTURE_LANE, KNOWLEDGE_DIRS, KNOWLEDGE_LANE,
                                     _ROOT_LOWER)  # _ROOT_LOWER: one shared source of case-folded root names
from mainframe.core.hashing import content_hash
from mainframe.core.indexer import LaneFile
from mainframe.core.paths import canonical
from mainframe.memory.atomic import write_atomic
from mainframe.memory.frontmatter import split

logger = logging.getLogger(__name__)

SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "extracted", "raw",
                       ".lancedb", "index.lancedb", ".models", ".claude", "eval", "build", "dist"})
SKIP_FILES = frozenset({"README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE.md",
                        "TODO.md", "NEWS.md", "SCORECARD.md"})
ADHOC_PROJECT = "_adhoc"
QUARANTINE = "quarantine"
OVERFLOW = "overflow"
SAFE_NAME_RE = re.compile(r"^(?:_adhoc|[A-Za-z0-9][A-Za-z0-9._-]*)$")

_SKIP_DIRS_L = frozenset(d.lower() for d in SKIP_DIRS)
_SKIP_FILES_L = frozenset(f.lower() for f in SKIP_FILES)
_KNOWLEDGE_DIRS_L = frozenset(d.lower() for d in KNOWLEDGE_DIRS)


def safe_name(name: str) -> str:
    """Reject anything that could escape a lane when joined into a path."""
    if not isinstance(name, str) or not SAFE_NAME_RE.match(name):
        raise ValueError(f"unsafe project/slug name: {name!r}")
    return name


def by_canonical_path(files: list) -> list:
    """One LaneFile per canonical path, first spelling wins.

    A link that reaches an in-root file under a second name (a junction from
    `docs/` into `research/`) walks the same document twice. The canonical path
    is a document's identity everywhere else in the system (`core/paths.py`),
    so the enumerators speak it too rather than handing the pipeline the same
    file twice in one batch."""
    out: dict = {}
    for lf in files:
        out.setdefault(lf.path, lf)      # first wins: the enumerators walk in sorted order
    return list(out.values())


def capture_id_of(path) -> str:
    """The stable identity of a capture: its frontmatter `capture_id`, else
    the first 16 hex of the normalized body hash (legacy files). Frontmatter
    always sits in the first few lines, so try a cheap 4 KB head read first —
    `pending()` calls this for every capture file on every check, and only a
    legacy (no capture_id) file needs to pay for reading the whole body."""
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="replace") as f:
        head = f.read(4096)
    head_fields, _ = split(head)
    head_cid = head_fields.get("capture_id", "").strip()
    if head_cid:
        return head_cid
    text = p.read_text(encoding="utf-8", errors="replace")
    fields, body = split(text)
    cid = fields.get("capture_id", "").strip()
    return cid or content_hash(body)[:16]


def load_receipts(path) -> set:
    p = Path(path)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return set()
        return set(data.get("capture_ids", []))
    except (OSError, ValueError, AttributeError):
        return set()


def save_receipts(path, capture_ids, meta: dict) -> None:
    data = {"capture_ids": sorted(capture_ids), **meta}
    write_atomic(Path(path), json.dumps(data, indent=2, ensure_ascii=False) + "\n")


class Lanes:
    def __init__(self, config: dict):
        self.repos_dir = Path(config["paths"]["repos_dir"])
        self.mainframe_dir = Path(config["paths"]["mainframe_dir"])
        self.include_projects = list(config["paths"].get("include_projects", []) or [])
        self.exclude_projects = list(config["paths"].get("exclude_projects", []) or [])
        self.skipped_projects = 0
        # realpath of a not-yet-existing dir is fine (no error, just no symlink to resolve);
        # cached once so resolve() and the walkers don't re-stat on every call.
        self._repos_real = os.path.realpath(self.repos_dir)
        self._mainframe_real = os.path.realpath(self.mainframe_dir)
        self._captures_real = os.path.realpath(self.mainframe_dir / "captures")

    @staticmethod
    def _within(root, candidate: str) -> bool:
        """Is `candidate` `root` itself or inside it? Both are canonicalized
        here, so either may be a raw realpath or an already-canonical string.

        This is the containment `resolve()` gets for free by realpath-ing
        first and then asking `relative_to`. The enumerators walk the tree
        instead, so they must ask it explicitly — see the callers."""
        r = os.path.normcase(str(root)).replace("\\", "/").rstrip("/")
        c = os.path.normcase(str(candidate)).replace("\\", "/")
        return c == r or c.startswith(r + "/")

    def in_scope(self, project_name: str) -> bool:
        included = not self.include_projects or any(
            fnmatchcase(project_name, g) for g in self.include_projects)
        excluded = any(fnmatchcase(project_name, g) for g in self.exclude_projects)
        return included and not excluded

    # ---- knowledge ----

    @staticmethod
    def _classify(parts, name: str) -> bool:
        """THE knowledge rule, shared by knowledge_files() and resolve():
        `parts` = directory parts below the project dir, `name` = file name.
        Case-insensitive throughout so a mixed-case dir/file spelling on disk
        still matches (Windows especially)."""
        n = name.lower()
        if not n.endswith(".md") or n in _SKIP_FILES_L:
            return False
        low = [p.lower() for p in parts]
        if any(p in _SKIP_DIRS_L for p in low):
            return False
        if any(a == "research" and b == "sessions" for a, b in zip(low, low[1:])):
            # The v1 raw capture lane is separate from v2 knowledge and captures.
            return False
        if not low:
            return n in _ROOT_LOWER
        return any(p in _KNOWLEDGE_DIRS_L for p in low)

    def _contained(self, path, root: str, lane: str, project: str):
        """The LaneFile for `path`, or None when the CANONICAL path is not a
        member of this lane under `root`.

        A walked path can lie about itself in two ways, and both are settled on
        the canonical spelling — the identity the index stores.

        It can LEAVE the root: what gets indexed is a link's TARGET, so
        `docs/leak.md -> C:/secrets/x.md` pulls foreign, possibly secret
        content into the index. `_within` against the root the file was
        ENUMERATED from also keeps attribution honest — a link into another
        repo is that repo's file, and its own enumeration finds it under its
        own name.

        It can stay inside and still NOT BE A MEMBER: with
        `proj/docs/vendor -> proj/src`, the walk classified `docs/vendor/x.md`
        as knowledge while the stored path was `proj/src/x.md`, which
        `resolve()` rejects. `rescan` then found the same document in both
        `todo` and `vanished` and upserted and deleted it forever, calling
        `optimize()` each round — the version churn the optimize cooldown
        exists to prevent. So membership is re-asked of `resolve()`, the one
        authoritative rule, on the canonical path.

        Lane and project stay with the ENUMERATION: it knows the on-disk
        spelling of the project name, which the case-folded canonical path no
        longer carries."""
        c = canonical(path)
        if not self._within(root, c):
            logger.warning("skipping %s: resolves to %s, outside %s", path, c, root)
            return None
        member = self.resolve(c)
        if member is None or member.lane != lane:
            logger.warning("skipping %s: %s is not a member of the %s lane", path, c, lane)
            return None
        return LaneFile(c, lane, project)

    def knowledge_files(self) -> list:
        self.skipped_projects = 0
        out = []
        if not self.repos_dir.exists():
            return out
        for project_dir in sorted(self.repos_dir.iterdir()):
            if not project_dir.is_dir() or project_dir.name.startswith("."):
                continue
            if os.path.normcase(os.path.realpath(project_dir)) != os.path.normcase(
                    os.path.join(self._repos_real, project_dir.name)):
                continue    # junction/symlink: the real dir is enumerated under its own name (or is outside repos)
            project = project_dir.name
            if not self.in_scope(project):
                self.skipped_projects += 1
                continue
            project_root = canonical(project_dir)
            for dirpath, dirnames, filenames in os.walk(project_dir):
                rel = Path(dirpath).relative_to(project_dir).parts
                parent = rel[-1].lower() if rel else None
                dirnames[:] = sorted(
                    d for d in dirnames
                    if d.lower() not in _SKIP_DIRS_L
                    and not (parent == "research" and d.lower() == "sessions"))
                for fn in sorted(filenames):
                    if not self._classify(rel, fn):
                        continue
                    lf = self._contained(Path(dirpath) / fn, project_root, KNOWLEDGE_LANE, project)
                    if lf is not None:
                        out.append(lf)
        return by_canonical_path(out)

    def ensure_dirs(self):
        (self.mainframe_dir / "captures").mkdir(parents=True, exist_ok=True)

    def watch_roots(self) -> list:
        """The directories the watcher observes recursively. With an
        `include_projects` allowlist this is the in-scope project dirs, NOT all
        of `repos_dir`: excluded repositories' build/git churn
        would otherwise land on the single watchdog dispatcher thread and can
        overflow ReadDirectoryChangesW. Re-evaluated on every watcher start, so
        a project (or `repos_dir` itself) created later is picked up."""
        roots = []
        if self.include_projects:
            try:
                roots = [d for d in sorted(self.repos_dir.iterdir())
                         if d.is_dir() and not d.name.startswith(".") and self.in_scope(d.name)]
            except OSError:
                roots = []
        elif self.repos_dir.exists():
            roots = [self.repos_dir]
        captures = self.mainframe_dir / "captures"
        if captures.exists():
            roots.append(captures)
        return roots

    # ---- watcher predicates (pure string; no realpath, no stat) ----

    @staticmethod
    def _rel_parts(path_fwd: str, root):
        """`path_fwd`'s parts below `root`, () for the root itself, None when it
        is not under it. Case-folded like `canonical`, string-only: the watcher
        calls this for EVERY filesystem event, and the event may describe a
        path that no longer exists."""
        root_fwd = str(root).replace("\\", "/").rstrip("/")
        a, b = os.path.normcase(path_fwd), os.path.normcase(root_fwd)
        if a == b:
            return ()
        if not a.startswith(b + os.path.normcase("/")):
            return None
        return tuple(p for p in path_fwd[len(root_fwd) + 1:].split("/") if p)

    def could_belong(self, path) -> bool:
        """Could this PATH be a lane member, judged from its spelling alone?

        The watcher's file-relevance predicate. `resolve()` cannot serve here:
        it starts with `realpath` + `is_file`, so a delete event (and the source
        half of a rename) is always rejected — the file is already gone — and
        removals silently depended on a full-lane rescan. It is also a
        syscall pair per event before any cheap rejection.

        Deliberately an OVER-approximation (it cannot see symlinks or whether
        the path is a file): everything it admits is resolved authoritatively
        by the pipeline, which no-ops a non-member and turns a vanished ledger
        path into a delete."""
        p = str(path).replace("\\", "/")
        rel = self._rel_parts(p, self._captures_real)
        if rel is not None:
            return (len(rel) == 2 and bool(SAFE_NAME_RE.match(rel[0]))
                    and rel[1].lower().endswith(".md"))
        rel = self._rel_parts(p, self._repos_real)
        if rel is None or len(rel) < 2:
            return False
        project = rel[0]
        if project.startswith(".") or not self.in_scope(project):
            return False
        return self._classify(rel[1:-1], rel[-1])

    def dir_matters(self, path) -> bool:
        """Should a DIRECTORY event on this path trigger a reconciling rescan?

        True for `repos_dir`, the captures root and a `captures/<project>` dir
        (a new project may have appeared), for a project directory itself, and
        for a KNOWLEDGE-dir subtree inside an in-scope project.

        The knowledge-dir requirement is the point: watchdog emits a
        DirModified on the PARENT of every changed file, so without it a build
        under `src/`, a test-fixture write or a `.pytest_cache` refresh in any
        in-scope repo costs a full-lane rescan. Nothing outside a knowledge dir
        (or a project root, where the root context files live) can change what
        an enumeration would find. Nested layouts still work: `packages/x/docs`
        has `docs` among its parts.

        ALWAYS False for `mainframe_dir` and everything under it except the
        capture lane — otherwise, with `mainframe_dir` nested under
        `repos_dir`, the index's own `.lancedb`/`events.db`/`daemon.log` writes
        would flag a rescan after every batch, forever."""
        p = str(path).replace("\\", "/")
        if self._rel_parts(p, self._mainframe_real) is not None:
            rel = self._rel_parts(p, self._captures_real)
            # The capture lane is flat: quarantine/ and overflow/ (and anything
            # deeper) are neither indexed nor pending.
            return rel is not None and len(rel) <= 1
        rel = self._rel_parts(p, self._repos_real)
        if rel is None:
            return False
        if not rel:
            return True
        project = rel[0]
        if project.startswith(".") or not self.in_scope(project):
            return False
        low = [d.lower() for d in rel[1:]]
        if any(d in _SKIP_DIRS_L for d in low):
            return False
        if any(a == "research" and b == "sessions" for a, b in zip(low, low[1:])):
            return False
        if not low:
            return True                        # the project dir itself
        return any(d in _KNOWLEDGE_DIRS_L for d in low)

    def resolve(self, path):
        """Lane membership for ONE path — same rule as the enumerators."""
        real = Path(os.path.realpath(path))                     # on-disk spelling, symlinks resolved
        if not real.is_file():
            return None
        c = canonical(real)
        try:
            rel = real.relative_to(Path(self._captures_real))
        except ValueError:
            rel = None
        if rel is not None:
            if len(rel.parts) == 2 and SAFE_NAME_RE.match(rel.parts[0]) and rel.parts[1].lower().endswith(".md"):
                return LaneFile(c, CAPTURE_LANE, rel.parts[0])
            return None
        try:
            rel = real.relative_to(Path(self._repos_real))
        except ValueError:
            return None
        if len(rel.parts) < 2:
            return None
        project = rel.parts[0]
        if project.startswith(".") or not self.in_scope(project):
            return None
        if self._classify(rel.parts[1:-1], rel.parts[-1]):
            return LaneFile(c, KNOWLEDGE_LANE, project)
        return None

    # ---- capture ----

    def capture_dir(self, project: str) -> Path:
        return self.mainframe_dir / "captures" / safe_name(project)

    def quarantine_dir(self, project: str) -> Path:
        return self.capture_dir(project) / QUARANTINE

    def overflow_dir(self, project: str) -> Path:
        return self.capture_dir(project) / OVERFLOW

    def capture_files(self, project: str | None = None) -> list:
        if project is not None:
            safe_name(project)
        root = self.mainframe_dir / "captures"
        if not root.exists():
            return []
        projects = [project] if project else sorted(p.name for p in root.iterdir() if p.is_dir() and SAFE_NAME_RE.match(p.name))
        out = []
        for proj in projects:
            d = root / proj
            if not d.is_dir():
                continue
            if os.path.normcase(os.path.realpath(d)) != os.path.normcase(
                    os.path.join(self._captures_real, proj)):
                continue    # junction/symlink: not enumerated here
            project_root = canonical(d)
            # top level only, case-insensitive .md (POSIX glob is case-sensitive): quarantine/overflow excluded
            for f in sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".md"):
                lf = self._contained(f, project_root, CAPTURE_LANE, proj)
                if lf is not None:
                    out.append(lf)
        return by_canonical_path(out)

    def all_files(self) -> list:
        return self.knowledge_files() + self.capture_files()

    # ---- notes / drafts / receipts ----

    def note_path(self, project: str, slug: str) -> Path:
        return self.repos_dir / safe_name(project) / "research" / "memory" / f"{safe_name(slug)}.md"

    def note_receipts_path(self, project: str, slug: str) -> Path:
        return self.repos_dir / safe_name(project) / "research" / "memory" / f"{safe_name(slug)}.sources.json"

    def draft_path(self, project: str, slug: str) -> Path:
        return self.mainframe_dir / "drafts" / safe_name(project) / f"{safe_name(slug)}.md"

    def draft_receipts_path(self, project: str, slug: str) -> Path:
        return self.mainframe_dir / "drafts" / safe_name(project) / f"{safe_name(slug)}.sources.json"

    def pending(self, project: str, slug: str | None = None) -> list:
        """Captures not yet absorbed by the PROMOTED note (draft receipts never
        define pending — the draft is disposable)."""
        promoted = load_receipts(self.note_receipts_path(project, slug or project))
        out = []
        for f in self.capture_files(project):
            try:
                cid = capture_id_of(f.path)
            except OSError as e:
                logger.warning(f"pending: skipping unreadable capture {f.path}: {e}")
                continue
            if cid not in promoted:
                out.append(f)
        return out
