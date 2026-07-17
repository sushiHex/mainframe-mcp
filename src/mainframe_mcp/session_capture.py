"""Zero-VRAM extractive session capture (SessionEnd backstop).

Reads a Claude Code transcript JSONL, does cheap regex/extractive capture (no
LLM, no network, no torch), secret-scrubs it, and writes ONE uniquely-named
immutable markdown file under <repo>/research/sessions/ (or the library
fallback). It deliberately does NOT call the MCP server — the stdio server may
be torn down at SessionEnd; the file is the durable handoff, ingested later by
the SessionStart loop.

Schema verified against a real transcript (2026-06-27): each line is a JSON
object; message entries have type in {user, assistant} with message.role and
message.content (a str, or a list of blocks of type text/thinking/tool_use/
tool_result). Top-level cwd/gitBranch/sessionId/isSidechain are present.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

try:  # share the canonical scrubber when the package is importable
    from mainframe_mcp.secrets import scrub_secrets
except Exception:  # pragma: no cover - standalone fallback
    def scrub_secrets(text: str) -> str:
        return text

try:  # default dirs from the single config source of truth when importable
    from mainframe_mcp.config import DEFAULTS as _DEFAULTS
    DEFAULT_REPOS_DIR = _DEFAULTS["paths"]["repos_dir"]
    DEFAULT_MAINFRAME_DIR = _DEFAULTS["paths"]["mainframe_dir"]
except Exception:  # pragma: no cover - standalone fallback
    DEFAULT_REPOS_DIR = str(Path.home() / "repos")
    DEFAULT_MAINFRAME_DIR = str(Path.home() / ".claude" / "mainframe")

# Bound extraction work/memory on very long transcripts (the SessionEnd hook has
# a wall-clock timeout). The Summary uses only users[0]/assistants[-1]; the
# regex scans need only a bounded tail.
_MAX_USER_TURNS = 12
_MAX_ASSISTANT_TURNS = 40

# Noise prefixes: harness command wrappers (incl. <local-command-caveat>) that
# are not user intent. NOT bare "caveat:" — a real prompt may start "Caveat:".
_NOISE_PREFIXES = (
    "<local-command", "<command-", "<bash-",
    "<system-reminder", "<user-prompt-submit-hook", "<post-tool-use",
)

# Windows path ending in an extension. Greedy body up to the last .ext, stopping
# at whitespace — this correctly handles dotted directories (e.g. .claude) and
# multiple paths on one line. (A non-greedy form truncates at the first dot;
# allowing spaces makes it over-grab across paths.)
# Two guards added after the FIRST real production capture (2026-07-13):
# - lookbehind: `https://github.com` must not yield the fake path `s://github.com`
#   (any letter directly before the "drive letter" means it's a URL scheme tail);
# - the extension needs at least one LETTER (`[0-9]*[A-Za-z][A-Za-z0-9]*`), so
#   a version-number dir like `plugins\superpowers\6.0.3` doesn't pass with
#   "extension" `3`. Tradeoff: all-digit extensions (e.g. `log.2026`) are
#   skipped — noise wins that one.
# Best-effort heuristic, deliberately not a parser: if a THIRD class of false
# positive shows up, stop patching the regex and tokenize instead.
_FILE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s:*?\"<>|]+\.[0-9]*[A-Za-z][A-Za-z0-9]*(?::\d+)?")
_DECISION_RE = re.compile(r"(?im)^.*\b(decision|decided|chose|we will|going with)\b.*$")
_ACTION_RE = re.compile(r"(?im)^.*(act-now|\btodo\b|\[ \]|next step|action item).*$")
# An assistant tool_use of capture_memory means the agent already captured this
# session in-context, so the backstop must skip (deterministic dual-trigger dedup).
_CAPTURE_TOOL = "capture_memory"


def _blocks_text(content) -> list[str]:
    """Extract human-readable text from a message.content (str or block list)."""
    if isinstance(content, str):
        return [content]
    out = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                if t:
                    out.append(t)
    return out


def _has_capture_tool_use(content) -> bool:
    """True if an assistant message block is a tool_use of capture_memory."""
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            if _CAPTURE_TOOL in str(block.get("name", "")):
                return True
    return False


def parse_transcript(transcript_path: str) -> dict:
    """Return {session_id, cwd, git_branch, user_texts, assistant_texts,
    capture_memory_called}. Text lists are bounded for long transcripts."""
    user_texts: list[str] = []
    assistant_texts: list[str] = []
    session_id = ""
    cwd = ""
    git_branch = ""
    capture_memory_called = False

    p = Path(transcript_path)
    if not p.exists():
        return {
            "session_id": "", "cwd": "", "git_branch": "",
            "user_texts": [], "assistant_texts": [], "capture_memory_called": False,
        }

    with p.open("r", encoding="utf-8-sig") as f:  # -sig strips a leading BOM
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("type")
            content = (obj.get("message") or {}).get("content") if t in ("user", "assistant") else None
            # Detect a capture_memory call even in subagent (sidechain) traffic —
            # the backstop must skip whenever ANY agent already captured.
            if t == "assistant" and not capture_memory_called and _has_capture_tool_use(content):
                capture_memory_called = True
            if obj.get("isSidechain"):
                continue  # subagent traffic — skip its text/metadata
            session_id = session_id or obj.get("sessionId", "")
            cwd = cwd or obj.get("cwd", "")
            git_branch = git_branch or obj.get("gitBranch", "")
            if t not in ("user", "assistant"):
                continue
            for text in _blocks_text(content):
                stripped = text.strip()
                if not stripped:
                    continue
                if t == "user":
                    if stripped.lower().startswith(_NOISE_PREFIXES):
                        continue
                    if len(user_texts) < _MAX_USER_TURNS:
                        user_texts.append(stripped)
                else:
                    assistant_texts.append(stripped)
                    if len(assistant_texts) > _MAX_ASSISTANT_TURNS:
                        del assistant_texts[0]  # keep the most recent tail

    return {
        "session_id": session_id,
        "cwd": cwd,
        "git_branch": git_branch,
        "user_texts": user_texts,
        "assistant_texts": assistant_texts,
        "capture_memory_called": capture_memory_called,
    }


def _collect(pattern, text: str, cap: int) -> list[str]:
    """First `cap` distinct stripped matches of `pattern` in `text` (early-exit,
    so cost is bounded by cap, not by total match count)."""
    seen, out = set(), []
    for m in pattern.finditer(text):
        item = m.group(0).strip()[:300]
        if item and item not in seen:
            seen.add(item)
            out.append(item)
            if len(out) >= cap:
                break
    return out


def extract_capture(parsed: dict) -> tuple[str, str]:
    """Build (title, markdown_content) via cheap extraction. Caller scrubs."""
    users = parsed["user_texts"]
    assistants = parsed["assistant_texts"]
    assistant_text = "\n".join(assistants)
    all_text = assistant_text + "\n" + "\n".join(users)

    title = "session"
    if users:
        title = users[0].splitlines()[0].strip()[:80] or "session"

    sections = []

    summary_bits = []
    if users:
        summary_bits.append(f"Task: {users[0][:600]}")
    if assistants:
        summary_bits.append(f"Outcome: {assistants[-1][:600]}")
    if summary_bits:
        sections.append("## Summary\n" + "\n\n".join(summary_bits))

    def _bullets(heading, items):
        if items:
            sections.append(f"## {heading}\n" + "\n".join(f"- {i}" for i in items))

    # Decisions/actions come from the agent's own text only — user-role turns
    # also carry injected skill/checklist content (e.g. a TDD "- [ ]" list).
    _bullets("Decisions", _collect(_DECISION_RE, assistant_text, 8))
    _bullets("Action items (ACT-NOW)", _collect(_ACTION_RE, assistant_text, 10))
    _bullets("Files touched", _collect(_FILE_PATH_RE, all_text, 20))

    return title, "\n\n".join(sections) + "\n"


def library_sessions_dir(mainframe_dir: str) -> Path:
    """The out-of-repo fallback session lane. Single source of truth for the
    `<mainframe_dir>/library/sessions/` path (used by resolve_sessions_dir and
    the index sync)."""
    return Path(mainframe_dir) / "library" / "sessions"


def session_dirs(repos_dir: str, mainframe_dir: str, project: str | None = None) -> list[Path]:
    """All session-capture directories — single source of truth for the session
    lane layout. One project's lane, or every repo's lane plus the library
    fallback. Shared by the index sync and the SessionEnd/Start tooling."""
    repos = Path(repos_dir)
    if project:
        return [repos / project / "research" / "sessions"]
    return list(repos.glob("*/research/sessions")) + [library_sessions_dir(mainframe_dir)]


def pending_sessions_by_project(repos_dir: str, mainframe_dir: str) -> dict[str, int]:
    """Count un-consolidated session captures per project (files still sitting
    in a session lane; consolidation archives them away). The library fallback
    lane is reported as 'library'. This is THE memory-loop trigger signal —
    surfaced through sync_index/status so the agent sees when a project's
    backlog is worth a consolidate_sessions call."""
    counts: dict[str, int] = {}
    lib_dir = library_sessions_dir(mainframe_dir)
    for d in session_dirs(repos_dir, mainframe_dir):
        if not d.exists():
            continue
        n = sum(1 for _ in d.glob("*.md"))
        if not n:
            continue
        key = "library" if d == lib_dir else d.parts[-3]  # <repo>/research/sessions
        counts[key] = counts.get(key, 0) + n
    return counts


def resolve_sessions_dir(cwd: str, repos_dir: str, mainframe_dir: str, project: str | None = None):
    """Return (sessions_dir, repo_dir) — the single sessions-dir policy.

    In-repo primary: <repos_dir>/<project or cwd's repo>/research/sessions/.
    Fallback (cwd outside repos_dir): <mainframe_dir>/library/sessions/ — never
    the Mainframe's own code repo. repo_dir is None for the fallback. Shared by
    capture_memory (server.py) and the SessionEnd hook so the policy lives once.
    """
    repos = Path(repos_dir)
    repo_dir = None
    if project:
        repo_dir = repos / project
    else:
        try:
            rel = Path(cwd).resolve().relative_to(repos.resolve())
            if rel.parts:
                repo_dir = repos / rel.parts[0]
        except (ValueError, OSError):
            repo_dir = None
    if repo_dir is not None:
        return repo_dir / "research" / "sessions", repo_dir
    return library_sessions_dir(mainframe_dir), None


def already_captured(sessions_dir: Path, session8: str) -> bool:
    """Dual-trigger dedup: skip if the SKILL (or an earlier hook) already wrote."""
    if not session8:
        return False
    return any(sessions_dir.glob(f"*{session8}*.md"))


def git_branch_from_repo(repo_dir) -> str:
    """Best-effort current branch by reading <repo>/.git/HEAD (no subprocess)."""
    if repo_dir is None:
        return ""
    try:
        txt = (Path(repo_dir) / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return txt.split("/", 2)[-1] if txt.startswith("ref:") else txt[:12]


def unique_session_path(sessions_dir: Path, base: str) -> Path:
    """One immutable file per capture — never overwrite an existing file."""
    path = sessions_dir / f"{base}.md"
    n = 2
    while path.exists():
        path = sessions_dir / f"{base}-{n}.md"
        n += 1
    return path


_VALID_KINDS = ("session", "decision", "note")


def _yaml_scalar(v: str) -> str:
    """Flatten a value to a single safe YAML line so an embedded newline can't
    close the frontmatter block early or inject extra keys (finding #11)."""
    return " ".join(str(v).splitlines()).strip()


def build_frontmatter(
    kind: str, project: str, session_id: str, date: str,
    hhmmss: str, cwd: str, branch: str, captured_by: str,
) -> str:
    """Compose the session-capture YAML frontmatter. Shared by both capture
    paths; all values are flattened to one line and `kind` is enum-validated."""
    kind = kind if kind in _VALID_KINDS else "note"
    return (
        "---\n"
        f"type: {kind}\n"
        f"project: {_yaml_scalar(project)}\n"
        f"session_id: {_yaml_scalar(session_id)}\n"
        f"date: {_yaml_scalar(date)}\n"
        f"time: {_yaml_scalar(hhmmss)}\n"
        f"cwd: {_yaml_scalar(str(cwd).replace(chr(92), '/'))}\n"
        f"git_branch: {_yaml_scalar(branch)}\n"
        f"captured_by: {_yaml_scalar(captured_by)}\n"
        "---\n"
    )


def capture_session(
    transcript_path: str,
    session_id: str = "",
    cwd: str = "",
    repos_dir: str = DEFAULT_REPOS_DIR,
    mainframe_dir: str = DEFAULT_MAINFRAME_DIR,
    now: datetime | None = None,
):
    """Parse -> dedup-skip -> scrub -> write one immutable file. Returns path|None."""
    parsed = parse_transcript(transcript_path)
    session_id = session_id or parsed["session_id"]
    cwd = cwd or parsed["cwd"] or os.getcwd()
    session8 = (session_id or "")[:8]

    # Deterministic dual-trigger dedup: if the agent already called
    # capture_memory in-session, the skill captured a higher-quality note —
    # skip the backstop. This works even when the skill used session_id="adhoc"
    # (the filename-glob guard below would miss that case; finding #5).
    if parsed.get("capture_memory_called"):
        return None

    sessions_dir, repo_dir = resolve_sessions_dir(cwd, repos_dir, mainframe_dir)

    if already_captured(sessions_dir, session8):
        return None  # an earlier hook already wrote this session — avoid a near-dupe

    title, content = extract_capture(parsed)
    if not content.strip():
        return None  # nothing worth capturing

    safe_title = scrub_secrets(title).strip() or "session"
    safe_content = scrub_secrets(content)
    branch = parsed["git_branch"] or git_branch_from_repo(repo_dir)

    now = now or datetime.now()
    date = now.strftime("%Y-%m-%d")
    hhmmss = now.strftime("%H%M%S")
    sid = session8 or "adhoc"

    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = unique_session_path(sessions_dir, f"{date}-{hhmmss}-{sid}")

    proj = repo_dir.name if repo_dir is not None else "adhoc"
    frontmatter = build_frontmatter(
        "session", proj, sid, date, hhmmss, cwd, branch, "sessionend-hook"
    )
    md = f"{frontmatter}# Session {date} — {safe_title}\n\n{safe_content.strip()}\n"
    path.write_text(md, encoding="utf-8")
    return str(path)


def main():  # pragma: no cover - exercised via the hook shim
    import sys

    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {}

    transcript_path = data.get("transcript_path") or ""
    if not transcript_path:
        return  # best-effort; nothing to do without a transcript

    repos_dir = os.environ.get("MAINFRAME_REPOS_DIR", DEFAULT_REPOS_DIR)
    mainframe_dir = os.environ.get("MAINFRAME_DIR", DEFAULT_MAINFRAME_DIR)
    try:
        out = capture_session(
            transcript_path=transcript_path,
            session_id=data.get("session_id", ""),
            cwd=data.get("cwd", ""),
            repos_dir=repos_dir,
            mainframe_dir=mainframe_dir,
        )
        if out:
            print(f"capture_session wrote {out}")
    except Exception as e:
        print(f"capture_session skipped: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
