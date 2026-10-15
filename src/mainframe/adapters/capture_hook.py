"""The Claude Code capture hooks (spec 5.1): zero-VRAM extractive capture.

A capture covers one SEGMENT of a session: the transcript since its last
compaction (its `compact_boundary` record), or since its start. Long sessions
are compacted many times and rarely end, so one capture per session would be a
capture a week, its "task" the session's first prompt. `mainframe hook
pre-compact` captures the segment a compaction is about to summarize;
`mainframe hook session-end` captures the last one.

Each reads the hook event JSON on stdin ({session_id, transcript_path, cwd,
reason | trigger}), parses the segment within `capture.max_transcript_bytes`,
applies the gate, and exclusive-creates
ONE scrubbed file through the same writer the MCP tool and the CLI use
(`write_capture`) -- so a hook capture has exactly their shape: one frontmatter
block, one `capture_id`, sections in the body. It contacts no daemon: the
capture root is a watch root and the periodic rescan is the backstop, and the
WAL event store is multi-writer precisely so an out-of-process hook can append
`capture.written` itself. (REST `/capture` could not carry this provenance
anyway: it records `rest` for whatever posts to it, by design.)

Every decision is one line in `<mainframe_dir>/hook.log`. `main` never raises:
a capture is never worth a broken session teardown or compaction. Nothing here
may import the store, the models or the MCP stack: SessionEnd's default hook
timeout is 1.5 s (test_capture_hook pins the import set).
"""

import json
import os
import re
import sys
from datetime import datetime
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace

from mainframe.config import DEFAULTS, load_config
from mainframe.memory.capture import write_capture
from mainframe.memory.lanes import ADHOC_PROJECT, SAFE_NAME_RE, Lanes
from mainframe.service.events import EventStore

CAPTURED_BY = "hook"
LOG_NAME = "hook.log"
HEAD_BYTES = 1024 * 1024        # of an oversize segment: the task is at the head, the outcome at the tail
SCAN_BYTES = 4 * 1024 * 1024    # backward-scan block when looking for the segment's start
_BOUNDARY = b'"subtype":"compact_boundary"'   # unescaped quotes: only ever a record's own key
_MAX_USER_TURNS = 12
_MAX_ASSISTANT_TURNS = 40

# Harness wrappers that are not user intent. NOT bare "caveat:" -- a real
# prompt may start "Caveat:".
_NOISE_PREFIXES = ("<local-command", "<command-", "<bash-", "<system-reminder",
                   "<user-prompt-submit-hook", "<post-tool-use")
# Windows path ending in an extension (drive letter not preceded by a letter, so
# `https://` yields no path; the extension needs a letter, so `6.0.3` is not one).
_FILE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s:*?\"<>|]+\.[0-9]*[A-Za-z][A-Za-z0-9]*(?::\d+)?")
_DECISION_RE = re.compile(r"(?im)^.*\b(decision|decided|chose|we will|going with)\b.*$")
_ACTION_RE = re.compile(r"(?im)^.*(act-now|\btodo\b|\[ \]|next step|action item).*$")


def _is_capture_tool(name) -> bool:
    """The Mainframe MCP `capture` tool under any host's server prefix (`mcp__mainframe__capture`)."""
    return "mainframe" in name and name.endswith("capture")


def segment_start(f, size: int) -> tuple[int, str]:
    """(offset just past the last compact_boundary record, its timestamp), or
    (0, "") for a session never compacted. Scans backwards in blocks; a block
    that is all one line is carried into the next, so a hit is always parsed
    as a whole line."""
    pos, carry = size, b""
    while pos > 0:
        lo = max(0, pos - SCAN_BYTES)
        f.seek(lo)
        buf = f.read(pos - lo) + carry
        first = buf.find(b"\n") if lo else -1    # buf[:first + 1] may be a torn line
        if lo and first == -1:
            pos, carry = lo, buf
            continue
        hit = len(buf)
        while (hit := buf.rfind(_BOUNDARY, first + 1, hit)) != -1:
            start, end = buf.rfind(b"\n", 0, hit) + 1, buf.find(b"\n", hit)
            end = len(buf) if end == -1 else end
            try:
                rec = json.loads(buf[start:end])
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("type") == "system" and rec.get("subtype") == "compact_boundary":
                return min(lo + end + 1, size), str(rec.get("timestamp") or "")
        pos, carry = lo, buf[:first + 1]
    return 0, ""


def read_transcript(path, cap: int) -> tuple[list[str], bool, str]:
    """(lines of the current segment, bounded, the segment's start timestamp,
    "" when it starts the session). `bounded` is True when the segment was
    larger than `cap` and only its first HEAD_BYTES and the tail that fits were
    read -- the counters are then unknown, and the gate fails open."""
    p = Path(path)
    size = p.stat().st_size
    with p.open("rb") as f:
        start, began = segment_start(f, size)
        f.seek(start)
        if size - start <= cap:
            data, bounded = f.read(), False
        else:
            head_n = min(HEAD_BYTES, cap // 2)
            head = f.read(head_n)
            f.seek(size - (cap - head_n))
            tail = f.read()
            data, bounded = head[:head.rfind(b"\n") + 1] + tail[tail.find(b"\n") + 1:], True
    return data.decode("utf-8-sig", errors="replace").splitlines(), bounded, began


def parse(lines) -> dict:
    """Per-MESSAGE counters and bounded text: a user message counts once when it
    is non-sidechain with a non-noise text block; an assistant message counts
    once when it is non-sidechain with a tool_use block. A capture tool_use is
    noticed even on a sidechain: SOME agent captured this segment. The summary
    a compaction leaves is the previous segment's, not a prompt."""
    s = {"session_id": "", "cwd": "", "git_branch": "", "users": [], "assistants": [],
         "prompts": 0, "tool_uses": 0, "agent_captured": False}
    for line in lines:
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict) or obj.get("type") not in ("user", "assistant"):
            continue
        content = (obj.get("message") or {}).get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        blocks = [b for b in blocks if isinstance(b, dict)]
        tools = [str(b.get("name", "")) for b in blocks if b.get("type") == "tool_use"]
        if any(_is_capture_tool(t) for t in tools):
            s["agent_captured"] = True
        if obj.get("isSidechain") or obj.get("isCompactSummary"):
            continue
        for key, src in (("session_id", "sessionId"), ("cwd", "cwd"), ("git_branch", "gitBranch")):
            s[key] = s[key] or str(obj.get(src) or "")
        texts = [b["text"].strip() for b in blocks
                 if b.get("type") == "text" and isinstance(b.get("text"), str) and b["text"].strip()]
        if obj["type"] == "user":
            texts = [t for t in texts if not t.lower().startswith(_NOISE_PREFIXES)]
            s["prompts"] += bool(texts)
            s["users"].extend(texts[:max(0, _MAX_USER_TURNS - len(s["users"]))])
        else:
            s["tool_uses"] += bool(tools)
            s["assistants"].extend(texts)
            del s["assistants"][:-_MAX_ASSISTANT_TURNS]
    return s


def _collect(pattern, text: str, cap: int) -> list[str]:
    seen, out = set(), []
    for m in pattern.finditer(text):
        item = m.group(0).strip()[:300]
        if item and item not in seen:
            seen.add(item)
            out.append(item)
            if len(out) >= cap:
                break
    return out


def extract(parsed: dict) -> tuple[str, str, bool]:
    """(title, markdown sections, found_signal). Decisions and actions come from
    the agent's own text only: user turns also carry injected checklists."""
    users, assistants = parsed["users"], parsed["assistants"]
    assistant_text = "\n".join(assistants)
    title = (users[0].splitlines()[0].strip()[:80] if users else "") or "session"
    sections, found = [], False
    summary = ([f"Task: {users[0][:600]}"] if users else []) + \
              ([f"Outcome: {assistants[-1][:600]}"] if assistants else [])
    if summary:
        sections.append("## Summary\n" + "\n\n".join(summary))
    for heading, items in (("Decisions", _collect(_DECISION_RE, assistant_text, 8)),
                           ("Action items (ACT-NOW)", _collect(_ACTION_RE, assistant_text, 10))):
        if items:
            found = True
            sections.append(f"## {heading}\n" + "\n".join(f"- {i}" for i in items))
    files = _collect(_FILE_PATH_RE, assistant_text + "\n" + "\n".join(users), 20)
    if files:
        sections.append("## Files touched\n" + "\n".join(f"- {f}" for f in files))
    return title, "\n\n".join(sections), found


def project_for(cwd: str, lanes: Lanes) -> str:
    """The repo under `repos_dir` that `cwd` is in, else the out-of-repo lane."""
    try:
        rel = Path(os.path.realpath(cwd)).relative_to(os.path.realpath(lanes.repos_dir))
    except ValueError:
        return ADHOC_PROJECT
    name = rel.parts[0] if rel.parts else ""
    return name if SAFE_NAME_RE.match(name) else ADHOC_PROJECT


def _stamp(began: str) -> str:
    """A boundary's UTC timestamp as a capture filename's local
    `YYYY-MM-DD-HHMMSS` prefix, so the two compare as strings."""
    return datetime.fromisoformat(began).astimezone().strftime("%Y-%m-%d-%H%M%S")


def _event_reason(event: dict) -> str:
    """SessionEnd says why the session ended; PreCompact says what compacted it."""
    if event.get("hook_event_name") == "PreCompact":
        return f"pre-compact:{event.get('trigger') or 'unknown'}"
    return str(event.get("reason") or "")


def capture_segment(config: dict, event: dict, now: datetime | None = None) -> dict:
    """Gate and act for one hook event; the decision is the return value.
    Raises only on a bug -- `main` is the never-raise boundary."""
    cap, lanes, now = config["capture"], Lanes(config), now or datetime.now()
    sid = re.sub(r"[^A-Za-z0-9-]", "", str(event.get("session_id") or ""))
    out = {"decision": "skip", "project": "", "session_id": sid[:8]}
    transcript = str(event.get("transcript_path") or "")
    if not transcript or not Path(transcript).is_file():
        return dict(out, reason="no transcript")
    lines, bounded, began = read_transcript(transcript, int(cap["max_transcript_bytes"]))
    parsed = parse(lines)
    out["session_id"] = sid8 = (sid or parsed["session_id"])[:8]
    cwd = str(event.get("cwd") or parsed["cwd"])
    out["project"] = project = project_for(cwd, lanes) if cwd else ADHOC_PROJECT
    # The gate, in spec order; the first hit decides.
    if any(fnmatchcase(project, g) for g in cap.get("exclude_projects") or []):
        return dict(out, reason="excluded project")
    title, body, found = extract(parsed)
    bar = 2 if event.get("reason") == "prompt_input_exit" else int(cap["min_prompts"])
    if not bounded and parsed["prompts"] < bar and not parsed["tool_uses"] and not found:
        return dict(out, reason=f"low signal: {parsed['prompts']} prompts, no tool use, no decisions")
    if not body.strip():
        return dict(out, reason="nothing extractable")
    if parsed["agent_captured"]:
        return dict(out, reason="the agent captured this segment itself")
    # Captured already: a capture of this session written since the segment
    # began (a session resumed and ended twice; a compaction retried).
    since = _stamp(began) if began else ""
    if sid8 and any(p.name >= since for p in (lanes.mainframe_dir / "captures").glob(f"*/*-{sid8}-*.md")):
        return dict(out, reason="a capture for this segment exists")
    today = now.strftime("%Y-%m-%d")
    full = sum(1 for _ in lanes.capture_dir(project).glob(f"{today}-*.md")) >= int(cap["daily_cap"])
    # At the daily cap the capture goes to overflow/ (neither indexed nor
    # pending), never dropped: `write_capture` asks its first argument only
    # for the directory.
    target = SimpleNamespace(capture_dir=lanes.overflow_dir) if full else lanes
    path = write_capture(target, body, title, project, session_id=sid8, captured_by=CAPTURED_BY,
                         cwd=cwd, branch=parsed["git_branch"], reason=_event_reason(event), now=now)
    EventStore(lanes.mainframe_dir / "events.db").emit(
        "capture.written", project=project, path=str(path), captured_by=CAPTURED_BY,
        **({"overflow": True} if full else {}))
    return dict(out, decision="overflow" if full else "written", path=str(path))


def _log(mainframe_dir: Path, hook: str, line: str) -> None:
    try:
        mainframe_dir.mkdir(parents=True, exist_ok=True)
        with (mainframe_dir / LOG_NAME).open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {hook} {line}\n")
    except OSError:
        pass


def main(config: dict | None = None, stdin=None, now: datetime | None = None, hook: str = "session-end") -> int:
    """The hook process: one decision line in hook.log, exit 0 whatever happened."""
    mainframe_dir = Path(DEFAULTS["paths"]["mainframe_dir"])
    try:
        config = config or load_config()
        mainframe_dir = Path(config["paths"]["mainframe_dir"])
        raw = (stdin or sys.stdin).read()
        event = json.loads(raw) if raw.strip() else {}
        r = capture_segment(config, event if isinstance(event, dict) else {}, now=now)
        line = (f"{r['decision']} project={r['project'] or '-'} session={r['session_id'] or '-'}: "
                f"{r.get('path') or r.get('reason')}")
    except Exception as e:
        line = f"error {type(e).__name__}: {e}"
    _log(mainframe_dir, hook, line)
    return 0
