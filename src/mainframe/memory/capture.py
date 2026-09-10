"""The capture-lane writer: one immutable, scrubbed file per
capture with a content-derived capture_id, written through the exclusive-create
descriptor so a crash can never leave an empty capture behind."""

import re
from datetime import datetime
from pathlib import Path

from mainframe.core.hashing import content_hash
from mainframe.core.secrets import scrub_secrets
from mainframe.memory.atomic import create_exclusive
from mainframe.memory.frontmatter import render
from mainframe.memory.lanes import ADHOC_PROJECT

KINDS = ("session", "decision", "note")


def write_capture(lanes, content: str, title: str, project, session_id: str = "", kind: str = "session",
                  captured_by: str = "capture_memory", cwd: str = "", branch: str = "", reason: str = "",
                  now: datetime | None = None) -> Path:
    if kind not in KINDS:
        raise ValueError(f"unknown capture kind {kind!r}; expected one of {KINDS}")
    now = now or datetime.now()
    date, hhmmss = now.strftime("%Y-%m-%d"), now.strftime("%H%M%S")
    proj = project or ADHOC_PROJECT
    target = lanes.capture_dir(proj)                       # validates the name
    sid = re.sub(r"[^A-Za-z0-9-]", "", session_id or "")[:8] or "adhoc"
    safe_title = " ".join(scrub_secrets(title or "").split()) or "untitled"
    body = (f"# Session {date} — {safe_title}" if kind == "session" else f"# {safe_title}") + \
           f"\n\n{scrub_secrets(content or '').strip()}\n"
    cid = content_hash(body)[:16]
    fm = render({"type": kind, "project": proj, "capture_id": cid, "session_id": sid, "reason": scrub_secrets(reason),
                 "date": date, "time": hhmmss, "cwd": scrub_secrets(cwd), "git_branch": scrub_secrets(branch),
                 "captured_by": scrub_secrets(captured_by)})
    return create_exclusive(target, f"{date}-{hhmmss}-{sid}-{cid[:8]}", text=fm + body)
