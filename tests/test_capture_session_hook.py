"""§9.7 — capture_session.py hook logic (offline, against a sample transcript).

Fixture matches the schema verified against a real transcript: message entries
with str or block-list content (text/thinking/tool_use/tool_result blocks),
top-level sessionId/cwd/gitBranch/isSidechain.
"""

import json
from pathlib import Path

from mainframe_mcp import session_capture as sc

SID = "deadbeef1234"
SECRET = "ghp_" + "C" * 36


def _write_transcript(tmp_path, cwd):
    lines = [
        {"type": "mode", "mode": "normal", "sessionId": SID},
        {
            "type": "user", "isSidechain": False, "sessionId": SID,
            "cwd": cwd, "gitBranch": "master",
            "message": {"role": "user",
                        "content": f"Implement the widget. Pasted token {SECRET} for deploy."},
        },
        {
            "type": "assistant", "isSidechain": False, "sessionId": SID, "cwd": cwd,
            "message": {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "secret private reasoning here"},
                {"type": "text", "text": "Working on the widget now."},
                {"type": "tool_use", "name": "Edit", "input": {"x": 1}},
            ]},
        },
        {  # sidechain (subagent) entry must be ignored
            "type": "assistant", "isSidechain": True, "sessionId": SID,
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "subagent chatter that should not be captured"}]},
        },
        {
            "type": "user", "isSidechain": False, "sessionId": SID,
            "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]},
        },
        {
            "type": "assistant", "isSidechain": False, "sessionId": SID,
            "message": {"role": "assistant", "content": [
                {"type": "text",
                 "text": "DECISION: we chose LanceDB. Edited C:/Users/x/repos/proj/foo.py:10."}]},
        },
    ]
    t = tmp_path / "transcript.jsonl"
    t.write_text("\n".join(json.dumps(o) for o in lines), encoding="utf-8")
    return t


def test_capture_session_writes_scrubbed_unique_file(tmp_path):
    repos = tmp_path / "repos"
    (repos / "proj").mkdir(parents=True)
    cwd = str(repos / "proj")
    transcript = _write_transcript(tmp_path, cwd)

    out = sc.capture_session(
        str(transcript), repos_dir=str(repos), mainframe_dir=str(tmp_path / "mainframe")
    )
    assert out is not None
    p = Path(out)
    assert p.exists()
    assert SID[:8] in p.name  # uniquely named per session
    assert p.parent == repos / "proj" / "research" / "sessions"

    body = p.read_text(encoding="utf-8")
    assert SECRET not in body and "REDACTED" in body           # secret scrubbed
    assert "secret private reasoning" not in body              # thinking never leaks
    assert "subagent chatter" not in body                      # sidechain ignored
    assert "captured_by: sessionend-hook" in body
    assert "DECISION" in body and "foo.py:10" in body          # extractive capture


def test_capture_session_skips_when_already_captured(tmp_path):
    repos = tmp_path / "repos"
    (repos / "proj").mkdir(parents=True)
    cwd = str(repos / "proj")
    transcript = _write_transcript(tmp_path, cwd)

    first = sc.capture_session(
        str(transcript), repos_dir=str(repos), mainframe_dir=str(tmp_path / "mainframe")
    )
    assert first is not None
    # Dual-trigger dedup: a *-<session8>*.md already exists -> skip.
    second = sc.capture_session(
        str(transcript), repos_dir=str(repos), mainframe_dir=str(tmp_path / "mainframe")
    )
    assert second is None


def test_file_path_re_ignores_urls_and_version_dirs():
    """Found in the FIRST real production capture (2026-07-13): 'Files touched'
    listed `s://github.com` (the drive-letter regex ate the tail of https://)
    and a plugin-cache dir ending in a version number (`...superpowers\6.0.3`
    — all-digit 'extension'). Real paths must still match."""
    from mainframe_mcp.session_capture import _FILE_PATH_RE

    assert _FILE_PATH_RE.findall("clone https://github.com/foo/bar.git today") == []
    assert _FILE_PATH_RE.findall("see http://example.com/a.md and ftp://x.io/b.py") == []
    assert _FILE_PATH_RE.findall(r"cache C:\Users\dev\plugins\superpowers\6.0.3 loaded") == []
    # real paths keep matching, incl. dotted dirs, :line suffixes, digit-bearing exts
    assert _FILE_PATH_RE.findall(r"edit C:\Users\dev\.claude\a.py then C:/x/b.md:12 ok") == \
        [r"C:\Users\dev\.claude\a.py", "C:/x/b.md:12"]
    assert _FILE_PATH_RE.findall("dump in C:/tmp/archive.7z now") == ["C:/tmp/archive.7z"]
