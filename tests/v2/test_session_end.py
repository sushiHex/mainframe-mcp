"""The SessionEnd hook: gate branches, project derivation, body shape, dedup
against the agent's own capture, and silence with no daemon anywhere."""
import io
import json
from datetime import datetime
from pathlib import Path

import pytest

from mainframe.adapters import cli, session_end as se
from mainframe.memory.frontmatter import split
from mainframe.memory.lanes import ADHOC_PROJECT, Lanes, capture_id_of
from mainframe.service.events import EventStore

SID = "0123456789abcdef-session"
NOW = datetime(2026, 10, 14, 9, 30, 0)


def _msg(role, blocks, **top):
    return {"type": role, "isSidechain": False, "sessionId": SID, "gitBranch": "main",
            "message": {"role": role, "content": blocks}, **top}


def user(text, **top):
    return _msg("user", [{"type": "text", "text": text}], **top)


def assistant(text=None, tool=None, **top):
    blocks = ([{"type": "text", "text": text}] if text else []) + \
             ([{"type": "tool_use", "name": tool, "input": {}}] if tool else [])
    return _msg("assistant", blocks, **top)


def transcript(tmp_path, entries, name="t.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return str(p)


def event(tmp_path, entries, cwd, reason="other", session_id=SID):
    return {"session_id": session_id, "transcript_path": transcript(tmp_path, entries),
            "cwd": cwd, "reason": reason, "hook_event_name": "SessionEnd"}


REAL = [user("Port the hook to v2"), assistant("Reading.", tool="Read"),
        user("Go ahead"), assistant("Decided: write through write_capture. TODO: wire the shim.\n"
                                    "Edited C:/x/repos/proj/src/mainframe/adapters/session_end.py")]


@pytest.fixture
def repo(cfg, tmp_path):
    d = tmp_path / "repos" / "proj"
    d.mkdir(parents=True)
    return d


def test_real_session_is_written_in_v2_shape_with_hook_provenance(cfg, repo, tmp_path):
    r = se.session_end(cfg, event(tmp_path, REAL, str(repo)), now=NOW)
    assert r["decision"] == "written" and r["project"] == "proj" and r["session_id"] == "01234567"
    text = (path := Path(r["path"])).read_text(encoding="utf-8")
    fields, body = split(text)
    assert fields["captured_by"] == "hook" and fields["session_id"] == "01234567"
    assert fields["reason"] == "other" and fields["git_branch"] == "main"
    assert fields["capture_id"] == capture_id_of(path) and path.parent == Lanes(cfg).capture_dir("proj")
    # exactly ONE frontmatter block: the body is sections only (the v1 port nested a second one)
    assert "---" not in body and text.count("---\n") == 2
    assert body.startswith("# Session 2026-10-14 — Port the hook to v2\n\n## Summary\nTask: Port the hook")
    assert "## Decisions\n- Decided: write through write_capture." in body
    assert "## Action items (ACT-NOW)\n- Decided: write through" in body        # the TODO line
    assert "## Files touched\n- C:/x/repos/proj/src/mainframe/adapters/session_end.py" in body
    ev = EventStore(tmp_path / "mainframe" / "events.db").recent(1)[0]
    assert ev["kind"] == "capture.written" and ev["project"] == "proj" and ev["detail"]["captured_by"] == "hook"


def test_low_signal_acceptance_junk_is_skipped_and_each_signal_alone_passes(cfg, repo, tmp_path):
    junk = [user("Reply DONE"), assistant("DONE")]
    assert se.session_end(cfg, event(tmp_path, junk, str(repo)), now=NOW)["reason"].startswith("low signal")
    assert not list((tmp_path / "mainframe" / "captures").rglob("*.md"))
    # enough prompts
    two = [user("a"), assistant("b"), user("c"), assistant("d")]
    assert se.session_end(cfg, event(tmp_path, two, str(repo)), now=NOW)["decision"] == "written"
    # one prompt but a tool use
    tool = [user("a"), assistant("b", tool="Bash")]
    assert se.session_end(cfg, event(tmp_path, tool, str(repo), session_id="s2"), now=NOW)["decision"] == "written"
    # one prompt but a decision in the agent's text (a user-turn checklist does not count)
    dec = [user("- [ ] do it"), assistant("We decided to go with X.")]
    assert se.session_end(cfg, event(tmp_path, dec, str(repo), session_id="s3"), now=NOW)["decision"] == "written"
    chk = [user("- [ ] do it"), assistant("ok")]
    assert "low signal" in se.session_end(cfg, event(tmp_path, chk, str(repo), session_id="s4"), now=NOW)["reason"]


def test_min_prompts_is_read_from_config_and_prompt_input_exit_lowers_it(cfg, repo, tmp_path):
    cfg["capture"]["min_prompts"] = 3
    two = [user("a"), assistant("b"), user("c"), assistant("d")]
    assert "low signal" in se.session_end(cfg, event(tmp_path, two, str(repo)), now=NOW)["reason"]
    r = se.session_end(cfg, event(tmp_path, two, str(repo), reason="prompt_input_exit"), now=NOW)
    assert r["decision"] == "written"


def test_noise_and_sidechain_turns_do_not_count(cfg, repo, tmp_path):
    t = [user("<system-reminder>injected</system-reminder>"), user("<command-name>/x</command-name>"),
         user("real", isSidechain=True), assistant("sub", tool="Bash", isSidechain=True),
         user("only prompt"), assistant("done")]
    r = se.session_end(cfg, event(tmp_path, t, str(repo)), now=NOW)
    assert r["reason"] == "low signal: 1 prompts, no tool use, no decisions"


def test_project_from_cwd_else_adhoc_and_exclude_projects(cfg, repo, tmp_path):
    assert se.session_end(cfg, event(tmp_path, REAL, str(repo / "sub" / "dir")), now=NOW)["project"] == "proj"
    r = se.session_end(cfg, event(tmp_path, REAL, str(tmp_path / "elsewhere"), session_id="s2"), now=NOW)
    assert r["project"] == ADHOC_PROJECT and r["decision"] == "written"
    assert (tmp_path / "mainframe" / "captures" / ADHOC_PROJECT).is_dir()
    cfg["capture"]["exclude_projects"] = ["pr*"]
    r = se.session_end(cfg, event(tmp_path, REAL, str(repo), session_id="s3"), now=NOW)
    assert r == {"decision": "skip", "project": "proj", "session_id": "s3", "reason": "excluded project"}


def test_skips_when_the_agent_captured_this_session_itself(cfg, repo, tmp_path):
    # (a) the transcript shows the Mainframe MCP capture tool, even on a sidechain
    t = REAL + [assistant("saving", tool="mcp__mainframe__capture", isSidechain=True)]
    r = se.session_end(cfg, event(tmp_path, t, str(repo)), now=NOW)
    assert r["reason"] == "the agent captured this session itself"
    # (b) a capture carrying this session id already exists, in ANY project
    other = Lanes(cfg).capture_dir("other")
    other.mkdir(parents=True)
    (other / "2026-10-13-120000-01234567-deadbeef.md").write_text("x", encoding="utf-8")
    r = se.session_end(cfg, event(tmp_path, REAL, str(repo)), now=NOW)
    assert r["reason"] == "a capture for this session exists"
    # a screenshot tool named capture is not ours
    t = REAL + [assistant("shot", tool="mcp__playwright__capture")]
    assert se.session_end(cfg, event(tmp_path, t, str(repo), session_id="fresh"), now=NOW)["decision"] == "written"


def test_daily_cap_diverts_to_overflow_never_drops(cfg, repo, tmp_path):
    cfg["capture"]["daily_cap"] = 2
    L = Lanes(cfg)
    for i in range(2):
        L.capture_dir("proj").mkdir(parents=True, exist_ok=True)
        (L.capture_dir("proj") / f"2026-10-14-00000{i}-aaaaaaaa-bbbbbbbb.md").write_text("x", encoding="utf-8")
    (L.capture_dir("proj") / "2026-10-13-000000-cccccccc-dddddddd.md").write_text("y", encoding="utf-8")
    r = se.session_end(cfg, event(tmp_path, REAL, str(repo)), now=NOW)
    assert r["decision"] == "overflow" and Path(r["path"]).parent == L.overflow_dir("proj")
    assert EventStore(tmp_path / "mainframe" / "events.db").recent(1)[0]["detail"]["overflow"] is True
    assert Path(r["path"]).name not in {Path(f.path).name for f in L.capture_files("proj")}   # not a lane member


def test_oversize_transcript_reads_head_and_tail_and_fails_open(cfg, repo, tmp_path):
    cfg["capture"]["max_transcript_bytes"] = 4096
    filler = [assistant("x" * 200) for _ in range(40)]        # ~10 KB, one prompt, no tool use
    t = [user("The task line")] + filler + [assistant("Final outcome api_key=hunter2secret")]
    lines, bounded = se.read_transcript(transcript(tmp_path, t), 4096)
    assert bounded and all(json.loads(line) for line in lines)      # no torn line at either seam
    r = se.session_end(cfg, event(tmp_path, t, str(repo)), now=NOW)
    assert r["decision"] == "written"                               # counters unknown: never skip
    body = split(Path(r["path"]).read_text(encoding="utf-8"))[1]
    assert "Task: The task line" in body and "Outcome: Final outcome" in body
    assert "hunter2secret" not in body and "[REDACTED]" in body


def test_no_daemon_is_not_a_failure_and_nothing_ever_raises(cfg, repo, tmp_path, monkeypatch):
    log = tmp_path / "mainframe" / "hook.log"
    assert not (tmp_path / "mainframe" / "daemon.json").exists()
    stdin = io.StringIO(json.dumps(event(tmp_path, REAL, str(repo))))
    assert se.main(config=cfg, stdin=stdin, now=NOW) == 0
    assert "session-end written project=proj session=01234567:" in log.read_text(encoding="utf-8")
    assert len(list(Lanes(cfg).capture_dir("proj").glob("*.md"))) == 1
    # garbage on stdin, a missing transcript, and a config that cannot load: one log line each, exit 0
    assert se.main(config=cfg, stdin=io.StringIO("not json")) == 0
    assert se.main(config=cfg, stdin=io.StringIO(json.dumps({"transcript_path": str(tmp_path / "nope")}))) == 0
    monkeypatch.setattr(se, "load_config", lambda: (_ for _ in ()).throw(RuntimeError("bad config")))
    monkeypatch.setitem(se.DEFAULTS["paths"], "mainframe_dir", str(tmp_path / "mainframe"))
    assert se.main(stdin=io.StringIO("{}")) == 0
    text = log.read_text(encoding="utf-8")
    assert "error JSONDecodeError" in text and "skip project=- session=-: no transcript" in text
    assert "error RuntimeError: bad config" in text


def test_cli_hook_subcommand_reads_the_event_on_stdin(cfg, repo, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event(tmp_path, REAL, str(repo)))))
    assert cli.main(["hook", "session-end"], config=cfg) == 0
    assert len(list(Lanes(cfg).capture_dir("proj").glob("*.md"))) == 1
