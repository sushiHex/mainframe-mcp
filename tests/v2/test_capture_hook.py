"""The capture hooks: gate branches, project derivation, body shape, dedup
against the agent's own capture, and silence with no daemon anywhere."""
import io
import itertools
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from mainframe.adapters import capture_hook as se, cli
from mainframe.memory.frontmatter import split
from mainframe.memory.lanes import ADHOC_PROJECT, Lanes, capture_id_of
from mainframe.service.events import EventStore

SID = "0123456789abcdef-session"
NOW = datetime(2026, 10, 14, 9, 30, 0)


_UUIDS = itertools.count(1)


def _msg(role, blocks, **top):
    return {"type": role, "uuid": f"r-{next(_UUIDS)}", "isSidechain": False, "sessionId": SID, "gitBranch": "main",
            "message": {"role": role, "content": blocks}, **top}


def user(text, **top):
    return _msg("user", [{"type": "text", "text": text}], **top)


def assistant(text=None, tool=None, **top):
    blocks = ([{"type": "text", "text": text}] if text else []) + \
             ([{"type": "tool_use", "name": tool, "input": {}}] if tool else [])
    return _msg("assistant", blocks, **top)


def transcript(tmp_path, entries, name="t.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(e, separators=(",", ":")) for e in entries) + "\n",
                 encoding="utf-8")                                    # compact, as Claude Code writes it
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
    r = se.capture_segment(cfg, event(tmp_path, REAL, str(repo)), now=NOW)
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
    assert se.capture_segment(cfg, event(tmp_path, junk, str(repo)), now=NOW)["reason"].startswith("low signal")
    assert not list((tmp_path / "mainframe" / "captures").rglob("*.md"))
    # enough prompts
    two = [user("a"), assistant("b"), user("c"), assistant("d")]
    assert se.capture_segment(cfg, event(tmp_path, two, str(repo)), now=NOW)["decision"] == "written"
    # one prompt but a tool use
    tool = [user("a"), assistant("b", tool="Bash")]
    assert se.capture_segment(cfg, event(tmp_path, tool, str(repo), session_id="s2"), now=NOW)["decision"] == "written"
    # one prompt but a decision in the agent's text (a user-turn checklist does not count)
    dec = [user("- [ ] do it"), assistant("We decided to go with X.")]
    assert se.capture_segment(cfg, event(tmp_path, dec, str(repo), session_id="s3"), now=NOW)["decision"] == "written"
    chk = [user("- [ ] do it"), assistant("ok")]
    assert "low signal" in se.capture_segment(cfg, event(tmp_path, chk, str(repo), session_id="s4"), now=NOW)["reason"]


def test_min_prompts_is_read_from_config_and_prompt_input_exit_lowers_it(cfg, repo, tmp_path):
    cfg["capture"]["min_prompts"] = 3
    two = [user("a"), assistant("b"), user("c"), assistant("d")]
    assert "low signal" in se.capture_segment(cfg, event(tmp_path, two, str(repo)), now=NOW)["reason"]
    r = se.capture_segment(cfg, event(tmp_path, two, str(repo), reason="prompt_input_exit"), now=NOW)
    assert r["decision"] == "written"


def test_noise_and_sidechain_turns_do_not_count(cfg, repo, tmp_path):
    t = [user("<system-reminder>injected</system-reminder>"), user("<command-name>/x</command-name>"),
         user("real", isSidechain=True), assistant("sub", tool="Bash", isSidechain=True),
         user("only prompt"), assistant("done")]
    r = se.capture_segment(cfg, event(tmp_path, t, str(repo)), now=NOW)
    assert r["reason"] == "low signal: 1 prompts, no tool use, no decisions"


def test_project_from_cwd_else_adhoc_and_exclude_projects(cfg, repo, tmp_path):
    assert se.capture_segment(cfg, event(tmp_path, REAL, str(repo / "sub" / "dir")), now=NOW)["project"] == "proj"
    r = se.capture_segment(cfg, event(tmp_path, REAL, str(tmp_path / "elsewhere"), session_id="s2"), now=NOW)
    assert r["project"] == ADHOC_PROJECT and r["decision"] == "written"
    assert (tmp_path / "mainframe" / "captures" / ADHOC_PROJECT).is_dir()
    cfg["capture"]["exclude_projects"] = ["pr*"]
    r = se.capture_segment(cfg, event(tmp_path, REAL, str(repo), session_id="s3"), now=NOW)
    assert r == {"decision": "skip", "project": "proj", "session_id": "s3", "reason": "excluded project"}


def test_skips_when_the_agent_captured_this_session_itself(cfg, repo, tmp_path):
    # (a) the transcript shows the Mainframe MCP capture tool, even on a sidechain
    t = REAL + [assistant("saving", tool="mcp__mainframe__capture", isSidechain=True)]
    r = se.capture_segment(cfg, event(tmp_path, t, str(repo)), now=NOW)
    assert r["reason"] == "the agent captured this segment itself"
    # (b) an earlier capture of this session WITHOUT `through` (the agent's own MCP capture) blocks nothing
    other = Lanes(cfg).capture_dir("other")
    other.mkdir(parents=True)
    (other / "2026-10-13-120000-01234567-deadbeef.md").write_text("x", encoding="utf-8")
    r = se.capture_segment(cfg, event(tmp_path, REAL, str(repo)), now=NOW)
    assert r["decision"] == "written"
    assert split(Path(r["path"]).read_text(encoding="utf-8"))[0]["through"] == REAL[-1]["uuid"]
    # (c) a capture whose `through` is the transcript's last record: the rerun has nothing new
    r = se.capture_segment(cfg, event(tmp_path, REAL, str(repo)), now=NOW)
    assert r["decision"] == "skip" and r["reason"].startswith(("low signal", "nothing extractable"))
    assert len(list(Lanes(cfg).capture_dir("proj").glob("*.md"))) == 1
    # a screenshot tool named capture is not ours
    t = REAL + [assistant("shot", tool="mcp__playwright__capture")]
    assert se.capture_segment(cfg, event(tmp_path, t, str(repo), session_id="fresh"), now=NOW)["decision"] == "written"


def test_daily_cap_diverts_to_overflow_never_drops(cfg, repo, tmp_path):
    cfg["capture"]["daily_cap"] = 2
    L = Lanes(cfg)
    for i in range(2):
        L.capture_dir("proj").mkdir(parents=True, exist_ok=True)
        (L.capture_dir("proj") / f"2026-10-14-00000{i}-aaaaaaaa-bbbbbbbb.md").write_text("x", encoding="utf-8")
    (L.capture_dir("proj") / "2026-10-13-000000-cccccccc-dddddddd.md").write_text("y", encoding="utf-8")
    r = se.capture_segment(cfg, event(tmp_path, REAL, str(repo)), now=NOW)
    assert r["decision"] == "overflow" and Path(r["path"]).parent == L.overflow_dir("proj")
    assert EventStore(tmp_path / "mainframe" / "events.db").recent(1)[0]["detail"]["overflow"] is True
    assert Path(r["path"]).name not in {Path(f.path).name for f in L.capture_files("proj")}   # not a lane member


def test_oversize_transcript_reads_head_and_tail_and_fails_open(cfg, repo, tmp_path):
    cfg["capture"]["max_transcript_bytes"] = 4096
    filler = [assistant("x" * 200) for _ in range(40)]        # ~10 KB, one prompt, no tool use
    t = [user("The task line")] + filler + [assistant("Final outcome api_key=hunter2secret")]
    lines, bounded = se.read_transcript(transcript(tmp_path, t), 4096)
    assert bounded and all(json.loads(line) for line in lines)   # no torn line at either seam
    r = se.capture_segment(cfg, event(tmp_path, t, str(repo)), now=NOW)
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


BOUNDARY_TS = "2026-10-14T13:47:21.428Z"
SUMMARY = "This session is being continued from a previous conversation..."


def boundary(ts=BOUNDARY_TS, **top):
    return {"type": "system", "subtype": "compact_boundary", "content": "Conversation compacted",
            "timestamp": ts, "uuid": f"b-{next(_UUIDS)}", **top}


def test_task_comes_from_the_current_segment_not_the_compacted_past(cfg, repo, tmp_path):
    t = [user("old task from last week"), assistant("working", tool="Bash"), boundary(),
         user(SUMMARY, isCompactSummary=True), user("the new task"), assistant("done", tool="Edit")]
    r = se.capture_segment(cfg, event(tmp_path, t, str(repo)), now=NOW)
    assert r["decision"] == "written"
    body = split(Path(r["path"]).read_text(encoding="utf-8"))[1]
    assert "Task: the new task" in body and "# Session 2026-10-14 — the new task" in body
    assert "old task" not in body and "This session is being continued" not in body


def test_compact_summary_is_not_a_prompt(cfg, repo, tmp_path):
    t = [boundary(), user(SUMMARY, isCompactSummary=True), assistant("Understood.")]
    r = se.capture_segment(cfg, event(tmp_path, t, str(repo)), now=NOW)
    assert r["decision"] == "skip" and r["reason"] == "low signal: 0 prompts, no tool use, no decisions"
    assert not list((tmp_path / "mainframe" / "captures").rglob("*.md"))


def test_pre_compact_event_records_the_trigger_as_the_reason(cfg, repo, tmp_path):
    ev = event(tmp_path, REAL, str(repo))
    del ev["reason"]
    ev.update(hook_event_name="PreCompact", trigger="auto")
    r = se.capture_segment(cfg, ev, now=NOW)
    assert r["decision"] == "written"
    assert split(Path(r["path"]).read_text(encoding="utf-8"))[0]["reason"] == "pre-compact:auto"
    ev["session_id"] = "other-session"
    assert se.main(config=cfg, stdin=io.StringIO(json.dumps(ev)), now=NOW, hook="pre-compact") == 0
    line = (tmp_path / "mainframe" / "hook.log").read_text(encoding="utf-8").splitlines()[-1]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d pre-compact written project=proj .*", line)


def _fields_body(r):
    return split(Path(r["path"]).read_text(encoding="utf-8"))


def _files(cfg):
    return list((Lanes(cfg).mainframe_dir / "captures").rglob("*.md"))


def test_resume_after_a_capture_starts_after_its_through_record(cfg, repo, tmp_path):
    t = [user("first task"), assistant("working", tool="Bash")]
    a = se.capture_segment(cfg, event(tmp_path, t, str(repo)), now=NOW)
    assert a["decision"] == "written" and _fields_body(a)[0]["through"] == t[-1]["uuid"]
    t += [user("second task"), assistant("more", tool="Edit")]          # no boundary: a resumed session
    b = se.capture_segment(cfg, event(tmp_path, t, str(repo)), now=NOW)
    fields, body = _fields_body(b)
    assert b["decision"] == "written" and b["path"] != a["path"] and fields["through"] == t[-1]["uuid"]
    assert "Task: second task" in body and "first task" not in body
    assert se.capture_segment(cfg, event(tmp_path, t, str(repo)), now=NOW)["decision"] == "skip"
    assert len(_files(cfg)) == 2


def test_a_failed_compaction_loses_nothing_and_repeats_nothing(cfg, repo, tmp_path):
    def pre_compact(t):
        ev = event(tmp_path, t, str(repo))
        del ev["reason"]
        ev.update(hook_event_name="PreCompact", trigger="auto")
        return se.capture_segment(cfg, ev, now=NOW)
    t = [user("before compaction"), assistant("working", tool="Bash")]
    assert pre_compact(t)["decision"] == "written"
    t += [user("work after the failed compaction"), assistant("more", tool="Edit")]   # no boundary landed
    r = pre_compact(t)
    body = _fields_body(r)[1]
    assert r["decision"] == "written" and "Task: work after the failed compaction" in body
    assert "before compaction" not in body


def test_a_through_in_an_overflow_capture_is_honoured_too(cfg, repo, tmp_path):
    cfg["capture"]["daily_cap"] = 0                                     # every capture diverts to overflow/
    t = [user("first task"), assistant("working", tool="Bash")]
    a = se.capture_segment(cfg, event(tmp_path, t, str(repo)), now=NOW)
    assert a["decision"] == "overflow" and Path(a["path"]).parent == Lanes(cfg).overflow_dir("proj")
    assert se.capture_segment(cfg, event(tmp_path, t, str(repo)), now=NOW)["decision"] == "skip"
    t += [user("second task"), assistant("more", tool="Edit")]
    b = se.capture_segment(cfg, event(tmp_path, t, str(repo)), now=NOW)
    body = _fields_body(b)[1]
    assert b["decision"] == "overflow" and "Task: second task" in body and "first task" not in body


def test_meta_and_transcript_only_records_are_not_prompts(cfg, repo, tmp_path):
    caveat = "Caveat: The messages below were generated by the user while running local commands."
    t = [user(caveat, isMeta=True), user("shown only in the transcript", isVisibleInTranscriptOnly=True),
         user("the real task"), assistant("done", tool="Bash")]
    parsed = se.parse([json.dumps(e) for e in t])
    assert parsed["prompts"] == 1 and parsed["users"] == ["the real task"]
    body = _fields_body(se.capture_segment(cfg, event(tmp_path, t, str(repo)), now=NOW))[1]
    assert "Task: the real task" in body and "Caveat" not in body and "transcript" not in body


def test_cli_hook_takes_any_event_label_and_still_exits_zero(cfg, repo, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event(tmp_path, REAL, str(repo)))))
    assert cli.main(["hook", "no-such-event"], config=cfg) == 0
    assert len(list(Lanes(cfg).capture_dir("proj").glob("*.md"))) == 1
    assert " no-such-event written project=proj" in (tmp_path / "mainframe" / "hook.log").read_text(encoding="utf-8")


def _segment_file(tmp_path, pad, long_boundary):
    head = (json.dumps(user("x" * pad)) + "\n").encode()
    rec = boundary(content="c" * (300 if long_boundary else 0))
    line = (json.dumps(rec, separators=(",", ":")) + "\n").encode()
    tail = (json.dumps(assistant("L" * 400)) + "\n").encode()                  # several blocks, one line
    tail += (json.dumps(user('quoting "subtype":"compact_boundary" in a prompt')) + "\n").encode()
    tail += (json.dumps({"type": "user", "subtype": "compact_boundary"}, separators=(",", ":")) + "\n").encode()
    tail += (json.dumps(user("last")) + "\n").encode()
    p = tmp_path / "seg.jsonl"
    p.write_bytes(head + line + tail)
    return p, len(head) + len(line)


@pytest.mark.parametrize("long_boundary", [False, True])
def test_segment_start_finds_the_boundary_wherever_it_falls_in_a_block(tmp_path, monkeypatch, long_boundary):
    monkeypatch.setattr(se, "SCAN_BYTES", 64)
    for pad in range(70):
        p, expected = _segment_file(tmp_path, pad, long_boundary)
        with p.open("rb") as f:
            assert se.segment_start(f, p.stat().st_size) == expected, pad


def test_segment_start_ignores_decoys_and_returns_zero_without_a_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "SCAN_BYTES", 64)
    lines = [user('quoting "subtype":"compact_boundary" in a prompt'), assistant("L" * 400),
             {"type": "user", "subtype": "compact_boundary"}, user("last")]
    p = tmp_path / "none.jsonl"
    p.write_text("\n".join(json.dumps(e, separators=(",", ":")) for e in lines) + "\n", encoding="utf-8")
    with p.open("rb") as f:
        assert se.segment_start(f, p.stat().st_size) == 0
    p.write_bytes(b"")
    with p.open("rb") as f:
        assert se.segment_start(f, 0) == 0


def _offsets(tmp_path, recs, **dumps):
    """Write `recs` as JSONL; return the path and the offset just past each line."""
    lines = [(json.dumps(r, **dumps) + "\n").encode() for r in recs]
    p = tmp_path / "off.jsonl"
    p.write_bytes(b"".join(lines))
    return p, list(itertools.accumulate(len(x) for x in lines))


def _start(p, through=frozenset()):
    with p.open("rb") as f:
        return se.segment_start(f, p.stat().st_size, through)


def test_segment_start_finds_a_boundary_serialized_with_spaces(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "SCAN_BYTES", 64)
    for pad in range(0, 70, 7):
        p, ends = _offsets(tmp_path, [user("x" * pad), boundary(), user("tail"), assistant("L" * 300)])
        assert b'"subtype": "compact_boundary"' in p.read_bytes() and _start(p) == ends[1], pad


def test_segment_start_through_matches_a_record_uuid_not_quoted_text(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "SCAN_BYTES", 64)
    quoted = user('see {"uuid":"zz-target"} and "uuid" : "zz-target"')
    p, ends = _offsets(tmp_path, [user("first"), quoted, assistant("L" * 300), user("last")])
    assert _start(p, frozenset({"zz-target"})) == 0
    p, ends = _offsets(tmp_path, [user("first"), assistant("hit", uuid="zz-target"), quoted, user("last")])
    assert _start(p, frozenset({"zz-target"})) == ends[1]


def test_segment_start_the_later_of_boundary_and_through_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "SCAN_BYTES", 64)
    through = frozenset({"zz-target"})
    hit = assistant("hit", uuid="zz-target")
    p, ends = _offsets(tmp_path, [user("a"), boundary(), user("b"), hit, user("c")])
    assert _start(p, through) == ends[3]
    p, ends = _offsets(tmp_path, [user("a"), hit, user("b"), boundary(), user("c")])
    assert _start(p, through) == ends[3]


def test_oversize_segment_is_bounded_from_the_segment_head_not_the_file_head(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "SCAN_BYTES", 64)
    t = [user("ancient prompt"), boundary(), user("segment first prompt")] + \
        [assistant("x" * 200) for _ in range(40)] + [assistant("Final outcome")]
    lines, bounded = se.read_transcript(transcript(tmp_path, t), 4096)
    assert bounded
    assert "segment first prompt" in lines[0] and "Final outcome" in lines[-1]
    assert not any("ancient" in line or "compact_boundary" in line for line in lines)
    assert all(json.loads(line) for line in lines)


def test_hook_imports_stay_light_because_session_end_times_out_at_1_5_seconds():
    code = ("import sys\n"
            "import mainframe.adapters.cli, mainframe.adapters.capture_hook\n"
            "heavy = ['lancedb', 'pyarrow', 'torch', 'sentence_transformers', 'mcp',\n"
            "         'mainframe.service.daemon', 'mainframe.core.store']\n"
            "bad = [m for m in heavy if m in sys.modules]\n"
            "assert not bad, bad\n")
    src = str(Path(__file__).resolve().parents[2] / "src")
    env = {**os.environ, "PYTHONPATH": src}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_cli_pre_compact_subcommand_captures_the_segment(cfg, repo, tmp_path, monkeypatch):
    ev = event(tmp_path, REAL, str(repo))
    del ev["reason"]
    ev.update(hook_event_name="PreCompact", trigger="manual")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(ev)))
    assert cli.main(["hook", "pre-compact"], config=cfg) == 0
    files = list(Lanes(cfg).capture_dir("proj").glob("*.md"))
    assert len(files) == 1
    assert split(files[0].read_text(encoding="utf-8"))[0]["reason"] == "pre-compact:manual"
    assert "pre-compact written" in (tmp_path / "mainframe" / "hook.log").read_text(encoding="utf-8")
