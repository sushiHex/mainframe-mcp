"""Regression tests for the second code-review round."""

import json
from pathlib import Path

from mainframe_mcp.store import classify_source_type
from mainframe_mcp.session_capture import _FILE_PATH_RE


def test_machines_playbooks_classify_as_docs():
    assert classify_source_type("C:/repos/x/machines/m.md") == "docs"
    assert classify_source_type("C:/repos/x/playbooks/p.md") == "docs"


def test_classify_respects_custom_mainframe_dir():
    md = "C:/Users/x/.knowledge"  # relocated, no "mainframe" in the name
    assert classify_source_type(md + "/library/sessions/s.md", md) == "session"
    assert classify_source_type(md + "/library/foo.md", md) == "library"
    assert classify_source_type(md + "/archive/a.md", md) == "archive"
    # default-name path still works without the arg
    assert classify_source_type("C:/Users/x/.claude/mainframe/library/sessions/s.md") == "session"


def test_file_path_re_handles_dotted_dirs_and_multipath():
    s = r"edited C:\Users\dev\.claude\hooks\capture_session.py for the fix"
    assert _FILE_PATH_RE.findall(s) == [r"C:\Users\dev\.claude\hooks\capture_session.py"]
    s2 = r"see C:\a\one.py and C:\b\two.md done"
    assert _FILE_PATH_RE.findall(s2) == [r"C:\a\one.py", r"C:\b\two.md"]


def test_ingest_paths_counts_all_chunks_duplicate(mainframe):
    mf = mainframe
    d = Path(mf.config["paths"]["repos_dir"]) / "p" / "research"
    d.mkdir(parents=True)
    body = "# t\n\n## Summary\nIdentical content about Qwen3 bge reranker chunks here.\n"
    (d / "a.md").write_text(body, encoding="utf-8")
    (d / "b.md").write_text(body, encoding="utf-8")  # identical -> b is all-dup of a
    mf.ingest_file(str(d / "a.md"))
    assert mf.ingest_file(str(d / "b.md"))["status"] == "all_chunks_duplicate"
    # _ingest_paths must count a (unchanged) AND b (all_chunks_duplicate) as skipped
    ingested, skipped, errors = mf._ingest_paths([str(d / "a.md"), str(d / "b.md")])
    assert ingested == 0 and skipped == 2 and errors == []


def test_manifest_preserves_first_ingested_at(tmp_path):
    from mainframe_mcp.manifest import Manifest

    m = Manifest(tmp_path)
    m.record_ingest("f.md", "h1", 1)
    first = m.data["files"]["f.md"]["ingested_at"]
    m.record_ingest("f.md", "h2", 2)  # re-ingest
    assert m.data["files"]["f.md"]["ingested_at"] == first
    assert m.data["files"]["f.md"]["content_hash"] == "h2"


def test_manifest_remove_many(tmp_path):
    from mainframe_mcp.manifest import Manifest

    m = Manifest(tmp_path)
    m.record_ingest("a.md", "h", 1)
    m.record_ingest("b.md", "h", 1)
    m.remove_many(["a.md", "b.md", "missing.md"])
    assert "a.md" not in m.data["files"] and "b.md" not in m.data["files"]


def test_capture_session_skips_subagent_capture_memory(tmp_path):
    from mainframe_mcp import session_capture as sc

    repos = tmp_path / "repos"
    (repos / "proj").mkdir(parents=True)
    sid = "cafef00d1234"
    lines = [
        {"type": "user", "isSidechain": False, "sessionId": sid, "cwd": str(repos / "proj"),
         "message": {"role": "user", "content": "do the work"}},
        # a SUBAGENT (sidechain) calls capture_memory — must still suppress the backstop
        {"type": "assistant", "isSidechain": True, "sessionId": sid,
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "name": "mcp__mainframe__capture_memory", "input": {}}]}},
    ]
    t = tmp_path / "t.jsonl"
    t.write_text("\n".join(json.dumps(o) for o in lines), encoding="utf-8")
    assert sc.capture_session(str(t), repos_dir=str(repos), mainframe_dir=str(tmp_path / "mf")) is None
