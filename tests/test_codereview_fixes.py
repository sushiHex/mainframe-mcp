"""Regression tests for the code-review findings."""

import json
from pathlib import Path

import pytest


def _sec(n, body):
    return f"## Section {n}\n{body} this section has well over ten tokens so it stands alone as a chunk.\n"


def test_reingest_preserves_unchanged_sections(mainframe):
    """Finding #1: editing one section must not evict the unchanged ones."""
    mf = mainframe
    d = Path(mf.config["paths"]["repos_dir"]) / "p" / "research"
    d.mkdir(parents=True)
    doc = d / "analysis.md"
    doc.write_text("# Analysis\n\n" + _sec(1, "alpha") + "\n" + _sec(2, "beta") + "\n" + _sec(3, "gamma"), encoding="utf-8")
    assert mf.ingest_file(str(doc))["chunks"] == 3

    doc.write_text("# Analysis\n\n" + _sec(1, "alpha") + "\n" + _sec(2, "beta") + "\n" + _sec(3, "gamma EDITED distinct xyzzy plugh"), encoding="utf-8")
    mf.ingest_file(str(doc))

    texts = mf.store.table.to_pandas().query("doc_path == @s", local_dict={"s": str(doc)})["text"].tolist()
    assert any("Section 1" in t for t in texts), "section 1 was evicted"
    assert any("Section 2" in t for t in texts), "section 2 was evicted"
    assert any("Section 3" in t for t in texts)


def test_classify_source_type_anchors_and_fallback():
    """Findings #14 + scanner reuse: one classifier, fallback sessions are 'session'."""
    from mainframe_mcp.store import classify_source_type

    assert classify_source_type("C:/Users/x/.claude/mainframe/library/sessions/2026-01-01-x.md") == "session"
    assert classify_source_type("C:/Users/x/repos/p/research/sessions/foo.md") == "session"
    assert classify_source_type("C:/Users/x/repos/p/research/bar.md") == "research"
    assert classify_source_type("C:/Users/x/.claude/mainframe/library/foo.md") == "library"
    assert classify_source_type("C:/Users/x/.claude/mainframe/archive/a.md") == "archive"
    assert classify_source_type("C:/Users/x/repos/p/CLAUDE.md") == "project"
    assert classify_source_type("C:/Users/x/repos/p/docs/d.md") == "docs"


def test_get_stats_and_dedup_embeddings_populated(mainframe):
    """Findings #2 + #4: column-projected to_pandas raises on this LanceDB; both must work."""
    mf = mainframe
    d = Path(mf.config["paths"]["repos_dir"]) / "p" / "research"
    d.mkdir(parents=True)
    f = d / "x.md"
    f.write_text("# t\n\n## Summary\nQwen3 RTX 3090 bge reranker tokens chunks embeddings.\n", encoding="utf-8")
    mf.ingest_file(str(f))
    assert mf.store.get_stats()["total_chunks"] >= 1
    embs, ids = mf.store.get_embeddings_for_dedup(limit=500)
    assert len(embs) >= 1 and len(ids) == len(embs)


def test_capture_session_skips_when_capture_memory_already_called(tmp_path):
    """Finding #5: deterministic dedup — skip the backstop if the agent already captured."""
    from mainframe_mcp import session_capture as sc

    repos = tmp_path / "repos"
    (repos / "proj").mkdir(parents=True)
    sid = "feedface1234"
    lines = [
        {"type": "user", "isSidechain": False, "sessionId": sid, "cwd": str(repos / "proj"),
         "message": {"role": "user", "content": "do the thing"}},
        {"type": "assistant", "isSidechain": False, "sessionId": sid,
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "name": "mcp__mainframe__capture_memory", "input": {"title": "x", "content": "y"}}]}},
    ]
    t = tmp_path / "t.jsonl"
    t.write_text("\n".join(json.dumps(o) for o in lines), encoding="utf-8")
    assert sc.capture_session(str(t), repos_dir=str(repos), mainframe_dir=str(tmp_path / "mf")) is None


def test_build_frontmatter_sanitizes_injection():
    """Finding #11: a newline in a field must not break the YAML block."""
    from mainframe_mcp.session_capture import build_frontmatter

    fm = build_frontmatter("session\ninjected: pwned", "proj", "sid", "2026-01-01", "120000", "C:/x", "main", "capture_memory")
    assert fm.count("---") == 2
    assert "injected: pwned" not in fm
    assert fm.endswith("---\n")


def test_scrub_redacts_special_char_password():
    """Finding #12: KV value charset must cover password specials."""
    from mainframe_mcp.secrets import scrub_secrets

    out = scrub_secrets("password=P@ssw0rd!val")
    assert "P@ssw0rd!val" not in out
    assert "REDACTED" in out


def test_preset_still_applies_env_override(monkeypatch):
    """Finding #6: MAINFRAME_PRESET must not drop MAINFRAME_* env overrides."""
    from mainframe_mcp import config as cfg

    monkeypatch.setenv("MAINFRAME_PRESET", "gpu-max")
    monkeypatch.setenv("MAINFRAME_EMBEDDER_MODEL", "my/custom-model")
    c = cfg.load_config()
    assert c["embedder"]["model"] == "my/custom-model"
