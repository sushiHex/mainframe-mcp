"""MCP-ergonomics pass (2026-07-17, from the Hermes/ChatGPT bake-off feedback):
make the server a clean, portable MCP citizen — not Claude-Code-coupled — for
any always-on multi-agent gateway sharing the GPU.

- MAINFRAME_CONFIG: portable config path override (item 2)
- MAINFRAME_PREWARM: GPU prewarm is opt-in, lazy by default (item 3)
- MAINFRAME_READ_ONLY: expose only recall tools, reject mutations (item 4)
- line_start/line_end citations in search results (item 5)
- structured JSON search response, no appended prose (item 6)
- AGENTS.md / HERMES.md / .hermes.md as root context files; NOT SOUL.md (item 7)
"""

import asyncio
import json
from pathlib import Path

import pytest

import mainframe_mcp.server as server_mod
from conftest import make_mainframe
from mainframe_mcp import config as config_mod
from mainframe_mcp.scanner import ROOT_CONTEXT_FILES, scan_projects
from mainframe_mcp.store import classify_source_type


@pytest.fixture
def mainframe(tmp_path):
    return make_mainframe(tmp_path)


def _dispatch(name, args):
    return asyncio.run(server_mod.call_tool(name, args))[0].text


# ---------- item 2: MAINFRAME_CONFIG ----------

def test_mainframe_config_env_overrides_path(tmp_path, monkeypatch):
    cfg = tmp_path / "custom.json"
    cfg.write_text(json.dumps({"tiers": {"library": 0.42}}), encoding="utf-8")
    monkeypatch.setenv("MAINFRAME_CONFIG", str(cfg))
    loaded = config_mod.load_config()  # no explicit path -> env wins
    assert loaded["tiers"]["library"] == 0.42


def test_mainframe_config_absent_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("MAINFRAME_CONFIG", raising=False)
    # must not raise and must return a populated default config
    loaded = config_mod.load_config()
    assert "embedder" in loaded and "paths" in loaded


def test_explicit_path_still_wins_over_env(tmp_path, monkeypatch):
    envcfg = tmp_path / "env.json"
    envcfg.write_text(json.dumps({"tiers": {"library": 0.1}}), encoding="utf-8")
    argcfg = tmp_path / "arg.json"
    argcfg.write_text(json.dumps({"tiers": {"library": 0.9}}), encoding="utf-8")
    monkeypatch.setenv("MAINFRAME_CONFIG", str(envcfg))
    loaded = config_mod.load_config(config_path=argcfg)
    assert loaded["tiers"]["library"] == 0.9  # explicit arg beats env


# ---------- item 3: opt-in prewarm ----------

def test_prewarm_off_by_default(monkeypatch):
    monkeypatch.delenv("MAINFRAME_PREWARM", raising=False)
    assert server_mod._prewarm_enabled() is False


def test_prewarm_opt_in(monkeypatch):
    for v in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("MAINFRAME_PREWARM", v)
        assert server_mod._prewarm_enabled() is True
    monkeypatch.setenv("MAINFRAME_PREWARM", "0")
    assert server_mod._prewarm_enabled() is False


# ---------- item 4: read-only mode ----------

def test_read_only_filters_tool_list(monkeypatch):
    monkeypatch.setenv("MAINFRAME_READ_ONLY", "1")
    tools = asyncio.run(server_mod.list_tools())
    names = {t.name for t in tools}
    assert names == server_mod.READ_ONLY_TOOLS
    assert "delete_file" not in names and "consolidate_sessions" not in names


def test_full_tool_list_by_default(monkeypatch):
    monkeypatch.delenv("MAINFRAME_READ_ONLY", raising=False)
    tools = asyncio.run(server_mod.list_tools())
    names = {t.name for t in tools}
    assert "delete_file" in names and "search" in names and len(names) >= 11


def test_read_only_rejects_mutation(mainframe, monkeypatch):
    monkeypatch.setattr(server_mod, "mainframe", mainframe)
    monkeypatch.setenv("MAINFRAME_READ_ONLY", "1")
    out = _dispatch("delete_file", {"file_path": "C:/whatever.md"})
    assert "read-only" in out.lower()
    # a read tool still works in read-only mode
    out2 = _dispatch("search", {"query": "anything"})
    assert "results" in out2


# ---------- item 5: line-number citations ----------

def test_search_returns_line_spans(mainframe):
    mf = mainframe
    d = Path(mf.config["paths"]["repos_dir"]) / "proj" / "docs"
    d.mkdir(parents=True, exist_ok=True)
    # heading on line 1, blank line 2, the target sentence on line 3
    f = d / "lines.md"
    f.write_text("# Title\n\nQwen3 reranker distinct citation sentence here.\n", encoding="utf-8")
    mf.ingest_file(str(f))
    res = mf.search("Qwen3 reranker distinct citation sentence", top_k=1)
    assert res
    r = res[0]
    assert r["line_start"] >= 1 and r["line_end"] >= r["line_start"]
    # the chunk's char span must map to the real line it starts on
    text = f.read_text(encoding="utf-8")
    assert r["line_start"] == text.count("\n", 0, r["char_start"]) + 1


def test_line_spans_survive_missing_file(mainframe):
    """A vanished/moved source file must not crash search — line fields are
    simply omitted (char_start/char_end are still returned)."""
    mf = mainframe
    d = Path(mf.config["paths"]["repos_dir"]) / "proj" / "docs"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "gone.md"
    f.write_text("# G\n\nEphemeral unique lancedb fragment content here.\n", encoding="utf-8")
    mf.ingest_file(str(f))
    f.unlink()
    res = mf.search("ephemeral unique lancedb fragment content", top_k=1)
    assert res and "char_start" in res[0]
    assert "line_start" not in res[0]


# ---------- item 6: structured JSON response ----------

def test_search_response_is_valid_json(mainframe, monkeypatch):
    monkeypatch.setattr(server_mod, "mainframe", mainframe)  # empty index
    out = _dispatch("search", {"query": "zzz nothing here at all"})
    payload = json.loads(out)  # must parse — no appended prose
    assert payload["results"] == []
    assert payload["confidence"] == "none"
    assert "guidance" in payload and "technical terms" in payload["guidance"].lower()


def test_search_response_hit_has_no_guidance(mainframe, monkeypatch):
    mf = mainframe
    monkeypatch.setattr(server_mod, "mainframe", mf)
    d = Path(mf.config["paths"]["repos_dir"]) / "proj" / "docs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "h.md").write_text("# H\n\ntantivy hybrid reranker qwen chunk fact.\n", encoding="utf-8")
    mf.ingest_file(str(d / "h.md"))
    payload = json.loads(_dispatch("search", {"query": "tantivy hybrid reranker qwen chunk"}))
    assert payload["results"]
    assert payload["confidence"] in ("high", "low")


# ---------- item 7: agent context files ----------

def test_agent_context_files_classify_as_project():
    for name in ("CLAUDE.md", "AGENTS.md", "HERMES.md", ".hermes.md"):
        assert classify_source_type(f"C:/repos/x/{name}") == "project", name
    assert "SOUL.md" not in ROOT_CONTEXT_FILES  # identity files are not evidence


def test_scanner_finds_agent_context_files_not_soul(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "AGENTS.md").write_text("# agents\n\nbuild rules here.\n", encoding="utf-8")
    (proj / "HERMES.md").write_text("# hermes\n\ngateway notes.\n", encoding="utf-8")
    (proj / ".hermes.md").write_text("# dot hermes\n\nmore.\n", encoding="utf-8")
    (proj / "SOUL.md").write_text("# soul\n\nidentity — must NOT index.\n", encoding="utf-8")
    found = {Path(f["path"]).name for f in scan_projects(repos_dir=tmp_path)}
    assert {"AGENTS.md", "HERMES.md", ".hermes.md"} <= found
    assert "SOUL.md" not in found
