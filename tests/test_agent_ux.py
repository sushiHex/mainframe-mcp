"""Agent-UX + memory-loop batch (10-POV review Tiers 1-3, 2026-07-01):
limit actually caps results, candidate pool decoupled from output size,
truncation flag + char offsets, empty/low-confidence hints at the MCP layer,
file_path alias, days=0 = all pending sessions, consolidation bloat guardrail."""

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

import mainframe_mcp.server as server_mod
from conftest import FakeConsolidator, make_mainframe


@pytest.fixture
def mainframe(tmp_path):
    return make_mainframe(tmp_path)


def _seed_docs(mf, n=6):
    d = Path(mf.config["paths"]["repos_dir"]) / "proj" / "docs"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        f = d / f"doc{i}.md"
        f.write_text(f"# Doc {i}\n\nQwen3 reranker hybrid search chunks topic{i} facts.\n",
                     encoding="utf-8")
        mf.ingest_file(str(f))


# ---------- limit caps results; pool is decoupled ----------

def test_limit_actually_caps_results(mainframe):
    mf = mainframe
    _seed_docs(mf, 6)
    assert len(mf.search("Qwen3 reranker hybrid search", top_k=1)) == 1
    assert len(mf.search("Qwen3 reranker hybrid search", top_k=2)) == 2
    assert len(mf.search("Qwen3 reranker hybrid search", top_k=5)) == 5


def test_candidate_pool_floor_decoupled_from_limit(mainframe, monkeypatch):
    mf = mainframe
    _seed_docs(mf, 2)
    seen = {}
    orig = mf.store.search

    def spy(qe, top_k, **kw):
        seen["pool"] = top_k
        return orig(qe, top_k=top_k, **kw)

    monkeypatch.setattr(mf.store, "search", spy)
    mf.search("Qwen3 reranker hybrid", top_k=3)
    pool_floor = mf.config["search"].get("candidate_pool", 20)
    assert seen["pool"] >= pool_floor, \
        "a small limit must not starve the reranker's candidate pool"


# ---------- truncation flag + char offsets ----------

def test_truncation_flag_and_offsets(mainframe):
    mf = mainframe
    d = Path(mf.config["paths"]["repos_dir"]) / "proj" / "docs"
    d.mkdir(parents=True, exist_ok=True)
    long_body = "Qwen3 reranker hybrid search facts. " + ("verbose filler sentence here. " * 40)
    (d / "long.md").write_text(f"# Long\n\n{long_body}\n", encoding="utf-8")
    mf.ingest_file(str(d / "long.md"))
    res = mf.search("Qwen3 reranker hybrid search facts", top_k=1)
    assert res and res[0]["truncated"] is True
    assert len(res[0]["text"]) == 500
    assert isinstance(res[0]["char_start"], int) and isinstance(res[0]["char_end"], int)

    (d / "short.md").write_text("# Short\n\nQwen3 embedder distinct tiny note.\n", encoding="utf-8")
    mf.ingest_file(str(d / "short.md"))
    res = mf.search("Qwen3 embedder distinct tiny note", top_k=1)
    assert res and res[0]["truncated"] is False


# ---------- MCP dispatch: hints + file_path alias ----------

def _dispatch(name, args):
    return asyncio.run(server_mod.call_tool(name, args))[0].text


def test_search_empty_result_returns_hint(mainframe, monkeypatch):
    monkeypatch.setattr(server_mod, "mainframe", mainframe)  # empty index
    out = _dispatch("search", {"query": "zzz nothing matches this"})
    payload = json.loads(out)  # structured JSON, not prose-appended
    assert payload["results"] == [] and payload["confidence"] == "none"
    assert "specific technical terms" in payload["guidance"].lower()


def test_ingest_file_accepts_both_param_spellings(mainframe, monkeypatch):
    mf = mainframe
    monkeypatch.setattr(server_mod, "mainframe", mf)
    d = Path(mf.config["paths"]["repos_dir"]) / "proj" / "docs"
    d.mkdir(parents=True, exist_ok=True)
    f1 = d / "a.md"
    f1.write_text("# A\n\nalpha content for ingest here.\n", encoding="utf-8")
    f2 = d / "b.md"
    f2.write_text("# B\n\nbeta content for ingest here.\n", encoding="utf-8")
    assert "ingested" in _dispatch("ingest_file", {"file_path": str(f1)})
    assert "ingested" in _dispatch("ingest_file", {"filePath": str(f2)})  # legacy alias


# ---------- days=0 consolidates ALL pending sessions ----------

def _seed_old_session(mf, age_days=100):
    sdir = Path(mf.config["paths"]["repos_dir"]) / "proj" / "research" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    f = sdir / "2026-01-01-000000-old1-note.md"
    f.write_text("---\ntype: session\n---\n# Old\n\n## Summary\nAncient decision recorded.\n",
                 encoding="utf-8")
    mf.ingest_file(str(f))
    old = time.time() - age_days * 86400
    os.utime(f, (old, old))
    return f


def test_days_zero_means_all_pending(mainframe):
    mf = mainframe
    mf.consolidator = FakeConsolidator()
    _seed_old_session(mf, age_days=100)
    # a 30-day window must NOT see the 100-day-old session...
    r = mf.consolidate_sessions(project="proj", topic="old stuff", days=30)
    assert r["status"] == "no_sessions"
    # ...but days=0 (the new default: all pending) must consolidate it
    r = mf.consolidate_sessions(project="proj", topic="old stuff", days=0)
    assert r["status"] == "consolidated"


# ---------- consolidation bloat guardrail (B) ----------

class BloatingConsolidator(FakeConsolidator):
    """Models runaway accretion: restates the whole prior note twice + padding
    (multiplicative growth — the failure mode the 2x guardrail targets)."""

    def consolidate(self, reports, topic="", existing=""):
        super().consolidate(reports, topic, existing)
        body = "## Facts\n\nRestated everything verbosely again. " * 30
        return (existing + "\n\n" + existing + "\n\n" + body) if existing else body


def test_reconsolidation_flags_note_bloat(mainframe):
    mf = mainframe
    mf.consolidator = BloatingConsolidator()
    sdir = Path(mf.config["paths"]["repos_dir"]) / "proj" / "research" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    for i in (1, 2):
        f = sdir / f"2026-02-0{i}-000000-id{i}-note.md"
        f.write_text(f"---\ntype: session\n---\n# S{i}\n\n## Summary\nFact {i} decided here.\n",
                     encoding="utf-8")
        mf.ingest_file(str(f))
        r = mf.consolidate_sessions(project="proj", topic="bloat check", days=0)
        assert r["status"] == "consolidated"
    # second run had a prior note and a wildly larger merge -> must be flagged
    assert r["note_bloat_warning"] is True


def test_search_limit_clamped_before_pool_sizing(mainframe, monkeypatch):
    """limit=100000 must not size the LanceDB fetch to the corpus (schema 'max 10'
    is advisory only — the clamp must live in code)."""
    mf = mainframe
    _seed_docs(mf, 3)
    seen = {}
    orig = mf.store.search

    def spy(qe, top_k, **kw):
        seen["pool"] = top_k
        return orig(qe, top_k=top_k, **kw)

    monkeypatch.setattr(mf.store, "search", spy)
    res = mf.search("Qwen3 reranker hybrid search", top_k=100000)
    assert len(res) <= server_mod.MAX_SEARCH_RESULTS
    fetch_mult = mf.config["search"].get("fetch_multiplier", 2)
    assert seen["pool"] <= max(mf.config["search"]["candidate_pool"],
                               server_mod.MAX_SEARCH_RESULTS * fetch_mult)


def test_file_path_not_hard_required_in_schema():
    """Hard-requiring file_path would make the MCP SDK reject legacy filePath
    callers at schema validation, before dispatch's alias handling ever runs."""
    tools = {t.name: t for t in asyncio.run(server_mod.list_tools())}
    for name in ("ingest_file", "delete_file"):
        schema = tools[name].inputSchema
        assert not schema.get("required"), f"{name} must not hard-require file_path"
        assert "filePath" in schema["properties"]  # legacy alias stays declared


def test_consolidate_budget_packs_sources(mainframe, monkeypatch):
    """Sources beyond the char budget stay PENDING (not archived/pruned) instead
    of being tail-truncated out of the merge and then archived unseen."""
    mf = mainframe
    mf.consolidator = FakeConsolidator()
    sdir = Path(mf.config["paths"]["repos_dir"]) / "proj" / "research" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    files = []
    for i in (1, 2):
        f = sdir / f"2026-04-0{i}-000000-id{i}-note.md"
        f.write_text(f"---\ntype: session\n---\n# S{i}\n\n## Summary\n"
                     f"Long recorded fact number {i} with plenty of padding text here.\n",
                     encoding="utf-8")
        mf.ingest_file(str(f))
        files.append(f)
    monkeypatch.setattr(server_mod, "CONSOLIDATION_CHAR_BUDGET", 50)  # < one file

    r1 = mf.consolidate_sessions(project="proj", topic="budget", days=0)
    assert r1["status"] == "consolidated"
    assert r1["pending"] == 1
    assert r1["truncated_sources"] == 1  # file1 alone exceeds the tiny budget -> clamped visibly
    assert not files[0].exists()          # consolidated + archived
    assert files[1].exists(), "over-budget source must stay pending on disk"
    df = mf.store.table.to_pandas()
    assert (df["doc_path"] == str(files[1])).any(), "pending source must stay indexed"

    r2 = mf.consolidate_sessions(project="proj", topic="budget", days=0)
    assert r2["status"] == "consolidated" and r2["pending"] == 0
    assert not files[1].exists()          # picked up by the next run


def test_normal_reconsolidation_not_flagged(mainframe):
    mf = mainframe
    mf.consolidator = FakeConsolidator()  # returns compact merges
    sdir = Path(mf.config["paths"]["repos_dir"]) / "proj" / "research" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    for i in (1, 2):
        f = sdir / f"2026-03-0{i}-000000-id{i}-note.md"
        f.write_text(f"---\ntype: session\n---\n# S{i}\n\n## Summary\nCompact fact {i}.\n",
                     encoding="utf-8")
        mf.ingest_file(str(f))
        r = mf.consolidate_sessions(project="proj", topic="lean check", days=0)
        assert r["status"] == "consolidated"
    assert r["note_bloat_warning"] is False
