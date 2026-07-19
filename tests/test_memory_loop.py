"""Memory-loop V1 (head-to-toe 2026-07-02): pending-session visibility +
per-project consolidation default. The loop-closer is a SIGNAL (pending counts
surfaced through sync_index/status) plus a friction-remover (topic defaults to
the project name), not an autonomous trigger — the caller stays in control."""

from pathlib import Path

import pytest

from conftest import FakeConsolidator, make_mainframe
from mainframe_mcp.session_capture import pending_sessions_by_project


@pytest.fixture
def mainframe(tmp_path):
    return make_mainframe(tmp_path)


def _seed_sessions(repos_dir, project, n):
    sdir = Path(repos_dir) / project / "research" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (sdir / f"2026-01-{i+1:02d}-000000-id{i}-note.md").write_text(
            f"---\ntype: session\n---\n# S{i}\n\n## Summary\nFact {i}.\n", encoding="utf-8")


def test_pending_sessions_by_project(mainframe):
    mf = mainframe
    repos = mf.config["paths"]["repos_dir"]
    _seed_sessions(repos, "alpha", 3)
    _seed_sessions(repos, "beta", 1)
    lib = Path(mf.config["paths"]["mainframe_dir"]) / "library" / "sessions"
    lib.mkdir(parents=True, exist_ok=True)
    (lib / "2026-01-01-000000-adhoc-x.md").write_text("# s\n\nbody\n", encoding="utf-8")

    pending = pending_sessions_by_project(repos, mf.config["paths"]["mainframe_dir"])
    assert pending == {"alpha": 3, "beta": 1, "library": 1}


def test_sync_index_surfaces_pending_and_hint(mainframe):
    mf = mainframe
    _seed_sessions(mf.config["paths"]["repos_dir"], "alpha", 5)  # >= threshold
    res = mf.sync_index()
    assert res["pending_sessions"] == {"alpha": 5}
    assert "alpha" in res["consolidation_hint"]
    assert "consolidate_sessions" in res["consolidation_hint"]


def test_sync_index_no_hint_below_threshold(mainframe):
    mf = mainframe
    _seed_sessions(mf.config["paths"]["repos_dir"], "alpha", 2)
    res = mf.sync_index()
    assert res["pending_sessions"] == {"alpha": 2}
    assert res.get("consolidation_hint") is None


def test_consolidate_topic_defaults_to_project(mainframe):
    mf = mainframe
    mf.consolidator = FakeConsolidator()
    _seed_sessions(mf.config["paths"]["repos_dir"], "alpha", 2)
    for f in (Path(mf.config["paths"]["repos_dir"]) / "alpha" / "research" / "sessions").glob("*.md"):
        mf.ingest_file(str(f))
    r = mf.consolidate_sessions(project="alpha")  # no topic
    assert r["status"] == "consolidated"
    assert Path(r["memory_file"]).name == "alpha.md"


def test_status_reports_pending_and_index_health(mainframe):
    mf = mainframe
    _seed_sessions(mf.config["paths"]["repos_dir"], "alpha", 2)
    s = mf.status()
    assert s["pending_sessions"] == {"alpha": 2}
    assert "reranker" in s
    ih = s["index_health"]
    assert ih["versions"] >= 1 and ih["fragments"] >= 0


def test_reload_config_patches_live_knobs(mainframe, monkeypatch):
    """Hot-reload must re-read config and patch the Store's live-read fields
    without touching resident models; model-section changes are reported as
    restart-required, not silently half-applied."""
    mf = mainframe
    import copy
    new_cfg = copy.deepcopy(mf.config)
    new_cfg["search"]["recency_weight"] = 0.3
    new_cfg["tiers"]["library"] = 0.5
    new_cfg["reranker"] = {"model": "some/other-model"}

    import mainframe_mcp.server as server_mod
    monkeypatch.setattr(server_mod, "load_config", lambda: new_cfg, raising=False)
    r = mf.reload_config()
    assert r["status"] == "reloaded"
    assert mf.store._recency_weight == 0.3
    assert mf.store.tier_boost["library"] == 0.5
    assert "reranker" in r["restart_required_for"]
