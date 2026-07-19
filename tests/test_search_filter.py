"""§9.3 — search default-excludes sessions; include_sessions=True opts in."""

from pathlib import Path

SHARED = "Qwen3 Embedding bge reranker RTX 3090 token chunks adaptive prefix hybrid"


def _seed(mf):
    repos = Path(mf.config["paths"]["repos_dir"])
    sess = repos / "p" / "research" / "sessions"
    sess.mkdir(parents=True)
    (sess / "2026-06-27-000000-abcd1234-s.md").write_text(
        f"# s\n\n## Summary\n{SHARED} session variant.\n", encoding="utf-8"
    )
    res = repos / "p" / "research"
    (res / "curated.md").write_text(
        f"# r\n\n## Summary\n{SHARED} curated variant.\n", encoding="utf-8"
    )
    for f in sorted(sess.glob("*.md")) + [res / "curated.md"]:
        mf.ingest_file(str(f))


def test_default_search_excludes_sessions(mainframe):
    mf = mainframe
    _seed(mf)
    hits = mf.search(SHARED, top_k=10)
    types = {h["source_type"] for h in hits}
    assert "session" not in types
    assert "research" in types


def test_include_sessions_returns_sessions(mainframe):
    mf = mainframe
    _seed(mf)
    hits = mf.search(SHARED, top_k=10, include_sessions=True)
    types = {h["source_type"] for h in hits}
    assert "session" in types
