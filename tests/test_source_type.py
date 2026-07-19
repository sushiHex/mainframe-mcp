"""§9.2 — source_type derivation: anchored session branch before /research/."""

from pathlib import Path


def test_research_sessions_path_is_session(mainframe):
    mf = mainframe
    repos = Path(mf.config["paths"]["repos_dir"])
    sess = repos / "p" / "research" / "sessions"
    sess.mkdir(parents=True)
    f = sess / "2026-06-27-000000-abcd1234-note.md"
    f.write_text(
        "# t\n\n## Summary\nQwen3-Embedding-8B on the RTX 3090 with bge-reranker.\n",
        encoding="utf-8",
    )
    assert mf.ingest_file(str(f))["source_type"] == "session"


def test_plain_research_path_is_research(mainframe):
    mf = mainframe
    repos = Path(mf.config["paths"]["repos_dir"])
    res = repos / "p" / "research"
    res.mkdir(parents=True)
    f = res / "bar.md"
    f.write_text(
        "# t\n\n## Summary\nCurated research on embeddings and reranking pipelines.\n",
        encoding="utf-8",
    )
    assert mf.ingest_file(str(f))["source_type"] == "research"
