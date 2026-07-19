"""§9.1 — capture_memory round-trip (primary unit test)."""

from pathlib import Path


def test_capture_memory_roundtrip(mainframe):
    mf = mainframe
    repos = Path(mf.config["paths"]["repos_dir"])
    (repos / "computers").mkdir(parents=True)

    planted_token = "ghp_" + "A" * 36  # looks like a GitHub PAT
    content = (
        "## Summary\n"
        "Switched the embedder to Qwen3-Embedding-8B INT8 on the RTX 3090; the "
        "reranker is bge-reranker-v2-m3 and chunks are 256 tokens.\n\n"
        "## Decisions\n"
        f"- Keep 256-token chunks — pasted token {planted_token} for the deploy.\n"
    )

    result = mf.capture_memory(
        content=content, title="GPU embedder swap", project="computers"
    )

    # (b) return dict is ingest_file's shape
    assert result["status"] == "ingested"
    assert result["source_type"] == "session"
    assert result["chunks"] >= 1

    # (a) a uniquely-named file exists under research/sessions/
    sess_dir = repos / "computers" / "research" / "sessions"
    files = sorted(sess_dir.glob("*.md"))
    assert len(files) == 1
    assert str(files[0]).replace("\\", "/") == result["file"].replace("\\", "/")

    # (d) the secret-scrub strips the planted token
    written = files[0].read_text(encoding="utf-8")
    assert planted_token not in written
    assert "REDACTED" in written

    # (c) re-calling with new content writes a *second* file (never overwrite)
    result2 = mf.capture_memory(
        content=content + "\n## Open questions\n- anything left?\n",
        title="GPU embedder swap",
        project="computers",
    )
    files_after = sorted(sess_dir.glob("*.md"))
    assert len(files_after) == 2
    assert result2["file"] != result["file"]
    assert result2["status"] == "ingested"
