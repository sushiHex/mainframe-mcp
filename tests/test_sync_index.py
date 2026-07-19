"""Phase 2 — sync_index: auto-ingest new/modified + prune deleted files."""

from pathlib import Path


def test_sync_index_ingests_new_and_prunes_deleted(mainframe):
    mf = mainframe
    repos = Path(mf.config["paths"]["repos_dir"])
    sdir = repos / "proj" / "research" / "sessions"
    sdir.mkdir(parents=True)
    sess = sdir / "2026-01-01-000000-abcd1234-note.md"
    sess.write_text("# t\n\n## Summary\nQwen3 RTX 3090 bge reranker chunks session note.\n", encoding="utf-8")
    curated = repos / "proj" / "research" / "curated.md"
    curated.write_text("# c\n\n## Summary\nCurated research about embeddings and reranking pipelines.\n", encoding="utf-8")

    r1 = mf.sync_index()
    assert r1["ingested"] >= 2 and r1["pruned"] == 0
    df = mf.store.table.to_pandas()
    assert (df["doc_path"] == str(sess)).any()
    assert str(sess) in mf.manifest.data["files"]

    # delete the session file from disk -> next sync prunes it from the index
    sess.unlink()
    r2 = mf.sync_index()
    assert r2["pruned"] >= 1
    df2 = mf.store.table.to_pandas()
    assert not (df2["doc_path"] == str(sess)).any(), "deleted file's chunks not pruned"
    assert str(sess) not in mf.manifest.data["files"], "deleted file's manifest entry not pruned"
    # the curated file that still exists must NOT be pruned
    assert str(curated) in mf.manifest.data["files"]


def test_sync_index_is_idempotent(mainframe):
    mf = mainframe
    repos = Path(mf.config["paths"]["repos_dir"])
    f = repos / "proj" / "research" / "a.md"
    f.parent.mkdir(parents=True)
    f.write_text("# a\n\n## Summary\nSome curated content about LanceDB and chunking strategies.\n", encoding="utf-8")
    mf.sync_index()
    r2 = mf.sync_index()
    assert r2["ingested"] == 0 and r2["pruned"] == 0
    assert r2["unchanged"] >= 1


def test_diff_files_prunes_only_truly_missing(mainframe):
    """A manifest entry is 'deleted' iff its file is gone — not merely absent
    from the scanned subset (so manually-ingested files aren't wrongly pruned)."""
    from mainframe_mcp.watcher import diff_files

    mf = mainframe
    repos = Path(mf.config["paths"]["repos_dir"])
    f = repos / "proj" / "research" / "b.md"
    f.parent.mkdir(parents=True)
    f.write_text("# b\n\n## Summary\nContent here.\n", encoding="utf-8")
    mf.ingest_file(str(f))

    # f exists but is NOT in the passed file list -> must NOT be flagged deleted
    diff = diff_files([], mf.manifest)
    assert str(f) not in diff["deleted"]

    f.unlink()
    diff2 = diff_files([], mf.manifest)
    assert str(f) in diff2["deleted"]
