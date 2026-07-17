"""§9.4 — delete_file keeps the manifest in sync (no ghost entry)."""

from pathlib import Path


def test_delete_file_syncs_manifest_and_allows_reingest(mainframe):
    mf = mainframe
    repos = Path(mf.config["paths"]["repos_dir"])
    (repos / "computers").mkdir(parents=True)

    result = mf.capture_memory(
        content="## Summary\nRTX 3090 Qwen3-Embedding-8B bge-reranker-v2-m3 256 token chunks.\n",
        title="delete test note",
        project="computers",
    )
    path = result["file"]

    # Chunks are present and the manifest tracks the file. Use the full
    # to_pandas() (column-projected reads can return empty — store gotcha).
    assert (mf.store.table.to_pandas()["doc_path"] == path).any()
    assert path in mf.manifest.data["files"]

    delete_result = mf.delete_file(path)
    assert delete_result["status"] == "deleted"

    # store chunks gone AND the manifest no longer holds the path (no ghost).
    remaining = mf.store.table.to_pandas()
    assert (remaining["doc_path"] != path).all()
    assert path not in mf.manifest.data["files"]

    # The on-disk file is untouched (delete only prunes the index). Re-ingesting
    # the identical file must re-index, not return "unchanged" (the ghost-entry
    # bug would have made needs_reindex False -> silently never re-indexed).
    assert Path(path).exists()
    reingest = mf.ingest_file(path)
    assert reingest["status"] == "ingested"
