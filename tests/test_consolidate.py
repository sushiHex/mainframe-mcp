"""Phase 3 — consolidate_sessions: merge raw session logs into a curated note."""

from pathlib import Path

from conftest import FakeConsolidator


def _seed_sessions(mf, n=2):
    repos = Path(mf.config["paths"]["repos_dir"])
    sdir = repos / "proj" / "research" / "sessions"
    sdir.mkdir(parents=True)
    files = []
    for i in range(n):
        f = sdir / f"2026-01-0{i+1}-000000-id{i}-note.md"
        f.write_text(
            f"---\ntype: session\n---\n# S{i}\n\n## Summary\nDecision {i}: Qwen3 bge reranker fact {i}.\n",
            encoding="utf-8",
        )
        mf.ingest_file(str(f))  # in the index as session
        files.append(f)
    return files


def test_consolidate_sessions_writes_memory_and_archives_sources(mainframe):
    mf = mainframe
    mf.consolidator = FakeConsolidator()  # pre-set so the lazy GPU load is skipped
    sources = _seed_sessions(mf, 2)

    res = mf.consolidate_sessions(project="proj", topic="reranker decisions",
                                  days=3650, archive_sources=True)

    assert res["status"] == "consolidated" and res["sources"] == 2

    repos = Path(mf.config["paths"]["repos_dir"])
    notes = list((repos / "proj" / "research" / "memory").glob("*.md"))
    assert len(notes) == 1, "memory note not written under research/memory/"

    df = mf.store.table.to_pandas()
    note_rows = df[df["doc_path"] == str(notes[0])]
    assert len(note_rows) >= 1, "memory note not ingested"
    assert note_rows["source_type"].iloc[0] == "research", "memory note should be curated research"

    # raw sources pruned from the index AND moved to the archive (provenance)
    for s in sources:
        assert not (df["doc_path"] == str(s)).any(), "session chunks not pruned"
        assert not s.exists(), "session file not archived/removed"
    archive = Path(mf.config["paths"]["mainframe_dir"]) / "archive"
    assert len(list(archive.glob("*.md"))) == 2


def test_consolidate_sessions_no_sessions_is_noop(mainframe):
    mf = mainframe
    mf.consolidator = FakeConsolidator()
    (Path(mf.config["paths"]["repos_dir"]) / "proj").mkdir(parents=True)
    res = mf.consolidate_sessions(project="proj", topic="x", days=30)
    assert res["status"] == "no_sessions" and res["sources"] == 0


def test_consolidate_keeps_sources_when_note_ingest_fails(mainframe, monkeypatch):
    mf = mainframe
    mf.consolidator = FakeConsolidator()
    sources = _seed_sessions(mf, 2)

    orig = mf.ingest_file

    def fake_ingest(path):
        if "/memory/" in str(path).replace("\\", "/"):
            return {"status": "error", "error": "No content to chunk", "chunks": 0,
                    "source_type": "research"}
        return orig(path)

    monkeypatch.setattr(mf, "ingest_file", fake_ingest)
    res = mf.consolidate_sessions(project="proj", topic="t", days=3650)

    assert res["status"] != "consolidated"  # failure surfaced, not silently 'consolidated'
    # sources must NOT be pruned/archived when the note never indexed
    df = mf.store.table.to_pandas()
    for s in sources:
        assert s.exists(), "source removed despite failed note ingest"
        assert (df["doc_path"] == str(s)).any(), "source pruned despite failed note ingest"


def test_consolidate_archive_collision_keeps_both(mainframe):
    mf = mainframe
    mf.consolidator = FakeConsolidator()
    # two sessions in different projects with the SAME filename
    repos = Path(mf.config["paths"]["repos_dir"])
    name = "2026-01-01-000000-dup-note.md"
    paths = []
    for proj in ("a", "b"):
        d = repos / proj / "research" / "sessions"
        d.mkdir(parents=True)
        f = d / name
        f.write_text("---\ntype: session\n---\n# s\n\n## Summary\nfact.\n", encoding="utf-8")
        mf.ingest_file(str(f))
        paths.append(f)
    # consolidate all repos (project=None) -> both same-named files archived
    mf.consolidate_sessions(project=None, topic="dup test", days=3650, archive_sources=True)
    archive = Path(mf.config["paths"]["mainframe_dir"]) / "archive"
    assert len(list(archive.glob("*.md"))) == 2, "archive collision dropped a file"


def test_consolidate_sessions_delete_mode_removes_files(mainframe):
    mf = mainframe
    mf.consolidator = FakeConsolidator()
    sources = _seed_sessions(mf, 1)
    res = mf.consolidate_sessions(project="proj", topic="t", days=3650, archive_sources=False)
    assert res["status"] == "consolidated"
    assert not sources[0].exists()  # deleted, not archived
    archive = Path(mf.config["paths"]["mainframe_dir"]) / "archive"
    assert not archive.exists() or not list(archive.glob("*.md"))
