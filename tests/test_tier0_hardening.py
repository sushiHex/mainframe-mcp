"""Tier-0 ops/security hardening (10-POV review, 2026-07-01):
read-only search, atomic manifest, poison-file resilience, optimize wiring,
secret-scrub gaps, consolidation-output scrub, path-traversal validation."""

from pathlib import Path

from conftest import FakeConsolidator, make_mainframe
from mainframe_mcp.manifest import Manifest
from mainframe_mcp.secrets import REDACTED, scrub_secrets

import pytest


@pytest.fixture
def mainframe(tmp_path):
    return make_mainframe(tmp_path)


def _seed_doc(mf, name="doc.md", proj="proj"):
    d = Path(mf.config["paths"]["repos_dir"]) / proj / "docs"
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_text("# Doc\n\nQwen3 reranker hybrid search chunks embeddings facts.\n", encoding="utf-8")
    mf.ingest_file(str(f))
    return f


# ---------- read-only search hot path ----------

def test_search_is_read_only(mainframe, monkeypatch):
    mf = mainframe
    _seed_doc(mf)
    writes = {"table_update": 0, "manifest_save": 0}
    monkeypatch.setattr(mf.store.table, "update",
                        lambda *a, **k: writes.__setitem__("table_update", writes["table_update"] + 1),
                        raising=False)
    monkeypatch.setattr(mf.manifest, "save",
                        lambda: writes.__setitem__("manifest_save", writes["manifest_save"] + 1))
    for _ in range(3):
        assert mf.search("Qwen3 reranker hybrid")
    assert writes == {"table_update": 0, "manifest_save": 0}, \
        "search must not write to LanceDB or rewrite the manifest per query"


def test_search_flushes_manifest_periodically(mainframe, monkeypatch):
    mf = mainframe
    _seed_doc(mf)
    saves = []
    monkeypatch.setattr(mf.manifest, "save", lambda: saves.append(1))
    for _ in range(25):
        mf.search("Qwen3 reranker hybrid")
    assert len(saves) == 1  # analytics still persisted, just batched


# ---------- atomic manifest + corruption recovery ----------

def test_manifest_survives_corruption(tmp_path):
    m = Manifest(tmp_path)
    m.record_ingest("C:/a.md", "h1", 1)
    m.record_ingest("C:/b.md", "h2", 2)  # second save rotates a .bak (holds a.md)
    (tmp_path / ".manifest.json").write_text("{truncated garbage", encoding="utf-8")
    m2 = Manifest(tmp_path)  # must NOT raise
    assert "C:/a.md" in m2.data["files"], "should recover from .bak"
    # no bak at all + corrupt main -> empty default, still no crash
    (tmp_path / ".manifest.json").write_text("nope", encoding="utf-8")
    (tmp_path / ".manifest.json.bak").unlink()
    m3 = Manifest(tmp_path)
    assert m3.data == {"files": {}, "version": 1}


# ---------- poison file must not abort a batch ----------

def test_ingest_paths_survives_poison_file(mainframe, monkeypatch):
    mf = mainframe
    d = Path(mf.config["paths"]["repos_dir"]) / "proj" / "docs"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(3):
        f = d / f"d{i}.md"
        f.write_text(f"# D{i}\n\nunique content {i} for chunking here.\n", encoding="utf-8")
        paths.append(str(f))
    orig = mf.ingest_file

    def boom(path, **kw):  # mirrors ingest_file(path, donor_cache=...)
        if "d1.md" in path:
            raise RuntimeError("CUDA error: device-side assert")
        return orig(path, **kw)

    monkeypatch.setattr(mf, "ingest_file", boom)
    ingested, skipped, errors = mf._ingest_paths(paths)
    assert ingested == 2, "the two good files must still ingest"
    assert any("d1.md" in e["file"] for e in errors)


# ---------- optimize wiring (compaction + FTS freshness) ----------

def test_sync_index_triggers_optimize(mainframe, monkeypatch):
    mf = mainframe
    d = Path(mf.config["paths"]["repos_dir"]) / "proj" / "docs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "n.md").write_text("# N\n\nnew knowledge file content for sync.\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(mf.store, "optimize", lambda: calls.append(1))
    res = mf.sync_index()
    assert res["ingested"] >= 1
    assert calls, "sync_index must optimize after changes (compaction + FTS refresh)"
    calls.clear()
    mf.sync_index()  # no changes -> no pointless compaction
    assert not calls


def test_store_optimize_smoke(mainframe):
    mf = mainframe
    _seed_doc(mf)
    mf.store.optimize()  # must not raise on this LanceDB version


# ---------- secrets.py denylist gaps ----------

def test_scrub_prefixed_env_vars():
    out = scrub_secrets("export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY")
    assert "wJalrXUtnFEMIK7MDENG" not in out and REDACTED in out
    out = scrub_secrets("DB_PASSWORD=hunter2pass")
    assert "hunter2pass" not in out and REDACTED in out
    out = scrub_secrets("MYAPP_API_KEY: abc123def456")
    assert "abc123def456" not in out
    out = scrub_secrets("AccountKey=Zm9vYmFyYmF6cXV4MTIzNDU2Nzg5MA==")
    assert "Zm9vYmFy" not in out


def test_scrub_stripe_and_url_creds():
    out = scrub_secrets("use sk_live_abcDEF12345678901234 for prod")
    assert "sk_live_abcDEF" not in out and REDACTED in out
    out = scrub_secrets("whsec_AbCdEf123456789012345678 is the webhook secret")
    assert "whsec_AbCdEf" not in out
    out = scrub_secrets("conn = postgres://admin:s3cretpw@db.host:5432/app")
    assert "s3cretpw" not in out
    assert "postgres://admin:" in out  # scheme+user preserved, password redacted


def test_scrub_still_ignores_benign_prose():
    benign = "the 256-token chunks and token_count fields; the bearer of bad news"
    assert scrub_secrets(benign) == benign


# ---------- consolidation output is scrubbed before the git-tracked write ----------

class LeakyConsolidator(FakeConsolidator):
    def consolidate(self, reports, topic="", existing=""):
        super().consolidate(reports, topic, existing)
        return "## Facts\n\nThe deploy uses api_key=SUPERSECRETVALUE123 for auth.\n"


def test_consolidate_scrubs_note(mainframe):
    mf = mainframe
    mf.consolidator = LeakyConsolidator()
    sdir = Path(mf.config["paths"]["repos_dir"]) / "proj" / "research" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    f = sdir / "2026-01-01-000000-id1-note.md"
    f.write_text("---\ntype: session\n---\n# S\n\n## Summary\nDeploy config decided.\n", encoding="utf-8")
    mf.ingest_file(str(f))
    r = mf.consolidate_sessions(project="proj", topic="deploy", days=3650)
    assert r["status"] == "consolidated"
    note = Path(r["memory_file"]).read_text(encoding="utf-8")
    assert "SUPERSECRETVALUE123" not in note, "LLM output must be scrubbed before git-tracked write"
    assert REDACTED in note


def test_manifest_recovers_in_flight_generation_from_tmp(tmp_path):
    """Crash between _save's two os.replace calls leaves main missing, .bak =
    previous generation, .tmp = the complete in-flight generation — _load must
    prefer .tmp so the newest save isn't silently lost."""
    import json as _json
    m = Manifest(tmp_path)
    m.record_ingest("C:/old.md", "h-old", 1)
    # simulate the crash window: main rotated away, newer generation stranded in .tmp
    main = tmp_path / ".manifest.json"
    (tmp_path / ".manifest.json.tmp").write_text(_json.dumps(
        {"files": {"C:/new.md": {"content_hash": "h-new", "chunk_count": 2}}, "version": 1}),
        encoding="utf-8")
    main.replace(tmp_path / ".manifest.json.bak")
    m2 = Manifest(tmp_path)
    assert "C:/new.md" in m2.data["files"], "in-flight .tmp generation must win"


def test_contradiction_section_is_scrubbed(mainframe):
    """The ⚠ prior/newer excerpts come from RAW session text — the scrub must
    run AFTER the section is appended, or a missed secret is promoted into the
    git-tracked note through the NLI path."""
    from conftest import FakeContradictionDetector
    from mainframe_mcp.server import CONTRADICTION_HEADING
    mf = mainframe
    mf.consolidator = FakeConsolidator()
    mf.nli = FakeContradictionDetector(trigger="gpu")
    sdir = Path(mf.config["paths"]["repos_dir"]) / "proj" / "research" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)

    f1 = sdir / "2026-05-01-000000-ida-note.md"
    f1.write_text("---\ntype: session\n---\n# S1\n\n## Summary\n"
                  "The GPU is an RTX 4090 with 24GB VRAM in the rig.\n", encoding="utf-8")
    mf.ingest_file(str(f1))
    assert mf.consolidate_sessions(project="proj", topic="gpu secrets", days=0)["status"] == "consolidated"

    f2 = sdir / "2026-05-02-000000-idb-note.md"
    f2.write_text("---\ntype: session\n---\n# S2\n\n## Summary\n"
                  "Correction: the GPU auth uses api_key=SEKRETVALUE1234567 not a 4090 token.\n",
                  encoding="utf-8")
    f2_raw = f2.read_text(encoding="utf-8")
    assert "SEKRETVALUE1234567" in f2_raw  # raw session really carries it
    mf.ingest_file(str(f2))
    r = mf.consolidate_sessions(project="proj", topic="gpu secrets", days=0)
    assert r["status"] == "consolidated" and r["contradictions"] >= 1
    note = Path(r["memory_file"]).read_text(encoding="utf-8")
    assert CONTRADICTION_HEADING in note
    assert "SEKRETVALUE1234567" not in note, "⚠ excerpts must be scrubbed too"
    assert REDACTED in note


# ---------- path traversal validation ----------

def test_capture_rejects_traversal_project(mainframe):
    r = mainframe.capture_memory("some captured content here", "title", project="../evil")
    assert r["status"] == "error"
    r = mainframe.consolidate_sessions(project="..\\evil", topic="x", days=30)
    assert r["status"] == "error"


def test_capture_sanitizes_session_id(mainframe):
    r = mainframe.capture_memory("captured content body here", "title",
                                 project=None, cwd=str(Path(mainframe.config["paths"]["repos_dir"]) / "proj"),
                                 session_id="../../zz")
    assert r["status"] in ("ingested", "all_chunks_duplicate")
    p = Path(r["file"])
    assert ".." not in p.name and "/" not in p.name
    sessions_dir = Path(mainframe.config["paths"]["repos_dir"]) / "proj" / "research" / "sessions"
    assert p.parent.resolve() == sessions_dir.resolve(), "file must stay inside the sessions lane"
