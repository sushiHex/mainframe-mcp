"""A: recency-aware ranking (store.search). B: conflict-aware consolidation."""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

from conftest import (
    FakeConsolidator,
    FakeContradictionDetector,
    FakeEmbedder,
    make_mainframe,
)
from mainframe_mcp.server import CONTRADICTION_HEADING, _canonical_body
from mainframe_mcp.store import Store


def test_canonical_body_strips_scaffolding():
    note = (
        "# Hardware notes\n\n"
        "_Consolidated 2026-06-30 from 3 session capture(s)._\n\n"
        "The GPU is an RTX 3090 with 24GB VRAM.\n\n"
        f"{CONTRADICTION_HEADING}\n"
        "- **prior:** RTX 4090\n  **newer:** RTX 3090  _(NLI 0.95)_\n\n"
        "## Sources\n- 2026-06-01-000000-id1-note.md\n"
    )
    body = _canonical_body(note)
    assert "The GPU is an RTX 3090 with 24GB VRAM." in body  # knowledge kept
    assert "_Consolidated" not in body                       # preamble dropped
    assert "# Hardware notes" not in body                    # title dropped
    assert "⚠" not in body and "RTX 4090" not in body        # ⚠ section dropped
    assert "## Sources" not in body and "id1-note.md" not in body  # provenance dropped


# ---------- A. recency-aware ranking ----------

def test_recency_halflife_clamped_to_positive(tmp_path):
    # 0 and negative halflife must NOT pass through (negative inverts ranking:
    # 0.5 ** (age / -hl) = 2 ** age -> oldest first).
    for bad in (0, -5):
        st = Store(db_path=tmp_path / f"db{bad}", embedding_dim=8, tier_boost={},
                   recency_weight=0.1, recency_halflife_days=bad)
        assert st._recency_halflife == 90.0


def test_recency_off_by_default_is_eval_neutral(tmp_path):
    # default recency_weight=0 -> identical ranking to no recency at all
    emb = FakeEmbedder(dim=32)
    st = Store(db_path=tmp_path / ".lancedb", embedding_dim=32, tier_boost={})
    assert st._recency_weight == 0.0  # opt-in
    text = "Qwen3 RTX 3090 bge reranker chunks embeddings hybrid search note"
    e = emb.embed([text])[0]
    h = hashlib.sha256(text.encode()).hexdigest()
    st.insert_chunks([{"text": text, "heading": "H", "chunk_index": 0, "char_start": 0,
                       "char_end": 1, "token_count": 9}], "d", "C:/d.md", "research", [e], [h])
    assert st.search(emb.embed_query(text), top_k=5, query_text="")  # no crash, returns hits


def test_search_prefers_recent_among_equal(tmp_path):
    emb = FakeEmbedder(dim=32)
    st = Store(db_path=tmp_path / ".lancedb", embedding_dim=32, tier_boost={},
               recency_weight=0.2, recency_halflife_days=90)
    text = "Qwen3 RTX 3090 bge reranker chunks embeddings hybrid search note"
    e = emb.embed([text])[0]
    h = hashlib.sha256(text.encode()).hexdigest()
    # two identical-text chunks for two docs -> identical vector distance
    st.insert_chunks([{"text": text, "heading": "H", "chunk_index": 0, "char_start": 0,
                       "char_end": 1, "token_count": 9}], "dnew", "C:/new.md", "research", [e], [h])
    st.insert_chunks([{"text": text, "heading": "H", "chunk_index": 0, "char_start": 0,
                       "char_end": 1, "token_count": 9}], "dold", "C:/old.md", "research", [e], [h + "x"])
    # age the "old" doc
    old_ts = (datetime.now() - timedelta(days=400)).isoformat()
    st.table.update(where='doc_path = "C:/old.md"', values={"created_at": old_ts})

    res = st.search(emb.embed_query(text), top_k=5, query_text="")
    paths = [r["doc_path"] for r in res]
    assert paths and paths[0] == "C:/new.md", f"recent doc should rank first, got {paths}"


# ---------- B. conflict-aware consolidation ----------

def _seed(mf, n, start=0):
    repos = Path(mf.config["paths"]["repos_dir"])
    sdir = repos / "proj" / "research" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + n):
        f = sdir / f"2026-01-{i+1:02d}-000000-id{i}-note.md"
        f.write_text(f"---\ntype: session\n---\n# S{i}\n\n## Summary\nDecision {i}: fact {i}.\n",
                     encoding="utf-8")
        mf.ingest_file(str(f))


def test_consolidate_feeds_existing_note_to_consolidator(mainframe):
    mf = mainframe
    mf.consolidator = FakeConsolidator()

    _seed(mf, 2, start=0)
    r1 = mf.consolidate_sessions(project="proj", topic="reranker decisions", days=3650)
    assert r1["status"] == "consolidated"
    note = Path(r1["memory_file"])
    assert note.exists()
    assert mf.consolidator.last_existing == ""  # first run: no prior memory

    # a later run on NEW sessions must hand the existing note to the consolidator
    _seed(mf, 1, start=5)
    r2 = mf.consolidate_sessions(project="proj", topic="reranker decisions", days=3650)
    assert r2["status"] == "consolidated"
    fed = mf.consolidator.last_existing
    assert fed, "existing note not passed to consolidator"
    assert "Decision 0" in fed                # body content fed back
    assert "_Consolidated" not in fed         # preamble not re-fed (no accretion)
    assert "## Sources" not in fed            # provenance not re-fed


def test_consolidate_prunes_sources_when_note_unchanged(mainframe, monkeypatch):
    """An identical-content re-merge returns ingest 'unchanged' — the note IS in
    the index, so sources must still be archived/pruned (else they re-consolidate
    forever)."""
    mf = mainframe
    mf.consolidator = FakeConsolidator()
    _seed(mf, 2, start=0)
    sources = list((Path(mf.config["paths"]["repos_dir"]) / "proj" / "research" / "sessions").glob("*.md"))
    assert sources

    orig = mf.ingest_file

    def fake_ingest(path):
        if "/memory/" in str(path).replace("\\", "/"):
            return {"status": "unchanged", "file": str(path)}
        return orig(path)

    monkeypatch.setattr(mf, "ingest_file", fake_ingest)
    res = mf.consolidate_sessions(project="proj", topic="reranker decisions", days=3650)
    assert res["status"] == "consolidated"          # 'unchanged' is not a failure
    for s in sources:
        assert not s.exists(), "source not archived when note was 'unchanged'"


# ---------- C. contradiction surfacing (NLI) ----------

def test_consolidate_surfaces_contradictions(mainframe):
    mf = mainframe
    mf.consolidator = FakeConsolidator()
    mf.nli = FakeContradictionDetector(trigger="gpu")
    sdir = Path(mf.config["paths"]["repos_dir"]) / "proj" / "research" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)

    f1 = sdir / "2026-01-01-000000-id1-note.md"
    f1.write_text("---\ntype: session\n---\n# S1\n\n## Summary\n"
                  "The GPU is a RTX 4090 with 24GB VRAM in the rig.\n", encoding="utf-8")
    mf.ingest_file(str(f1))
    r1 = mf.consolidate_sessions(project="proj", topic="hardware", days=3650)
    assert r1["status"] == "consolidated"
    assert r1.get("contradictions", 0) == 0  # no prior note -> nothing to contradict

    f2 = sdir / "2026-01-05-000000-id2-note.md"
    f2.write_text("---\ntype: session\n---\n# S2\n\n## Summary\n"
                  "Correction: the GPU is a RTX 3090 not a 4090 with 24GB VRAM.\n", encoding="utf-8")
    mf.ingest_file(str(f2))
    r2 = mf.consolidate_sessions(project="proj", topic="hardware", days=3650)
    assert r2["status"] == "consolidated"
    assert r2.get("contradictions", 0) >= 1, "contradiction not detected/surfaced"
    assert r2.get("nli") == "ran"  # status distinguishes ran-none from never-ran
    note_text = Path(r2["memory_file"]).read_text(encoding="utf-8")
    assert "Contradiction" in note_text  # surfaced in the curated note, not silently dropped


def test_consolidate_strips_prior_contradiction_section(mainframe):
    """The ## ⚠ block must be stripped before re-feeding the note, so it isn't
    re-NLI'd / re-accreted and its markup isn't fed back to the consolidator."""
    mf = mainframe
    mf.consolidator = FakeConsolidator()
    mf.nli = FakeContradictionDetector(trigger="zzz")  # triggers no NEW contradictions

    mem = Path(mf.config["paths"]["repos_dir"]) / "proj" / "research" / "memory"
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "hardware.md").write_text(
        "# hardware\n\nFact A is true and stable here.\n\n"
        f"{CONTRADICTION_HEADING}\n"
        "- **prior:** old claim x is wrong\n  **newer:** new claim y  _(NLI 0.95)_\n\n"
        "## Sources\n- old.md\n", encoding="utf-8")

    sdir = Path(mf.config["paths"]["repos_dir"]) / "proj" / "research" / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    f = sdir / "2026-02-01-000000-id9-note.md"
    f.write_text("---\ntype: session\n---\n# S\n\n## Summary\nFact B is a new addition here.\n", encoding="utf-8")
    mf.ingest_file(str(f))

    r = mf.consolidate_sessions(project="proj", topic="hardware", days=3650)
    assert r["status"] == "consolidated"
    fed = mf.consolidator.last_existing
    assert "⚠" not in fed and "old claim x" not in fed  # ⚠ section stripped
    assert "Fact A is true" in fed                       # surgical: body preserved
