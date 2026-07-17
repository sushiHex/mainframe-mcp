"""Index-health fixes from the 2026-07-02 head-to-toe evaluation:

1. FTS index detection must recognize LanceDB 0.30's index_type='FTS' —
   the old check only matched 'INVERTED', so EVERY process start re-created
   the index (observed live: 26 unpruned FTS generations, +193MB).
2. optimize() must CREATE a missing FTS index, not just refresh an existing
   one — a brand-new mainframe_dir otherwise runs its whole first server
   session vector-only ("hybrid" claim silently false until restart).
3. authored_at: chunks carry the source file's mtime so recency can rank by
   authored time instead of ingest time; existing tables are migrated via
   add_columns and recency prefers authored_at when present.
"""

import hashlib

from conftest import FakeEmbedder
from mainframe_mcp.store import Store


def _mk_store(tmp_path, **kw):
    return Store(db_path=tmp_path / ".lancedb", embedding_dim=32, tier_boost={}, **kw)


def _insert(st, emb, text, doc="C:/d.md", authored_at=""):
    e = emb.embed([text])[0]
    h = hashlib.sha256(text.encode()).hexdigest()
    st.insert_chunks([{"text": text, "heading": "H", "chunk_index": 0, "char_start": 0,
                       "char_end": 1, "token_count": 9}], "d", doc, "research", [e], [h],
                     authored_at=authored_at)


def test_fts_detection_recognizes_existing_index(tmp_path):
    """After an index exists, _ensure_fts_index must be a no-op — the 'FTS'
    index_type string must be recognized (the 'INVERTED'-only check re-created
    the index on every process start)."""
    emb = FakeEmbedder(dim=32)
    st = _mk_store(tmp_path)
    _insert(st, emb, "qwen reranker hybrid chunks facts")
    st.table.create_fts_index("text", replace=True)
    assert any("fts" in str(getattr(i, "index_type", "")).lower()
               for i in st.table.list_indices()), "precondition: index exists as FTS type"

    calls = []
    real_create = st.table.create_fts_index
    st.table.create_fts_index = lambda *a, **k: calls.append(1) or real_create(*a, **k)
    st._ensure_fts_index()
    assert not calls, "existing FTS index must be detected — no re-creation"


def test_optimize_creates_missing_fts_index(tmp_path):
    """Fresh table (no FTS index yet): the first optimize() pass — which runs
    after every ingest batch — must create the index so keyword search works
    in the very first session."""
    emb = FakeEmbedder(dim=32)
    st = _mk_store(tmp_path)
    _insert(st, emb, "tantivy keyword searchable unique zebra token")
    assert not any("fts" in str(getattr(i, "index_type", "")).lower()
                   for i in st.table.list_indices()), "precondition: new table has no FTS"

    st.optimize()

    hits = st.table.search("zebra", query_type="fts").limit(5).to_pandas()
    assert not hits.empty, "keyword search must work after the first optimize()"


def test_authored_at_roundtrip_and_migration(tmp_path):
    emb = FakeEmbedder(dim=32)
    st = _mk_store(tmp_path)
    _insert(st, emb, "authored timestamp roundtrip fact", authored_at="2026-01-15T10:00:00")
    df = st.table.to_pandas()
    assert df.iloc[0]["authored_at"] == "2026-01-15T10:00:00"

    # migration: reopening a pre-authored_at table must add the column
    # (backfilled from created_at) instead of crashing inserts
    st.table.drop_columns(["authored_at"])
    st2 = _mk_store(tmp_path)
    assert "authored_at" in [f.name for f in st2.table.schema], "add_columns migration"


def test_recency_prefers_authored_at_over_ingest_time(tmp_path):
    """Two docs ingested at the SAME moment; the one AUTHORED long ago must
    rank below the recently-authored one when recency is on — created_at
    (ingest time) can no longer misrank re-ingested old files."""
    emb = FakeEmbedder(dim=32)
    st = _mk_store(tmp_path, recency_weight=0.2, recency_halflife_days=90)
    text = "identical text so vector distance ties exactly here"
    _insert(st, emb, text, doc="C:/fresh.md", authored_at="2026-07-01T00:00:00")
    _insert(st, emb, text, doc="C:/ancient.md", authored_at="2024-01-01T00:00:00")
    res = st.search(emb.embed_query(text), top_k=5, query_text="")
    assert res and res[0]["doc_path"] == "C:/fresh.md", \
        f"authored-recent doc must outrank ancient one, got {[r['doc_path'] for r in res]}"
