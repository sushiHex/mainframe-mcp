import pytest

from mainframe.core.paths import chunk_key, doc_id
from mainframe.core.store import DocRows, Store


def _rows(embedder, path, texts, lane="knowledge", source_type="docs", file_hash="fh1", project="proj"):
    did = doc_id(path)
    embs = embedder.embed(texts)
    return DocRows(doc_id=did, doc_path=path, rows=[{
        "chunk_key": chunk_key(did, i), "doc_id": did, "doc_path": path, "project": project,
        "lane": lane, "source_type": source_type, "file_hash": file_hash, "content_hash": f"c{i}",
        "text": t, "heading": "H", "chunk_index": i, "char_start": 0, "char_end": len(t),
        "token_count": len(t.split()), "vector": e, "authored_at": "2026-01-01T00:00:00",
        "indexed_at": "2026-01-01T00:00:00",
    } for i, (t, e) in enumerate(zip(texts, embs))])


def test_upsert_batch_is_one_version(store, fake_embedder):
    before = len(store.table.list_versions())
    a = _rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha one", "alpha two"])
    b = _rows(fake_embedder, "c:/r/p/docs/b.md", ["beta one"])
    res = store.upsert_batch([a, b])
    assert res.upserted_rows == 3 and res.docs == 2
    assert len(store.table.list_versions()) == before + 1
    assert store.ledger() == {"c:/r/p/docs/a.md": "fh1", "c:/r/p/docs/b.md": "fh1"}


def test_shrinking_doc_deletes_only_its_stale_keys(store, fake_embedder):
    store.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["a0", "a1", "a2"]),
                        _rows(fake_embedder, "c:/r/p/docs/b.md", ["b0", "b1"])])
    store.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["a0", "a1x"], file_hash="fh2")])
    df = store._scan(["doc_path", "chunk_index", "text"]).sort_values(["doc_path", "chunk_index"])
    assert df[df.doc_path.str.endswith("a.md")]["text"].tolist() == ["a0", "a1x"]
    assert df[df.doc_path.str.endswith("b.md")]["text"].tolist() == ["b0", "b1"]  # untouched
    assert store.ledger()["c:/r/p/docs/a.md"] == "fh2"


def test_doc_emptied_and_doc_deleted(store, fake_embedder):
    store.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["a0"]),
                        _rows(fake_embedder, "c:/r/p/docs/b.md", ["b0"])])
    empty_a = DocRows(doc_id=doc_id("c:/r/p/docs/a.md"), doc_path="c:/r/p/docs/a.md", rows=[])
    store.upsert_batch([empty_a], deleted_doc_ids=[doc_id("c:/r/p/docs/b.md")])
    assert store.ledger() == {} and store.stats()["total_chunks"] == 0


def test_failed_doc_is_simply_absent_from_batch(store, fake_embedder):
    store.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["a0", "a1"])])
    store.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/b.md", ["b0"])])  # a not in batch
    assert store.ledger()["c:/r/p/docs/a.md"] == "fh1" and store.stats()["total_chunks"] == 3


def test_upsert_wins_over_stale_delete_of_same_doc(store, fake_embedder):
    """A doc present in both `docs` and `deleted_doc_ids` in the same batch must
    survive — it was just upserted, so a stale delete request for it must not
    silently drop its rows."""
    a = _rows(fake_embedder, "c:/r/p/docs/a.md", ["a0", "a1"])
    res = store.upsert_batch([a], deleted_doc_ids=[a.doc_id])
    assert res.upserted_rows == 2 and res.deleted_docs == 0
    assert store.ledger() == {"c:/r/p/docs/a.md": "fh1"}
    assert store.stats()["total_chunks"] == 2


def test_search_excludes_captures_by_default_and_is_stably_ordered(store, fake_embedder):
    store.upsert_batch([
        _rows(fake_embedder, "c:/r/p/docs/z.md", ["qwen reranker hybrid search"]),
        _rows(fake_embedder, "c:/r/p/docs/a.md", ["qwen reranker hybrid search"]),  # identical text
        _rows(fake_embedder, "c:/cap/p/x.md", ["qwen reranker hybrid search"], lane="capture", source_type="session"),
    ])
    store.optimize()
    q = fake_embedder.embed_query("qwen reranker")
    res = store.search(q, top_k=10, query_text="qwen reranker")
    assert {r["lane"] for r in res} == {"knowledge"}
    # equal scores -> tie broken by doc_path then chunk_index (a.md before z.md)
    assert [r["doc_path"] for r in res][:2] == ["c:/r/p/docs/a.md", "c:/r/p/docs/z.md"]
    assert all("vector" not in r for r in res)  # the vector column is dropped before to_dict
    res_all = store.search(q, top_k=10, include_captures=True, query_text="qwen reranker")
    assert any(r["lane"] == "capture" for r in res_all)
    assert all("vector" not in r for r in res_all)


def test_optimize_still_ensures_indexes_when_compaction_fails(store, fake_embedder, monkeypatch):
    store.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])

    def raising_optimize(*args, **kwargs):
        raise RuntimeError("compaction boom")

    monkeypatch.setattr(type(store.table), "optimize", raising_optimize)
    assert store.optimize() is False
    cols = {tuple(getattr(i, "columns", [])) for i in store.table.list_indices()}
    assert ("chunk_key",) in cols and ("text",) in cols
    h = store.index_health()
    idx_cols = {tuple(i["columns"]) for i in h["indexes"]}
    assert ("chunk_key",) in idx_cols and ("text",) in idx_cols


def test_optimize_creates_indexes_and_health_reports(store, fake_embedder):
    store.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    store.optimize()
    kinds = {tuple(getattr(i, "columns", [])) for i in store.table.list_indices()}
    assert ("text",) in kinds and ("chunk_key",) in kinds
    h = store.index_health()
    assert h["rows"] == 1 and h["versions"] >= 1 and h["fragments"] >= 1


def test_key_lookup_uses_scalar_index(store, fake_embedder):
    store.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    store.optimize()
    plan = store.explain_key_lookup(chunk_key(doc_id("c:/r/p/docs/a.md"), 0))
    if plan is None:
        pytest.skip("explain_plan not available on this LanceDB build")
    assert "ScalarIndexQuery" in plan or "index" in plan.lower()


def test_cleanup_window_must_exceed_reader_refresh(tmp_path):
    with pytest.raises(ValueError):
        Store(db_path=tmp_path / "db", embedding_dim=8, read_consistency_seconds=60, cleanup_minutes=1)


def test_drop_rebuilds_empty_table(store, fake_embedder):
    store.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    store.drop()
    assert store.stats()["total_chunks"] == 0 and store.ledger() == {}


def test_in_rejects_non_hex_ids():
    with pytest.raises(ValueError):
        Store._in(["zz"])


def test_store_defers_table_until_first_upsert(tmp_path, fake_embedder):
    s = Store(db_path=tmp_path / "lazy", embedding_dim=None)
    assert s.table is None and s.ledger() == {} and s.count_rows() == 0
    assert s.search([0.0] * 64, top_k=5, query_text="x") == []
    assert s.index_health()["rows"] == 0 and s.stats()["total_chunks"] == 0 and s.optimize() is True
    s.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    assert s.table is not None and s.embedding_dim == 64 and s.count_rows() == 1
    s2 = Store(db_path=tmp_path / "lazy", embedding_dim=None)
    assert s2.embedding_dim == 64 and s2.count_rows() == 1


def test_index_health_reports_index_presence(store, fake_embedder):
    assert store.index_health()["indexes"] == []
    store.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    assert store.optimize() is True
    cols = sorted(tuple(i["columns"]) for i in store.index_health()["indexes"])
    assert cols == [("chunk_key",), ("text",)]


def test_search_snapshots_the_table_against_a_concurrent_drop(store, fake_embedder):
    """`search` runs on to_thread workers while `rebuild` -> `drop()` nulls
    `self.table` on the writer thread. Re-reading the attribute inside the
    retry meant `self.table.checkout_latest()` hit None and raised an
    UNCAUGHT AttributeError — a bare 500. One snapshot at the top fixes it."""
    store.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["qwen reranker hybrid search"])])
    store.optimize()

    class DroppingTable:
        """Fails the first read and nulls store.table, as drop() would."""
        def __init__(self, real):
            self.real, self.calls = real, 0

        def search(self, *a, **kw):
            self.calls += 1
            if self.calls == 1:
                store.table = None
                raise RuntimeError("dataset version was removed")
            return self.real.search(*a, **kw)

        def __getattr__(self, name):
            return getattr(self.real, name)

    store.table = DroppingTable(store.table)
    q = fake_embedder.embed_query("qwen reranker")
    res = store.search(q, top_k=5, query_text="qwen reranker")      # must not raise
    assert [r["doc_path"] for r in res] == ["c:/r/p/docs/a.md"]
    store.table = None
    assert store.search(q, top_k=5, query_text="qwen reranker") == []


def test_dim_mismatch_on_reopen_is_typed(tmp_path, fake_embedder):
    s1 = Store(db_path=tmp_path / "dimdb", embedding_dim=64)
    s1.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    s2 = Store(db_path=tmp_path / "dimdb", embedding_dim=8)
    assert s2.embedding_dim == 64
    bad = _rows(fake_embedder, "c:/r/p/docs/b.md", ["beta"])
    for row in bad.rows:
        row["vector"] = [0.0] * 8
    with pytest.raises(ValueError, match="embedding dim"):
        s2.upsert_batch([bad])
