"""§9.6 — build_raptor skips source_type="session" docs."""

from conftest import FakeConsolidator

from mainframe_mcp.dedup import content_hash
from mainframe_mcp.raptor import build_raptor_summaries


def _add_doc(store, embedder, doc_path, source_type, texts):
    chunks = [
        {
            "text": t,
            "heading": "Notes",
            "chunk_index": i,
            "char_start": 0,
            "char_end": len(t),
            "token_count": 10,
        }
        for i, t in enumerate(texts)
    ]
    store.insert_chunks(
        chunks=chunks,
        doc_id=content_hash(doc_path)[:12],
        doc_path=doc_path,
        source_type=source_type,
        embeddings=embedder.embed(texts),
        content_hashes=[content_hash(t) for t in texts],
    )


def test_raptor_excludes_session_docs(mainframe):
    mf = mainframe
    # Three near-identical chunks per doc so each doc forms one cluster of >=3.
    _add_doc(
        mf.store, mf.embedder, "S/research/sessions/log.md", "session",
        ["alpha beta gamma delta", "alpha beta gamma delta epsilon", "alpha beta gamma delta zeta"],
    )
    _add_doc(
        mf.store, mf.embedder, "R/research/curated.md", "research",
        ["one two three four", "one two three four five", "one two three four six"],
    )

    build_raptor_summaries(
        mf.store, mf.embedder, FakeConsolidator(),
        distance_threshold=0.8, min_cluster_size=3,
    )

    df = mf.store.table.to_pandas()
    tier1 = df[df["tier"] == 1]
    # RAPTOR still summarizes curated docs...
    assert len(tier1) >= 1
    # ...but emits no tier-1 "session" summaries.
    assert (tier1["source_type"] != "session").all()
