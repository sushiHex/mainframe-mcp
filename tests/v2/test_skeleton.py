def test_kept_modules_import():
    from mainframe.core.chunker import chunk_markdown, Chunk
    from mainframe.core.secrets import scrub_secrets, REDACTED
    from mainframe.core.embedder import Embedder
    from mainframe.core.contextual import Contextualizer
    chunks = chunk_markdown("# T\n\nhello world " * 5, max_tokens=256, overlap_ratio=0.35)
    assert chunks and isinstance(chunks[0], Chunk)
    assert scrub_secrets("x=1") == "x=1" and REDACTED == "[REDACTED]"
    assert callable(Embedder) and callable(Contextualizer)


def test_fakes(fake_embedder, fake_reranker):
    v = fake_embedder.embed(["alpha beta"])[0]
    assert len(v) == 64 and abs(sum(x * x for x in v) - 1.0) < 1e-5
    assert fake_reranker.rerank("q", ["a", "b", "c"], top_k=2) == [(0, 1.0), (1, 0.99)]
