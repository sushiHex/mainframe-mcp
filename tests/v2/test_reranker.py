import inspect

from mainframe.core import reranker as rr


def test_backend_detection():
    assert rr.detect_backend("Qwen/Qwen3-Reranker-4B") == "qwen3-logit"
    assert rr.detect_backend("tomaarsen/Qwen3-Reranker-0.6B-seq-cls") == "cross-encoder"
    assert rr.detect_backend("BAAI/bge-reranker-v2-m3") == "cross-encoder"
    assert rr.detect_backend("Qwen/Qwen3-Reranker-4B", cfg_backend="cross-encoder") == "cross-encoder"


def test_pair_text_bounds_query_but_leaves_document_for_token_budget():
    t = rr.qwen3_pair_text("q" * 2000, "d" * 5000, "inst")
    assert t.startswith("<Instruct>: inst\n<Query>: ") and "<Doc>: " in t
    assert t.count("q") == rr._QWEN3_QUERY_CHAR_CLAMP and t.count("d") == 5000


def test_category_machinery_is_gone():
    assert not hasattr(rr, "CATEGORY_INSTRUCTIONS")
    assert not hasattr(rr.Reranker, "resolve_instruction")
    assert "category" not in inspect.signature(rr.Reranker.rerank).parameters


def test_disabled_reranker_keeps_candidate_order():
    cfg = {"reranker": {"model": "x", "enabled": False}, "search": {"rerank_top_k": 3}}
    r = rr.Reranker(cfg)
    assert r.rerank("q", ["a", "b", "c", "d"], top_k=2) == [(0, 0.0), (1, 0.0)]
    assert r.info["backend"] == "disabled"
