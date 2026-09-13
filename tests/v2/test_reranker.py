import inspect

from mainframe.core import reranker as rr


def test_backend_detection():
    assert rr.detect_backend("Qwen/Qwen3-Reranker-4B") == "qwen3-logit"
    assert rr.detect_backend("tomaarsen/Qwen3-Reranker-0.6B-seq-cls") == "cross-encoder"
    assert rr.detect_backend("BAAI/bge-reranker-v2-m3") == "cross-encoder"
    assert rr.detect_backend("Qwen/Qwen3-Reranker-4B", cfg_backend="cross-encoder") == "cross-encoder"


def test_pair_text_bounds_query_but_leaves_document_for_token_budget():
    t = rr.qwen3_pair_text("q" * 2000, "d" * 5000, "inst")
    assert t.startswith("<Instruct>: inst\n<Query>: ") and "<Document>: " in t
    assert t.count("q") == rr._QWEN3_QUERY_CHAR_CLAMP and t.count("d") == 5000


def test_pair_text_is_shared_with_v1():
    from mainframe_mcp.qwen import qwen3_pair_text

    assert rr.qwen3_pair_text is qwen3_pair_text


def test_category_machinery_is_gone():
    assert not hasattr(rr, "CATEGORY_INSTRUCTIONS")
    assert not hasattr(rr.Reranker, "resolve_instruction")
    assert "category" not in inspect.signature(rr.Reranker.rerank).parameters


def test_disabled_reranker_keeps_candidate_order():
    cfg = {"reranker": {"model": "x", "enabled": False}, "search": {"rerank_top_k": 3}}
    r = rr.Reranker(cfg)
    assert r.rerank("q", ["a", "b", "c", "d"], top_k=2) == [(0, 0.0), (1, 0.0)]
    assert r.info["backend"] == "disabled"


def test_none_instruction_falls_back_to_the_model_card_default(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(rr, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(rr.torch.cuda, "is_available", lambda: False)
    cfg = {"reranker": {"model": "example/cross", "backend": "cross-encoder",
                         "enabled": True, "instruction": None},
           "search": {"rerank_top_k": 3}}
    r = rr.Reranker(cfg)
    assert r.instruction == rr._QWEN3_DEFAULT_INSTRUCTION


def test_configured_instruction_replaces_the_default(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(rr, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(rr.torch.cuda, "is_available", lambda: False)
    custom = "Given a legal contract clause, retrieve the passage that defines it"
    cfg = {"reranker": {"model": "example/cross", "backend": "cross-encoder",
                         "enabled": True, "instruction": custom},
           "search": {"rerank_top_k": 3}}
    r = rr.Reranker(cfg)
    assert r.instruction == custom
