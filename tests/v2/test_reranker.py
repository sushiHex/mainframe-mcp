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


def test_pair_text_comes_from_the_qwen_helper():
    from mainframe.core.qwen import qwen3_pair_text

    assert rr.qwen3_pair_text is qwen3_pair_text


def test_qwen_defaults_to_bf16_and_reports_score_contract(monkeypatch):
    monkeypatch.setattr(rr.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(rr.Reranker, "_load_qwen3",
                        lambda self, device: setattr(self, "_precision", "bf16"))
    r = rr.Reranker({"reranker": {"model": "Qwen/Qwen3-Reranker-0.6B", "enabled": True},
                     "search": {"rerank_top_k": 3}})
    assert r.quantize is False
    assert r.instruction == rr._QWEN3_DEFAULT_INSTRUCTION
    assert r.info["score_type"] == "conditional-yes-probability"
    assert r.info["context_length"] == 2048


def test_score_qwen3_retries_per_item_on_oom():
    """A VRAM spike mid-batch must degrade to item-at-a-time scoring, not kill
    the search (parity with the embedder's OOM fallback)."""
    import torch

    r = rr.Reranker.__new__(rr.Reranker)
    r._quantized = False
    r.instruction = "i"
    r._yes_id, r._no_id = 0, 1
    calls = []

    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        def get_vocab(self):
            return {"x": 0}

        def __call__(self, batch, **kw):
            calls.append(len(batch))
            return {"input_ids": [[0] * 4 for _ in batch]}

        def encode(self, text, **kw):
            return [0]

        def pad(self, inputs, **kw):
            return FakeInputs(input_ids=torch.tensor(inputs["input_ids"]))

    class FakeModel:
        device = "cpu"

        def __call__(self, **inputs):
            n = inputs["input_ids"].shape[0]
            if n > 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            out = type("O", (), {})()
            out.logits = torch.zeros((n, 4, 2))
            out.logits[:, -1, 0] = 1.0  # yes wins
            return out

    r.tokenizer = FakeTokenizer()
    r.model = FakeModel()
    scores = r._score_qwen3("q", ["a", "b", "c"])
    assert len(scores) == 3 and all(s > 0.5 for s in scores)
    assert calls[0] == 3 and all(c == 1 for c in calls[1:]), \
        f"full batch then per-item retries, got {calls}"


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
