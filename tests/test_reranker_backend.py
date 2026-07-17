"""Qwen3-Reranker native backend: pure (GPU-free) pieces — backend detection
and the model-card prompt format. The logit scoring itself is validated by the
GPU eval harness."""

from mainframe_mcp.reranker import (
    CATEGORY_INSTRUCTIONS,
    _QWEN3_DOC_CHAR_CLAMP,
    _QWEN3_QUERY_CHAR_CLAMP,
    Reranker,
    detect_backend,
    qwen3_pair_text,
)


def test_backend_detection():
    assert detect_backend("BAAI/bge-reranker-v2-m3") == "cross-encoder"
    assert detect_backend("Qwen/Qwen3-Reranker-4B") == "qwen3-logit"
    assert detect_backend("Qwen/Qwen3-Reranker-0.6B") == "qwen3-logit"
    # the seq-cls conversions ARE CrossEncoders — must NOT take the logit path
    assert detect_backend("tomaarsen/Qwen3-Reranker-0.6B-seq-cls") == "cross-encoder"
    # explicit config override wins either way
    assert detect_backend("BAAI/bge-reranker-v2-m3", "qwen3-logit") == "qwen3-logit"
    assert detect_backend("Qwen/Qwen3-Reranker-4B", "cross-encoder") == "cross-encoder"


def test_qwen3_pair_text_format():
    out = qwen3_pair_text("grid overflow fix", "The batch cap prevents it.", "Find the doc")
    assert out == ("<Instruct>: Find the doc\n"
                   "<Query>: grid overflow fix\n"
                   "<Doc>: The batch cap prevents it.")


def test_qwen3_pair_text_clamps_pathological_docs():
    huge = "x" * (_QWEN3_DOC_CHAR_CLAMP * 3)
    out = qwen3_pair_text("q", huge, "i")
    # clamped so tokenizer truncation can never eat the assistant suffix
    assert len(out) < _QWEN3_DOC_CHAR_CLAMP + 100


def test_qwen3_pair_text_clamps_pathological_query():
    huge_q = "w " * 4000
    out = qwen3_pair_text(huge_q, "doc", "i")
    assert len(out) < _QWEN3_QUERY_CHAR_CLAMP + _QWEN3_DOC_CHAR_CLAMP


def test_category_instruction_resolution():
    """The Reranker owns the category gate: callers pass their classification
    unconditionally. Gate off (default) -> always the instance default; gate on
    -> per-category with fallback for unknown/absent; config overrides win."""
    r = Reranker.__new__(Reranker)
    r.instruction = "default instruction"
    r.category_instructions = dict(CATEGORY_INSTRUCTIONS)
    r.category_enabled = False  # shipped default: measured 0.3760 vs 0.4187
    assert r.resolve_instruction("code") == "default instruction"
    r.category_enabled = True
    assert r.resolve_instruction("code") == CATEGORY_INSTRUCTIONS["code"]
    assert r.resolve_instruction("general") == "default instruction"
    assert r.resolve_instruction(None) == "default instruction"
    r.category_instructions = {**CATEGORY_INSTRUCTIONS, "code": "my override"}
    assert r.resolve_instruction("code") == "my override"


def test_score_qwen3_retries_per_item_on_oom():
    """VRAM spike mid-batch must degrade to item-at-a-time scoring, not kill
    the search (parity with the embedder's OOM fallback)."""
    import torch

    r = Reranker.__new__(Reranker)
    r._quantized = False
    r.instruction = "i"
    r.category_instructions = {}
    r._yes_id, r._no_id = 0, 1
    calls = []

    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        def __call__(self, batch, **kw):
            calls.append(len(batch))
            return FakeInputs(input_ids=torch.zeros((len(batch), 4), dtype=torch.long))

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
