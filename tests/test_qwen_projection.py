"""Native Qwen scoring must not materialize unused vocabulary projections."""

import importlib
import re
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("module", ["mainframe_mcp.reranker", "mainframe.core.reranker"])
def test_final_token_projection_preserves_scores_and_int8_batch_order(module):
    cls = importlib.import_module(module).Reranker
    reranker = cls.__new__(cls)
    reranker.instruction = "Find relevant evidence"
    reranker._quantized = True
    reranker._yes_id, reranker._no_id = 0, 1
    batches, projected = [], []

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            ids = [int(re.search(r"<Doc>: document-(\d+)", text)[1]) for text in texts]
            batches.append(ids)
            return {"input_ids": [[i, i] for i in ids]}

        def encode(self, text, **kwargs):
            return [0]

        def pad(self, inputs, **kwargs):
            return Inputs(input_ids=torch.tensor(inputs["input_ids"]))

    class Model:
        device = "cpu"

        def __call__(self, input_ids, logits_to_keep=0):
            # This fake implements the causal-LM output contract: retaining
            # fewer positions changes allocation, not the final-token answer.
            n, length = input_ids.shape
            length = logits_to_keep or length
            logits = torch.zeros((n, length, 2))
            logits[:, -1, 0] = input_ids[:, -1].float() / 4
            projected.append(length)
            return SimpleNamespace(logits=logits)

    reranker.tokenizer = Tokenizer()
    reranker.model = Model()
    scores = reranker._score_qwen3("query", [f"document-{i}" for i in range(11)])
    assert scores == pytest.approx(torch.sigmoid(torch.arange(11).float() / 4).tolist())
    assert batches == [list(range(8)), [8, 9, 10]]
    assert projected == [1, 1]
