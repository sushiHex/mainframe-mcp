"""Embedder VRAM resilience: OOM at a batch retries at batch_size=1."""

import numpy as np
import pytest
import torch

from mainframe.core.embedder import Embedder


def _legacy_embedder(batch_size):
    emb = Embedder.__new__(Embedder)  # bypass model loading
    emb._use_manual = False
    emb.encoding = None  # legacy Sentence Transformers path: model.encode
    emb.batch_size = batch_size
    emb._safe_batch = 64  # grid cap (irrelevant here; > batch_size)
    return emb


def test_embed_retries_at_batch_one_on_oom():
    emb = _legacy_embedder(batch_size=8)
    calls = []

    class FakeModel:
        def encode(self, texts, batch_size, show_progress_bar=False, normalize_embeddings=True):
            calls.append(batch_size)
            if batch_size > 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return np.zeros((len(texts), 4), dtype="float32")

    emb.model = FakeModel()
    out = emb.embed(["a", "b", "c"])
    assert calls == [8, 1], "should try the full batch, then retry at 1 after OOM"
    assert len(out) == 3 and len(out[0]) == 4


def test_embed_reraises_oom_at_batch_one():
    emb = _legacy_embedder(batch_size=1)

    class AlwaysOOM:
        def encode(self, texts, batch_size, show_progress_bar=False, normalize_embeddings=True):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    emb.model = AlwaysOOM()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        emb.embed(["a"])
