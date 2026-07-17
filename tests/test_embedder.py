"""Embedder VRAM resilience: OOM at a batch retries at batch_size=1."""

import numpy as np
import torch

from mainframe_mcp.embedder import Embedder


def test_embed_retries_at_batch_one_on_oom():
    emb = Embedder.__new__(Embedder)  # bypass model loading
    emb._use_manual = False
    emb.batch_size = 8
    emb._safe_batch = 64  # grid cap (irrelevant here; > batch_size)
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
    emb = Embedder.__new__(Embedder)
    emb._use_manual = False
    emb.batch_size = 1
    emb._safe_batch = 64

    class AlwaysOOM:
        def encode(self, texts, batch_size, show_progress_bar=False, normalize_embeddings=True):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    emb.model = AlwaysOOM()
    try:
        emb.embed(["a"])
        assert False, "expected OOM to propagate when already at batch_size=1"
    except torch.cuda.OutOfMemoryError:
        pass
