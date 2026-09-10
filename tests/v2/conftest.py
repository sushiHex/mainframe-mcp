"""GPU-free fixtures for the v2 package: deterministic fakes + real LanceDB on disk."""
import contextlib
import hashlib
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@contextlib.contextmanager
def held(daemon):
    """Hold the daemon's file lock for a block, and ALWAYS release it.

    A bare `assert d.acquire()` ... `d.release()` pair leaks the lock to every
    later test in the process the moment an assertion between them fails —
    turning one real failure into a cascade of unrelated ones."""
    assert daemon.acquire(), "the daemon lock was already held"
    try:
        yield daemon
    finally:
        daemon.release()


class FakeEmbedder:
    """Hashed bag-of-words -> unit vector. Shared tokens => nonzero cosine."""

    def __init__(self, dim: int = 64):
        self.dimension = dim
        self.batch_size = 8
        self.query_prefix = ""
        self.model_name = "fake"

    def _vec(self, text: str):
        import numpy as np
        v = np.zeros(self.dimension, dtype="float32")
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.dimension] += 1.0
        n = float(np.linalg.norm(v))
        if n == 0.0:
            v[0] = 1.0
            n = 1.0
        return (v / n).tolist()

    def embed(self, texts, batch_size=None):
        return [self._vec(t) for t in texts]

    def embed_query(self, query):
        return self._vec(query)

    @staticmethod
    def _classify_query(query):
        from mainframe.core.embedder import Embedder
        return Embedder._classify_query(query)

    @property
    def info(self):
        return {"model": "fake", "dimension": self.dimension, "device": "cpu"}


class FakeReranker:
    """Order-preserving: returns the first top_k candidates with descending scores."""

    def __init__(self, top_k: int = 3):
        self.default_top_k = top_k
        self.enabled = True
        self.heading_inject = False
        self.model_name = "fake"

    def rerank(self, query, documents, top_k=None, headings=None):
        k = top_k or self.default_top_k
        return [(i, round(1.0 - i * 0.01, 4)) for i in range(min(k, len(documents)))]

    @property
    def info(self):
        return {"model": "fake", "backend": "fake", "quantized": None, "enabled": True}


@pytest.fixture
def fake_embedder():
    return FakeEmbedder(dim=64)


@pytest.fixture
def fake_reranker():
    return FakeReranker(top_k=3)


@pytest.fixture
def cfg(tmp_path):
    """DEFAULTS with every path sandboxed under tmp_path."""
    from mainframe.config import DEFAULTS
    c = json.loads(json.dumps(DEFAULTS))
    c["paths"]["repos_dir"] = str(tmp_path / "repos")
    c["paths"]["mainframe_dir"] = str(tmp_path / "mainframe")
    c["paths"]["model_cache"] = str(tmp_path / "models")
    (tmp_path / "repos").mkdir(exist_ok=True)
    (tmp_path / "mainframe").mkdir(exist_ok=True)
    return c


@pytest.fixture
def store(tmp_path, cfg):
    from mainframe.core.store import INDEX_DIRNAME, Store
    return Store(db_path=tmp_path / "mainframe" / INDEX_DIRNAME, embedding_dim=64,
                 **Store.search_kwargs_from_config(cfg))
