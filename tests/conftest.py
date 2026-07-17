"""Test harness for Mainframe MCP.

Builds a MainframeMCP instance with a deterministic CPU embedder and a real
on-disk LanceDB store, bypassing the GPU __init__ (which would load Qwen3-8B +
bge-reranker on CUDA). This keeps the unit tests fast and GPU-free while still
exercising the real ingest/search/delete code paths verbatim.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

# src-layout: make the package importable without an editable install.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeEmbedder:
    """Deterministic, GPU-free embedder.

    Hashed bag-of-words -> L2-normalized vector. Texts that share tokens get
    non-zero cosine similarity, which is enough to exercise vector-search and
    the source_type filter without loading a real model.
    """

    def __init__(self, dim: int = 64):
        self.dimension = dim
        self.batch_size = 8
        self.query_prefix = ""

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
        from mainframe_mcp.embedder import Embedder
        return Embedder._classify_query(query)  # the real (pure) classifier

    @property
    def info(self):
        return {"model": "fake", "dimension": self.dimension, "device": "cpu"}


class FakeReranker:
    """No-op reranker: keeps candidate order, returns the first top_k indices."""

    def __init__(self, top_k: int = 3):
        self.default_top_k = top_k
        self.enabled = True
        self.heading_inject = False
        self.last_category = None

    def rerank(self, query, documents, top_k=None, headings=None, category=None):
        self.last_category = category
        k = top_k or self.default_top_k
        return [(i, 1.0 - i * 0.01) for i in range(min(k, len(documents)))]

    @property
    def info(self):
        return {"model": "fake", "backend": "fake", "quantized": None, "enabled": True}


class FakeConsolidator:
    """Stand-in for the Qwen2.5-3B consolidator used by RAPTOR + consolidation."""

    def __init__(self):
        self.last_existing = ""  # per-instance (no shared class-attr footgun)

    def summarize_cluster(self, texts, headings, doc_path):
        return "Cluster summary: " + " ".join(texts)[:200]

    def consolidate(self, reports, topic="", existing=""):
        self.last_existing = existing
        bodies = " ".join(r["text"] for r in reports)
        return f"## Consolidated: {topic}\nMerged {len(reports)} reports. {bodies[:200]}"


class FakeContradictionDetector:
    """Flags a pair as contradictory when both mention the trigger token."""

    enabled = True

    def __init__(self, trigger: str = "gpu"):
        self.trigger = trigger

    def check_contradiction(self, text_a: str, text_b: str) -> dict:
        hit = self.trigger in text_a.lower() and self.trigger in text_b.lower()
        return {"is_contradiction": hit, "confidence": 0.95 if hit else 0.0,
                "label": "contradiction" if hit else "neutral", "all_scores": {}}


def base_config(tmp_path: Path) -> dict:
    """A DEFAULTS-derived config with paths sandboxed into tmp_path."""
    from mainframe_mcp.config import DEFAULTS

    cfg = json.loads(json.dumps(DEFAULTS))
    cfg["paths"]["repos_dir"] = str(tmp_path / "repos")
    cfg["paths"]["mainframe_dir"] = str(tmp_path / "mainframe")
    cfg["paths"]["model_cache"] = str(tmp_path / "models")
    return cfg


def make_mainframe(tmp_path: Path, embedder=None, reranker=None):
    """Construct a MainframeMCP wired to fakes + a real CPU LanceDB store."""
    from mainframe_mcp.manifest import Manifest
    from mainframe_mcp.server import MainframeMCP
    from mainframe_mcp.store import Store

    cfg = base_config(tmp_path)
    mf = MainframeMCP.__new__(MainframeMCP)  # bypass GPU model loading in __init__
    mf.config = cfg
    mf.mainframe_dir = Path(cfg["paths"]["mainframe_dir"])
    mf.mainframe_dir.mkdir(parents=True, exist_ok=True)
    mf.embedder = embedder or FakeEmbedder(dim=64)
    mf.reranker = reranker or FakeReranker(top_k=cfg["search"]["rerank_top_k"])
    mf.store = Store(
        db_path=mf.mainframe_dir / ".lancedb",
        embedding_dim=mf.embedder.dimension,
        **Store.search_kwargs_from_config(cfg),
    )
    mf.manifest = Manifest(mf.mainframe_dir)
    return mf


@pytest.fixture
def mainframe(tmp_path):
    return make_mainframe(tmp_path)
