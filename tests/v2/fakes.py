"""Shared fakes for daemon-level tests: a Models registry that never touches a GPU."""
from v2.conftest import FakeEmbedder, FakeReranker


def fake_models(config: dict, dim: int = 64, **kw):
    from mainframe.core.models import Models
    return Models(config, embedder_factory=lambda cfg: FakeEmbedder(dim=dim),
                  reranker_factory=lambda cfg: FakeReranker(top_k=3), **kw)
