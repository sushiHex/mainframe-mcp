import sys
from types import SimpleNamespace
import weakref

import pytest

from mainframe.core import embedder


@pytest.mark.parametrize("failure_point", ["constructor", "metadata"])
def test_fallback_releases_primary_allocations_before_loading_another_model(tmp_path, monkeypatch, failure_point):
    refs, cache_liveness, quantizers = [], [], []

    class PartialModel:
        def __init__(self):
            self.cycle = self
            refs.append(weakref.ref(self))

        def get_sentence_embedding_dimension(self):
            raise OSError("synthetic allocation failure (os error 1455)")

    def primary(*args, **kwargs):
        quantizers.append(kwargs["model_kwargs"]["quantization_config"])
        model = PartialModel()
        if failure_point == "constructor":
            raise OSError("synthetic allocation failure (os error 1455)")
        return model

    def fallback(self, name, cache, quantization):
        assert all(ref() is None for ref in refs), "the failed primary model is still alive"
        assert cache_liveness == [False], "clear the allocator only after releasing the model"
        assert quantization is quantizers[0] and quantization == {"load_in_8bit": True}
        self._use_manual = True
        self.dimension = 8

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(BitsAndBytesConfig=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=primary))
    monkeypatch.setattr(embedder, "empty_cuda_cache",
                        lambda: cache_liveness.append(any(ref() is not None for ref in refs)), raising=False)
    monkeypatch.setattr(embedder.Embedder, "_load_manual_quantized", fallback)
    model = object.__new__(embedder.Embedder)
    model._load_quantized("synthetic-model", tmp_path)
    assert model._use_manual and model.dimension == 8


def test_primary_success_keeps_the_model_without_fallback_or_cache_flush(tmp_path, monkeypatch):
    model = SimpleNamespace(get_sentence_embedding_dimension=lambda: 8)
    calls = []
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(BitsAndBytesConfig=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *a, **kw: model))
    monkeypatch.setattr(embedder.torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(embedder, "empty_cuda_cache", lambda: calls.append("cache"), raising=False)
    monkeypatch.setattr(embedder.Embedder, "_load_manual_quantized", lambda *args: calls.append("fallback"))
    owner = object.__new__(embedder.Embedder)
    owner._load_quantized("synthetic-model", tmp_path)
    assert owner.model is model and owner.dimension == 8 and not calls
