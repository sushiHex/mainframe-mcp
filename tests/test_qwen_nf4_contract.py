"""Loader contracts for the selected native Qwen NF4/BF16 reranker."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch


MODULES = ("mainframe_mcp.reranker", "mainframe.core.reranker")
REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"


def install_fake_transformers(monkeypatch, *, loaded_in_4bit, calls=None):
    calls = {} if calls is None else calls

    class Tokenizer:
        def convert_tokens_to_ids(self, token):
            return {"yes": 3, "no": 4}[token]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["tokenizer"] = (args, kwargs)
            return Tokenizer()

    class Model:
        is_loaded_in_4bit = loaded_in_4bit

        def eval(self):
            calls["eval"] = True
            return self

    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["model"] = (args, kwargs)
            return Model()

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            calls["bnb"] = self

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModel,
        BitsAndBytesConfig=BitsAndBytesConfig))
    return calls


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.parametrize("device,quantize,loaded_in_4bit,precision", [
    ("cuda", True, True, "nf4-bf16"),
    ("cuda", False, False, "bf16"),
    ("cpu", True, False, "bf16"),
])
def test_qwen_loader_uses_pinned_nf4_or_bf16(
        monkeypatch, module_name, device, quantize, loaded_in_4bit, precision):
    module = importlib.import_module(module_name)
    calls = install_fake_transformers(monkeypatch, loaded_in_4bit=loaded_in_4bit)
    reranker = module.Reranker.__new__(module.Reranker)
    reranker.model_name, reranker.revision, reranker.quantize = "Qwen/test", REVISION, quantize
    reranker._backend, reranker.enabled = "qwen3-logit", True
    reranker._load_qwen3(device)

    assert calls["tokenizer"][1] == {"padding_side": "left", "trust_remote_code": True,
                                      "revision": REVISION}
    kwargs = calls["model"][1]
    assert kwargs["dtype"] is torch.bfloat16
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["revision"] == REVISION and calls["eval"] is True
    assert reranker.info["precision"] == precision
    assert reranker.info["batch_size"] == 8
    if quantize and device == "cuda":
        assert kwargs["device_map"] == "cuda"
        assert calls["bnb"].load_in_4bit and calls["bnb"].bnb_4bit_quant_type == "nf4"
        assert calls["bnb"].bnb_4bit_use_double_quant is True
        assert calls["bnb"].bnb_4bit_compute_dtype is torch.bfloat16
    else:
        assert kwargs["device_map"] == device and "quantization_config" not in kwargs


@pytest.mark.parametrize("module_name", MODULES)
def test_qwen_cuda_nf4_load_requires_actual_4bit_model(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    install_fake_transformers(monkeypatch, loaded_in_4bit=False)
    reranker = module.Reranker.__new__(module.Reranker)
    reranker.model_name, reranker.revision, reranker.quantize = "Qwen/test", REVISION, True
    reranker._backend, reranker.enabled = "qwen3-logit", True

    with pytest.raises(RuntimeError, match="did not produce a 4-bit model"):
        reranker._load_qwen3("cuda")


@pytest.mark.parametrize("module_name", MODULES)
def test_cross_encoder_receives_requested_revision(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    calls = {}

    class CrossEncoder:
        def __init__(self, *args, **kwargs):
            calls["args"], calls["kwargs"] = args, kwargs

    monkeypatch.setattr(module, "CrossEncoder", CrossEncoder)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    config = {"reranker": {"model": "example/cross", "revision": REVISION,
                             "backend": "cross-encoder", "enabled": True},
              "search": {"rerank_top_k": 3}}
    module.Reranker(config)
    assert calls["kwargs"]["revision"] == REVISION
