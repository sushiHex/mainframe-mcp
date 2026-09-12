"""Pinned reranker revisions must not follow a different configured model."""

import importlib
import json

import pytest


REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"


@pytest.mark.parametrize("module_name", ("mainframe.config", "mainframe_mcp.config"))
def test_reranker_model_override_clears_default_pin_but_keeps_explicit_pin(
        module_name, tmp_path, monkeypatch):
    config_module = importlib.import_module(module_name)
    path = tmp_path / "config.json"
    monkeypatch.delenv("MAINFRAME_PRESET", raising=False)
    monkeypatch.delenv("MAINFRAME_RERANKER_MODEL", raising=False)
    path.write_text(json.dumps({"reranker": {"model": "example/reranker"}}), encoding="utf-8")
    assert config_module.load_config(path)["reranker"]["revision"] is None
    path.write_text(json.dumps({"reranker": {"model": "example/reranker", "revision": "a" * 40}}),
                    encoding="utf-8")
    assert config_module.load_config(path)["reranker"]["revision"] == "a" * 40
    monkeypatch.setenv("MAINFRAME_RERANKER_MODEL", "example/env-reranker")
    assert config_module.load_config(tmp_path / "missing.json")["reranker"]["revision"] is None


@pytest.mark.parametrize("module_name", ("mainframe.config", "mainframe_mcp.config"))
def test_default_reranker_revision_is_pinned(module_name, tmp_path, monkeypatch):
    config_module = importlib.import_module(module_name)
    monkeypatch.delenv("MAINFRAME_PRESET", raising=False)
    monkeypatch.delenv("MAINFRAME_RERANKER_MODEL", raising=False)
    assert config_module.load_config(tmp_path / "missing.json")["reranker"]["revision"] == REVISION
