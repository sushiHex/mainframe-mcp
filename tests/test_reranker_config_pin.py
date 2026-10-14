"""Pinned reranker revisions must not follow a different configured model."""

import json

from mainframe.config import load_config


REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"


def test_reranker_model_override_clears_default_pin_but_keeps_explicit_pin(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.delenv("MAINFRAME_PRESET", raising=False)
    monkeypatch.delenv("MAINFRAME_RERANKER_MODEL", raising=False)
    path.write_text(json.dumps({"reranker": {"model": "example/reranker"}}), encoding="utf-8")
    assert load_config(path)["reranker"]["revision"] is None
    path.write_text(json.dumps({"reranker": {"model": "example/reranker", "revision": "a" * 40}}),
                    encoding="utf-8")
    assert load_config(path)["reranker"]["revision"] == "a" * 40
    monkeypatch.setenv("MAINFRAME_RERANKER_MODEL", "example/env-reranker")
    assert load_config(tmp_path / "missing.json")["reranker"]["revision"] is None


def test_default_reranker_revision_is_pinned(tmp_path, monkeypatch):
    monkeypatch.delenv("MAINFRAME_PRESET", raising=False)
    monkeypatch.delenv("MAINFRAME_RERANKER_MODEL", raising=False)
    assert load_config(tmp_path / "missing.json")["reranker"]["revision"] == REVISION
