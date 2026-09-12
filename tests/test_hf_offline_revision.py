"""Offline Hub admission must cover the exact configured model revisions."""

import importlib
import os
from types import SimpleNamespace

import huggingface_hub
import pytest


MODULES = ("mainframe_mcp.config", "mainframe.config")
COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
EMBED_COMMIT = "c" * 40


def _config(revision=COMMIT):
    return {
        "embedder": {"model": "example/embedder", "revision": EMBED_COMMIT},
        "reranker": {"model": "example/reranker", "revision": revision, "enabled": True},
        "nli": {"model": "example/nli", "enabled": False},
        "consolidator": {"model": "example/consolidator", "enabled": False},
        "paths": {"model_cache": "/alternate/cache"},
    }


@pytest.mark.parametrize("module_name", MODULES)
def test_cached_revision_inventory_unions_commits_and_refs_across_cache_roots(
        monkeypatch, module_name):
    module = importlib.import_module(module_name)
    default_repo = SimpleNamespace(
        repo_id="example/reranker",
        revisions=[SimpleNamespace(commit_hash=OTHER_COMMIT, refs=frozenset({"main"}))],
    )
    alternate_repo = SimpleNamespace(
        repo_id="example/reranker",
        revisions=[SimpleNamespace(commit_hash=COMMIT, refs=frozenset({"release/v1"}))],
    )
    calls = []

    def scan_cache_dir(path=None):
        calls.append(path)
        return SimpleNamespace(repos=[default_repo] if path is None else [alternate_repo])

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", scan_cache_dir)
    assert module._cached_repo_revisions([None, "/alternate/cache"]) == {
        "example/reranker": {OTHER_COMMIT, COMMIT, "main", "release/v1"},
    }
    assert calls == [None, "/alternate/cache"]


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.parametrize("revision", [COMMIT, "release/v1"], ids=["commit", "ref"])
def test_hf_offline_accepts_exact_cached_commit_or_ref(monkeypatch, module_name, revision):
    module = importlib.import_module(module_name)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(module, "_cached_repo_revisions", lambda dirs: {
        "example/embedder": {EMBED_COMMIT},
        "example/reranker": {COMMIT, "release/v1"},
    })

    assert module.hf_offline_if_cached(_config(revision)) is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    monkeypatch.delenv("HF_HUB_OFFLINE")


@pytest.mark.parametrize("module_name", MODULES)
@pytest.mark.parametrize("missing", ["embedder", "reranker"])
def test_hf_stays_online_when_repo_exists_but_pinned_revision_is_missing(
        monkeypatch, module_name, missing):
    module = importlib.import_module(module_name)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    revisions = {
        "embedder": OTHER_COMMIT if missing == "embedder" else EMBED_COMMIT,
        "reranker": OTHER_COMMIT if missing == "reranker" else COMMIT,
    }
    repos = [SimpleNamespace(
        repo_id=f"example/{role}",
        revisions=[SimpleNamespace(commit_hash=commit, refs=frozenset({"main"}))],
    ) for role, commit in revisions.items()]
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir",
                        lambda path=None: SimpleNamespace(repos=repos))

    assert module.hf_offline_if_cached(_config()) is False
    assert "HF_HUB_OFFLINE" not in os.environ


@pytest.mark.parametrize("module_name", MODULES)
def test_hf_unpinned_model_keeps_repo_only_admission(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(module, "_cached_repo_revisions", lambda dirs: {
        "example/embedder": {EMBED_COMMIT},
        "example/reranker": {OTHER_COMMIT},
    })

    assert module.hf_offline_if_cached(_config(revision=None)) is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    monkeypatch.delenv("HF_HUB_OFFLINE")


@pytest.mark.parametrize("module_name", MODULES)
def test_hf_offline_explicit_setting_still_wins_before_cache_scan(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(module, "_cached_repo_revisions",
                        lambda dirs: pytest.fail("explicit setting must skip cache scan"), raising=False)

    assert module.hf_offline_if_cached(_config()) is False
    assert os.environ["HF_HUB_OFFLINE"] == "0"
