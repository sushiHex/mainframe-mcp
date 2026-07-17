"""Improvement pass 2026-07-16 — all three items motivated by the first real
production cycle, not speculation:

1. Empty files reported as "errors" during the first real sync (4 stub files)
   — they're skips, not failures; error counts must stay signal.
2. The HF Hub outage that killed the first sync attempt would equally hang the
   server's pre-warm — go offline at startup when every configured model is
   already cached.
3. That 88-file sync ran 88 full-table hash scans (37K rows each): the exact-
   dedup donor set must be built once per batch and maintained incrementally.
"""

import os
from pathlib import Path

import pytest

from conftest import make_mainframe
from mainframe_mcp import config as config_mod
from mainframe_mcp.dedup import DonorCache


@pytest.fixture
def mainframe(tmp_path):
    return make_mainframe(tmp_path)


def _write(mf, name, body, proj="proj"):
    d = Path(mf.config["paths"]["repos_dir"]) / proj / "docs"
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_text(body, encoding="utf-8")
    return str(f)


# ---------- 1. empty files are skips, not errors ----------

def test_empty_file_is_skipped_not_error(mainframe):
    mf = mainframe
    f = _write(mf, "stub.md", "")
    r = mf.ingest_file(f)
    assert r["status"] == "empty"
    assert "error" not in r
    ingested, skipped, errors = mf._ingest_paths([f])
    assert errors == [] and skipped == 1 and ingested == 0


# ---------- 2. HF offline mode when everything is cached ----------

def _cfg(model_cache="/tmp/models"):
    return {
        "embedder": {"model": "m/emb"},
        "reranker": {"model": "m/rr", "enabled": True},
        "nli": {"model": "m/nli", "enabled": True},
        "consolidator": {"model": "m/cons", "enabled": True},
        "paths": {"model_cache": model_cache},
    }


def test_hf_offline_set_when_all_models_cached(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(config_mod, "_cached_repo_ids",
                        lambda dirs: {"m/emb", "m/rr", "m/nli", "m/cons"})
    assert config_mod.hf_offline_if_cached(_cfg()) is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    monkeypatch.delenv("HF_HUB_OFFLINE")


def test_hf_stays_online_when_any_model_missing(monkeypatch):
    """A missing model (first run, model swap) must keep hub access — going
    offline would turn the lazy consolidator/NLI load into a confusing crash."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(config_mod, "_cached_repo_ids", lambda dirs: {"m/emb", "m/rr"})
    assert config_mod.hf_offline_if_cached(_cfg()) is False
    assert "HF_HUB_OFFLINE" not in os.environ


def test_hf_offline_respects_explicit_user_setting(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")  # user explicitly wants online
    monkeypatch.setattr(config_mod, "_cached_repo_ids",
                        lambda dirs: {"m/emb", "m/rr", "m/nli", "m/cons"})
    assert config_mod.hf_offline_if_cached(_cfg()) is False
    assert os.environ["HF_HUB_OFFLINE"] == "0"  # untouched


def test_hf_disabled_models_not_required(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    cfg = _cfg()
    cfg["consolidator"]["enabled"] = False
    cfg["nli"]["enabled"] = False
    monkeypatch.setattr(config_mod, "_cached_repo_ids", lambda dirs: {"m/emb", "m/rr"})
    assert config_mod.hf_offline_if_cached(cfg) is True
    monkeypatch.delenv("HF_HUB_OFFLINE")


# ---------- 3. one hash scan per batch, not per file ----------

def test_batch_ingest_scans_hash_table_once(mainframe, monkeypatch):
    mf = mainframe
    paths = [_write(mf, f"d{i}.md", f"# D{i}\n\nunique fact number {i} about chunk {i}.\n")
             for i in range(3)]
    scans = []
    orig = mf.store.get_hash_doc_map
    monkeypatch.setattr(mf.store, "get_hash_doc_map", lambda **kw: scans.append(1) or orig(**kw))
    per_file = []
    monkeypatch.setattr(mf.store, "get_existing_hashes",
                        lambda **kw: per_file.append(1) or set())
    ingested, _, errors = mf._ingest_paths(paths)
    assert ingested == 3 and not errors
    assert scans == [1], "batch must build the donor map exactly once"
    assert per_file == [], "no per-file full-table scans inside a batch"


def test_batch_intra_dedup_still_works(mainframe):
    """File 2 duplicating file 1's content within the SAME batch must still be
    caught (the donor cache must absorb file 1's fresh hashes)."""
    mf = mainframe
    body = "# T\n\nIdentical knowledge about qwen reranker chunks here.\n"
    f1 = _write(mf, "one.md", body)
    f2 = _write(mf, "two.md", body)
    ingested, skipped, errors = mf._ingest_paths([f1, f2])
    assert ingested == 1 and skipped == 1 and not errors


def test_batch_modified_file_does_not_self_dupe(mainframe):
    """Re-ingesting an edited file inside a batch must not see its own prior
    chunks as dupes (the data-loss bug class from review finding #1)."""
    mf = mainframe
    f = _write(mf, "doc.md", "# D\n\nStable section about lancedb fragments.\n\n## B\n\nOld extra bit.\n")
    assert mf.ingest_file(f)["status"] == "ingested"
    Path(f).write_text("# D\n\nStable section about lancedb fragments.\n\n## B\n\nNew changed bit.\n",
                       encoding="utf-8")
    ingested, _, errors = mf._ingest_paths([f])
    assert ingested == 1 and not errors
    hits = mf.search("stable section lancedb fragments")
    assert hits, "unchanged section must survive the batched re-ingest"


def test_donor_cache_remove_then_add_roundtrip():
    dc = DonorCache({"h1": {"A", "B"}, "h2": {"A"}})
    assert dc.is_donor("h1", "A")                # h1 also lives in B -> counts
    assert not dc.is_donor("h2", "A")            # h2 only in A -> own-doc exclusion
    assert dc.is_donor("h2", "C")
    dc.remove_doc("A")
    assert not dc.is_donor("h2", "C")            # h2 gone with A
    dc.add_doc("C", {"h3"}, is_donor=True)
    assert dc.is_donor("h3", "A")
    dc.add_doc("S", {"h4"}, is_donor=False)      # session lane: never a donor
    assert not dc.is_donor("h4", "A")
    assert not dc.is_donor("missing", "A")       # unknown hash -> not a donor
