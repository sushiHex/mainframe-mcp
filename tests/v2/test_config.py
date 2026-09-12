import json
import logging
from pathlib import Path

import pytest

from mainframe.config import DEFAULTS, ConfigError, deep_copy, load_config, load_preset


def test_default_stack_uses_the_measured_harrier_contract(tmp_path, monkeypatch):
    from mainframe.core.encoding import native_contract

    for name in ("MAINFRAME_PRESET", "MAINFRAME_EMBEDDER_MODEL", "MAINFRAME_RERANKER_MODEL",
                 "MAINFRAME_RERANKER_QUANTIZE"):
        monkeypatch.delenv(name, raising=False)
    config = load_config(tmp_path / "first-run.json")
    assert config["embedder"]["model"] == "microsoft/harrier-oss-v1-0.6b"
    contract = native_contract(config)
    assert contract["revision"] == "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"
    assert contract["query_prompt"] == "web_search_query"
    assert contract["dtype"] == "bfloat16" and contract["max_seq_length"] == 2048
    assert config["reranker"]["model"] == "Qwen/Qwen3-Reranker-0.6B"
    assert config["reranker"]["revision"] == "e61197ed45024b0ed8a2d74b80b4d909f1255473"
    assert config["reranker"]["quantize"] is False
    assert not config["consolidator"]["enabled"] and not config["nli"]["enabled"]


@pytest.mark.parametrize("name", ["gpu-max", "gpu-light", "gpu-shared", "v2-harrier"])
def test_gpu_presets_use_the_same_harrier_encoding(name):
    from mainframe.core.encoding import native_contract

    config = load_preset(name)
    assert config["embedder"]["model"] == "microsoft/harrier-oss-v1-0.6b"
    assert native_contract(config) == native_contract(DEFAULTS)
    assert config["reranker"]["model"] == "Qwen/Qwen3-Reranker-0.6B"
    assert config["reranker"]["revision"] == "e61197ed45024b0ed8a2d74b80b4d909f1255473"
    assert config["reranker"]["quantize"] is False


def test_defaults_have_required_sections():
    for key in ("paths", "service", "embedder", "reranker", "consolidator", "nli", "models",
                "chunker", "search", "tiers", "index", "capture", "memory", "contextual"):
        assert key in DEFAULTS, key
    assert DEFAULTS["service"]["port"] == 7433 and DEFAULTS["service"]["prewarm"] is False
    assert DEFAULTS["index"]["cleanup_minutes"] == 120
    assert DEFAULTS["index"]["read_consistency_seconds"] == 30
    assert DEFAULTS["capture"]["daily_cap"] == 20 and DEFAULTS["capture"]["min_prompts"] == 2
    assert DEFAULTS["memory"]["threshold"] == 5 and DEFAULTS["memory"]["vram_footprint_gb"] == 3.5
    assert DEFAULTS["search"]["candidate_pool"] == 20
    assert DEFAULTS["tiers"]["session"] == 1.05


def test_load_config_merges_file_and_env(tmp_path, monkeypatch):
    cfgp = tmp_path / "config.json"
    cfgp.write_text(json.dumps({"service": {"port": 8000}, "paths": {"repos_dir": "~/r"}}), encoding="utf-8")
    monkeypatch.setenv("MAINFRAME_CONFIG", str(cfgp))
    monkeypatch.setenv("MAINFRAME_PORT", "9001")
    monkeypatch.setenv("MAINFRAME_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    c = load_config()
    assert c["service"]["port"] == 9001                      # env beats file
    assert c["reranker"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert "~" not in c["paths"]["repos_dir"]                 # expanded
    assert c["chunker"]["chunk_size"] == 256                  # default retained


def test_v1_embedder_config_without_encoding_keeps_legacy_semantics(tmp_path, monkeypatch):
    old = tmp_path / "old-config.json"
    old.write_text(json.dumps({"embedder": {
        "model": "Qwen/Qwen3-Embedding-8B", "quantize": True,
        "batch_size": 8, "max_seq_length": 2048,
        "query_prefix": "Instruct: Find project knowledge\nQuery: ",
    }}), encoding="utf-8")
    monkeypatch.delenv("MAINFRAME_PRESET", raising=False)
    monkeypatch.delenv("MAINFRAME_EMBEDDER_MODEL", raising=False)

    embedder = load_config(old)["embedder"]

    assert embedder["model"] == "Qwen/Qwen3-Embedding-8B"
    assert embedder["encoding"] == "legacy"
    assert embedder["revision"] is None and embedder["query_prompt"] is None
    assert embedder["quantize"] is True
    assert embedder["query_prefix"].startswith("Instruct:")


def test_embedder_model_env_override_replaces_native_contract(tmp_path, monkeypatch):
    configured = tmp_path / "native.json"
    configured.write_text(json.dumps({"embedder": {"batch_size": 3}}), encoding="utf-8")
    monkeypatch.delenv("MAINFRAME_PRESET", raising=False)
    monkeypatch.setenv("MAINFRAME_EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5")

    embedder = load_config(configured)["embedder"]

    assert embedder["model"] == "BAAI/bge-small-en-v1.5"
    assert embedder["encoding"] == "legacy"
    assert embedder["revision"] is None and embedder["query_prompt"] is None
    assert embedder["batch_size"] == 3


def test_changed_native_model_must_supply_its_own_contract(tmp_path, monkeypatch):
    custom = tmp_path / "custom-native.json"
    custom.write_text(json.dumps({"embedder": {
        "model": "example/native-embedder", "encoding": "native", "quantize": False,
    }}), encoding="utf-8")
    monkeypatch.delenv("MAINFRAME_PRESET", raising=False)
    monkeypatch.delenv("MAINFRAME_EMBEDDER_MODEL", raising=False)

    with pytest.raises(ConfigError, match="pinned 40-character embedder.revision"):
        load_config(custom)


def test_changed_native_model_accepts_an_explicit_complete_contract(tmp_path, monkeypatch):
    custom = tmp_path / "custom-native.json"
    custom.write_text(json.dumps({"embedder": {
        "model": "example/native-embedder", "encoding": "native",
        "revision": "b" * 40, "query_prompt": None, "quantize": False,
    }}), encoding="utf-8")
    monkeypatch.delenv("MAINFRAME_PRESET", raising=False)
    monkeypatch.delenv("MAINFRAME_EMBEDDER_MODEL", raising=False)

    embedder = load_config(custom)["embedder"]

    assert embedder["encoding"] == "native"
    assert embedder["revision"] == "b" * 40
    assert embedder["query_prompt"] is None


def test_explicit_path_beats_env(tmp_path, monkeypatch):
    a = tmp_path / "a.json"; a.write_text(json.dumps({"service": {"port": 1}}), encoding="utf-8")
    b = tmp_path / "b.json"; b.write_text(json.dumps({"service": {"port": 2}}), encoding="utf-8")
    monkeypatch.setenv("MAINFRAME_CONFIG", str(a))
    assert load_config(b)["service"]["port"] == 2


def test_presets_still_load():
    c = load_preset("cpu-only")
    assert c["consolidator"]["enabled"] is False
    assert c["service"]["port"] == 7433  # preset merges over DEFAULTS


def test_reranker_quantization_can_be_configured_in_json(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"reranker": {"quantize": False}}), encoding="utf-8")
    monkeypatch.delenv("MAINFRAME_RERANKER_QUANTIZE", raising=False)
    assert load_config(path)["reranker"]["quantize"] is False
    monkeypatch.setenv("MAINFRAME_RERANKER_QUANTIZE", "true")
    assert load_config(path)["reranker"]["quantize"] is True
    assert DEFAULTS["reranker"]["quantize"] is False


def test_a_broken_config_refuses_instead_of_widening_the_scope(tmp_path, monkeypatch):
    """A truncated file used to log a warning and fall back to DEFAULTS — where
    `include_projects` is empty, i.e. EVERY project under repos_dir. A daemon
    scoped to two repos would silently start ingesting two hundred."""
    scoped = {"paths": {"include_projects": ["mainframe-mcp"]}}
    good = tmp_path / "good.json"
    good.write_text(json.dumps(scoped), encoding="utf-8")
    assert load_config(good)["paths"]["include_projects"] == ["mainframe-mcp"]

    truncated = tmp_path / "truncated.json"
    truncated.write_text(json.dumps(scoped)[:-8], encoding="utf-8")
    with pytest.raises(ConfigError, match="truncated.json"):
        load_config(truncated)

    not_an_object = tmp_path / "list.json"
    not_an_object.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON object"):
        load_config(not_an_object)

    # a MISSING file is a first run, and the defaults are right for it
    monkeypatch.delenv("MAINFRAME_CONFIG", raising=False)
    assert load_config(tmp_path / "absent.json")["paths"]["include_projects"] == []


def test_an_unknown_config_key_is_refused_by_name(tmp_path):
    """A typo used to merge silently, so the user believed a setting was in
    force and nothing read it. `chunker.min_section_tokens` and
    `index.vector_index_rows` were exactly that in reverse: shipped defaults
    with no reader at all."""
    bad = tmp_path / "typo.json"
    bad.write_text(json.dumps({"chunker": {"chunk_sizes": 512}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="chunker.chunk_sizes"):
        load_config(bad)
    assert "chunk_size" in str(pytest.raises(ConfigError, load_config, bad).value)   # names the neighbours

    section = tmp_path / "section.json"
    section.write_text(json.dumps({"chunkr": {"chunk_size": 512}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="chunkr"):
        load_config(section)

    assert "min_section_tokens" not in DEFAULTS["chunker"]
    assert "vector_index_rows" not in DEFAULTS["index"]


def test_v1_era_keys_and_comments_are_tolerated(tmp_path, caplog):
    """v1 is still installed and reads the SAME config file and presets, so
    keys only IT owns are not typos. Refusing them would brick the daemon on
    every machine that has both — but an exemption that passes in SILENCE is,
    from the operator's chair, indistinguishable from a setting in force."""
    shared = tmp_path / "shared.json"
    shared.write_text(json.dumps({
        "_note": "a comment", "reranker": {"top_k": 3}, "search": {"fetch_multiplier": 2},
        "chunker": {"min_section_tokens": 10}, "tiers": {"archive": 1.1},
        "memory": {"consolidate_threshold": 5}, "service": {"port": 8123},
    }), encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="mainframe.config"):
        assert load_config(shared)["service"]["port"] == 8123
    said = "\n".join(caplog.messages)
    for key in ("reranker.top_k", "search.fetch_multiplier", "chunker.min_section_tokens",
                "tiers.archive", "memory.consolidate_threshold"):
        assert key in said and "does not read it" in said
    for name in ("cpu-only", "gpu-light", "gpu-max", "gpu-minimal"):
        assert load_preset(name)["service"]["port"] == 7433


def test_deep_copy_is_independent():
    a = deep_copy(DEFAULTS)
    a["search"]["candidate_pool"] = 1
    assert DEFAULTS["search"]["candidate_pool"] == 20
