"""Configuration — single source of truth for models, lanes, index, capture,
memory, and service knobs. Top-level model sections keep the v1
shape so the kept modules and the shipped presets read them unchanged."""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".claude" / "mainframe" / "config.json"
PRESETS_DIR = Path(__file__).resolve().parents[2] / "configs"


class ConfigError(RuntimeError):
    """A config file that EXISTS but cannot be read or parsed.

    Falling back to DEFAULTS looks harmless until you notice what the defaults
    say: `include_projects` is empty, which means EVERY project under
    `repos_dir`. A truncated file would turn a carefully scoped daemon into an
    all-repository ingest, quietly, on the next restart. Refusing to start is
    the safe failure; a MISSING file still gets the defaults, which are
    correct for a first run."""

DEFAULTS = {
    "paths": {
        "mainframe_dir": str(Path.home() / ".claude" / "mainframe"),
        "repos_dir": str(Path.home() / "repos"),
        "model_cache": str(Path.home() / ".claude" / "mainframe" / ".models"),
        "include_projects": [],
        "exclude_projects": [],
    },
    "service": {"port": 7433, "token": None, "prewarm": False, "tick_seconds": 5,
                "shutdown_timeout_seconds": 300},
    "embedder": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "quantize": True,
        "batch_size": 8,
        "max_seq_length": 2048,
        "query_prefix": "Instruct: Find the most relevant code documentation or knowledge base entry\nQuery: ",
        "encoding": "legacy", "revision": None, "query_prompt": None,
    },
    "reranker": {"model": "Qwen/Qwen3-Reranker-4B", "enabled": True,
                 "heading_inject": True, "quantize": True},
    "consolidator": {"model": "Qwen/Qwen2.5-3B-Instruct", "enabled": True, "quantize": True,
                     "max_new_tokens": 4096},
    "nli": {"model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "enabled": True,
            "threshold": 0.7},
    "models": {"retry_minutes": 15},
    "chunker": {"chunk_size": 256, "overlap_ratio": 0.35, "strip_frontmatter": False},
    "search": {"candidate_pool": 20, "rerank_top_k": 3, "recency_weight": 0.0,
               "recency_halflife_days": 90},
    "tiers": {"library": 0.90, "project": 0.93, "docs": 0.95, "research": 0.97, "session": 1.05},
    "index": {"debounce_seconds": 10, "rescan_hours": 6, "cleanup_minutes": 120,
              "read_consistency_seconds": 30, "batch_cap": 200,
              "optimize_versions": 20, "optimize_fragments": 10, "optimize_cooldown_seconds": 900},
    "capture": {"exclude_projects": [], "min_prompts": 2, "daily_cap": 20,
                "max_transcript_bytes": 8 * 1024 * 1024},
    "memory": {"threshold": 5, "settle_minutes": 30, "check_interval_minutes": 5,
               "vram_footprint_gb": 3.5, "vram_margin_gb": 1.5, "char_budget": 80_000},
    "contextual": {"enabled": False, "model": "claude-haiku-4-5"},
}

_ENV_OVERRIDES = {
    "MAINFRAME_EMBEDDER_MODEL": ("embedder", "model", str),
    "MAINFRAME_RERANKER_MODEL": ("reranker", "model", str),
    "MAINFRAME_RERANKER_QUANTIZE": ("reranker", "quantize",
                                    lambda v: v.strip().lower() in ("1", "true", "yes")),
    "MAINFRAME_NLI_MODEL": ("nli", "model", str),
    "MAINFRAME_CONSOLIDATOR_MODEL": ("consolidator", "model", str),
    "MAINFRAME_CHUNK_SIZE": ("chunker", "chunk_size", int),
    "MAINFRAME_OVERLAP": ("chunker", "overlap_ratio", float),
    "MAINFRAME_REPOS_DIR": ("paths", "repos_dir", str),
    "MAINFRAME_DIR": ("paths", "mainframe_dir", str),
    "MAINFRAME_PORT": ("service", "port", int),
}


def deep_copy(d: dict) -> dict:
    return json.loads(json.dumps(d))


# v1 (`mainframe_mcp`) is still installed and reads the SAME
# `~/.claude/mainframe/config.json` and the same `configs/` presets, so a live
# config legitimately carries keys v2 does not own. These are not typos, and
# refusing to start over them would brick the daemon on every machine that has
# both. Keep these exemptions while v1 and v2 share configuration files.
_V1_ONLY_KEYS = frozenset({"reranker.top_k", "chunker.min_section_tokens", "search.fetch_multiplier",
                           "tiers.archive", "memory.consolidate_threshold"})


def _deep_merge(base: dict, override: dict, source: str = "", prefix: str = "") -> None:
    """Merge `override` into `base`.

    With a `source`, a key `base` does not define is REFUSED instead of merged:
    an unknown key is a typo, and a typo that merges silently becomes a setting
    the user believes is in force and which nothing ever reads. A leading
    underscore marks a comment (`_name`, `_vram` in the presets) and is allowed
    anywhere. Presets pass no `source` — they are repo-shipped, not hand-typed,
    and deliberately shared with v1."""
    for key, val in override.items():
        path = f"{prefix}{key}"
        if source and key not in base and not key.startswith("_"):
            if path not in _V1_ONLY_KEYS:
                raise ConfigError(
                    f"unknown config key {path!r} in {source}: nothing reads it, so it would be "
                    f"silently ignored. Known keys here: {', '.join(sorted(base)) or '(none)'}")
            # Say so out loud. An exemption that passes in silence is
            # indistinguishable, from the operator's chair, from a setting that
            # is in force — which is the very confusion the refusal exists to end.
            logger.info("config key %r in %s belongs to v1 (mainframe_mcp); the v2 daemon does not read it",
                        path, source)
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val, source, f"{path}.")
        else:
            base[key] = val


def load_preset(name: str) -> dict:
    preset_path = PRESETS_DIR / f"{name}.json"
    if not preset_path.exists():
        available = [p.stem for p in PRESETS_DIR.glob("*.json")] if PRESETS_DIR.exists() else []
        raise FileNotFoundError(f"Preset '{name}' not found. Available: {', '.join(available)}")
    config = deep_copy(DEFAULTS)
    _deep_merge(config, json.loads(preset_path.read_text(encoding="utf-8")))
    return config


def list_presets() -> list[dict]:
    if not PRESETS_DIR.exists():
        return []
    out = []
    for p in sorted(PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({"name": p.stem, "description": data.get("_name", ""),
                    "vram": data.get("_vram", "unknown")})
    return out


def load_config(config_path: Path | None = None) -> dict:
    """explicit path > MAINFRAME_CONFIG > ~/.claude/mainframe/config.json; then
    MAINFRAME_PRESET as the base; env overrides applied last; paths expanded."""
    if config_path is None:
        env_path = os.environ.get("MAINFRAME_CONFIG")
        config_path = Path(env_path).expanduser() if env_path else DEFAULT_CONFIG_PATH
    preset = os.environ.get("MAINFRAME_PRESET")
    config = load_preset(preset) if preset else deep_copy(DEFAULTS)
    if Path(config_path).exists():
        try:
            data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ConfigError(f"cannot read config {config_path}: {e}") from e
        if not isinstance(data, dict):
            raise ConfigError(f"config {config_path} must be a JSON object, got {type(data).__name__}")
        _deep_merge(config, data, source=str(config_path))
        logger.info(f"Loaded config from {config_path}")
    for env_key, (section, key, cast) in _ENV_OVERRIDES.items():
        val = os.environ.get(env_key)
        if val is not None:
            config.setdefault(section, {})[key] = cast(val)
    for k, v in config.get("paths", {}).items():
        if isinstance(v, str):  # include_projects/exclude_projects are glob lists, not paths
            config["paths"][k] = str(Path(v).expanduser())
    from mainframe.core.encoding import native_contract
    try:
        native_contract(config)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    return config


def _cached_repo_ids(cache_dirs: list) -> set:
    from huggingface_hub import scan_cache_dir
    repos = set()
    for d in cache_dirs:
        try:
            info = scan_cache_dir(d) if d else scan_cache_dir()
            repos |= {r.repo_id for r in info.repos}
        except Exception:
            continue
    return repos


def hf_offline_if_cached(config: dict) -> bool:
    """Set HF_HUB_OFFLINE=1 when every ENABLED model is already cached, so a hub
    outage cannot hang a model load. Respects an explicit
    user setting; stays online when any model is missing."""
    if "HF_HUB_OFFLINE" in os.environ:
        return False
    models = [config["embedder"]["model"]]
    for key in ("reranker", "nli", "consolidator"):
        section = config.get(key, {})
        if section.get("enabled", True):
            models.append(section["model"])
    cached = _cached_repo_ids([None, config["paths"].get("model_cache")])
    if all(m in cached for m in models):
        os.environ["HF_HUB_OFFLINE"] = "1"
        return True
    return False
