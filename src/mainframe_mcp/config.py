"""Mainframe MCP configuration — single source of truth for all models and parameters.

Users swap models by editing config.json or setting environment variables.
No source code changes needed.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path.home() / ".claude" / "mainframe" / "config.json"

# Default configuration — used if no config.json exists
DEFAULTS = {
    # Embedding model
    "embedder": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "quantize": True,           # INT8 via bitsandbytes
        "batch_size": 8,
        "max_seq_length": 2048,     # 8x the 256-tok chunk target; 8192 made dense
                                    # chunks O(seq^2)-slow and overflowed the INT8
                                    # kernel grid (batch*seq) at batch 8
        "query_prefix": "Instruct: Find the most relevant code documentation or knowledge base entry\nQuery: ",
    },

    # Reranker. Qwen3-Reranker-4B via the native qwen3-logit backend scored
    # 0.4187 vs bge's 0.3320 (+26%) on the pool-20 harness (2026-07-01d) at the
    # cost of ~8s/query (vs ~2-4s) and ~4.5GB VRAM (vs 1.2GB). Rollback:
    # MAINFRAME_RERANKER_MODEL=BAAI/bge-reranker-v2-m3.
    "reranker": {
        "model": "Qwen/Qwen3-Reranker-4B",
        "enabled": True,
        "top_k": 3,
        "heading_inject": True,
    },

    # NLI contradiction detection
    "nli": {
        "model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "enabled": True,
        "threshold": 0.7,
    },

    # Consolidation LLM
    "consolidator": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "enabled": True,
        "quantize": True,           # Q4 via bitsandbytes
        "max_new_tokens": 4096,
    },

    # Chunking
    "chunker": {
        "chunk_size": 256,          # tokens (autoresearch optimized)
        "overlap_ratio": 0.35,      # autoresearch optimized
        "min_section_tokens": 10,
    },

    # Search
    "search": {
        "fetch_multiplier": 2,      # autoresearch optimized
        "rerank_top_k": 3,          # EVAL-ONLY: the live search path now honors
                                    # the caller's `limit` (default 3, cap 10)
                                    # instead of this; eval + conftest still read it
        # Floor for the vector+FTS candidate pool fed to the reranker, decoupled
        # from the caller's result limit — a small `limit` must not starve the
        # cross-encoder (10-POV review: pool starvation drove the 0.48->0.38
        # regression as the corpus tripled). Brute-force exact search makes a
        # wider pool nearly free; tune against the pool-sweep results.
        "candidate_pool": 20,
        # Temporal ranking (mem0-style): nudges a recent chunk higher in the
        # candidate pool (the reranker may still reorder — not a guarantee). OFF
        # by default: it keys off created_at = INGEST time (not authored time), so
        # re-ingesting an old file makes it look recent. Opt in only if your corpus
        # has meaningfully divergent, stable ingest ages.
        "recency_weight": 0.0,      # 0 disables; max relative boost for a brand-new chunk
        "recency_halflife_days": 90,
    },

    # Memory loop: sync_index/status surface per-project pending session-capture
    # counts; at this threshold the sync result carries a consolidation hint.
    "memory": {
        "consolidate_threshold": 5,
    },

    # API contextual retrieval at ingest (Anthropic method) — the ONE deliberate
    # exception to 100%-local, and OFF by default: enabling sends document
    # content to the Anthropic API at INGEST time (serving stays local). Needs
    # ANTHROPIC_API_KEY / an `ant auth login` profile in the server's env.
    "contextual": {
        "enabled": False,
        "model": "claude-haiku-4-5",
    },

    # Tier boosting (lower = higher priority)
    "tiers": {
        "library": 0.90,
        "project": 0.93,
        "docs": 0.95,
        "research": 0.97,
        "session": 1.05,   # below research, above archive — marginal FTS-tail aid only
        "archive": 1.10,
    },

    # Paths
    "paths": {
        "mainframe_dir": str(Path.home() / ".claude" / "mainframe"),
        "repos_dir": str(Path.home() / "repos"),
        "model_cache": str(Path.home() / ".claude" / "mainframe" / ".models"),
    },
}


PRESETS_DIR = Path(__file__).parent.parent.parent / "configs"


def load_preset(name: str) -> dict:
    """Load a named preset config (e.g., 'gpu-max', 'cpu-only').

    Presets live in the configs/ directory of the mainframe-mcp repo.
    """
    preset_path = PRESETS_DIR / f"{name}.json"
    if not preset_path.exists():
        available = [p.stem for p in PRESETS_DIR.glob("*.json")] if PRESETS_DIR.exists() else []
        raise FileNotFoundError(
            f"Preset '{name}' not found. Available: {', '.join(available)}"
        )
    config = _deep_copy(DEFAULTS)
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    _deep_merge(config, preset)
    logger.info(f"Loaded preset: {name} ({preset.get('_name', '')})")
    return config


def list_presets() -> list[dict]:
    """List all available presets with their descriptions."""
    if not PRESETS_DIR.exists():
        return []
    presets = []
    for p in sorted(PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            presets.append({
                "name": p.stem,
                "description": data.get("_name", ""),
                "vram": data.get("_vram", "unknown"),
            })
        except Exception:
            pass
    return presets


def load_config(config_path: Path | None = None) -> dict:
    """Load config from JSON file, falling back to defaults.

    Priority: MAINFRAME_PRESET env > config.json > environment variables > defaults.

    The config file path resolves as: explicit `config_path` arg > `MAINFRAME_CONFIG`
    env (portable override for non-Claude MCP hosts) > the Claude-Code default
    (`~/.claude/mainframe/config.json`). The default is kept for back-compat.
    """
    if config_path is None:
        env_path = os.environ.get("MAINFRAME_CONFIG")
        config_path = Path(env_path).expanduser() if env_path else DEFAULT_CONFIG_PATH
    # Base: a named preset (if selected) else DEFAULTS + config.json.
    preset_name = os.environ.get("MAINFRAME_PRESET")
    if preset_name:
        config = load_preset(preset_name)
    else:
        config = _deep_copy(DEFAULTS)
        if config_path.exists():
            try:
                user_config = json.loads(config_path.read_text(encoding="utf-8"))
                _deep_merge(config, user_config)
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

    # Environment variable overrides — applied LAST so they win over a preset too
    # (e.g. MAINFRAME_PRESET=gpu-max plus MAINFRAME_EMBEDDER_MODEL=...).
    env_overrides = {
        "MAINFRAME_EMBEDDER_MODEL": ("embedder", "model"),
        "MAINFRAME_RERANKER_MODEL": ("reranker", "model"),
        "MAINFRAME_RERANKER_QUANTIZE": ("reranker", "quantize",
                                        lambda v: v.strip().lower() in ("1", "true", "yes")),
        "MAINFRAME_NLI_MODEL": ("nli", "model"),
        "MAINFRAME_CONSOLIDATOR_MODEL": ("consolidator", "model"),
        "MAINFRAME_CHUNK_SIZE": ("chunker", "chunk_size", int),
        "MAINFRAME_OVERLAP": ("chunker", "overlap_ratio", float),
        "MAINFRAME_REPOS_DIR": ("paths", "repos_dir"),
        "MAINFRAME_DIR": ("paths", "mainframe_dir"),
    }

    for env_key, path in env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            if len(path) == 3:
                section, key, cast = path
                config[section][key] = cast(val)
            else:
                section, key = path
                config[section][key] = val
            logger.info(f"Override from env: {env_key}={val}")

    # Expand portable "~/..." paths (config.example.json ships these so the
    # example never hardcodes a username).
    for k, v in config.get("paths", {}).items():
        config["paths"][k] = str(Path(v).expanduser())

    return config


def _cached_repo_ids(cache_dirs: list) -> set:
    """repo_ids present across the given HF cache dirs (None = default cache)."""
    from huggingface_hub import scan_cache_dir
    repos = set()
    for d in cache_dirs:
        try:
            info = scan_cache_dir(d) if d else scan_cache_dir()
            repos |= {r.repo_id for r in info.repos}
        except Exception:  # missing dir, hub lib quirk — just means "not cached here"
            continue
    return repos


def hf_offline_if_cached(config: dict) -> bool:
    """Set HF_HUB_OFFLINE=1 when every ENABLED model is already in an HF cache.

    Hit live 2026-07-16: a HuggingFace Hub outage put sentence-transformers
    into minutes of per-config-file 504 retries with all weights cached,
    killing a sync and (identically) able to hang the server pre-warm. Offline
    mode skips the hub checks entirely (also faster). Stays ONLINE when any
    model is missing (first run / model swap — the lazy consolidator/NLI load
    would otherwise crash confusingly), and respects an explicit user setting.
    Checks BOTH the default cache and paths.model_cache (embedder/consolidator
    download there)."""
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
        logger.info("All models cached — HF_HUB_OFFLINE=1 (hub outages can't hang loads)")
        return True
    return False


def save_config(config: dict, config_path: Path = DEFAULT_CONFIG_PATH):
    """Save current config to JSON file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Saved config to {config_path}")


def _deep_copy(d: dict) -> dict:
    return json.loads(json.dumps(d))


def _deep_merge(base: dict, override: dict):
    """Merge override into base, recursively."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
