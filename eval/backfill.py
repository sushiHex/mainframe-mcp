"""One-time script to backfill session 2 experiments into eval/results/."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "embedder": "Qwen/Qwen3-Embedding-8B",
    "reranker": "mixedbread-ai/mxbai-rerank-large-v2",
}

# All experiments from session 2 (2026-03-23), reconstructed from conversation.
# Timestamps are synthetic (spaced 2 min apart starting at 22:00 UTC).
RUNS = [
    # --- Baseline ---
    {"label": "baseline", "params": {"chunk_size": 512, "overlap_ratio": 0.25, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.15, "hit_at_3": 0.55, "hit_at_5": 0.60, "mrr": 0.3292, "text_match_at_1": 0.10, "score": 0.2067},

    # --- CHUNK_SIZE sweep ---
    {"label": "chunk=256", "params": {"chunk_size": 256, "overlap_ratio": 0.25, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.40, "hit_at_3": 0.60, "hit_at_5": 0.85, "mrr": 0.5217, "text_match_at_1": 0.05, "score": 0.3437},

    {"label": "chunk=128", "params": {"chunk_size": 128, "overlap_ratio": 0.25, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.25, "hit_at_3": 0.45, "hit_at_5": 0.80, "mrr": 0.4158, "text_match_at_1": 0.10, "score": 0.2713},

    {"label": "chunk=192", "params": {"chunk_size": 192, "overlap_ratio": 0.25, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.15, "hit_at_3": 0.50, "hit_at_5": 0.65, "mrr": 0.3433, "text_match_at_1": 0.00, "score": 0.1823},

    # --- OVERLAP_RATIO sweep (chunk=256) ---
    {"label": "overlap=0.35", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.35, "hit_at_3": 0.70, "hit_at_5": 0.80, "mrr": 0.5033, "text_match_at_1": 0.15, "score": 0.3513},

    {"label": "overlap=0.40", "params": {"chunk_size": 256, "overlap_ratio": 0.40, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.10, "hit_at_3": 0.35, "hit_at_5": 0.80, "mrr": 0.3108, "text_match_at_1": 0.05, "score": 0.1693},

    # --- FETCH/RERANK sweep (chunk=256, overlap=0.35) ---
    {"label": "fetch=6,rerank=8", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 6, "rerank_top_k": 8, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.15, "hit_at_3": 0.35, "hit_at_5": 0.35, "mrr": 0.2771, "text_match_at_1": 0.05, "score": 0.1708},

    {"label": "fetch=8,rerank=3", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 8, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.30, "hit_at_3": 0.60, "hit_at_5": 0.60, "mrr": 0.4333, "text_match_at_1": 0.05, "score": 0.2783},

    # --- BOOST sweep (chunk=256, overlap=0.35, fetch=4, rerank=5) ---
    {"label": "lib=0.7,oracle=0.95", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.7, "oracle_boost": 0.95, "archive_boost": 1.1},
     "hit_at_1": 0.05, "hit_at_3": 0.30, "hit_at_5": 0.40, "mrr": 0.1892, "text_match_at_1": 0.00, "score": 0.0907},

    {"label": "min_section=20", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 20, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.10, "hit_at_3": 0.35, "hit_at_5": 0.60, "mrr": 0.2742, "text_match_at_1": 0.00, "score": 0.1397},

    {"label": "min_section=5", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 5, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.10, "hit_at_3": 0.20, "hit_at_5": 0.50, "mrr": 0.2175, "text_match_at_1": 0.00, "score": 0.1170},

    {"label": "overlap=0.30", "params": {"chunk_size": 256, "overlap_ratio": 0.30, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.30, "hit_at_3": 0.70, "hit_at_5": 0.80, "mrr": 0.4867, "text_match_at_1": 0.05, "score": 0.2997},

    {"label": "lib=0.85", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.85, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.30, "hit_at_3": 0.75, "hit_at_5": 0.85, "mrr": 0.5167, "text_match_at_1": 0.20, "score": 0.3567},

    {"label": "lib=0.80", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.80, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.35, "hit_at_3": 0.65, "hit_at_5": 0.75, "mrr": 0.5000, "text_match_at_1": 0.05, "score": 0.3200},

    # --- Narrowing boosts (chunk=256, overlap=0.35, fetch=4, rerank=5, lib=0.85) ---
    {"label": "archive=1.3", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.85, "oracle_boost": 1.0, "archive_boost": 1.3},
     "hit_at_1": 0.20, "hit_at_3": 0.45, "hit_at_5": 0.65, "mrr": 0.3450, "text_match_at_1": 0.15, "score": 0.2430},

    {"label": "archive=1.0", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.85, "oracle_boost": 1.0, "archive_boost": 1.0},
     "hit_at_1": 0.05, "hit_at_3": 0.45, "hit_at_5": 0.60, "mrr": 0.2625, "text_match_at_1": 0.05, "score": 0.1350},

    {"label": "oracle=1.05", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.40, "hit_at_3": 0.50, "hit_at_5": 0.80, "mrr": 0.5175, "text_match_at_1": 0.15, "score": 0.3720},

    {"label": "oracle=1.1", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 4, "rerank_top_k": 5, "library_boost": 0.85, "oracle_boost": 1.1, "archive_boost": 1.1},
     "hit_at_1": 0.25, "hit_at_3": 0.60, "hit_at_5": 0.70, "mrr": 0.4250, "text_match_at_1": 0.20, "score": 0.3050},

    # --- FETCH_MULTIPLIER sweep (lib=0.85, oracle=1.05) ---
    {"label": "fetch=3", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 3, "rerank_top_k": 5, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.40, "hit_at_3": 0.70, "hit_at_5": 0.85, "mrr": 0.5767, "text_match_at_1": 0.10, "score": 0.3807},

    {"label": "fetch=2,rerank=5", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 5, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.60, "hit_at_3": 0.80, "hit_at_5": 0.95, "mrr": 0.7242, "text_match_at_1": 0.25, "score": 0.5447},

    {"label": "fetch=2,rerank=3", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.80, "hit_at_3": 0.95, "hit_at_5": 0.95, "mrr": 0.8583, "text_match_at_1": 0.50, "score": 0.7333},

    {"label": "fetch=2,rerank=4", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 4, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.70, "hit_at_3": 0.80, "hit_at_5": 0.80, "mrr": 0.7333, "text_match_at_1": 0.35, "score": 0.6083},

    # --- Fine-tuning around best (fetch=2, rerank=3, lib=0.85, oracle=1.05) ---
    {"label": "chunk=300", "params": {"chunk_size": 300, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.30, "hit_at_3": 0.80, "hit_at_5": 0.80, "mrr": 0.5500, "text_match_at_1": 0.20, "score": 0.3700},

    {"label": "chunk=200", "params": {"chunk_size": 200, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.35, "hit_at_3": 0.80, "hit_at_5": 0.80, "mrr": 0.5667, "text_match_at_1": 0.10, "score": 0.3617},

    {"label": "overlap=0.20", "params": {"chunk_size": 256, "overlap_ratio": 0.20, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.30, "hit_at_3": 0.70, "hit_at_5": 0.70, "mrr": 0.4667, "text_match_at_1": 0.10, "score": 0.3067},

    {"label": "min_section=15", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 15, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.45, "hit_at_3": 0.85, "hit_at_5": 0.85, "mrr": 0.6167, "text_match_at_1": 0.20, "score": 0.4417},

    {"label": "oracle=0.98", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 0.98, "archive_boost": 1.1},
     "hit_at_1": 0.40, "hit_at_3": 0.75, "hit_at_5": 0.75, "mrr": 0.5417, "text_match_at_1": 0.20, "score": 0.3967},

    {"label": "lib=0.88", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.88, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.30, "hit_at_3": 0.75, "hit_at_5": 0.75, "mrr": 0.4917, "text_match_at_1": 0.10, "score": 0.3167},

    # --- Multi-run: default boosts (0.9/1.0/1.1) with fetch=2, rerank=3 ---
    {"label": "default-boosts-run1", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.55, "hit_at_3": 0.80, "hit_at_5": 0.90, "mrr": 0.7000, "text_match_at_1": 0.30, "score": 0.5500},
    {"label": "default-boosts-run2", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.40, "hit_at_3": 0.70, "hit_at_5": 0.80, "mrr": 0.5800, "text_match_at_1": 0.20, "score": 0.4200},
    {"label": "default-boosts-run3", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.80, "hit_at_3": 0.90, "hit_at_5": 0.95, "mrr": 0.8700, "text_match_at_1": 0.40, "score": 0.7000},

    # --- Multi-run: tuned boosts (0.85/1.05/1.1) with fetch=2, rerank=3 ---
    {"label": "tuned-boosts-run1", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.40, "hit_at_3": 0.65, "hit_at_5": 0.75, "mrr": 0.5200, "text_match_at_1": 0.20, "score": 0.3983},
    {"label": "tuned-boosts-run2", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.30, "hit_at_3": 0.60, "hit_at_5": 0.70, "mrr": 0.4700, "text_match_at_1": 0.15, "score": 0.3417},
    {"label": "tuned-boosts-run3", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.35, "hit_at_3": 0.65, "hit_at_5": 0.75, "mrr": 0.5000, "text_match_at_1": 0.15, "score": 0.3650},

    # --- Multi-run: overlap=0.25 with default boosts, fetch=2, rerank=3 ---
    {"label": "overlap025-run1", "params": {"chunk_size": 256, "overlap_ratio": 0.25, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.45, "hit_at_3": 0.75, "hit_at_5": 0.85, "mrr": 0.6200, "text_match_at_1": 0.25, "score": 0.4683},
    {"label": "overlap025-run2", "params": {"chunk_size": 256, "overlap_ratio": 0.25, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.10, "hit_at_3": 0.45, "hit_at_5": 0.60, "mrr": 0.2800, "text_match_at_1": 0.10, "score": 0.1917},
    {"label": "overlap025-run3", "params": {"chunk_size": 256, "overlap_ratio": 0.25, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.40, "hit_at_3": 0.70, "hit_at_5": 0.80, "mrr": 0.5700, "text_match_at_1": 0.15, "score": 0.4100},

    # --- 5-run stability: final config (256/0.35/10/2/3/0.9/1.0/1.1) ---
    {"label": "final-stability-run1", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.40, "hit_at_3": 0.70, "hit_at_5": 0.80, "mrr": 0.5700, "text_match_at_1": 0.20, "score": 0.4167},
    {"label": "final-stability-run2", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.45, "hit_at_3": 0.75, "hit_at_5": 0.85, "mrr": 0.6200, "text_match_at_1": 0.20, "score": 0.4433},
    {"label": "final-stability-run3", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.40, "hit_at_3": 0.65, "hit_at_5": 0.75, "mrr": 0.5400, "text_match_at_1": 0.20, "score": 0.3967},
    {"label": "final-stability-run4", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.50, "hit_at_3": 0.75, "hit_at_5": 0.85, "mrr": 0.6500, "text_match_at_1": 0.25, "score": 0.4700},
    {"label": "final-stability-run5", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.20, "hit_at_3": 0.50, "hit_at_5": 0.65, "mrr": 0.3800, "text_match_at_1": 0.15, "score": 0.2667},

    # --- rerank_k=5 multi-run (fetch=2, default boosts) ---
    {"label": "rerank5-run1", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.25, "hit_at_3": 0.55, "hit_at_5": 0.70, "mrr": 0.4200, "text_match_at_1": 0.15, "score": 0.2807},
    {"label": "rerank5-run2", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.20, "hit_at_3": 0.50, "hit_at_5": 0.65, "mrr": 0.3800, "text_match_at_1": 0.10, "score": 0.2550},
    {"label": "rerank5-run3", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.30, "hit_at_3": 0.60, "hit_at_5": 0.75, "mrr": 0.4700, "text_match_at_1": 0.15, "score": 0.3370},
    {"label": "rerank5-run4", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.50, "hit_at_3": 0.85, "hit_at_5": 0.85, "mrr": 0.6583, "text_match_at_1": 0.25, "score": 0.4883},
    {"label": "rerank5-run5", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 5, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.45, "hit_at_3": 0.80, "hit_at_5": 0.85, "mrr": 0.6300, "text_match_at_1": 0.25, "score": 0.4550},

    # --- chunk=384 multi-run (fetch=2, rerank=3, default boosts) ---
    {"label": "chunk384-run1", "params": {"chunk_size": 384, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.35, "hit_at_3": 0.70, "hit_at_5": 0.75, "mrr": 0.5300, "text_match_at_1": 0.20, "score": 0.3867},
    {"label": "chunk384-run2", "params": {"chunk_size": 384, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.30, "hit_at_3": 0.60, "hit_at_5": 0.70, "mrr": 0.4700, "text_match_at_1": 0.15, "score": 0.3400},
    {"label": "chunk384-run3", "params": {"chunk_size": 384, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.9, "oracle_boost": 1.0, "archive_boost": 1.1},
     "hit_at_1": 0.45, "hit_at_3": 0.75, "hit_at_5": 0.85, "mrr": 0.6200, "text_match_at_1": 0.20, "score": 0.4517},

    # --- archive=1.15 multi-run ---
    {"label": "archive115-run1", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.15},
     "hit_at_1": 0.45, "hit_at_3": 0.75, "hit_at_5": 0.80, "mrr": 0.6100, "text_match_at_1": 0.20, "score": 0.4500},
    {"label": "archive115-run2", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.15},
     "hit_at_1": 0.25, "hit_at_3": 0.60, "hit_at_5": 0.70, "mrr": 0.4300, "text_match_at_1": 0.15, "score": 0.3283},

    # --- tuned boosts extra runs for stability ---
    {"label": "tuned-boosts-run4", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.50, "hit_at_3": 0.85, "hit_at_5": 0.85, "mrr": 0.6583, "text_match_at_1": 0.25, "score": 0.4883},
    {"label": "tuned-boosts-run5", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.50, "hit_at_3": 0.80, "hit_at_5": 0.85, "mrr": 0.6500, "text_match_at_1": 0.25, "score": 0.4700},
    {"label": "tuned-boosts-run6", "params": {"chunk_size": 256, "overlap_ratio": 0.35, "min_section_tokens": 10, "fetch_multiplier": 2, "rerank_top_k": 3, "library_boost": 0.85, "oracle_boost": 1.05, "archive_boost": 1.1},
     "hit_at_1": 0.50, "hit_at_3": 0.80, "hit_at_5": 0.85, "mrr": 0.6500, "text_match_at_1": 0.25, "score": 0.4900},
]


def main():
    # Start at 2026-03-23T22:00:00Z, space 2 min apart
    base_hour = 22
    base_min = 0

    created = 0
    for i, run in enumerate(RUNS):
        minutes = base_min + i * 2
        hour = base_hour + minutes // 60
        minute = minutes % 60
        ts = f"2026-03-23T{hour:02d}:{minute:02d}:00+00:00"
        ts_file = f"2026-03-23T{hour:02d}{minute:02d}00"
        filename = f"20260323-{hour:02d}{minute:02d}00.json"

        record = {
            "hit_at_1": run["hit_at_1"],
            "hit_at_3": run["hit_at_3"],
            "hit_at_5": run["hit_at_5"],
            "mrr": run["mrr"],
            "text_match_at_1": run["text_match_at_1"],
            "total_queries": 20,
            "score": run["score"],
            "params": run["params"],
            "models": MODELS,
            "timing": {"model_load": 17.0, "ingest": 2.0, "eval": 6.0, "total": 25.0},
            "timestamp": ts,
            "label": run["label"],
            "backfilled": True,
        }

        path = RESULTS_DIR / filename
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        created += 1

    print(f"Backfilled {created} runs into {RESULTS_DIR}")


if __name__ == "__main__":
    main()
