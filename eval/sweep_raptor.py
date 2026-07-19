"""Sweep RAPTOR distance_threshold and min_cluster_size."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mainframe_mcp.config import hf_offline_if_cached, load_config
from mainframe_mcp.consolidator import Consolidator
from mainframe_mcp.embedder import Embedder
from mainframe_mcp.raptor import build_raptor_summaries
from mainframe_mcp.reranker import Reranker
from mainframe_mcp.store import SESSION_SOURCE_TYPE, Store

MAINFRAME_DIR = Path.home() / ".claude" / "mainframe"
EVAL_DIR = Path(__file__).parent
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"
BOOST = {"library": 0.9, "project": 0.93, "docs": 0.95, "research": 0.97, "session": 1.05, "archive": 1.1}
FM, RK = 2, 3


def evaluate(embedder, reranker, store, test_queries):
    hits_at_1 = 0
    hits_at_3 = 0
    text_matches = 0
    reciprocal_ranks = []

    for tq in test_queries:
        q_emb = embedder.embed_query(tq["query"])
        raw = store.search(q_emb, top_k=RK * FM, query_text=tq["query"], exclude_source_type=SESSION_SOURCE_TYPE)
        if not raw:
            reciprocal_ranks.append(0.0)
            continue
        texts = [r["text"] for r in raw]
        headings = [r.get("heading", "") for r in raw]
        reranked = reranker.rerank(tq["query"], texts, top_k=RK, headings=headings)
        found_rank = None
        for rank, (oi, sc) in enumerate(reranked, 1):
            if tq["expected_file"] in raw[oi].get("doc_path", ""):
                if found_rank is None:
                    found_rank = rank
                if rank == 1:
                    hits_at_1 += 1
                    if tq.get("expected_text_contains", "").lower() in raw[oi]["text"].lower():
                        text_matches += 1
                if rank <= 3:
                    hits_at_3 += 1
                break
        reciprocal_ranks.append(1 / found_rank if found_rank else 0)

    n = len(test_queries)
    mrr = sum(reciprocal_ranks) / n
    score = mrr * 0.4 + hits_at_1 / n * 0.3 + text_matches / n * 0.3
    return {
        "score": round(score, 4), "h@1": round(hits_at_1 / n, 4),
        "h@3": round(hits_at_3 / n, 4), "mrr": round(mrr, 4),
        "txt": round(text_matches / n, 4),
    }


def main():
    config = load_config()
    hf_offline_if_cached(config)  # hub outages must not hang cached loads
    test_queries = json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))

    print("Loading all models...", flush=True)
    embedder = Embedder(config)
    reranker = Reranker(config)
    consolidator = Consolidator(config)
    store = Store(
        db_path=MAINFRAME_DIR / ".lancedb",
        embedding_dim=embedder.dimension,
        tier_boost=BOOST,
    )
    print("All models loaded.\n", flush=True)

    # First, eval without any tier-1 (remove existing)
    try:
        store.table.delete("tier = 1")
        store.table = store.db.open_table("chunks")
    except Exception:
        pass

    r = evaluate(embedder, reranker, store, test_queries)
    print(f"No RAPTOR:     score={r['score']:.4f}  h@1={r['h@1']:.4f}  h@3={r['h@3']:.4f}  mrr={r['mrr']:.4f}  txt={r['txt']:.4f}\n", flush=True)

    # Sweep thresholds
    configs = [
        (0.4, 3), (0.5, 3), (0.6, 3), (0.7, 3), (0.8, 3), (1.0, 3),
        (0.6, 2), (0.6, 5), (0.8, 2), (0.8, 5),
    ]

    for threshold, min_size in configs:
        t0 = time.time()
        stats = build_raptor_summaries(
            store=store, embedder=embedder, consolidator=consolidator,
            distance_threshold=threshold, min_cluster_size=min_size,
        )
        build_time = time.time() - t0

        r = evaluate(embedder, reranker, store, test_queries)
        total_time = time.time() - t0

        n_summaries = stats["summaries_generated"]
        print(f"t={threshold:.1f} min={min_size}:  score={r['score']:.4f}  h@1={r['h@1']:.4f}  h@3={r['h@3']:.4f}  mrr={r['mrr']:.4f}  txt={r['txt']:.4f}  summaries={n_summaries:3d}  ({build_time:.0f}s build, {total_time:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
