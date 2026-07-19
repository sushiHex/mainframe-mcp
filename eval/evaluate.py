"""RAG evaluation harness for autoresearch parameter optimization.

Two modes:
  --live     Use the existing Mainframe index (fast, search params only)
  --rebuild  Rebuild index from scratch (slow, tests chunking params too)

Metrics: hit@k, MRR, text_match@1. Composite score for autoresearch.
"""

import json
import sys
import time
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mainframe_mcp.chunker import chunk_markdown
from mainframe_mcp.dedup import content_hash
from mainframe_mcp.embedder import Embedder
from mainframe_mcp.reranker import Reranker
from mainframe_mcp.store import SESSION_SOURCE_TYPE, Store

# ----- Configurable parameters (autoresearch modifies these) -----
CHUNK_SIZE = 256            # tokens per chunk
OVERLAP_RATIO = 0.35        # overlap between chunks
MIN_SECTION_TOKENS = 10     # merge threshold for tiny sections
FETCH_MULTIPLIER = 2        # overfetch ratio for reranking
RERANK_TOP_K = 3            # how many results after reranking
CANDIDATE_POOL = None       # vector+FTS pool floor fed to the reranker. None ->
                            # read production's search.candidate_pool from config
                            # at startup (single source of truth); set an int here
                            # only to override for a sweep. The old harness used
                            # RERANK_TOP_K*FETCH_MULTIPLIER=6 while prod used 20,
                            # so the eval measured a narrower pool than users ever
                            # hit (10-POV review).
LIBRARY_BOOST = 0.9         # tier boost for library (lower = more boosted)
PROJECT_BOOST = 0.93        # tier boost for project CLAUDE.md
DOCS_BOOST = 0.95           # tier boost for docs
RESEARCH_BOOST = 0.97       # tier boost for research
ARCHIVE_BOOST = 1.1         # tier boost for archive
# ------------------------------------------------------------------

MAINFRAME_DIR = Path.home() / ".claude" / "mainframe"
EVAL_DIR = Path(__file__).parent
# Your own corpus-specific query set lives in test_queries.json (gitignored —
# eval fixtures inevitably describe YOUR private corpus). The tracked
# test_queries.example.json shows the format and targets this repo's own docs.
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"
if not TEST_QUERIES_PATH.exists():
    TEST_QUERIES_PATH = EVAL_DIR / "test_queries.example.json"
TEMP_DB = EVAL_DIR / ".eval-lancedb"


def evaluate(
    embedder: Embedder,
    reranker: Reranker,
    store: Store,
    test_queries: list[dict],
) -> dict:
    """Run test queries and compute retrieval metrics."""
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    text_matches = 0
    reciprocal_ranks = []

    for tq in test_queries:
        query = tq["query"]
        expected_file = tq["expected_file"]
        expected_text = tq.get("expected_text_contains", "")

        q_emb = embedder.embed_query(query)
        # Mirror production: session captures are excluded from default search,
        # and the candidate pool is floored independently of the result count.
        # `or 0` guards a direct evaluate() call before main() resolved the
        # None sentinel (max(None, int) raises TypeError).
        pool = max(CANDIDATE_POOL or 0, RERANK_TOP_K * FETCH_MULTIPLIER)
        raw_results = store.search(q_emb, top_k=pool,
                                   query_text=query, exclude_source_type=SESSION_SOURCE_TYPE)

        if not raw_results:
            reciprocal_ranks.append(0.0)
            continue

        texts = [r["text"] for r in raw_results]
        headings = [r.get("heading", "") for r in raw_results]
        # Mirror production: category passed unconditionally — the Reranker
        # owns the category_instructions gate (off by default).
        reranked = reranker.rerank(query, texts, top_k=RERANK_TOP_K, headings=headings,
                                   category=embedder._classify_query(query))

        found_rank = None
        for rank, (orig_idx, score) in enumerate(reranked, 1):
            r = raw_results[orig_idx]
            doc_path = r.get("doc_path", "")
            if expected_file in doc_path:
                if found_rank is None:
                    found_rank = rank
                if rank == 1:
                    hits_at_1 += 1
                    if expected_text.lower() in r["text"].lower():
                        text_matches += 1
                if rank <= 3:
                    hits_at_3 += 1
                if rank <= 5:
                    hits_at_5 += 1
                break

        reciprocal_ranks.append(1.0 / found_rank if found_rank else 0.0)

    n = len(test_queries)
    mrr = sum(reciprocal_ranks) / n if n else 0

    return {
        "hit_at_1": hits_at_1 / n if n else 0,
        "hit_at_3": hits_at_3 / n if n else 0,
        "hit_at_5": hits_at_5 / n if n else 0,
        "mrr": mrr,
        "text_match_at_1": text_matches / n if n else 0,
        "total_queries": n,
        "score": (mrr * 0.4 + hits_at_1 / n * 0.3 + text_matches / n * 0.3) if n else 0,
    }


def main():
    """Run evaluation. Use --live for existing index, --rebuild for fresh index."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use existing Mainframe index (fast)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild temp index from library files")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Regression gate: exit 1 if the composite score falls below this")
    args = parser.parse_args()

    use_live = args.live or not args.rebuild  # default to live if index exists

    t0 = time.time()

    tier_boost = {
        "library": LIBRARY_BOOST, "project": PROJECT_BOOST,
        "docs": DOCS_BOOST, "research": RESEARCH_BOOST,
        "session": 1.05, "archive": ARCHIVE_BOOST,
    }

    print(f"Parameters: fetch={FETCH_MULTIPLIER}x, rerank_k={RERANK_TOP_K}, "
          f"boosts={LIBRARY_BOOST}/{PROJECT_BOOST}/{DOCS_BOOST}/{RESEARCH_BOOST}/{ARCHIVE_BOOST}",
          file=sys.stderr)

    from mainframe_mcp.config import hf_offline_if_cached, load_config
    config = load_config()
    hf_offline_if_cached(config)  # hub outages must not hang cached loads

    global CANDIDATE_POOL
    if CANDIDATE_POOL is None:
        CANDIDATE_POOL = config["search"].get("candidate_pool", 20)

    embedder = Embedder(config)
    reranker = Reranker(config)
    t1 = time.time()
    print(f"Models loaded: {t1-t0:.1f}s", file=sys.stderr)

    live_db = MAINFRAME_DIR / ".lancedb"
    if use_live and live_db.exists():
        print("Using LIVE Mainframe index", file=sys.stderr)
        # Mirror production Store construction (incl. recency ranking wiring) so
        # the harness measures the pipeline users actually hit.
        store = Store(db_path=live_db, embedding_dim=embedder.dimension, tier_boost=tier_boost,
                      recency_weight=config["search"].get("recency_weight", 0.0),
                      recency_halflife_days=config["search"].get("recency_halflife_days", 90))
        t2 = t1
    else:
        print("Rebuilding temp index...", file=sys.stderr)
        if TEMP_DB.exists():
            shutil.rmtree(str(TEMP_DB))
        store = Store(db_path=TEMP_DB, embedding_dim=embedder.dimension, tier_boost=tier_boost)
        # Rebuild from the MANIFEST — the authoritative list of everything the
        # live index ingested (multi-repo research/docs/CLAUDE.md). The old flat
        # library/*.md glob could not reproduce the real corpus, which made
        # chunking-parameter experiments (--rebuild's whole purpose) unrunnable
        # at production scale. Falls back to library/ for a fresh setup.
        from mainframe_mcp.manifest import Manifest
        from mainframe_mcp.store import classify_source_type
        files = [Path(p) for p in Manifest(MAINFRAME_DIR).data.get("files", {})]
        files = sorted(p for p in files if p.exists())
        if not files:
            lib = MAINFRAME_DIR / "library"
            files = sorted(lib.glob("*.md")) if lib.exists() else []
        print(f"Rebuild corpus: {len(files)} files (manifest-driven)", file=sys.stderr)
        for md_file in files:
            try:
                text = md_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            chunks = chunk_markdown(text, max_tokens=CHUNK_SIZE, overlap_ratio=OVERLAP_RATIO)
            if not chunks:
                continue
            texts = [c.text for c in chunks]
            embeddings = embedder.embed(texts)
            hashes = [content_hash(t) for t in texts]
            chunk_dicts = [{"text": c.text, "heading": c.heading, "chunk_index": c.chunk_index,
                           "char_start": c.char_start, "char_end": c.char_end, "token_count": c.token_count}
                          for c in chunks]
            doc_id = hashlib.md5(str(md_file).encode()).hexdigest()[:12]
            store.insert_chunks(chunk_dicts, doc_id, str(md_file),
                                classify_source_type(str(md_file), str(MAINFRAME_DIR)),
                                embeddings, hashes)
        store.optimize()  # build the FTS index over the fresh corpus
        t2 = time.time()
        print(f"Index built: {t2-t1:.1f}s", file=sys.stderr)

    # Load test queries — say WHICH set loudly: silently falling back to the
    # 8-entry example set produces meaningless scores in eighths.
    test_queries = json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))
    print(f"Queries: {len(test_queries)} from {TEST_QUERIES_PATH.name}"
          + (" (EXAMPLE SET — scores not comparable to real baselines!)"
             if "example" in TEST_QUERIES_PATH.name else ""),
          file=sys.stderr)

    # Evaluate
    results = evaluate(embedder, reranker, store, test_queries)
    t3 = time.time()
    print(f"Evaluated {results['total_queries']} queries in {t3-t2:.1f}s", file=sys.stderr)

    # Build results record
    results["params"] = {
        "chunk_size": CHUNK_SIZE, "overlap_ratio": OVERLAP_RATIO,
        "min_section_tokens": MIN_SECTION_TOKENS, "fetch_multiplier": FETCH_MULTIPLIER,
        "candidate_pool": CANDIDATE_POOL,
        "rerank_top_k": RERANK_TOP_K, "library_boost": LIBRARY_BOOST,
        "project_boost": PROJECT_BOOST, "docs_boost": DOCS_BOOST,
        "research_boost": RESEARCH_BOOST, "archive_boost": ARCHIVE_BOOST,
    }
    results["models"] = {"embedder": embedder.model_name, "reranker": reranker.model_name}
    results["timing"] = {"model_load": t1 - t0, "ingest": t2 - t1, "eval": t3 - t2, "total": t3 - t0}
    results["mode"] = "live" if (use_live and live_db.exists()) else "rebuild"
    results["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Save to results history
    results_dir = EVAL_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    result_file = results_dir / f"{ts}.json"
    result_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved to {result_file.name}", file=sys.stderr)

    print(json.dumps(results, indent=2))
    print(f"\nval_score: {results['score']:.4f}", file=sys.stderr)

    # Cleanup temp DB if used
    if not (use_live and live_db.exists()):
        shutil.rmtree(str(TEMP_DB), ignore_errors=True)

    # Regression gate for CI/nightly use.
    if args.min_score is not None and results["score"] < args.min_score:
        print(f"REGRESSION: score {results['score']:.4f} < min {args.min_score:.4f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
