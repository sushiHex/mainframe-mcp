"""RAG evaluation harness for autoresearch parameter optimization.

Modes (resolved by `choose_mode`):
  (default) / --live   Score the live index. Routed through the RUNNING daemon's
             /search when mainframed is up — it is the only process that loads a
             model — and in-process otherwise. `--in-process` forces the latter.
  --daemon   Always score via the daemon (errors if it is not running).
  --rebuild  Rebuild index from scratch (slow, tests chunking params too) — the
             temp index is kept at eval/.eval-lancedb afterward (for --db
             re-scoring) and replaced by the next --rebuild.
  --db PATH  Score an existing v2 index without rebuilding it.
`--rebuild`/`--db` load models here, so they REFUSE while the daemon is running.

Metrics: hit@k, MRR, text_match@1. Composite score for autoresearch. The daemon
path scores the same top-3 rank window as the in-process one (`hit_at_5` is
None there), so the two are comparable and the --min-score gate means the same
thing in both.
"""

import json
import sys
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mainframe.config import hf_offline_if_cached, load_config
from mainframe.core.embedder import Embedder
from mainframe.core.indexer import LaneFile, prepare_document, DocFailure, UNCHANGED
from mainframe.core.reranker import Reranker
from mainframe.core.store import INDEX_DIRNAME, Store
from mainframe.memory.lanes import Lanes

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
                            # hit.
# ------------------------------------------------------------------
CANDIDATE_DUMP = None       # None, or a dict populated by --dump-candidates
                            # (query -> [[doc_path, chunk_index], ...] over the
                            # raw candidate pool) — the determinism gate.

EVAL_DIR = Path(__file__).parent
# Your own corpus-specific query set lives in test_queries.json (gitignored —
# eval fixtures inevitably describe YOUR private corpus). The tracked
# test_queries.example.json shows the format and targets this repo's own docs.
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"
if not TEST_QUERIES_PATH.exists():
    TEST_QUERIES_PATH = EVAL_DIR / "test_queries.example.json"
TEMP_DB = EVAL_DIR / ".eval-lancedb"


def corpus_from_manifest(config: dict, manifest_path) -> list:
    """The v1 manifest's non-session files that still exist on disk, as knowledge
    LaneFiles (project = first path component under repos_dir, else '_manifest').
    Used by the parity gates so the rebuild measures the same documents as the
    reference run instead of whatever repos_dir holds today."""
    import json
    from mainframe.core.paths import canonical
    repos = canonical(config["paths"]["repos_dir"]).rstrip("/") + "/"
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out = []
    for key in data.get("files", {}):
        kk = key.replace("\\", "/")
        if "/research/sessions/" in kk or "/library/sessions/" in kk:
            continue
        c = canonical(key)
        if not Path(c).exists():
            continue
        project = c[len(repos):].split("/", 1)[0] if c.startswith(repos) else "_manifest"
        out.append(LaneFile(c, "knowledge", project))
    return sorted(out, key=lambda f: f.path)


def rebuild_index(store, embedder, files, chunk_cfg, batch_size: int = 50) -> int:
    """Index `files` (LaneFile list) in sorted canonical order, `batch_size`
    documents per merge — deterministic by construction."""
    files = sorted(files, key=lambda f: f.path)
    written = 0
    for i in range(0, len(files), batch_size):
        docs = []
        for lf in files[i:i + batch_size]:
            d = prepare_document(lf, chunk_cfg, embedder)
            if d is UNCHANGED or isinstance(d, DocFailure):
                if isinstance(d, DocFailure):
                    print(f"  skip {d.doc_path}: {d.error}", file=sys.stderr)
                continue
            docs.append(d)
        if docs:
            written += store.upsert_batch(docs).upserted_rows
    store.optimize()
    return written


def open_existing_index(path, embedding_dim: int, config: dict):
    """Open an EXISTING v2 index for evaluation. Refuses (SystemExit) when the
    directory has no `chunks` table, the table is empty, or it is a v1 schema —
    Store's create-if-missing would otherwise score a silent 0."""
    import lancedb
    p = Path(path)
    if p.exists():
        resp = lancedb.connect(str(p)).list_tables()
        tables = getattr(resp, "tables", resp)
    else:
        tables = []
    if "chunks" not in tables:
        sys.exit(f"no v2 index at {p} (no `chunks` table) — run --rebuild or point --db at a built index")
    store = Store(db_path=p, embedding_dim=embedding_dim, fingerprint=Store.fingerprint_from_config(config),
                  **Store.search_kwargs_from_config(config))
    if "lane" not in [f.name for f in store.table.schema]:
        sys.exit(f"index at {p} is v1 — use the separate v2 index or run --rebuild")
    if store.table.count_rows() == 0:
        sys.exit(f"index at {p} is empty — run --rebuild")
    return store


def matches_expected(doc_path: str, expected_file: str) -> bool:
    """Case-insensitive, separator-insensitive substring match — doc paths are
    canonical (lower-cased on Windows, forward slashes) while the query set was
    written with native spellings."""
    return expected_file.replace("\\", "/").lower() in doc_path.replace("\\", "/").lower()


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
        raw_results = store.search(q_emb, top_k=pool, query_text=query, include_captures=False)

        if CANDIDATE_DUMP is not None:
            CANDIDATE_DUMP[query] = [[r["doc_path"], int(r["chunk_index"])] for r in raw_results]

        if not raw_results:
            reciprocal_ranks.append(0.0)
            continue

        texts = [r["text"] for r in raw_results]
        headings = [r.get("heading", "") for r in raw_results]
        reranked = reranker.rerank(query, texts, top_k=RERANK_TOP_K, headings=headings)

        found_rank = None
        for rank, (orig_idx, score) in enumerate(reranked, 1):
            r = raw_results[orig_idx]
            doc_path = r.get("doc_path", "")
            if matches_expected(doc_path, expected_file):
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


def evaluate_via_daemon(client, test_queries: list, limit: int = RERANK_TOP_K) -> dict:
    """Score the RUNNING daemon's search: no
    model load here, the daemon answers. Candidate dumps are not available.

    Asks for exactly `limit` (= RERANK_TOP_K = 3) results, the SAME rank window
    in-process evaluate() scores. Asking for 5 let ranks 4-5 contribute
    0.25/0.20 to MRR that the in-process harness can never earn, so a --daemon
    score came out systematically higher than the in-process baselines the
    --min-score gate is calibrated against. `hit_at_5` is reported as None for
    the same reason: ranks 4-5 do not exist in this window, and reporting 0
    (or hit_at_3's value) would both be lies.

    Uses response_format="full" (not "detailed") — SearchService's "detailed"
    caps `text` at TEXT_CHARS=500, while in-process evaluate() matches
    expected_text_contains against the FULL chunk text. Scoring "detailed"
    here would under-count text_match_at_1 whenever the expected phrase sits
    past char 500, making --daemon scores incomparable to evaluate()'s."""
    n = len(test_queries)
    hits1 = hits3 = text = 0
    rr = []
    for qi, tq in enumerate(test_queries, 1):
        try:
            out = client.post("/search", json={"query": tq["query"], "limit": limit,
                                               "include_sessions": False, "response_format": "full"})
        except Exception as e:
            sys.exit(f"daemon became unreachable at query {qi}/{n}: {e}")
        rank = None
        for i, r in enumerate(out.get("results", []), 1):
            if matches_expected(r["file"], tq["expected_file"]):
                rank = i
                if i == 1 and tq.get("expected_text_contains", "").lower() in r.get("text", "").lower():
                    text += 1
                break
        rr.append(1.0 / rank if rank else 0.0)
        hits1 += rank == 1
        hits3 += bool(rank and rank <= 3)
    mrr = sum(rr) / n if n else 0
    return {"total_queries": n, "hit_at_1": hits1 / n if n else 0, "hit_at_3": hits3 / n if n else 0,
            "hit_at_5": None, "mrr": mrr, "text_match_at_1": text / n if n else 0,
            "score": (mrr * 0.4 + hits1 / n * 0.3 + text / n * 0.3) if n else 0}


def choose_mode(args, daemon_alive: bool) -> str:
    """Pure mode resolution. "daemon" scores through the running
    daemon's /search with no model load here; "in-process" builds an Embedder
    and a Reranker in THIS process; "refuse" means the requested mode needs the
    GPU that mainframed is holding.

    `--live` (and the no-flag default) uses the running daemon so that only
    one process loads models. The
    default used to double-load ~15 GB against a resident daemon. `--rebuild`
    and `--db` must build/read their own index in-process, so they refuse while
    the daemon runs.
    `--in-process` is the escape hatch; `--dump-candidates` needs the raw
    candidate pool, which only the in-process path has."""
    if getattr(args, "daemon", False):
        return "daemon"                       # explicit: main() errors if it is not alive
    if getattr(args, "rebuild", False) or getattr(args, "db", None):
        return "refuse" if daemon_alive else "in-process"
    if getattr(args, "in_process", False) or getattr(args, "dump_candidates", None):
        return "in-process"
    return "daemon" if daemon_alive else "in-process"


def main():
    """Run evaluation. Use --live for existing index, --rebuild for fresh index,
    --daemon to score the running daemon's /search without loading any models."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use existing Mainframe index (fast)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild temp index from library files")
    parser.add_argument("--daemon", action="store_true",
                        help="Score via the running daemon's /search (no model load)")
    parser.add_argument("--in-process", action="store_true",
                        help="Force the in-process path for --live/default even when the daemon is up")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Regression gate: exit 1 if the composite score falls below this")
    parser.add_argument("--dump-candidates", type=str, default=None,
                        help="Write per-query candidate lists (determinism gate)")
    parser.add_argument("--corpus-manifest", type=str, default=None,
                        help="Rebuild from exactly the files listed in this v1 manifest (parity gates)")
    parser.add_argument("--db", type=str, default=None,
                        help="Evaluate an existing v2 index at PATH without rebuilding or deleting it")
    args = parser.parse_args()

    if args.db and args.rebuild:
        parser.error("--db and --rebuild are mutually exclusive")
    if args.daemon and (args.rebuild or args.live or args.db or args.dump_candidates or args.in_process):
        parser.error("--daemon is mutually exclusive with --rebuild/--live/--db/--dump-candidates/--in-process")

    global CANDIDATE_DUMP
    CANDIDATE_DUMP = {} if args.dump_candidates else None

    use_live = args.live or not args.rebuild  # default to live if index exists

    t0 = time.time()

    config = load_config()
    hf_offline_if_cached(config)  # hub outages must not hang cached loads

    # Load test queries — say WHICH set loudly: silently falling back to the
    # 8-entry example set produces meaningless scores in eighths.
    test_queries = json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))
    print(f"Queries: {len(test_queries)} from {TEST_QUERIES_PATH.name}"
          + (" (EXAMPLE SET — scores not comparable to real baselines!)"
             if "example" in TEST_QUERIES_PATH.name else ""),
          file=sys.stderr)

    # Mode resolution happens BEFORE any model is constructed:
    # mainframed is the only process that loads a model, so the default routes
    # through it whenever it is running.
    from mainframe.adapters.cli import DaemonClient
    client = DaemonClient(config)
    daemon_alive = client.alive()
    mode = choose_mode(args, daemon_alive)
    if mode == "refuse":
        sys.exit("mainframed is running and holds the GPU - stop it first (`mainframe down`) for "
                 "--rebuild/--db, or score the live index through it with no flags")

    if mode == "daemon":
        # Score the RUNNING daemon's search directly — no embedder/reranker/
        # store construction, so this path never touches the GPU or torch.
        if not daemon_alive:                        # only reachable via an explicit --daemon
            sys.exit("daemon not running - mainframe up")
        print(f"Scoring via daemon at {client.base_url}", file=sys.stderr)
        t1 = t2 = time.time()
        results = evaluate_via_daemon(client, test_queries)
        t3 = time.time()
        print(f"Evaluated {results['total_queries']} queries in {t3-t2:.1f}s", file=sys.stderr)
        results["models"] = {"embedder": "daemon", "reranker": "daemon"}
        results["timing"] = {"model_load": t1 - t0, "ingest": t2 - t1, "eval": t3 - t2, "total": t3 - t0}
        results["mode"] = "daemon"
    else:
        print(f"Parameters: fetch={FETCH_MULTIPLIER}x, rerank_k={RERANK_TOP_K}", file=sys.stderr)

        global CANDIDATE_POOL
        if CANDIDATE_POOL is None:
            CANDIDATE_POOL = config["search"].get("candidate_pool", 20)

        embedder = Embedder(config)
        reranker = Reranker(config)
        t1 = time.time()
        print(f"Models loaded: {t1-t0:.1f}s", file=sys.stderr)

        live_db = Path(config["paths"]["mainframe_dir"]) / INDEX_DIRNAME
        # Mode resolution order: --db > --rebuild > --live/default.
        if args.db:
            print(f"Using existing index at {args.db}", file=sys.stderr)
            store = open_existing_index(args.db, embedder.dimension, config)
            t2 = t1
        elif use_live and live_db.exists():
            print("Using LIVE Mainframe index", file=sys.stderr)
            # Mirror production Store construction (incl. recency ranking wiring) so
            # the harness measures the pipeline users actually hit.
            store = open_existing_index(live_db, embedder.dimension, config)
            t2 = t1
        else:
            print("Rebuilding temp index...", file=sys.stderr)
            if TEMP_DB.exists():
                shutil.rmtree(str(TEMP_DB))
            store = Store(db_path=TEMP_DB, embedding_dim=embedder.dimension,
                          fingerprint=Store.fingerprint_from_config(config),
                          **Store.search_kwargs_from_config(config))
            if args.corpus_manifest:
                files = corpus_from_manifest(config, args.corpus_manifest)
                source_label = "manifest-scoped"
            else:
                files = Lanes(config).knowledge_files()
                source_label = "knowledge lane"
            print(f"Rebuild corpus: {len(files)} files ({source_label})", file=sys.stderr)
            # Use the module-level knobs (autoresearch's sweep target), not
            # config["chunker"] — results["params"] below reports these same
            # constants, so the reported params and the actual chunking must
            # come from the same source or a sweep silently measures nothing.
            chunk_cfg = {"chunk_size": CHUNK_SIZE, "overlap_ratio": OVERLAP_RATIO,
                        "min_section_tokens": MIN_SECTION_TOKENS}
            rows = rebuild_index(store, embedder, files, chunk_cfg)
            t2 = time.time()
            print(f"Index built: {rows} rows in {t2-t1:.1f}s", file=sys.stderr)

        # Evaluate
        results = evaluate(embedder, reranker, store, test_queries)

        # Write the candidate dump the instant it's available — a bad
        # --dump-candidates path must not lose a ~25-minute rebuild's results.
        if CANDIDATE_DUMP is not None:
            dump_path = Path(args.dump_candidates)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text(json.dumps(CANDIDATE_DUMP, indent=1), encoding="utf-8")

        t3 = time.time()
        print(f"Evaluated {results['total_queries']} queries in {t3-t2:.1f}s", file=sys.stderr)

        # Build results record
        results["params"] = {
            "chunk_size": CHUNK_SIZE, "overlap_ratio": OVERLAP_RATIO,
            "min_section_tokens": MIN_SECTION_TOKENS, "fetch_multiplier": FETCH_MULTIPLIER,
            "candidate_pool": CANDIDATE_POOL, "rerank_top_k": RERANK_TOP_K,
        }
        results["models"] = {"embedder": embedder.model_name, "reranker": reranker.model_name}
        results["timing"] = {"model_load": t1 - t0, "ingest": t2 - t1, "eval": t3 - t2, "total": t3 - t0}
        results["mode"] = "db" if args.db else ("live" if (use_live and live_db.exists()) else "rebuild")

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

    # Regression gate for CI/nightly use.
    if args.min_score is not None and results["score"] < args.min_score:
        print(f"REGRESSION: score {results['score']:.4f} < min {args.min_score:.4f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
