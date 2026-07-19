"""View and compare eval result history.

Usage:
    python eval/history.py                  # show all runs, newest first
    python eval/history.py --best           # show top 10 by val_score
    python eval/history.py --model qwen     # filter by model name (substring)
    python eval/history.py --last 5         # show last 5 runs
    python eval/history.py --compare        # side-by-side: best per model pair
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_results() -> list[dict]:
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_file"] = f.name
            results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def fmt_row(r: dict) -> str:
    models = r.get("models", {})
    emb = models.get("embedder", "?").split("/")[-1]
    rnk = models.get("reranker", "?").split("/")[-1]
    p = r.get("params", {})
    return (
        f"{r.get('timestamp', '?')[:19]:>19}  "
        f"{r.get('score', 0):6.4f}  "
        f"{r.get('mrr', 0):5.3f}  "
        f"{r.get('hit_at_1', 0):5.3f}  "
        f"{r.get('text_match_at_1', 0):5.3f}  "
        f"chunk={p.get('chunk_size', '?'):>4} "
        f"ovlp={p.get('overlap_ratio', '?'):<4} "
        f"fetch={p.get('fetch_multiplier', '?')} "
        f"rnk_k={p.get('rerank_top_k', '?')} "
        f"boost={p.get('library_boost', '?')}/{p.get('oracle_boost', '?')}/{p.get('archive_boost', '?')}  "
        f"{emb}/{rnk}"
    )


HEADER = (
    f"{'timestamp':>19}  {'score':>6}  {'mrr':>5}  {'h@1':>5}  {'tm@1':>5}  "
    f"{'params':<58}  models"
)
SEP = "-" * 160


def show_table(results: list[dict]):
    print(HEADER)
    print(SEP)
    for r in results:
        print(fmt_row(r))
    print(SEP)
    print(f"{len(results)} run(s)")


def show_compare(results: list[dict]):
    """Best score per unique model pair."""
    by_models: dict[str, dict] = {}
    for r in results:
        models = r.get("models", {})
        key = f"{models.get('embedder', '?')} + {models.get('reranker', '?')}"
        if key not in by_models or r.get("score", 0) > by_models[key].get("score", 0):
            by_models[key] = r

    print("Best score per model combination:\n")
    print(HEADER)
    print(SEP)
    for key in sorted(by_models, key=lambda k: by_models[k].get("score", 0), reverse=True):
        print(fmt_row(by_models[key]))
    print(SEP)


def main():
    parser = argparse.ArgumentParser(description="View eval result history")
    parser.add_argument("--best", action="store_true", help="Top 10 by score")
    parser.add_argument("--last", type=int, help="Show last N runs")
    parser.add_argument("--model", type=str, help="Filter by model name substring")
    parser.add_argument("--compare", action="store_true", help="Best per model pair")
    args = parser.parse_args()

    results = load_results()
    if not results:
        print("No results found in eval/results/")
        return

    if args.model:
        q = args.model.lower()
        results = [
            r for r in results
            if q in json.dumps(r.get("models", {})).lower()
        ]

    if args.compare:
        show_compare(results)
        return

    if args.best:
        results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)[:10]
    elif args.last:
        results = results[-args.last:]
    else:
        results = list(reversed(results))  # newest first

    show_table(results)


if __name__ == "__main__":
    main()
