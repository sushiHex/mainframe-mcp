"""Diff two --details JSON files (written by eval/evaluate.py's
write_details_file/build_detail_record) to say WHICH queries moved between
two eval runs, and whose fault it looks like: the embedder's candidate pool
(the expected passage fell out before the reranker ever saw it), or the
reranker's ordering of a pool that already had the right passage in it.

Usage:
    python eval/details_diff.py BASE.json OTHER.json [--markdown]

Pure functions only — no model, LanceDB, or mainframe imports — so this runs
against archived --details files with nothing GPU-resident.
"""
import argparse
import json
from pathlib import Path


def _in_pool(record: dict):
    """True/False, or None when the record has no pool info at all (the
    --daemon path always writes pool: null)."""
    pool = record.get("pool")
    return pool["expected_in_pool"] if pool else None


def _pool_rank(record: dict):
    pool = record.get("pool")
    return pool["expected_best_pool_rank"] if pool else None


def _rank_key(found_rank):
    """Sort key where a SMALLER key is a BETTER (higher) rank; not-found
    (None) sorts worse than any numeric rank — "null counts as worse than
    any rank"."""
    return (found_rank is None, found_rank or 0)


def classify(base: dict, other: dict) -> str:
    """One query's before/after judgement records -> a fault class.

    EMBEDDER   the expected passage fell OUT of the candidate pool between
               the two runs — the reranker never had a chance at it; the
               regression is upstream of it.
    RERANKER   still in the pool under both runs, but the reranker now ranks
               it worse (a non-find counts as worse than any numeric rank).
    RECOVERED  still in the pool under both runs, and the reranker now ranks
               it better.
    POOL_RANK  in the pool under both runs and the reranked rank is
               unchanged, but WHERE it sat in the pre-rerank pool moved —
               informational: the hybrid ranking shuffled without changing
               the outcome.
    OTHER      anything else that still differs (e.g. only text_match_at_1
               changed, or pool membership is unknown for one side).
    """
    b_in, o_in = _in_pool(base), _in_pool(other)
    if b_in is True and o_in is False:
        return "EMBEDDER"
    if b_in is True and o_in is True:
        b_key, o_key = _rank_key(base.get("found_rank")), _rank_key(other.get("found_rank"))
        if o_key > b_key:
            return "RERANKER"
        if o_key < b_key:
            return "RECOVERED"
        if _pool_rank(base) != _pool_rank(other):
            return "POOL_RANK"
    return "OTHER"


def _changed(base: dict, other: dict) -> bool:
    return (base.get("found_rank") != other.get("found_rank")
            or base.get("text_match_at_1") != other.get("text_match_at_1"))


def diff_records(base_queries: list, other_queries: list) -> list:
    """One row per query whose found_rank or text_match_at_1 differs, in
    query-set order. Queries are matched by `index` — both files must come
    from the same query set (the on-disk order the harness ran them in)."""
    other_by_index = {r["index"]: r for r in other_queries}
    rows = []
    for base in base_queries:
        other = other_by_index.get(base["index"])
        if other is None or not _changed(base, other):
            continue
        rows.append({
            "index": base["index"],
            "expected_file": base["expected_file"],
            "base_rank": base.get("found_rank"),
            "other_rank": other.get("found_rank"),
            "base_in_pool": _in_pool(base),
            "other_in_pool": _in_pool(other),
            "class": classify(base, other),
        })
    return rows


def summarize_classes(rows: list) -> dict:
    """class -> count, over the rows diff_records produced."""
    counts = {}
    for row in rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    return counts


def _aggregates(queries: list) -> dict:
    """hit@1, hit@3, MRR, text@1 recomputed straight from the records — the
    same formulas eval/evaluate.py's evaluate() uses."""
    n = len(queries)
    if not n:
        return {"hit_at_1": 0.0, "hit_at_3": 0.0, "mrr": 0.0, "text_match_at_1": 0.0}
    hit1 = sum(1 for q in queries if q.get("found_rank") == 1)
    hit3 = sum(1 for q in queries if q.get("found_rank") is not None and q["found_rank"] <= 3)
    text1 = sum(1 for q in queries if q.get("text_match_at_1"))
    mrr = sum(1.0 / q["found_rank"] if q.get("found_rank") else 0.0 for q in queries) / n
    return {"hit_at_1": hit1 / n, "hit_at_3": hit3 / n, "mrr": mrr, "text_match_at_1": text1 / n}


def aggregate_deltas(base_queries: list, other_queries: list) -> dict:
    """{"metric": (base, other, delta), ...} recomputed from the records
    themselves, so the diff is self-checking against whatever the two
    results files reported rather than trusting them blindly."""
    b, o = _aggregates(base_queries), _aggregates(other_queries)
    return {k: (b[k], o[k], o[k] - b[k]) for k in b}


def render_text(rows: list, counts: dict, deltas: dict) -> str:
    lines = [f"[{r['index']}] {r['expected_file']}: rank {r['base_rank']} -> {r['other_rank']}, "
             f"pool {r['base_in_pool']} -> {r['other_in_pool']} :: {r['class']}"
             for r in rows]
    lines.append("")
    lines.append("Summary: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no changes"))
    lines.append("")
    for metric, (b, o, d) in deltas.items():
        lines.append(f"{metric}: {b:.4f} -> {o:.4f} ({d:+.4f})")
    return "\n".join(lines)


def render_markdown(rows: list, counts: dict, deltas: dict) -> str:
    lines = ["| index | expected_file | base rank | other rank | base pool | other pool | class |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['index']} | {r['expected_file']} | {r['base_rank']} | {r['other_rank']} | "
              f"{r['base_in_pool']} | {r['other_in_pool']} | {r['class']} |"
              for r in rows]
    lines += ["", "| class | count |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    lines += ["", "| metric | base | other | delta |", "| --- | --- | --- | --- |"]
    lines += [f"| {metric} | {b:.4f} | {o:.4f} | {d:+.4f} |" for metric, (b, o, d) in deltas.items()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", help="Baseline --details JSON file")
    parser.add_argument("other", help="Other-run --details JSON file")
    parser.add_argument("--markdown", action="store_true", help="Render as a markdown table")
    args = parser.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    other = json.loads(Path(args.other).read_text(encoding="utf-8"))
    rows = diff_records(base["queries"], other["queries"])
    counts = summarize_classes(rows)
    deltas = aggregate_deltas(base["queries"], other["queries"])
    render = render_markdown if args.markdown else render_text
    print(render(rows, counts, deltas))


if __name__ == "__main__":
    main()
