"""Diff two --details JSON files (written by eval/evaluate.py's
write_details_file/build_detail_record) to say WHICH queries moved between
two eval runs, and whose fault it looks like: the embedder's candidate pool
(the expected passage fell out before the reranker ever saw it), or the
reranker's ordering of a pool that already had the right passage in it.

That attribution is only honest when both runs scored the SAME pool — same
index, same embedder config — because a different embedder changes which
chunks the reranker even sees, the FTS fallback distances, and batch
composition. `compare()` reads `cell.models.embedder` from both files and
switches to factual-only labels (no fault attributed) when they differ.

Usage:
    python eval/details_diff.py BASE.json OTHER.json [--markdown]

Pure functions only — no model, LanceDB, or mainframe imports — so this runs
against archived --details files with nothing GPU-resident.
"""
import argparse
import json
from pathlib import Path


def embedder_of(details: dict):
    return details.get("cell", {}).get("models", {}).get("embedder")


def mode_of(details: dict):
    return details.get("cell", {}).get("mode")


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


def _passage_lost(base: dict, other: dict) -> bool:
    """True when the expected FILE stayed in the pool under both runs but
    the answer PASSAGE (a chunk of that file containing expected_text) left
    it — Gap 2's `expected_text_in_pool` makes this distinguishable from a
    plain pool-position shuffle. Silently False when either record predates
    Gap 2 (no `expected_text_in_pool` key) or has no pool at all (--daemon)
    — an absent signal must not read as a confirmed answer."""
    bp, op = base.get("pool"), other.get("pool")
    if not bp or not op or "expected_text_in_pool" not in bp or "expected_text_in_pool" not in op:
        return False
    return (bool(bp.get("expected_in_pool")) and bool(op.get("expected_in_pool"))
            and bool(bp.get("expected_text_in_pool")) and not op.get("expected_text_in_pool"))


def classify(base: dict, other: dict) -> str:
    """One query's before/after judgement records -> a fault class. Only
    meaningful when both runs scored the SAME pool (see `classify_cross`
    otherwise — `compare()` picks between them).

    EMBEDDER      the expected passage fell OUT of the candidate pool
                  between the two runs — the reranker never had a chance at
                  it; the regression is upstream of it.
    RERANKER      still in the pool under both runs, but the reranker now
                  ranks it worse (a non-find counts as worse than any
                  numeric rank).
    RECOVERED     still in the pool under both runs, and the reranker now
                  ranks it better.
    PASSAGE_LOST  the file stayed in the pool and the reranked rank did not
                  get worse, but the specific answer passage left the pool
                  (see `_passage_lost`) — the file is still there, its
                  answer chunk is not.
    POOL_RANK     in the pool under both runs and the reranked rank is
                  unchanged, but WHERE it sat in the pre-rerank pool moved —
                  informational: the hybrid ranking shuffled without
                  changing the outcome.
    OTHER         anything else that still differs (e.g. only
                  text_match_at_1 changed for an unrelated reason, or pool
                  membership is unknown for one side).
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
        if _passage_lost(base, other):
            return "PASSAGE_LOST"
        if _pool_rank(base) != _pool_rank(other):
            return "POOL_RANK"
    return "OTHER"


def classify_cross(base: dict, other: dict) -> str:
    """Factual (non-attributing) labels for two runs that scored DIFFERENT
    pools (different embedder) — a rank change cannot be pinned on the
    reranker when the candidates it saw were not even the same set.

    POOL_LOST    the expected file fell out of the candidate pool.
    POOL_GAINED  the expected file newly entered the candidate pool.
    RANK_WORSE   in the pool under both runs; the final rank got worse.
    RANK_BETTER  in the pool under both runs; the final rank got better.
    PASSAGE_LOST see `classify`'s PASSAGE_LOST — meaningful regardless of
                 whether the embedder changed.
    TEXT_ONLY    found_rank is unchanged but text_match_at_1 differs, with
                 no pool-level explanation available.
    OTHER        anything else that still differs.
    """
    b_in, o_in = _in_pool(base), _in_pool(other)
    if b_in is True and o_in is False:
        return "POOL_LOST"
    if b_in is False and o_in is True:
        return "POOL_GAINED"
    if b_in is True and o_in is True:
        b_key, o_key = _rank_key(base.get("found_rank")), _rank_key(other.get("found_rank"))
        if o_key > b_key:
            return "RANK_WORSE"
        if o_key < b_key:
            return "RANK_BETTER"
        if _passage_lost(base, other):
            return "PASSAGE_LOST"
        if base.get("text_match_at_1") != other.get("text_match_at_1"):
            return "TEXT_ONLY"
    return "OTHER"


def _changed(base: dict, other: dict) -> bool:
    return (base.get("found_rank") != other.get("found_rank")
            or base.get("text_match_at_1") != other.get("text_match_at_1"))


def diff_records(base_queries: list, other_queries: list, classify_fn=classify) -> list:
    """One row per query whose found_rank or text_match_at_1 differs, in
    query-set order. Queries are matched by `index` — both files must come
    from the same query set (the on-disk order the harness ran them in).
    `classify_fn` is `classify` (same-embedder, attributing) or
    `classify_cross` (different-embedder, factual-only) — `compare()` picks
    which one applies to a given pair of files."""
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
            "class": classify_fn(base, other),
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


def compare(base_details: dict, other_details: dict) -> dict:
    """Top-level comparison between two --details files: picks the
    attributing (`classify`) or factual-only (`classify_cross`) label set
    based on whether `cell.models.embedder` matches, and flags the
    same-embedder branch's EMBEDDER-cannot-occur consistency assumption
    with a warning rather than silently trusting it.

    A missing/empty embedder string on either side is treated as "not the
    same" (the conservative default: an unrecorded pool cannot be assumed
    identical) — this also covers --details files predating Gap 4's
    provenance fields."""
    base_queries, other_queries = base_details["queries"], other_details["queries"]
    be, oe = embedder_of(base_details), embedder_of(other_details)
    same_embedder = bool(be) and be == oe
    classify_fn = classify if same_embedder else classify_cross
    rows = diff_records(base_queries, other_queries, classify_fn)

    warnings = []
    if same_embedder:
        if any(r["class"] == "EMBEDDER" for r in rows):
            warnings.append(
                f"WARNING: pool membership changed (a query's expected file left the pool) even "
                f"though both runs report the same embedder ({be!r}) — the 'same pool' assumption "
                f"this label set relies on may not hold (different corpus, index, or config?)."
            )
    else:
        warnings.append(
            f"NOTE: base used embedder {be!r} (mode={mode_of(base_details)!r}), other used "
            f"{oe!r} (mode={mode_of(other_details)!r}) — the two runs did not necessarily score "
            f"the same candidate pool, so rank changes below are NOT attributable to the "
            f"reranker. Labels are factual only."
        )

    return {
        "rows": rows,
        "counts": summarize_classes(rows),
        "deltas": aggregate_deltas(base_queries, other_queries),
        "warnings": warnings,
        "same_embedder": same_embedder,
    }


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
    result = compare(base, other)
    for warning in result["warnings"]:
        print(warning)
        print()
    render = render_markdown if args.markdown else render_text
    print(render(result["rows"], result["counts"], result["deltas"]))


if __name__ == "__main__":
    main()
