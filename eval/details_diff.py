"""Diff two --details JSON files (written by eval/evaluate.py's
write_details_file/build_detail_record) to say WHICH queries moved between
two eval runs, and whose fault it looks like: the embedder's candidate pool
(the expected passage fell out before the reranker ever saw it), or the
reranker's ordering of a pool that already had the right passage in it.

That attribution is only honest when both runs scored the SAME pool. Round
2 approximated "same pool" by comparing the embedder NAME — a proxy that
can't tell two differently-named configs producing the same retrieval from
one same-named config that quietly diverged. Records now carry candidate
IDENTITIES (`full_order`: every pool entry as doc_path + chunk_index +
content_sha256 + pool_rank) and FULL ranks, not just a top-3 window, so
`compare()` gates attribution on the real thing: per query, `pool_identity()`
— the ORDERED tuple of (doc_path, chunk_index, content_sha256) sorted by
`pool_rank` (order matters: Reranker.rerank() sorts stably, so equal scores
inherit the pre-rerank pool order) — must be IDENTICAL between the two
runs, AND `cell.params` (candidate_pool/rerank_top_k/chunk_size/
overlap_ratio) must match file-wide. The embedder name is kept only for the
header note — informational, not the gate.

Three concerns, each owned by exactly one function (Round 5 — the same
identity/movement/query-equality logic used to live at more than one call
site and drifted): `pool_identity()` decides whether two runs saw the same
candidates in the same order; `rank_movement()` decides how far a rank
moved at each granularity (window/full/text-full), read by BOTH the
attributing and factual classifiers; `same_query()` decides whether two
query records describe the same question.

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


def _rank_key(rank):
    """Sort key where a SMALLER key is a BETTER (higher) rank; None (not
    found / not in the pool) sorts worse than any numeric rank — "null
    counts as worse than any rank". Works for found_rank AND
    expected_text_full_rank alike."""
    return (rank is None, rank or 0)


def _delta(base_rank, other_rank) -> int:
    """+1 when `other` ranks worse than `base`, -1 when better, 0 tied —
    using `_rank_key` so None is worse than any numeric rank at every
    granularity this is applied to."""
    b_key, o_key = _rank_key(base_rank), _rank_key(other_rank)
    if o_key > b_key:
        return 1
    if o_key < b_key:
        return -1
    return 0


def rank_movement(base: dict, other: dict) -> tuple:
    """(window_delta, full_delta, text_full_delta): the ONE place that
    computes how far a query's rank moved, at each of the three
    granularities a top-3 window, the reranker's full ordering, and the
    answer passage's position within that full ordering can independently
    move at. `classify()` (attributing) and `classify_cross` (factual) both
    read this instead of each recomputing rank comparisons — the label sets
    differ, but "did it get better or worse, and at which granularity" does
    not, and duplicating that answer let it drift (Round 4 gave `classify`
    a full-rank fallback on a found_rank tie but not `classify_cross`)."""
    return (
        _delta(base.get("found_rank"), other.get("found_rank")),
        _delta(base.get("expected_full_rank"), other.get("expected_full_rank")),
        _delta(base.get("expected_text_full_rank"), other.get("expected_text_full_rank")),
    )


def pool_identity(record: dict):
    """The ORDERED tuple of (doc_path, chunk_index, content_sha256) for
    every `full_order` entry, sorted by `pool_rank` — the position the
    reranker's INPUT (the pre-rerank pool) held it at, not its reranked
    output position. Reranker.rerank() sorts stably, so equal scores
    inherit that pre-rerank order — two runs with the same candidate SET in
    a DIFFERENT pool order are not the same pool, so order is part of the
    identity, not just set membership (Round 5).

    None when `full_order` is absent (records older than Gap 1, or
    --daemon mode, which has no full pool to report) OR when any entry is
    missing doc_path/chunk_index/content_sha256/pool_rank (records older
    than Round 4/5) — an absent signal must never read as "empty/missing
    and therefore equal"."""
    full_order = record.get("full_order")
    if full_order is None:
        return None
    entries = []
    for e in full_order:
        doc_path, chunk_index = e.get("doc_path"), e.get("chunk_index")
        content, rank = e.get("content_sha256"), e.get("pool_rank")
        if doc_path is None or chunk_index is None or not content or rank is None:
            return None
        entries.append((rank, doc_path, chunk_index, content))
    entries.sort(key=lambda t: t[0])
    return tuple((doc_path, chunk_index, content) for _, doc_path, chunk_index, content in entries)


def _pool_status(base: dict, other: dict) -> str:
    """'identical' | 'order_differs' | 'differs' | 'unrecorded': whether
    `pool_identity()` proves the two runs scored the same pool, and if not,
    WHY — the same candidate SET in a different order is a narrower,
    differently-explained problem than a genuinely different set."""
    b_id, o_id = pool_identity(base), pool_identity(other)
    if b_id is None or o_id is None:
        return "unrecorded"
    if b_id == o_id:
        return "identical"
    if set(b_id) == set(o_id):
        return "order_differs"
    return "differs"


def _params_match(base_details: dict, other_details: dict) -> bool:
    """Whether the two FILES (not one query) were scored under the same
    chunking/pooling configuration — part of the same-pool proof alongside
    per-query candidate identity (`pool_identity`/`_pool_status`)."""
    keys = ("candidate_pool", "rerank_top_k", "chunk_size", "overlap_ratio")
    bp = base_details.get("cell", {}).get("params") or {}
    op = other_details.get("cell", {}).get("params") or {}
    return all(bp.get(k) == op.get(k) for k in keys)


def _passage_lost(base: dict, other: dict) -> bool:
    """True when the expected FILE stayed in the pool under both runs but
    the answer PASSAGE (a chunk of that file containing expected_text) left
    it entirely — Gap 2's `expected_text_in_pool` makes this distinguishable
    from a plain pool-position shuffle (see `classify`'s use of
    `rank_movement`'s text_full delta for that case). Silently False when
    either record predates Gap 2 (no
    `expected_text_in_pool` key) or has no pool at all (--daemon) — an
    absent signal must not read as a confirmed answer."""
    bp, op = base.get("pool"), other.get("pool")
    if not bp or not op or "expected_text_in_pool" not in bp or "expected_text_in_pool" not in op:
        return False
    return (bool(bp.get("expected_in_pool")) and bool(op.get("expected_in_pool"))
            and bool(bp.get("expected_text_in_pool")) and not op.get("expected_text_in_pool"))


def classify(base: dict, other: dict) -> str:
    """One query's before/after judgement records -> a fault class. Callers
    must have already PROVEN the two runs scored the same pool (see
    `pool_identity`/`_params_match` — `compare()`/`diff_records()` only
    call this when that proof holds); otherwise use `classify_cross`. Rank
    movement at every granularity comes from `rank_movement` — the ONE
    function both this and `classify_cross` read, so the two label sets
    cannot compute "did it get worse" differently (Round 5).

    EMBEDDER            the expected passage fell OUT of the candidate pool
                        between the two runs. Given a proven-identical pool
                        this cannot legitimately happen — reaching it means
                        the pool proof and the pool membership field
                        disagree, which is a data problem, not a fault
                        class, but the branch is kept rather than hidden.
    RERANKER            still in the pool under both runs, but the reranker
                        now ranks it worse — the top-k window moved, or (on
                        a window tie, including both null) the FULL ranking
                        a top-k window cannot see past did.
    RECOVERED           the mirror of RERANKER: still in the pool under both
                        runs, and the reranker now ranks it better, at
                        whichever granularity `rank_movement` says moved
                        (window, full, or the answer passage's own rank).
    PASSAGE_LOST        the file's rank (window AND full) is unchanged, but
                        the specific answer passage left the pool entirely
                        (see `_passage_lost`) — the file is still there, its
                        answer chunk is not.
    PASSAGE_REORDERED   the file's rank (window AND full) is unchanged, but
                        the answer chunk's own rank (`expected_text_full_rank`)
                        got worse — the reranker now prefers a DIFFERENT
                        chunk of that same file over the one with the answer.
    POOL_RANK           the file's rank and the answer chunk's rank are both
                        unchanged, but WHERE the file sat in the pre-rerank
                        pool moved — informational: the hybrid ranking
                        shuffled without changing the outcome.
    OTHER               anything else that still differs.
    """
    b_in, o_in = _in_pool(base), _in_pool(other)
    if b_in is True and o_in is False:
        return "EMBEDDER"
    if b_in is True and o_in is True:
        window, full, text_full = rank_movement(base, other)
        if window != 0:
            return "RERANKER" if window > 0 else "RECOVERED"
        if full != 0:
            return "RERANKER" if full > 0 else "RECOVERED"
        if _passage_lost(base, other):
            return "PASSAGE_LOST"
        if text_full != 0:
            return "PASSAGE_REORDERED" if text_full > 0 else "RECOVERED"
        if _pool_rank(base) != _pool_rank(other):
            return "POOL_RANK"
    return "OTHER"


def classify_cross(base: dict, other: dict) -> str:
    """Factual (non-attributing) labels for a query whose two runs did NOT
    prove they scored the same pool — a rank change cannot be pinned on the
    reranker when the candidates it saw were not even known to be the same
    set. Reads the SAME `rank_movement` as `classify`, cascading through the
    same window -> full -> text-full fallback (Round 5 — previously only
    `classify` escalated to the full rank on a window tie).

    POOL_LOST    the expected file fell out of the candidate pool.
    POOL_GAINED  the expected file newly entered the candidate pool.
    RANK_WORSE   in the pool under both runs; the rank (window, or full on a
                 window tie) got worse.
    RANK_BETTER  the mirror of RANK_WORSE.
    PASSAGE_LOST see `classify`'s PASSAGE_LOST — meaningful regardless of
                 pool identity.
    TEXT_ONLY    the file's rank (window and full) is unchanged, but the
                 answer passage's own rank moved — no pool-identity-based
                 attribution available for which direction that is, unlike
                 `classify`'s PASSAGE_REORDERED/RECOVERED split.
    OTHER        anything else that still differs.
    """
    b_in, o_in = _in_pool(base), _in_pool(other)
    if b_in is True and o_in is False:
        return "POOL_LOST"
    if b_in is False and o_in is True:
        return "POOL_GAINED"
    if b_in is True and o_in is True:
        window, full, text_full = rank_movement(base, other)
        if window != 0:
            return "RANK_WORSE" if window > 0 else "RANK_BETTER"
        if full != 0:
            return "RANK_WORSE" if full > 0 else "RANK_BETTER"
        if _passage_lost(base, other):
            return "PASSAGE_LOST"
        if text_full != 0:
            return "TEXT_ONLY"
        if base.get("text_match_at_1") != other.get("text_match_at_1"):
            return "TEXT_ONLY"
    return "OTHER"


def _changed(base: dict, other: dict) -> bool:
    """A row is worth reporting when anything the attribution gate or the
    fault diagnosis reads has moved — not just the scored top-3 window
    (found_rank/text_match_at_1). A miss sliding from rank 4 to rank 20 is
    invisible to found_rank (null either way) but is exactly the kind of
    regression --details exists to surface."""
    scalar_fields = ("found_rank", "text_match_at_1", "expected_full_rank",
                     "expected_text_full_rank")
    if any(base.get(f) != other.get(f) for f in scalar_fields):
        return True
    pool_fields = ("expected_in_pool", "expected_text_in_pool",
                  "expected_best_pool_rank", "expected_text_pool_rank")
    bp, op = base.get("pool") or {}, other.get("pool") or {}
    return any(bp.get(f) != op.get(f) for f in pool_fields)


def diff_records(base_queries: list, other_queries: list, params_match: bool = True) -> list:
    """One row per query whose judgement moved (see `_changed`), in
    query-set order. Queries are matched by `index` — callers (`compare`)
    must have already verified both files scored the same query set.

    Attribution (the `classify` label set) is allowed for a row ONLY when
    `params_match` (a file-pair-wide fact) AND this query's `pool_identity`
    is `identical` on both sides (see `_pool_status`) — otherwise the row
    gets the factual-only `classify_cross` labels and a `reason` saying why
    (`"params differ"`, `"pool: order differs"`, `"pool: differs"`, or
    `"pool: unrecorded"`)."""
    other_by_index = {r["index"]: r for r in other_queries}
    rows = []
    for base in base_queries:
        other = other_by_index.get(base["index"])
        if other is None or not _changed(base, other):
            continue
        identity = _pool_status(base, other)
        allowed = params_match and identity == "identical"
        if allowed:
            reason = None
        elif not params_match:
            reason = "params differ"
        else:
            reason = f"pool: {identity.replace('_', ' ')}"
        rows.append({
            "index": base["index"],
            "expected_file": base["expected_file"],
            "base_rank": base.get("found_rank"),
            "other_rank": other.get("found_rank"),
            "base_full_rank": base.get("expected_full_rank"),
            "other_full_rank": other.get("expected_full_rank"),
            "base_text_full_rank": base.get("expected_text_full_rank"),
            "other_text_full_rank": other.get("expected_text_full_rank"),
            "base_in_pool": _in_pool(base),
            "other_in_pool": _in_pool(other),
            "pool_identity": identity,
            "class": classify(base, other) if allowed else classify_cross(base, other),
            "reason": reason,
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


def same_query(base: dict, other: dict) -> tuple:
    """(True, None) when two query records describe the SAME question, else
    (False, field) naming the FIRST of `query`/`expected_file`/
    `expected_text_contains` that differs — the ONE place that decides
    query equality, so an error message can always say exactly what
    diverged instead of just "the query differs" when it was really the
    expected file."""
    for field in ("query", "expected_file", "expected_text_contains"):
        if base.get(field) != other.get(field):
            return False, field
    return True, None


def _check_query_sets(base_details: dict, other_details: dict) -> None:
    """Refuse (ValueError) to pair queries by `index` across two --details
    files that did not provably score the SAME query set — a positional
    index means nothing if question 5 in one file is not question 5 in the
    other. Prefers `cell.query_set_sha256` (Gap 4) when both files carry it.

    When either lacks it, the fallback has two stages: FIRST the sets of
    `index` themselves must match exactly — a strict subset (one file simply
    has fewer queries) used to be silently ignored, since the old fallback
    only ever compared indices present on BOTH sides. Only once the index
    sets agree does it check `same_query` at each index, raising on the
    first mismatch and naming which field diverged."""
    b_hash = base_details.get("cell", {}).get("query_set_sha256")
    o_hash = other_details.get("cell", {}).get("query_set_sha256")
    if b_hash and o_hash:
        if b_hash != o_hash:
            raise ValueError(f"query sets differ: base query_set_sha256={b_hash!r}, "
                             f"other query_set_sha256={o_hash!r}")
        return

    base_queries = base_details.get("queries", [])
    other_queries = other_details.get("queries", [])
    b_indices = {q["index"] for q in base_queries}
    o_indices = {q["index"] for q in other_queries}
    if b_indices != o_indices:
        unmatched = sorted(b_indices ^ o_indices)
        raise ValueError(
            f"query sets differ: base has {len(b_indices)} indices, other has {len(o_indices)} "
            f"indices; unmatched indices (up to 5 of {len(unmatched)}): {unmatched[:5]}"
        )

    other_by_index = {q["index"]: q for q in other_queries}
    for q in base_queries:
        match = other_by_index[q["index"]]
        same, field = same_query(q, match)
        if not same:
            raise ValueError(f"query sets differ at index {q['index']} (field {field!r}): "
                             f"base {field}={q.get(field)!r}, other {field}={match.get(field)!r}")


def compare(base_details: dict, other_details: dict) -> dict:
    """Top-level comparison between two --details files.

    Refuses (ValueError, `_check_query_sets`) when the two files are not
    provably scoring the same query set — pairing by `index` would
    otherwise silently compare unrelated questions.

    Attribution is gated on PROOF, not the embedder's name: `_params_match`
    (file-wide) and, per query, `pool_identity` from `full_order` (see
    `diff_records`/`_pool_status`). The embedder string is read only for an
    informational header note — it no longer decides which label set a row
    gets."""
    _check_query_sets(base_details, other_details)

    base_queries, other_queries = base_details["queries"], other_details["queries"]
    params_match = _params_match(base_details, other_details)
    rows = diff_records(base_queries, other_queries, params_match)

    be, oe = embedder_of(base_details), embedder_of(other_details)
    same_embedder = bool(be) and be == oe
    warnings = []
    if not same_embedder:
        warnings.append(
            f"NOTE: base used embedder {be!r} (mode={mode_of(base_details)!r}), other used "
            f"{oe!r} (mode={mode_of(other_details)!r}) — rank changes are attributable to the "
            f"reranker only for rows whose candidate pool is PROVEN identical (see each row's "
            f"pool_identity/reason); the embedder name itself no longer gates that."
        )
    if not params_match:
        warnings.append(
            "NOTE: cell.params (candidate_pool/rerank_top_k/chunk_size/overlap_ratio) differ "
            "between the two runs — every row below is factual-only regardless of candidate "
            "identity (reason: 'params differ')."
        )

    return {
        "rows": rows,
        "counts": summarize_classes(rows),
        "deltas": aggregate_deltas(base_queries, other_queries),
        "warnings": warnings,
        "same_embedder": same_embedder,
        "params_match": params_match,
    }


def render_text(rows: list, counts: dict, deltas: dict) -> str:
    lines = []
    for r in rows:
        reason = f" [{r['reason']}]" if r.get("reason") else ""
        lines.append(
            f"[{r['index']}] {r['expected_file']}: rank {r['base_rank']} -> {r['other_rank']}, "
            f"full {r['base_full_rank']} -> {r['other_full_rank']}, "
            f"text_full {r['base_text_full_rank']} -> {r['other_text_full_rank']}, "
            f"pool {r['base_in_pool']} -> {r['other_in_pool']} :: {r['class']}{reason}"
        )
    lines.append("")
    lines.append("Summary: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no changes"))
    lines.append("")
    for metric, (b, o, d) in deltas.items():
        lines.append(f"{metric}: {b:.4f} -> {o:.4f} ({d:+.4f})")
    return "\n".join(lines)


def render_markdown(rows: list, counts: dict, deltas: dict) -> str:
    lines = ["| index | expected_file | base rank | other rank | base full | other full | "
             "base text full | other text full | base pool | other pool | class | reason |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['index']} | {r['expected_file']} | {r['base_rank']} | {r['other_rank']} | "
              f"{r['base_full_rank']} | {r['other_full_rank']} | {r['base_text_full_rank']} | "
              f"{r['other_text_full_rank']} | {r['base_in_pool']} | {r['other_in_pool']} | "
              f"{r['class']} | {r['reason'] or ''} |"
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
