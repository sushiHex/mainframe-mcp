"""Diff two --details JSON files (written by eval/evaluate.py's
write_details_file/build_detail_record) to say WHICH queries moved between
two eval runs, and whose fault it looks like: the embedder's candidate pool
(the expected passage fell out before the reranker ever saw it), or the
reranker's ordering of a pool that already had the right passage in it.

That attribution is only honest when both runs scored the SAME pool. Round
2 approximated "same pool" by comparing the embedder NAME — a proxy that
can't tell two differently-named configs producing the same retrieval from
one same-named config that quietly diverged. Records now carry candidate
IDENTITIES (`full_order`: every pool entry as doc_path + chunk_index) and
FULL ranks, not just a top-3 window, so `compare()` gates attribution on the
real thing: per query, whether the `{(doc_path, chunk_index)}` set taken
from `full_order` is IDENTICAL between the two runs, AND whether
`cell.params` (candidate_pool/rerank_top_k/chunk_size/overlap_ratio) match
file-wide. The embedder name is kept only for the header note — informational,
not the gate.

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


def _candidate_set(record: dict):
    """{(doc_path, chunk_index, content_sha256), ...} — the actual candidate
    IDENTITIES this query's `full_order` names, order-insensitive (the
    pool's own order is not what proves two runs saw the same candidates).
    CONTENT is part of the identity (Round 4): (doc_path, chunk_index) alone
    is the same before and after a document is edited and re-indexed, even
    though the reranker scored different text — content_sha256 breaks that
    false match.

    None when `full_order` is absent (records older than Gap 1, or
    --daemon mode, which has no full pool to report) OR when any entry in
    it lacks `content_sha256` (records older than Round 4) — an absent
    signal must never read as "empty/missing and therefore equal"."""
    full_order = record.get("full_order")
    if full_order is None:
        return None
    triples = []
    for e in full_order:
        content = e.get("content_sha256")
        if not content:
            return None
        triples.append((e.get("doc_path"), e.get("chunk_index"), content))
    return frozenset(triples)


def _pool_identity(base: dict, other: dict) -> str:
    """'identical' | 'differs' | 'unrecorded': whether THIS query's
    candidate set proves the two runs scored the same pool. This is the
    real proof attribution requires — an embedder NAME match is only a
    proxy for it (see the module docstring)."""
    b_set, o_set = _candidate_set(base), _candidate_set(other)
    if b_set is None or o_set is None:
        return "unrecorded"
    return "identical" if b_set == o_set else "differs"


def _params_match(base_details: dict, other_details: dict) -> bool:
    """Whether the two FILES (not one query) were scored under the same
    chunking/pooling configuration — part of the same-pool proof alongside
    per-query candidate identity (`_pool_identity`)."""
    keys = ("candidate_pool", "rerank_top_k", "chunk_size", "overlap_ratio")
    bp = base_details.get("cell", {}).get("params") or {}
    op = other_details.get("cell", {}).get("params") or {}
    return all(bp.get(k) == op.get(k) for k in keys)


def _passage_lost(base: dict, other: dict) -> bool:
    """True when the expected FILE stayed in the pool under both runs but
    the answer PASSAGE (a chunk of that file containing expected_text) left
    it entirely — Gap 2's `expected_text_in_pool` makes this distinguishable
    from a plain pool-position shuffle (see `_passage_reordering_class` for
    that case). Silently False when either record predates Gap 2 (no
    `expected_text_in_pool` key) or has no pool at all (--daemon) — an
    absent signal must not read as a confirmed answer."""
    bp, op = base.get("pool"), other.get("pool")
    if not bp or not op or "expected_text_in_pool" not in bp or "expected_text_in_pool" not in op:
        return False
    return (bool(bp.get("expected_in_pool")) and bool(op.get("expected_in_pool"))
            and bool(bp.get("expected_text_in_pool")) and not op.get("expected_text_in_pool"))


def _passage_reordering_class(base: dict, other: dict):
    """Rule 3 (generalized by Round 4 Rule 2): the expected FILE's rank is
    genuinely unchanged (the caller has already tied out found_rank AND
    expected_full_rank) but `expected_text_full_rank` — the ANSWER chunk's
    position in the full reranked order — moved: worse -> PASSAGE_REORDERED
    (an attributing label — the reranker now prefers a different chunk of
    the same file over the one containing the answer), better -> RECOVERED.
    None (not applicable, or the ranks tied) when this rule does not decide
    the query — e.g. when neither record carries expected_text_full_rank at
    all, both sides read as None and tie, which is the correct "no signal"
    outcome."""
    t_b = _rank_key(base.get("expected_text_full_rank"))
    t_o = _rank_key(other.get("expected_text_full_rank"))
    if t_o > t_b:
        return "PASSAGE_REORDERED"
    if t_o < t_b:
        return "RECOVERED"
    return None


def classify(base: dict, other: dict) -> str:
    """One query's before/after judgement records -> a fault class. Callers
    must have already PROVEN the two runs scored the same pool (see
    `_pool_identity`/`_params_match` — `compare()`/`diff_records()` only
    call this when that proof holds); otherwise use `classify_cross`.

    EMBEDDER            the expected passage fell OUT of the candidate pool
                        between the two runs. Given a proven-identical pool
                        this cannot legitimately happen — reaching it means
                        the pool proof and the pool membership field
                        disagree, which is a data problem, not a fault
                        class, but the branch is kept rather than hidden.
    RERANKER            still in the pool under both runs, but the reranker
                        now ranks it worse — either found_rank (the scored
                        top-k window) moved, or (Round 4 Rule 2) found_rank
                        TIED (including both null) while expected_full_rank
                        — the FULL ranking a top-k window cannot see past —
                        got worse. A non-find counts as worse than any
                        numeric rank at both granularities.
    RECOVERED           the mirror of RERANKER: still in the pool under both
                        runs, and the reranker now ranks it better, at
                        whichever granularity (found_rank, expected_full_rank,
                        or Rule 3's expected_text_full_rank) actually moved.
    PASSAGE_LOST        the file's rank (top-k AND full) is unchanged, but
                        the specific answer passage left the pool entirely
                        (see `_passage_lost`) — the file is still there, its
                        answer chunk is not.
    PASSAGE_REORDERED   the file's rank (top-k AND full) is unchanged, but
                        the reranker now prefers a DIFFERENT chunk of that
                        same file over the one containing the answer (see
                        `_passage_reordering_class`, keyed on
                        expected_text_full_rank).
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
        b_key, o_key = _rank_key(base.get("found_rank")), _rank_key(other.get("found_rank"))
        if o_key > b_key:
            return "RERANKER"
        if o_key < b_key:
            return "RECOVERED"
        # found_rank tied (including both None): a top-3 window cannot see
        # past itself, so escalate to the FULL rank (Round 4 Rule 2) before
        # concluding "unchanged".
        fb_key = _rank_key(base.get("expected_full_rank"))
        fo_key = _rank_key(other.get("expected_full_rank"))
        if fo_key > fb_key:
            return "RERANKER"
        if fo_key < fb_key:
            return "RECOVERED"
        if _passage_lost(base, other):
            return "PASSAGE_LOST"
        reordered = _passage_reordering_class(base, other)
        if reordered is not None:
            return reordered
        if _pool_rank(base) != _pool_rank(other):
            return "POOL_RANK"
    return "OTHER"


def classify_cross(base: dict, other: dict) -> str:
    """Factual (non-attributing) labels for a query whose two runs did NOT
    prove they scored the same pool — a rank change cannot be pinned on the
    reranker when the candidates it saw were not even known to be the same
    set (Rule 3: without pool identity, a text_match_at_1 flip is TEXT_ONLY,
    never PASSAGE_REORDERED).

    POOL_LOST    the expected file fell out of the candidate pool.
    POOL_GAINED  the expected file newly entered the candidate pool.
    RANK_WORSE   in the pool under both runs; the final rank got worse.
    RANK_BETTER  in the pool under both runs; the final rank got better.
    PASSAGE_LOST see `classify`'s PASSAGE_LOST — meaningful regardless of
                 pool identity.
    TEXT_ONLY    found_rank is unchanged but text_match_at_1 differs, with
                 no pool-identity-based explanation available.
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
    `params_match` (a file-pair-wide fact) AND this query's candidate set is
    `identical` per `_pool_identity` — otherwise the row gets the
    factual-only `classify_cross` labels and a `reason` saying why
    (`"params differ"`, `"pool: differs"`, or `"pool: unrecorded"`)."""
    other_by_index = {r["index"]: r for r in other_queries}
    rows = []
    for base in base_queries:
        other = other_by_index.get(base["index"])
        if other is None or not _changed(base, other):
            continue
        identity = _pool_identity(base, other)
        allowed = params_match and identity == "identical"
        if allowed:
            reason = None
        elif not params_match:
            reason = "params differ"
        else:
            reason = f"pool: {identity}"
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


def _check_query_sets(base_details: dict, other_details: dict) -> None:
    """Refuse (ValueError) to pair queries by `index` across two --details
    files that did not provably score the SAME query set — a positional
    index means nothing if question 5 in one file is not question 5 in the
    other. Prefers `cell.query_set_sha256` (Gap 4) when both files carry it.

    When either lacks it, the fallback has two stages (Round 4 Rule 3): FIRST
    the sets of `index` themselves must match exactly — a strict subset (one
    file simply has fewer queries) used to be silently ignored, since the
    old fallback only ever compared indices present on BOTH sides. Only once
    the index sets agree does it compare the actual `query` string at each
    index, raising on the first mismatch."""
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
        if match.get("query") != q.get("query"):
            raise ValueError(f"query sets differ at index {q['index']}: "
                             f"base query={q.get('query')!r}, other query={match.get('query')!r}")


def compare(base_details: dict, other_details: dict) -> dict:
    """Top-level comparison between two --details files.

    Refuses (ValueError, `_check_query_sets`) when the two files are not
    provably scoring the same query set — pairing by `index` would
    otherwise silently compare unrelated questions.

    Attribution is gated on PROOF, not the embedder's name: `_params_match`
    (file-wide) and, per query, `_pool_identity` from `full_order`'s
    candidate set (see `diff_records`). The embedder string is read only for
    an informational header note — it no longer decides which label set a
    row gets."""
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
