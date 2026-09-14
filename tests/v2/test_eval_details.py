"""GPU-free tests for --details per-query judgement records (eval/evaluate.py)
and the eval/details_diff.py diff tool. No model, LanceDB, or mainframe
imports — everything here is fed fake raw_results/reranked rows shaped like
the real ones store.search()/reranker.rerank() produce."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- 1. judge_query: pool membership + reranked rank ----------

def test_judge_query_pool_membership_and_reranked_rank():
    ev = _load("eval_evaluate_details_1", "eval/evaluate.py")

    # Six unrelated candidates, then the expected file at pool position 7
    # (1-based), reranked to position 2.
    raw_results = [{"doc_path": f"c:/r/p/docs/other{i}.md", "text": "irrelevant"} for i in range(6)]
    raw_results.append({"doc_path": "c:/r/p/docs/target.md", "text": "the answer is here"})
    reranked = [(0, 0.9), (6, 0.8), (1, 0.7)]

    # `full_reranked` omitted -> falls back to the `reranked` window, so
    # expected_full_rank/expected_text_full_rank reflect that window (rank 2
    # here), not the true pool position (7) -- Gap 1's full-pool visibility
    # is exercised separately by test_judge_query_full_ordering_survives_a_
    # narrow_top_k below.
    judged = ev.judge_query(raw_results, reranked, "target.md", "the answer")
    assert judged == {"expected_in_pool": True, "expected_best_pool_rank": 7,
                       "expected_text_in_pool": True, "expected_text_pool_rank": 7,
                       "found_rank": 2, "text_match_at_1": False,
                       "expected_full_rank": 2, "expected_text_full_rank": 2}

    # Expected file entirely absent from the pool.
    judged_absent = ev.judge_query(raw_results[:6], [(0, 0.9)], "target.md", "the answer")
    assert judged_absent == {"expected_in_pool": False, "expected_best_pool_rank": None,
                              "expected_text_in_pool": False, "expected_text_pool_rank": None,
                              "found_rank": None, "text_match_at_1": False,
                              "expected_full_rank": None, "expected_text_full_rank": None}

    # Reranked at rank 1 AND the expected text is present -> text_match_at_1 True.
    raw_rank1 = [{"doc_path": "c:/r/p/docs/target.md", "text": "the answer is here"},
                 {"doc_path": "c:/r/p/docs/other.md", "text": "irrelevant"}]
    judged_rank1 = ev.judge_query(raw_rank1, [(0, 0.9), (1, 0.5)], "target.md", "the answer")
    assert judged_rank1 == {"expected_in_pool": True, "expected_best_pool_rank": 1,
                             "expected_text_in_pool": True, "expected_text_pool_rank": 1,
                             "found_rank": 1, "text_match_at_1": True,
                             "expected_full_rank": 1, "expected_text_full_rank": 1}


def test_judge_query_distinguishes_file_level_from_text_level_pool_rank():
    """Gap 2: pool presence is judged on doc_path (any chunk of the expected
    FILE), but text_match needs the ANSWER passage -- the file can sit in
    the pool well before the chunk that actually contains the phrase."""
    ev = _load("eval_evaluate_details_1b", "eval/evaluate.py")

    raw_results = [{"doc_path": "c:/r/p/docs/other.md", "text": "irrelevant"}]           # rank 1
    raw_results.append({"doc_path": "c:/r/p/docs/target.md", "text": "no phrase here"})  # rank 2: file, no text
    raw_results += [{"doc_path": f"c:/r/p/docs/other{i}.md", "text": "irrelevant"}
                    for i in range(6)]                                                    # ranks 3-8
    raw_results.append({"doc_path": "c:/r/p/docs/target.md", "text": "here is the answer"})  # rank 9: file + text

    judged = ev.judge_query(raw_results, [], "target.md", "the answer")
    assert judged["expected_best_pool_rank"] == 2
    assert judged["expected_text_in_pool"] is True
    assert judged["expected_text_pool_rank"] == 9


def test_judge_query_full_ordering_survives_a_narrow_top_k():
    """Gap 1: expected_full_rank comes from the reranker's FULL ordering
    (full_reranked), not the scored top-k window -- it can see rank 7 even
    when found_rank (bounded by the window) reports null."""
    ev = _load("eval_evaluate_details_1c", "eval/evaluate.py")

    raw_results = [{"doc_path": f"c:/r/p/docs/other{i}.md", "text": "irrelevant"} for i in range(9)]
    raw_results.append({"doc_path": "c:/r/p/docs/target.md", "text": "the answer"})
    full_reranked = [(0, 0.99), (1, 0.95), (2, 0.90), (3, 0.85), (4, 0.80), (5, 0.75),
                     (9, 0.70), (6, 0.65), (7, 0.60), (8, 0.55)]   # target (idx 9) at full rank 7
    window = full_reranked[:3]   # RERANK_TOP_K-sized scored window: target not in it

    judged = ev.judge_query(raw_results, window, "target.md", "the answer", full_reranked=full_reranked)
    assert judged["found_rank"] is None
    assert judged["expected_full_rank"] == 7
    assert judged["expected_text_full_rank"] == 7


# ---------- 2. evaluate() aggregates, computed through judge_query, match hand math ----------

class _FixtureStore:
    def __init__(self, per_query):
        self.per_query = per_query

    def search(self, query_embedding, top_k=20, query_text="", include_captures=False):
        return self.per_query[query_text]


class _FixtureReranker:
    def __init__(self, per_query):
        self.per_query = per_query

    def rerank(self, query, texts, top_k=3, headings=None):
        return self.per_query[query]


class _FixtureEmbedder:
    def embed_query(self, query):
        return [0.0]


def test_evaluate_aggregates_match_hand_computed_values():
    ev = _load("eval_evaluate_details_2", "eval/evaluate.py")

    raw_a = [{"doc_path": "c:/r/p/docs/x.md", "text": "irrelevant"},
             {"doc_path": "c:/r/p/docs/a.md", "text": "the answer is 42"}]
    raw_b = [{"doc_path": "c:/r/p/docs/x.md", "text": "irrelevant"}]
    store = _FixtureStore({"query a": raw_a, "query b": raw_b})
    # query a: expected file reranked to rank 1 with the expected text present.
    # query b: expected file never appears in the pool at all.
    reranker = _FixtureReranker({"query a": [(1, 0.9), (0, 0.5)], "query b": [(0, 0.9)]})
    embedder = _FixtureEmbedder()
    test_queries = [
        {"query": "query a", "expected_file": "a.md", "expected_text_contains": "the answer"},
        {"query": "query b", "expected_file": "b.md", "expected_text_contains": "nope"},
    ]

    results = ev.evaluate(embedder, reranker, store, test_queries)

    # Hand-computed: query a hits at 1/3/5 with a text match (rr=1.0); query b
    # never found (rr=0.0) -- the exact arithmetic the pre-refactor inline
    # loop produced, now routed through judge_query.
    assert results["total_queries"] == 2
    assert results["hit_at_1"] == pytest.approx(0.5)
    assert results["hit_at_3"] == pytest.approx(0.5)
    assert results["hit_at_5"] == pytest.approx(0.5)
    assert results["text_match_at_1"] == pytest.approx(0.5)
    assert results["mrr"] == pytest.approx(0.5)
    assert results["score"] == pytest.approx(0.5 * 0.4 + 0.5 * 0.3 + 0.5 * 0.3)


class _RecordingReranker:
    """Simulates the REAL contract confirmed by reading reranker.rerank():
    every candidate is scored regardless of top_k, and top_k only slices the
    already-sorted result afterward. Records the top_k it was called with
    per query, so a test can see what --details actually asks for."""
    def __init__(self, per_query_order):
        self.calls = []            # [(query, top_k), ...]
        self.per_query_order = per_query_order

    def rerank(self, query, texts, top_k=None, headings=None):
        self.calls.append((query, top_k))
        order = self.per_query_order[query]
        k = top_k if top_k is not None else len(order)
        return order[:k]


def test_evaluate_details_requests_full_ordering_without_moving_aggregates():
    """Gap 1: with --details active, evaluate() must ask the reranker for
    the FULL ordering (top_k=len(texts)), not RERANK_TOP_K, and slice the
    scored window off the front of it. Aggregates must come out
    byte-identical either way, and the details record must see rank 7 where
    the aggregate-only path only ever reports null."""
    ev = _load("eval_evaluate_details_gap1", "eval/evaluate.py")

    raw_deep = [{"doc_path": f"c:/r/p/docs/other{i}.md", "text": "irrelevant"} for i in range(9)]
    raw_deep.append({"doc_path": "c:/r/p/docs/target.md", "text": "the answer is here"})
    order_deep = [(0, 0.99), (1, 0.95), (2, 0.90), (3, 0.85), (4, 0.80), (5, 0.75),
                  (9, 0.70), (6, 0.65), (7, 0.60), (8, 0.55)]   # target (idx 9) at rank 7

    raw_shallow = [{"doc_path": "c:/r/p/docs/target2.md", "text": "irrelevant"},
                   {"doc_path": "c:/r/p/docs/o1.md", "text": "irrelevant"},
                   {"doc_path": "c:/r/p/docs/o2.md", "text": "irrelevant"},
                   {"doc_path": "c:/r/p/docs/o3.md", "text": "irrelevant"},
                   {"doc_path": "c:/r/p/docs/o4.md", "text": "irrelevant"}]
    order_shallow = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)]   # target2 at rank 1

    fixture_store = _FixtureStore({"deep": raw_deep, "shallow": raw_shallow})
    embedder = _FixtureEmbedder()
    test_queries = [
        {"query": "deep", "expected_file": "target.md", "expected_text_contains": "the answer"},
        {"query": "shallow", "expected_file": "target2.md", "expected_text_contains": ""},
    ]

    reranker_without = _RecordingReranker({"deep": order_deep, "shallow": order_shallow})
    results_without = ev.evaluate(embedder, reranker_without, fixture_store, test_queries)
    assert reranker_without.calls == [("deep", ev.RERANK_TOP_K), ("shallow", ev.RERANK_TOP_K)]

    reranker_with = _RecordingReranker({"deep": order_deep, "shallow": order_shallow})
    ev.DETAILS_DUMP = []
    results_with = ev.evaluate(embedder, reranker_with, fixture_store, test_queries)
    assert reranker_with.calls == [("deep", len(raw_deep)), ("shallow", len(raw_shallow))]

    assert results_without == results_with          # --details never moves an aggregate

    deep_record, shallow_record = ev.DETAILS_DUMP
    assert deep_record["found_rank"] is None                  # outside the scored top-3 window
    assert deep_record["expected_full_rank"] == 7              # visible in the full ordering
    assert shallow_record["found_rank"] == 1
    assert shallow_record["expected_full_rank"] == 1
    ev.DETAILS_DUMP = None


# ---------- 3. details_diff.classify + aggregate_deltas ----------

def _record(found_rank, pool_rank, in_pool=True, text_match=False):
    pool = None if in_pool is None else {"size": 20, "expected_in_pool": in_pool,
                                          "expected_best_pool_rank": pool_rank}
    return {"found_rank": found_rank, "text_match_at_1": text_match, "pool": pool}


def test_classify_returns_each_class_exactly_once():
    dd = _load("eval_details_diff_3", "eval/details_diff.py")

    embedder_case = dd.classify(_record(2, 5, True), _record(None, None, False))
    reranker_case = dd.classify(_record(1, 3, True), _record(3, 3, True))
    recovered_case = dd.classify(_record(3, 3, True), _record(1, 3, True))
    pool_rank_case = dd.classify(_record(2, 4, True, text_match=False),
                                  _record(2, 9, True, text_match=True))
    other_case = dd.classify(_record(1, 1, True, text_match=True),
                              _record(1, 1, True, text_match=False))

    assert embedder_case == "EMBEDDER"
    assert reranker_case == "RERANKER"
    assert recovered_case == "RECOVERED"
    assert pool_rank_case == "POOL_RANK"
    assert other_case == "OTHER"


def test_aggregate_deltas_matches_hand_computed_value():
    dd = _load("eval_details_diff_3b", "eval/details_diff.py")

    base_queries = [
        {"index": 0, "found_rank": 1, "text_match_at_1": True},
        {"index": 1, "found_rank": None, "text_match_at_1": False},
    ]
    other_queries = [
        {"index": 0, "found_rank": 2, "text_match_at_1": False},
        {"index": 1, "found_rank": 1, "text_match_at_1": True},
    ]

    deltas = dd.aggregate_deltas(base_queries, other_queries)

    # base: hit@1=0.5 (q0 only), hit@3=0.5, mrr=(1+0)/2=0.5, text@1=0.5
    # other: hit@1=0.5 (q1 only), hit@3=1.0 (both within rank 3), mrr=(0.5+1)/2=0.75, text@1=0.5
    assert deltas["hit_at_1"] == pytest.approx((0.5, 0.5, 0.0))
    assert deltas["hit_at_3"] == pytest.approx((0.5, 1.0, 0.5))
    assert deltas["mrr"] == pytest.approx((0.5, 0.75, 0.25))
    assert deltas["text_match_at_1"] == pytest.approx((0.5, 0.5, 0.0))


def test_classify_passage_lost_in_both_branches():
    """Gap 3: the file stayed in the pool under both runs, but the answer
    PASSAGE (not just the file) left it -- distinct from POOL_RANK/OTHER,
    and available whether or not the two runs used the same embedder."""
    dd = _load("eval_details_diff_gap3_passage", "eval/details_diff.py")

    base = {"found_rank": 1, "text_match_at_1": True,
            "pool": {"expected_in_pool": True, "expected_best_pool_rank": 1,
                     "expected_text_in_pool": True, "expected_text_pool_rank": 1}}
    other = {"found_rank": 1, "text_match_at_1": False,
             "pool": {"expected_in_pool": True, "expected_best_pool_rank": 1,
                      "expected_text_in_pool": False, "expected_text_pool_rank": None}}

    assert dd.classify(base, other) == "PASSAGE_LOST"
    assert dd.classify_cross(base, other) == "PASSAGE_LOST"


def _full_order_entry(doc_path, chunk_index, rank=1, expected=True, text_match=False,
                      content_sha256="c0ffee0000000000", pool_rank=None):
    # pool_rank defaults to `rank` -- every pre-Round-5 test built full_order
    # lists in ascending rank order already, so this default reproduces
    # their intended pool order without touching every call site.
    return {"rank": rank, "doc_path": doc_path, "chunk_index": chunk_index,
            "rerank_score": 0.5, "expected": expected, "text_match": text_match,
            "content_sha256": content_sha256, "pool_rank": pool_rank if pool_rank is not None else rank}


# ---------- Round 3, Rule 1: attribution requires the SAME CANDIDATE SET,
# not the same embedder name ----------

def test_diff_records_uses_attributing_label_when_candidate_sets_are_identical():
    dd = _load("eval_details_diff_r3_rule1a", "eval/details_diff.py")
    shared_full_order = [_full_order_entry("a.md", 0), _full_order_entry("b.md", 1, rank=2, expected=False)]
    common_pool = {"size": 2, "expected_in_pool": True, "expected_best_pool_rank": 1,
                   "expected_text_in_pool": True, "expected_text_pool_rank": 1}
    base_q = [{"index": 0, "expected_file": "a.md", "found_rank": 1, "text_match_at_1": True,
               "expected_full_rank": 1, "expected_text_full_rank": 1,
               "pool": common_pool, "full_order": shared_full_order}]
    other_q = [{"index": 0, "expected_file": "a.md", "found_rank": 2, "text_match_at_1": False,
                "expected_full_rank": 2, "expected_text_full_rank": 2,
                "pool": common_pool, "full_order": shared_full_order}]

    rows = dd.diff_records(base_q, other_q, params_match=True)
    assert len(rows) == 1
    assert rows[0]["pool_identity"] == "identical"
    assert rows[0]["class"] == "RERANKER"
    assert rows[0]["reason"] is None


def test_diff_records_falls_back_to_factual_label_when_one_candidate_swapped():
    """Same embedder-name situation as before, but the pool ACTUALLY
    changed (one candidate swapped) -- must get the factual label with a
    reason, not RERANKER."""
    dd = _load("eval_details_diff_r3_rule1b", "eval/details_diff.py")
    base_full_order = [_full_order_entry("a.md", 0), _full_order_entry("b.md", 1, rank=2)]
    other_full_order = [_full_order_entry("a.md", 0), _full_order_entry("c.md", 1, rank=2)]  # b -> c
    common_pool = {"size": 2, "expected_in_pool": True, "expected_best_pool_rank": 1,
                   "expected_text_in_pool": True, "expected_text_pool_rank": 1}
    base_q = [{"index": 0, "expected_file": "a.md", "found_rank": 1, "text_match_at_1": True,
               "expected_full_rank": 1, "expected_text_full_rank": 1,
               "pool": common_pool, "full_order": base_full_order}]
    other_q = [{"index": 0, "expected_file": "a.md", "found_rank": 2, "text_match_at_1": False,
                "expected_full_rank": 2, "expected_text_full_rank": 2,
                "pool": common_pool, "full_order": other_full_order}]

    rows = dd.diff_records(base_q, other_q, params_match=True)
    assert len(rows) == 1
    assert rows[0]["pool_identity"] == "differs"
    assert rows[0]["class"] == "RANK_WORSE"          # classify_cross, never RERANKER
    assert rows[0]["reason"] == "pool: differs"


def test_diff_records_treats_missing_full_order_as_unrecorded():
    dd = _load("eval_details_diff_r3_rule1c", "eval/details_diff.py")
    base_q = [{"index": 0, "expected_file": "a.md", "found_rank": 1, "text_match_at_1": True,
               "expected_full_rank": 1, "expected_text_full_rank": 1,
               "pool": {"size": 2, "expected_in_pool": True, "expected_best_pool_rank": 1,
                        "expected_text_in_pool": True, "expected_text_pool_rank": 1},
               "full_order": None}]                       # e.g. --daemon-style record
    other_q = [{"index": 0, "expected_file": "a.md", "found_rank": 2, "text_match_at_1": False,
                "expected_full_rank": 2, "expected_text_full_rank": 2,
                "pool": None, "full_order": None}]

    rows = dd.diff_records(base_q, other_q, params_match=True)
    assert len(rows) == 1
    assert rows[0]["pool_identity"] == "unrecorded"
    assert rows[0]["reason"] == "pool: unrecorded"
    assert rows[0]["class"] != "RERANKER"              # factual label only


def test_diff_records_forces_factual_label_when_params_differ_despite_identical_candidates():
    dd = _load("eval_details_diff_r3_rule1d", "eval/details_diff.py")
    shared_full_order = [_full_order_entry("a.md", 0), _full_order_entry("b.md", 1, rank=2)]
    common_pool = {"size": 2, "expected_in_pool": True, "expected_best_pool_rank": 1,
                   "expected_text_in_pool": True, "expected_text_pool_rank": 1}
    base_q = [{"index": 0, "expected_file": "a.md", "found_rank": 1, "text_match_at_1": True,
               "expected_full_rank": 1, "expected_text_full_rank": 1,
               "pool": common_pool, "full_order": shared_full_order}]
    other_q = [{"index": 0, "expected_file": "a.md", "found_rank": 2, "text_match_at_1": False,
                "expected_full_rank": 2, "expected_text_full_rank": 2,
                "pool": common_pool, "full_order": shared_full_order}]

    rows = dd.diff_records(base_q, other_q, params_match=False)
    assert rows[0]["pool_identity"] == "identical"      # candidates matched...
    assert rows[0]["reason"] == "params differ"          # ...but params didn't
    assert rows[0]["class"] == "RANK_WORSE"


def test_compare_gates_attribution_on_candidate_identity_not_embedder_name():
    """The root-cause fix: a DIFFERENT embedder name no longer forces
    factual labels for every row -- query 0's candidate set happens to be
    identical (attribution allowed), query 1's differs (factual only)."""
    dd = _load("eval_details_diff_r3_compare", "eval/details_diff.py")
    shared_full_order = [_full_order_entry("a.md", 0), _full_order_entry("x.md", 1, rank=2)]
    differing_full_order = [_full_order_entry("b.md", 0), _full_order_entry("y.md", 1, rank=2)]
    common_params = {"candidate_pool": 20, "rerank_top_k": 3, "chunk_size": 256, "overlap_ratio": 0.35}
    pool_a = {"size": 2, "expected_in_pool": True, "expected_best_pool_rank": 1,
             "expected_text_in_pool": True, "expected_text_pool_rank": 1}
    pool_b = {"size": 2, "expected_in_pool": True, "expected_best_pool_rank": 1,
             "expected_text_in_pool": True, "expected_text_pool_rank": 1}

    base_details = {"cell": {"models": {"embedder": "harrier-v1"}, "mode": "live", "params": common_params},
                    "queries": [
        {"index": 0, "query": "q0", "expected_file": "a.md", "found_rank": 1, "text_match_at_1": True,
         "expected_full_rank": 1, "expected_text_full_rank": 1, "pool": pool_a,
         "full_order": shared_full_order},
        {"index": 1, "query": "q1", "expected_file": "b.md", "found_rank": 1, "text_match_at_1": True,
         "expected_full_rank": 1, "expected_text_full_rank": 1, "pool": pool_b,
         "full_order": differing_full_order},
    ]}
    other_details = {"cell": {"models": {"embedder": "harrier-v2"}, "mode": "live", "params": common_params},
                     "queries": [
        {"index": 0, "query": "q0", "expected_file": "a.md", "found_rank": 2, "text_match_at_1": False,
         "expected_full_rank": 2, "expected_text_full_rank": 2, "pool": pool_a,
         "full_order": shared_full_order},                          # SAME candidates -> attribution OK
        {"index": 1, "query": "q1", "expected_file": "b.md", "found_rank": 2, "text_match_at_1": False,
         "expected_full_rank": 2, "expected_text_full_rank": 2, "pool": pool_b,
         "full_order": [_full_order_entry("b.md", 0), _full_order_entry("z.md", 1, rank=2)]},  # differs
    ]}

    result = dd.compare(base_details, other_details)
    assert result["same_embedder"] is False
    assert result["rows"][0]["class"] == "RERANKER"        # attributing despite different embedder NAME
    assert result["rows"][0]["reason"] is None
    assert result["rows"][1]["class"] == "RANK_WORSE"       # factual: candidate set differed
    assert result["rows"][1]["reason"] == "pool: differs"
    assert len(result["warnings"]) == 1
    assert "embedder" in result["warnings"][0].lower()


def test_compare_same_embedder_and_identical_pool_uses_attribution_labels():
    dd = _load("eval_details_diff_r3_compare_same", "eval/details_diff.py")
    shared_full_order = [_full_order_entry("a.md", 0), _full_order_entry("x.md", 1, rank=2)]
    common_params = {"candidate_pool": 20, "rerank_top_k": 3, "chunk_size": 256, "overlap_ratio": 0.35}
    common_pool = {"size": 2, "expected_in_pool": True, "expected_best_pool_rank": 1,
                   "expected_text_in_pool": True, "expected_text_pool_rank": 1}

    base_details = {"cell": {"models": {"embedder": "qwen3-8b"}, "mode": "live", "params": common_params},
                    "queries": [{"index": 0, "query": "q0", "expected_file": "a.md", "found_rank": 1,
                                 "text_match_at_1": True, "expected_full_rank": 1, "expected_text_full_rank": 1,
                                 "pool": common_pool, "full_order": shared_full_order}]}
    other_details = {"cell": {"models": {"embedder": "qwen3-8b"}, "mode": "live", "params": common_params},
                     "queries": [{"index": 0, "query": "q0", "expected_file": "a.md", "found_rank": 3,
                                  "text_match_at_1": False, "expected_full_rank": 3, "expected_text_full_rank": 3,
                                  "pool": common_pool, "full_order": shared_full_order}]}

    result = dd.compare(base_details, other_details)
    assert result["same_embedder"] is True
    assert result["params_match"] is True
    assert result["rows"][0]["class"] == "RERANKER"
    assert result["rows"][0]["reason"] is None
    assert result["warnings"] == []


# ---------- Round 3, Rule 2: a row is "changed" when full ranks moved,
# even if found_rank/text_match_at_1 did not ----------

def test_changed_full_rank_movement_produces_one_row_even_with_found_rank_unchanged():
    dd = _load("eval_details_diff_r3_rule2", "eval/details_diff.py")
    common_pool = {"size": 20, "expected_in_pool": True, "expected_best_pool_rank": 5,
                   "expected_text_in_pool": True, "expected_text_pool_rank": 5}
    base_q = [{"index": 0, "expected_file": "a.md", "found_rank": None, "text_match_at_1": False,
               "expected_full_rank": 4, "expected_text_full_rank": 4, "pool": dict(common_pool),
               "full_order": None}]
    other_q = [{"index": 0, "expected_file": "a.md", "found_rank": None, "text_match_at_1": False,
                "expected_full_rank": 20, "expected_text_full_rank": 20, "pool": dict(common_pool),
                "full_order": None}]

    rows = dd.diff_records(base_q, other_q)
    assert len(rows) == 1
    assert rows[0]["base_full_rank"] == 4
    assert rows[0]["other_full_rank"] == 20


def test_render_text_and_markdown_include_full_rank_columns():
    dd = _load("eval_details_diff_r3_render", "eval/details_diff.py")
    rows = [{"index": 0, "expected_file": "a.md", "base_rank": None, "other_rank": None,
             "base_full_rank": 4, "other_full_rank": 20, "base_text_full_rank": 4,
             "other_text_full_rank": 20, "base_in_pool": True, "other_in_pool": True,
             "pool_identity": "identical", "class": "POOL_RANK", "reason": None}]
    text = dd.render_text(rows, {}, {})
    assert "4" in text and "20" in text
    md = dd.render_markdown(rows, {}, {})
    assert "4" in md and "20" in md


# ---------- Round 3, Rule 3: passage reordering WITHIN the expected file ----------

def test_classify_passage_reordered_within_expected_file():
    dd = _load("eval_details_diff_r3_rule3", "eval/details_diff.py")

    base = {"found_rank": 1, "text_match_at_1": True, "expected_text_full_rank": 1,
            "pool": {"expected_in_pool": True, "expected_best_pool_rank": 1,
                     "expected_text_in_pool": True, "expected_text_pool_rank": 1}}
    other = {"found_rank": 1, "text_match_at_1": False, "expected_text_full_rank": 3,
             "pool": {"expected_in_pool": True, "expected_best_pool_rank": 1,
                      "expected_text_in_pool": True, "expected_text_pool_rank": 3}}

    assert dd.classify(base, other) == "PASSAGE_REORDERED"     # worse
    assert dd.classify(other, base) == "RECOVERED"             # better (swapped direction)
    assert dd.classify_cross(base, other) == "TEXT_ONLY"       # without pool identity


# ---------- Round 3, Rule 4: refuse mismatched query sets ----------

def test_compare_raises_on_unequal_query_set_hashes():
    dd = _load("eval_details_diff_r3_rule4a", "eval/details_diff.py")
    base_details = {"cell": {"query_set_sha256": "aaa"}, "queries": []}
    other_details = {"cell": {"query_set_sha256": "bbb"}, "queries": []}
    with pytest.raises(ValueError) as exc_info:
        dd.compare(base_details, other_details)
    assert "aaa" in str(exc_info.value) and "bbb" in str(exc_info.value)


def test_compare_passes_on_equal_query_set_hashes():
    dd = _load("eval_details_diff_r3_rule4b", "eval/details_diff.py")
    base_details = {"cell": {"query_set_sha256": "same-hash", "models": {"embedder": "e"}, "mode": "live"},
                    "queries": [{"index": 0, "query": "q0", "expected_file": "a.md", "found_rank": 1,
                                 "text_match_at_1": True, "pool": None, "full_order": None}]}
    other_details = {"cell": {"query_set_sha256": "same-hash", "models": {"embedder": "e"}, "mode": "live"},
                     "queries": [{"index": 0, "query": "q0", "expected_file": "a.md", "found_rank": 1,
                                  "text_match_at_1": True, "pool": None, "full_order": None}]}
    result = dd.compare(base_details, other_details)      # must not raise
    assert result["rows"] == []


def test_compare_raises_on_legacy_files_with_differing_query_at_index():
    """Neither file carries query_set_sha256 (pre-Gap-4) -- fall back to
    comparing the actual query text at each shared index."""
    dd = _load("eval_details_diff_r3_rule4c", "eval/details_diff.py")
    base_details = {"cell": {"models": {"embedder": "e"}, "mode": "live"}, "queries": [
        {"index": 0, "query": "alpha query", "expected_file": "a.md", "found_rank": 1,
         "text_match_at_1": True, "pool": None, "full_order": None}]}
    other_details = {"cell": {"models": {"embedder": "e"}, "mode": "live"}, "queries": [
        {"index": 0, "query": "a totally different query", "expected_file": "a.md", "found_rank": 1,
         "text_match_at_1": True, "pool": None, "full_order": None}]}
    with pytest.raises(ValueError) as exc_info:
        dd.compare(base_details, other_details)
    assert "0" in str(exc_info.value)          # the index is named in the message


def test_compare_raises_when_hashless_files_have_different_index_sets():
    """Round 4, Rule 3: a strict-subset query set must be refused even
    though every SHARED index's query string matches -- the old fallback
    silently ignored indices present on only one side."""
    dd = _load("eval_details_diff_r4_rule3", "eval/details_diff.py")
    base_details = {"cell": {"models": {"embedder": "e"}, "mode": "live"}, "queries": [
        {"index": 0, "query": "q0"}, {"index": 1, "query": "q1"}, {"index": 2, "query": "q2"}]}
    other_details = {"cell": {"models": {"embedder": "e"}, "mode": "live"}, "queries": [
        {"index": 0, "query": "q0"}, {"index": 1, "query": "q1"}]}   # strict subset -- missing index 2
    with pytest.raises(ValueError) as exc_info:
        dd.compare(base_details, other_details)
    msg = str(exc_info.value)
    assert "3" in msg and "2" in msg           # counts on each side (3 vs 2)


# ---------- Round 4: candidate identity includes CONTENT, classify escalates
# to full ranks on a top-k tie ----------

def test_build_detail_record_content_sha256_prefers_row_content_hash():
    """store.py's schema carries `content_hash` on every in-process
    raw_results row (sha256 of the normalized chunk text, computed once at
    ingest) -- build_detail_record must reuse it rather than re-hashing,
    truncated to the first 16 hex chars."""
    ev = _load("eval_evaluate_r4_content_hash_a", "eval/evaluate.py")
    full_hash = "abc123def4567890" * 4
    raw_results = [{"doc_path": "c:/r/p/docs/a.md", "chunk_index": 0, "heading": "",
                    "text": "hello world", "content_hash": full_hash}]
    reranked = [(0, 0.9)]
    judged = ev.judge_query(raw_results, reranked, "a.md", "")

    record = ev.build_detail_record(0, {"query": "q", "expected_file": "a.md"}, raw_results,
                                    reranked, judged, has_pool=True, full_reranked=reranked)

    assert record["top"][0]["content_sha256"] == full_hash[:16]
    assert record["full_order"][0]["content_sha256"] == full_hash[:16]


def test_build_detail_record_hashes_text_when_content_hash_missing():
    """The --daemon path's normalized rows carry no `content_hash` (never
    on the wire) -- fall back to hashing `text` at record time."""
    ev = _load("eval_evaluate_r4_content_hash_b", "eval/evaluate.py")
    import hashlib
    raw_results = [{"doc_path": "c:/r/p/docs/a.md", "chunk_index": 0, "heading": "", "text": "hello world"}]
    reranked = [(0, 0.9)]
    judged = ev.judge_query(raw_results, reranked, "a.md", "")

    record = ev.build_detail_record(0, {"query": "q", "expected_file": "a.md"}, raw_results,
                                    reranked, judged, has_pool=True, full_reranked=reranked)

    expected = hashlib.sha256(b"hello world").hexdigest()[:16]
    assert record["top"][0]["content_sha256"] == expected
    assert record["full_order"][0]["content_sha256"] == expected


def test_build_detail_record_pool_rank_is_the_pre_rerank_pool_position():
    """Round 5, Rule 1: pool_rank is orig_idx + 1 -- the position the
    reranker's INPUT (the pool BEFORE reranking) held each candidate at,
    independent of where reranking later placed it."""
    ev = _load("eval_evaluate_r5_pool_rank", "eval/evaluate.py")
    raw_results = [{"doc_path": "c:/r/p/docs/a.md", "chunk_index": 0, "heading": "", "text": "alpha"},
                   {"doc_path": "c:/r/p/docs/b.md", "chunk_index": 0, "heading": "", "text": "beta"}]
    reranked = [(1, 0.9), (0, 0.5)]      # reranker promoted pool position 2 (b.md) to rank 1
    judged = ev.judge_query(raw_results, reranked, "a.md", "")

    record = ev.build_detail_record(0, {"query": "q", "expected_file": "a.md"}, raw_results,
                                    reranked, judged, has_pool=True, full_reranked=reranked)

    top_by_doc = {e["doc_path"]: e for e in record["top"]}
    assert top_by_doc["c:/r/p/docs/b.md"]["rank"] == 1 and top_by_doc["c:/r/p/docs/b.md"]["pool_rank"] == 2
    assert top_by_doc["c:/r/p/docs/a.md"]["rank"] == 2 and top_by_doc["c:/r/p/docs/a.md"]["pool_rank"] == 1
    full_by_doc = {e["doc_path"]: e for e in record["full_order"]}
    assert full_by_doc["c:/r/p/docs/a.md"]["pool_rank"] == 1
    assert full_by_doc["c:/r/p/docs/b.md"]["pool_rank"] == 2


def test_diff_records_treats_same_path_and_chunk_but_different_content_as_pool_differs():
    """Round 4, Rule 1: (doc_path, chunk_index) alone survives a doc
    edit + re-index -- the reranker saw DIFFERENT text even though the
    positional identity looks unchanged. content_sha256 must break the
    false match."""
    dd = _load("eval_details_diff_r4_rule1", "eval/details_diff.py")
    base_full_order = [_full_order_entry("a.md", 0, content_sha256="hash-before"),
                       _full_order_entry("b.md", 1, rank=2)]
    other_full_order = [_full_order_entry("a.md", 0, content_sha256="hash-AFTER-edit"),  # same path+chunk!
                        _full_order_entry("b.md", 1, rank=2)]
    common_pool = {"size": 2, "expected_in_pool": True, "expected_best_pool_rank": 1,
                   "expected_text_in_pool": True, "expected_text_pool_rank": 1}
    base_q = [{"index": 0, "expected_file": "a.md", "found_rank": 1, "text_match_at_1": True,
               "expected_full_rank": 1, "expected_text_full_rank": 1,
               "pool": common_pool, "full_order": base_full_order}]
    other_q = [{"index": 0, "expected_file": "a.md", "found_rank": 2, "text_match_at_1": False,
                "expected_full_rank": 2, "expected_text_full_rank": 2,
                "pool": common_pool, "full_order": other_full_order}]

    rows = dd.diff_records(base_q, other_q, params_match=True)
    assert len(rows) == 1
    assert rows[0]["pool_identity"] == "differs"
    assert rows[0]["class"] != "RERANKER"
    assert rows[0]["reason"] == "pool: differs"


def test_diff_records_treats_full_order_missing_content_sha256_as_unrecorded():
    """A record whose full_order predates Round 4 (no content_sha256 at
    all) must not be silently treated as identical just because doc_path/
    chunk_index still line up."""
    dd = _load("eval_details_diff_r4_rule1b", "eval/details_diff.py")
    legacy_entry = {"rank": 1, "doc_path": "a.md", "chunk_index": 0, "rerank_score": 0.5,
                    "expected": True, "text_match": False}          # no content_sha256 key
    common_pool = {"size": 1, "expected_in_pool": True, "expected_best_pool_rank": 1,
                   "expected_text_in_pool": False, "expected_text_pool_rank": None}
    base_q = [{"index": 0, "expected_file": "a.md", "found_rank": 1, "text_match_at_1": False,
               "expected_full_rank": 1, "expected_text_full_rank": None,
               "pool": common_pool, "full_order": [legacy_entry]}]
    other_q = [{"index": 0, "expected_file": "a.md", "found_rank": 2, "text_match_at_1": False,
                "expected_full_rank": 2, "expected_text_full_rank": None,
                "pool": common_pool, "full_order": [_full_order_entry("a.md", 0)]}]

    rows = dd.diff_records(base_q, other_q, params_match=True)
    assert rows[0]["pool_identity"] == "unrecorded"
    assert rows[0]["reason"] == "pool: unrecorded"


def test_classify_escalates_to_full_rank_when_found_rank_ties():
    """Round 4, Rule 2: found_rank tied at None (neither run's top-3 window
    contains the file) must not silently fall through to POOL_RANK/OTHER --
    the FULL rank moving from 4 to 20 is exactly the regression --details
    exists to surface."""
    dd = _load("eval_details_diff_r4_rule2", "eval/details_diff.py")
    common_pool = {"expected_in_pool": True, "expected_best_pool_rank": 1}
    base = {"found_rank": None, "expected_full_rank": 4, "pool": common_pool}
    other = {"found_rank": None, "expected_full_rank": 20, "pool": common_pool}

    assert dd.classify(base, other) == "RERANKER"
    assert dd.classify(other, base) == "RECOVERED"


# ---------- Round 5: unify identity/movement/query-equality into ONE
# function each, instead of duplicating the logic at each call site ----------

def test_pool_identity_sorts_by_pool_rank_not_list_order():
    """pool_identity() must key off pool_rank (the reranker's INPUT order),
    not the order full_order entries happen to be listed in the JSON."""
    dd = _load("eval_details_diff_r5_rule1_sort", "eval/details_diff.py")

    record_a = {"full_order": [_full_order_entry("x.md", 0, pool_rank=2),
                               _full_order_entry("y.md", 1, pool_rank=1)]}
    record_b = {"full_order": [_full_order_entry("y.md", 1, pool_rank=1),
                               _full_order_entry("x.md", 0, pool_rank=2)]}
    # Same entries, same pool_rank values, listed in a different order in
    # the array -> pool_identity must agree (it sorts).
    assert dd.pool_identity(record_a) == dd.pool_identity(record_b)

    record_c = {"full_order": [_full_order_entry("x.md", 0, pool_rank=1),
                               _full_order_entry("y.md", 1, pool_rank=2)]}
    # Same SET as record_a, but pool_rank values swapped -> a genuinely
    # different pool order, not just a different list order.
    assert dd.pool_identity(record_a) != dd.pool_identity(record_c)
    assert set(dd.pool_identity(record_a)) == set(dd.pool_identity(record_c))


def test_pool_identity_none_when_any_field_missing():
    dd = _load("eval_details_diff_r5_rule1_none", "eval/details_diff.py")
    assert dd.pool_identity({"full_order": None}) is None
    no_pool_rank = {"full_order": [{"doc_path": "a.md", "chunk_index": 0, "content_sha256": "h"}]}
    assert dd.pool_identity(no_pool_rank) is None
    no_content = {"full_order": [{"doc_path": "a.md", "chunk_index": 0, "pool_rank": 1}]}
    assert dd.pool_identity(no_content) is None


def test_diff_records_treats_pool_order_swap_as_order_differs():
    """Round 5, Rule 1: the reranker sorts stably, so equal scores inherit
    the PRE-rerank input order -- two runs with the same candidate set in a
    different pool order are not the same pool. Must get a factual label
    and a reason distinct from a genuine set difference."""
    dd = _load("eval_details_diff_r5_rule1_order", "eval/details_diff.py")
    base_full_order = [_full_order_entry("a.md", 0, pool_rank=1),
                       _full_order_entry("b.md", 1, rank=2, pool_rank=2)]
    other_full_order = [_full_order_entry("a.md", 0, pool_rank=2),      # same set...
                        _full_order_entry("b.md", 1, rank=2, pool_rank=1)]  # ...swapped pool order
    common_pool = {"size": 2, "expected_in_pool": True, "expected_best_pool_rank": 1,
                   "expected_text_in_pool": True, "expected_text_pool_rank": 1}
    base_q = [{"index": 0, "expected_file": "a.md", "found_rank": 1, "text_match_at_1": True,
               "expected_full_rank": 1, "expected_text_full_rank": 1,
               "pool": common_pool, "full_order": base_full_order}]
    other_q = [{"index": 0, "expected_file": "a.md", "found_rank": 2, "text_match_at_1": False,
                "expected_full_rank": 2, "expected_text_full_rank": 2,
                "pool": common_pool, "full_order": other_full_order}]

    rows = dd.diff_records(base_q, other_q, params_match=True)
    assert len(rows) == 1
    assert rows[0]["pool_identity"] == "order_differs"
    assert rows[0]["reason"] == "pool: order differs"
    assert rows[0]["class"] != "RERANKER"          # factual label only


def test_rank_movement_reports_window_full_and_text_full_deltas():
    dd = _load("eval_details_diff_r5_rule2_movement", "eval/details_diff.py")
    worse = {"found_rank": 1, "expected_full_rank": 1, "expected_text_full_rank": 1}
    better_side = {"found_rank": 2, "expected_full_rank": 2, "expected_text_full_rank": 2}
    assert dd.rank_movement(worse, better_side) == (1, 1, 1)     # other is worse on all three
    assert dd.rank_movement(better_side, worse) == (-1, -1, -1)  # other is better on all three
    assert dd.rank_movement(worse, dict(worse)) == (0, 0, 0)


def test_classify_cross_falls_back_to_full_rank_on_window_tie():
    """Round 5, Rule 2: classify_cross must use the SAME cascading
    window -> full rank fallback as classify() -- previously only the
    attributing path escalated to expected_full_rank on a found_rank tie."""
    dd = _load("eval_details_diff_r5_rule2_cross", "eval/details_diff.py")
    common_pool = {"expected_in_pool": True}
    base = {"found_rank": None, "expected_full_rank": 4, "pool": common_pool}
    other = {"found_rank": None, "expected_full_rank": 20, "pool": common_pool}

    assert dd.classify_cross(base, other) == "RANK_WORSE"
    assert dd.classify_cross(other, base) == "RANK_BETTER"


def test_same_query_names_the_first_differing_field():
    dd = _load("eval_details_diff_r5_rule3_field", "eval/details_diff.py")
    base = {"query": "q0", "expected_file": "a.md", "expected_text_contains": "x"}
    other = {"query": "q0", "expected_file": "b.md", "expected_text_contains": "x"}

    same, field = dd.same_query(base, other)
    assert same is False
    assert field == "expected_file"

    same_true, field_none = dd.same_query(base, dict(base))
    assert same_true is True and field_none is None


def test_compare_raises_naming_field_when_query_matches_but_expected_file_differs():
    """Round 5, Rule 3: the hashless fallback used to compare only `query`
    text -- two records that ask the same question but score against a
    different expected_file must also be refused, with the differing field
    named."""
    dd = _load("eval_details_diff_r5_rule3_compare", "eval/details_diff.py")
    base_details = {"cell": {"models": {"embedder": "e"}, "mode": "live"}, "queries": [
        {"index": 0, "query": "q0", "expected_file": "a.md", "expected_text_contains": "x"}]}
    other_details = {"cell": {"models": {"embedder": "e"}, "mode": "live"}, "queries": [
        {"index": 0, "query": "q0", "expected_file": "b.md", "expected_text_contains": "x"}]}
    with pytest.raises(ValueError) as exc_info:
        dd.compare(base_details, other_details)
    msg = str(exc_info.value)
    assert "expected_file" in msg
    assert "0" in msg


# ---------- 4. --details writer round-trips ----------

def test_write_details_file_round_trips(tmp_path):
    ev = _load("eval_evaluate_details_4", "eval/evaluate.py")

    records = [
        {"index": 0, "query": "q0", "expected_file": "a.md", "expected_text_contains": "",
         "pool": {"size": 20, "expected_in_pool": True, "expected_best_pool_rank": 1},
         "found_rank": 1, "text_match_at_1": False, "top": []},
        {"index": 1, "query": "q1", "expected_file": "b.md", "expected_text_contains": "x",
         "pool": None, "found_rank": None, "text_match_at_1": False, "top": []},
    ]
    results = {"params": {"chunk_size": 256}, "models": {"embedder": "e", "reranker": "r"},
               "mode": "live", "score": 0.5, "timing": {}, "hit_at_1": 0.5}

    out = tmp_path / "details.json"
    ev.write_details_file(out, results, records)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["cell"] == {"params": results["params"], "models": results["models"],
                               "mode": results["mode"]}
    assert [r["index"] for r in loaded["queries"]] == [0, 1]
    assert loaded["queries"] == records


# ---------- 5. Gap 4: provenance in `cell` ----------

def test_build_cell_provenance_pure_fields(tmp_path):
    """The deterministic fields: derived only from `config`, the .info dicts
    the harness already builds, and the caller-supplied paths -- no new
    Embedder/Reranker attributes."""
    ev = _load("eval_evaluate_details_gap4a", "eval/evaluate.py")
    import hashlib

    query_set = tmp_path / "test_queries.json"
    query_set.write_text('[{"query": "q"}]', encoding="utf-8")
    expected_hash = hashlib.sha256(query_set.read_bytes()).hexdigest()

    config = {"embedder": {"quantize": False}}
    embedder_info = {"model": "harrier", "dimension": 4096, "device": "cpu",
                     "encoding": {"revision": "abc123"}}
    reranker_info = {"model": "qwen3-reranker-4b", "revision": "rev1", "precision": "nf4-bf16"}
    results = {"mode": "live"}
    repo_root = ev.EVAL_DIR.parent

    prov = ev.build_cell_provenance(config, results, embedder_info, reranker_info,
                                    query_set, "manifest.json", repo_root, "eval/evaluate.py")

    assert prov["harness"] == "eval/evaluate.py"
    assert prov["reranker_revision"] == "rev1"
    assert prov["reranker_quantize"] is True                 # precision == "nf4-bf16"
    assert prov["embedder_encoding"] == {"revision": "abc123"}
    assert prov["embedder_quantize"] is False                # from config, not a model attribute
    assert prov["query_set_sha256"] == expected_hash
    assert prov["corpus_manifest"] == "manifest.json"
    # Git and package lookups are real but environment-dependent -- presence
    # only, per the brief ("do not test git or package lookups beyond
    # 'returns a dict with these keys'").
    assert "commit" in prov
    assert "packages" in prov and isinstance(prov["packages"], dict)


def test_build_cell_provenance_defaults_when_nothing_is_available():
    ev = _load("eval_evaluate_details_gap4b", "eval/evaluate.py")

    prov = ev.build_cell_provenance({"embedder": {}}, {}, {}, {}, None, None,
                                    ev.EVAL_DIR.parent, "eval/evaluate.py")

    assert prov["embedder_encoding"] == "legacy"       # no "encoding" key -> legacy contract
    assert prov["reranker_revision"] is None
    assert prov["reranker_quantize"] is False           # no "precision" key -> not quantized
    assert prov["embedder_quantize"] is True            # config default (quantize: True)
    assert prov["query_set_sha256"] is None             # no query set path given
    assert prov["corpus_manifest"] is None


def test_write_details_file_merges_provenance_into_cell(tmp_path):
    ev = _load("eval_evaluate_details_gap4c", "eval/evaluate.py")

    results = {"params": {"chunk_size": 256}, "models": {"embedder": "e"}, "mode": "live"}
    provenance = {"commit": "deadbeef", "harness": "eval/evaluate.py", "packages": {"torch": "2.0"}}
    out = tmp_path / "details.json"

    ev.write_details_file(out, results, [], provenance=provenance)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["cell"]["commit"] == "deadbeef"
    assert loaded["cell"]["harness"] == "eval/evaluate.py"
    assert loaded["cell"]["packages"] == {"torch": "2.0"}
    assert loaded["cell"]["mode"] == "live"             # existing fields survive the merge


# ---------- bonus: --daemon mode routes through the same judgement, pool always null ----------

def test_evaluate_via_daemon_populates_details_with_null_pool():
    ev = _load("eval_evaluate_details_5", "eval/evaluate.py")

    class C:
        def post(self, path, json=None):
            return {"results": [{"file": "c:/r/p/docs/a.md", "text": "the answer is here",
                                  "rerank_score": 0.9}], "confidence": "high"}

    queries = [{"query": "alpha", "expected_file": "a.md", "expected_text_contains": "the answer"}]
    ev.DETAILS_DUMP = []
    m = ev.evaluate_via_daemon(C(), queries)
    assert m["hit_at_1"] == 1.0
    assert len(ev.DETAILS_DUMP) == 1
    record = ev.DETAILS_DUMP[0]
    assert record["pool"] is None
    assert record["found_rank"] == 1
    assert record["text_match_at_1"] is True
    assert record["top"][0]["doc_path"] == "c:/r/p/docs/a.md"
    assert record["full_order"] is None                 # Gap 1: daemon has no full pool to show
    assert record["expected_full_rank"] is None
    assert record["expected_text_full_rank"] is None
    ev.DETAILS_DUMP = None
