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

    judged = ev.judge_query(raw_results, reranked, "target.md", "the answer")
    assert judged == {"expected_in_pool": True, "expected_best_pool_rank": 7,
                       "found_rank": 2, "text_match_at_1": False}

    # Expected file entirely absent from the pool.
    judged_absent = ev.judge_query(raw_results[:6], [(0, 0.9)], "target.md", "the answer")
    assert judged_absent == {"expected_in_pool": False, "expected_best_pool_rank": None,
                              "found_rank": None, "text_match_at_1": False}

    # Reranked at rank 1 AND the expected text is present -> text_match_at_1 True.
    raw_rank1 = [{"doc_path": "c:/r/p/docs/target.md", "text": "the answer is here"},
                 {"doc_path": "c:/r/p/docs/other.md", "text": "irrelevant"}]
    judged_rank1 = ev.judge_query(raw_rank1, [(0, 0.9), (1, 0.5)], "target.md", "the answer")
    assert judged_rank1 == {"expected_in_pool": True, "expected_best_pool_rank": 1,
                             "found_rank": 1, "text_match_at_1": True}


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
    ev.DETAILS_DUMP = None
