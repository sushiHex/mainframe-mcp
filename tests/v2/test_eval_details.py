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


def test_compare_same_embedder_uses_attribution_labels():
    """Gap 3: same cell.models.embedder string on both sides -> the existing
    EMBEDDER/RERANKER/RECOVERED/POOL_RANK label set, no warning."""
    dd = _load("eval_details_diff_gap3_same", "eval/details_diff.py")

    base_details = {"cell": {"models": {"embedder": "qwen3-8b"}, "mode": "live"}, "queries": [
        {"index": 0, "expected_file": "a.md", "found_rank": 1, "text_match_at_1": True,
         "pool": {"size": 20, "expected_in_pool": True, "expected_best_pool_rank": 1,
                  "expected_text_in_pool": True, "expected_text_pool_rank": 1}},
    ]}
    other_details = {"cell": {"models": {"embedder": "qwen3-8b"}, "mode": "live"}, "queries": [
        {"index": 0, "expected_file": "a.md", "found_rank": 3, "text_match_at_1": False,
         "pool": {"size": 20, "expected_in_pool": True, "expected_best_pool_rank": 1,
                  "expected_text_in_pool": True, "expected_text_pool_rank": 1}},
    ]}

    result = dd.compare(base_details, other_details)
    assert result["same_embedder"] is True
    assert result["rows"][0]["class"] == "RERANKER"
    assert result["warnings"] == []


def test_compare_different_embedder_uses_factual_labels_and_warns():
    """Gap 3: different cell.models.embedder strings -> factual-only labels
    (POOL_LOST/RANK_WORSE, not EMBEDDER/RERANKER) plus a header warning that
    rank changes are not attributable to the reranker."""
    dd = _load("eval_details_diff_gap3_cross", "eval/details_diff.py")

    base_details = {"cell": {"models": {"embedder": "harrier-v1"}, "mode": "live"}, "queries": [
        {"index": 0, "expected_file": "a.md", "found_rank": 1, "text_match_at_1": True,
         "pool": {"size": 20, "expected_in_pool": True, "expected_best_pool_rank": 1,
                  "expected_text_in_pool": True, "expected_text_pool_rank": 1}},
        {"index": 1, "expected_file": "b.md", "found_rank": 1, "text_match_at_1": True,
         "pool": {"size": 20, "expected_in_pool": True, "expected_best_pool_rank": 2,
                  "expected_text_in_pool": True, "expected_text_pool_rank": 2}},
    ]}
    other_details = {"cell": {"models": {"embedder": "harrier-v2"}, "mode": "live"}, "queries": [
        {"index": 0, "expected_file": "a.md", "found_rank": None, "text_match_at_1": False,
         "pool": {"size": 20, "expected_in_pool": False, "expected_best_pool_rank": None,
                  "expected_text_in_pool": False, "expected_text_pool_rank": None}},
        {"index": 1, "expected_file": "b.md", "found_rank": 3, "text_match_at_1": False,
         "pool": {"size": 20, "expected_in_pool": True, "expected_best_pool_rank": 2,
                  "expected_text_in_pool": True, "expected_text_pool_rank": 2}},
    ]}

    result = dd.compare(base_details, other_details)
    assert result["same_embedder"] is False
    assert result["rows"][0]["class"] == "POOL_LOST"       # not EMBEDDER: pools were not the same
    assert result["rows"][1]["class"] == "RANK_WORSE"       # not RERANKER
    assert len(result["warnings"]) == 1
    assert "embedder" in result["warnings"][0].lower()
    assert "reranker" in result["warnings"][0].lower()


def test_compare_same_embedder_warns_if_embedder_label_still_occurs():
    """EMBEDDER 'cannot occur' when both runs report the same embedder -- if
    pool membership still flipped, that assumption didn't hold (different
    corpus/index/config even with the same model), and compare() must say
    so instead of silently trusting the label."""
    dd = _load("eval_details_diff_gap3_warn", "eval/details_diff.py")

    base_details = {"cell": {"models": {"embedder": "harrier-v1"}, "mode": "live"}, "queries": [
        {"index": 0, "expected_file": "a.md", "found_rank": 1, "text_match_at_1": True,
         "pool": {"size": 20, "expected_in_pool": True, "expected_best_pool_rank": 1,
                  "expected_text_in_pool": True, "expected_text_pool_rank": 1}},
    ]}
    other_details = {"cell": {"models": {"embedder": "harrier-v1"}, "mode": "rebuild"}, "queries": [
        {"index": 0, "expected_file": "a.md", "found_rank": None, "text_match_at_1": False,
         "pool": {"size": 20, "expected_in_pool": False, "expected_best_pool_rank": None,
                  "expected_text_in_pool": False, "expected_text_pool_rank": None}},
    ]}

    result = dd.compare(base_details, other_details)
    assert result["same_embedder"] is True
    assert result["rows"][0]["class"] == "EMBEDDER"
    assert len(result["warnings"]) == 1
    assert "same embedder" in result["warnings"][0].lower()


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
