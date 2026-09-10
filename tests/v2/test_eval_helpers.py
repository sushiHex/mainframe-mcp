import importlib.util
import json
import sys
from pathlib import Path

from mainframe.core.indexer import LaneFile
from mainframe.core.paths import canonical
from v2.helpers import write_md

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_rebuild_index_is_deterministic(cfg, store, fake_embedder, tmp_path):
    ev = _load("eval_evaluate", "eval/evaluate.py")
    files = [LaneFile(canonical(write_md(tmp_path / "repos" / "p" / "docs" / f"{n}.md", f"# {n}\n\ntext {n} qwen.\n")), "knowledge", "p")
             for n in ("b", "a", "c")]
    n1 = ev.rebuild_index(store, fake_embedder, files, cfg["chunker"], batch_size=2)
    v_after_first = len(store.table.list_versions())
    n2 = ev.rebuild_index(store, fake_embedder, files, cfg["chunker"], batch_size=2)
    assert n1 == n2 == 3
    assert len(store.table.list_versions()) > v_after_first  # a rebuild writes again
    assert sorted(store.ledger()) == sorted(f.path for f in files)


def test_rebuild_index_honors_the_chunk_cfg_it_is_given(cfg, store, fake_embedder, tmp_path):
    """rebuild_index must chunk with whatever chunk_cfg the caller passes it —
    not a module-level default the caller silently ignored."""
    ev = _load("eval_evaluate", "eval/evaluate.py")
    paras = [f"Paragraph {i} qwen reranker hybrid search words here more padding padding." for i in range(20)]
    body = "# T\n\n" + "\n\n".join(paras) + "\n"
    p = write_md(tmp_path / "repos" / "p" / "docs" / "big.md", body)
    files = [LaneFile(canonical(p), "knowledge", "p")]
    rows_default = ev.rebuild_index(store, fake_embedder, files, {"chunk_size": 256, "overlap_ratio": 0.35})
    store.drop()
    rows_small = ev.rebuild_index(store, fake_embedder, files, {"chunk_size": 20, "overlap_ratio": 0.35})
    assert rows_small > rows_default


def test_corpus_identity_compares_by_path_and_hash(cfg, tmp_path):
    ci = _load("eval_corpus_identity", "eval/corpus_identity.py")
    a = write_md(tmp_path / "repos" / "p" / "docs" / "a.md", "# a\n")
    manifest = tmp_path / ".manifest.json"
    from mainframe.core.hashing import file_hash
    manifest.write_text(json.dumps({"files": {
        str(a): {"content_hash": file_hash("# a\n")},
        str(tmp_path / "repos" / "p" / "research" / "sessions" / "s.md"): {"content_hash": "x"},  # ignored
    }}), encoding="utf-8")
    assert ci.compare(cfg, manifest) == ([], [])          # no missing, no extra
    write_md(tmp_path / "repos" / "p" / "docs" / "b.md", "# b\n")
    missing, extra = ci.compare(cfg, manifest)
    assert missing == [] and [Path(e).name for e in extra] == ["b.md"]


def test_corpus_from_manifest_keeps_existing_non_session_files(cfg, tmp_path):
    ev = _load("eval_evaluate", "eval/evaluate.py")
    a = write_md(tmp_path / "repos" / "p" / "docs" / "a.md", "# a\n")
    gone = tmp_path / "repos" / "p" / "docs" / "gone.md"          # in manifest, not on disk
    sess = write_md(tmp_path / "repos" / "p" / "research" / "sessions" / "s.md", "# s\n")
    manifest = tmp_path / ".manifest.json"
    manifest.write_text(json.dumps({"files": {str(a): {}, str(gone): {}, str(sess): {}}}), encoding="utf-8")
    files = ev.corpus_from_manifest(cfg, manifest)
    assert [Path(f.path).name for f in files] == ["a.md"]
    assert files[0].lane == "knowledge" and files[0].project == "p"


def test_gate_passes_only_when_missing_files_are_gone(tmp_path):
    ci = _load("eval_corpus_identity", "eval/corpus_identity.py")
    present = write_md(tmp_path / "present.md", "# x\n")
    assert ci.gate_passes([]) is True
    assert ci.gate_passes([str(tmp_path / "gone.md")]) is True
    assert ci.gate_passes([str(present)]) is False


def test_matches_expected_is_case_and_separator_insensitive():
    ev = _load("eval_evaluate", "eval/evaluate.py")
    assert ev.matches_expected("c:/fixtures/repos/example-project/claude.md", "example-project\\CLAUDE.md")
    assert ev.matches_expected("c:/fixtures/repos/example-project/docs/python-conventions.md", "python-conventions.md")
    assert not ev.matches_expected("c:/fixtures/repos/example-project/claude.md", "otherproj\\CLAUDE.md")


def test_open_existing_index_refuses_missing_and_empty(cfg, tmp_path, store, fake_embedder):
    import pytest
    ev = _load("eval_evaluate", "eval/evaluate.py")
    with pytest.raises(SystemExit):
        ev.open_existing_index(tmp_path / "nowhere", 64, cfg)          # missing
    with pytest.raises(SystemExit):
        ev.open_existing_index(store.db_path, 64, cfg)                  # exists but empty
    files = [LaneFile(canonical(write_md(tmp_path / "repos" / "p" / "docs" / "a.md", "# a\n\ntext a qwen.\n")), "knowledge", "p")]
    ev.rebuild_index(store, fake_embedder, files, cfg["chunker"])
    opened = ev.open_existing_index(store.db_path, 64, cfg)             # populated -> OK
    assert opened.stats()["total_chunks"] == 1


def test_evaluate_via_daemon_scores_from_search_results():
    ev = _load("eval_evaluate", "eval/evaluate.py")

    class C:
        def post(self, path, json=None):
            q = json["query"]
            return {"results": [{"file": "c:/r/p/docs/" + ("a.md" if "alpha" in q else "z.md"),
                                 "text": "alpha body text", "rerank_score": 0.9}], "confidence": "high"}
    queries = [{"query": "alpha one", "expected_file": "docs\\A.md", "expected_text_contains": "alpha body"},
               {"query": "beta two", "expected_file": "b.md"}]
    m = ev.evaluate_via_daemon(C(), queries)
    assert m["hit_at_1"] == 0.5 and m["text_match_at_1"] == 0.5 and m["total_queries"] == 2 and 0 < m["score"] < 1


def test_evaluate_via_daemon_requests_full_format_and_matches_past_char_500():
    """SearchService's "detailed" truncates `text` at 500 chars; evaluate() (the
    in-process path) matches expected_text_contains against the FULL chunk
    text, so --daemon must request response_format="full" or it silently
    under-counts text_match_at_1 whenever the phrase sits past char 500."""
    ev = _load("eval_evaluate", "eval/evaluate.py")
    long_text = ("padding " * 90) + "the target phrase"  # > 500 chars before the phrase
    assert len(long_text) > 600 and long_text.index("the target phrase") > 500

    class C:
        requested_formats = []

        def post(self, path, json=None):
            self.requested_formats.append(json["response_format"])
            return {"results": [{"file": "c:/r/p/docs/a.md", "text": long_text, "rerank_score": 0.9}],
                    "confidence": "high"}

    client = C()
    queries = [{"query": "alpha", "expected_file": "a.md", "expected_text_contains": "the target phrase"}]
    m = ev.evaluate_via_daemon(client, queries)
    assert m["text_match_at_1"] == 1.0
    assert client.requested_formats == ["full"]


def test_evaluate_via_daemon_matches_the_in_process_rank_window():
    """the daemon harness asked for max(limit, 5) results while in-process
    evaluate() reranks top-3 — ranks 4 and 5 contributed MRR the in-process
    baseline can never earn, so a --daemon score was systematically >= the
    in-process one it is compared against by --min-score."""
    ev = _load("eval_evaluate", "eval/evaluate.py")

    class C:
        def __init__(self):
            self.limits = []

        def post(self, path, json=None):
            self.limits.append(json["limit"])
            return {"results": [{"file": "c:/r/p/docs/a.md", "text": "alpha body", "rerank_score": 0.9}],
                    "confidence": "high"}

    client = C()
    m = ev.evaluate_via_daemon(client, [{"query": "alpha", "expected_file": "a.md"}])
    assert client.limits == [3] and ev.RERANK_TOP_K == 3
    assert m["hit_at_5"] is None            # a rank window the in-process harness cannot produce
    assert m["hit_at_1"] == 1.0 and m["hit_at_3"] == 1.0


def _args(**kw):
    from types import SimpleNamespace
    base = dict(live=False, rebuild=False, daemon=False, db=None, in_process=False, dump_candidates=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_choose_mode_truth_table():
    """The NO-FLAG default — what CLAUDE.md documents — used to build an
    Embedder and a Reranker in a second process while mainframed held ~15 GB."""
    ev = _load("eval_evaluate", "eval/evaluate.py")
    cm = ev.choose_mode
    assert cm(_args(), True) == "daemon" and cm(_args(), False) == "in-process"
    assert cm(_args(live=True), True) == "daemon" and cm(_args(live=True), False) == "in-process"
    assert cm(_args(live=True, in_process=True), True) == "in-process"     # the escape hatch
    assert cm(_args(daemon=True), True) == "daemon" and cm(_args(daemon=True), False) == "daemon"
    # a second model set cannot share the card with the daemon
    assert cm(_args(rebuild=True), True) == "refuse" and cm(_args(rebuild=True), False) == "in-process"
    assert cm(_args(db="x"), True) == "refuse" and cm(_args(db="x"), False) == "in-process"
    # candidate dumps only exist on the in-process path
    assert cm(_args(dump_candidates="d.json"), True) == "in-process"


def test_evaluate_via_daemon_exits_cleanly_when_daemon_becomes_unreachable():
    import pytest
    ev = _load("eval_evaluate", "eval/evaluate.py")

    class C:
        def post(self, path, json=None):
            raise ConnectionError("connection refused")

    queries = [{"query": "alpha", "expected_file": "a.md"}]
    with pytest.raises(SystemExit) as exc_info:
        ev.evaluate_via_daemon(C(), queries)
    assert "unreachable" in str(exc_info.value)


def test_live_evaluation_uses_the_configured_state_directory(cfg, tmp_path, store,
                                                          fake_embedder, fake_reranker, monkeypatch):
    """A scoped evaluation must never open a different deployment's index."""
    from mainframe.adapters import cli
    from mainframe.core.store import INDEX_DIRNAME

    ev = _load("eval_evaluate", "eval/evaluate.py")
    configured = Path(cfg["paths"]["mainframe_dir"]) / INDEX_DIRNAME
    default_dir = tmp_path / "other-deployment"
    (default_dir / INDEX_DIRNAME).mkdir(parents=True)
    queries = tmp_path / "queries.json"
    queries.write_text("[]", encoding="utf-8")
    # Trap the legacy module-level path even after its removal.
    monkeypatch.setattr(ev, "MAINFRAME_DIR", default_dir, raising=False)
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(ev, "TEST_QUERIES_PATH", queries)
    monkeypatch.setattr(ev, "load_config", lambda: cfg)
    monkeypatch.setattr(ev, "hf_offline_if_cached", lambda config: False)
    monkeypatch.setattr(ev, "Embedder", lambda config: fake_embedder)
    monkeypatch.setattr(ev, "Reranker", lambda config: fake_reranker)
    monkeypatch.setattr(cli.DaemonClient, "alive", lambda self: False)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--live"])
    opened = []

    def open_index(path, dimension, config):
        assert path == configured
        opened.append(path)
        return store

    monkeypatch.setattr(ev, "open_existing_index", open_index)
    ev.main()
    assert opened == [configured]
