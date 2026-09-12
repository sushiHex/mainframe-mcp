import sys
import threading
from pathlib import Path

import pytest

from mainframe.app import JOBS, Mainframe
from mainframe.core.paths import canonical
from mainframe.core.pipeline import IndexRunResult
from mainframe.service.events import EventStore
from mainframe.service.mutations import MutationQueue, QueueStopped
from mainframe.service.watcher import Invalidation
from v2.fakes import fake_models
from v2.helpers import write_md


def _app(cfg, store, tmp_path):
    return Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)


def _attach(app):
    app.mutations = MutationQueue(app.events)
    app.invalidation = Invalidation()
    app.mutations.start()
    return app


def test_search_status_offline(cfg, store, tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    app = _app(cfg, store, tmp_path)
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    assert app.submit("rescan")["upserted_rows"] == 1
    out = app.search("qwen3 reranker")
    assert out["confidence"] == "unavailable" and out["results"][0]["file"].endswith("a.md")
    st = app.status()
    assert st["index"]["rows"] == 1 and st["index"]["docs"] == 1 and st["index"]["last_run"]["kind"] == "rescan"
    assert st["models"]["embedder"]["state"] == "loaded" and st["vram"] is None and "torch" not in sys.modules
    assert st["memory"]["mode"] == "index-and-search only" and st["memory"]["pending"] == {}
    assert st["daemon"]["queue_depth"] is None and st["recent_events"][0]["kind"] == "index.rescan"
    with pytest.raises(ValueError):
        app.submit("nope")


def test_capture_writes_a_pending_file_and_the_ordinary_scan_indexes_it(cfg, store, tmp_path):
    """`capture` submits nothing: the capture root is a watch root, so the file
    reaches the index the same way every other file does. The write is the
    durable part; the scan (watcher debounce in production, explicit here) is
    what indexes it."""
    app = _attach(_app(cfg, store, tmp_path))
    res = app.capture("## Summary\nqwen3 decided.\n", "decision", project="p", session_id="abcd1234")
    assert res["project"] == "p" and Path(res["file"]).exists()
    assert app.store.count_rows() == 0 and app.status()["memory"]["pending"] == {"p": 1}
    app.submit("index", [res["file"]])
    assert app.search("qwen3", include_sessions=True)["results"][0]["file"] == canonical(res["file"])
    app.mutations.stop()


def test_submit_inline_on_worker_and_requeue_on_failure(cfg, store, tmp_path, monkeypatch):
    app = _attach(_app(cfg, store, tmp_path))
    seen = {}

    def nested():
        seen["on_worker"] = app.mutations.is_worker_thread()
        return app.submit("optimize")          # inline: must NOT deadlock
    assert app.mutations.run("outer", nested, timeout=5) == {"ok": True} and seen["on_worker"]
    monkeypatch.setattr(app.pipeline, "index", lambda paths: (_ for _ in ()).throw(RuntimeError("cuda")))
    paths = ["C:/x.md", "C:/y.md"]
    with pytest.raises(RuntimeError):
        app.submit("index", paths, on_retryable_failure=lambda: [app.invalidation.touch(p) for p in paths])
    assert sorted(app.invalidation.take_paths()) == paths      # given back for the next tick
    fut = app.submit("optimize", wait=False)
    assert fut.result(timeout=5) == {"ok": True} and JOBS == ("index", "rescan", "optimize", "reload")
    app.mutations.stop()


def test_reload_is_the_deliberate_escape_hatch_for_a_stuck_model(cfg, store, tmp_path):
    """`Models.invoke` heals a device failure it can RECOGNIZE. Nothing could
    force a reload when one escaped that predicate, so the only recovery was a
    daemon restart — the deleted clock-driven `resume` had covered that by
    luck. `maintain("reload")` makes the hatch deliberate."""
    app = _app(cfg, store, tmp_path)
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    app.submit("rescan")
    app.search("qwen3 reranker")
    assert app.models.loaded("embedder") and app.models.loaded("reranker")

    assert app.maintain("reload") == {"reloaded": ["embedder", "reranker"]}
    assert not app.models.loaded("embedder") and not app.models.loaded("reranker")
    assert app.maintain("reload") == {"reloaded": []}          # idempotent
    assert app.search("qwen3 reranker")["results"]             # and they come back on next use


def test_reload_clears_a_role_stuck_in_its_load_backoff(cfg, store, tmp_path):
    """The state the hatch most exists for. A GPU fault AT LOAD TIME leaves the
    role `failed` for fifteen minutes and NOT `loaded` — so a resident-only
    filter skipped exactly the case an operator is trying to clear, and the
    backoff still blocked every retry."""
    from mainframe.core.models import Models

    attempts = []

    def boom(_c):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("CUDA error: no CUDA-capable device is detected")
        return "healthy"

    models = Models(cfg, embedder_factory=boom, reranker_factory=lambda c: None)
    app = Mainframe(cfg, models=models, events=EventStore(tmp_path / "ev.db"), store=store)

    with pytest.raises(RuntimeError, match="CUDA"):
        models.invoke("embedder", lambda e: e)
    assert models.state()["embedder"]["state"] == "failed" and not models.loaded("embedder")
    with pytest.raises(RuntimeError, match="unavailable"):
        models.invoke("embedder", lambda e: e)      # backed off: the factory is not called again

    assert app.maintain("reload") == {"reloaded": ["embedder"]}
    assert models.state()["embedder"]["state"] == "absent"
    assert models.invoke("embedder", lambda e: e) == "healthy"   # backoff cleared, not waited out
    assert len(attempts) == 2


def test_maintain_is_the_one_maintenance_entry_point(cfg, store, tmp_path):
    """the action list and the index->rescan alias used to be written
    three times (app.JOBS, mcp_server.ACTIONS, api.py), and `project` reached
    a filesystem join and a delete predicate unvalidated."""
    app = _app(cfg, store, tmp_path)
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    assert app.maintain("index")["upserted_rows"] == 1        # `index` means "reconcile"
    assert app.maintain("rescan", "p")["files"] == 1
    assert app.maintain("optimize") == {"ok": True}
    for bad in ("../x", "C:/evil", "a/b", ".hidden", ""):
        with pytest.raises(ValueError):
            app.maintain("rescan", bad)
    with pytest.raises(ValueError):
        app.maintain("explode")


def test_search_heals_a_device_failure(cfg, store, tmp_path):
    """The rebuild-on-identity-change dance is gone: `SearchService` holds the
    model REGISTRY, so a model that dies and reloads underneath a search is
    invisible to the caller and needs nothing rebuilt."""
    from mainframe.core.models import Models
    from v2.conftest import FakeEmbedder, FakeReranker

    loads, fails = [], {"n": 1}

    class FlakyEmbedder(FakeEmbedder):
        def embed_query(self, query):
            if fails["n"]:
                fails["n"] -= 1
                raise RuntimeError("CUDA error: unknown error")
            return super().embed_query(query)

    models = Models(cfg, embedder_factory=lambda c: loads.append(1) or FlakyEmbedder(dim=64),
                    reranker_factory=lambda c: FakeReranker(top_k=3))
    app = Mainframe(cfg, models=models, events=EventStore(tmp_path / "ev.db"), store=store)
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    app.submit("rescan")
    searcher = app.searcher
    assert app.search("qwen3 reranker")["results"][0]["file"].endswith("a.md")
    assert len(loads) == 2 and app.searcher is searcher


def test_submit_and_capture_when_queue_is_stopped(cfg, store, tmp_path):
    app = _attach(_app(cfg, store, tmp_path))
    app.mutations.stop()
    outcome = {}

    def _try():
        try:
            app.submit("optimize")
        except QueueStopped as e:
            outcome["error"] = e
    t = threading.Thread(target=_try)
    t.start()
    t.join(2)
    assert not t.is_alive() and isinstance(outcome.get("error"), QueueStopped)   # returned promptly, didn't hang
    # A capture never touches the queue at all, so a stopped one cannot fail it.
    res = app.capture("body text", "t", project="p")
    assert Path(res["file"]).exists()


def test_status_docs_avoids_ledger_scan(cfg, store, tmp_path, monkeypatch):
    app = _app(cfg, store, tmp_path)
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    app.submit("rescan")
    assert app.pipeline.doc_count == 1
    assert app.status()["index"]["docs"] == 1

    def _boom():
        raise AssertionError("status() must not scan the ledger when pipeline.doc_count is already known")
    monkeypatch.setattr(app.store, "ledger", _boom)
    assert app.status()["index"]["docs"] == 1


def test_submit_enqueues_exactly_once(cfg, store, tmp_path, monkeypatch):
    app = _attach(_app(cfg, store, tmp_path))
    calls = []
    orig_submit = app.mutations.submit

    def spy(name, fn, *a, **kw):
        calls.append(name)
        return orig_submit(name, fn, *a, **kw)
    monkeypatch.setattr(app.mutations, "submit", spy)
    assert app.submit("optimize") == {"ok": True}
    assert calls == ["optimize"]
    fut = app.submit("optimize", wait=False)
    assert fut.result(timeout=5) == {"ok": True}
    assert calls == ["optimize", "optimize"]
    app.mutations.stop()


def test_on_retryable_failure_runs_on_both_paths_and_only_when_transient(cfg, store, tmp_path, monkeypatch):
    """ONE mechanism, and `submit` no longer knows what the caller consumed.
    It used to key on `kind == "index"` and reach into `self.invalidation`,
    which is exactly why `rescan` was missed when it started consuming the
    reasons too."""
    from mainframe.core.store import IndexStaleError

    app = _attach(_app(cfg, store, tmp_path))
    undone = []

    def _boom(exc):
        monkeypatch.setattr(app.pipeline, "index", lambda paths: (_ for _ in ()).throw(exc))

    _boom(RuntimeError("cuda"))
    fut = app.submit("index", ["C:/x.md"], wait=False, on_retryable_failure=lambda: undone.append("async"))
    assert fut.exception(timeout=5) is not None
    assert undone == ["async"]

    with pytest.raises(RuntimeError):
        app.submit("index", ["C:/x.md"], on_retryable_failure=lambda: undone.append("wait"))
    assert undone == ["async", "wait"]

    # A permanent failure undoes nothing: retrying cannot clear it.
    _boom(IndexStaleError("chunk_size drifted"))
    with pytest.raises(IndexStaleError):
        app.submit("index", ["C:/x.md"], on_retryable_failure=lambda: undone.append("stale"))
    fut = app.submit("index", ["C:/x.md"], wait=False, on_retryable_failure=lambda: undone.append("stale"))
    assert fut.exception(timeout=5) is not None
    assert undone == ["async", "wait"]
    app.mutations.stop()


@pytest.mark.parametrize("mode", ["offline", "worker", "wait", "async"])
def test_a_job_timeout_restores_work_in_every_execution_mode(cfg, store, tmp_path, monkeypatch, mode):
    app = _app(cfg, store, tmp_path)
    if mode != "offline":
        _attach(app)
    undone = []

    def job(_paths):
        raise TimeoutError("storage timed out")

    monkeypatch.setattr(app.pipeline, "index", job)

    def submit():
        return app.submit("index", (), wait=mode != "async", on_retryable_failure=lambda: undone.append(1))

    try:
        with pytest.raises(TimeoutError, match="storage timed out"):
            if mode == "worker":
                app.mutations.run("outer", submit, timeout=5)
            elif mode == "async":
                submit().result(timeout=5)
            else:
                submit()
        assert undone == [1]
    finally:
        if app.mutations is not None:
            assert app.mutations.stop(timeout=5)


def test_a_wait_timeout_does_not_abandon_the_jobs_later_recovery(cfg, store, tmp_path, monkeypatch):
    app = _attach(_app(cfg, store, tmp_path))
    release = threading.Event()
    undone = []

    def job(_paths):
        assert release.wait(5)
        raise RuntimeError("later storage failure")

    monkeypatch.setattr(app.pipeline, "index", job)
    try:
        with pytest.raises(TimeoutError):
            app.submit("index", (), timeout=0, on_retryable_failure=lambda: undone.append(1))
        assert undone == [], "a caller's wait timeout is not a job failure"
        release.set()
        app.mutations.run("barrier", lambda: None, timeout=5)
        assert undone == [1], "the job still owes recovery after its caller stops waiting"
    finally:
        release.set()
        assert app.mutations.stop(timeout=5)


def test_a_future_is_not_finished_until_recovery_finishes(cfg, store, tmp_path, monkeypatch):
    app = _attach(_app(cfg, store, tmp_path))
    run_job, undo_started, finish_undo = threading.Event(), threading.Event(), threading.Event()

    def job(_paths):
        assert run_job.wait(5)
        raise RuntimeError("storage failed")

    def undo():
        undo_started.set()
        assert finish_undo.wait(5)

    monkeypatch.setattr(app.pipeline, "index", job)
    try:
        fut = app.submit("index", (), wait=False, on_retryable_failure=undo)
        run_job.set()
        assert undo_started.wait(5)
        assert not fut.done(), "completion exposed the failure before its recovery finished"
        finish_undo.set()
        with pytest.raises(RuntimeError, match="storage failed"):
            fut.result(timeout=5)
    finally:
        run_job.set()
        finish_undo.set()
        assert app.mutations.stop(timeout=5)


@pytest.mark.parametrize("mode", ["offline", "worker", "wait", "async"])
def test_success_is_recorded_once_before_the_result_is_visible(cfg, store, tmp_path, monkeypatch, mode):
    app = _app(cfg, store, tmp_path)
    if mode != "offline":
        _attach(app)
    order = []
    report = IndexRunResult(upserted_rows=1)

    def job(_paths):
        order.append("job")
        return report

    def completed():
        if app.mutations is not None:
            assert app.mutations.is_worker_thread()
        order.append("success")

    monkeypatch.setattr(app.pipeline, "index", job)

    def submit():
        return app.submit("index", (), wait=mode != "async", on_success=completed,
                          on_retryable_failure=lambda: order.append("failure"))

    try:
        if mode == "worker":
            result = app.mutations.run("outer", submit, timeout=5)
        elif mode == "async":
            result = submit().result(timeout=5)
        else:
            result = submit()
        assert result == report.as_dict() and order == ["job", "success"]
    finally:
        if app.mutations is not None:
            assert app.mutations.stop(timeout=5)
