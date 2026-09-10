import logging
import threading
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from mainframe.app import Mainframe
from mainframe.service import daemon as dmod
from mainframe.service.daemon import LOCK_HELD_EXIT, Daemon, read_discovery, write_discovery
from mainframe.service.events import EventStore
from mainframe.service.mutations import QueueStopped
from v2.conftest import held
from v2.fakes import fake_models
from v2.helpers import write_md

# FastMCP's transport security rejects a portless Host header, and httpx
# normalizes away a default-port URL's port — so the base_url carries an
# explicit (fictitious) port, exactly as tests/v2/test_mcp_server.py does.
BASE = "http://127.0.0.1:8420"


def _daemon(cfg, store, tmp_path):
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    return Daemon(cfg, app=app)


def _raises(exc):
    """A method body that raises — the models must stay untouched."""
    def _boom(*_args, **_kw):
        raise exc
    return _boom


def test_healthz_never_touches_models(cfg, store, tmp_path, monkeypatch):
    """Structural, not timing-based: /healthz must answer even if resolving a
    model would explode. The initial rescan then fails on the same call, which
    proves it ran and that its failure never reached /healthz."""
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    d = _daemon(cfg, store, tmp_path)
    monkeypatch.setattr(type(d.app.models), "_get", _raises(RuntimeError("touched")))
    with held(d), TestClient(d.build(), base_url=BASE) as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        h = r.json()
        assert h["ok"] is True and h["nonce"] == d.nonce
        assert h["models"]["embedder"]["state"] == "absent"
        d.mutations.run("drain", lambda: None, timeout=10)       # initial rescan already queued
        assert c.get("/events?n=50&kind=job.failed").json()       # it ran, and it failed on the model
        assert c.get("/healthz").json()["models"]["embedder"]["state"] == "absent"


def test_initial_rescan_indexes_the_lanes(cfg, store, tmp_path):
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    d = _daemon(cfg, store, tmp_path)
    with held(d), TestClient(d.build(), base_url=BASE) as c:
        d.mutations.run("drain", lambda: None, timeout=10)          # initial rescan already queued
        assert c.get("/status").json()["index"]["rows"] == 1
        out = c.post("/search", json={"query": "qwen3 reranker"}).json()
        assert out["results"][0]["file"].endswith("a.md")
        assert c.post("/jobs/nope").status_code == 400 and c.post("/search", json={"nope": 1}).status_code == 400
        assert c.post("/jobs/rescan?wait=0").json() == {"queued": True}
        info = c.post("/capture", json={"content": "## Summary\nx", "title": "t", "project": "p"}).json()
        assert Path(info["file"]).exists()
        assert "captured_by: rest" in Path(info["file"]).read_text(encoding="utf-8")
        assert c.get("/events?n=50").json()[-1]["kind"] == "daemon.start"
        # no uvicorn server is attached under TestClient, so nothing will
        # actually stop — say so instead of reporting a stop that never happens.
        assert c.post("/shutdown").json() == {"stopping": False}


def test_captured_by_is_provenance_not_caller_input(cfg, store, tmp_path):
    """`setdefault` let a REST caller's own `captured_by` survive into
    `events.emit`, which stores it verbatim and `/events` hands back — the
    capture FILE was scrubbed, the event was not. The surface decides."""
    d = _daemon(cfg, store, tmp_path)
    with held(d), TestClient(d.build(), base_url=BASE) as c:
        info = c.post("/capture", json={"content": "## Summary\nx", "title": "t", "project": "p",
                                        "captured_by": "sk_live_" + "a" * 24}).json()
        assert "captured_by: rest" in Path(info["file"]).read_text(encoding="utf-8")
        written = c.get("/events?n=50&kind=capture.written").json()
        assert written[0]["detail"]["captured_by"] == "rest"
        assert "sk_live_" not in c.get("/events?n=50").text


def test_captured_by_is_scrubbed_on_every_surface(cfg, store, tmp_path):
    """Belt and braces behind the REST overwrite: `capture()` scrubs the value
    once, so the file and the event can never disagree about it."""
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    app.capture("body", "t", project="p", captured_by="hook sk_live_" + "a" * 24)
    detail = app.events.recent(1, kind="capture.written")[0]["detail"]
    assert "sk_live_" not in detail["captured_by"] and detail["captured_by"].startswith("hook ")


def test_host_header_must_be_loopback(cfg, store, tmp_path):
    """FastMCP guards its own sub-apps, but /search, /capture, /jobs,
    /events and /shutdown are registered on the parent — with the default
    token=None a rebound DNS name reached /shutdown."""
    d = _daemon(cfg, store, tmp_path)
    with held(d), TestClient(d.build(), base_url=BASE) as c:
        assert c.get("/healthz").status_code == 200                       # 127.0.0.1:8420
        assert c.get("/healthz", headers={"Host": "localhost:7433"}).status_code == 200
        assert c.get("/healthz", headers={"Host": "127.0.0.1"}).status_code == 200
        for bad in ("evil.com", "evil.com:8420", "127.0.0.1.evil.com:8420", ""):
            r = c.post("/shutdown", headers={"Host": bad})
            assert r.status_code == 421, bad
            assert r.json()["error"] and r.json()["guidance"]
        assert c.get("/status", headers={"Host": "evil.com"}).status_code == 421


def test_events_n_is_clamped(cfg, store, tmp_path):
    """`int(-1)` reached SQLite's `LIMIT -1`, which means unlimited."""
    d = _daemon(cfg, store, tmp_path)
    with held(d), TestClient(d.build(), base_url=BASE) as c:
        for i in range(5):
            d.app.events.emit("test.event", i=i)
        assert len(c.get("/events?n=-1").json()) == 1
        assert len(c.get("/events?n=0").json()) == 1
        assert len(c.get("/events?n=3").json()) == 3
        assert len(c.get("/events?n=99999").json()) <= 1000
        assert c.get("/events?n=abc").status_code == 400


def test_status_never_installs_a_doc_count(cfg, store, tmp_path):
    """`status()` runs on a reader thread; assigning `pipeline.doc_count` there
    could overwrite a newer count the writer thread had just set."""
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    assert app.status()["index"]["docs"] == 0
    assert app.pipeline.doc_count is None, "a reader must not install the writer's value"

    app.submit("rescan")
    assert app.pipeline.doc_count == 1 and app.status()["index"]["docs"] == 1

    # and an unreadable ledger degrades the field instead of raising
    fresh = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev2.db"), store=store)

    def _boom(*_a, **_kw):
        raise RuntimeError("scan boom")
    store._scan = _boom
    assert fresh.status()["index"]["docs"] is None


def test_shutdown_reports_whether_a_server_will_actually_stop(cfg, store, tmp_path):
    d = _daemon(cfg, store, tmp_path)
    with held(d), TestClient(d.build(), base_url=BASE) as c:
        assert c.post("/shutdown").json() == {"stopping": False}

        class FakeServer:
            should_exit = False

        d._server = FakeServer()
        assert c.post("/shutdown").json() == {"stopping": True}
        assert d._server.should_exit is True


def test_the_queue_stays_attached_and_rejects_after_shutdown(cfg, store, tmp_path):
    """`_stop` used to null `app.mutations`, so a write arriving after
    shutdown would run INLINE on a foreign thread — the exact opposite of the
    single-writer guarantee `submit` documents. The queue stays attached and
    rejects instead, and `status` still reports an idle queue."""
    d = _daemon(cfg, store, tmp_path)
    with held(d):
        with TestClient(d.build(), base_url=BASE):
            pass
        assert d.app.mutations is not None
        with pytest.raises(QueueStopped):
            d.app.submit("optimize")
        daemon_status = d.app.status()["daemon"]
        assert daemon_status["queue_depth"] == 0 and daemon_status["running"] is None


def test_rest_never_raises_across_the_transport(cfg, store, tmp_path, monkeypatch):
    """Handlers only translated TypeError/ValueError/QueueStopped,
    so the MOST LIKELY production failure — `RuntimeError: embedder unavailable:
    ...; retry in 873s` during the models backoff — became a bare 500 that the
    CLI then reported as "could not reach the daemon".

    raise_server_exceptions=False: Starlette's ServerErrorMiddleware sends the
    JSON response and THEN re-raises so the server logs it (uvicorn does), and
    the wire response is what a client sees.
    """
    d = _daemon(cfg, store, tmp_path)
    with held(d), TestClient(d.build(), base_url=BASE, raise_server_exceptions=False) as c:
        d.mutations.run("drain", lambda: None, timeout=10)
        monkeypatch.setattr(d.app, "search", _raises(RuntimeError("embedder unavailable: boom; retry in 9s")))
        r = c.post("/search", json={"query": "x"})
        assert r.status_code == 503
        assert "unavailable" in r.json()["error"] and r.json()["guidance"]

        monkeypatch.setattr(d.app, "search", _raises(RuntimeError("kaboom")))
        r = c.post("/search", json={"query": "x"})
        assert r.status_code == 500
        assert r.json()["error"] == "kaboom" and r.json()["guidance"]

        # A stale index is a 409: the request was fine and the server is
        # healthy, but its state conflicts with the configured settings and the
        # remedy is a specific operator action. As a 500 that remedy reached
        # only the log.
        from mainframe.core.store import IndexStaleError
        monkeypatch.setattr(d.app, "search", _raises(
            IndexStaleError("search refused: ... Stop the daemon and run `mainframe rebuild`")))
        r = c.post("/search", json={"query": "x"})
        assert r.status_code == 409
        assert "mainframe rebuild" in r.json()["error"] and "mainframe rebuild" in r.json()["guidance"]

        monkeypatch.setattr(d.app, "status", _raises(OSError("capture dir vanished")))
        assert c.get("/status").status_code == 500 and c.get("/status").json()["guidance"]


def test_search_bad_response_format_is_400(cfg, store, tmp_path):
    """SearchService raises ValueError for an unknown response_format; the
    REST handler must turn that into the same 400 {error, guidance} shape as
    a TypeError, not let it surface as a 500."""
    d = _daemon(cfg, store, tmp_path)
    with held(d), TestClient(d.build(), base_url=BASE) as c:
        d.mutations.run("drain", lambda: None, timeout=10)
        r = c.post("/search", json={"query": "x", "response_format": "bogus"})
        assert r.status_code == 400
        body = r.json()
        assert body["error"] and body["guidance"]


def test_jobs_index_reconciles_and_rescan_takes_a_project(cfg, store, tmp_path):
    """`index` needs paths, which a REST caller does not have — over REST it
    means "reconcile", exactly like the MCP `maintain` tool."""
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    d = _daemon(cfg, store, tmp_path)
    with held(d), TestClient(d.build(), base_url=BASE) as c:
        d.mutations.run("drain", lambda: None, timeout=10)
        r = c.post("/jobs/index")
        assert r.status_code == 200, r.text
        rep = r.json()
        assert rep["unchanged"] == 1 and rep["upserted_rows"] == 0
        assert c.post("/jobs/rescan?project=p").json()["files"] == 1
        assert c.post("/jobs/rescan?project=nope").json()["files"] == 0
        # `project` reaches a filesystem join and a delete
        # predicate — it is validated, not passed through.
        bad = c.post("/jobs/rescan?project=../x")
        assert bad.status_code == 400 and bad.json()["error"] and bad.json()["guidance"]


def test_jobs_return_503_while_stopping(cfg, store, tmp_path):
    d = _daemon(cfg, store, tmp_path)
    with held(d), TestClient(d.build(), base_url=BASE) as c:
        assert d.mutations.stop(timeout=5) is True
        r = c.post("/jobs/rescan")
        assert r.status_code == 503
        assert r.json()["error"] and r.json()["guidance"]


def test_shutdown_never_frees_the_lock_while_the_writer_lives(cfg, store, tmp_path, monkeypatch):
    """`mutations.stop(timeout)` returning False left the worker running,
    yet `_stop` removed discovery and returned and `serve()` then released
    `.daemon.lock` — so a replacement daemon could open LanceDB behind a thread
    that was still mid-merge. The stop timeout is a report, not a cancellation.

    Here the worker is wedged on an event: `_stop` must not come back, must
    keep discovery published, and must keep the lock — and must finish the
    moment the worker does.
    """
    monkeypatch.setattr(dmod, "SLOW_STOP_WARN_SECONDS", 0.05)
    cfg["service"]["shutdown_timeout_seconds"] = 0.2
    d = _daemon(cfg, store, tmp_path)
    d._components()
    d.mutations.start()
    write_discovery(cfg, 1234, d.nonce)
    running, release = threading.Event(), threading.Event()

    def wedged():
        running.set()
        release.wait(30)

    assert d.acquire()
    try:
        d.mutations.submit("wedged", wedged)
        assert running.wait(5)
        stopping = threading.Thread(target=d._stop, name="stopping")
        stopping.start()

        stopping.join(1.0)                       # well past the 0.2 s stop timeout
        assert stopping.is_alive(), "_stop returned while the writer thread was still running"
        assert read_discovery(cfg) is not None
        assert Daemon(cfg, app=d.app).acquire() is False       # still ours

        release.set()
        stopping.join(10)
        assert not stopping.is_alive()
        assert read_discovery(cfg) is None
        assert d.app.events.recent(50, kind="daemon.stop")[0]["detail"]["clean"] is False
    finally:
        release.set()
        d.release()
    replacement = Daemon(cfg, app=d.app)
    assert replacement.acquire(), "the lock must be free once the writer is gone"
    replacement.release()


def test_release_holds_the_lock_even_when_stop_raises(cfg, store, tmp_path, monkeypatch):
    """`_stop` DECLARED the invariant, but `serve()`'s `finally: release()` is
    the actual release site and had no guard — so any raise inside `_stop`
    before the wait (a non-numeric `shutdown_timeout_seconds` reaching
    `float()`, a scheduler or watcher that throws on stop) freed the lock with
    the worker alive.

    `release()` is called on the acquiring thread, as `serve()` calls it:
    `filelock` is thread-local by default, so a release from anywhere else is
    a no-op that would prove nothing."""
    monkeypatch.setattr(dmod, "SLOW_STOP_WARN_SECONDS", 0.05)
    cfg["service"]["shutdown_timeout_seconds"] = "not-a-number"
    d = _daemon(cfg, store, tmp_path)
    d._components()
    d.mutations.start()
    running, unblock = threading.Event(), threading.Event()
    d.mutations.submit("wedged", lambda: (running.set(), unblock.wait(30)))
    assert running.wait(5)
    assert d.acquire()
    probe = {}

    def probe_then_unblock():
        """While `release()` waits, the lock must still be ours."""
        time.sleep(0.3)
        other = Daemon(cfg, app=d.app)
        probe["taken"] = other.acquire()
        if probe["taken"]:
            other.release()
        unblock.set()

    helper = threading.Thread(target=probe_then_unblock, name="probe")
    try:
        with pytest.raises(ValueError):
            d._stop()                       # dies long before it reaches the wait
        helper.start()
        t0 = time.monotonic()
        d.release()
        waited = time.monotonic() - t0
    finally:
        unblock.set()
        helper.join(10)
        d.release()

    assert probe["taken"] is False, "release() freed the lock with the writer alive"
    assert waited >= 0.3, "release() did not wait for the writer"
    assert d.mutations._thread is None      # it returned only once the worker was gone
    replacement = Daemon(cfg, app=d.app)
    assert replacement.acquire(), "the lock must be free once the writer is gone"
    replacement.release()


def test_discovery_roundtrip_and_bind_order(cfg, store, tmp_path):
    d = _daemon(cfg, store, tmp_path)
    assert read_discovery(cfg) is None
    with held(d):
        d.port = 0
        sock = d.bind()                              # binds first, then writes discovery
        try:
            info = read_discovery(cfg)
            assert info["port"] == sock.getsockname()[1] and info["nonce"] == d.nonce and info["pid"]
        finally:
            sock.close()


def test_stop_keeps_a_foreign_discovery(cfg, store, tmp_path):
    """A straggler must never unpublish a newer instance's daemon.json."""
    d = _daemon(cfg, store, tmp_path)
    with held(d):
        write_discovery(cfg, 1, "someone-else")
        with TestClient(d.build(), base_url=BASE):
            pass
        assert read_discovery(cfg)["nonce"] == "someone-else"

    later = _daemon(cfg, store, tmp_path)
    with held(later):
        later.port = 0
        sock = later.bind()
        try:
            assert read_discovery(cfg)["nonce"] == later.nonce  # the live instance republishes
        finally:
            sock.close()


def test_serve_failure_after_bind_removes_discovery(cfg, store, tmp_path, monkeypatch):
    """Discovery is published between bind() and the lifespan; if the launch
    dies in that window it must not survive with our nonce and a dead port."""
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    monkeypatch.setattr(Daemon, "_configure_process", lambda self: None)   # no env/logging side effects here
    monkeypatch.setattr(Daemon, "build", _raises(RuntimeError("boom")))
    d = Daemon(cfg, app=app, port=0)
    with pytest.raises(RuntimeError, match="boom"):
        d.serve()
    assert read_discovery(cfg) is None
    other = Daemon(cfg, app=app)
    assert other.acquire()                                                 # and the lock was released
    other.release()


def test_second_instance_cannot_take_the_lock(cfg, store, tmp_path):
    a = _daemon(cfg, store, tmp_path)
    assert a.acquire()
    b = Daemon(cfg, app=a.app)
    assert b.acquire() is False
    a.release()
    assert b.acquire()
    b.release()


def test_lock_holds_across_processes(cfg, store, tmp_path):
    import subprocess, sys, textwrap
    a = _daemon(cfg, store, tmp_path)
    with held(a):
        code = textwrap.dedent(f"""
            from filelock import FileLock, Timeout
            try:
                FileLock(r"{tmp_path / 'mainframe' / '.daemon.lock'}").acquire(timeout=0); print("ACQUIRED")
            except Timeout:
                print("LOCKED")
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60).stdout
        assert "LOCKED" in out


def test_token_middleware_covers_rest_and_mcp(cfg, store, tmp_path):
    cfg["service"]["token"] = "s3cret"
    d = _daemon(cfg, store, tmp_path)
    mcp_headers = {"Accept": "application/json, text/event-stream"}
    with held(d), TestClient(d.build(), base_url=BASE) as c:
        assert c.get("/healthz").status_code == 401
        assert c.post("/mcp/ro", content="{}", headers=mcp_headers).status_code == 401
        assert c.post("/mcp", content="{}", headers=mcp_headers).status_code == 401
        assert c.get("/healthz", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/healthz", headers={"Authorization": "Bearer s3cre"}).status_code == 401


def test_serve_refuses_to_start_on_a_broken_config(tmp_path, monkeypatch):
    """The daemon lets `ConfigError` through rather than starting on DEFAULTS,
    whose empty `include_projects` means every project under repos_dir. It
    raises before the lock, the socket or LanceDB is touched."""
    from mainframe.config import ConfigError
    broken = tmp_path / "config.json"
    broken.write_text('{"paths": {"include_projects": ["a"]', encoding="utf-8")
    monkeypatch.setenv("MAINFRAME_CONFIG", str(broken))
    with pytest.raises(ConfigError):
        dmod.serve()


def test_serve_exits_3_when_locked(cfg, store, tmp_path, monkeypatch):
    """The LOSING instance configures nothing: no HF_HUB_OFFLINE, no handlers."""
    a = _daemon(cfg, store, tmp_path)
    with held(a):
        offline_calls = []
        monkeypatch.setattr(dmod, "hf_offline_if_cached", lambda c: offline_calls.append(c))
        handlers_before = list(logging.getLogger("mainframe").handlers)
        with pytest.raises(SystemExit) as e:
            dmod.serve(cfg)
        assert e.value.code == LOCK_HELD_EXIT
        assert offline_calls == []
        assert logging.getLogger("mainframe").handlers == handlers_before
