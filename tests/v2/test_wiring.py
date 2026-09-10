"""End-to-end in-process: watcher event → scheduler tick → queue → index → search over MCP and REST."""
import json
from pathlib import Path

from starlette.testclient import TestClient

from mainframe.app import Mainframe
from mainframe.service import watcher as watcher_mod
from mainframe.service.daemon import Daemon
from mainframe.service.events import EventStore
from v2.conftest import held
from v2.fakes import fake_models
from v2.helpers import write_md
from v2.test_watcher import FakeObserver

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
# FastMCP's transport security rejects a portless Host header, and httpx
# normalizes away a default-port URL's port — base_url carries an explicit
# (fictitious) port, as tests/v2/test_daemon.py and test_mcp_server.py do.
BASE = "http://127.0.0.1:8420"


def test_end_to_end_index_loop(cfg, store, tmp_path):
    cfg["index"]["debounce_seconds"] = 0
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    d = Daemon(cfg, app=app)
    with held(d), TestClient(d.build(), base_url=BASE) as c:
        d.mutations.run("drain-initial-rescan", lambda: None, timeout=10)
        f = write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
        d.watcher.on_event(str(f))                 # what watchdog would deliver
        d.scheduler.tick()                         # debounce 0 → index job submitted
        d.mutations.run("drain", lambda: None, timeout=10)
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "search", "arguments": {"query": "qwen3 reranker"}}}
        payload = json.loads(c.post("/mcp/ro", content=json.dumps(body), headers=HEADERS).json()["result"]["content"][0]["text"])
        assert payload["results"][0]["file"].endswith("a.md")
        assert c.post("/search", json={"query": "qwen3 reranker"}).json()["results"][0]["file"].endswith("a.md")
        kinds = [e["kind"] for e in c.get("/events?n=20").json()]
        assert "index.batch" in kinds and "daemon.start" in kinds and "index.rescan" in kinds
        st = c.get("/status").json()
        assert st["index"]["rows"] == 1 and st["daemon"]["nonce"] == d.nonce


def _no_os_observer(monkeypatch):
    """Fake ONLY the OS observer: `_components` still builds the real watcher,
    predicates, scheduler and queue, but no asynchronous filesystem event can
    race the ones the test delivers by hand (a real DirModifiedEvent on `docs/`
    legitimately flags a rescan, which would make these assertions flaky)."""
    monkeypatch.setattr(watcher_mod, "_default_observer_factory", FakeObserver)


def test_delete_event_removes_the_doc_without_a_rescan(cfg, store, tmp_path, monkeypatch):
    """a delete (and the source half of a rename) is dispatched AFTER the
    file is gone, so `resolve`'s is_file() check dropped it and removal
    depended entirely on a directory event triggering a full-lane rescan. With `could_belong` the path reaches the dirty
    set and the pipeline turns a vanished ledger entry into a delete."""
    _no_os_observer(monkeypatch)
    cfg["index"]["debounce_seconds"] = 0
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    f = write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    d = Daemon(cfg, app=app)
    with held(d), TestClient(d.build(), base_url=BASE):
        d.mutations.run("drain-initial-rescan", lambda: None, timeout=10)
        assert store.count_rows() == 1
        rescans = app.events.count("index.rescan")

        f.unlink()
        d.watcher.on_event(str(f))                 # what watchdog delivers for a deletion
        assert d.invalidation.state(0)[0] == 1
        d.scheduler.tick()
        d.mutations.run("drain", lambda: None, timeout=10)

        assert store.count_rows() == 0 and store.ledger() == {}
        assert app.events.recent(1, kind="index.batch")[0]["detail"]["deleted_docs"] == 1
        assert app.events.count("index.rescan") == rescans        # the batch did it, not a rescan


def test_rename_indexes_the_destination_and_drops_the_source(cfg, store, tmp_path, monkeypatch):
    """Without the rename source path, both the old and the new
    doc stayed in the index and search kept returning the stale path."""
    _no_os_observer(monkeypatch)
    cfg["index"]["debounce_seconds"] = 0
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    docs = Path(cfg["paths"]["repos_dir"]) / "p" / "docs"
    old = write_md(docs / "old.md", "# T\n\nqwen3 reranker text.\n")
    d = Daemon(cfg, app=app)
    with held(d), TestClient(d.build(), base_url=BASE) as c:
        d.mutations.run("drain-initial-rescan", lambda: None, timeout=10)
        assert store.count_rows() == 1
        rescans = app.events.count("index.rescan")

        new = docs / "new.md"
        old.rename(new)
        d.watcher.on_event(str(old), dest_path=str(new))
        d.scheduler.tick()
        d.mutations.run("drain", lambda: None, timeout=10)

        assert [Path(p).name for p in store.ledger()] == ["new.md"]
        assert app.events.count("index.rescan") == rescans
        assert c.post("/search", json={"query": "qwen3 reranker"}).json()["results"][0]["file"].endswith("new.md")
