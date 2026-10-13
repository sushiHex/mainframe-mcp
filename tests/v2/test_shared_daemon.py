"""Independent stdio clients on ONE daemon (#19), through the real shim and
the real routes: discovery, `/healthz`, the nonce check and `/mcp` are the
production code. Only the socket is replaced - Starlette's TestClient stands
in for loopback, so no port is opened and no model is loaded."""

import io
import json
from pathlib import Path

import httpx
from starlette.testclient import TestClient

from mainframe.adapters.mcp_stdio_shim import forward, make_post, run
from mainframe.app import Mainframe
from mainframe.core.paths import canonical
from mainframe.service.daemon import Daemon, write_discovery
from mainframe.service.events import EventStore
from v2.conftest import held
from v2.fakes import fake_models
from v2.helpers import write_md

PORT = 8420                                     # fictitious: TestClient answers every port
BASE = f"http://127.0.0.1:{PORT}"


class Loopback:
    """What a client sees on 127.0.0.1: the daemon once one is up, a refused
    connection before - which is also what a stale daemon.json points at."""

    def __init__(self):
        self.daemon = None

    def _up(self):
        if self.daemon is None:
            raise httpx.ConnectError("[WinError 10061] No connection could be made because "
                                     "the target machine actively refused it")
        return self.daemon

    def get(self, url, headers=None):
        return self._up().get(url, headers=headers)

    def post(self, url, content=None, headers=None):
        return self._up().post(url, content=content, headers=headers)


def _daemon(cfg, store, tmp_path):
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    return Daemon(cfg, app=app)


def _line(id_, method, params=None) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}})


def _rpc(post, id_, method, params=None) -> dict:
    """One line in, one line out, exactly as the host drives the shim."""
    return json.loads(forward(_line(id_, method, params), post))


def _tool(post, id_, name, **arguments) -> dict:
    res = _rpc(post, id_, "tools/call", {"name": name, "arguments": arguments})["result"]
    assert not res.get("isError"), res
    return json.loads(res["content"][0]["text"])


def test_two_clients_share_one_daemon_and_captures_stay_opt_in(cfg, store, tmp_path):
    """Two shims, two sessions, one daemon: both search the same index and see
    the same daemon identity. One captures; the other retrieves it with
    sessions enabled, while its ordinary search still excludes the capture."""
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    d = _daemon(cfg, store, tmp_path)
    loopback = Loopback()
    alice, bob = make_post(cfg, http=loopback), make_post(cfg, http=loopback)
    with held(d), TestClient(d.build(), base_url=BASE) as daemon:
        loopback.daemon = daemon
        write_discovery(cfg, PORT, d.nonce)                 # what bind() publishes
        d.mutations.run("drain", lambda: None, timeout=10)  # the initial rescan has run

        for client in (alice, bob):
            hit = _tool(client, 1, "search", query="qwen3 reranker")
            assert hit["results"][0]["file"].endswith("a.md")
            assert _tool(client, 2, "status")["daemon"]["nonce"] == d.nonce

        note = _tool(alice, 3, "capture", content="## Summary\nzebra quokka rollback decided.\n",
                     title="decision", project="p")
        # Production indexes it after the watcher's debounce; a client that
        # wants it now asks for the same job the watcher would submit.
        _tool(bob, 4, "maintain", action="rescan")

        hidden = _tool(bob, 5, "search", query="zebra quokka")
        assert canonical(note["file"]) not in [r["file"] for r in hidden["results"]]
        shown = _tool(bob, 6, "search", query="zebra quokka", include_sessions=True)
        assert shown["results"][0]["file"] == canonical(note["file"])


def test_a_client_started_before_the_daemon_recovers_once_it_is_up(cfg, store, tmp_path):
    """A host may launch the shim before the daemon answers (a logon race).
    Every call until then is told to run `mainframe up` - never a socket error
    - and the SAME shim serves once the daemon publishes discovery. Nothing is
    restarted on the client side."""
    loopback = Loopback()
    shim = make_post(cfg, http=loopback)

    def ask(id_) -> dict:
        out = io.StringIO()
        run(io.StringIO(_line(id_, "tools/list") + "\n"), out, shim)
        return json.loads(out.getvalue())

    err = ask(1)["error"]                                   # no daemon.json at all
    assert err["code"] == -32000 and "run `mainframe up`" in err["message"] and err["message"].isascii()

    write_discovery(cfg, PORT, "previous")                  # a dead daemon's file, nothing listening
    err = ask(2)["error"]
    assert "run `mainframe up`" in err["message"] and "10061" not in err["message"]

    d = _daemon(cfg, store, tmp_path)
    with held(d), TestClient(d.build(), base_url=BASE) as daemon:
        loopback.daemon = daemon
        write_discovery(cfg, PORT, d.nonce)
        tools = ask(3)["result"]["tools"]
        assert [t["name"] for t in tools] == ["search", "capture", "status", "maintain"]
