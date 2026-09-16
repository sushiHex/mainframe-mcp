import json
from pathlib import Path

import pytest
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mainframe.adapters.mcp_server import build_mcp, mount
from mainframe.app import Mainframe
from mainframe.service.events import EventStore
from v2.fakes import fake_models
from v2.helpers import write_md

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def rpc(client, path, method, params=None, id=1):
    r = client.post(path, content=json.dumps({"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}}),
                    headers=HEADERS)
    assert r.status_code == 200, r.text
    return r.json()


def _client(cfg, store, tmp_path):
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    app.submit("rescan")
    full, ro = build_mcp(app)
    # FastMCP's default transport security (DNS-rebinding protection) allows
    # only Host headers of the form "127.0.0.1:<port>" (httpx normalizes away
    # a bare default-port URL's port, so a portless base_url produces a Host
    # header of "127.0.0.1" and is rejected with 421) — give TestClient an
    # explicit, if fictitious, port so its Host header matches.
    # follow_redirects=False: every MCP request is a POST, and the Python MCP
    # SDK's httpx client does not follow redirects by default either — the
    # bare "/mcp" and "/mcp/ro" paths below must work with NO redirect.
    return TestClient(mount(full, ro), base_url="http://127.0.0.1:8420", follow_redirects=False), app


def test_tool_lists_and_annotations(cfg, store, tmp_path):
    c, _ = _client(cfg, store, tmp_path)
    with c:
        tools = rpc(c, "/mcp", "tools/list")["result"]["tools"]
        assert [t["name"] for t in tools] == ["search", "capture", "status", "maintain"]
        by = {t["name"]: t for t in tools}
        assert by["search"]["annotations"]["readOnlyHint"] is True and by["status"]["annotations"]["readOnlyHint"] is True
        assert not (by["maintain"].get("annotations") or {}).get("readOnlyHint")
        assert [t["name"] for t in rpc(c, "/mcp/ro", "tools/list")["result"]["tools"]] == ["search", "status"]


def test_search_status_maintain_over_mcp(cfg, store, tmp_path):
    c, app = _client(cfg, store, tmp_path)
    with c:
        res = rpc(c, "/mcp", "tools/call", {"name": "search", "arguments": {"query": "qwen3 reranker"}})["result"]
        payload = json.loads(res["content"][0]["text"])
        assert payload["reranked"] is True and payload["results"][0]["file"].endswith("a.md")
        st = json.loads(rpc(c, "/mcp/ro", "tools/call", {"name": "status", "arguments": {}})["result"]["content"][0]["text"])
        assert st["index"]["rows"] == 1
        rep = json.loads(rpc(c, "/mcp", "tools/call", {"name": "maintain", "arguments": {"action": "index"}})["result"]["content"][0]["text"])
        assert rep["unchanged"] == 1
        bad = json.loads(rpc(c, "/mcp", "tools/call", {"name": "maintain", "arguments": {"action": "explode"}})["result"]["content"][0]["text"])
        assert "guidance" in bad
        # an unsafe project is refused on this surface too.
        unsafe = json.loads(rpc(c, "/mcp", "tools/call", {"name": "maintain",
                                                          "arguments": {"action": "rescan", "project": "../x"}})
                            ["result"]["content"][0]["text"])
        assert unsafe["error"] and unsafe["guidance"]


def test_ro_mount_refuses_mutations_and_capture_writes(cfg, store, tmp_path):
    c, _ = _client(cfg, store, tmp_path)
    with c:
        out = rpc(c, "/mcp/ro", "tools/call", {"name": "maintain", "arguments": {"action": "rescan"}})
        assert out["result"]["isError"] is True
        info = json.loads(rpc(c, "/mcp", "tools/call", {"name": "capture", "arguments": {"content": "## Summary\nx", "title": "t", "project": "p"}})["result"]["content"][0]["text"])
        assert Path(info["file"]).exists()
        # provenance says which surface wrote it.
        assert "captured_by: capture_memory" in Path(info["file"]).read_text(encoding="utf-8")


def test_bare_mount_paths_work_without_redirect(cfg, store, tmp_path):
    c, _ = _client(cfg, store, tmp_path)
    with c:
        for path, expected in (("/mcp", ["search", "capture", "status", "maintain"]),
                                ("/mcp/ro", ["search", "status"])):
            r = c.post(path, content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}),
                       headers=HEADERS)
            assert r.status_code == 200, r.text
            assert r.history == []
            assert [t["name"] for t in r.json()["result"]["tools"]] == expected


def test_mount_runs_extra_routes_and_hooks(cfg, store, tmp_path):
    """The ONE lifespan serves the daemon's extra routes and runs its hooks —
    and `on_shutdown` still runs when `on_startup` blows up half-way."""
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    calls = []
    full, ro = build_mcp(app)
    mounted = mount(full, ro, extra_routes=[Route("/ping", lambda request: JSONResponse({"pong": True}))],
                    on_startup=lambda: calls.append("startup"), on_shutdown=lambda: calls.append("shutdown"))
    with TestClient(mounted, base_url="http://127.0.0.1:8420") as c:
        assert c.get("/ping").json() == {"pong": True}
        assert calls == ["startup"]
    assert calls == ["startup", "shutdown"]

    boom_calls = []

    def boom():
        boom_calls.append("startup")
        raise RuntimeError("startup exploded")

    full2, ro2 = build_mcp(app)                      # a session manager can only be entered once
    failing = mount(full2, ro2, on_startup=boom, on_shutdown=lambda: boom_calls.append("shutdown"))
    # The session manager's anyio task group re-raises as an ExceptionGroup.
    with pytest.raises(BaseExceptionGroup) as excinfo:
        with TestClient(failing, base_url="http://127.0.0.1:8420"):
            pass
    assert excinfo.group_contains(RuntimeError, match="startup exploded")
    assert boom_calls == ["startup", "shutdown"]
