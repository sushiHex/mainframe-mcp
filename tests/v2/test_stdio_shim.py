import io
import json

import pytest

from mainframe.adapters.mcp_stdio_shim import forward, make_post, run
from mainframe.service.daemon import write_discovery


def test_forward_request_and_notification():
    seen = []

    def post(body):
        seen.append(body)
        return {"jsonrpc": "2.0", "id": body.get("id"), "result": {"tools": []}}
    out = forward(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/list"}), post)
    assert json.loads(out)["id"] == 7 and seen[-1]["method"] == "tools/list"
    assert forward(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}), post) is None


def test_forward_errors():
    def post(body):
        raise ConnectionError("down")
    assert json.loads(forward(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}), post))["error"]["code"] == -32000
    assert json.loads(forward("not json", post))["error"]["code"] == -32700
    assert json.loads(forward("[1,2]", post))["error"]["code"] == -32600
    assert forward(json.dumps({"jsonrpc": "2.0", "method": "n"}), post) is None     # failed notification: silent


def test_run_streams_lines():
    stdin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "a"}) + "\n\n" +
                        json.dumps({"jsonrpc": "2.0", "method": "n"}) + "\n" +
                        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "b"}) + "\n")
    stdout = io.StringIO()
    run(stdin, stdout, lambda body: {"jsonrpc": "2.0", "id": body.get("id"), "result": body["method"]})
    lines = [json.loads(l) for l in stdout.getvalue().splitlines()]
    assert [l["id"] for l in lines] == [1, 2] and lines[1]["result"] == "b"


def test_forward_empty_response_is_an_error():
    def post(body):
        return None
    out = json.loads(forward(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}), post))
    assert out["error"]["code"] == -32000 and out["id"] == 1


def test_make_post_rejects_stale_discovery(cfg):
    write_discovery(cfg, 1234, "n1")

    class FakeHttp:
        def get(self, url, headers=None):
            class R:
                status_code = 200
                def json(self):
                    return {"ok": True, "nonce": "other"}
            return R()

    post = make_post(cfg, http=FakeHttp())
    try:
        post({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert False, "should have raised ConnectionError"
    except ConnectionError as e:
        assert "not running" in str(e) and str(e).isascii()     # no em dash through cp1252

    # Now test with matching nonce
    class FakeHttpMatching:
        def __init__(self):
            self.posted_url = None
            self.posted_headers = None

        def get(self, url, headers=None):
            class R:
                status_code = 200
                def json(self):
                    return {"ok": True, "nonce": "n1"}
            return R()

        def post(self, url, content=None, headers=None):
            self.posted_url = url
            self.posted_headers = headers
            class R:
                status_code = 200
                content = b'{"result": "ok"}'
                def json(self):
                    return {"result": "ok"}
                def raise_for_status(self):
                    pass
            return R()

    fake = FakeHttpMatching()
    post = make_post(cfg, http=fake)
    result = post({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert result["result"] == "ok"
    assert fake.posted_url == "http://127.0.0.1:1234/mcp"
    assert fake.posted_headers["Accept"] == "application/json, text/event-stream"
    assert fake.posted_headers["Content-Type"] == "application/json"


def test_make_post_rediscovers_after_daemon_appears(cfg):
    """A shim started before the daemon (a logon race) resolves discovery
    lazily: the first call fails cleanly, and once the daemon appears (writes
    discovery, comes healthy) the next call re-discovers and succeeds."""
    write_discovery(cfg, 1234, "old")

    class FakeHttp:
        def __init__(self):
            self.get_calls = 0

        def get(self, url, headers=None):
            self.get_calls += 1
            status = 404 if self.get_calls == 1 else 200
            body = {} if self.get_calls == 1 else {"ok": True, "nonce": "new"}
            class R:
                status_code = status
                def json(self_inner):
                    return body
            return R()

        def post(self, url, content=None, headers=None):
            class R:
                status_code = 200
                content = b'{"result": "ok"}'
                def json(self_inner):
                    return {"result": "ok"}
                def raise_for_status(self_inner):
                    pass
            return R()

    fake = FakeHttp()
    post = make_post(cfg, http=fake)

    with pytest.raises(ConnectionError):
        post({"jsonrpc": "2.0", "id": 1, "method": "ping"})

    write_discovery(cfg, 1234, "new")   # models the daemon starting late

    result = post({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert result["result"] == "ok"
    assert fake.get_calls == 2
