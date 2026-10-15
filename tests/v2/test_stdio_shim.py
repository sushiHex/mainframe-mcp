import io
import json

import httpx
import pytest

from mainframe.adapters.mcp_stdio_shim import forward, make_post, run
from mainframe.service.daemon import write_discovery


def _req(id_):
    return {"jsonrpc": "2.0", "id": id_, "method": "ping"}


class _Response:
    def __init__(self, url, status_code, body):
        self.url, self.status_code, self._body = url, status_code, body
        self.content = json.dumps(body).encode()

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            httpx.Response(self.status_code, request=httpx.Request("POST", self.url)).raise_for_status()


class FakeLoopback:
    """One healthy daemon on `port`; every OTHER port refuses the connection,
    which is what loopback does once a process has exited. Records the port of
    every POST attempted and the body of every POST actually delivered, so a
    test can tell "tried twice" from "applied twice"."""

    def __init__(self, port, nonce, status=200, post_error=None):
        self.port, self.nonce = port, nonce
        self.status, self.post_error = status, post_error
        self.attempts, self.received, self.probes = [], [], []

    def restart(self, port, nonce):
        """A new daemon process: `service.port` is 0, so the OS hands it a
        fresh ephemeral port and the old one stops answering."""
        self.port, self.nonce = port, nonce

    @staticmethod
    def _port(url):
        return int(url.rsplit(":", 1)[1].split("/")[0])

    @staticmethod
    def _refuse():
        raise httpx.ConnectError("[WinError 10061] No connection could be made because "
                                 "the target machine actively refused it")

    def get(self, url, headers=None):
        self.probes.append(self._port(url))
        if self._port(url) != self.port:
            self._refuse()
        return _Response(url, 200, {"ok": True, "nonce": self.nonce})

    def post(self, url, content=None, headers=None):
        self.attempts.append(self._port(url))
        if self._port(url) != self.port:
            self._refuse()
        if self.post_error is not None:
            raise self.post_error
        self.received.append(json.loads(content))
        return _Response(url, self.status, {"result": "ok"})


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


def test_a_stale_cached_port_heals_within_the_call(cfg):
    """The daemon takes a FRESH ephemeral port every start (`service.port` 0),
    so a shim that outlives a restart holds a dead address. Healing only on the
    NEXT call left this one to fail with a raw WinError 10061 - which an
    operator read as "the health endpoint is down", because the tool that
    happened to be called first was the one that ate the stale address."""
    write_discovery(cfg, 9100, "first")
    daemon = FakeLoopback(9100, "first")
    post = make_post(cfg, http=daemon)

    assert post(_req(1))["result"] == "ok"              # caches 9100

    daemon.restart(9239, "second")
    write_discovery(cfg, 9239, "second")

    assert post(_req(2))["result"] == "ok"              # no error reaches the caller
    assert daemon.attempts == [9100, 9100, 9239]        # the dead port, then the live one
    assert [b["id"] for b in daemon.received] == [1, 2]  # each request applied exactly once


def test_a_status_error_is_not_retried(cfg):
    """A status error means the daemon RECEIVED the request. Replaying it would
    apply a mutation twice, and `capture` writes a file."""
    write_discovery(cfg, 9100, "n")
    daemon = FakeLoopback(9100, "n")
    post = make_post(cfg, http=daemon)
    assert post(_req(1))["result"] == "ok"              # warm the cache

    daemon.status = 500
    with pytest.raises(httpx.HTTPStatusError):
        post(_req(2))
    assert daemon.attempts == [9100, 9100]              # one attempt per call, not two
    assert [b["id"] for b in daemon.received] == [1, 2]


def test_a_read_failure_is_not_retried(cfg):
    """Only a connect failure proves the bytes never left. A read timeout may
    already have been processed, so it must not be replayed either."""
    write_discovery(cfg, 9100, "n")
    daemon = FakeLoopback(9100, "n")
    post = make_post(cfg, http=daemon)
    assert post(_req(1))["result"] == "ok"

    daemon.post_error = httpx.ReadTimeout("timed out")
    with pytest.raises(httpx.ReadTimeout):
        post(_req(2))
    assert daemon.attempts == [9100, 9100]


def test_a_connect_failure_on_a_freshly_probed_port_is_not_retried(cfg):
    """Nothing was cached, so the address came from a probe moments earlier: a
    second attempt would be noise, not healing."""
    write_discovery(cfg, 9100, "n")
    daemon = FakeLoopback(9100, "n", post_error=httpx.ConnectError("refused"))
    post = make_post(cfg, http=daemon)

    with pytest.raises(httpx.ConnectError):
        post(_req(1))
    assert daemon.attempts == [9100]


def test_a_dead_daemon_reports_the_actionable_message_not_a_socket_error(cfg):
    """With the daemon really gone, the caller is told to run `mainframe up`
    instead of being handed [WinError 10061] to interpret."""
    write_discovery(cfg, 9100, "first")
    daemon = FakeLoopback(9100, "first")
    post = make_post(cfg, http=daemon)
    assert post(_req(1))["result"] == "ok"

    daemon.restart(0, "gone")                           # nothing listening anywhere
    with pytest.raises(ConnectionError) as e:
        post(_req(2))
    assert "not running" in str(e.value) and str(e.value).isascii()


def test_a_failed_send_never_leaves_an_address_cached(cfg):
    """The cache holds an address only for as long as it works - on the retry
    path too. Otherwise a send that fails AFTER re-discovery leaves the shim
    trusting an address it has never successfully used."""
    write_discovery(cfg, 9100, "first")
    daemon = FakeLoopback(9100, "first")
    post = make_post(cfg, http=daemon)
    assert post(_req(1))["result"] == "ok"

    daemon.restart(9239, "second")                      # moved, and failing there too
    write_discovery(cfg, 9239, "second")
    daemon.status = 500
    with pytest.raises(httpx.HTTPStatusError):
        post(_req(2))
    assert daemon.attempts == [9100, 9100, 9239]

    daemon.status = 200
    assert post(_req(3))["result"] == "ok"
    assert daemon.probes == [9100, 9239, 9239]          # probed again, not reused blind
