"""The ONE discovery+healthz+nonce check.

`cli.DaemonClient.alive()` and the stdio shim's `_discover()` were the same six
lines with different error handling; a stale `daemon.json` pointing at a
recycled port must be rejected by both, identically.
"""

from mainframe.service.discovery import probe, write_discovery


class FakeHTTP:
    def __init__(self, nonce=None, status=200, raises=False):
        self.nonce, self.status, self.raises = nonce, status, raises
        self.headers, self.urls = None, []

    def get(self, url, headers=None):
        self.urls.append(url)
        self.headers = headers
        if self.raises:
            raise ConnectionError("refused")
        body, status = {"ok": True, "nonce": self.nonce}, self.status

        class R:
            status_code = status

            def json(self_inner):
                return body
        return R()


def test_probe_returns_the_info_only_for_a_matching_live_daemon(cfg):
    assert probe(cfg, http=FakeHTTP(nonce="n1")) is None          # no daemon.json at all

    write_discovery(cfg, 1234, "n1")
    assert probe(cfg, http=FakeHTTP(nonce="other")) is None       # a foreign service on a recycled port
    assert probe(cfg, http=FakeHTTP(nonce="n1", status=503)) is None
    assert probe(cfg, http=FakeHTTP(raises=True)) is None         # unreachable is not an exception

    http = FakeHTTP(nonce="n1")
    info = probe(cfg, http=http)
    assert info["port"] == 1234 and info["nonce"] == "n1" and info["pid"]
    assert http.urls == ["http://127.0.0.1:1234/healthz"] and http.headers == {}


def test_probe_sends_the_bearer_token_when_one_is_configured(cfg):
    write_discovery(cfg, 1234, "n1")
    cfg["service"]["token"] = "s3cret"
    http = FakeHTTP(nonce="n1")
    assert probe(cfg, http=http)["port"] == 1234
    assert http.headers == {"Authorization": "Bearer s3cret"}
