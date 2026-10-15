"""stdio → HTTP shim for hosts that only speak MCP over stdio:
one newline-delimited JSON-RPC message in, one POST to the daemon's stateless
mount, one line out. It never spawns the daemon (Task Scheduler owns it)."""

import json
import sys

from mainframe.config import load_config
from mainframe.service.discovery import probe

_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
# ASCII: this line reaches a host's log, often through cp1252 on Windows.
_NOT_RUNNING = "mainframed is not running - run `mainframe up`"


def _error(id_, code: int, message: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}})


def _never_sent(exc) -> bool:
    """True when `exc` PROVES the request never reached a daemon.

    Only a connect-phase failure proves it, and only such a failure may be
    replayed: a read timeout or a reset mid-response may already have been
    processed, and a status error certainly was. `capture` writes a file, so a
    retry that cannot make that distinction would apply a mutation twice.
    """
    import httpx
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))


def forward(line: str, post):
    try:
        msg = json.loads(line)
    except ValueError:
        return _error(None, -32700, "parse error")
    if not isinstance(msg, dict):
        return _error(None, -32600, "invalid request: expected a JSON object")
    is_request = "id" in msg
    try:
        resp = post(msg)
    except Exception as e:
        return _error(msg.get("id"), -32000, f"mainframed unreachable: {e}") if is_request else None
    if not is_request:
        return None
    return json.dumps(resp) if resp is not None else _error(msg.get("id"), -32000, "empty response")


def run(stdin, stdout, post):
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        out = forward(line, post)
        if out is not None:
            stdout.write(out + "\n")
            stdout.flush()


def make_post(config, http=None):
    """Return a post callable that resolves discovery/healthz lazily, on
    first use and again after any transport failure.

    A shim started before the daemon (a logon race) used to resolve discovery
    once at startup and stay permanently broken if that first check failed.
    Now each `post()` call re-checks discovery when it has no cached base URL,
    so a shim that starts ahead of the daemon heals itself once the daemon
    comes up.

    A shim that OUTLIVES a daemon needs more than that. `service.port` is 0, so
    every daemon start takes a fresh ephemeral port, and a shim holding the old
    one is pointed at an address nothing answers. Healing on the next call left
    THIS call to fail, so a restart surfaced one raw `[WinError 10061]` to
    whichever tool happened to be called first - reported, reasonably, as that
    tool being broken while the others served. So when the failed address came
    from the cache and the failure proves the request never left (`_never_sent`),
    re-discover and send it once more. Exactly once: see that predicate.

    Args:
        config: Mainframe config dict
        http: Optional fake HTTP client for testing (default: httpx.Client)

    Returns:
        A callable that POSTs to the daemon's /mcp endpoint, or raises
        ConnectionError if the daemon is not running or the healthz check
        fails.
    """
    state = {"base": None, "headers": None}
    token = config["service"].get("token")

    def _http():
        nonlocal http
        if http is None:
            import httpx
            http = httpx.Client(timeout=600)
        return http

    def _discover():
        # ONE liveness check, shared with the CLI: a daemon.json alone
        # proves nothing — the port may have been recycled by another service.
        info = probe(config, _http())
        if not info:
            raise ConnectionError(_NOT_RUNNING)

        mcp_headers = dict(_HEADERS)
        if token:
            mcp_headers["Authorization"] = f"Bearer {token}"
        state["base"] = f"http://127.0.0.1:{info['port']}/mcp"
        state["headers"] = mcp_headers

    def _send(body):
        try:
            r = _http().post(state["base"], content=json.dumps(body), headers=state["headers"])
            r.raise_for_status()
            return r.json() if r.content else None
        except Exception:
            state["base"] = None      # an address stays cached only while it works
            raise

    def post(body):
        cached = state["base"] is not None
        if not cached:
            _discover()
        try:
            return _send(body)
        except Exception as e:
            if not (cached and _never_sent(e)):
                raise
        # The address came from the cache and the request never left this
        # process: the daemon moved. Find it and send the request once more,
        # so the move costs the caller nothing rather than one raw socket error.
        _discover()
        return _send(body)

    return post


def main():
    # Reconfigure stdio for UTF-8 on Windows (prevents UnicodeEncodeError on em-dash, etc.)
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")

    config = load_config()
    post = make_post(config)
    run(sys.stdin, sys.stdout, post)


if __name__ == "__main__":
    main()
