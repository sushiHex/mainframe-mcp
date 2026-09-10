"""Daemon discovery: `<mainframe_dir>/daemon.json` plus THE
liveness probe every client shares.

Kept out of `daemon.py` on purpose: the CLI and the stdio shim only need to
find a running daemon, and importing `daemon.py` would drag the whole ASGI/MCP
stack into a process that just wants to POST one line of JSON-RPC.

A `daemon.json` alone proves nothing — the process may be gone and its port
recycled by an unrelated service — so `probe` also checks `/healthz` and
requires the nonce to match the file. `cli.DaemonClient.alive()` and the shim's
`_discover()` were the same six lines with different error handling.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from mainframe import __version__
from mainframe.memory.atomic import write_atomic


def discovery_path(config) -> Path:
    return Path(config["paths"]["mainframe_dir"]) / "daemon.json"


def write_discovery(config, port: int, nonce: str):
    write_atomic(discovery_path(config), json.dumps({"pid": os.getpid(), "port": port, "nonce": nonce,
                                                     "started_at": datetime.now().isoformat(timespec="seconds"),
                                                     "version": __version__}, indent=2))


def read_discovery(config):
    p = discovery_path(config)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


def probe(config, http=None) -> dict | None:
    """The discovery record of a daemon that is REALLY running, else None.

    Never raises: an unreachable port, a foreign service, a stale nonce and a
    missing file are all simply "not running" to a caller. `http` is any object
    with `.get(url, headers=...)` (the caller's pooled client, or a fake).
    """
    info = read_discovery(config)
    if not info:
        return None
    token = config["service"].get("token")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    client = http
    if client is None:
        import httpx
        client = httpx.Client(timeout=5)
    try:
        r = client.get(f"http://127.0.0.1:{info['port']}/healthz", headers=headers)
        body = r.json()
        if r.status_code == 200 and body.get("ok") and body.get("nonce") == info.get("nonce"):
            return info
    except Exception:
        return None
    return None
