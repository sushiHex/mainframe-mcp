"""Local REST for hooks and the CLI: tiny, JSON, loopback.
Every blocking call is offloaded so the event loop stays responsive."""

import asyncio
import json
import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from mainframe import __version__
from mainframe.app import JOBS
from mainframe.service.mutations import QueueStopped


def _bad(error: str, guidance: str, status: int = 400):
    return JSONResponse({"error": error, "guidance": guidance}, status_code=status)


async def _body(request: Request):
    """Parse the JSON body, or return a 400 response instead of a 500."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return None, _bad("body is not valid JSON", "send a JSON object of keyword arguments")
    if not isinstance(body, dict):
        return None, _bad("body must be a JSON object", "send a JSON object of keyword arguments")
    return body, None


def routes(app, shutdown) -> list:
    async def healthz(_: Request):
        # Never touches the models: `state()` reads the registry's bookkeeping,
        # it does not load anything.
        return JSONResponse({"ok": True, "pid": os.getpid(), "version": __version__, "nonce": app.nonce,
                             "models": app.models.state(),
                             "queue_depth": app.mutations.depth if app.mutations else None})

    async def status(_: Request):
        return JSONResponse(await asyncio.to_thread(app.status))

    async def search(request: Request):
        body, err = await _body(request)
        if err is not None:
            return err
        try:
            return JSONResponse(await asyncio.to_thread(lambda: app.search(**body)))
        except (TypeError, ValueError) as e:
            return _bad(str(e), "body keys: query, limit, include_sessions, response_format")

    async def capture(request: Request):
        body, err = await _body(request)
        if err is not None:
            return err
        # Provenance is observed, not accepted: `captured_by` records which
        # SURFACE wrote the capture, and this is that surface. `setdefault` let
        # a caller's own value through into the event log, where /events
        # returns it verbatim.
        body["captured_by"] = "rest"
        try:
            return JSONResponse(await asyncio.to_thread(lambda: app.capture(**body)))
        except (TypeError, ValueError) as e:
            return _bad(str(e), "body keys: content, title, project, session_id, kind, cwd")

    async def jobs(request: Request):
        # One maintenance surface: `Mainframe.maintain` owns the action list,
        # the `index` -> `rescan` alias (the queue's `index` job takes explicit
        # paths, which no REST caller has) and the `project` validation.
        kind = request.path_params["kind"]
        project = request.query_params.get("project") or None
        wait = request.query_params.get("wait", "1") != "0"
        try:
            out = await asyncio.to_thread(app.maintain, kind, project, wait)
            return JSONResponse({"queued": True} if not wait else out)
        except ValueError as e:
            return _bad(str(e), f"job: one of {', '.join(JOBS)}; ?project= a plain repo name")
        except QueueStopped as e:
            return _bad(str(e), "the daemon is shutting down; retry against the next instance", status=503)

    async def events(request: Request):
        raw = request.query_params.get("n", "20")
        try:
            n = int(raw)
        except ValueError:
            return _bad(f"n must be an integer, got {raw!r}", "GET /events?n=20&kind=daemon.start")
        # SQLite reads `LIMIT -1` as unlimited, so `?n=-1` dumped the whole
        # unrotated events.db through the response.
        n = max(1, min(1000, n))
        return JSONResponse(await asyncio.to_thread(app.events.recent, n, request.query_params.get("kind")))

    async def shutdown_route(_: Request):
        # False when no server is attached: nothing will actually stop, and
        # saying "stopping" then sends `mainframe down` into a 30 s lock wait.
        return JSONResponse({"stopping": bool(shutdown())})

    return [Route("/healthz", healthz), Route("/status", status), Route("/search", search, methods=["POST"]),
            Route("/capture", capture, methods=["POST"]), Route("/jobs/{kind}", jobs, methods=["POST"]),
            Route("/events", events), Route("/shutdown", shutdown_route, methods=["POST"])]


class TokenMiddleware(BaseHTTPMiddleware):
    """Loopback is not authentication when any local process can reach the port.
    Applied outermost, so it covers `/mcp` and `/mcp/ro` too.

    `BaseHTTPMiddleware` is the discouraged base for anything that streams: it
    is safe here ONLY because both FastMCP instances are built with
    `json_response=True` (see `mcp_server._server`). If that flag is ever
    flipped to SSE, this must become pure ASGI middleware."""

    def __init__(self, app, token: str):
        super().__init__(app)
        self.expected = f"Bearer {token}".encode("utf-8")

    async def dispatch(self, request, call_next):
        # compare_digest, not `!=`: a plain comparison short-circuits on the
        # first differing byte and leaks the token's prefix by timing.
        presented = (request.headers.get("authorization") or "").encode("utf-8")
        if not secrets.compare_digest(presented, self.expected):
            return JSONResponse({"error": "unauthorized",
                                 "guidance": "send Authorization: Bearer <service.token>"}, status_code=401)
        return await call_next(request)
