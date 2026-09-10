"""MCP adapter: two FastMCP instances over ONE Mainframe —
`/mcp` (search, capture, status, maintain) and `/mcp/ro` (search, status) —
stateless Streamable HTTP with JSON responses. Tools are async and offload
blocking work to a thread so /healthz and /shutdown stay responsive."""

import asyncio
import contextlib
from urllib.parse import urlsplit

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount

from mainframe.app import JOBS
from mainframe.core.store import IndexStaleError

READ_ONLY = ToolAnnotations(readOnlyHint=True)
_MAINTAIN_GUIDANCE = (f"action: one of {', '.join(JOBS)}; "
                      "project: a plain repo name (no separators, no leading dot)")


def _register(mcp: FastMCP, app, read_only: bool):
    @mcp.tool(name="search", annotations=READ_ONLY,
              description=("Search the Mainframe (hybrid vector+keyword, reranked). Use SPECIFIC technical "
                           "terms; trust rerank_score (higher = better). response_format: concise (default) "
                           "or detailed. Session captures are excluded unless include_sessions=true."))
    async def search(query: str, limit: int = 3, include_sessions: bool = False, response_format: str = "concise") -> dict:
        return await asyncio.to_thread(app.search, query, limit, include_sessions, response_format)

    if not read_only:
        @mcp.tool(name="capture",
                  description=("Write one immutable, secret-scrubbed session note into the capture lane "
                               "(indexed automatically; excluded from default search). Author dense markdown "
                               "sections: ## Summary, ## Decisions, ## Action items, ## Key facts."))
        async def capture(content: str, title: str, project: str | None = None, session_id: str = "",
                          kind: str = "session", cwd: str = "") -> dict:
            try:
                return await asyncio.to_thread(app.capture, content, title, project, session_id, kind, cwd,
                                               "capture_memory")
            except ValueError as e:
                return {"error": str(e), "guidance": "project must be a plain name; kind is session|decision|note"}

    @mcp.tool(name="status", annotations=READ_ONLY,
              description="Daemon health: models, VRAM, index rows/versions/indexes, pending captures, recent events.")
    async def status() -> dict:
        return await asyncio.to_thread(app.status)

    if not read_only:
        @mcp.tool(name="maintain",
                  description=("Run maintenance now: index/rescan (reconcile the lanes; optional project), "
                               "optimize (compact), reload (drop the resident models so the next use "
                               "reloads them — for a GPU fault that survives a retry). A full rebuild is "
                               "offline only: `mainframe down` then `mainframe rebuild`."))
        async def maintain(action: str, project: str | None = None) -> dict:
            try:
                return await asyncio.to_thread(app.maintain, action, project)
            except ValueError as e:
                return {"error": str(e), "guidance": _MAINTAIN_GUIDANCE}


def _server(name: str) -> FastMCP:
    return FastMCP(name, stateless_http=True, json_response=True, streamable_http_path="/")


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


class _LocalRequestsOnly:
    """Guard REST and MCP together, before either dispatches a request.

    Literal loopback Hosts prevent DNS rebinding. Browser Origins must also
    match the request's origin: a cross-site simple POST can otherwise reach
    a valid loopback Host without a CORS preflight. Native clients omit Origin.
    Neither check replaces bearer authentication against other local clients.
    """

    def __init__(self, app, allowed=LOOPBACK_HOSTS):
        self.app, self.allowed = app, frozenset(allowed)

    @staticmethod
    def _hostname(scope) -> str:
        raw = ""
        for key, value in scope.get("headers", ()):
            if key == b"host":
                raw = value.decode("latin-1").strip()
                break
        name, _, port = raw.rpartition(":")
        return name if name and port.isdigit() else raw       # ':' inside an IPv6 literal is not a port

    @staticmethod
    def _origin(value):
        try:
            parsed = urlsplit(value)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None
                    or parsed.path or parsed.query or parsed.fragment):
                return None
            port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
            return parsed.scheme, parsed.hostname, port
        except ValueError:
            return None

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self._hostname(scope).lower() not in self.allowed:
            await JSONResponse({"error": "invalid Host header",
                                "guidance": "the daemon serves 127.0.0.1 only"},
                               status_code=421)(scope, receive, send)
            return
        if scope["type"] == "http":
            headers = scope.get("headers", ())
            origins = [v.decode("latin-1") for k, v in headers if k == b"origin"]
            host = next((v.decode("latin-1") for k, v in headers if k == b"host"), "")
            expected = self._origin(f"{scope['scheme']}://{host}")
            if origins and (len(origins) != 1 or expected is None or self._origin(origins[0]) != expected):
                await JSONResponse({"error": "invalid Origin header",
                                    "guidance": "browser requests must use the daemon's own origin"},
                                   status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


class _BareMountPaths:
    """`/mcp` and `/mcp/ro` (no trailing slash) are what users configure; Starlette's
    Mount only matches with the slash and would otherwise redirect a POST."""
    def __init__(self, app, paths):
        self.app, self.paths = app, set(paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") in self.paths:
            scope = dict(scope, path=scope["path"] + "/", raw_path=(scope["path"] + "/").encode())
        await self.app(scope, receive, send)


def build_mcp(app):
    full, ro = _server("mainframe"), _server("mainframe-ro")
    _register(full, app, read_only=False)
    _register(ro, app, read_only=True)
    return full, ro


def _unhandled(_request, exc: Exception) -> JSONResponse:
    """Anything a handler did not translate, as the documented error shape.

    A stale index is a 409: the request was well formed and the server is
    healthy, but its state conflicts with the settings now configured, and the
    remedy is a specific operator action. Without this the carefully written
    "stop the daemon and run `mainframe rebuild`" reached only the log while
    REST answered an opaque 500. A model still in its load backoff says
    "unavailable" and is a 503 (retry later); everything else is a 500."""
    if isinstance(exc, IndexStaleError):
        return JSONResponse({"error": str(exc), "guidance": "`mainframe down`, then `mainframe rebuild`"},
                            status_code=409)
    status = 503 if "unavailable" in str(exc).lower() else 500
    return JSONResponse({"error": str(exc), "guidance": "check `mainframe status` and daemon.log"},
                        status_code=status)


def mount(full: FastMCP, ro: FastMCP, extra_routes=None, on_startup=None, on_shutdown=None) -> Starlette:
    """Parent app: `/mcp/ro` before `/mcp`; both session managers and the
    daemon's start/stop hooks run in ONE lifespan (mounted sub-app lifespans
    do not run on their own)."""
    full_app, ro_app = full.streamable_http_app(), ro.streamable_http_app()

    @contextlib.asynccontextmanager
    async def lifespan(_):
        async with full.session_manager.run():
            async with ro.session_manager.run():
                # on_startup INSIDE the try: a half-started daemon (queue up,
                # watcher not) must still get its on_shutdown, or the mutation
                # thread and the discovery file outlive the failed launch.
                try:
                    if on_startup:
                        await asyncio.to_thread(on_startup)
                    yield
                finally:
                    if on_shutdown:
                        await asyncio.to_thread(on_shutdown)

    routes = list(extra_routes or []) + [Mount("/mcp/ro", app=ro_app), Mount("/mcp", app=full_app)]
    # Translate failures at the transport boundary. The REST handlers
    # translate the errors they expect; this catches everything else — above
    # all `RuntimeError: <role> unavailable: ...; retry in Ns` from the model
    # backoff, which is the most likely 500 in production and which the CLI
    # would otherwise report as "could not reach the daemon". Starlette runs
    # this from ServerErrorMiddleware, so the JSON goes out and the exception
    # is still re-raised for the server's own log. (The MCP mounts are already
    # safe: FastMCP turns a tool exception into an isError result.)
    app = Starlette(routes=routes, lifespan=lifespan, exception_handlers={Exception: _unhandled})
    # Starlette's Mount pattern for e.g. "/mcp/ro" only matches a request path
    # that ALREADY has a trailing "/mcp/ro/..." — a bare "/mcp/ro" (what a
    # client actually configures) would otherwise either 404 (silently
    # swallowed by the shorter "/mcp" mount's greedy subpath match) or, for
    # bare "/mcp" itself, cost a 307 via Starlette's redirect_slashes fallback
    # — and every MCP request is a POST that the Python MCP SDK's httpx
    # client does NOT redirect by default. Normalizing the two bare paths in
    # the ASGI scope before routing serves both with no redirect at all.
    app.add_middleware(_BareMountPaths, paths=("/mcp", "/mcp/ro"))
    # Added last, so it runs before any route, mount or path rewrite sees the
    # request. Not outermost overall: `Daemon.build()` adds TokenMiddleware
    # after this and Starlette inserts at index 0, so with a token configured
    # an unauthorized request off a rebound host answers 401 before 421 — the
    # right order anyway.
    app.add_middleware(_LocalRequestsOnly)
    return app
