"""mainframed lifecycle: lock → open store → bind → discovery →
components → serve. Nothing touches the LanceDB directory before the lock is
held; discovery is written only after the socket is bound; models never load
here. A second instance exits 3."""

import contextlib
import logging
import os
import secrets
import socket
import sys
from pathlib import Path

from filelock import FileLock, Timeout

from mainframe import __version__
from mainframe.adapters.mcp_server import build_mcp, mount
from mainframe.app import Mainframe
from mainframe.config import hf_offline_if_cached, load_config
from mainframe.service.api import TokenMiddleware, routes
# Re-exported: discovery lives in its own module so the CLI and the stdio shim
# can find a daemon without importing the ASGI/MCP stack.
from mainframe.service.discovery import discovery_path, read_discovery, write_discovery  # noqa: F401
from mainframe.service.mutations import MutationQueue
from mainframe.service.scheduler import Scheduler
from mainframe.service.watcher import Invalidation, Watcher

logger = logging.getLogger(__name__)
LOCK_HELD_EXIT = 3
SLOW_STOP_WARN_SECONDS = 30.0    # how often a shutdown that is waiting out its writer says so
_LOGGING_CONFIGURED = False      # handlers are process-wide: attach them exactly once


def lock_path(config) -> Path:
    return Path(config["paths"]["mainframe_dir"]) / ".daemon.lock"


class Daemon:
    def __init__(self, config: dict, app: Mainframe | None = None, port: int | None = None):
        self.config = config
        self._app = app
        self.port = port if port is not None else int(config["service"].get("port", 7433))
        self.nonce = secrets.token_hex(8)
        Path(config["paths"]["mainframe_dir"]).mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(lock_path(config)))
        self._server = None
        self._started = False
        self.mutations = self.invalidation = self.watcher = self.scheduler = None

    # ---- lock (before anything else is opened) ----
    def acquire(self) -> bool:
        try:
            self._lock.acquire(timeout=0)
            return True
        except Timeout:
            return False

    def release(self):
        """THE invariant, enforced where it is actually true: this process
        never voluntarily releases `.daemon.lock` while its writer thread is
        alive. This is the one release site — `serve()`'s `finally` calls it on
        every path out, including a raise partway through `_stop` — so the
        wait belongs here and not in the orderly-shutdown path, which any
        exception can skip.

        A stop timeout was a report, not a cancellation, and there is nothing
        to cancel: a merge cannot be interrupted safely. So the only two ways
        the lock can be freed are (a) returning from here, after the worker is
        gone, and (b) this process dying, which the OS answers by releasing the
        lock at exactly the instant the thread dies with it. Either way there
        is never a second live writer opening LanceDB behind a thread that is
        still mid-merge. A long shutdown is correct; an operator who
        cannot wait kills the process.

        The queue is asked to stop first (idempotent — `_stop` normally sent
        that already): the wait would otherwise hang forever on a worker that
        is merely idle, and letting go of the lock while the writer still
        accepts work is the very thing this refuses to do.

        `filelock` is thread-local by default, so `is_locked` is only true on
        the thread that acquired — which is the thread `serve()` releases on."""
        if not self._lock.is_locked:
            return
        if self.mutations is not None:
            self.mutations.stop(timeout=0)
            self._await_writer()
        self._lock.release()

    @property
    def app(self) -> Mainframe:
        if self._app is None:
            self._app = Mainframe(self.config)          # opens LanceDB + events: only after the lock
        return self._app

    def open(self):
        return self.app

    # ---- components ----
    def _components(self):
        app = self.app
        self.mutations = MutationQueue(app.events)
        self.invalidation = Invalidation()
        # `watch_roots` is passed as a CALLABLE so every (re)start re-enumerates
        # it, and both predicates are pure-string lane tests: a delete or
        # the source half of a rename must be recorded even though the
        # path is already gone, and every event must be rejected before any
        # syscall. `dir_matters` keeps a directory event from flagging a
        # full rescan unless it could actually hold lane files.
        self.watcher = Watcher(app.lanes.watch_roots, self.invalidation,
                               is_relevant=app.lanes.could_belong,
                               is_relevant_dir=app.lanes.dir_matters)
        self.scheduler = Scheduler(app, self.invalidation, self.watcher, self.mutations, self.config)
        app.mutations, app.invalidation, app.nonce = self.mutations, self.invalidation, self.nonce

    def _start(self):
        app = self.app
        self._components()
        self.mutations.start()
        try:
            self.watcher.start()
        except Exception as e:      # a dead watcher must not stop the daemon: the scheduler restarts it
            logger.warning(f"watcher failed to start: {e}")
        self.scheduler.start()
        app.events.emit("daemon.start", pid=os.getpid(), port=self.port, version=__version__, nonce=self.nonce)
        app.submit("rescan", wait=False)                   # reconcile what changed while we were down
        if self.config["service"].get("prewarm"):
            app.models.prewarm(background=True)
        self._started = True        # from here the lifespan owns the discovery file's removal

    def _stop(self):
        # Tolerates components that never started: `mount()` runs on_shutdown
        # even when on_startup raised half-way through.
        if self.scheduler:
            self.scheduler.stop()
        if self.watcher:
            self.watcher.stop()
        clean = self.mutations.stop(
            timeout=float(self.config["service"].get("shutdown_timeout_seconds", 300))) if self.mutations else True
        app = self._app
        if app is not None:
            # The queue stays ATTACHED: a stopped MutationQueue rejects
            # with QueueStopped, while detaching it would make a late write run
            # INLINE on a foreign thread — the opposite of what `submit`
            # guarantees. A stopped queue reports depth 0 / running None.
            with contextlib.suppress(Exception):
                app.events.emit("daemon.stop", pid=os.getpid(), clean=clean)
        if not clean:
            # The lock is held until `release()` anyway; waiting here as well
            # keeps the ORDER right — discovery must not be withdrawn while
            # this daemon can still be writing.
            self._await_writer()
        self._remove_own_discovery()

    def _await_writer(self):
        """Block until the mutation worker is gone. Silent when it already is,
        so `release()` can call it unconditionally; loud when it is not,
        because the wait is unbounded on purpose (see `release`)."""
        if self.mutations is None or self.mutations.join(timeout=0):
            return
        logger.error("mutation worker is still running; holding %s until it exits",
                     lock_path(self.config))
        waited = 0.0
        while not self.mutations.join(timeout=SLOW_STOP_WARN_SECONDS):
            waited += SLOW_STOP_WARN_SECONDS
            logger.warning("still waiting for the mutation worker after %.0fs", waited)

    def _remove_own_discovery(self):
        """Unpublish daemon.json, but ONLY while it is still ours: a newer
        instance may already have written its own, and a straggler must never
        delete a live daemon's discovery file."""
        mine = read_discovery(self.config)
        if mine and mine.get("nonce") == self.nonce:
            with contextlib.suppress(OSError):
                discovery_path(self.config).unlink()

    def stop(self) -> bool:
        """True when a server was actually asked to exit. False (e.g. under
        TestClient, or before serve() built one) means nothing will happen —
        `/shutdown` reports that instead of promising a stop."""
        if self._server is None:
            return False
        self._server.should_exit = True
        return True

    def build(self):
        """The ASGI app; its lifespan starts/stops the components (tests drive it
        with TestClient; production goes through serve()). Built fresh every
        call: a FastMCP session manager can only be entered once."""
        full, ro = build_mcp(self.app)
        base = mount(full, ro, extra_routes=routes(self.app, self.stop),
                     on_startup=self._start, on_shutdown=self._stop)
        token = self.config["service"].get("token")
        if token:
            base.add_middleware(TokenMiddleware, token=token)     # outermost: covers /mcp too
        return base

    def bind(self) -> socket.socket:
        """Bind the loopback socket FIRST, then publish discovery."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if os.name == "nt":
                # On Windows SO_REUSEADDR lets a SECOND process steal a bound
                # address, which is the exact split-brain the lock exists to
                # prevent; SO_EXCLUSIVEADDRUSE is the hardening equivalent.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self.port))
            sock.listen(128)
            self.port = sock.getsockname()[1]
            write_discovery(self.config, self.port, self.nonce)
        except BaseException:
            sock.close()
            raise
        return sock

    def _configure_process(self):
        """Process-wide setup for the instance that WON the lock: a loser must
        exit 3 without touching the env or the logging config. Handlers go on
        the `mainframe` logger, not root, so library noise stays out of
        daemon.log."""
        global _LOGGING_CONFIGURED
        hf_offline_if_cached(self.config)
        if _LOGGING_CONFIGURED:
            return
        log_path = Path(self.config["paths"]["mainframe_dir"]) / "daemon.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        pkg = logging.getLogger("mainframe")
        pkg.setLevel(logging.INFO)
        for handler in (logging.StreamHandler(sys.stderr),
                        logging.FileHandler(log_path, encoding="utf-8", delay=True)):
            handler.setFormatter(fmt)
            pkg.addHandler(handler)
        _LOGGING_CONFIGURED = True

    def serve(self):
        """acquire → configure → open → bind → discovery → uvicorn (components
        start in the ASGI lifespan). Raises SystemExit(3) when another instance
        holds the lock. If the launch dies after bind() but before the lifespan
        ran, the discovery file we just published is withdrawn again."""
        import uvicorn
        if not self.acquire():
            raise SystemExit(LOCK_HELD_EXIT)
        sock = None
        try:
            self._configure_process()
            self.open()
            sock = self.bind()
            app = self.build()
            self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="info"))
            self._server.run(sockets=[sock])
        finally:
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()          # uvicorn closes it too; close() is idempotent
            if not self._started:
                self._remove_own_discovery()
            self.release()


def serve(config: dict | None = None):
    """Process entry point: ONE acquisition path (Daemon.serve owns the lock),
    with a friendly message when a second instance loses the race."""
    config = config or load_config()
    try:
        Daemon(config).serve()
    except SystemExit as e:
        if e.code == LOCK_HELD_EXIT:
            print(f"mainframed: another instance holds {lock_path(config)}", file=sys.stderr)
        raise
