"""Filesystem watcher: one recursive watchdog observer per lane
root records into an Invalidation; the scheduler drains it.
A real ReadDirectoryChangesW overflow surfaces as a dead emitter thread (which
`alive` detects), not as an event. An event on a root path itself signals a
root deletion/move (FILE_ACTION_REMOVED_SELF), not overflow."""

import contextlib
import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class Invalidation:
    """Everything that says "the index is behind, act later", in one place.

    A file event names ONE path. A directory event, an event on a watch root, a
    relevance check that raised, or a gap in the daemon's own tick loop can only
    say "we may have missed events here — reconcile by enumeration". Those were
    two objects with two disciplines for one statement: a lock-synchronized path
    set, and a bare bool written by the watcher thread and cleared by the
    scheduler thread with no synchronization at all.

    A full scan SUBSUMES the path set: re-enumerating the lanes re-indexes every
    changed file and prunes every vanished one, so any path recorded before it
    is already covered. Draining therefore answers either "index these paths" or
    "reconcile everything, because X" — never both."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._paths: set = set()
        self._reasons: set = set()      # a small closed vocabulary: it is a SET, and must stay bounded
        self._last = None
        self._lock = threading.Lock()

    def touch(self, path):
        """One file changed. Debounced — `quiet_for` measures from the last one."""
        with self._lock:
            self._paths.add(str(path))
            self._last = self._clock()

    def reconcile(self, reason: str):
        """A full scan is due, because `reason`. Deliberately NOT debounced:
        every reason means "we may have missed events", and waiting for churn to
        settle does nothing to answer that."""
        with self._lock:
            if reason not in self._reasons:
                logger.info("full reconcile requested: %s", reason)
            self._reasons.add(reason)

    @property
    def full(self) -> bool:
        with self._lock:
            return bool(self._reasons)

    def state(self, now: float) -> tuple:
        """`(pending_paths, quiet_for, full)` under ONE lock. The scheduler
        weighs all three together, and three separate reads can interleave with
        the watcher thread into a view no instant ever had."""
        with self._lock:
            quiet = float("inf") if self._last is None else max(0.0, now - self._last)
            return len(self._paths), quiet, bool(self._reasons)

    def take_paths(self) -> tuple:
        """The pending paths, LEAVING any reconcile request standing.

        An index job does not subsume a full scan (the reverse of `drain`), and
        the watcher thread can record a reason between the scheduler's snapshot
        — which is what chose `index` — and this call. Clearing it here would
        throw away a request nobody acted on: the same written-by-one-thread,
        cleared-by-another loss this class exists to end."""
        with self._lock:
            paths = tuple(sorted(self._paths))
            self._paths.clear()
            self._last = None
            return paths

    def drain(self) -> tuple:
        """The reasons a full scan is due, and reset — paths cleared too,
        because a full scan SUBSUMES them: re-enumerating the lanes re-indexes
        every changed file and prunes every vanished one anyway."""
        with self._lock:
            reasons = tuple(sorted(self._reasons))
            self._paths.clear()
            self._reasons.clear()
            self._last = None
            return reasons


def _default_observer_factory():
    from watchdog.observers import Observer
    return Observer()


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher):
        super().__init__()
        self.watcher = watcher

    def on_any_event(self, event):
        self.watcher.on_event(getattr(event, "src_path", ""), getattr(event, "dest_path", None),
                              bool(getattr(event, "is_directory", False)))


class Watcher:
    """`roots` is a list OR a zero-arg callable. A callable is re-evaluated on
    every `start()`/`restart()`, so a root created after launch — a
    `repos_dir` that did not exist yet, a project that just came into scope —
    is picked up by the next restart instead of never; `self.roots` always
    reflects the last evaluation."""

    def __init__(self, roots, invalidation: Invalidation, is_relevant, observer_factory=None,
                 is_relevant_dir=None):
        self._roots_source = roots
        self.roots = []
        self._root_strs: set = set()
        self._resolve_roots()
        self.invalidation = invalidation
        self.is_relevant = is_relevant
        self.is_relevant_dir = is_relevant_dir or (lambda _p: True)
        self._factory = observer_factory or _default_observer_factory
        self._observer = None

    def _resolve_roots(self) -> list:
        src = self._roots_source
        self.roots = [Path(r) for r in (src() if callable(src) else src)]
        self._root_strs = {str(r) for r in self.roots}
        return self.roots

    def on_event(self, src_path, dest_path=None, is_directory: bool = False):
        try:
            src = str(src_path or "")
            dest = str(dest_path) if dest_path else ""
            if src in self._root_strs or (dest and dest in self._root_strs):
                self.invalidation.reconcile("root_event")   # a deletion/move (FILE_ACTION_REMOVED_SELF)
                return
            if is_directory:
                # Reconcile by enumeration, but ONLY for a directory that could
                # hold lane files: every directory event under ~/repos used to
                # flag this, so an out-of-scope repo's .git/node_modules churn
                # drove a full rescan every tick.
                if self.is_relevant_dir(src) or (dest and self.is_relevant_dir(dest)):
                    self.invalidation.reconcile("directory_event")
                return
            for p in (src, dest):
                if p and self.is_relevant(p):
                    self.invalidation.touch(p)
        except Exception as e:
            logger.warning("watcher event failed (%s); scheduling a rescan", e)
            self.invalidation.reconcile("watcher_error")

    def start(self):
        self._resolve_roots()
        obs = self._factory()
        handler = _Handler(self)
        for root in self.roots:
            obs.schedule(handler, str(root), recursive=True)
        try:
            obs.start()
        except BaseException:
            with contextlib.suppress(Exception):
                obs.stop()
                obs.join(5)
            raise
        self._observer = obs
        logger.info(f"watching {len(self.roots)} root(s)")

    def stop(self):
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(5)
            except Exception as e:
                logger.debug(f"observer stop: {e}")
            self._observer = None

    def restart(self):
        self.stop()
        self.start()

    @property
    def alive(self) -> bool:
        obs = self._observer
        if obs is None or not obs.is_alive():
            return False
        emitters = list(getattr(obs, "emitters", ()))       # watchdog: one EventEmitter thread per root
        if emitters and (len(emitters) < len(self.roots) or not all(e.is_alive() for e in emitters)):
            return False
        return True
