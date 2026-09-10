"""The serialized writer: one FIFO thread, so that operations
needing MUTUAL EXCLUSION never overlap. "One process" is not "one writer" —
this queue is what makes it so.

What it serializes, exactly: every LanceDB mutation, and every
replace/compare-and-swap on a file. What it deliberately does NOT: an immutable
exclusive-create (a capture cannot collide with anything, by construction) and
an append to the WAL-mode event store, which is multi-writer on purpose so
out-of-process hooks can append too. Claiming "every mutation goes through the
queue" was never true and would rule out both.

`is_worker_thread()` lets Mainframe.submit run inline when already on it."""

import logging
import queue
import threading
from concurrent.futures import Future

logger = logging.getLogger(__name__)

_STOP = object()


class QueueStopped(RuntimeError):
    """Raised by `MutationQueue.submit`/`run` when the queue is stopping or
    has stopped — a caller must not enqueue after that, and must not be
    left hanging on a write it is about to lose."""


class MutationQueue:
    def __init__(self, events=None):
        self.events = events
        self._q: "queue.Queue" = queue.Queue()
        self._thread: threading.Thread | None = None
        self.running: str | None = None
        self._stop_sent: bool = False
        self._lock = threading.Lock()

    @property
    def depth(self) -> int:
        return max(0, self._q.qsize() - (1 if self._stop_sent else 0) + (1 if self.running else 0))

    @property
    def accepting(self) -> bool:
        """False before `start()`, and as soon as `stop()` is called (even
        while the worker is still draining) — a caller must not enqueue
        after that, and must not be told a write it is about to lose."""
        return self._thread is not None and not self._stop_sent

    def is_worker_thread(self) -> bool:
        return self._thread is not None and threading.get_ident() == self._thread.ident

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="mutations", daemon=True)
            self._thread.start()

    def stop(self, timeout=None) -> bool:
        """Ask the worker to finish and wait up to `timeout` for it. False means
        it is STILL RUNNING — the caller owns what happens next (see
        `Daemon._stop`), because a job that outlives its stop request is still
        a live writer."""
        if self._thread is None:
            return True
        with self._lock:
            if not self._stop_sent:
                self._q.put(_STOP)
                self._stop_sent = True
        self._thread.join(timeout)
        return self._finish()

    def join(self, timeout=None) -> bool:
        """Wait for the worker to exit WITHOUT asking it to (`stop` already
        did). True when the thread is gone. This is how a shutdown that timed
        out keeps waiting instead of walking away from a live writer."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return self._finish()

    def _finish(self) -> bool:
        """Tear the queue down at the moment the worker is actually gone —
        whichever of `stop`/`join` observes it. Idempotent; False while the
        thread is still alive."""
        thread = self._thread
        if thread is None:
            return True
        if thread.is_alive():
            return False
        # Belt and braces: normally _STOP is enqueued behind every job
        # already submitted, so nothing is left. Fail anything that is
        # (e.g. a caller that bypassed submit()'s lock) so no waiter hangs.
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                continue
            _, _, _, _, fut = item
            fut.set_exception(QueueStopped("mutation queue is stopping"))
        self._thread = None
        self._stop_sent = False
        return True

    def submit(self, name: str, fn, *args, **kwargs) -> Future:
        with self._lock:
            if not self.accepting:
                raise QueueStopped("mutation queue is stopping")
            fut: Future = Future()
            self._q.put((name, fn, args, kwargs, fut))
        return fut

    def run(self, name: str, fn, *args, timeout=None, **kwargs):
        return self.submit(name, fn, *args, **kwargs).result(timeout=timeout)

    def _loop(self):
        while True:
            item = self._q.get()
            if item is _STOP:
                return
            name, fn, args, kwargs, fut = item
            self.running = name
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as e:  # noqa: BLE001 — a job must never kill the writer
                logger.warning(f"job {name} failed: {e}")
                if self.events is not None:
                    try:
                        self.events.emit("job.failed", name=name, error=str(e))
                    except Exception:
                        pass
                fut.set_exception(e)
            finally:
                self.running = None
