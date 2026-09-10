"""Composition root: the one object every adapter talks to. Reads
(search/status) run on the caller's thread.

Every write that needs MUTUAL EXCLUSION enters through `submit`, which is
queue-aware: inline when no daemon queue is attached or when already on the
writer thread, enqueued exactly once otherwise — so no caller can deadlock the
writer by re-entering it. `capture` deliberately does not: it exclusive-creates
an immutable file, which collides with nothing, and the watcher carries it into
the index like any other file."""

import os
import sys
from datetime import datetime
from pathlib import Path

from mainframe import __version__ as VERSION
from mainframe.core.models import Models
from mainframe.core.pipeline import IndexPipeline
from mainframe.core.search import SearchService
from mainframe.core.secrets import scrub_secrets
from mainframe.core.store import INDEX_DIRNAME, IndexStaleError, Store
from mainframe.memory.capture import write_capture
from mainframe.memory.lanes import Lanes, capture_id_of, safe_name
from mainframe.service.events import EventStore

# `rebuild` is deliberately absent: it drops the table, so it runs OFFLINE only
# (`cli.run_offline_rebuild`, under `.daemon.lock`), never as a daemon job while
# readers are being served.
JOBS = ("index", "rescan", "optimize", "reload")


def _worth_retrying(exc) -> bool:
    """Should what this job consumed be given back for another attempt?

    The undo is for TRANSIENT faults — a CUDA blip, a LanceDB hiccup, a file
    being written. `IndexStaleError` is permanent: neither re-indexing the same
    paths nor re-running a rescan can clear it before an offline rebuild, so
    restoring the trigger would re-attempt forever. A rebuild re-enumerates
    everything anyway, so nothing is lost by letting it go."""
    return exc is not None and not isinstance(exc, IndexStaleError)


class Mainframe:
    def __init__(self, config: dict, *, models=None, events=None, store=None):
        self.config = config
        self.mainframe_dir = Path(config["paths"]["mainframe_dir"])
        self.mainframe_dir.mkdir(parents=True, exist_ok=True)
        self.lanes = Lanes(config)
        self.lanes.ensure_dirs()
        self.events = events or EventStore(self.mainframe_dir / "events.db")
        self.store = store or Store(db_path=self.mainframe_dir / INDEX_DIRNAME, embedding_dim=None,
                                    fingerprint=Store.fingerprint_from_config(config),
                                    **Store.search_kwargs_from_config(config))
        self.models = models or Models(config, events=self.events)
        self.pipeline = IndexPipeline(self.lanes, self.store, self.models, config, self.events)
        # Built once: SearchService holds the model REGISTRY, not the models, so
        # nothing here needs rebuilding when a model is reloaded underneath it.
        self.searcher = SearchService(self.models, self.store, config)
        self.mutations = None        # attached by the daemon
        self.invalidation = None     # attached by the daemon
        self.nonce = None            # set by the daemon
        self.started_at = datetime.now()

    @classmethod
    def from_config(cls, config: dict) -> "Mainframe":
        return cls(config)

    # ---- reads ----

    def search(self, query: str, limit: int = 3, include_sessions: bool = False,
               response_format: str = "concise") -> dict:
        return self.searcher.search(query, limit=limit, include_sessions=include_sessions,
                                    response_format=response_format)

    def status(self) -> dict:
        vram = None
        torch = sys.modules.get("torch")           # never import torch here: status must not init CUDA
        try:
            if torch is not None and torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                vram = {"free_gb": round(free / 1e9, 2), "total_gb": round(total / 1e9, 2)}
        except Exception:
            pass
        pending = {}
        for proj in sorted({f.project for f in self.lanes.capture_files()}):
            n = len(self.lanes.pending(proj))
            if n:
                pending[proj] = n
        # `status` runs on a reader thread. Installing the count it computes
        # would let a stale read overwrite a newer one the writer thread just
        # set (`pipeline.doc_count` belongs to `_run`), so the scan pays for
        # this response only. `None` when the scan itself is unavailable —
        # status must degrade, never raise.
        docs = self.pipeline.doc_count
        if docs is None:
            try:
                docs = len(self.store.ledger())
            except Exception:
                docs = None
        return {
            "daemon": {"pid": os.getpid(), "version": VERSION, "nonce": self.nonce,
                       "uptime": int((datetime.now() - self.started_at).total_seconds()),
                       "queue_depth": self.mutations.depth if self.mutations is not None else None,
                       "running": self.mutations.running if self.mutations is not None else None},
            "models": self.models.state(),
            "vram": vram,
            "index": {**self.store.index_health(), "docs": docs,
                      "config_drift": self.store.config_drift,
                      "rebuild_required": bool(self.store.config_drift),
                      "empty_files": len(self.pipeline.empty_paths), "last_run": self.pipeline.last_run},
            "memory": {"mode": "index-and-search only",      # Consolidation is not implemented in v2.
                       "pending": pending, "drafts": []},
            "scope": {"include": self.config["paths"].get("include_projects", []),
                      "exclude": self.config["paths"].get("exclude_projects", []),
                      "skipped_projects": getattr(self.lanes, "skipped_projects", 0)},
            "recent_events": self.events.recent(20),
        }

    # ---- writes ----

    def capture(self, content: str, title: str, project=None, session_id: str = "",
                kind: str = "session", cwd: str = "", captured_by: str = "capture_memory") -> dict:
        # `captured_by` is provenance: which SURFACE wrote this
        # capture — the MCP tool, the CLI, or REST — not a constant. Scrubbed
        # ONCE here so the file and the event carry the same vetted string: the
        # capture file was scrubbed and the event was not.
        captured_by = scrub_secrets(captured_by)
        path = write_capture(self.lanes, content, title, project, session_id=session_id, kind=kind, cwd=cwd,
                             captured_by=captured_by)
        proj = path.parent.name
        self.events.emit("capture.written", project=proj, path=str(path), captured_by=captured_by)
        # Nothing is submitted here. The watcher already owns file-to-index
        # propagation and the capture root is one of its watch roots, so this
        # file reaches the index through the ordinary debounce, with the
        # periodic rescan as the backstop. The cost is honestly bounded by that
        # backstop, not by the debounce: with the observer dead (which the
        # repair backoff can hold for up to an hour) a capture waits out
        # `index.rescan_hours`. Acceptable — a capture is a note just written,
        # not a query anyone is blocked on — and worth one propagation path.
        return {"file": str(path), "project": proj, "capture_id": capture_id_of(path)}

    def maintain(self, action: str, project: str | None = None, wait: bool = True):
        """THE maintenance entry point, shared by the MCP `maintain` tool and
        REST `/jobs/{kind}` — the action list and the `index` -> `rescan` alias
        used to be written three times. `project` is validated here:
        it reaches a `repos_dir` join and a delete predicate in
        `IndexPipeline.rescan`. Raises ValueError for either rejection; each
        adapter renders that in its own error shape."""
        if action not in JOBS:
            raise ValueError(f"unknown action {action!r}; expected one of {', '.join(JOBS)}")
        if project is not None:
            project = safe_name(project)
        if action in ("index", "rescan"):
            return self.submit("rescan", project, wait=wait)
        return self.submit(action, wait=wait)

    def _job(self, kind: str):
        if kind == "index":
            return lambda paths: self.pipeline.index(paths).as_dict()
        if kind == "rescan":
            return lambda project=None: self.pipeline.rescan(project).as_dict()
        if kind == "optimize":
            def _opt():
                ok = self.store.optimize()
                self.events.emit("optimize.done" if ok else "optimize.failed", **self.store.index_health())
                return {"ok": ok}
            return _opt
        if kind == "reload":
            def _reload():
                """THE deliberate escape hatch. `Models.invoke` heals a device
                failure it can RECOGNIZE; this is what an operator reaches for
                when one escapes that predicate — a post-wake CUDA context
                failing in a shape we do not match — instead of restarting the
                daemon. Without it the only recovery was a restart, and the
                deleted clock-driven `resume` covered that case by luck.

                On the writer thread like every other job, so a reload cannot
                land in the middle of an index batch. `invalidate_all` includes
                a role stuck in its load backoff — the state an operator most
                needs cleared after a GPU fault at load time."""
                return {"reloaded": self.models.invalidate_all()}
            return _reload
        raise ValueError(f"unknown job {kind!r}; expected one of {JOBS}")

    def submit(self, kind: str, *args, wait: bool = True, timeout=None,
               on_retryable_failure=None, on_success=None):
        """Run a mutation and its completion action as one job.

        Completion actions take no arguments and run on the executing thread,
        before the result or exception is published. This applies to queued,
        inline, and reentrant calls. The caller owns restoring consumed work;
        `_worth_retrying` excludes permanent failures from that restoration.

        `wait` and `timeout` control only how the caller observes the job. A
        timed-out wait leaves the job and its completion action intact; a
        TimeoutError raised BY the job is handled like any other failure.

        Pass materialized arguments, since queued work consumes them on another
        thread. Queue admission errors propagate without running completion
        actions: no job was accepted, so the submitting caller owns recovery.
        """
        fn = self._job(kind)

        def execute():
            try:
                result = fn(*args)
            except BaseException as exc:
                if on_retryable_failure is not None and _worth_retrying(exc):
                    on_retryable_failure()
                raise
            if on_success is not None:
                on_success()
            return result

        q = self.mutations
        if q is None or q.is_worker_thread():
            return execute()
        fut = q.submit(kind, execute)
        return fut.result(timeout=timeout) if wait else fut
