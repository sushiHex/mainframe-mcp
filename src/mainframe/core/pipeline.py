"""The index loop's work unit: paths → canonical de-dup →
resolve lanes → prepare rows → ONE merge per batch of ≤ batch_cap files (and
≤ batch_cap deletions) → optimize once → events. Vanished paths the ledger
still holds are deleted. Thread-agnostic: the MutationQueue provides exclusion."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from mainframe.core.indexer import UNCHANGED, DocFailure, prepare_document
from mainframe.core.paths import canonical, doc_id

logger = logging.getLogger(__name__)


@dataclass
class IndexRunResult:
    """One index()/rescan()/rebuild() call's outcome.

    `empty` = files that currently chunk to nothing (including emptied ones).
    `emptied` (subset of `empty`) = those that previously had rows in the
    store — a genuine change, since their stale rows are deleted. A file that
    has ALWAYS been empty is counted in `empty` but is never written to the
    store, so it never bumps `emptied` or `changed` (a never-indexed
    empty file must not write a delete on every scan)."""

    files: int = 0
    upserted_rows: int = 0
    deleted_docs: int = 0
    emptied: int = 0
    unchanged: int = 0
    empty: int = 0
    failed: list = field(default_factory=list)
    batches: int = 0
    seconds: float = 0.0
    prune_skipped: bool = False    # this run deleted nothing, deliberately (see _run)

    @property
    def changed(self) -> bool:
        return self.upserted_rows > 0 or self.deleted_docs > 0 or self.emptied > 0

    def as_dict(self) -> dict:
        return {"files": self.files, "upserted_rows": self.upserted_rows, "deleted_docs": self.deleted_docs,
                "emptied": self.emptied, "unchanged": self.unchanged, "empty": self.empty,
                "failed": self.failed[:10], "failed_count": len(self.failed), "batches": self.batches,
                "seconds": round(self.seconds, 1), "prune_skipped": self.prune_skipped}


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class IndexPipeline:
    def __init__(self, lanes, store, models, config: dict, events):
        self.lanes, self.store, self.models, self.config, self.events = lanes, store, models, config, events
        self.chunk_cfg = config["chunker"]
        self.batch_cap = int(config["index"].get("batch_cap", 200))
        if self.batch_cap < 1:
            raise ValueError("index.batch_cap must be >= 1")
        self.empty_paths: set = set()
        self.last_run = None
        self.doc_count = None      # set by _run(); status() uses it to avoid a ledger scan per call
        self._contextualizer = None
        self._ledger: dict | None = None   # {doc_path: file_hash}; see _ledger_view()
        self.ledger_degraded = False       # True while the scan is failing: nothing may be pruned

    def _ctx(self):
        if not self.config.get("contextual", {}).get("enabled"):
            return None
        if self._contextualizer is None:
            from mainframe.core.contextual import Contextualizer
            self._contextualizer = Contextualizer(self.config)
        return self._contextualizer

    def _ledger_view(self) -> dict:
        """THE ledger: derived from a projected scan the first time
        it is needed, then maintained in memory by `_run_batch` — every write
        goes through the single mutation thread, so this dict is authoritative
        between batches. It is rebuilt from the scan when the process restarts
        and reset to {} by `rebuild()`. There is no manifest file.

        A FAILED scan is not an empty index. The failure is never cached,
        so the next run retries it; until one succeeds `ledger_degraded` stands
        and `_run` prunes nothing. Caching `{}` here — which is what swallowing
        the exception amounted to — made every ghost row unprunable for the
        life of the process."""
        if self._ledger is None:
            try:
                self._ledger = self.store.ledger()
            except Exception as e:
                logger.warning(f"ledger unavailable ({e}); this run will index but prune nothing")
                self.ledger_degraded = True
                return {}
            self.ledger_degraded = False
        return self._ledger

    # ---- public ----

    def index(self, paths) -> IndexRunResult:
        ledger = self._ledger_view()
        todo, deleted = [], []
        for c in dict.fromkeys(canonical(p) for p in paths):      # canonical FIRST, then de-dup
            if not Path(c).exists():
                if c in ledger:
                    deleted.append(c)
                self.empty_paths.discard(c)
                continue
            lf = self.lanes.resolve(c)
            if lf is not None:
                todo.append(lf)
        return self._run(todo, deleted, "index")

    def rescan(self, project: str | None = None) -> IndexRunResult:
        """Reconcile the lanes with the ledger. A known doc is `vanished` when
        it no longer RESOLVES — gone from disk, or still there but no longer a
        lane member because the scope changed. Without the second half,
        narrowing `include_projects` left the dropped project's rows searchable
        forever and only a full `rebuild` could reconcile them."""
        lanefiles = [lf for lf in self.lanes.all_files() if project is None or lf.project == project]
        known = self._ledger_view()
        prune_skipped = not self.lanes.repos_dir.exists()
        if prune_skipped:
            logger.warning(f"rescan: repos_dir {self.lanes.repos_dir} is unavailable; skipping prune "
                           "so an absent root cannot wipe the index")
            vanished = []
        elif project is not None:
            prefix = canonical(self.lanes.repos_dir / project) + "/"
            vanished = [p for p in known if p.startswith(prefix) and self.lanes.resolve(p) is None]
        else:
            vanished = [p for p in known if self.lanes.resolve(p) is None]
        rep = self._run(lanefiles, vanished, "rescan", prune_skipped=prune_skipped)
        self.events.emit("index.rescan", project=project, **rep.as_dict())
        return rep

    def rebuild(self) -> IndexRunResult:
        self.store.drop()
        self._ledger = {}                  # the table is gone: the cache is authoritative again
        self.ledger_degraded = False
        self.empty_paths.clear()
        rep = self.rescan()
        self.last_run["kind"] = "rebuild"
        self.events.emit("index.rebuild", **rep.as_dict())
        return rep

    # ---- internals ----

    def _run(self, todo: list, deleted: list, kind: str, prune_skipped: bool = False) -> IndexRunResult:
        """The shared batching loop for index()/rescan(): merge per ≤batch_cap
        group of files and of deletions, optimize once at the end if anything
        changed, emit index.missing when the chunk_key index still isn't
        there. A mid-run upsert_batch exception leaves earlier batches already
        committed; the in-memory ledger was updated only for the batches that
        actually landed, so the next call picks up where it left off — the
        caller (Mainframe.submit) is responsible for requeuing any paths that
        never got attempted."""
        # BEFORE anything is embedded. The store refuses a stale write at the
        # merge, which is the invariant; asking here as well is what keeps a
        # drifted index from costing a full GPU pass over the batch every
        # debounce window for a write that cannot land until a rebuild.
        self.store.refuse_if_stale("indexing")
        t0 = time.monotonic()
        started_at = datetime.now()
        rep = IndexRunResult()
        ledger = self._ledger_view()
        if prune_skipped or self.ledger_degraded:
            # Never delete against a ledger we could not read, or while the
            # root the deletions were derived from is unavailable: both make
            # "absent" indistinguishable from "unreadable".
            rep.prune_skipped = True
            deleted = []
        todo = sorted(todo, key=lambda f: f.path)
        deleted = sorted(set(deleted))
        rep.files = len(todo)
        file_batches = list(_chunks(todo, self.batch_cap))
        del_batches = list(_chunks(deleted, self.batch_cap))
        for i in range(max(len(file_batches), len(del_batches))):
            chunk = file_batches[i] if i < len(file_batches) else []
            dels = del_batches[i] if i < len(del_batches) else []
            if not chunk and not dels:
                continue
            self._run_batch(chunk, dels, ledger, rep)
            rep.batches += 1
        if rep.changed:
            ok = self.store.optimize()
            health = self.store.index_health()
            self.events.emit("optimize.done" if ok else "optimize.failed", **health)
            if not any(list(i["columns"]) == ["chunk_key"] for i in health["indexes"]):
                self.events.emit("index.missing", column="chunk_key")
        rep.seconds = time.monotonic() - t0
        self.last_run = {"kind": kind, "at": started_at.isoformat(timespec="seconds"), **rep.as_dict()}
        if not self.ledger_degraded:
            self.doc_count = len(ledger)  # the in-memory ledger is authoritative after a run
        return rep

    def _run_batch(self, chunk, deleted_paths, ledger, rep: IndexRunResult):
        failed_before = len(rep.failed)
        docs = []
        ctx = self._ctx()
        for lf in chunk:
            out = self.models.invoke("embedder", lambda e, lf=lf: prepare_document(
                lf, self.chunk_cfg, e, known_hash=ledger.get(lf.path), contextualizer=ctx))
            if out is UNCHANGED:
                rep.unchanged += 1
            elif isinstance(out, DocFailure):
                rep.failed.append({"file": out.doc_path, "error": out.error})
            elif out.rows:
                self.empty_paths.discard(lf.path)
                docs.append(out)
            else:
                # Chunks to nothing. Only a genuine emptying (the store still
                # holds old rows for this doc) is worth a write — a file that
                # was NEVER indexed must not issue a delete every scan.
                rep.empty += 1
                self.empty_paths.add(lf.path)
                if lf.path in ledger:
                    rep.emptied += 1
                    docs.append(out)
        deleted_ids = [doc_id(p) for p in deleted_paths]
        upserted = 0
        if docs or deleted_ids:
            res = self.store.upsert_batch(docs, deleted_ids)
            upserted = res.upserted_rows
            rep.upserted_rows += upserted
            rep.deleted_docs += res.deleted_docs
            for d in docs:
                if d.rows:
                    ledger[d.doc_path] = d.rows[0]["file_hash"]
                else:
                    ledger.pop(d.doc_path, None)
            for p in deleted_paths:
                ledger.pop(p, None)
        self.events.emit("index.batch", files=len(chunk), upserted_rows=upserted, deleted_docs=len(deleted_ids),
                         failed=[f["file"] for f in rep.failed[failed_before:]])
