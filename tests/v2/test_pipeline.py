import os
import shutil
from pathlib import Path

import pytest

from mainframe.core.paths import canonical
from mainframe.core.pipeline import IndexPipeline
from mainframe.memory.lanes import Lanes
from mainframe.service.events import EventStore
from v2.conftest import FakeEmbedder
from v2.fakes import fake_models
from v2.helpers import write_md


def _pipe(cfg, store, tmp_path):
    return IndexPipeline(Lanes(cfg), store, fake_models(cfg), cfg, EventStore(tmp_path / "ev.db"))


def _raises_scan(*_a, **_kw):
    raise RuntimeError("scan boom")


def test_index_new_modified_deleted_unchanged_emptied(cfg, store, tmp_path):
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    a = write_md(r / "docs" / "a.md", "# a\n\nalpha text here.\n")
    b = write_md(r / "docs" / "b.md", "# b\n\nbeta text here.\n")
    pipe = _pipe(cfg, store, tmp_path)
    rep = pipe.index([str(a), str(b), str(a)])               # duplicate raw path
    assert rep.files == 2 and rep.upserted_rows == 2 and rep.changed and rep.batches == 1
    assert store.ledger().keys() == {canonical(a), canonical(b)}
    assert pipe.index([str(a)]).unchanged == 1
    a.write_text("# a\n\nalpha text CHANGED.\n", encoding="utf-8")
    b.unlink()
    rep = pipe.index([str(a), str(b)])
    assert rep.upserted_rows == 1 and rep.deleted_docs == 1 and store.ledger().keys() == {canonical(a)}
    a.write_text("   \n", encoding="utf-8")                   # emptied: rows go, and that IS a change
    rep = pipe.index([str(a)])
    assert rep.emptied == 1 and rep.empty == 1 and rep.changed and store.ledger() == {}
    assert pipe.empty_paths == {canonical(a)}
    assert pipe.last_run["emptied"] == 1


def test_index_ignores_unresolvable_and_reports_failures(cfg, store, tmp_path):
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    outside = write_md(r / "src" / "notes.md", "# not knowledge")
    empty = write_md(r / "docs" / "e.md", "  \n")
    pipe = _pipe(cfg, store, tmp_path)
    rep = pipe.index([str(outside), str(empty), str(r / "docs" / "missing.md")])
    assert rep.files == 1 and rep.empty == 1 and rep.upserted_rows == 0 and not rep.changed
    assert store.count_rows() == 0 and pipe.empty_paths == {canonical(empty)}
    # a never-indexed empty file must never write to the store — not even
    # a no-op delete — so repeated scans cannot grow LanceDB version history.
    v = store.index_health()["versions"]
    pipe.index([str(empty)])
    pipe.index([str(empty)])
    assert store.index_health()["versions"] == v
    assert store.count_rows() == 0


def test_failures_are_attributed_to_their_batch(cfg, store, tmp_path, monkeypatch):
    # Three files, one failure in the MIDDLE batch — this discriminates a
    # per-batch slice (rep.failed[failed_before:]) from the full rep.failed
    # list, which a two-file case (failure only in the last batch) cannot.
    cfg["index"]["batch_cap"] = 1
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    a = write_md(r / "docs" / "a.md", "# a\n\nalpha.\n")
    b = write_md(r / "docs" / "b.md", "# b\n\nbeta.\n")
    c = write_md(r / "docs" / "c.md", "# c\n\ngamma.\n")
    ev = EventStore(tmp_path / "ev.db")
    pipe = IndexPipeline(Lanes(cfg), store, fake_models(cfg), cfg, ev)
    import mainframe.core.pipeline as pmod
    real = pmod.prepare_document

    def flaky(lf, *args, **kw):
        if lf.path.endswith("b.md"):
            from mainframe.core.indexer import DocFailure
            return DocFailure(doc_path=lf.path, error="boom")
        return real(lf, *args, **kw)
    monkeypatch.setattr(pmod, "prepare_document", flaky)
    rep = pipe.index([str(a), str(b), str(c)])
    assert rep.upserted_rows == 2 and rep.failed == [{"file": canonical(b), "error": "boom"}]
    assert rep.as_dict()["failed_count"] == 1 and canonical(b) not in store.ledger()
    batches = [e["detail"]["failed"] for e in reversed(ev.recent(10, kind="index.batch"))]
    assert batches == [[], [canonical(b)], []]                   # attributed to batch 2 only


def test_batches_respect_cap_for_files_and_deletions(cfg, store, tmp_path):
    cfg["index"]["batch_cap"] = 2
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    files = [write_md(r / "docs" / f"{i}.md", f"# {i}\n\ntext {i}.\n") for i in range(5)]
    ev = EventStore(tmp_path / "ev.db")
    pipe = IndexPipeline(Lanes(cfg), store, fake_models(cfg), cfg, ev)
    rep = pipe.index([str(f) for f in files])
    assert rep.batches == 3 and rep.upserted_rows == 5 and ev.count("index.batch") == 3
    for f in files:
        f.unlink()
    rep = pipe.index([str(f) for f in files])
    assert rep.deleted_docs == 5 and rep.batches == 3 and store.count_rows() == 0
    with pytest.raises(ValueError):
        cfg["index"]["batch_cap"] = 0
        IndexPipeline(Lanes(cfg), store, fake_models(cfg), cfg, ev)


def test_rescan_project_scope_and_rebuild(cfg, store, tmp_path):
    ra, rb = Path(cfg["paths"]["repos_dir"]) / "a", Path(cfg["paths"]["repos_dir"]) / "b"
    write_md(ra / "docs" / "a.md", "# a\n\nalpha.\n")
    write_md(rb / "docs" / "b.md", "# b\n\nbeta.\n")
    pipe = _pipe(cfg, store, tmp_path)
    assert pipe.rescan(project="a").upserted_rows == 1 and store.count_rows() == 1
    assert pipe.rescan().upserted_rows == 1 and store.count_rows() == 2
    (ra / "docs" / "a.md").unlink()
    # pruning is scoped to the requested project — a's vanished doc must
    # NOT be pruned by a rescan restricted to project b.
    rep_b = pipe.rescan(project="b")
    assert rep_b.upserted_rows == 0 and store.count_rows() == 2
    rep = pipe.rescan()
    assert rep.deleted_docs == 1 and store.count_rows() == 1
    assert pipe.rebuild().upserted_rows == 1 and store.count_rows() == 1


def test_ledger_is_scanned_once_and_maintained_in_memory(cfg, store, tmp_path, monkeypatch):
    """The daemon caches the ledger and updates it after each batch.
    index() and rescan() each loaded it, then _run() loaded it AGAIN — two full
    projected scans of the whole table per job, including the one-file job
    every capture fires."""
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    a = write_md(r / "docs" / "a.md", "# a\n\nalpha.\n")
    pipe = _pipe(cfg, store, tmp_path)
    calls, real = {"n": 0}, store.ledger

    def counting():
        calls["n"] += 1
        return real()
    monkeypatch.setattr(store, "ledger", counting)

    assert pipe.index([str(a)]).upserted_rows == 1
    assert pipe.index([str(a)]).unchanged == 1          # the CACHED ledger still knows the hash
    assert pipe.rescan().unchanged == 1
    assert calls["n"] == 1

    write_md(r / "docs" / "b.md", "# b\n\nbeta.\n")
    assert pipe.rebuild().upserted_rows == 2            # drop() resets the cache to {}
    assert calls["n"] == 1
    monkeypatch.undo()
    assert sorted(Path(p).name for p in store.ledger()) == ["a.md", "b.md"]


def test_rescan_prunes_docs_that_left_the_scope(cfg, store, tmp_path):
    """`vanished` was computed purely from file non-existence, so narrowing
    include_projects / widening exclude_projects left the dropped project's
    rows in the index forever — searchable, while status.scope reported the
    new narrower scope."""
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# a\n\nalpha.\n")
    write_md(Path(cfg["paths"]["repos_dir"]) / "q" / "docs" / "b.md", "# b\n\nbeta.\n")
    assert _pipe(cfg, store, tmp_path).rescan().upserted_rows == 2

    cfg["paths"]["exclude_projects"] = ["p"]
    scoped = _pipe(cfg, store, tmp_path)                # a fresh Lanes with the new scope
    rep = scoped.rescan(project="q")
    assert rep.deleted_docs == 0 and store.count_rows() == 2   # a scoped rescan stays in its prefix
    rep = scoped.rescan()
    assert rep.deleted_docs == 1
    assert [Path(p).name for p in store.ledger()] == ["b.md"]


def test_a_linked_dir_cannot_put_one_doc_in_both_todo_and_vanished(cfg, store, tmp_path):
    """The churn `Lanes._contained`'s classification fix prevents: a document
    that `knowledge_files()` yields but `resolve()` rejects lands in `todo`
    AND in `vanished`, so every rescan upserts and deletes it and calls
    `store.optimize()` — unbounded version growth from a stable corpus."""
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    write_md(r / "src" / "x.md", "# not knowledge\n")
    write_md(r / "docs" / "real.md", "# knowledge\n")
    link = r / "docs" / "vendor"
    try:
        import _winapi
        _winapi.CreateJunction(str(r / "src"), str(link))
    except (ImportError, AttributeError, OSError):
        try:
            os.symlink(r / "src", link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("cannot create a directory link here")

    pipe = _pipe(cfg, store, tmp_path)
    assert pipe.rescan().upserted_rows == 1
    versions = store.index_health()["versions"]
    for _ in range(3):
        rep = pipe.rescan()                      # an unchanged corpus, three times over
        assert rep.deleted_docs == 0 and rep.upserted_rows == 0 and rep.unchanged == 1
    assert store.count_rows() == 1
    assert store.index_health()["versions"] == versions      # nothing was written at all


def test_a_failed_ledger_scan_prunes_nothing_and_repairs_itself(cfg, store, tmp_path, monkeypatch):
    """`Store.ledger` swallowed every exception and returned {} — the same
    answer as a genuinely empty index — and the pipeline then CACHED that for
    the life of the process, so no ghost row could ever be pruned again.

    A failed scan now raises, the pipeline refuses to prune against a ledger it
    could not read, and the next run tries again."""
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    a = write_md(r / "docs" / "a.md", "# a\n\nalpha.\n")
    b = write_md(r / "docs" / "b.md", "# b\n\nbeta.\n")
    ev = EventStore(tmp_path / "ev.db")
    pipe = IndexPipeline(Lanes(cfg), store, fake_models(cfg), cfg, ev)
    assert pipe.rescan().upserted_rows == 2

    fresh = IndexPipeline(Lanes(cfg), store, fake_models(cfg), cfg, ev)   # a restart: no cached ledger
    monkeypatch.setattr(store, "_scan", _raises_scan)
    with pytest.raises(RuntimeError, match="scan boom"):
        store.ledger()
    assert fresh._ledger_view() == {} and fresh._ledger is None      # the failure is NOT cached
    assert fresh.ledger_degraded is True

    b.unlink()
    rep = fresh.rescan()
    assert rep.deleted_docs == 0 and rep.prune_skipped is True
    assert store.count_rows() == 2, "a ledger we could not read must not delete rows"
    assert ev.recent(1, kind="index.rescan")[0]["detail"]["prune_skipped"] is True

    monkeypatch.undo()
    rep = fresh.rescan()
    assert fresh.ledger_degraded is False and rep.prune_skipped is False
    assert rep.deleted_docs == 1 and [Path(p).name for p in store.ledger()] == [Path(a).name]


class _RecordingModels:
    """A `Models` stand-in that records every `invoke()` call instead of
    loading anything: `calls` collects the (role, operation) pairs it was
    given, in order. `invoke` still runs the operation against `embedder` so
    the pipeline sees real embeddings and behaves normally around the fake."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.calls = []

    def invoke(self, role, operation):
        self.calls.append((role, operation))
        return operation(self.embedder)


def test_unchanged_document_never_takes_the_model_lock(cfg, store, tmp_path):
    """The rescan win: `known_hash` is checked before any embedding call, so
    an unchanged file must never reach `Models.invoke` — today (pre-fix) it
    is invoked once per file regardless of whether anything changed."""
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    a = write_md(r / "docs" / "a.md", "# a\n\nalpha text here.\n")
    assert _pipe(cfg, store, tmp_path).index([str(a)]).upserted_rows == 1

    recording = _RecordingModels(FakeEmbedder(dim=64))
    pipe = IndexPipeline(Lanes(cfg), store, recording, cfg, EventStore(tmp_path / "ev2.db"))
    rep = pipe.index([str(a)])

    assert rep.unchanged == 1
    assert recording.calls == []


def test_only_the_embedding_runs_under_the_lock(cfg, store, tmp_path):
    """The locked operation is the embed call and nothing else: chunking and
    contextualizing happen OUTSIDE `Models.invoke`, and what invoke receives,
    run against any embedder, returns exactly that embedder's vectors for the
    texts it embedded."""
    cfg["contextual"]["enabled"] = True
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    a = write_md(r / "docs" / "a.md", "# a\n\nalpha text here.\n")
    log = []

    class LoggingContextualizer:
        enabled = True

        def contextualize(self, doc, chunks):
            log.append("contextualize")
            return [""] * len(chunks)

    class _LoggingRecordingModels(_RecordingModels):
        def invoke(self, role, operation):
            log.append("invoke")
            return super().invoke(role, operation)

    class SpyEmbedder(FakeEmbedder):
        def __init__(self, dim=64):
            super().__init__(dim)
            self.embed_calls = []

        def embed(self, texts, batch_size=None):
            self.embed_calls.append(list(texts))
            return super().embed(texts, batch_size)

    recording = _LoggingRecordingModels(FakeEmbedder(dim=64))
    pipe = IndexPipeline(Lanes(cfg), store, recording, cfg, EventStore(tmp_path / "ev2.db"))
    pipe._contextualizer = LoggingContextualizer()

    rep = pipe.index([str(a)])

    assert rep.upserted_rows == 1
    assert log == ["contextualize", "invoke"]
    assert len(recording.calls) == 1
    role, operation = recording.calls[0]
    assert role == "embedder"

    spy = SpyEmbedder(dim=64)
    result = operation(spy)
    assert spy.embed_calls, "the locked operation must call embedder.embed(texts)"
    assert result == spy.embed(spy.embed_calls[0])   # a pure fn of texts: re-embedding matches


def _flaky_models(cfg, exc, fail_on):
    """A registry whose FIRST embedder raises `exc` once, from the embed of the
    chunk containing `fail_on`. Only that instance is sick, so RELOADING is the
    only thing that can make the batch finish — which is exactly the claim
    under test."""
    from mainframe.core.models import Models
    from v2.conftest import FakeEmbedder, FakeReranker

    loads = []

    class Flaky(FakeEmbedder):
        def __init__(self, sick):
            super().__init__(dim=64)
            self.sick = sick

        def embed(self, texts, batch_size=None):
            if self.sick and any(fail_on in t for t in texts):
                self.sick = False
                raise exc
            return super().embed(texts, batch_size)

    def _load(_cfg):
        loads.append(1)
        return Flaky(sick=len(loads) == 1)

    return Models(cfg, embedder_factory=_load,
                  reranker_factory=lambda c: FakeReranker(top_k=3)), loads


def test_a_device_fault_mid_batch_reloads_and_the_batch_completes(cfg, store, tmp_path):
    """THE claim this branch makes: models heal at the point of failure.

    It was false on the index path. `prepare_document` turned every exception
    into a `DocFailure`, so a dead CUDA context never reached `Models.invoke`,
    nothing reloaded, and the job SUCCEEDED with every file marked failed —
    after `take_paths` had already consumed them. The daemon then indexed
    nothing until a human read `rep.failed`."""
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    a = write_md(r / "docs" / "a.md", "# a\n\nalpha text here.\n")
    b = write_md(r / "docs" / "b.md", "# b\n\nbeta text here.\n")
    models, loads = _flaky_models(cfg, RuntimeError("CUDA error: unknown error"), "alpha")
    pipe = IndexPipeline(Lanes(cfg), store, models, cfg, EventStore(tmp_path / "ev.db"))

    rep = pipe.index([str(a), str(b)])
    assert len(loads) == 2, "the fault never reached invoke, so nothing reloaded"
    assert rep.failed == [] and rep.upserted_rows == 2
    assert store.ledger().keys() == {canonical(a), canonical(b)}


def test_an_ordinary_embed_failure_still_only_costs_its_own_document(cfg, store, tmp_path):
    """The other direction: a per-document fault must NOT reload the model or
    fail the batch — its neighbours are still indexed and it alone is reported."""
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    a = write_md(r / "docs" / "a.md", "# a\n\nalpha text here.\n")
    b = write_md(r / "docs" / "b.md", "# b\n\nbeta text here.\n")
    models, loads = _flaky_models(cfg, ValueError("token id out of range"), "alpha")
    pipe = IndexPipeline(Lanes(cfg), store, models, cfg, EventStore(tmp_path / "ev.db"))

    rep = pipe.index([str(a), str(b)])
    assert len(loads) == 1, "an ordinary document error cost a two-minute model reload"
    assert [f["file"] for f in rep.failed] == [canonical(a)]
    assert "token id out of range" in rep.failed[0]["error"]
    assert rep.upserted_rows == 1 and store.ledger().keys() == {canonical(b)}


def test_a_model_load_failure_fails_the_job_not_the_document(cfg, store, tmp_path):
    """`Models._get` failing means the registry could not resolve a model at
    ALL — that is a whole-run problem, not one document's, exactly like a
    device fault that survives reload. It must propagate out of `index()` (so
    `Mainframe.submit`'s retry/requeue machinery sees it and restores the
    paths) rather than becoming a per-document `DocFailure` that lets the run
    "succeed" while silently marking every file failed and erasing the
    ledger's evidence that anything went wrong.

    This is the scenario `tests/v2/test_daemon.py::test_healthz_never_touches_
    models` exercises through the daemon; this test exercises it directly
    through the pipeline, including the OOM case `is_device_failure` is
    documented to exclude — the one case a reload genuinely cannot fix."""
    from mainframe.core.indexer import EmbedderUnavailable
    from mainframe.core.models import Models
    from v2.conftest import FakeReranker

    r = Path(cfg["paths"]["repos_dir"]) / "p"
    a = write_md(r / "docs" / "a.md", "# a\n\nalpha text here.\n")

    for message in ("no weights on disk", "CUDA out of memory"):
        def _dead_factory(_cfg, message=message):
            raise RuntimeError(message)
        models = Models(cfg, embedder_factory=_dead_factory, reranker_factory=lambda c: FakeReranker(top_k=3))
        pipe = IndexPipeline(Lanes(cfg), store, models, cfg, EventStore(tmp_path / "ev.db"))

        with pytest.raises(EmbedderUnavailable, match=message):
            pipe.index([str(a)])
        assert store.ledger() == {}, "a load failure must not look like a successfully-processed file"


def test_rescan_skips_prune_when_root_is_missing(cfg, store, tmp_path):
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    write_md(r / "docs" / "a.md", "# a\n\nalpha.\n")
    ev = EventStore(tmp_path / "ev.db")
    pipe = IndexPipeline(Lanes(cfg), store, fake_models(cfg), cfg, ev)
    pipe.rescan()
    assert store.count_rows() == 1
    shutil.rmtree(cfg["paths"]["repos_dir"])
    pipe.rescan()
    assert store.count_rows() == 1                            # an absent root must not wipe the index
    events = ev.recent(1, kind="index.rescan")
    assert events[0]["detail"]["prune_skipped"] is True


def _delete_when_read(monkeypatch, target):
    """Make the file disappear after the pipeline has resolved its lane."""
    original = Path.read_text
    target = canonical(target)

    def read_then_delete(path, *args, **kwargs):
        if canonical(path) == target:
            path.unlink()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_then_delete)


def test_index_deletion_during_prepare_removes_an_indexed_document(cfg, store, tmp_path, monkeypatch):
    """Exercise the resolve → read TOCTOU through IndexPipeline, not a fake result."""
    path = write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "race.md", "# race\n\ncontent\n")
    pipe = _pipe(cfg, store, tmp_path)
    assert pipe.index([str(path)]).upserted_rows == 1

    _delete_when_read(monkeypatch, path)
    report = pipe.index([str(path)])

    assert report.failed == [] and report.deleted_docs == 1
    assert store.count_rows() == 0 and store.ledger() == {}


def test_index_deletion_during_prepare_skips_an_unindexed_document(cfg, store, tmp_path, monkeypatch):
    path = write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "race.md", "# race\n\ncontent\n")
    pipe = _pipe(cfg, store, tmp_path)

    _delete_when_read(monkeypatch, path)
    report = pipe.index([str(path)])

    assert report.failed == [] and report.deleted_docs == 0 and report.upserted_rows == 0
    assert store.count_rows() == 0


def test_index_keeps_an_indexed_document_on_a_real_read_failure(cfg, store, tmp_path, monkeypatch):
    path = write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "locked.md", "# locked\n\ncontent\n")
    pipe = _pipe(cfg, store, tmp_path)
    assert pipe.index([str(path)]).upserted_rows == 1
    original = Path.read_text

    def denied(read_path, *args, **kwargs):
        if canonical(read_path) == canonical(path):
            raise PermissionError("synthetic denied read")
        return original(read_path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    report = pipe.index([str(path)])

    assert report.deleted_docs == 0 and len(report.failed) == 1
    assert "synthetic denied read" in report.failed[0]["error"]
    assert store.count_rows() == 1


def test_vanished_prepare_does_not_prune_when_ledger_is_degraded(cfg, store, tmp_path, monkeypatch):
    path = write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "race.md", "# race\n\ncontent\n")
    assert _pipe(cfg, store, tmp_path).index([str(path)]).upserted_rows == 1
    pipe = _pipe(cfg, store, tmp_path)  # no cached ledger: make its scan fail
    monkeypatch.setattr(store, "_scan", _raises_scan)
    _delete_when_read(monkeypatch, path)

    report = pipe.index([str(path)])

    assert report.prune_skipped is True and report.deleted_docs == 0
    assert store.count_rows() == 1


def test_vanished_prepare_does_not_prune_when_root_disappears_after_enumeration(cfg, store, tmp_path, monkeypatch):
    """A populated ledger must still honor rescan's root-unavailable guard."""
    repos = Path(cfg["paths"]["repos_dir"])
    path = write_md(repos / "p" / "docs" / "race.md", "# race\n\ncontent\n")
    assert _pipe(cfg, store, tmp_path).index([str(path)]).upserted_rows == 1
    pipe = _pipe(cfg, store, tmp_path)
    original = pipe.lanes.all_files

    def enumerate_then_remove_root():
        files = original()
        shutil.rmtree(repos)
        return files

    monkeypatch.setattr(pipe.lanes, "all_files", enumerate_then_remove_root)
    report = pipe.rescan()

    assert report.prune_skipped is True and report.deleted_docs == 0
    assert store.count_rows() == 1


# ---- a rescan heals capture rows that predate their published kind ----

_DECISION = "---\ntype: decision\nproject: p\n---\n# T\n\n## Summary\n{body}\n"


def _kinds(store) -> dict:
    df = store._scan(["doc_path", "source_type"])
    return {p: set(df[df["doc_path"] == p]["source_type"]) for p in set(df["doc_path"])}


def _session_era_capture(cfg, store, tmp_path, monkeypatch, body="healme body text here."):
    """Index a `type: decision` capture while nothing is published, so its rows say `session`."""
    cap = write_md(Lanes(cfg).capture_dir("p") / "2026-01-01-000000-ab12cd34-ef56ab78.md",
                   _DECISION.format(body=body))
    with monkeypatch.context() as m:
        m.setattr("mainframe.core.classify.PUBLISHED_KINDS", frozenset())
        assert _pipe(cfg, store, tmp_path).index([str(cap)]).upserted_rows > 0
    assert _kinds(store) == {canonical(cap): {"session"}}
    return cap


def test_a_rescan_reindexes_a_capture_whose_rows_predate_its_kind(cfg, store, tmp_path, monkeypatch):
    cap = _session_era_capture(cfg, store, tmp_path, monkeypatch)
    calls = []
    real = FakeEmbedder.embed

    def counting(self, texts, *a, **kw):
        calls.append(list(texts))
        return real(self, texts, *a, **kw)
    monkeypatch.setattr(FakeEmbedder, "embed", counting)
    pipe = _pipe(cfg, store, tmp_path)
    rep = pipe.rescan()
    assert rep.unchanged == 0 and rep.upserted_rows > 0 and not rep.failed
    assert _kinds(store) == {canonical(cap): {"decision"}}
    assert len(calls) == 1
    keys = list(store._scan(["chunk_key"])["chunk_key"])
    assert len(keys) == len(set(keys)) == store.count_rows()
    assert pipe.rescan().unchanged == 1 and len(calls) == 1          # healed: nothing embeds again


def test_session_capture_paths_lists_only_session_capture_rows(cfg, store, tmp_path, monkeypatch):
    """Pin: no table and an emptied table answer [], a populated one only its session captures."""
    assert store.session_capture_paths() == []
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    doc = write_md(r / "docs" / "k.md", "# k\n\nknowledge text.\n")
    pipe = _pipe(cfg, store, tmp_path)
    pipe.index([str(doc)])
    doc.write_text("   \n", encoding="utf-8")                  # rows go, the table stays
    pipe.index([str(doc)])
    assert store.count_rows() == 0 and store.session_capture_paths() == []
    legacy = _session_era_capture(cfg, store, tmp_path, monkeypatch)
    published = write_md(Lanes(cfg).capture_dir("p") / "2026-01-02-000000-ab12cd34-ef56ab79.md",
                         _DECISION.format(body="already published body."))
    doc.write_text("# k\n\nknowledge text again.\n", encoding="utf-8")
    pipe.index([str(published), str(doc)])
    assert _kinds(store)[canonical(published)] == {"decision"}
    assert store.session_capture_paths() == [canonical(legacy)]


def test_a_watcher_run_does_not_heal(cfg, store, tmp_path, monkeypatch):
    """Pin: only the periodic rescan reclassifies; index() trusts the ledger."""
    cap = _session_era_capture(cfg, store, tmp_path, monkeypatch)
    other = write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "o.md", "# o\n\nunrelated text.\n")
    pipe = _pipe(cfg, store, tmp_path)
    assert pipe.index([str(other)]).upserted_rows == 1
    assert pipe.index([str(cap)]).unchanged == 1
    assert _kinds(store)[canonical(cap)] == {"session"} and canonical(cap) in store.ledger()


def test_a_failed_reclassification_scan_forces_nothing_and_prunes_on(cfg, store, tmp_path, monkeypatch):
    cap = _session_era_capture(cfg, store, tmp_path, monkeypatch)       # a legacy `session` row set
    r = Path(cfg["paths"]["repos_dir"]) / "p"
    gone = write_md(r / "docs" / "gone.md", "# gone\n\nvanishing text.\n")
    write_md(r / "docs" / "kept.md", "# kept\n\nstays put.\n")
    pipe = _pipe(cfg, store, tmp_path)
    with monkeypatch.context() as m:
        m.setattr(store, "session_capture_paths", _raises_scan)
        assert pipe.rescan().upserted_rows == 2                          # the docs; the capture is not forced
        gone.unlink()
        calls = []
        real = FakeEmbedder.embed

        def counting(self, texts, *a, **kw):
            calls.append(list(texts))
            return real(self, texts, *a, **kw)
        m.setattr(FakeEmbedder, "embed", counting)
        rep = pipe.rescan()
        assert rep.prune_skipped is False and rep.deleted_docs == 1 and not rep.failed
        assert calls == [] and _kinds(store)[canonical(cap)] == {"session"}       # forced nothing
        assert pipe.ledger_degraded is False
        assert set(store.ledger()) == {canonical(r / "docs" / "kept.md"), canonical(cap)}
    rep = pipe.rescan()                                                  # scan restored: the next rescan heals
    assert not rep.failed and _kinds(store)[canonical(cap)] == {"decision"}


def test_a_forced_doc_that_fails_keeps_its_rows_and_heals_next_rescan(cfg, store, tmp_path, monkeypatch):
    cap = _session_era_capture(cfg, store, tmp_path, monkeypatch, body="flaky body text here.")
    real = FakeEmbedder.embed
    armed = [True]

    def flaky(self, texts, *a, **kw):
        if armed and any("flaky body" in t for t in texts):
            armed.clear()                                          # raise once, scoped to this doc's text
            raise RuntimeError("synthetic embed failure")
        return real(self, texts, *a, **kw)
    monkeypatch.setattr(FakeEmbedder, "embed", flaky)
    pipe = _pipe(cfg, store, tmp_path)
    rep = pipe.rescan()
    assert [f["file"] for f in rep.failed] == [canonical(cap)] and "synthetic embed failure" in rep.failed[0]["error"]
    assert _kinds(store) == {canonical(cap): {"session"}} and canonical(cap) in store.ledger()
    rep = pipe.rescan()
    assert not rep.failed and _kinds(store) == {canonical(cap): {"decision"}}


# ---- retraction: a capture moved into quarantine/ leaves the index ----

def _retractable(cfg, store, tmp_path, project="p"):
    """A published capture plus a knowledge doc, indexed; returns (pipe, capture, doc)."""
    cap = write_md(Lanes(cfg).capture_dir(project) / "2026-01-01-000000-ab12cd34-ef56ab78.md",
                   _DECISION.format(body="retract body text here."))
    doc = write_md(Path(cfg["paths"]["repos_dir"]) / project / "docs" / "k.md", "# k\n\nknowledge text.\n")
    pipe = _pipe(cfg, store, tmp_path)
    assert pipe.rescan().failed == []
    assert canonical(cap) in pipe._ledger_view() and canonical(doc) in pipe._ledger_view()
    return pipe, cap, doc


def _retract(lanes, cap, project="p"):
    q = lanes.quarantine_dir(project)
    q.mkdir(parents=True, exist_ok=True)
    return Path(shutil.move(str(cap), str(q / cap.name)))


def _search_paths(store, fake_embedder, include_captures):
    store.optimize()
    res = store.search(fake_embedder.embed_query("retract body text"), top_k=10, include_captures=include_captures,
                       query_text="retract body text")
    return {r["doc_path"] for r in res}


def test_a_project_rescan_prunes_the_projects_retracted_capture(cfg, store, fake_embedder, tmp_path):
    pipe, cap, doc = _retractable(cfg, store, tmp_path)
    assert canonical(cap) in _search_paths(store, fake_embedder, False)         # published: a default search finds it
    _retract(pipe.lanes, cap)
    rep = pipe.rescan("p")
    assert rep.deleted_docs == 1
    assert set(store.ledger()) == {canonical(doc)}
    for include in (False, True):
        assert canonical(cap) not in _search_paths(store, fake_embedder, include)


def test_a_rescan_of_another_project_leaves_a_retracted_capture_alone(cfg, store, tmp_path):
    """Pin: isolation holds before and after the prune covers captures."""
    pipe, cap, doc = _retractable(cfg, store, tmp_path)
    _retract(pipe.lanes, cap)
    assert pipe.rescan("q").deleted_docs == 0
    assert {canonical(cap), canonical(doc)} <= set(store.ledger())


def test_indexing_the_old_path_of_a_moved_capture_deletes_its_rows(cfg, store, tmp_path):
    """Pin: the watcher delivers a move's old path; index() sees it gone and deletes."""
    pipe, cap, doc = _retractable(cfg, store, tmp_path)
    _retract(pipe.lanes, cap)
    assert pipe.index([str(cap)]).deleted_docs == 1
    assert set(store.ledger()) == {canonical(doc)}


def test_a_missing_repos_dir_leaves_a_retracted_capture_until_it_returns(cfg, store, tmp_path):
    """Pin, documents a limit: rescan prunes nothing without repos_dir, though capture
    pruning does not depend on it (public #79)."""
    pipe, cap, doc = _retractable(cfg, store, tmp_path)
    _retract(pipe.lanes, cap)
    shutil.rmtree(cfg["paths"]["repos_dir"])
    rep = pipe.rescan("p")
    assert rep.prune_skipped is True and canonical(cap) in store.ledger()


def test_a_rescan_of_another_project_does_not_heal_this_projects_capture(cfg, store, tmp_path, monkeypatch):
    """Pin: _reclassify's project prefix keeps rescan("q") from re-embedding p's capture."""
    cap = _session_era_capture(cfg, store, tmp_path, monkeypatch)
    calls = []
    real = FakeEmbedder.embed

    def counting(self, texts, *a, **kw):
        calls.append(list(texts))
        return real(self, texts, *a, **kw)
    monkeypatch.setattr(FakeEmbedder, "embed", counting)
    pipe = _pipe(cfg, store, tmp_path)
    pipe.rescan("q")
    assert calls == [] and _kinds(store) == {canonical(cap): {"session"}}
    pipe.rescan("p")
    assert len(calls) == 1 and _kinds(store) == {canonical(cap): {"decision"}}
