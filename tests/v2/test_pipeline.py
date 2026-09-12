import os
import shutil
from pathlib import Path

import pytest

from mainframe.core.paths import canonical
from mainframe.core.pipeline import IndexPipeline
from mainframe.memory.lanes import Lanes
from mainframe.service.events import EventStore
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
