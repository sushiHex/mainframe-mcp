"""the index says what BUILT it.

One sidecar next to the database records the settings the rows were produced
under (embedder, chunk size, overlap, frontmatter, contextualization). Changing
any of them leaves every existing row untouched while new documents enter a
different regime, and nothing about a row itself records which regime made it.

There is no ownership half any more: v2 keeps its rows in
`<mainframe_dir>/index.lancedb`, v1 keeps `.lancedb`, so no directory is ever
claimed by two builds.
"""
import json

import pytest

from mainframe.core.store import INDEX_DIRNAME, IndexStaleError, Store
from v2.test_store import _rows


def _marker(store) -> dict:
    return json.loads(store._marker_path().read_text(encoding="utf-8"))


def _fp(**over) -> dict:
    base = {"embedder_model": "fake", "chunk_size": 256, "overlap_ratio": 0.35,
            "strip_frontmatter": False, "contextual": False}
    base.update(over)
    return base


def test_creating_a_table_writes_the_marker(tmp_path):
    s = Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp())
    assert _marker(s) == {"format_version": 1, "fingerprint": _fp()}
    assert s.config_drift == {}


def test_an_unmarked_index_takes_todays_fingerprint_as_its_baseline(tmp_path, fake_embedder):
    """An index built before the marker existed records nothing about its rows,
    and an unrecorded past cannot be reported as drift."""
    s = Store(db_path=tmp_path / "db", embedding_dim=64)
    s.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    s._marker_path().unlink()

    reopened = Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp())
    assert reopened.count_rows() == 1
    assert reopened.config_drift == {} and _marker(reopened)["fingerprint"] == _fp()


def test_a_corrupt_marker_re_establishes_the_baseline(tmp_path, fake_embedder):
    s = Store(db_path=tmp_path / "db", embedding_dim=64)
    s.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    s._marker_path().write_text("{not json", encoding="utf-8")
    reopened = Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp())
    assert reopened.count_rows() == 1 and _marker(reopened)["fingerprint"] == _fp()


def test_config_drift_is_reported(tmp_path, fake_embedder):
    s = Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp())
    s.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])

    changed = Store(db_path=tmp_path / "db", embedding_dim=64,
                    fingerprint=_fp(chunk_size=512, strip_frontmatter=True))
    assert changed.count_rows() == 1                       # the rows are untouched
    assert changed.config_drift == {"chunk_size": [256, 512], "strip_frontmatter": [False, True]}
    # the stored marker still describes what BUILT the rows, until a rebuild
    assert _marker(changed)["fingerprint"]["chunk_size"] == 256

    clean = Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp())
    assert clean.config_drift == {}
    # a caller that does not supply a fingerprint (eval harness) skips the check
    assert Store(db_path=tmp_path / "db", embedding_dim=64).config_drift == {}


def test_drift_refuses_indexing(tmp_path, fake_embedder):
    """Drift used to be computed, surfaced and then PERMITTED, so one table
    could hold rows from two chunking regimes while the marker claimed
    otherwise. A write is what mixes them, so a write is refused."""
    s = Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp())
    s.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])

    stale = Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp(chunk_size=512))
    with pytest.raises(IndexStaleError, match="chunk_size") as e:
        stale.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/b.md", ["beta"])])
    assert "rebuild" in str(e.value)                    # the message names the remedy
    assert stale.count_rows() == 1                      # nothing was written

    stale.drop()                                        # the rebuild the error asked for
    assert stale.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/b.md", ["beta"])]).upserted_rows == 1


def test_only_an_embedder_change_refuses_search(tmp_path, fake_embedder):
    """The asymmetry is deliberate: a different embedder puts the stored
    vectors in another space, so an answer would be meaningless. A different
    chunk size only makes the index heterogeneous, and refusing to answer would
    be worse than answering."""
    s = Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp())
    s.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["qwen reranker"])])
    q = fake_embedder.embed_query("qwen reranker")

    chunked = Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp(chunk_size=512))
    assert chunked.search(q, top_k=5, query_text="qwen reranker")       # still answers

    swapped = Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp(embedder_model="other"))
    with pytest.raises(IndexStaleError, match="embedder_model"):
        swapped.search(q, top_k=5, query_text="qwen reranker")


def test_drop_lets_a_new_embedder_dimension_take_hold(tmp_path, fake_embedder):
    """The remedy the stale error MANDATES has to work.

    Production opens the store with `embedding_dim=None` and learns the dim
    from disk, so re-creating the table inside `drop()` re-created it at the
    OLD dim: swapping to a 128-dim embedder emptied the index and then died on
    the first rebuild batch with `embedding dim 128 != index dim 64`. Every
    other test here pins `embedding_dim=64`, which is why it passed."""
    from v2.conftest import FakeEmbedder

    db = tmp_path / "db"
    s = Store(db_path=db, embedding_dim=None, fingerprint=_fp())
    s.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    assert s.embedding_dim == 64

    swapped = Store(db_path=db, embedding_dim=None, fingerprint=_fp(embedder_model="other"))
    assert swapped.embedding_dim == 64 and swapped.config_drift
    swapped.drop()
    assert swapped.config_drift == {}

    big = FakeEmbedder(dim=128)
    assert swapped.upsert_batch([_rows(big, "c:/r/p/docs/a.md", ["alpha"])]).upserted_rows == 1
    assert swapped.embedding_dim == 128
    assert Store(db_path=db, embedding_dim=None,
                 fingerprint=_fp(embedder_model="other")).config_drift == {}


def test_a_stale_index_refuses_before_it_embeds_and_is_not_requeued(cfg, tmp_path, fake_embedder):
    """Fail-closed drift must not cost a GPU pass per debounce window. The
    refusal used to arrive inside `upsert_batch`, after the batch had been
    embedded, and `submit` then requeued the paths to be re-embedded on the
    next tick, forever: `IndexStaleError` is permanent, not transient."""
    from mainframe.app import Mainframe
    from mainframe.core.models import Models
    from mainframe.service.events import EventStore
    from mainframe.service.watcher import Invalidation
    from v2.conftest import FakeEmbedder, FakeReranker
    from v2.helpers import write_md

    db = tmp_path / "mainframe" / INDEX_DIRNAME
    Store(db_path=db, embedding_dim=64,
          fingerprint=Store.fingerprint_from_config(cfg)).upsert_batch(
        [_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    cfg["chunker"]["chunk_size"] = 384                  # drift

    embeds = []

    class CountingEmbedder(FakeEmbedder):
        def embed(self, texts, batch_size=None):
            embeds.append(len(texts))
            return super().embed(texts, batch_size)

    models = Models(cfg, embedder_factory=lambda c: CountingEmbedder(dim=64),
                    reranker_factory=lambda c: FakeReranker(top_k=3))
    app = Mainframe(cfg, models=models, events=EventStore(tmp_path / "ev.db"))
    app.invalidation = Invalidation()
    p = str(write_md(tmp_path / "repos" / "p" / "docs" / "b.md", "# T\n\nbeta text.\n"))

    # The undo is offered exactly as the scheduler offers it, so the assertion
    # below has teeth: `_worth_retrying` is what must refuse it.
    with pytest.raises(IndexStaleError):
        app.submit("index", [p], on_retryable_failure=lambda: app.invalidation.touch(p))
    assert embeds == [], "the batch was embedded for a write that could never land"
    assert app.invalidation.take_paths() == (), "a permanent failure was queued for retry"


def test_status_reports_rebuild_required(cfg, tmp_path, fake_embedder):
    from mainframe.app import Mainframe
    from mainframe.service.events import EventStore
    from v2.fakes import fake_models

    Store(db_path=tmp_path / "mainframe" / INDEX_DIRNAME, embedding_dim=64,
          fingerprint=Store.fingerprint_from_config(cfg)).upsert_batch(
        [_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"))
    assert app.status()["index"]["rebuild_required"] is False
    cfg["chunker"]["chunk_size"] = 384
    stale = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"))
    assert stale.status()["index"]["rebuild_required"] is True


def test_drop_rewrites_the_marker(tmp_path, fake_embedder):
    s = Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp())
    s.upsert_batch([_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    changed = Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp(chunk_size=512))
    assert changed.config_drift

    changed.drop()                       # every row is about to be built again
    assert _marker(changed)["fingerprint"]["chunk_size"] == 512
    assert changed.config_drift == {}
    assert Store(db_path=tmp_path / "db", embedding_dim=64, fingerprint=_fp(chunk_size=512)).config_drift == {}


def test_status_surfaces_config_drift(cfg, tmp_path, fake_embedder):
    from mainframe.app import Mainframe
    from mainframe.service.events import EventStore
    from v2.fakes import fake_models

    db = tmp_path / "mainframe" / INDEX_DIRNAME
    Store(db_path=db, embedding_dim=64,
          fingerprint=Store.fingerprint_from_config(cfg)).upsert_batch(
        [_rows(fake_embedder, "c:/r/p/docs/a.md", ["alpha"])])
    cfg["chunker"]["chunk_size"] = 384
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"))
    assert app.status()["index"]["config_drift"] == {"chunk_size": [256, 384]}
