import threading
import weakref

import pytest

from mainframe.core.models import Models, is_device_failure
from mainframe.service.events import EventStore
from v2.fakes import fake_models


def _ident(role, m):
    """Inspect identity in tests only; production operations return results."""
    return m.invoke(role, lambda obj: obj)


def test_lazy_load_and_state(cfg):
    calls = []
    m = Models(cfg, embedder_factory=lambda c: calls.append("e") or object(),
               reranker_factory=lambda c: calls.append("r") or object())
    assert m.state()["embedder"]["state"] == "absent" and calls == []
    e1 = _ident("embedder", m)
    assert m.state()["embedder"]["state"] == "loaded" and calls == ["e"]
    assert _ident("embedder", m) is e1 and calls == ["e"]
    assert m.state()["reranker"]["state"] == "absent"


def test_a_device_failure_reloads_once_and_retries(cfg, tmp_path):
    """Repair at the point of failure, not on a clock: a dead CUDA context is
    something that HAPPENED, not something elapsed time predicts."""
    loads, ev = [], EventStore(tmp_path / "ev.db")
    m = Models(cfg, embedder_factory=lambda c: loads.append(1) or f"model{len(loads)}",
               reranker_factory=lambda c: None, events=ev)
    calls = []

    def op(obj):
        calls.append(obj)
        if len(calls) == 1:
            raise RuntimeError("CUDA error: unknown error")
        return "ok"

    assert m.invoke("embedder", op) == "ok"
    assert calls == ["model1", "model2"] and len(loads) == 2      # invalidated, reloaded, retried
    kinds = [e["kind"] for e in ev.recent(10)]
    assert "model.device_failure" in kinds and "model.invalidated" in kinds


@pytest.mark.parametrize("role", ["embedder", "reranker"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_recovery_releases_the_failed_model_before_cache_clear_and_reload(cfg, monkeypatch, role, wrapped):
    """A one-model memory budget must suffice, including cycles and traceback
    references. Pytest's log capture also keeps records alive across reload."""
    refs, cache_liveness = [], []

    class Model:
        def __init__(self, fails):
            self.fails = fails
            self.cycle = self

        def run(self):
            if self.fails:
                try:
                    raise RuntimeError("CUDA error: synthetic device failure")
                except RuntimeError as exc:
                    if wrapped:
                        raise RuntimeError("operation failed") from exc
                    raise
            return "ok"

    def load(_cfg):
        assert all(ref() is None for ref in refs), "replacement loaded while the failed model was alive"
        if refs:
            assert cache_liveness == [False], "release the model before clearing the CUDA cache"
        model = Model(fails=not refs)
        refs.append(weakref.ref(model))
        return model

    monkeypatch.setattr("mainframe.core.models.empty_cuda_cache",
                        lambda: cache_liveness.append(any(ref() is not None for ref in refs)))
    m = Models(cfg, **{f"{role}_factory": load})
    assert m.invoke(role, lambda model: model.run()) == "ok"
    assert len(refs) == 2 and refs[0]() is None and refs[1]() is not None
    assert m.state()[role]["state"] == "loaded"


def test_a_replacement_load_failure_keeps_its_backoff(cfg):
    loads = []

    def load(_cfg):
        loads.append(1)
        if len(loads) > 1:
            raise RuntimeError("replacement load failed")
        return object()

    def fail(_model):
        raise RuntimeError("CUDA error: synthetic device failure")

    m = Models(cfg, embedder_factory=load, clock=lambda: 0)
    with pytest.raises(RuntimeError, match="replacement load failed"):
        m.invoke("embedder", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        m.invoke("embedder", fail)
    assert len(loads) == 2 and m.state()["embedder"]["retry_after_s"] == 900


def test_a_second_device_failure_propagates(cfg):
    """ONE retry, not a loop: a second identical failure is a real fault and
    the caller must see it."""
    loads = []
    m = Models(cfg, embedder_factory=lambda c: loads.append(1) or object(),
               reranker_factory=lambda c: None)
    attempts = []

    def always_bad(_obj):
        attempts.append(1)
        raise RuntimeError("CUDA error: illegal memory access was encountered")

    with pytest.raises(RuntimeError, match="illegal memory access"):
        m.invoke("embedder", always_bad)
    assert len(attempts) == 2 and len(loads) == 2


def test_waiting_callers_share_one_replacement_without_retaining_the_failed_model(cfg):
    refs, calls, results, errors = [], [], [], []
    entered, release = threading.Event(), threading.Event()
    ready = threading.Barrier(8)

    class Model:
        def __init__(self, serial):
            self.serial = serial

    def load(_cfg):
        assert all(ref() is None for ref in refs), "a caller still holds the failed model"
        model = Model(len(refs) + 1)
        refs.append(weakref.ref(model))
        return model

    def op(model):
        calls.append(model.serial)
        if model.serial == 1:
            entered.set()
            assert release.wait(5), "test did not release the failing operation"
            raise RuntimeError("CUDA error: synthetic device failure")
        return model.serial

    m = Models(cfg, embedder_factory=load)

    def invoke(wait=False):
        try:
            if wait:
                ready.wait(5)
            results.append(m.invoke("embedder", op))
        except Exception as exc:
            errors.append(str(exc))

    first = threading.Thread(target=invoke, daemon=True)
    waiting = [threading.Thread(target=invoke, args=(True,), daemon=True) for _ in range(7)]
    first.start()
    try:
        assert entered.wait(5)
        # Health remains readable while a model operation owns the lock.
        assert m.state()["embedder"]["state"] == "loaded"
        for thread in waiting:
            thread.start()
        ready.wait(5)
    finally:
        release.set()
        for thread in [first, *waiting]:
            if thread.ident is not None:
                thread.join(10)
    assert not any(thread.is_alive() for thread in [first, *waiting])
    assert not errors
    assert results == [2] * 8 and calls == [1] + [2] * 8
    assert len(refs) == 2 and refs[0]() is None


def test_only_device_failures_heal(cfg):
    """A wrong argument must not cost a two-minute model reload."""
    loads = []
    m = Models(cfg, embedder_factory=lambda c: loads.append(1) or object(),
               reranker_factory=lambda c: None)
    with pytest.raises(ValueError):
        m.invoke("embedder", lambda _o: (_ for _ in ()).throw(ValueError("bad input")))
    assert len(loads) == 1 and m.loaded("embedder")


def test_is_device_failure_matches_type_and_message(cfg):
    assert is_device_failure(RuntimeError("CUDA error: unknown error"))
    assert is_device_failure(RuntimeError("cuDNN error: CUDNN_STATUS_NOT_INITIALIZED"))
    assert is_device_failure(RuntimeError("no kernel image is available for execution"))
    assert not is_device_failure(ValueError("dimension mismatch"))
    # OOM is a CAPACITY failure: reloading the same model cannot make room, and
    # the two-minute reload would be pure loss.
    assert not is_device_failure(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    # Every pattern must NAME the device stack. Dropping a healthy 8B model and
    # reloading it for two minutes is too expensive to spend on a phrase this
    # broad, and all three of these once matched.
    assert not is_device_failure(RuntimeError("unknown error"))
    assert not is_device_failure(RuntimeError("tokenizer initialization error"))
    assert not is_device_failure(ValueError("invalid device argument for this op"))


def test_is_device_failure_walks_the_cause_chain():
    """The layer that reports a fault is rarely the layer that raised it: a
    wrapper saying "embedding failed" `from` the real CUDA error is the
    ordinary shape, and matching only the outermost message missed it."""
    try:
        try:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        except RuntimeError as e:
            raise RuntimeError("embedding failed") from e
    except RuntimeError as wrapped:
        assert is_device_failure(wrapped)

    try:
        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        except RuntimeError as e:
            raise RuntimeError("embedding failed") from e
    except RuntimeError as wrapped:
        assert not is_device_failure(wrapped), "OOM stays excluded wherever it sits in the chain"

    # `__context__` alone is not a claim about this failure: whatever happened
    # to be in flight is not evidence that the GPU died.
    try:
        try:
            raise RuntimeError("CUDA error: unknown error")
        except RuntimeError:
            raise ValueError("bad input")
    except ValueError as implicit:
        assert not is_device_failure(implicit)


def test_failed_load_backs_off_then_retries(cfg, tmp_path):
    cfg["models"]["retry_minutes"] = 15
    clk = {"t": 0.0}
    attempts = []

    def boom(c):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("cuda gone")
        return "ok"
    ev = EventStore(tmp_path / "ev.db")
    m = Models(cfg, embedder_factory=boom, reranker_factory=lambda c: None, events=ev, clock=lambda: clk["t"])
    with pytest.raises(RuntimeError):
        _ident("embedder", m)
    st = m.state()["embedder"]
    assert st["state"] == "failed" and "cuda gone" in st["error"] and st["retry_after_s"] == 900
    with pytest.raises(RuntimeError, match="unavailable"):
        _ident("embedder", m)                        # backoff: no second factory call
    assert len(attempts) == 1
    clk["t"] += 15 * 60
    assert _ident("embedder", m) == "ok" and m.state()["embedder"]["state"] == "loaded"
    assert [e["kind"] for e in ev.recent(5)] == ["model.loaded", "model.failed"]


def test_invalidate_frees_the_cuda_cache_when_torch_is_already_imported(cfg, monkeypatch):
    """The unload path must release objects and clear the CUDA cache: dropping the
    references is not enough — the allocator keeps the blocks cached, so the
    reload finds less free VRAM than it should. Never IMPORTS torch: a daemon
    that has never loaded a model must not initialize CUDA here."""
    import sys
    from types import SimpleNamespace

    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: calls.append("empty_cache"))))
    m = fake_models(cfg)
    m.prewarm(background=False)
    assert m.loaded("embedder") and m.loaded("reranker")
    m.invalidate("embedder")
    assert calls == ["empty_cache"]
    assert not m.loaded("embedder") and m.state()["embedder"]["state"] == "absent"
    assert m.loaded("reranker"), "invalidation is per ROLE: a bad embedder says nothing about the reranker"

    monkeypatch.delitem(sys.modules, "torch")
    m.prewarm(background=False)
    m.invalidate("embedder")                    # no torch imported: still fine, still no import
    assert "torch" not in sys.modules and not m.loaded("embedder")


def test_invalidate_survives_a_broken_cuda_runtime(cfg, monkeypatch):
    import sys
    from types import SimpleNamespace

    def boom():
        raise RuntimeError("CUDA driver gone")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=boom)))
    m = fake_models(cfg)
    m.prewarm(background=False)
    m.invalidate("embedder")                    # best-effort: a dead runtime must not block the reload
    assert not m.loaded("embedder") and m.state()["embedder"]["state"] == "absent"


def test_reload_releases_all_models_before_one_cache_clear(cfg, monkeypatch):
    refs, cache_liveness = [], []

    class Model:
        def __init__(self):
            self.cycle = self

    def load(_cfg):
        model = Model()
        refs.append(weakref.ref(model))
        return model

    monkeypatch.setattr("mainframe.core.models.empty_cuda_cache",
                        lambda: cache_liveness.append([ref() is not None for ref in refs]))
    m = Models(cfg, embedder_factory=load, reranker_factory=load)
    m.prewarm(background=False)
    assert m.invalidate_all() == ["embedder", "reranker"]
    assert cache_liveness == [[False, False]]
    assert m.invalidate_all() == []
    assert cache_liveness == [[False, False]], "an empty reload needs no allocator work"


def test_health_snapshots_stay_readable_through_loading_failure_and_release(cfg):
    entered, release = threading.Event(), threading.Event()
    refs = []

    class Model:
        pass

    def load_embedder(_cfg):
        entered.set()
        assert release.wait(5), "test did not release model loading"
        model = Model()
        refs.append(weakref.ref(model))
        return model

    def load_reranker(_cfg):
        raise RuntimeError("synthetic load failure")

    m = Models(cfg, embedder_factory=load_embedder, reranker_factory=load_reranker, clock=lambda: 0)
    worker = threading.Thread(target=lambda: m.prewarm(background=False), daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        loading = m.state()
        assert loading["embedder"]["state"] == "loading"
        assert loading["embedder"]["error"] is None and loading["embedder"]["retry_after_s"] == 0
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    settled = m.state()
    assert settled["embedder"]["state"] == "loaded"
    assert settled["reranker"]["state"] == "failed"
    assert settled["reranker"]["error"] == "synthetic load failure"
    assert settled["reranker"]["retry_after_s"] == 900
    assert m.invalidate_all() == ["embedder", "reranker"]
    assert refs[0]() is None, "retained health snapshots must not retain a model"
    assert loading["embedder"]["state"] == "loading" and settled["embedder"]["state"] == "loaded"
    assert all(s["state"] == "absent" and s["error"] is None and s["retry_after_s"] == 0
               for s in m.state().values())


def test_concurrent_first_access_loads_once(cfg):
    calls = []
    gate = threading.Event()

    def slow_factory(c):
        gate.wait(5)
        calls.append(1)
        return object()
    m = Models(cfg, embedder_factory=slow_factory, reranker_factory=lambda c: None)
    got = []
    threads = [threading.Thread(target=lambda: got.append(_ident("embedder", m))) for _ in range(8)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(10)
    assert len(calls) == 1 and len(got) == 8 and all(o is got[0] for o in got)


def test_state_reports_live_backoff_without_touching_the_factory(cfg):
    cfg["models"]["retry_minutes"] = 1
    clk = {"t": 0.0}
    attempts = []

    def boom(c):
        attempts.append(1)
        raise RuntimeError("no gpu")
    m = Models(cfg, embedder_factory=boom, reranker_factory=lambda c: None, clock=lambda: clk["t"])
    with pytest.raises(RuntimeError):
        _ident("embedder", m)
    assert m.state()["embedder"]["retry_after_s"] == 60
    clk["t"] += 45
    assert m.state()["embedder"]["retry_after_s"] == 15 and len(attempts) == 1
    clk["t"] += 30
    assert m.state()["embedder"]["retry_after_s"] == 0 and len(attempts) == 1   # state() never loads
