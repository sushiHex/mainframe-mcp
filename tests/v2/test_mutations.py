import threading
import time
from concurrent.futures import Future

import pytest

from mainframe.service.events import EventStore
from mainframe.service.mutations import _STOP, MutationQueue, QueueStopped


def test_fifo_single_thread_and_worker_detection(tmp_path):
    q = MutationQueue(EventStore(tmp_path / "ev.db"))
    q.start()
    order, threads, on_worker = [], set(), []

    def job(i):
        threads.add(threading.get_ident())
        on_worker.append(q.is_worker_thread())
        time.sleep(0.01)
        order.append(i)
        return i * 2
    futs = [q.submit(f"j{i}", job, i) for i in range(5)]
    assert [f.result(timeout=5) for f in futs] == [0, 2, 4, 6, 8]
    assert order == [0, 1, 2, 3, 4] and len(threads) == 1 and all(on_worker) and not q.is_worker_thread()
    assert q.stop(timeout=5) is True


def test_exception_isolated_and_reported(tmp_path):
    ev = EventStore(tmp_path / "ev.db")
    q = MutationQueue(ev)
    q.start()

    def boom():
        raise RuntimeError("bad")
    with pytest.raises(RuntimeError):
        q.submit("boom", boom).result(timeout=5)
    assert q.run("ok", lambda: 42, timeout=5) == 42
    assert ev.recent(1, "job.failed")[0]["detail"]["name"] == "boom"
    q.stop()


def test_depth_counts_running_and_stop_reports_timeout(tmp_path):
    q = MutationQueue()
    q.start()
    gate = threading.Event()
    started = threading.Event()

    def slow():
        started.set()
        gate.wait()

    q.submit("slow", slow)
    q.submit("next", lambda: None)
    assert started.wait(timeout=5)
    assert q.running == "slow" and q.depth == 2
    assert q.stop(timeout=0.1) is False          # in-flight job still blocked
    assert q.depth == 2                          # the queued STOP sentinel is excluded
    gate.set()
    assert q.stop(timeout=5) is True and q.running is None and q.depth == 0


def test_accepting_false_before_start_and_once_stopping(tmp_path):
    q = MutationQueue()
    assert q.accepting is False                  # before start()
    q.start()
    assert q.accepting is True                   # after start()
    gate = threading.Event()
    started = threading.Event()

    def slow():
        started.set()
        gate.wait()

    q.submit("slow", slow)
    assert started.wait(timeout=5)
    assert q.stop(timeout=0.1) is False           # stop() called, worker still draining
    assert q.accepting is False                   # false the moment stop() is called, not just once it returns
    gate.set()
    assert q.stop(timeout=5) is True
    assert q.accepting is False                   # still false once stop() has fully returned


def test_submit_after_stop_raises_queue_stopped(tmp_path):
    q = MutationQueue()
    q.start()
    gate = threading.Event()
    started = threading.Event()

    def slow():
        started.set()
        gate.wait()

    q.submit("slow", slow)
    assert started.wait(timeout=5)

    stop_thread = threading.Thread(target=q.stop, kwargs={"timeout": 5})
    stop_thread.start()
    while not q._stop_sent:            # wait for stop() to flip the flag; it then blocks on join
        time.sleep(0.001)

    outcome = {}

    def _submit():
        with pytest.raises(QueueStopped):
            q.submit("late", lambda: None)
        outcome["done"] = True
    submit_thread = threading.Thread(target=_submit)
    submit_thread.start()
    submit_thread.join(2)
    assert not submit_thread.is_alive() and outcome.get("done") is True   # returned promptly, didn't hang

    gate.set()
    stop_thread.join(5)
    assert not stop_thread.is_alive()


def test_stop_fails_orphaned_futures(tmp_path):
    q = MutationQueue()
    q.start()
    q._stop_sent = True                # bypass the lock — simulate the lost race
    q._q.put(_STOP)
    orphan: Future = Future()
    q._q.put(("orphan", lambda: None, (), {}, orphan))
    assert q.stop(timeout=5) is True
    assert isinstance(orphan.exception(timeout=1), QueueStopped)
