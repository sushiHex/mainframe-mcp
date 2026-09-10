import gc
import threading
import warnings

from mainframe.service.events import EventStore


def test_emit_recent_count(tmp_path):
    ev = EventStore(tmp_path / "events.db")
    ev.emit("daemon.start", pid=1)
    ev.emit("index.batch", project="p", files=3, failed=[])
    rows = ev.recent(10)
    assert [r["kind"] for r in rows] == ["index.batch", "daemon.start"]
    assert rows[0]["project"] == "p" and rows[0]["detail"]["files"] == 3
    assert ev.count() == 2 and ev.count("index.batch") == 1
    assert ev.recent(5, kind="daemon.start")[0]["detail"] == {"pid": 1}


def test_concurrent_writers(tmp_path):
    ev = EventStore(tmp_path / "events.db")

    def writer(i):
        for j in range(50):
            EventStore(tmp_path / "events.db").emit("t", i=i, j=j)
    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ev.count("t") == 200


def test_survives_missing_parent_and_bad_detail(tmp_path):
    ev = EventStore(tmp_path / "nested" / "dir" / "events.db")
    ev.emit("x", obj=object())
    assert ev.recent(1)[0]["detail"]["obj"].startswith("<object")


def test_connections_are_closed(tmp_path):
    ev = EventStore(tmp_path / "events.db")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        for i in range(5):
            ev.emit("t", i=i)
        ev.recent(3)
        ev.count()
        ev.count("t")
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)], [str(w.message) for w in caught]
