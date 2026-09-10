"""The one test that drives a REAL watchdog Observer.

Every other watcher test injects a FakeObserver and calls `on_event` by hand —
the right default (real observers are timing-dependent), but it is also why the
live smoke, not the suite, found the directory-event storm, and why a delete
event never reaching the dirty set was invisible. This covers the whole class:
watchdog really dispatches, the handler really runs on its thread, and a
CREATE and a DELETE both land in the dirty set.

Marked `slow` so it can be deselected (`-m "not slow"`); it stays in the
default run and is bounded by a 3 s deadline.
"""
import time
from pathlib import Path

import pytest

from mainframe.service.watcher import Invalidation, Watcher

DEADLINE_SECONDS = 3.0


def _wait_for(predicate, deadline: float = DEADLINE_SECONDS):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.mark.slow
def test_a_real_observer_delivers_creates_and_deletes(tmp_path):
    seen = Invalidation()
    w = Watcher([tmp_path], seen, is_relevant=lambda p: p.lower().endswith(".md"))
    try:
        w.start()
    except Exception as e:                       # no inotify handles / no FS events here
        pytest.skip(f"cannot start a real observer: {e}")
    try:
        assert w.alive
        f = tmp_path / "a.md"
        f.write_text("# a\n", encoding="utf-8")
        assert _wait_for(lambda: any(Path(p).name == "a.md" for p in _peek(seen))), \
            "the create event never reached the invalidation"
        seen.take_paths()
        f.unlink()
        assert _wait_for(lambda: any(Path(p).name == "a.md" for p in _peek(seen))), \
            "the delete event never reached the invalidation"
    finally:
        w.stop()
    assert not w.alive


def _peek(dirty: Invalidation) -> list:
    """Read the set without draining it (the poll loop runs many times)."""
    with dirty._lock:
        return sorted(dirty._paths)
