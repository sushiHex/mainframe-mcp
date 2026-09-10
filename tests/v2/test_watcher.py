from pathlib import Path
from types import SimpleNamespace

import pytest

from mainframe.service.watcher import Invalidation, Watcher


class FakeClock:
    def __init__(self):
        self.t = 100.0
    def __call__(self):
        return self.t


def test_invalidation_dedups_paths_and_measures_quiet():
    clk = FakeClock()
    d = Invalidation(clock=clk)
    assert d.state(clk()) == (0, float("inf"), False)
    d.touch("C:/a.md"); d.touch("C:/a.md"); d.touch("C:/b.md")
    clk.t += 4
    assert d.state(clk()) == (2, 4, False)          # one atomic view, not three reads
    assert d.take_paths() == ("C:/a.md", "C:/b.md")
    assert d.state(clk()) == (0, float("inf"), False)


def test_a_full_scan_subsumes_the_paths_and_is_not_debounced():
    """One statement, one accumulator: a rescan re-enumerates the lanes, so
    every path recorded before it is already covered."""
    clk = FakeClock()
    d = Invalidation(clock=clk)
    d.touch("C:/a.md")
    d.reconcile("directory_event")
    d.reconcile("directory_event")            # reasons are a set: no unbounded growth
    d.reconcile("root_event")
    assert d.full is True
    assert d.drain() == ("directory_event", "root_event")
    assert d.state(clk()) == (0, float("inf"), False)      # the paths went with it


def test_taking_paths_leaves_a_racing_reconcile_standing():
    """An index job does NOT subsume a full scan. `decide` chose `index`
    because nothing was pending at snapshot time; the watcher thread can record
    a reason in the gap before the drain, and clearing it there would throw
    away a request nobody acted on."""
    d = Invalidation()
    d.touch("C:/a.md")
    d.reconcile("directory_event")            # races in after the snapshot
    assert d.take_paths() == ("C:/a.md",)
    assert d.full is True
    assert d.drain() == ("directory_event",)


class FakeObserver:
    instances = []
    def __init__(self):
        self.scheduled, self.started, self.stopped = [], False, False
        self.emitters = []
        FakeObserver.instances.append(self)
    def schedule(self, handler, path, recursive=False):
        self.scheduled.append((handler, path, recursive))
    def start(self):
        self.started = True
    def stop(self):
        self.stopped = True
    def join(self, timeout=None):
        pass
    def is_alive(self):
        return self.started and not self.stopped


def _ev(src, dest=None, is_dir=False):
    return SimpleNamespace(src_path=src, dest_path=dest, is_directory=is_dir)


def test_watcher_schedules_real_handler_filters_and_flags_overflow(tmp_path):
    d = Invalidation()
    w = Watcher([tmp_path], d, is_relevant=lambda p: p.endswith(".md"), observer_factory=FakeObserver)
    w.start()
    obs = FakeObserver.instances[-1]
    handler, path, recursive = obs.scheduled[0]
    assert obs.started and path == str(tmp_path) and recursive is True
    assert hasattr(handler, "dispatch") and hasattr(handler, "on_any_event")     # a real watchdog handler
    handler.on_any_event(_ev(str(tmp_path / "a.md")))
    handler.on_any_event(_ev(str(tmp_path / "a.txt")))
    handler.on_any_event(_ev(str(tmp_path / "old.md"), dest=str(tmp_path / "new.md")))
    assert sorted(Path(p).name for p in d.take_paths()) == ["a.md", "new.md", "old.md"]
    assert d.full is False
    handler.on_any_event(_ev(str(tmp_path / "sub"), is_dir=True))                 # directory event
    assert d.full is True
    d.drain()
    handler.on_any_event(_ev(str(tmp_path)))                                      # event on the root = overflow
    assert d.full is True
    assert w.alive
    w.restart()
    assert obs.stopped and FakeObserver.instances[-1] is not obs and w.alive
    w.stop()
    assert not w.alive


def test_directory_events_are_filtered_by_is_relevant_dir(tmp_path):
    """a directory event anywhere under repos_dir used to flag a rescan, so
    an out-of-scope repo's .git/node_modules churn drove a full rescan every
    tick. The root path itself still flags unconditionally (a root deletion)."""
    d = Invalidation()
    w = Watcher([tmp_path], d, is_relevant=lambda p: p.endswith(".md"),
                observer_factory=FakeObserver, is_relevant_dir=lambda p: "keep" in p)
    w.start()
    handler = FakeObserver.instances[-1].scheduled[0][0]
    handler.on_any_event(_ev(str(tmp_path / "junk" / "objects"), is_dir=True))
    assert d.full is False and d.take_paths() == ()
    handler.on_any_event(_ev(str(tmp_path / "keep" / "docs"), is_dir=True))
    assert d.full is True
    d.drain()
    # a directory MOVED into scope is judged on its destination too
    handler.on_any_event(_ev(str(tmp_path / "junk"), dest=str(tmp_path / "keep2"), is_dir=True))
    assert d.full is True
    d.drain()
    handler.on_any_event(_ev(str(tmp_path), is_dir=True))          # the root itself: always
    assert d.full is True


def test_roots_may_be_a_callable_and_are_re_evaluated(tmp_path):
    """watch_roots() only returns directories that exist AT THAT MOMENT, and
    restart() reused the startup list — a repos_dir (or project) created after
    launch was never watched until a process restart."""
    (tmp_path / "one").mkdir()
    roots = [tmp_path / "one"]
    d = Invalidation()
    w = Watcher(lambda: list(roots), d, is_relevant=lambda p: p.endswith(".md"),
                observer_factory=FakeObserver)
    assert [str(r) for r in w.roots] == [str(tmp_path / "one")]
    w.start()
    (tmp_path / "two").mkdir()
    roots.append(tmp_path / "two")
    w.restart()
    assert [str(r) for r in w.roots] == [str(tmp_path / "one"), str(tmp_path / "two")]
    assert [p for _, p, _ in FakeObserver.instances[-1].scheduled] == [str(tmp_path / "one"), str(tmp_path / "two")]
    handler = FakeObserver.instances[-1].scheduled[0][0]
    handler.on_any_event(_ev(str(tmp_path / "two")))               # the NEW root path: overflow signal
    assert d.full is True


def test_relevance_errors_flag_a_rescan_instead_of_raising(tmp_path):
    d = Invalidation()
    w = Watcher([tmp_path], d, is_relevant=lambda p: 1/0, observer_factory=FakeObserver)
    w.start()
    obs = FakeObserver.instances[-1]
    handler = obs.scheduled[0][0]
    # Should not raise; should ask for a reconcile and record no path
    handler.on_any_event(_ev(str(tmp_path / "a.md")))
    assert d.full is True
    assert d.take_paths() == ()


def test_alive_reports_dead_emitters(tmp_path):
    d = Invalidation()
    w = Watcher([tmp_path], d, is_relevant=lambda p: p.endswith(".md"), observer_factory=FakeObserver)
    w.start()
    obs = FakeObserver.instances[-1]
    assert w.alive  # Default: no emitters, so alive is True

    # One dead emitter -> alive is False
    obs.emitters = [SimpleNamespace(is_alive=lambda: True), SimpleNamespace(is_alive=lambda: False)]
    assert w.alive is False

    # Only one emitter (one root), it's alive -> alive is True
    obs.emitters = [SimpleNamespace(is_alive=lambda: True)]
    assert w.alive is True


def test_start_failure_stops_the_partial_observer(tmp_path):
    class FailingObserver(FakeObserver):
        def start(self):
            raise RuntimeError("no handles")

    d = Invalidation()
    w = Watcher([tmp_path], d, is_relevant=lambda p: p.endswith(".md"), observer_factory=FailingObserver)
    with pytest.raises(RuntimeError, match="no handles"):
        w.start()
    obs = FailingObserver.instances[-1]
    assert obs.stopped is True
    assert w.alive is False
