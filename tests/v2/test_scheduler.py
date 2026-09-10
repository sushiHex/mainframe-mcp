"""What the daemon's clock must GUARANTEE.

The old suite tested how two clock baselines advanced, which is how the
scheduler happened to be built rather than what it owes anyone. These are the
requirements: work is submitted when it is due and never onto a busy writer, a
gap in the loop eventually reconciles, a dead watcher is repaired without
storming, cooldowns are honored, and nothing here ever touches the models.
"""
from mainframe.app import _worth_retrying
from mainframe.core.store import IndexStaleError
from mainframe.service import scheduler as scheduler_mod
from mainframe.service.scheduler import Scheduler, Snapshot, decide
from mainframe.service.watcher import Invalidation


class NoModels:
    """The scheduler must never touch the models. Elapsed time is not evidence
    about the GPU — a device failure is repaired where it happens, by
    `Models.invoke`."""

    def __getattr__(self, name):
        raise AssertionError(f"the scheduler touched models.{name}")


class RecordingApp:
    """A Mainframe stand-in that records submits and events.

    `raise_on` names the kinds whose SUBMIT blows up synchronously; `fails`
    maps a kind to the exception its JOB fails with afterwards. The failure
    path goes through the real `_worth_retrying`, so the fake cannot quietly
    disagree with production about which failures are worth undoing."""

    def __init__(self, raise_on=(), fails=None):
        self.calls, self.kinds = [], []
        self.raise_on = frozenset(raise_on)
        self.fails = dict(fails or {})
        outer = self

        class _Events:
            @staticmethod
            def emit(kind, **detail):
                outer.kinds.append((kind, detail))

        self.models, self.events = NoModels(), _Events()

    def submit(self, kind, *a, wait=True, timeout=None, on_retryable_failure=None, on_success=None):
        self.calls.append((kind, a, wait))
        if kind in self.raise_on:
            raise RuntimeError("stopping")
        exc = self.fails.get(kind)
        if exc is not None and on_retryable_failure is not None and _worth_retrying(exc):
            on_retryable_failure()
        if exc is None and on_success is not None:
            on_success()
        return None

    def submitted(self, kind):
        return [c for c in self.calls if c[0] == kind]

    def events_of(self, kind):
        return [d for k, d in self.kinds if k == kind]


class IdleQueue:
    depth = 0


class LiveWatcher:
    alive = True
    restarts = 0

    def restart(self):
        self.restarts += 1


def _scheduler(c, app, clk, wall, watcher=None, queue=None, health=None):
    """A Scheduler on fake clocks: `clk`/`wall` are one-key dicts the test
    advances by hand, so an hour of hibernate is a line of arithmetic."""
    return Scheduler(app, Invalidation(clock=lambda: clk["t"]), watcher or LiveWatcher(),
                     queue or IdleQueue(), c, clock=lambda: clk["t"], wall=lambda: wall["t"],
                     health=health or (lambda: {"versions": 1, "fragments": 1}))


def _cfg(cfg):
    cfg["service"]["tick_seconds"] = 5
    cfg["index"].update(debounce_seconds=10, rescan_hours=6, batch_cap=200, optimize_versions=20,
                        optimize_fragments=10, optimize_cooldown_seconds=900)
    return cfg


def _snap(**kw):
    base = dict(now_mono=1000.0, dirty_len=0, quiet_for=float("inf"), last_rescan_mono=1000.0,
                full=False, versions=1, fragments=1, queue_depth=0, watcher_alive=True,
                last_repair_mono=float("-inf"), repair_failures=0, last_optimize_mono=float("-inf"))
    base.update(kw)
    return Snapshot(**base)


def _acts(snap, c) -> list:
    return list(decide(snap, c))


def _run_ticks(s, clk, wall, n, step=5):
    for _ in range(n):
        clk["t"] += step
        wall["t"] += step
        s.tick()


def test_decide_truth_table(cfg):
    c = _cfg(cfg)
    assert _acts(_snap(), c) == []
    assert _acts(_snap(dirty_len=3, quiet_for=4), c) == []                  # inside the debounce
    assert _acts(_snap(dirty_len=3, quiet_for=10), c) == ["index"]
    assert _acts(_snap(dirty_len=200, quiet_for=0), c) == ["index"]         # batch_cap short-circuits it
    assert _acts(_snap(now_mono=1000 + 6 * 3600), c) == ["rescan"]          # periodic
    assert _acts(_snap(full=True), c) == ["rescan"]
    assert _acts(_snap(watcher_alive=False), c) == ["repair"]
    assert _acts(_snap(versions=21), c) == ["optimize"]
    assert _acts(_snap(fragments=11), c) == ["optimize"]
    # a rescan re-enumerates the lanes, so it SUBSUMES the pending paths: no
    # tick submits an index job and then a rescan that redoes the same work
    assert _acts(_snap(dirty_len=5, quiet_for=99, full=True, versions=30), c) == ["rescan", "optimize"]


def test_the_queue_gates_writer_work_but_not_watcher_repair(cfg):
    """Nothing is stacked onto a busy writer. Restarting an observer is not
    writer work, though, and a rescan that runs for minutes must not leave the
    watcher dead for its whole duration — the reconcile it asks for is recorded,
    not submitted."""
    c = _cfg(cfg)
    assert _acts(_snap(dirty_len=1, quiet_for=99, full=True, versions=30, queue_depth=1), c) == []
    assert _acts(_snap(watcher_alive=False, queue_depth=1), c) == ["repair"]


def test_a_gap_in_the_loop_eventually_reconciles_and_never_touches_the_models(cfg):
    """An hour with no tick means events may have been dropped, so the index is
    reconciled by enumeration. That is ALL it means: `NoModels` fails the test
    if the scheduler so much as reads `app.models`."""
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}
    app = RecordingApp()
    s = _scheduler(c, app, clk, wall)

    clk["t"] += 3600                     # both clocks in lockstep: divergence is zero,
    wall["t"] += 3600                    # which is exactly what Windows QPC does
    s.tick()
    assert len(app.submitted("rescan")) == 1
    assert app.events_of("daemon.reconcile") == [{"cause": "loop_gap"}]

    _run_ticks(s, clk, wall, 10)         # and the ticks that follow are quiet
    assert len(app.submitted("rescan")) == 1


def test_a_gap_while_the_queue_is_busy_is_not_lost(cfg):
    """A gated tick submits nothing, but the reconcile it recorded stays
    recorded — hibernating during a rescan is the ordinary laptop case."""
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}

    class Busy:
        depth = 1

    app, q = RecordingApp(), Busy()
    s = _scheduler(c, app, clk, wall, queue=q)
    clk["t"] += 3600
    wall["t"] += 3600
    s.tick()
    assert app.calls == [] and s.invalidation.full is True

    q.depth = 0
    _run_ticks(s, clk, wall, 1)
    assert len(app.submitted("rescan")) == 1
    _run_ticks(s, clk, wall, 5)
    assert len(app.submitted("rescan")) == 1          # consumed once, not re-offered forever


def test_a_long_job_is_not_a_gap(cfg):
    """The gap is measured against the HEARTBEAT, which every tick advances —
    not against a baseline that a busy queue freezes. Measured the other way, a
    ten-minute rescan read as a suspend and provoked another rescan that itself
    outran the threshold, forever."""
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}

    class Busy:
        depth = 1

    app, q = RecordingApp(), Busy()
    s = _scheduler(c, app, clk, wall, queue=q)
    _run_ticks(s, clk, wall, 60)          # 5 minutes of ticks while the writer is held
    q.depth = 0
    _run_ticks(s, clk, wall, 1)
    assert app.calls == [] and app.kinds == []


def test_time_spent_inside_a_tick_is_not_loop_downtime(cfg):
    """The heartbeat stamps FRESH readings at the END of a tick. Stamping the
    tick's own start billed its duration to the loop as downtime, so a 90 s
    block inside the tick manufactured a gap on the very next one."""
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}

    class SlowApp(RecordingApp):
        def submit(self, kind, *a, **kw):
            clk["t"] += 90                    # the tick body takes 90 s
            wall["t"] += 90
            return super().submit(kind, *a, **kw)

    app = SlowApp()
    s = _scheduler(c, app, clk, wall)
    s.invalidation.touch("C:/x.md")
    _run_ticks(s, clk, wall, 1, step=11)      # past the 10 s debounce
    assert app.submitted("index")
    _run_ticks(s, clk, wall, 1)
    assert app.submitted("rescan") == [], "the tick's own duration was billed to the loop as downtime"


def test_a_dead_watcher_is_repaired_and_then_reconciles(cfg):
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}

    class Flaky(LiveWatcher):
        alive = False
        restarts = 0

        def restart(self):
            self.restarts += 1
            type(self).alive = True

    app, w = RecordingApp(), Flaky()
    s = _scheduler(c, app, clk, wall, watcher=w)
    _run_ticks(s, clk, wall, 1)
    assert w.restarts == 1 and app.events_of("daemon.watcher_repair") == [{"after_failures": 0}]

    _run_ticks(s, clk, wall, 1)               # the reconcile it asked for lands next tick
    assert len(app.submitted("rescan")) == 1
    assert app.events_of("daemon.reconcile") == [{"cause": "watcher_restarted"}]

    _run_ticks(s, clk, wall, 10)
    assert w.restarts == 1 and len(app.submitted("rescan")) == 1


def test_a_permanently_dead_watcher_does_not_storm(cfg):
    """`Watcher.start()` keeps raising, so `alive` is False forever. 40
    ticks at 5 s must not enqueue a rescan per tick, and must space the retries
    by at least the 60 s floor."""
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}
    attempts = []

    class DeadWatcher:
        alive = False

        def restart(self):
            attempts.append(clk["t"])
            raise OSError("no handles")

    app = RecordingApp()
    s = _scheduler(c, app, clk, wall, watcher=DeadWatcher())
    _run_ticks(s, clk, wall, 40)

    assert app.submitted("rescan") == []      # a failed repair asks for nothing
    assert app.kinds == []
    assert len(attempts) >= 2, "the damper must still retry, just slowly"
    assert attempts[1] - attempts[0] >= 60
    assert scheduler_mod.retry_cooldown(0) == 60 and scheduler_mod.retry_cooldown(99) == 3600


def test_repair_backs_off_per_consecutive_failure(cfg):
    c = _cfg(cfg)
    assert _acts(_snap(watcher_alive=False, last_repair_mono=970.0), c) == []           # 30 s ago < 60 s
    assert _acts(_snap(watcher_alive=False, last_repair_mono=939.0), c) == ["repair"]   # 61 s ago
    # three consecutive failures -> 8 * 60 = 480 s floor
    assert _acts(_snap(watcher_alive=False, last_repair_mono=900.0, repair_failures=3), c) == []
    assert _acts(_snap(watcher_alive=False, last_repair_mono=500.0, repair_failures=3), c) == ["repair"]
    # a suppressed repair must not suppress the rest of the clock's work
    assert _acts(_snap(watcher_alive=False, last_repair_mono=990.0, full=True), c) == ["rescan"]


def test_optimize_has_a_cooldown(cfg):
    """LanceDB retains every version younger than `cleanup_minutes`, so a
    table steady at 21 versions is still at 21 after a successful optimize —
    and `Store.optimize` turns failure into False rather than raising. Without
    a cooldown that is one optimize every tick, forever."""
    c = _cfg(cfg)
    assert _acts(_snap(versions=21, last_optimize_mono=1000.0 - 899), c) == []
    assert _acts(_snap(versions=21, last_optimize_mono=1000.0 - 900), c) == ["optimize"]
    # a suppressed optimize never blocks the rest of the clock's work
    assert _acts(_snap(versions=21, full=True, last_optimize_mono=999.0), c) == ["rescan"]


def test_a_stable_version_count_optimizes_once_per_cooldown(cfg):
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}
    app = RecordingApp()
    s = _scheduler(c, app, clk, wall, health=lambda: {"versions": 21, "fragments": 1})
    _run_ticks(s, clk, wall, 120)             # 10 minutes of 5 s ticks
    assert len(app.submitted("optimize")) == 1
    _run_ticks(s, clk, wall, 180)             # keep ticking until the 900 s cooldown expires
    assert len(app.submitted("optimize")) == 2


def test_tick_indexes_debounced_paths_and_reconciles_on_request(cfg):
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}
    app = RecordingApp()
    s = _scheduler(c, app, clk, wall)
    s.tick()
    assert app.calls == []
    s.invalidation.touch("C:/x.md")
    _run_ticks(s, clk, wall, 1, step=11)
    assert app.calls[-1] == ("index", (("C:/x.md",),), False)

    s.invalidation.reconcile("directory_event")
    _run_ticks(s, clk, wall, 1)
    assert app.calls[-1][0] == "rescan" and s.invalidation.full is False
    assert app.events_of("daemon.reconcile")[-1] == {"cause": "directory_event"}


def test_a_reconcile_that_races_an_index_decision_is_not_lost(cfg):
    """`decide` chose `index` from a snapshot taken before the watcher thread
    recorded a reconcile. Draining everything in the index branch would clear a
    request this tick never acted on — the written-here, cleared-there loss the
    accumulator exists to end, reintroduced on a smaller edge."""
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}
    app = RecordingApp()
    s = _scheduler(c, app, clk, wall)
    s.invalidation.touch("C:/x.md")
    pre_race = _snap(dirty_len=1, quiet_for=99, full=False)
    s.invalidation.reconcile("directory_event")          # the watcher thread, in the gap
    s._act(pre_race)

    assert app.submitted("index") and s.invalidation.full is True
    _run_ticks(s, clk, wall, 1)
    assert app.submitted("rescan")
    assert app.events_of("daemon.reconcile") == [{"cause": "directory_event"}]


def test_a_failed_rescan_gives_back_what_it_drained(cfg):
    """The drain race again, on the FAILURE path. `rescan` consumes the reasons
    and `last_rescan` is advanced, so a job that later fails used to leave the
    signal permanently consumed — affected files stayed stale until the
    six-hour periodic. The caller that consumed it is the one that restores it."""
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}
    app = RecordingApp(fails={"rescan": RuntimeError("lancedb blip")})
    s = _scheduler(c, app, clk, wall)
    s.invalidation.reconcile("directory_event")

    _run_ticks(s, clk, wall, 1)
    assert len(app.submitted("rescan")) == 1
    assert s.invalidation.full is True, "the reasons died with the failed job"

    # ... and the retry is paced, not run at tick rate: the restore re-arms an
    # UNDEBOUNCED trigger, so the storm is one this restore itself creates.
    _run_ticks(s, clk, wall, 10)
    assert len(app.submitted("rescan")) == 1
    _run_ticks(s, clk, wall, 16)                   # past retry_cooldown(1) = 120 s
    assert len(app.submitted("rescan")) == 2
    assert app.events_of("daemon.reconcile")[-1] == {"cause": "directory_event"}


def test_a_successful_rescan_gives_back_nothing(cfg):
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}
    app = RecordingApp()
    s = _scheduler(c, app, clk, wall)
    s.invalidation.reconcile("directory_event")

    _run_ticks(s, clk, wall, 1)
    assert len(app.submitted("rescan")) == 1 and s.invalidation.full is False
    _run_ticks(s, clk, wall, 40)
    assert len(app.submitted("rescan")) == 1


def test_a_failed_periodic_rescan_is_still_restored(cfg):
    """A periodic rescan drains no reasons, so restoring "what it drained"
    would restore nothing — and `last_rescan` was already advanced, which is
    the same six-hour hole by another route."""
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}
    app = RecordingApp(fails={"rescan": RuntimeError("lancedb blip")})
    s = _scheduler(c, app, clk, wall)
    s.last_rescan = clk["t"] - 6 * 3600 - 1        # the periodic is due

    _run_ticks(s, clk, wall, 1)
    assert app.events_of("daemon.reconcile") == [{"cause": "periodic"}]
    assert s.invalidation.full is True
    assert s.invalidation.drain() == ("rescan_failed",)


def test_a_stale_index_rescan_restores_nothing(cfg):
    """`IndexStaleError` is permanent: re-running a rescan cannot clear it, and
    a rebuild re-enumerates everything anyway."""
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}
    app = RecordingApp(fails={"rescan": IndexStaleError("chunk_size drifted")})
    s = _scheduler(c, app, clk, wall)
    s.invalidation.reconcile("directory_event")

    _run_ticks(s, clk, wall, 1)
    assert len(app.submitted("rescan")) == 1 and s.invalidation.full is False
    _run_ticks(s, clk, wall, 60)
    assert len(app.submitted("rescan")) == 1


def test_consecutive_rescan_failures_back_off(cfg):
    """Each failure restores the reason, so without a growing damper the retry
    is immediate and permanent. Ticks stay at 5 s so the loop-gap detector
    never fires — a big clock jump would ask for its own reconcile and hide
    the thing under test."""
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}
    app = RecordingApp(fails={"rescan": RuntimeError("lancedb blip")})
    s = _scheduler(c, app, clk, wall)
    s.invalidation.reconcile("directory_event")

    attempts = []
    for _ in range(500):                            # ~42 minutes of 5 s ticks
        before = len(app.submitted("rescan"))
        _run_ticks(s, clk, wall, 1)
        if len(app.submitted("rescan")) > before:
            attempts.append(clk["t"])
    gaps = [b - a for a, b in zip(attempts, attempts[1:])]
    assert len(gaps) >= 3 and gaps == sorted(gaps), f"not backing off: {gaps}"
    assert gaps[0] >= 120 and gaps[-1] >= 4 * gaps[0]
    assert scheduler_mod.retry_cooldown(0) == 60 and scheduler_mod.retry_cooldown(99) == 3600


def test_a_successful_rescan_clears_the_failure_count(cfg):
    """A successful job clears its backoff as part of completion."""
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}
    app = RecordingApp(fails={"rescan": RuntimeError("lancedb blip")})
    s = _scheduler(c, app, clk, wall)
    s.invalidation.reconcile("directory_event")
    _run_ticks(s, clk, wall, 1)
    assert s.rescan_failures == 1

    app.fails.clear()
    _run_ticks(s, clk, wall, 26)                    # past retry_cooldown(1); this one succeeds
    assert len(app.submitted("rescan")) == 2 and s.invalidation.full is False
    assert s.rescan_failures == 0

    s.invalidation.reconcile("directory_event")
    _run_ticks(s, clk, wall, 1)
    assert len(app.submitted("rescan")) == 3 and s.rescan_failures == 0


def test_tick_requeues_drained_paths_when_submit_raises(cfg):
    c = _cfg(cfg)
    clk, wall = {"t": 1000.0}, {"t": 1_000_000.0}
    s = _scheduler(c, RecordingApp(raise_on=("index",)), clk, wall)
    s.invalidation.touch("C:/x.md")
    s.invalidation.touch("C:/y.md")
    _run_ticks(s, clk, wall, 1, step=15)   # past the 10 s debounce; must not raise, though the submit does
    assert sorted(s.invalidation.take_paths()) == ["C:/x.md", "C:/y.md"]
