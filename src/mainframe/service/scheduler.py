"""The daemon's clock. `decide` is a pure
function of a Snapshot so every trigger is unit-testable; the Scheduler gathers
the snapshot and submits through Mainframe.submit.

An elapsed-time gap requests a full reconcile because filesystem events may
have been missed. It cannot establish that a suspend occurred or that a GPU
failed. Model recovery belongs at the operation that observes a device fault
(`core/models.Models.invoke`)."""

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

RETRY_FLOOR_SECONDS = 60
RETRY_CEILING_SECONDS = 3600

# A full reconcile the scheduler restores after a failed rescan. The reasons
# are a closed vocabulary (they live in a set), and a periodic rescan drains
# none, so its failure needs a name of its own to be restorable at all.
RESCAN_FAILED = "rescan_failed"


def retry_cooldown(failures: int) -> float:
    """Seconds between two attempts at something that keeps failing: a 60 s
    floor, doubling per consecutive failure, capped at an hour.

    ONE damper, two callers, because both storms have the same shape and the
    same cure. A watcher that can never restart (roots gone, handle exhaustion,
    an observer that cannot be recreated after hibernate) would be restarted
    every tick forever; a rescan that keeps failing would be retried every
    tick forever, because restoring its reason re-arms an UNDEBOUNCED trigger —
    a storm the restore itself creates, and therefore part of doing the restore
    correctly."""
    return float(min(RETRY_CEILING_SECONDS,
                     max(RETRY_FLOOR_SECONDS, 2 ** min(int(failures), 8) * RETRY_FLOOR_SECONDS)))


@dataclass(frozen=True)
class Snapshot:
    now_mono: float
    dirty_len: int
    quiet_for: float
    last_rescan_mono: float
    full: bool                          # a full reconcile is pending in the Invalidation
    versions: int
    fragments: int
    queue_depth: int
    watcher_alive: bool
    last_repair_mono: float = float("-inf")
    repair_failures: int = 0
    last_optimize_mono: float = float("-inf")
    last_rescan_failure_mono: float = float("-inf")
    rescan_failures: int = 0


def decide(s: Snapshot, cfg: dict) -> tuple:
    """What this tick should do."""
    idx = cfg["index"]
    out = []
    # Restarting a dead observer is not WRITER work, so the queue gate below
    # does not cover it: a rescan that runs for minutes must not leave the
    # watcher dead for its whole duration. It submits nothing — the reconcile it
    # asks for goes into the Invalidation and waits for an idle tick like
    # everything else.
    if not s.watcher_alive and s.now_mono - s.last_repair_mono >= retry_cooldown(s.repair_failures):
        out.append("repair")
    if s.queue_depth > 0:
        return tuple(out)               # never stack work onto a busy writer
    # The damper applies only AFTER a failure, so the first attempt (and every
    # attempt of a healthy rescan) is immediate — unlike `repair`, whose floor
    # can wait because a dead observer is not urgent.
    rescan_ready = (not s.rescan_failures
                    or s.now_mono - s.last_rescan_failure_mono >= retry_cooldown(s.rescan_failures))
    # A rescan re-enumerates the lanes, so it SUBSUMES the pending paths: doing
    # both in one tick indexed the same files twice, once per job. While a
    # rescan is backing off, individual file changes still flow through `index`.
    if rescan_ready and (s.full or s.now_mono - s.last_rescan_mono >= float(idx.get("rescan_hours", 6)) * 3600):
        out.append("rescan")
    elif s.dirty_len > 0 and (s.quiet_for >= float(idx.get("debounce_seconds", 10))
                              or s.dirty_len >= int(idx.get("batch_cap", 200))):
        out.append("index")
    # LanceDB keeps every version younger than `cleanup_minutes`, so a table
    # sitting at 21 versions stays at 21 after a successful optimize — without
    # a cooldown that is one optimize per tick, forever.
    if s.versions > int(idx.get("optimize_versions", 20)) or s.fragments > int(idx.get("optimize_fragments", 10)):
        if s.now_mono - s.last_optimize_mono >= float(idx.get("optimize_cooldown_seconds", 900)):
            out.append("optimize")
    return tuple(out)


class Scheduler:
    def __init__(self, app, invalidation, watcher, mutations, config, clock=time.monotonic,
                 wall=time.time, health=None):
        self.app, self.invalidation, self.watcher = app, invalidation, watcher
        self.mutations, self.config = mutations, config
        self.clock, self.wall = clock, wall
        self._health = health or (lambda: app.store.index_health())
        self.last_rescan = clock()
        self.last_tick_wall, self.last_tick_mono = wall(), clock()   # the heartbeat: every tick advances it
        self.last_repair = float("-inf")     # the FIRST repair attempt is immediate
        self.last_optimize = float("-inf")
        self.repair_failures = 0
        # Job completion updates these on the writer; admission failures and
        # snapshots run on the scheduler. The count and timestamp move together.
        self._rescan_lock = threading.Lock()
        self.last_rescan_failure = float("-inf")
        self.rescan_failures = 0
        self._stop = threading.Event()
        self._thread = None

    def _snapshot(self, now: float) -> Snapshot:
        h = self._health()
        dirty_len, quiet_for, full = self.invalidation.state(now)
        with self._rescan_lock:
            rescan_failures, last_rescan_failure = self.rescan_failures, self.last_rescan_failure
        return Snapshot(now_mono=now, dirty_len=dirty_len, quiet_for=quiet_for,
                        last_rescan_mono=self.last_rescan, full=full, versions=int(h.get("versions", 0)),
                        fragments=int(h.get("fragments", 0)), queue_depth=int(self.mutations.depth),
                        watcher_alive=bool(self.watcher.alive), last_repair_mono=self.last_repair,
                        repair_failures=self.repair_failures, last_optimize_mono=self.last_optimize,
                        last_rescan_failure_mono=last_rescan_failure, rescan_failures=rescan_failures)

    def _note_gap(self, now: float, now_wall: float):
        """Both clocks, one baseline. Neither alone survives every platform:
        `time.monotonic()` is QueryPerformanceCounter on Windows and may advance
        across a suspend exactly as the wall clock does, so a rule that
        subtracts one from the other cancels to zero in the very case it exists
        for. `max` of the two asks the answerable question instead — did
        this loop stop running, by whichever clock noticed?"""
        tick = float(self.config["service"].get("tick_seconds", 5))
        gap = max(now_wall - self.last_tick_wall, now - self.last_tick_mono)
        if gap > max(60.0, 12 * tick):
            logger.info("no tick for %.0fs; requesting a full reconcile", gap)
            self.invalidation.reconcile("loop_gap")

    def tick(self):
        # Read the clocks BEFORE the health probe, not after. `index_health()`
        # walks fragments and versions; large tables can make that slow.
        # Billing that duration to the interval since the last tick is
        # the same mistake as stamping the heartbeat at a tick's start — it
        # makes the loop's own work look like the loop having stopped.
        now, now_wall = self.clock(), self.wall()
        self._note_gap(now, now_wall)
        try:
            self._act(self._snapshot(now))
        finally:
            # The heartbeat records that this loop RAN, which it did — whether
            # or not the tick did anything, and whether or not an action raised.
            # Read FRESH, at the END: stamping the tick's own start time would
            # bill its duration to the loop as downtime, so a 90 s block inside
            # `_act` would manufacture a gap on the very next tick.
            self.last_tick_wall, self.last_tick_mono = self.wall(), self.clock()

    def _repair(self, s: Snapshot):
        """Restart a dead observer.

        Every attempt stamps `last_repair`, so a failure waits out its backoff.
        Only a SUCCESS asks for a reconcile: a watcher that is still dead has
        nothing new to reconcile that the periodic rescan will not catch, and a
        reconcile per attempt is exactly the storm the damper exists to stop."""
        self.last_repair = s.now_mono
        ok = True
        try:
            self.watcher.restart()
        except Exception as e:
            logger.warning(f"watcher restart failed: {e}")
            ok = False
        if ok and not self.watcher.alive:
            logger.warning("watcher restarted but reports itself dead")
            ok = False
        self.repair_failures = 0 if ok else self.repair_failures + 1
        if not ok:
            logger.warning("watcher repair attempt %d failed; retrying in >= %.0fs",
                           self.repair_failures, retry_cooldown(self.repair_failures))
            return
        self.app.events.emit("daemon.watcher_repair", after_failures=s.repair_failures)
        # Whatever changed while the observer was dead was never delivered.
        self.invalidation.reconcile("watcher_restarted")

    def _restore_paths(self, paths):
        """Give back paths a job consumed but never indexed."""
        for p in paths:
            self.invalidation.touch(p)

    def _rescan_failed(self, reasons):
        """Give back what the drain consumed, and pace the retry.

        Job failures run this on the writer, admission failures on the scheduler.
        The counters move under the lock. `reasons` is empty for a PERIODIC rescan —
        which drained nothing but did advance `last_rescan`, so without a name
        of its own its failure would be the same six-hour hole by another
        route."""
        with self._rescan_lock:
            self.rescan_failures += 1
            self.last_rescan_failure = self.clock()
        for reason in reasons or (RESCAN_FAILED,):
            self.invalidation.reconcile(reason)

    def _rescan_succeeded(self):
        with self._rescan_lock:
            self.rescan_failures = 0
            self.last_rescan_failure = float("-inf")

    def _act(self, s: Snapshot):
        for a in decide(s, self.config):
            if a == "repair":
                self._repair(s)
            elif a == "index":
                # `take_paths`, not `drain`: a reconcile the watcher recorded
                # since the snapshot is not this tick's to consume.
                paths = self.invalidation.take_paths()
                try:
                    self.app.submit("index", paths, wait=False,
                                    on_retryable_failure=lambda: self._restore_paths(paths))
                except Exception as e:
                    logger.warning(f"index submit failed: {e}; giving back {len(paths)} paths")
                    self._restore_paths(paths)
            elif a == "rescan":
                reasons = self.invalidation.drain()
                cause = ", ".join(reasons) or "periodic"
                logger.info("rescan: %s", cause)
                self.app.events.emit("daemon.reconcile", cause=cause)
                self.last_rescan = s.now_mono
                try:
                    self.app.submit("rescan", wait=False,
                                    on_retryable_failure=lambda: self._rescan_failed(reasons),
                                    on_success=self._rescan_succeeded)
                except Exception as e:
                    logger.warning(f"rescan submit failed: {e}; giving back {cause}")
                    self._rescan_failed(reasons)
            elif a == "optimize":
                # Stamped before the submit, like `_repair` stamps `last_repair`:
                # a submit that fails is still an attempt, and retrying it every
                # tick is the storm the cooldown exists for.
                self.last_optimize = s.now_mono
                self.app.submit("optimize", wait=False)

    def run(self):
        tick = float(self.config["service"].get("tick_seconds", 5))
        while not self._stop.wait(tick):
            try:
                self.tick()
            except Exception as e:      # the clock must never die
                logger.warning(f"scheduler tick failed: {e}")

    def start(self):
        self._thread = threading.Thread(target=self.run, name="scheduler", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)
