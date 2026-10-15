"""`mainframe` CLI: a thin REST client of the daemon, plus the
offline paths (serve, rebuild-under-lock) and the Task Scheduler installer."""

import argparse
import getpass
import io
import json
import os
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from xml.sax.saxutils import escape

import httpx

from mainframe.config import load_config
from mainframe.memory.capture import KINDS
from mainframe.service.discovery import LOCK_HELD_EXIT, SUPERVISOR_PID_ENV, lock_path, probe, read_discovery

DOWN_EXIT = 2
DOWN_WAIT_SECONDS = 30          # fallback only: service.shutdown_timeout_seconds is the real bound
DOWN_WAIT_MARGIN_SECONDS = 10   # the daemon's own bound, plus time to unwind and release
DOWN_PROGRESS_SECONDS = 10
TASK_NAME = "Mainframe daemon"
# Messages are ASCII on purpose: stderr is routinely piped through cp1252 on
# Windows, where an em dash arrives as mojibake.
NO_SCHTASKS = ("mainframe: Task Scheduler is Windows-only; run `mainframe serve` in a terminal "
               "(service units for other platforms are planned)")
NO_TASK_FALLBACK = ("mainframe: no logon task answered; starting a detached `serve` instead. "
                    "Run `mainframe install-task` once for a daemon that returns after a reboot.")
PREWARM_HINT = " (service.prewarm: true loads the models at start and makes `up` wait until it is ready)"
START_WAIT_SECONDS = 120
START_POLL_SECONDS = 1.0
LOG_WAIT_SECONDS = 5.0      # a handoff takes moments; anything longer is another holder
LOG_POLL_SECONDS = 0.25
SERVE_ARGV = ["-m", "mainframe.adapters.cli", "serve"]   # what the logon task and the detached fallback run
ONCE_ARGV = SERVE_ARGV + ["--once"]                      # one daemon process, no restarts: what `serve` supervises
RESTART_DELAYS = (1, 2, 4, 8, 16)   # seconds before each successive restart of a dead daemon; then give up
STABLE_SECONDS = 60                 # a daemon that served this long had started: its death earns a fresh budget

# The maintenance verbs the CLI exposes, each a plain POST to /jobs/<name>.
# `index` is deliberately absent: `Mainframe.maintain` treats it as an alias for
# `rescan`, and one name per action is enough on a command line. Keeping this
# list here rather than deriving it from `app.JOBS` is the point — the CLI
# chooses what to surface, and `rebuild` is offline-only so it is not in it.
JOB_COMMANDS = {"rescan": "reconcile the lanes with the index",
                "optimize": "compact the index and refresh its indexes",
                "reload": "drop the resident models; the next use reloads them"}


class DaemonClient:
    def __init__(self, config: dict, http=None):
        self.config = config
        self._http = http
        self.info = read_discovery(config)
        self.base_url = f"http://127.0.0.1:{self.info['port']}" if self.info else None

    @property
    def http(self):
        if self._http is None:
            import httpx
            tok = self.config["service"].get("token")
            self._http = httpx.Client(timeout=600, headers={"Authorization": f"Bearer {tok}"} if tok else {})
        return self._http

    def alive(self) -> bool:
        if not self.base_url:
            return False
        return probe(self.config, self.http) is not None

    def get(self, path: str):
        r = self.http.get(self.base_url + path)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, json=None):
        r = self.http.post(self.base_url + path, json=json)
        r.raise_for_status()
        return r.json()


def task_xml(config: dict) -> str:
    # An InteractiveToken task running python.exe opens a console at every
    # logon; pythonw.exe beside it is the same interpreter without one.
    windowless = Path(sys.executable).with_name("pythonw.exe")
    exe = escape(str(windowless) if windowless.exists() else sys.executable)
    wd = escape(config["paths"]["mainframe_dir"])
    # The task fires at, and runs as, the logon of the user who registers it.
    # A trigger on ANY user's logon is what made registering it need an
    # elevated shell; scoped to its own user it needs no prompt at all.
    domain = os.environ.get("USERDOMAIN")
    user = escape(f"{domain}\\{getpass.getuser()}" if domain else getpass.getuser())
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Mainframe v2 resident daemon (mainframed)</Description></RegistrationInfo>
  <Triggers><LogonTrigger><Enabled>true</Enabled><UserId>{user}</UserId></LogonTrigger></Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure><Interval>PT1M</Interval><Count>3</Count></RestartOnFailure>
    <StartWhenAvailable>true</StartWhenAvailable>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{exe}</Command>
      <Arguments>-m mainframe.adapters.cli serve</Arguments>
      <WorkingDirectory>{wd}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def schtasks_command(xml_path: Path) -> list:
    return ["schtasks", "/Create", "/F", "/TN", TASK_NAME, "/XML", str(xml_path)]


def quiet_call(argv) -> int:
    """The exit code, with the command's own output kept off the terminal:
    the `runner` for the `schtasks /Run` PROBE in `up`/`restart`, where a
    machine with no logon task printed schtasks' 'cannot find the file' right
    above our explanation of that very case. `install-task` runs the user's
    own command, and schtasks' output IS its report - that one stays loud."""
    return subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def run_offline_rebuild(config: dict, app=None) -> dict:
    """THE rebuild. It drops the table, so it never runs while the daemon is
    serving readers: this takes `.daemon.lock` itself, which is the exclusion —
    there is no queue to serialize on because there is no daemon."""
    from mainframe.service.daemon import Daemon
    d = Daemon(config, app=app)
    if not d.acquire():
        print(f"mainframe: a rebuild runs offline only and the daemon holds {lock_path(config)}; "
              "stop it first with `mainframe down`", file=sys.stderr)
        sys.exit(LOCK_HELD_EXIT)
    try:
        return d.app.pipeline.rebuild().as_dict()
    finally:
        d.release()


def _emit(obj):
    # ensure_ascii=True on purpose: the payload is document content (headings,
    # snippets) that routinely carries an arrow, an em dash, or other non-ASCII
    # text, and a Windows console is routinely cp1252 — printing raw non-ASCII
    # there is the same UnicodeEncodeError the messages-are-ASCII rule above
    # exists to avoid. Escaping to \uXXXX keeps the JSON valid and always
    # encodable, without reconfiguring stdout globally.
    print(json.dumps(obj, indent=2, ensure_ascii=True))


def _call(fn) -> int:
    """Run one daemon REST call (± an `_emit` of its result); a daemon that
    dies between the `/healthz` probe and this call raises a transport error,
    which we translate into one clean stderr line instead of a traceback.
    Single choke point — subcommands route their dispatch through here rather
    than each carrying its own try/except. Only actual connection/transport
    failures are reported as "could not reach the daemon" — anything else
    (a bug in handling an otherwise-successful response, say) surfaces with
    its own message instead of being mislabeled as connectivity."""
    try:
        fn()
    except (httpx.TransportError, ConnectionError) as e:
        print(f"mainframe: could not reach the daemon ({e})", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"mainframe: {e}", file=sys.stderr)
        return 1
    return 0


def daemon_out(config) -> Path:
    return Path(config["paths"]["mainframe_dir"]) / "daemon.out"


def _open_log(out):
    """`out` open for append, or None if it stays unavailable.

    The handle becomes the child's stdout, so a dying daemon holds it for a
    moment AFTER releasing the lock. Taking the lock as proof the previous
    daemon is gone is the same mistake twice: it proves only that the lock is
    free. So wait briefly, as `wait_for_lock` does for the resource it needs.

    It gives up rather than waiting forever, because a log file must never be
    able to prevent a daemon from starting - something else (an editor, a
    scanner) may hold it indefinitely, and that is not a reason to have no
    daemon.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + LOG_WAIT_SECONDS
    while True:
        try:
            return open(out, "ab")
        except PermissionError:
            if time.monotonic() >= deadline:
                return None
            time.sleep(LOG_POLL_SECONDS)


def spawn_detached(config) -> bool:
    """Start `serve` as a detached child. True when its output is being captured.

    The redirect is load-bearing, not tidiness. A detached child on Windows has
    no console, and the model loader writes progress to stdout; against an
    invalid handle that kills the daemon seconds after it starts, leaving no
    process and no log line - a start that reports success and produced nothing.

    The return value is not a courtesy: `start_daemon` reads an unchanged
    daemon.out as proof that nothing ran, and that reading is only valid while
    the daemon is actually writing there.
    """
    # DETACHED_PROCESS alone: Windows documents CREATE_NO_WINDOW and
    # DETACHED_PROCESS as mutually exclusive, and combining them can fail
    # CreateProcess outright. No console is wanted anyway - output goes to `out`.
    extra = ({"creationflags": subprocess.DETACHED_PROCESS}
             if sys.platform == "win32" else {"start_new_session": True})
    sink = _open_log(daemon_out(config))
    try:
        # DEVNULL, not an unredirected handle: what kills a detached daemon is an
        # INVALID stdout, and DEVNULL is valid. Losing the log costs diagnosis,
        # never the daemon.
        subprocess.Popen([sys.executable, *SERVE_ARGV],
                         stdout=sink or subprocess.DEVNULL, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, **extra)
    finally:
        if sink is not None:
            sink.close()
    return sink is not None


def supervise(run_child, sleep=time.sleep, clock=time.monotonic) -> int:
    """Keep ONE daemon alive until it stops on purpose.

    `run_child` runs `serve` to completion and returns its exit code. Exit 0
    is the daemon's own decision to stop - `mainframe down`, `/shutdown`, a
    signal uvicorn handles - and ends supervision: an intentional stop is
    never undone. `LOCK_HELD_EXIT` means another instance is serving, so
    there is nothing to restart. Anything else is a death - a crash, a kill,
    a startup failure - and the daemon is started again after a growing delay.

    What a kill looks like is the one platform question here. Windows has no
    graceful kill: TerminateProcess (taskkill /F, Task Manager, os.kill) sets
    a non-zero code, so every kill there is a death, and the only clean stops
    are the ones the daemon answers itself. On POSIX, SIGTERM/SIGINT reach
    uvicorn's handler and exit 0 (clean, as today's foreground `serve`), while
    SIGKILL cannot be handled and reports -9. One rule - zero is intentional -
    reads both correctly.

    The budget bounds a daemon that dies at startup every time: one restart
    per delay in `RESTART_DELAYS`, then this returns the last exit code (a
    restart storm would bury the cause under its own noise). A daemon that
    served for `STABLE_SECONDS` before dying earns the budget back - the
    failure was not in starting, and the next start is expected to work."""
    failures = 0
    while True:
        started = clock()
        rc = run_child()
        if rc == 0 or rc == LOCK_HELD_EXIT:
            return rc
        if clock() - started >= STABLE_SECONDS:
            failures = 0
        if failures >= len(RESTART_DELAYS):
            print(f"mainframe: daemon died {failures + 1} times in a row (last exit {rc}); giving up",
                  file=sys.stderr)
            return rc
        delay = RESTART_DELAYS[failures]
        failures += 1
        print(f"mainframe: daemon exited {rc}; restarting in {delay} s", file=sys.stderr)
        sleep(delay)


def _std(stream):
    """`stream` when it is a real file the child can write to, else None (inherit).
    None itself, a closed stream, and pytest's capture all answer None."""
    try:
        stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    return stream


def serve_child() -> int:
    """One `serve --once`, run to completion, writing where the supervisor writes:
    under pythonw that is daemon.out (see the `serve` branch of `main`), and
    handing the handle down is what keeps the child's crash readable there.

    On an interrupt the operator stopped BOTH processes - a console delivers
    Ctrl-C to each - so wait for the child to finish its own shutdown and
    report a clean stop. `subprocess.call` would kill it after a quarter
    second, cutting a graceful shutdown short."""
    # The child's discovery records this pid beside its own, so `down` can
    # wait for the supervisor too: it IS the logon task's instance, and a task
    # with IgnoreNew drops the next start while that instance runs (#59).
    env = dict(os.environ, **{SUPERVISOR_PID_ENV: str(os.getpid())})
    proc = subprocess.Popen([sys.executable, *ONCE_ARGV], stdin=subprocess.DEVNULL, env=env,
                            stdout=_std(sys.stdout), stderr=_std(sys.stderr))
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.wait()
        return 0


def start_daemon(config, runner, spawn, http=None, timeout=START_WAIT_SECONDS) -> int:
    """Make a daemon exist AND answer, or say why one does not.

    Prefers the logon task, so that a manual start runs what logon runs. When no
    task answers - none is registered yet - it falls back to a detached `serve`
    and says so, because the remedy for "no task" is not "give up".

    It returns only once `/healthz` answers - and then, if the daemon is
    warming up, once that is over (`wait_ready`). Returning when the LAUNCHER returned
    is the defect this exists to close: a launcher can exit cleanly having
    produced no process at all, and the caller cannot tell. On failure the growth
    of `daemon.out` separates "it started and died" from "nothing ever ran",
    which is the one fact that makes a silent start diagnosable.
    """
    # Anything the stop phase printed belongs above what follows: stdout is
    # block-buffered when piped while stderr is not, so without this the start's
    # messages overtake it and a restart reads out of order.
    sys.stdout.flush()
    out = daemon_out(config)
    before = out.stat().st_size if out.exists() else 0
    try:
        rc = runner(["schtasks", "/Run", "/TN", TASK_NAME])
    except OSError:
        rc = 1
    detached = rc != 0
    captured = True
    if detached:
        print(NO_TASK_FALLBACK, file=sys.stderr)
        captured = spawn(config) is not False

    deadline = time.monotonic() + timeout
    while True:
        if probe(config, http) is not None:
            return wait_ready(config, http, deadline)
        if time.monotonic() >= deadline:
            break
        time.sleep(START_POLL_SECONDS)

    grew = (out.stat().st_size if out.exists() else 0) - before
    how = "a detached `serve`" if detached else "the logon task"
    if not captured:
        # Without this the report would say "no process at all" about a daemon
        # whose output was never being watched - a confident wrong diagnosis in
        # the one place this instrumentation exists to be trusted.
        tail = (f"{out.name} could not be opened, so its size says nothing about "
                "this start and the daemon's own output was discarded")
    elif grew > 0:
        tail = f"{out.name} grew {grew} bytes; read it for the cause"
    else:
        tail = f"{out.name} did not change, so the launcher produced no process at all"
    print(f"mainframe: started {how} but no daemon answered within {int(timeout)} s - {tail}",
          file=sys.stderr)
    return 1


def wait_ready(config, http, deadline) -> int:
    """Answering is not ready. A daemon answers `/healthz` the moment it binds,
    with both models absent; `readiness` in that same body says whether a
    search would be served by resident models, and the DAEMON says which -
    no config is consulted here.

    `warming` is the one state worth waiting on: the daemon is working toward
    ready by itself. `ready` is 0. `idle` is 0 too - nothing loads until a
    search, so waiting would wait forever; say so and name the key that changes
    it. `blocked` is final (a failed load backs off for minutes; polling a
    timeout against it would report the timeout instead of the failure and its
    remedy), so it is reported at once. A body with no `readiness` at all is a
    daemon from before the field, answering the old contract: 0."""
    while True:
        info = probe(config, http)
        if info is None:
            print("mainframe: daemon stopped answering while warming up", file=sys.stderr)
            return 1
        rd = info["health"].get("readiness")
        if rd is None or rd.get("state") == "ready":
            return 0
        reason = rd.get("reason")
        if rd.get("state") == "idle":
            print(f"mainframe: daemon answering; {reason}{PREWARM_HINT}", file=sys.stderr)
            return 0
        if rd.get("state") != "warming":
            print(f"mainframe: daemon answering but {rd.get('state')} - {reason}", file=sys.stderr)
            return 1
        if time.monotonic() >= deadline:
            print(f"mainframe: daemon answering but still warming ({reason}); "
                  "raise `--timeout` to wait longer", file=sys.stderr)
            return 1
        time.sleep(START_POLL_SECONDS)


def process_exited(pid) -> bool:
    """True once `pid` has finished. Windows: a SYNCHRONIZE handle that is
    signalled, or cannot be opened because the process is gone. POSIX: signal 0."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, pid)        # SYNCHRONIZE
        if not handle:
            return True
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0     # WAIT_OBJECT_0: it exited
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def wait_for_lock(config, pids=()) -> int:
    """Wait until the daemon is really gone: nothing holds the lock AND every
    process in `pids` (from discovery: the daemon and, when it has one, its
    supervisor) has exited. The lock alone is not enough - the daemon releases
    it and then tears down for seconds more, and a logon task whose instance
    is still running ignores the next start, so an `up` right after a `down`
    that returned on the lock started nothing. The supervisor is that
    instance, and it exits only after its child has."""
    from filelock import FileLock, Timeout
    # The daemon finishes its IN-FLIGHT job before releasing the lock, and
    # that job can be a rescan of thousands of files. Wait for the daemon's
    # OWN bound (service.shutdown_timeout_seconds) plus a margin, or `down`
    # reports failure on a shutdown that is proceeding exactly as designed -
    # and say out loud that the wait is deliberate rather than a hang.
    limit = float(config["service"].get("shutdown_timeout_seconds", DOWN_WAIT_SECONDS)) \
        + DOWN_WAIT_MARGIN_SECONDS
    start = time.monotonic()
    next_progress = start + DOWN_PROGRESS_SECONDS
    released = False
    while time.monotonic() - start < limit:
        if not released:
            try:
                with FileLock(str(lock_path(config))).acquire(timeout=0):
                    released = True
            except Timeout:
                now = time.monotonic()
                if now >= next_progress:
                    print("mainframe: waiting for the daemon to finish its current job "
                          f"({int(now - start)}s elapsed)", file=sys.stderr)
                    next_progress = now + DOWN_PROGRESS_SECONDS
        if released and all(process_exited(pid) for pid in pids):
            return 0
        time.sleep(0.5)
    if released:
        alive = ", ".join(str(pid) for pid in pids if not process_exited(pid))
        print(f"mainframe: the daemon released its lock, but process {alive} is still running "
              f"after {int(limit)} s", file=sys.stderr)
    else:
        print(f"mainframe: daemon still holds the lock after {int(limit)} s", file=sys.stderr)
    return 1


def stop_daemon(client, config) -> int:
    """Ask a RUNNING daemon to stop, and return once it is gone - lock AND
    process tree: the daemon, and the supervisor that will exit after it."""
    info = client.info or {}                       # read before shutdown withdraws discovery
    pids = [p for p in (info.get("pid"), info.get("supervisor_pid")) if p]
    rc = _call(lambda: _emit(client.post("/shutdown")))
    return rc if rc != 0 else wait_for_lock(config, pids)


def main(argv=None, config=None, http=None, runner=None, spawn=None) -> int:
    p = argparse.ArgumentParser(prog="mainframe", description="Mainframe v2: local knowledge base daemon")
    sub = p.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve", help="run the daemon in the foreground, starting it again when it dies "
                                      "(a clean stop - `mainframe down` - ends it)")
    sv.add_argument("--once", action="store_true", help="run one daemon process, no restarts")
    u = sub.add_parser("up", help="start the daemon and wait until it answers (and is done warming up)")
    u.add_argument("--timeout", type=float, default=START_WAIT_SECONDS)
    sub.add_parser("down", help="ask the daemon to stop and wait for it")
    r = sub.add_parser("restart", help="stop the daemon if it is running, start it, and confirm it answers "
                                       "(and is done warming up)")
    r.add_argument("--timeout", type=float, default=START_WAIT_SECONDS)
    sub.add_parser("status")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--sessions", action="store_true")
    s.add_argument("--limit", type=int, default=3); s.add_argument("--detailed", action="store_true")
    for name, helptext in JOB_COMMANDS.items():
        sub.add_parser(name, help=helptext)
    sub.add_parser("rebuild", help="drop and re-index everything (offline; stop the daemon first)")
    e = sub.add_parser("events"); e.add_argument("--tail", type=int, default=20); e.add_argument("--kind")
    t = sub.add_parser("install-task"); t.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("capture"); c.add_argument("--file", required=True); c.add_argument("--project", required=True)
    c.add_argument("--title"); c.add_argument("--session-id", default="")
    c.add_argument("--kind", choices=KINDS, default="session",
                   help="decision and note are default-searchable; session is opt-in history (default)")
    h = sub.add_parser("hook", help="agent-host hook entry points (event JSON on stdin; always exits 0)")
    # A free-form name, not subcommands: argparse exits 2 on an unknown
    # choice, and a PreCompact hook that exits 2 blocks the compaction.
    h.add_argument("event", nargs="?", default="session-end",
                   help="session-end (Claude Code SessionEnd) or pre-compact (PreCompact); only labels hook.log")
    v = sub.add_parser("validate", help="run live synthetic smoke/soak checks in new state outside Git")
    v.add_argument("--output", required=True, help="new directory outside Git for state, credentials, and reports")
    v.add_argument("--duration", type=float, default=0, help="observation seconds; 0 runs smoke checks and restart")
    v.add_argument("--timeout", type=float, default=2400, help="maximum seconds for an indexing/visibility check")
    v.add_argument("--clients", default="", help="load phase client counts, e.g. 1,4,8 (default: no load phase)")
    v.add_argument("--load-seconds", type=float, default=30, help="closed-loop seconds per client count")
    args = p.parse_args(argv)
    if args.cmd == "hook":
        # Before load_config: a hook runs inside session teardown, so even a
        # malformed config is a hook.log line there, never a traceback here.
        from mainframe.adapters.capture_hook import main as capture_hook_main
        return capture_hook_main(config=config, hook=args.event)
    config = config or load_config()

    if args.cmd == "validate":
        from mainframe.adapters.validation import ValidationRun, parse_clients
        try:
            ValidationRun(args.output, config, args.duration, args.timeout,
                          clients=parse_clients(args.clients), load_seconds=args.load_seconds).run()
        except Exception as e:
            print(f"mainframe: validation failed ({e})", file=sys.stderr)
            return 1
        return 0
    if args.cmd == "serve":
        from mainframe.service.daemon import serve
        if sys.stderr is None:         # pythonw (the logon task): keep what a crash says
            sink = _open_log(daemon_out(config))
            if sink is not None:
                sys.stdout = sys.stderr = io.TextIOWrapper(sink, encoding="utf-8", errors="backslashreplace",
                                                           line_buffering=True)
        if args.once:
            serve(config)
            return 0
        return supervise(serve_child)
    if args.cmd == "install-task":
        xml_path = Path(config["paths"]["mainframe_dir"]) / "mainframe-task.xml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text(task_xml(config), encoding="utf-16")
        cmd = schtasks_command(xml_path)
        if args.dry_run:
            print(" ".join(cmd))
            return 0
        try:
            return (runner or subprocess.call)(cmd)   # the user's own command: schtasks' output is the report
        except OSError:               # no schtasks: not Windows
            print(NO_SCHTASKS, file=sys.stderr)
            return 1
    if args.cmd == "rebuild":
        _emit(run_offline_rebuild(config))       # never routed to a live daemon: see run_offline_rebuild
        return 0

    client = DaemonClient(config, http)
    runner = runner or quiet_call                 # from here `runner` only ever runs the `/Run` probe
    spawn = spawn or spawn_detached
    if args.cmd == "up":
        if client.alive():
            print("mainframe: daemon already running")     # answering - which is not yet ready
            return wait_ready(config, client.http, time.monotonic() + args.timeout)
        return start_daemon(config, runner, spawn, client.http, args.timeout)
    if args.cmd == "restart":
        if client.alive():
            rc = stop_daemon(client, config)
            if rc != 0:
                return rc
        return start_daemon(config, runner, spawn, client.http, args.timeout)

    alive = client.alive()
    if args.cmd == "down" and not alive:
        print("mainframe: daemon already stopped")   # already the desired end state
        return 0
    if not alive:
        print("mainframe: daemon not running; run `mainframe up` (or `mainframe serve`)", file=sys.stderr)
        return DOWN_EXIT
    if args.cmd == "down":
        return stop_daemon(client, config)
    if args.cmd == "status":
        return _call(lambda: _emit(client.get("/status")))
    elif args.cmd == "search":
        return _call(lambda: _emit(client.post("/search", json={
            "query": args.query, "limit": args.limit, "include_sessions": args.sessions,
            "response_format": "detailed" if args.detailed else "concise"})))
    elif args.cmd in JOB_COMMANDS:
        return _call(lambda: _emit(client.post(f"/jobs/{args.cmd}")))
    elif args.cmd == "events":
        params = {"n": args.tail}
        if args.kind is not None:
            params["kind"] = args.kind
        return _call(lambda: _emit(client.get("/events?" + urllib.parse.urlencode(params))))
    elif args.cmd == "capture":
        try:
            text = Path(args.file).read_text(encoding="utf-8")
        except OSError as e:
            print(f"mainframe: cannot read {args.file}: {e}", file=sys.stderr)
            return 1
        return _call(lambda: _emit(client.post("/capture", json={
            "content": text, "title": args.title or Path(args.file).stem,
            "project": args.project, "session_id": args.session_id, "kind": args.kind,
            "captured_by": "cli"})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
