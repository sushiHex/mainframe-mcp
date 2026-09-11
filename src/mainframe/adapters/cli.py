"""`mainframe` CLI: a thin REST client of the daemon, plus the
offline paths (serve, rebuild-under-lock) and the Task Scheduler installer."""

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from xml.sax.saxutils import escape

from mainframe.config import load_config
from mainframe.service.daemon import LOCK_HELD_EXIT, Daemon, lock_path
from mainframe.service.discovery import probe, read_discovery

DOWN_EXIT = 2
DOWN_WAIT_SECONDS = 30          # fallback only: service.shutdown_timeout_seconds is the real bound
DOWN_WAIT_MARGIN_SECONDS = 10   # the daemon's own bound, plus time to unwind and release
DOWN_PROGRESS_SECONDS = 10
TASK_NAME = "Mainframe daemon"
# Messages are ASCII on purpose: stderr is routinely piped through cp1252 on
# Windows, where an em dash arrives as mojibake.
NO_SCHTASKS = ("mainframe: Task Scheduler is Windows-only; run `mainframe serve` in a terminal "
               "(service units for other platforms are planned)")

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
    exe, wd = escape(sys.executable), escape(config["paths"]["mainframe_dir"])
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Mainframe v2 resident daemon (mainframed)</Description></RegistrationInfo>
  <Triggers><LogonTrigger><Enabled>true</Enabled></LogonTrigger></Triggers>
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


def run_offline_rebuild(config: dict, app=None) -> dict:
    """THE rebuild. It drops the table, so it never runs while the daemon is
    serving readers: this takes `.daemon.lock` itself, which is the exclusion —
    there is no queue to serialize on because there is no daemon."""
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
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _call(fn) -> int:
    """Run one daemon REST call (± an `_emit` of its result); a daemon that
    dies between the `/healthz` probe and this call raises a raw httpx/network
    error, which we translate into one clean stderr line instead of a
    traceback. Single choke point — subcommands route their dispatch through
    here rather than each carrying its own try/except."""
    try:
        fn()
    except Exception as e:
        print(f"mainframe: could not reach the daemon ({e})", file=sys.stderr)
        return 1
    return 0


def main(argv=None, config=None, http=None, runner=None) -> int:
    p = argparse.ArgumentParser(prog="mainframe", description="Mainframe v2: local knowledge base daemon")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the daemon in the foreground")
    sub.add_parser("up", help="start the daemon via the logon task (or say how)")
    sub.add_parser("down", help="ask the daemon to stop and wait for it")
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
    v = sub.add_parser("validate", help="run live synthetic smoke/soak checks in new state outside Git")
    v.add_argument("--output", required=True, help="new directory outside Git for state, credentials, and reports")
    v.add_argument("--duration", type=float, default=0, help="observation seconds; 0 runs smoke checks and restart")
    v.add_argument("--timeout", type=float, default=2400, help="maximum seconds for an indexing/visibility check")
    args = p.parse_args(argv)
    config = config or load_config()
    runner = runner or subprocess.call

    if args.cmd == "validate":
        from mainframe.adapters.validation import ValidationRun
        try:
            ValidationRun(args.output, config, args.duration, args.timeout).run()
        except Exception as e:
            print(f"mainframe: validation failed ({e})", file=sys.stderr)
            return 1
        return 0
    if args.cmd == "serve":
        from mainframe.service.daemon import serve
        serve(config)
        return 0
    if args.cmd == "install-task":
        xml_path = Path(config["paths"]["mainframe_dir"]) / "mainframe-task.xml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text(task_xml(config), encoding="utf-16")
        cmd = schtasks_command(xml_path)
        if args.dry_run:
            print(" ".join(cmd))
            return 0
        try:
            return runner(cmd)
        except OSError:               # no schtasks: not Windows
            print(NO_SCHTASKS, file=sys.stderr)
            return 1
    if args.cmd == "rebuild":
        _emit(run_offline_rebuild(config))       # never routed to a live daemon: see run_offline_rebuild
        return 0

    client = DaemonClient(config, http)
    if args.cmd == "up":
        if client.alive():
            print("mainframe: daemon already running")
            return 0
        try:
            rc = runner(["schtasks", "/Run", "/TN", TASK_NAME])
        except OSError:
            print(NO_SCHTASKS, file=sys.stderr)
            return 1
        if rc != 0:
            print("mainframe: no logon task; run `mainframe install-task` once, or `mainframe serve` in a terminal",
                  file=sys.stderr)
        return rc

    alive = client.alive()
    if args.cmd == "down" and not alive:
        print("mainframe: daemon already stopped")   # already the desired end state
        return 0
    if not alive:
        print("mainframe: daemon not running; run `mainframe up` (or `mainframe serve`)", file=sys.stderr)
        return DOWN_EXIT
    if args.cmd == "down":
        rc = _call(lambda: _emit(client.post("/shutdown")))
        if rc != 0:
            return rc
        from filelock import FileLock, Timeout
        # The daemon finishes its IN-FLIGHT job before releasing the lock, and
        # that job can be a rescan of thousands of files. Wait for the daemon's
        # OWN bound (service.shutdown_timeout_seconds) plus a margin, or `down`
        # reports failure on a shutdown that is proceeding exactly as designed —
        # and say out loud that the wait is deliberate rather than a hang.
        limit = float(config["service"].get("shutdown_timeout_seconds", DOWN_WAIT_SECONDS)) \
            + DOWN_WAIT_MARGIN_SECONDS
        start = time.monotonic()
        next_progress = start + DOWN_PROGRESS_SECONDS
        while time.monotonic() - start < limit:
            try:
                with FileLock(str(lock_path(config))).acquire(timeout=0):
                    return 0
            except Timeout:
                now = time.monotonic()
                if now >= next_progress:
                    print("mainframe: waiting for the daemon to finish its current job "
                          f"({int(now - start)}s elapsed)", file=sys.stderr)
                    next_progress = now + DOWN_PROGRESS_SECONDS
                time.sleep(0.5)
        print(f"mainframe: daemon still holds the lock after {int(limit)} s", file=sys.stderr)
        return 1
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
            "project": args.project, "session_id": args.session_id, "captured_by": "cli"})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
