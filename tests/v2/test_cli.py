import builtins
import json
import os
import subprocess
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import filelock
import pytest

from mainframe.adapters import cli
from mainframe.app import Mainframe
from mainframe.service.daemon import Daemon, write_discovery
from mainframe.service.events import EventStore
from v2.conftest import held
from v2.fakes import fake_models
from v2.helpers import write_md

REAL_PROCESS_EXITED = getattr(cli, "process_exited", None)


@pytest.fixture(autouse=True)
def _a_stopped_fake_daemon_has_exited(monkeypatch):
    """Discovery records THIS process's pid in these tests, and `down` waits for
    the daemon's process to exit. A fake daemon's process exits with its lock."""
    monkeypatch.setattr(cli, "process_exited", lambda pid: True, raising=False)


class FakeHTTP:
    def __init__(self, handlers):
        self.handlers, self.calls = handlers, []

    def get(self, url, **kw):
        return self._call("GET", url)

    def post(self, url, json=None, **kw):
        return self._call("POST", url, json)

    def _call(self, method, url, body=None):
        rest = url.split("127.0.0.1:1234", 1)[1]
        path, _, query = rest.partition("?")
        self.calls.append((method, path, body, query))
        fn = self.handlers.get((method, path))

        class R:
            status_code = 200 if fn else 404
            def json(self_inner):
                return fn(body) if fn else {"error": "nf"}
            def raise_for_status(self_inner):
                pass
        return R()


class RaisingHTTP:
    """Alive at /healthz, but every other call blows up mid-command (a daemon
    that dies between the healthz probe and the real request)."""
    def __init__(self, nonce):
        self.nonce = nonce

    def get(self, url, **kw):
        if url.endswith("/healthz"):
            nonce = self.nonce
            class R:
                status_code = 200
                def json(self_inner):
                    return {"ok": True, "nonce": nonce}
            return R()
        raise ConnectionError("reset")

    def post(self, url, json=None, **kw):
        raise ConnectionError("reset")


class BrokenHTTP:
    """Alive at /healthz, but the real response is unusable for a reason that
    has nothing to do with reaching the daemon (e.g. a malformed body) — this
    must surface with its own message, not the connectivity one."""
    def __init__(self, nonce):
        self.nonce = nonce

    def get(self, url, **kw):
        if url.endswith("/healthz"):
            nonce = self.nonce
            class R:
                status_code = 200
                def json(self_inner):
                    return {"ok": True, "nonce": nonce}
            return R()

        class R:
            status_code = 200
            def raise_for_status(self_inner):
                pass
            def json(self_inner):
                raise ValueError("boom: malformed response body")
        return R()


def _up(cfg, nonce="n1"):
    write_discovery(cfg, 1234, nonce)
    return {("GET", "/healthz"): lambda b: {"ok": True, "nonce": nonce}}


def test_every_maintenance_verb_reaches_its_job(cfg, capsys):
    """The surface is the same on all three fronts. `optimize` and `reload`
    existed over MCP and REST while the CLI had only `rescan`, so "exposed
    consistently across MCP, REST and CLI" was an overstatement. `index` stays
    out on purpose: it is `rescan`'s alias, and one name per action is enough."""
    http = FakeHTTP({**_up(cfg), **{("POST", f"/jobs/{name}"): (lambda n: lambda b: {"job": n})(name)
                                    for name in cli.JOB_COMMANDS}})
    for name in cli.JOB_COMMANDS:
        assert cli.main([name], config=cfg, http=http) == 0
        assert json.loads(capsys.readouterr().out) == {"job": name}
    assert set(cli.JOB_COMMANDS) == {"rescan", "optimize", "reload"}


def test_status_search_capture_are_rest_only(cfg, capsys, tmp_path):
    seen = {}
    http = FakeHTTP({**_up(cfg), ("GET", "/status"): lambda b: {"index": {"rows": 3}},
                     ("POST", "/search"): lambda b: {"results": [], "confidence": "none", "echo": b},
                     ("POST", "/capture"): lambda b: seen.update(b) or {"file": "x"}})
    assert cli.main(["status"], config=cfg, http=http) == 0
    assert json.loads(capsys.readouterr().out)["index"]["rows"] == 3
    assert cli.main(["search", "qwen3 reranker", "--sessions", "--limit", "5", "--detailed"], config=cfg, http=http) == 0
    assert json.loads(capsys.readouterr().out)["echo"] == {"query": "qwen3 reranker", "limit": 5,
                                                          "include_sessions": True, "response_format": "detailed"}
    note = tmp_path / "n.md"
    note.write_text("## Summary\nqwen3 decided.\n", encoding="utf-8")
    assert cli.main(["capture", "--file", str(note), "--project", "p", "--title", "t"], config=cfg, http=http) == 0
    assert seen["project"] == "p" and seen["content"].startswith("## Summary")
    # provenance is the CALLER, not the MCP tool name.
    assert seen["captured_by"] == "cli"
    assert not any(p == "/mcp" for _, p, _, _ in http.calls)


def test_daemon_down_and_stale_nonce(cfg, capsys):
    assert cli.main(["status"], config=cfg, http=FakeHTTP({})) == 2
    err = capsys.readouterr().err
    assert "not running" in err
    # stderr is often piped through cp1252 on Windows — an em dash in a
    # user-facing message arrives as mojibake.
    assert err.isascii(), err
    write_discovery(cfg, 1234, "old")
    http = FakeHTTP({("GET", "/healthz"): lambda b: {"ok": True, "nonce": "different"}})
    assert cli.main(["status"], config=cfg, http=http) == 2          # a foreign service on that port


def test_help_text_is_ascii(cfg, capsys):
    """`mainframe --help` goes through the same cp1252 pipes as stderr."""
    with pytest.raises(SystemExit):
        cli.main(["--help"], config=cfg)
    out = capsys.readouterr().out
    assert "Mainframe" in out and out.isascii(), out


def test_install_task_explains_a_missing_task_scheduler(cfg, capsys):
    """the CLI supports Windows and POSIX; `schtasks` does not exist off Windows
    and a traceback is not the promised guidance."""
    assert cli.main(["install-task"], config=cfg, runner=_no_schtasks) == 1
    err = capsys.readouterr().err
    assert "Windows-only" in err and err.isascii(), err


def _no_schtasks(argv):
    raise FileNotFoundError("schtasks")


def _never_spawn(config):
    raise AssertionError("should have started through the logon task")


def _health(state, nonce="n"):
    """A daemon that answers /healthz only while `state['up']` is true, and
    reports `state['readiness']` (the next one, when it is a list)."""
    def body(_):
        rd = state.get("readiness")
        if isinstance(rd, list):
            rd = rd.pop(0) if len(rd) > 1 else rd[0]
        return {"ok": state["up"], "nonce": nonce, "readiness": rd}
    return {("GET", "/healthz"): body}


_WARMING = {"state": "warming", "reason": "loading models"}
_READY = {"state": "ready", "reason": None}
_IDLE = {"state": "idle", "reason": "embedder absent; it loads on the first search"}
_BLOCKED = {"state": "blocked",
            "reason": "embedder failed: cuda gone; retry in 900s; `mainframe reload` clears the backoff"}


def test_up_waits_while_the_daemon_is_warming(cfg, monkeypatch):
    """A warming daemon becomes ready on its own, so `up` returning means
    ready - not merely answering. No config is consulted: the daemon says."""
    monkeypatch.setattr(cli, "START_POLL_SECONDS", 0.0)
    write_discovery(cfg, 1234, "n")
    state = {"up": False, "readiness": [_WARMING] * 4 + [_READY]}

    def runner(argv):
        state["up"] = True
        return 0

    http = FakeHTTP(_health(state))
    assert cli.main(["up"], config=cfg, http=http, runner=runner, spawn=_never_spawn) == 0
    assert state["readiness"] == [_READY]           # it polled through every warming answer
    assert len(http.calls) == 5


def test_up_returns_on_an_idle_daemon_and_names_the_key(cfg, capsys):
    """Idle means nothing will load until a search, so waiting would wait
    forever. Say what is true and where the other behaviour lives."""
    write_discovery(cfg, 1234, "n")
    state = {"up": False, "readiness": _IDLE}

    def runner(argv):
        state["up"] = True
        return 0

    assert cli.main(["up"], config=cfg, http=FakeHTTP(_health(state)), runner=runner, spawn=_never_spawn) == 0
    err = capsys.readouterr().err
    assert "service.prewarm: true" in err and "first search" in err and err.isascii(), err


def test_up_reports_a_blocked_daemon_at_once(cfg, capsys):
    """A failed load backs off for minutes; polling a 120 s timeout against it
    would report a timeout instead of the failure. Blocked is final: reported
    at once, with the remedy - whether or not prewarm is on."""
    write_discovery(cfg, 1234, "n")
    state = {"up": True, "readiness": _BLOCKED}
    http = FakeHTTP(_health(state))
    assert cli.main(["up", "--timeout", "1000"], config=cfg, http=http, runner=_no_schtasks,
                    spawn=_never_spawn) == 1
    assert len(http.calls) <= 3
    err = capsys.readouterr().err
    assert "blocked" in err and "cuda gone" in err and "mainframe reload" in err and err.isascii(), err


def test_up_already_running_still_waits_for_readiness(cfg, capsys, monkeypatch):
    """A daemon mid-warm-up answers already; `up` must not say "already
    running" and return 0 while it is not yet ready."""
    monkeypatch.setattr(cli, "START_POLL_SECONDS", 0.0)
    write_discovery(cfg, 1234, "n")
    state = {"up": True, "readiness": [_WARMING, _READY]}
    http = FakeHTTP(_health(state))
    assert cli.main(["up"], config=cfg, http=http, runner=_no_schtasks, spawn=_never_spawn) == 0
    assert "already running" in capsys.readouterr().out
    assert state["readiness"] == [_READY]


def test_up_names_the_phase_at_timeout(cfg, capsys):
    write_discovery(cfg, 1234, "n")
    state = {"up": True, "readiness": _WARMING}
    assert cli.main(["up", "--timeout", "0"], config=cfg, http=FakeHTTP(_health(state)),
                    runner=_no_schtasks, spawn=_never_spawn) == 1
    err = capsys.readouterr().err
    assert "still warming" in err and "loading models" in err and err.isascii(), err


def test_up_accepts_a_daemon_that_does_not_report_readiness(cfg, capsys):
    """A daemon from before this field (mid-upgrade) answers without it; that
    is the old contract, and the old contract is 0."""
    write_discovery(cfg, 1234, "n")
    state = {"up": True}
    assert cli.main(["up"], config=cfg, http=FakeHTTP(_health(state)), runner=_no_schtasks,
                    spawn=_never_spawn) == 0
    assert capsys.readouterr().err == ""


def test_only_the_start_probe_is_quiet(cfg, capsys, monkeypatch):
    """`schtasks /Run` is a probe: on a machine with no logon task its own
    'cannot find the file' landed on the terminal right above our explanation.
    `install-task` is the user's own command, and schtasks' output IS its
    report - that one stays loud."""
    seen = {"run": [], "call": []}
    write_discovery(cfg, 1234, "n")
    state = {"up": False, "readiness": _READY}

    def fake_run(argv, **kw):
        seen["run"].append((argv, kw))
        state["up"] = True
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli.subprocess, "call", lambda argv, **kw: seen["call"].append(argv) or 0)
    assert cli.main(["up"], config=cfg, http=FakeHTTP(_health(state)), spawn=_never_spawn) == 0
    [(argv, kw)] = seen["run"]
    assert argv[:2] == ["schtasks", "/Run"]
    assert kw["stdout"] == cli.subprocess.DEVNULL and kw["stderr"] == cli.subprocess.DEVNULL
    assert cli.main(["install-task"], config=cfg) == 0
    assert [a[:2] for a in seen["call"]] == [["schtasks", "/Create"]] and len(seen["run"]) == 1
    assert capsys.readouterr().err == ""


def test_up_falls_back_to_a_detached_serve_when_no_task_answers(cfg, capsys):
    """Task Scheduler refuses to register a task without elevation on some
    machines. "the scheduler is out of reach" is not a reason to leave the
    operator with no daemon; it is a reason to start one another way, and say so."""
    write_discovery(cfg, 1234, "n")
    state = {"up": False}
    spawned = []

    def spawn(config):
        spawned.append(config)
        state["up"] = True          # the detached serve comes up

    assert cli.main(["up"], config=cfg, http=FakeHTTP(_health(state)),
                    runner=_no_schtasks, spawn=spawn) == 0
    assert spawned == [cfg]
    err = capsys.readouterr().err
    assert "detached" in err and "install-task" in err and err.isascii(), err


def test_start_says_when_the_launcher_produced_no_process_at_all(cfg, capsys):
    """The failure this exists to close: a launcher returns cleanly, nothing
    starts, and no log line is written anywhere. The growth of daemon.out is what
    separates that from "it started and died"."""
    rc = cli.start_daemon(cfg, lambda argv: 0, _never_spawn, FakeHTTP({}), 0)
    assert rc == 1
    err = capsys.readouterr().err
    assert "no process at all" in err and err.isascii(), err


def test_start_says_when_something_started_and_then_died(cfg, capsys):
    out = cli.daemon_out(cfg)

    def spawn(config):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("Traceback (most recent call last): boom\n", encoding="utf-8")

    rc = cli.start_daemon(cfg, _no_schtasks, spawn, FakeHTTP({}), 0)
    assert rc == 1
    err = capsys.readouterr().err
    assert "grew" in err and "bytes" in err and err.isascii(), err


def test_a_detached_serve_redirects_its_output(cfg, monkeypatch):
    """Load-bearing, not tidiness: a detached child on Windows has no console,
    and the model loader's progress against an invalid handle kills the daemon
    seconds after it starts, leaving no process and no log line."""
    seen = {}

    def fake_popen(argv, **kw):
        seen.update(argv=argv, **kw)
        return object()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    cli.spawn_detached(cfg)
    assert seen["argv"][1:] == cli.SERVE_ARGV       # the supervised form: the fallback daemon comes back too
    assert seen["stdout"] is not None
    assert seen["stderr"] == cli.subprocess.STDOUT and seen["stdin"] == cli.subprocess.DEVNULL


def _refuse_open(monkeypatch, path, times=None):
    """Make opening `path` raise PermissionError - `times` times, or always.

    The real race is a dying daemon still holding the stdout handle it inherited,
    which is a moment wide and cannot be scheduled. Refusing the open reaches the
    same branch deterministically.
    """
    real_open = builtins.open
    state = {"n": 0}

    def opener(file, *a, **kw):
        if str(file) == str(path) and (times is None or state["n"] < times):
            state["n"] += 1
            raise PermissionError(13, "Permission denied")
        return real_open(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", opener)
    return state


def test_a_log_that_cannot_be_opened_never_prevents_a_start(cfg, monkeypatch):
    """A restart once died with PermissionError on daemon.out and left the daemon
    DOWN. Availability is not worth trading for a log file - and DEVNULL is a
    VALID handle, so the invalid-handle death the redirect exists to prevent
    still cannot happen."""
    out = cli.daemon_out(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("prior output\n", encoding="utf-8")
    _refuse_open(monkeypatch, out)
    monkeypatch.setattr(cli, "LOG_WAIT_SECONDS", 0.0)
    seen = {}
    monkeypatch.setattr(cli.subprocess, "Popen", lambda argv, **kw: seen.update(kw) or object())

    assert cli.spawn_detached(cfg) is False          # it started, and says it is not watching
    assert seen["stdout"] == cli.subprocess.DEVNULL
    assert out.read_text(encoding="utf-8") == "prior output\n"   # nothing truncated


def test_the_log_open_waits_for_a_previous_daemon_to_let_go(cfg, monkeypatch):
    """The handoff is the common case: the dying daemon releases the lock before
    the OS closes the handle it inherited, so a brief wait is what the resource
    actually needs."""
    out = cli.daemon_out(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    refusals = _refuse_open(monkeypatch, out, times=2)
    monkeypatch.setattr(cli, "LOG_POLL_SECONDS", 0.0)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda argv, **kw: object())

    assert cli.spawn_detached(cfg) is True
    assert refusals["n"] == 2                        # it really retried, then captured


def test_start_will_not_claim_nothing_ran_when_it_was_not_watching(cfg, capsys):
    """The dangerous failure: an unchanged daemon.out reads as "no process at
    all", which is a confident wrong diagnosis when output went to DEVNULL."""
    rc = cli.start_daemon(cfg, _no_schtasks, lambda config: False, FakeHTTP({}), 0)
    assert rc == 1
    err = capsys.readouterr().err
    assert "says nothing about this start" in err and "no process at all" not in err
    assert err.isascii(), err


def test_restart_is_stop_then_start(cfg):
    write_discovery(cfg, 1234, "n")
    state = {"up": True}

    def shutdown(_body):
        state["up"] = False
        return {"stopping": True}

    def runner(argv):
        state["up"] = True          # the logon task brought it back
        return 0

    http = FakeHTTP({**_health(state), ("POST", "/shutdown"): shutdown})
    assert cli.main(["restart"], config=cfg, http=http, runner=runner, spawn=_never_spawn) == 0
    assert "/shutdown" in [c[1] for c in http.calls]


def test_restart_does_not_try_to_stop_a_daemon_that_is_already_down(cfg):
    """restart is stop-then-start, and stopping what is already stopped is not
    a step - it is an error waiting to be reported."""
    write_discovery(cfg, 1234, "n")
    state = {"up": False}

    def runner(argv):
        state["up"] = True
        return 0

    http = FakeHTTP(_health(state))
    assert cli.main(["restart"], config=cfg, http=http, runner=runner, spawn=_never_spawn) == 0
    assert all(c[1] != "/shutdown" for c in http.calls)


def test_rebuild_is_offline_only_even_when_the_daemon_is_up(cfg, store, tmp_path, capsys):
    """A rebuild DROPS the table, so it is never routed to a daemon that is
    serving readers. The CLI takes `.daemon.lock` itself; a live daemon holding
    it is the refusal, and no request is sent to the daemon at all."""
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    http = FakeHTTP(_up(cfg))
    with held(Daemon(cfg, app=app)):
        with pytest.raises(SystemExit) as e:
            cli.main(["rebuild"], config=cfg, http=http)
        assert e.value.code == 3
    err = capsys.readouterr().err
    assert "mainframe down" in err and "offline only" in err and err.isascii()
    assert http.calls == []


def test_offline_rebuild_refuses_when_locked(cfg, store, tmp_path, capsys):
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    with held(Daemon(cfg, app=app)):
        with pytest.raises(SystemExit) as e:
            cli.run_offline_rebuild(cfg, app=app)
        assert e.value.code == 3
    assert capsys.readouterr().err.isascii()
    write_md(Path(cfg["paths"]["repos_dir"]) / "p" / "docs" / "a.md", "# a\n\nalpha.\n")
    assert cli.run_offline_rebuild(cfg, app=app)["upserted_rows"] == 1


def test_task_xml_and_install_dry_run(cfg, capsys):
    xml = cli.task_xml(cfg)
    root = ET.fromstring(xml)
    text = ET.tostring(root, encoding="unicode")
    assert "PT0S" in text and "RestartOnFailure" in text and "IgnoreNew" in text and "LogonTrigger" in text
    assert "mainframe.adapters.cli" in text and cfg["paths"]["mainframe_dir"].replace("\\", "/") in text.replace("\\", "/")
    assert cli.schtasks_command(Path("x.xml"))[:3] == ["schtasks", "/Create", "/F"]
    ran = []
    assert cli.main(["install-task"], config=cfg, runner=lambda argv: ran.append(argv) or 0) == 0
    assert ran and ran[0][0] == "schtasks" and Path(ran[0][-1]).exists()
    assert cli.main(["install-task", "--dry-run"], config=cfg) == 0 and "schtasks" in capsys.readouterr().out


def test_the_logon_task_runs_without_a_console_window(cfg, tmp_path, monkeypatch):
    """An InteractiveToken task running python.exe opens a console at every
    logon. pythonw.exe, installed beside it, is the same interpreter without
    one; where there is none (not Windows), the interpreter stays as it is."""
    (tmp_path / "python.exe").touch()
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    assert "<Command>" + str(tmp_path / "python.exe") + "</Command>" in cli.task_xml(cfg)
    (tmp_path / "pythonw.exe").touch()
    assert "<Command>" + str(tmp_path / "pythonw.exe") + "</Command>" in cli.task_xml(cfg)


def test_the_logon_task_belongs_to_the_user_who_registers_it(cfg, monkeypatch):
    """A trigger on ANY user's logon needs an elevated shell to register; one
    scoped to the registering user, run with their own non-elevated token, does
    not. That difference is the whole of 'install-task needs admin'."""
    monkeypatch.setenv("USERDOMAIN", "HOST")
    monkeypatch.setattr("getpass.getuser", lambda: "alice")
    root = ET.fromstring(cli.task_xml(cfg).split("\n", 1)[1])
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    assert root.find("t:Triggers/t:LogonTrigger/t:UserId", ns).text == "HOST\\alice"
    principal = root.find("t:Principals/t:Principal", ns)
    assert principal.find("t:UserId", ns).text == "HOST\\alice"
    assert principal.find("t:LogonType", ns).text == "InteractiveToken"
    assert principal.find("t:RunLevel", ns).text == "LeastPrivilege"


def test_a_windowless_serve_keeps_what_a_crash_says(cfg, monkeypatch):
    """Under pythonw there is no stderr, so nothing - not even a traceback -
    would say why the logon task's daemon died. `serve` adopts daemon.out, the
    file `spawn_detached` already hands its child."""
    import mainframe.service.daemon as daemon_mod

    def crashing_serve(_config):
        print("traceback would land here", file=sys.stderr)

    monkeypatch.setattr(daemon_mod, "serve", crashing_serve)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    assert cli.main(["serve", "--once"], config=cfg) == 0
    sys.stderr.flush()
    assert "traceback would land here" in cli.daemon_out(cfg).read_text(encoding="utf-8")
    # ... and the supervisor, which is what the task actually runs, writes there too
    monkeypatch.setattr(cli, "supervise", lambda run_child, **kw: print("supervisor says", file=sys.stderr) or 0)
    assert cli.main(["serve"], config=cfg) == 0
    sys.stderr.flush()
    assert "supervisor says" in cli.daemon_out(cfg).read_text(encoding="utf-8")


# ---- crash recovery: `serve` is a supervisor; `serve --once` is the daemon process ----

def _children(*codes):
    """A `run_child` that exits with each code in turn, counting its runs."""
    it = iter(codes)
    runs = []

    def run():
        runs.append(1)
        return next(it)
    return run, runs


def test_the_logon_task_action_is_unchanged_and_now_supervises(cfg, monkeypatch):
    """Task Scheduler's RestartOnFailure does not fire on an action that dies
    later (measured: a killed daemon stayed down for 150 s). What brings the
    daemon back is the process the task runs - and the task already registered
    was created from an elevated shell, so its Arguments cannot change without
    a prompt. `serve` itself is therefore the supervisor; `--once` is the bare
    process it runs, and the one a developer debugs."""
    assert "<Arguments>-m mainframe.adapters.cli serve</Arguments>" in cli.task_xml(cfg)
    assert cli.ONCE_ARGV == cli.SERVE_ARGV + ["--once"]
    seen = []
    monkeypatch.setattr(cli, "supervise", lambda run_child, **kw: seen.append(run_child) or 7)
    assert cli.main(["serve"], config=cfg) == 7
    assert seen == [cli.serve_child]
    import mainframe.service.daemon as daemon_mod
    monkeypatch.setattr(daemon_mod, "serve", lambda config: seen.append("once"))
    assert cli.main(["serve", "--once"], config=cfg) == 0
    assert seen == [cli.serve_child, "once"]


def test_supervise_restarts_a_daemon_that_died(capsys):
    run, runs = _children(1, 0)
    slept = []
    assert cli.supervise(run, sleep=slept.append, clock=lambda: 0.0) == 0
    assert len(runs) == 2 and slept == [cli.RESTART_DELAYS[0]]
    err = capsys.readouterr().err
    assert "restarting in 1 s" in err and err.isascii(), err


@pytest.mark.parametrize("code", [0, cli.LOCK_HELD_EXIT])
def test_supervise_never_undoes_an_intentional_stop(code):
    """Exit 0 is the daemon deciding to stop (`down`, `/shutdown`); exit 3 is
    another instance already serving. Neither is a death to recover from."""
    run, runs = _children(code, 0)
    slept = []
    assert cli.supervise(run, sleep=slept.append, clock=lambda: 0.0) == code
    assert len(runs) == 1 and slept == []


def test_supervise_gives_up_on_a_daemon_that_dies_at_startup(capsys):
    """A daemon that cannot start is restarted a bounded number of times, with
    growing delays, and the last exit code is then returned - no restart storm."""
    run, runs = _children(*([1] * 20))
    slept = []
    assert cli.supervise(run, sleep=slept.append, clock=lambda: 0.0) == 1
    assert len(runs) == len(cli.RESTART_DELAYS) + 1
    assert slept == list(cli.RESTART_DELAYS) and slept == sorted(slept)
    assert sum(slept) < 60                      # the whole budget fits in the minute the first restart promises
    assert "giving up" in capsys.readouterr().err


def test_a_daemon_that_served_a_while_earns_its_restarts_back():
    """Crashing after an hour of service is not failing to start: the next
    start is expected to work, so the budget does not run out across uptime."""
    now = [0.0]

    def run():
        now[0] += cli.STABLE_SECONDS           # each child lives exactly long enough to count as started
        return 1
    slept = []
    calls = {"n": 0}

    def sleep(s):
        slept.append(s)
        calls["n"] += 1
        if calls["n"] == 3 * len(cli.RESTART_DELAYS):
            raise StopIteration             # enough: it would go on forever, which is the point
    with pytest.raises(StopIteration):
        cli.supervise(run, sleep=sleep, clock=lambda: now[0])
    assert set(slept) == {cli.RESTART_DELAYS[0]}


def test_serve_child_runs_once_and_waits_out_an_interrupt(monkeypatch):
    """Ctrl-C reaches both processes. The child shuts down gracefully (that can
    take seconds); the supervisor waits for it and calls the stop clean."""
    seen = {}

    class FakeProc:
        waits = 0
        def wait(self):
            FakeProc.waits += 1
            if FakeProc.waits == 1:
                raise KeyboardInterrupt
            return 0

    def fake_popen(argv, **kw):
        seen.update(argv=argv, **kw)
        return FakeProc()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    assert cli.serve_child() == 0
    assert seen["argv"] == [sys.executable, *cli.ONCE_ARGV] and seen["stdin"] == cli.subprocess.DEVNULL
    assert seen["env"][cli.SUPERVISOR_PID_ENV] == str(os.getpid())     # so discovery can name the supervisor
    assert FakeProc.waits == 2


def test_discovery_names_the_supervisor_and_down_waits_for_it_too(cfg, monkeypatch):
    """The supervisor IS the logon task's instance. A task with IgnoreNew
    drops the next start while that instance runs, so a `down` that returned
    when only the child was gone would make the `up` after it start nothing -
    the #59 race again, one process up. The child records the pid its
    supervisor handed it, and `down` waits for both."""
    monkeypatch.setenv(cli.SUPERVISOR_PID_ENV, "4242")
    write_discovery(cfg, 1234, "n1")
    assert cli.read_discovery(cfg)["supervisor_pid"] == 4242 and cli.read_discovery(cfg)["pid"] == os.getpid()
    polls = []
    monkeypatch.setattr(cli, "process_exited", lambda pid: polls.append(pid) or pid != 4242 or len(polls) >= 4)
    http = FakeHTTP({("GET", "/healthz"): lambda b: {"ok": True, "nonce": "n1"},
                     ("POST", "/shutdown"): lambda b: {"stopping": True}})
    assert cli.main(["down"], config=cfg, http=http) == 0
    assert os.getpid() in polls and polls.count(4242) >= 2         # kept waiting on the supervisor alone
    monkeypatch.delenv(cli.SUPERVISOR_PID_ENV)
    write_discovery(cfg, 1234, "n1")
    assert "supervisor_pid" not in cli.read_discovery(cfg)         # a `--once` daemon has none to name


def test_serve_child_hands_its_log_to_the_child(monkeypatch, tmp_path):
    """Under pythonw the supervisor writes to daemon.out; the child must write
    there too, or a crash under the logon task would again say nothing."""
    seen = {}
    monkeypatch.setattr(cli.subprocess, "Popen",
                        lambda argv, **kw: seen.update(**kw) or type("P", (), {"wait": lambda self: 0})())
    with open(tmp_path / "daemon.out", "ab") as sink:
        monkeypatch.setattr(sys, "stdout", sink)
        monkeypatch.setattr(sys, "stderr", sink)
        assert cli.serve_child() == 0
    assert seen["stdout"] is sink and seen["stderr"] is sink
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    assert cli.serve_child() == 0
    assert seen["stdout"] is None and seen["stderr"] is None     # nothing to hand down: inherit


def test_down_waits_for_lock(cfg, monkeypatch):
    http = FakeHTTP({**_up(cfg), ("POST", "/shutdown"): lambda b: {"stopping": True}})
    assert cli.main(["down"], config=cfg, http=http) == 0            # lock is free immediately


def test_down_waits_for_the_daemon_process_not_just_its_lock(cfg, monkeypatch):
    """The daemon releases the lock, then tears down for seconds more. A logon
    task whose instance is still running IGNORES the next start, so an `up`
    right after a `down` that returned on the lock alone started nothing
    (measured: lock free at 24.8 s, process gone at 28.1 s)."""
    polls = []
    monkeypatch.setattr(cli, "process_exited", lambda pid: polls.append(pid) or len(polls) >= 3)
    http = FakeHTTP({**_up(cfg), ("POST", "/shutdown"): lambda b: {"stopping": True}})
    assert cli.main(["down"], config=cfg, http=http) == 0
    assert polls == [os.getpid()] * 3                                # the pid discovery recorded


def test_down_reports_a_daemon_process_that_outlives_its_lock(cfg, monkeypatch, capsys):
    monkeypatch.setattr(cli, "process_exited", lambda pid: False)
    clock = {"t": 0.0}

    def stepping_monotonic():
        clock["t"] += 5.0
        return clock["t"]
    monkeypatch.setattr(cli.time, "monotonic", stepping_monotonic)
    http = FakeHTTP({**_up(cfg), ("POST", "/shutdown"): lambda b: {"stopping": True}})
    assert cli.main(["down"], config=cfg, http=http) == 1
    assert f"process {os.getpid()} is still running" in capsys.readouterr().err


def test_process_exited_tells_a_running_process_from_a_finished_one():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert REAL_PROCESS_EXITED(child.pid) is False
    finally:
        child.kill()
        child.wait(10)
    assert REAL_PROCESS_EXITED(child.pid) is True


def test_daemon_unreachable_mid_command(cfg, capsys):
    write_discovery(cfg, 1234, "n1")
    http = RaisingHTTP("n1")
    assert cli.main(["status"], config=cfg, http=http) == 1
    assert "could not reach the daemon" in capsys.readouterr().err


def test_emit_is_ascii_safe_and_round_trips(capsys):
    """Document content (a heading, a snippet) can carry an arrow, an em dash,
    or any other non-cp1252 character. `_emit` must still print something a
    Windows cp1252 console can encode, and it must stay valid JSON that
    round-trips to the exact original string."""
    payload = {"results": [{"snippet": "lock → discovery — write"}]}
    cli._emit(payload)
    out = capsys.readouterr().out
    out.encode("cp1252")               # must not raise UnicodeEncodeError
    assert json.loads(out) == payload


def test_call_reports_connectivity_error_as_unreachable(cfg, capsys):
    """A real connection/transport failure still gets the connectivity
    message — this is the one case that message is accurate for."""
    write_discovery(cfg, 1234, "n1")
    http = RaisingHTTP("n1")
    assert cli.main(["status"], config=cfg, http=http) == 1
    assert "could not reach the daemon" in capsys.readouterr().err


def test_call_reports_non_connectivity_error_with_its_own_message(cfg, capsys):
    """The daemon answered fine; something else broke while handling the
    result (e.g. a printing/encoding bug). That must not be blamed on
    reachability — it should surface with its own message."""
    write_discovery(cfg, 1234, "n1")
    http = BrokenHTTP("n1")
    assert cli.main(["status"], config=cfg, http=http) == 1
    err = capsys.readouterr().err
    assert "could not reach the daemon" not in err
    assert "boom: malformed response body" in err


def test_down_is_idempotent_when_daemon_not_running(cfg, capsys):
    assert cli.main(["down"], config=cfg, http=FakeHTTP({})) == 0
    assert "already stopped" in capsys.readouterr().out


def test_capture_missing_file_prints_clean_error(cfg, capsys):
    http = FakeHTTP({**_up(cfg)})
    missing = str(Path(cfg["paths"]["mainframe_dir"]) / "does-not-exist.md")
    assert cli.main(["capture", "--file", missing, "--project", "p"], config=cfg, http=http) == 1
    err = capsys.readouterr().err
    assert "cannot read" in err and missing in err


def test_events_url_encodes_kind_filter(cfg):
    http = FakeHTTP({**_up(cfg), ("GET", "/events"): lambda b: {"events": []}})
    assert cli.main(["events", "--kind", "a b&c"], config=cfg, http=http) == 0
    query = next(q for m, p, b, q in http.calls if p == "/events")
    parsed = urllib.parse.parse_qs(query)
    assert parsed["kind"] == ["a b&c"] and parsed["n"] == ["20"]


def test_down_times_out_when_lock_is_held(cfg, store, monkeypatch, capsys, tmp_path):
    """The wait is the DAEMON's own shutdown bound (+10 s), not an unrelated
    30 s: the daemon finishes its in-flight job before releasing the lock, and
    that job can be a rescan of thousands of files. Reported live: `down` gave
    up at 30 s and returned 1 while a correct, bounded shutdown was underway."""
    app = Mainframe(cfg, models=fake_models(cfg), events=EventStore(tmp_path / "ev.db"), store=store)
    cfg["service"]["shutdown_timeout_seconds"] = 2          # -> the CLI waits 12 s
    holder = Daemon(cfg, app=app)
    assert holder.acquire()
    http = FakeHTTP({**_up(cfg), ("POST", "/shutdown"): lambda b: {"stopping": True}})

    # 3 s per monotonic() call: two full loop passes (each a real, still-Timeout
    # acquire + sleep) and a progress line, then past the 12 s bound.
    clock = {"t": 0.0}

    def stepping_monotonic():
        now = clock["t"]
        clock["t"] += 3.0
        return now
    monkeypatch.setattr(cli.time, "monotonic", stepping_monotonic)

    sleep_calls = {"n": 0}

    def counting_sleep(seconds):
        sleep_calls["n"] += 1

    monkeypatch.setattr(cli.time, "sleep", counting_sleep)

    acquire_calls = {"n": 0}
    original_acquire = filelock.FileLock.acquire

    def counting_acquire(self, *args, **kwargs):
        acquire_calls["n"] += 1
        return original_acquire(self, *args, **kwargs)     # still really tries; still really raises Timeout

    monkeypatch.setattr(filelock.FileLock, "acquire", counting_acquire)

    try:
        assert cli.main(["down"], config=cfg, http=http) == 1
        err = capsys.readouterr().err
        assert "still holds the lock" in err                      # the TRUE-timeout case still reports
        # the user is told the wait is deliberate, not a hang
        assert "waiting for the daemon to finish its current job" in err
        assert "elapsed)" in err and err.isascii()
        # proves the retry loop body actually ran against the genuinely-held lock,
        # not just the "deadline already past" branch
        assert sleep_calls["n"] >= 2
        assert acquire_calls["n"] >= 2
    finally:
        holder.release()
