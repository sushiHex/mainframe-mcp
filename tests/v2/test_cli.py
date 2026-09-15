import json
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


def test_up_and_install_task_explain_a_missing_task_scheduler(cfg, capsys):
    """the CLI supports Windows and POSIX; `schtasks` does not exist off Windows
    and a traceback is not the promised guidance."""
    def no_schtasks(argv):
        raise FileNotFoundError("schtasks")

    assert cli.main(["up"], config=cfg, http=FakeHTTP({}), runner=no_schtasks) == 1
    err = capsys.readouterr().err
    assert "Windows-only" in err and "mainframe serve" in err and err.isascii(), err
    assert cli.main(["install-task"], config=cfg, runner=no_schtasks) == 1
    err = capsys.readouterr().err
    assert "Windows-only" in err and err.isascii(), err


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


def test_down_waits_for_lock(cfg, monkeypatch):
    http = FakeHTTP({**_up(cfg), ("POST", "/shutdown"): lambda b: {"stopping": True}})
    assert cli.main(["down"], config=cfg, http=http) == 0            # lock is free immediately


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
