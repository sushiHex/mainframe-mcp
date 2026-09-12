import contextlib
import json
from pathlib import Path
import sys

import pytest
from starlette.testclient import TestClient

from mainframe.adapters.validation import ValidationRun
from mainframe.app import Mainframe
from mainframe.service.daemon import Daemon
from v2.conftest import held
from v2.fakes import fake_models


@contextlib.contextmanager
def local_session(config, log, timeout):
    """Real daemon, watcher, index, and routes; only models and sockets are fake."""
    app = Mainframe(config, models=fake_models(config))
    daemon = Daemon(config, app=app)
    headers = {"Authorization": "Bearer " + config["service"]["token"]}
    try:
        with held(daemon), TestClient(daemon.build(), base_url="http://127.0.0.1:8420",
                                     headers=headers) as client:
            yield client
    finally:
        app.models.invalidate_all()


def runner(tmp_path, cfg, **kwargs):
    cfg["service"]["tick_seconds"] = 0.02
    cfg["index"]["debounce_seconds"] = 0.02
    return ValidationRun(tmp_path / "validation", cfg, timeout=20,
                         session_factory=local_session, **kwargs)


@pytest.mark.slow
def test_smoke_uses_real_lifecycle_and_preserves_input_configuration(tmp_path, cfg):
    before = json.loads(json.dumps(cfg))
    run = runner(tmp_path, cfg)
    original = json.loads(json.dumps(cfg))
    report = run.run()
    assert cfg == original
    assert report["status"] == "passed"
    assert report["mutation_cycles"] == 1
    assert report["searches"] and report["health"]["requests"] > 0
    assert report["health"]["failures"] == []
    assert report["restart_upserted_rows"] == 0
    assert report["failed_jobs"] == 0
    assert report["rescan_count"] > 0
    assert {"first_search", "warm_search"} <= set(report["timings"])
    assert json.loads((run.output / "report.json").read_text())["status"] == "passed"
    assert list(Path(before["paths"]["mainframe_dir"]).iterdir()) == []
    assert run.config["paths"]["include_projects"] == ["validation"]
    assert run.config["contextual"]["enabled"] is False


def test_refuses_existing_output_without_modifying_it(tmp_path, cfg):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "report.json"
    sentinel.write_text("keep me")
    with pytest.raises(FileExistsError):
        ValidationRun(output, cfg).run()
    assert sentinel.read_text() == "keep me"
    assert list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize("git_entry", ["directory", "worktree-file"])
def test_refuses_output_inside_a_git_worktree(tmp_path, cfg, git_entry):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    marker = checkout / ".git"
    if git_entry == "directory":
        marker.mkdir()
    else:
        marker.write_text("gitdir: ../metadata")
    with pytest.raises(ValueError, match="outside Git"):
        ValidationRun(checkout / "nested" / "output", cfg).run()
    assert not (checkout / "nested").exists()


@pytest.mark.slow
def test_failure_still_closes_session_and_records_failure(tmp_path, cfg, monkeypatch):
    run = runner(tmp_path, cfg)
    closed = []

    @contextlib.contextmanager
    def session(*args):
        try:
            with local_session(*args) as client:
                yield client
        finally:
            closed.append(True)

    run.session_factory = session

    def fail(*args):
        raise RuntimeError("synthetic validation failure")

    monkeypatch.setattr(run, "mutation_cycle", fail)
    with pytest.raises(RuntimeError, match="synthetic validation failure"):
        run.run()
    assert closed == [True]
    report = json.loads((run.output / "report.json").read_text())
    assert report["status"] == "failed"
    assert "synthetic validation failure" in report["error"]
    from filelock import FileLock
    with FileLock(str(run.state / ".daemon.lock")).acquire(timeout=0):
        pass


def test_nonfinite_or_negative_duration_is_rejected(tmp_path, cfg):
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="duration"):
            ValidationRun(tmp_path / "output", cfg, duration=value)


@pytest.mark.parametrize("foreign", [None, "discovery", "health"])
def test_daemon_session_identifies_launch_independently_of_launcher_pid(tmp_path, cfg, monkeypatch, foreign):
    from unittest.mock import Mock
    import httpx
    from mainframe.adapters import validation

    cfg["service"]["token"] = "synthetic-validation-token"
    state = Path(cfg["paths"]["mainframe_dir"])
    discovery = state / "daemon.json"
    process = Mock(pid=100, returncode=None)
    process.poll.side_effect = lambda: process.returncode
    client = Mock()
    client.post.return_value = httpx.Response(200, json={"stopping": True},
                                             request=httpx.Request("POST", "http://localhost/shutdown"))

    def finish(*args, **kwargs):
        process.returncode = 0
        if not foreign:
            discovery.unlink(missing_ok=True)

    process.wait.side_effect = finish
    process.terminate.side_effect = finish

    def launch(command, **kwargs):
        # A Windows venv launcher can publish its child's PID, not Popen.pid.
        nonce = "unrelated-instance" if foreign == "discovery" else command[-1]
        discovery.write_text(json.dumps({"pid": 200, "port": 8420, "nonce": nonce}))
        health_nonce = "unrelated-instance" if foreign == "health" else nonce
        client.get.return_value = httpx.Response(200, json={"ok": True, "nonce": health_nonce})
        return process

    monkeypatch.setattr(validation.subprocess, "Popen", launch)
    monkeypatch.setattr(validation.httpx, "Client", lambda **kwargs: client)
    if foreign:
        with pytest.raises(TimeoutError):
            with validation.daemon_session(cfg, tmp_path / "trial.log", 0.01):
                pytest.fail("attached to an unrelated daemon")
        client.post.assert_not_called()
        assert discovery.exists()
    else:
        with validation.daemon_session(cfg, tmp_path / "trial.log", 0.5) as connected:
            assert connected is client
        client.post.assert_called_once_with("/shutdown", timeout=10)
        process.terminate.assert_not_called()
        assert not discovery.exists()


def test_query_crossing_deadline_does_not_start_another_mutation(tmp_path, cfg, monkeypatch):
    from mainframe.adapters import validation
    run = runner(tmp_path, cfg)
    now, mutations = [0.0], []
    monkeypatch.setattr(validation.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(validation.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(run, "search", lambda: now.__setitem__(0, 4.0))
    monkeypatch.setattr(run, "mutation_cycle", lambda: mutations.append(True))
    monkeypatch.setattr(run, "indexed", lambda: {"index": {"rows": 2}, "daemon": {"queue_depth": 0}, "vram": {}})
    monkeypatch.setattr(run, "mark", lambda *args, **kwargs: None)
    run.observe(deadline=3.0, next_change=2.0)
    assert not mutations


@pytest.mark.slow
def test_model_load_failure_stops_validation_without_waiting_for_index_timeout(tmp_path, cfg, monkeypatch):
    from mainframe.core.models import Models

    def fail(config):
        raise RuntimeError("synthetic model load failure")

    monkeypatch.setattr(sys.modules[__name__], "fake_models",
                        lambda config: Models(config, embedder_factory=fail))
    run = runner(tmp_path, cfg)
    with pytest.raises(RuntimeError, match="model load failed"):
        run.run()
    assert json.loads((run.output / "report.json").read_text())["status"] == "failed"


def test_recorded_model_failure_wins_over_a_loading_status_snapshot(tmp_path, cfg, monkeypatch):
    run = runner(tmp_path, cfg)
    # Model state and recent events can be read on opposite sides of a failure.
    monkeypatch.setattr(run, "request", lambda *args: {
        "models": {"embedder": {"state": "loading"}},
        "recent_events": [{"kind": "model.failed"}, {"kind": "job.failed"}],
    })
    with pytest.raises(RuntimeError, match="model load failed"):
        run.indexed()
