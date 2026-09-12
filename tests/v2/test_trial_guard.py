"""Resource observation, worker ownership, and pacing without models or CUDA."""

import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def load_guard():
    if 'trial_guard' in sys.modules:
        return sys.modules['trial_guard']
    spec = importlib.util.spec_from_file_location(
        "trial_guard", Path(__file__).resolve().parents[2] / "eval" / "trial_guard.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def guard():
    return load_guard()


def healthy(guard, **changes):
    values = dict(at=100., commit_used_gib=40., commit_limit_gib=100.,
                  gpu_free_gib=22., gpu_total_gib=24.)
    values.update(changes)
    return guard.Sample(**values)


@pytest.mark.parametrize('field,value', [('sample_s', 0), ('stale_s', float('nan')),
                                        ('idle_ratio', 0), ('min_idle_s', -1)])
def test_invalid_monitoring_settings_fail_closed(guard, field, value):
    with pytest.raises(ValueError):
        guard.TrialSettings(**{field: value})


@pytest.mark.parametrize('changes', [dict(gpu_free_gib=float('nan')),
                                      dict(commit_used_gib=101.), dict(gpu_free_gib=25.)])
def test_invalid_samples_are_not_headroom(guard, changes):
    with pytest.raises(ValueError, match='telemetry'):
        healthy(guard, **changes).validate()


def test_expired_or_cancelled_permit_prevents_forward(guard, tmp_path):
    path = tmp_path / 'permit.json'
    guard.write_json(path, {'nonce': 'test', 'at': 98., 'allowed': True})
    gate = guard.BatchGate(path, 'test', clock=lambda: 100.)
    with pytest.raises(guard.TrialAborted, match='stale'):
        gate.check()
    guard.write_json(path, {'nonce': 'test', 'at': 100., 'allowed': False,
                            'reason': 'host pressure'})
    with pytest.raises(guard.TrialAborted, match='host pressure'):
        gate.check()
    guard.write_json(path, {'nonce': 'different', 'at': 100., 'allowed': True})
    with pytest.raises(guard.TrialAborted, match='identity'):
        gate.check()


def test_pacing_keeps_real_forward_inputs_and_order(guard, tmp_path):
    import torch

    now = [100.]
    calls = []
    path = tmp_path / 'permit.json'

    def tick(seconds):
        now[0] += seconds
        guard.write_json(path, {'nonce': 'test', 'at': now[0], 'allowed': True})

    tick(0)
    gate = guard.BatchGate(path, 'test', clock=lambda: now[0], sleep=tick,
                           synchronize=lambda: None)

    class Model(torch.nn.Module):
        def forward(self, values):
            calls.append(values.tolist())
            tick(.1)
            return values * 2

    model = Model()
    gate.attach(model)
    try:
        assert model(torch.tensor([3, 1])).tolist() == [6, 2]
        assert model(torch.tensor([2])).tolist() == [4]
        assert calls == [[3, 1], [2]]
        assert gate.idle_seconds == pytest.approx(.3)
        assert gate.forwards == 2
    finally:
        gate.close()
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_failed_forward_cannot_retry_with_a_smaller_batch(guard, tmp_path):
    import torch

    path = tmp_path / 'permit.json'
    guard.write_json(path, {'nonce': 'test', 'at': 100., 'allowed': True})
    gate = guard.BatchGate(path, 'test', clock=lambda: 100., synchronize=lambda: None)
    calls = []

    class Model(torch.nn.Module):
        def forward(self, values):
            calls.append(values)
            raise torch.cuda.OutOfMemoryError('synthetic OOM')

    model = Model()
    gate.attach(model)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        model([1, 2])
    with pytest.raises(guard.TrialAborted):
        model([1])
    assert calls == [[1, 2]]
    gate.close()


def test_cuda_binding_leaves_the_allocator_uncapped(guard, tmp_path):
    from types import SimpleNamespace

    path = tmp_path / 'permit.json'
    guard.write_json(path, {'nonce': 'test', 'at': 100., 'allowed': True})
    gate = guard.BatchGate(path, 'test', clock=lambda: 100.)
    # Deliberately has no allocator or device-capacity API.
    cuda = SimpleNamespace(synchronize=lambda: None)
    gate.configure_cuda(SimpleNamespace(cuda=cuda))
    assert gate.synchronize is cuda.synchronize


class Lease:
    def __init__(self):
        self.events = []

    async def __aenter__(self):
        self.events.append('reserve')
        return self

    async def __aexit__(self, *exc):
        self.events.append('release')

    async def renew(self):
        self.events.append('renew')


class Worker:
    def __init__(self, codes):
        self.codes = iter(codes)
        self.stopped = False

    def poll(self):
        return next(self.codes, None)

    def stop(self):
        self.stopped = True


def test_memory_readings_and_elapsed_time_do_not_impose_policy_cutoffs(guard, tmp_path):
    lease = Lease()
    worker = Worker([0])
    now = [100.]

    def sample():
        now[0] += 1000
        return healthy(guard, at=now[0], commit_used_gib=99., gpu_free_gib=.5)

    def launch(permit, nonce):
        (tmp_path / 'worker').mkdir()
        guard.write_json(tmp_path / 'worker' / 'report.json', {'complete': True, 'guard_nonce': nonce})
        return worker

    result = asyncio.run(guard.supervise(
        guard.TrialSettings(), tmp_path, sample, launch, lease, clock=lambda: now[0]))
    assert result['status'] == 'completed' and worker.stopped
    assert lease.events == ['reserve', 'renew', 'release']
    assert result['last_sample']['commit_used_gib'] == 99.


@pytest.mark.parametrize('failure', ['telemetry', 'crash', 'cancel'])
def test_running_failure_stops_owned_worker_releases_lease_and_never_passes(guard, tmp_path, failure):
    lease = Lease()
    worker = Worker([17] if failure == 'crash' else [None])
    samples = iter([healthy(guard), healthy(guard)])

    def sample():
        value = next(samples, None)
        if value is not None:
            return value
        if failure == 'telemetry':
            raise OSError('synthetic sensor failure')
        if failure == 'cancel':
            raise asyncio.CancelledError()
        if failure == 'crash':
            return healthy(guard)

    result = asyncio.run(guard.supervise(guard.TrialSettings(), tmp_path, sample,
                                         lambda *a: worker, lease, clock=lambda: 100.))
    assert result['status'] == 'aborted'
    assert worker.stopped and lease.events[-1] == 'release'
    permit = json.loads((tmp_path / 'permit.json').read_text())
    assert permit['allowed'] is False
    if failure == 'crash':
        assert 'status 17' in result['reason']


def test_zero_exit_requires_complete_worker_report(guard, tmp_path):
    worker = Worker([0])
    result = asyncio.run(guard.supervise(guard.TrialSettings(), tmp_path, lambda: healthy(guard),
                                         lambda *a: worker, Lease(), clock=lambda: 100.))
    assert result['status'] == 'aborted'
    assert 'report' in result['reason']


def test_freshness_checked_even_when_resources_look_healthy(guard, tmp_path):
    result = asyncio.run(guard.supervise(guard.TrialSettings(), tmp_path,
                                         lambda: healthy(guard, at=90.),
                                         lambda *a: pytest.fail('worker launched'), Lease(), clock=lambda: 100.))
    assert result['status'] == 'refused' and 'stale' in result['reason']


def test_success_requires_matching_report_and_stops_before_releasing(guard, tmp_path):
    worker = Worker([0])

    class CheckedLease(Lease):
        async def __aexit__(self, *exc):
            assert worker.stopped
            await super().__aexit__(*exc)

    def launch(permit, nonce):
        (tmp_path / 'worker').mkdir()
        guard.write_json(tmp_path / 'worker' / 'report.json', {'complete': True, 'guard_nonce': nonce})
        return worker

    result = asyncio.run(guard.supervise(guard.TrialSettings(), tmp_path, lambda: healthy(guard),
                                         launch, CheckedLease(), clock=lambda: 100.))
    assert result['status'] == 'completed'
    assert result['responsiveness'] == 'unvalidated'


@pytest.mark.parametrize('failure', ['renewal', 'operator'])
def test_time_and_lease_failures_abort_the_worker(guard, tmp_path, failure):
    now = [100.]
    worker = Worker([None])

    class BrokenLease(Lease):
        async def renew(self):
            raise RuntimeError('lease lost')

    def sample():
        now[0] += 21
        return healthy(guard, at=now[0])

    if failure == 'operator':
        (tmp_path / 'cancel').touch()
    result = asyncio.run(guard.supervise(
        guard.TrialSettings(), tmp_path, sample,
        lambda *a: worker, BrokenLease(), clock=lambda: now[0]))
    assert result['status'] == 'aborted' and worker.stopped
    assert {'renewal': 'lease lost', 'operator': 'operator'}[failure] in result['reason']


def test_pacing_stops_when_telemetry_expires_during_idle(guard, tmp_path):
    import torch

    now = [100.]
    path = tmp_path / 'permit.json'
    guard.write_json(path, {'nonce': 'test', 'at': now[0], 'allowed': True})
    def cancel_after_delay(seconds):
        now[0] += seconds
        if now[0] > 101.2:
            guard.write_json(path, {'nonce': 'test', 'at': now[0], 'allowed': False})

    gate = guard.BatchGate(path, 'test', clock=lambda: now[0], min_idle_s=2.,
                           sleep=cancel_after_delay,
                           synchronize=lambda: None)
    model = torch.nn.Identity()
    gate.attach(model)
    model(torch.tensor([1]))
    with pytest.raises(guard.TrialAborted, match='cancelled'):
        model(torch.tensor([2]))
    assert gate.forwards == 1
    gate.close()


def test_permit_read_tolerates_transient_windows_sharing_violation(guard, tmp_path, monkeypatch):
    path = tmp_path / 'permit.json'
    guard.write_json(path, {'nonce': 'test', 'at': 100., 'allowed': True})
    original = Path.read_text
    calls = []

    def sharing_once(self, *args, **kwargs):
        calls.append(self)
        if len(calls) == 1:
            raise PermissionError('synthetic Windows file-replacement race')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', sharing_once)
    guard.BatchGate(path, 'test', clock=lambda: 100.).check()
    assert len(calls) == 2


def test_atomic_publication_tolerates_transient_reader_handle(guard, tmp_path, monkeypatch):
    path = tmp_path / 'permit.json'
    original = Path.replace
    calls = []

    def sharing_once(self, *args, **kwargs):
        calls.append(self)
        if len(calls) == 1:
            raise PermissionError('synthetic Windows reader handle')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'replace', sharing_once)
    guard.write_json(path, {'nonce': 'test', 'allowed': False})
    assert json.loads(path.read_text())['allowed'] is False
    assert len(calls) == 2


@pytest.mark.parametrize('cancel', [False, True])
def test_delayed_heartbeat_pauses_forward_until_fresh_or_cancelled(guard, tmp_path, cancel):
    import torch

    path = tmp_path / 'permit.json'
    now = [100.]
    calls = []
    guard.write_json(path, {'nonce': 'test', 'at': 98., 'allowed': True})

    def refresh(seconds):
        assert not calls  # Never submit GPU work under a stale permit.
        now[0] += seconds
        guard.write_json(path, {'nonce': 'test', 'at': now[0], 'allowed': not cancel})

    class Model(torch.nn.Module):
        def forward(self, values):
            calls.append(values)
            return values

    gate = guard.BatchGate(path, 'test', clock=lambda: now[0], sleep=refresh,
                           synchronize=lambda: None)
    model = Model()
    gate.attach(model)
    try:
        if cancel:
            with pytest.raises(guard.TrialAborted, match='cancelled'):
                model([1, 2])
        else:
            assert model([1, 2]) == [1, 2]
            assert calls == [[1, 2]]
            assert gate.monitor_wait_seconds > 0
    finally:
        gate.close()
