"""A cooperative claim can expire without invalidating completed model work."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest


def load_runner(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / 'eval'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('trial_runner_lease_test', directory / 'run_trial.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expired_reservation_is_replaced_and_the_new_claim_is_released(monkeypatch, tmp_path):
    runner = load_runner(monkeypatch)
    lease = runner.VramLease(15., tmp_path)
    lease.claim_id = 'expired'
    calls = []

    async def call(name, arguments):
        calls.append((name, arguments))
        return {'renew': {'ok': False}, 'reserve': {'ok': True, 'claim_id': 'replacement'},
                'release': {'ok': True}}[name]

    lease.call = call

    async def exercise():
        await lease.renew()
        await lease.__aexit__(None, None, None)

    asyncio.run(exercise())
    assert [name for name, args in calls] == ['renew', 'reserve', 'release']
    assert calls[1][1]['gb'] == 15.
    assert calls[2][1]['claim_id'] == 'replacement'
    events = [json.loads(line) for line in (tmp_path / 'lease-events.jsonl').read_text().splitlines()]
    assert events[-1]['replaces'] == 'expired'


@pytest.mark.parametrize('response', [{'ok': False}, {'ok': True}, {}])
def test_replacement_must_be_confirmed_before_continuing(monkeypatch, tmp_path, response):
    runner = load_runner(monkeypatch)
    lease = runner.VramLease(15., tmp_path)
    lease.claim_id = 'expired'

    async def call(name, arguments):
        return {'ok': False} if name == 'renew' else response

    lease.call = call
    with pytest.raises(RuntimeError, match='reservation'):
        asyncio.run(lease.renew())


def test_releasing_an_already_absent_claim_is_successful_cleanup(monkeypatch, tmp_path):
    runner = load_runner(monkeypatch)
    lease = runner.VramLease(15., tmp_path)
    lease.claim_id = 'expired'

    async def call(name, arguments):
        assert name == 'release'
        return {'ok': False}

    lease.call = call
    asyncio.run(lease.__aexit__(None, None, None))
