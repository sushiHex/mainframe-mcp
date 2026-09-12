"""Probe contracts and actual Windows child cleanup, without models or CUDA."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

EVAL = Path(__file__).resolve().parents[2] / 'eval'
sys.path.insert(0, str(EVAL))


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location('embedding_probe', EVAL / 'embedding_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrong_snapshot_and_changed_custom_code_refused_before_loading(probe, tmp_path):
    with pytest.raises(ValueError, match='pinned'):
        probe.validate_snapshot('nemotron', tmp_path)
    snapshot = tmp_path / probe.PROFILES['voyage']['revision']
    snapshot.mkdir()
    (snapshot / 'modeling_qwen3_bidirectional.py').write_text('raise RuntimeError("changed")')
    with pytest.raises(ValueError, match='reviewed'):
        probe.validate_snapshot('voyage', snapshot)


@pytest.mark.parametrize('vectors', [[[float('nan'), 0.]], [[0., 0.]], [[1.]]])
def test_bad_vectors_fail_contract(probe, vectors):
    with pytest.raises(ValueError):
        probe.validate_vectors(vectors, count=1, dimension=2)


def test_vector_tolerance_is_fixed_and_recorded(probe):
    assert probe.validate_vectors([[.6, .8]], count=1, dimension=2)['max_norm_error'] < .01
    assert len(probe.CASES) == 8
    assert any(text == '' for _, text in probe.CASES)
    assert len(dict(probe.CASES)['long'].split()) >= 2048


def test_snapshot_metadata_requires_native_prompt_contract(probe, tmp_path):
    profile = probe.PROFILES['harrier']
    snapshot = tmp_path / profile['revision']
    snapshot.mkdir()
    path = snapshot / 'config_sentence_transformers.json'
    path.write_text(json.dumps({'prompts': {'query': 'wrong'}}))
    with pytest.raises(ValueError, match='web_search_query'):
        probe.validate_snapshot('harrier', snapshot)


def test_importing_supervisor_does_not_import_models():
    code = "import run_trial,sys; assert not any(n in sys.modules for n in ('torch','sentence_transformers','transformers'))"
    subprocess.run([sys.executable, '-c', code], cwd=EVAL, check=True, timeout=15)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows Job Object integration')
def test_owned_job_stops_descendants_after_launcher_exits(tmp_path):
    from trial_worker import OwnedWorker
    psutil = pytest.importorskip('psutil')

    child_pid = tmp_path / 'child.pid'
    script = tmp_path / 'child.py'
    script.write_text(
        "import subprocess,sys\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "Path(sys.argv[1]).write_text(str(child.pid))\n")
    worker = OwnedWorker(sys.executable, script, [str(child_pid)], tmp_path)
    try:
        import ctypes
        # Native JOBOBJECT_BASIC_LIMIT_INFORMATION starts with two int64 times,
        # then LimitFlags. Check the real job has ownership but no memory cap.
        info = ctypes.create_string_buffer(144 if ctypes.sizeof(ctypes.c_void_p) == 8 else 112)
        assert worker.api.QueryInformationJobObject(worker.job, 9, info, len(info), None)
        flags = int.from_bytes(info.raw[16:20], 'little')
        assert flags & 0x2000 and not flags & (0x100 | 0x200)
        deadline = time.monotonic() + 10
        while (not child_pid.exists() or worker.poll() is None) and time.monotonic() < deadline:
            time.sleep(.05)
        assert child_pid.exists() and worker.poll() == 0
        child = psutil.Process(int(child_pid.read_text()))
        assert child.is_running()
    finally:
        worker.stop()
    child.wait(timeout=5)
    assert not child.is_running()
    worker.stop()  # Idempotent cleanup.


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows pipe integration')
def test_owned_worker_receives_permit_without_reading_a_file(tmp_path):
    from trial_worker import OwnedWorker

    script = tmp_path / 'receive.py'
    result = tmp_path / 'received.json'
    script.write_text(
        "import sys,json\nfrom pathlib import Path\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from trial_guard import open_gate\n"
        "gate,settings=open_gate(Path('missing-permit'), 'test', stdin=True)\n"
        "gate.wait()\nPath(sys.argv[2]).write_text(json.dumps({'received':True}))\n")
    worker = OwnedWorker(sys.executable, script, [str(EVAL), str(result)], tmp_path)
    try:
        worker.publish({'nonce': 'test', 'allowed': True, 'at': time.monotonic(),
                        'settings': {'stale_s': 10}})
        deadline = time.monotonic() + 10
        while worker.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert worker.poll() == 0
        assert json.loads(result.read_text()) == {'received': True}
    finally:
        worker.stop()


def test_closing_an_exited_workers_broken_pipe_does_not_fail_cleanup():
    import threading
    from types import SimpleNamespace
    from trial_worker import OwnedWorker

    class ClosedReader:
        closed = False

        def close(self):
            raise OSError(22, 'Invalid argument')

    worker = OwnedWorker.__new__(OwnedWorker)
    worker._condition = threading.Condition()
    worker.process = SimpleNamespace(stdin=ClosedReader(), wait=lambda timeout: 0)
    worker.job = worker.log = None
    worker.stop()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows supervisor integration')
def test_supervisor_completes_real_pipe_worker_and_releases_lease(tmp_path):
    import asyncio
    from trial_guard import Sample, TrialSettings, supervise
    from trial_worker import OwnedWorker

    class Lease:
        released = False

        async def __aenter__(self):
            return self

        async def renew(self):
            pass

        async def __aexit__(self, *args):
            self.released = True

    script = tmp_path / 'complete.py'
    script.write_text(
        "import sys,json\nfrom pathlib import Path\n"
        "sys.path.insert(0, sys.argv[1])\nfrom trial_guard import open_gate\n"
        "gate,settings=open_gate(Path('unused'), sys.argv[3], stdin=True)\n"
        "gate.wait()\noutput=Path(sys.argv[2]); output.mkdir()\n"
        "(output/'report.json').write_text(json.dumps({'complete':True,'guard_nonce':sys.argv[3]}))\n")
    lease = Lease()

    def launch(permit, nonce):
        return OwnedWorker(sys.executable, script, [EVAL, tmp_path / 'worker', nonce], tmp_path)

    result = asyncio.run(supervise(TrialSettings(), tmp_path,
                                   lambda: Sample(time.monotonic(), 1, 8, 20, 24), launch, lease))
    assert result['status'] == 'completed', result
    assert lease.released


def test_lease_rejects_api_errors_and_releases_only_its_claim(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from run_trial import VramLease

    calls = []

    class Session:
        async def call_tool(self, name, args):
            calls.append((name, args))
            return SimpleNamespace(isError=name != 'release', structuredContent={'ok': name == 'release'})

    lease = VramLease(7., tmp_path)
    lease.session = Session()
    lease.claim_id = 'synthetic-owned-claim'
    with pytest.raises(RuntimeError, match='vram-mcp renew failed'):
        asyncio.run(lease.renew())
    asyncio.run(lease.__aexit__(None, None, None))
    assert calls == [('renew', {'claim_id': 'synthetic-owned-claim', 'ttl_seconds': 60}),
                     ('release', {'claim_id': 'synthetic-owned-claim'})]


def test_output_refuses_git_and_reuse_before_writing(tmp_path):
    from run_trial import prepare_output

    with pytest.raises(ValueError, match='outside Git'):
        prepare_output(EVAL / 'must-not-exist')
    output = prepare_output(tmp_path / 'fresh')
    with pytest.raises(FileExistsError):
        prepare_output(output)


@pytest.mark.parametrize('fail', [False, True])
def test_native_factory_gates_inner_transformer_and_never_retries_oom(monkeypatch, tmp_path, fail):
    from collections import UserDict
    import sentence_transformers
    import torch
    from compare_models import experimental_factories
    from trial_guard import BatchGate, TrialAborted, write_json

    calls = []
    permit = tmp_path / 'permit.json'
    write_json(permit, {'nonce': 'test', 'at': 100., 'allowed': True})
    gate = BatchGate(permit, 'test', clock=lambda: 100., synchronize=lambda: None)

    class Transformer(torch.nn.Module):
        def forward(self, input_ids):
            calls.append(input_ids.tolist())
            if fail:
                raise torch.cuda.OutOfMemoryError('synthetic OOM')
            return torch.tensor([[1., 0.] for _ in input_ids])

    class Encoder(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.transformer = TransformerModule()

        def get_sentence_embedding_dimension(self):
            return 2

        def forward(self, inputs):
            return self.transformer(UserDict({'input_ids': inputs}))

        def encode_document(self, texts, **kwargs):
            # ST deliberately bypasses its own __call__; the inner hook matters.
            return self.forward(torch.tensor([[len(text)] for text in texts]))

    class TransformerModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.auto_model = Transformer()

        def forward(self, features):
            # ST 5.4 calls the HF forward method directly, also bypassing its
            # hooks. Only the enclosing ST transformer module is called normally.
            return self.auto_model.forward(**features)

    monkeypatch.setattr(sentence_transformers, 'SentenceTransformer', Encoder)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    factory, _ = experimental_factories(True, False, gate=gate)
    model = factory({'embedder': {'model': 'synthetic', 'quantize': False},
                     'paths': {'model_cache': str(tmp_path)}})
    try:
        if fail:
            with pytest.raises(TrialAborted, match='synthetic OOM'):
                model.embed(['abc', 'd'])
            assert gate.failed
        else:
            assert model.embed(['abc', 'd']) == [[1., 0.], [1., 0.]]
            assert gate.forwards == 1
            assert gate.input_shapes == [[2, 1]]
        assert calls == [[[3], [1]]]
    finally:
        gate.close()
