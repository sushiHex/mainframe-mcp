"""Resource observation and cooperative pacing for isolated evaluation workers.

This module imports no model library. Hardware readers and worker construction
are injected into the supervisor so failures can be tested without CUDA.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid


class TrialAborted(BaseException):
    """Stop evaluation without triggering production model recovery/fallback."""


class TelemetryStale(TrialAborted):
    """The owner is late; submit no work until it publishes a fresh permit."""


class PipePermit:
    """Latest complete heartbeat from the owned parent pipe; bounded memory."""

    def __init__(self, stream):
        self._latest = None
        self._error = None
        self._lock = threading.Lock()
        # A separate raw descriptor avoids holding Python's buffered-stdin
        # lock while interpreter shutdown tries to close standard streams.
        descriptor = os.dup(stream.fileno())
        threading.Thread(target=self._read, args=(descriptor,), daemon=True,
                         name='trial-permit-reader').start()

    def _read(self, descriptor):
        try:
            pending = b''
            while True:
                chunk = os.read(descriptor, 4096)
                if not chunk:
                    raise EOFError('supervisor pipe closed')
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        raise ValueError('permit must be an object')
                    with self._lock:
                        self._latest = data
                if len(pending) > 65536:
                    raise ValueError('permit frame is malformed')
        except (OSError, ValueError, EOFError) as error:
            with self._lock:
                self._error = str(error)
        finally:
            os.close(descriptor)

    def __call__(self):
        with self._lock:
            if self._error is not None:
                raise TrialAborted(self._error)
            if self._latest is None:
                raise TelemetryStale('waiting for supervisor')
            return self._latest


def open_gate(path, nonce, *, stdin=False):
    """Read settings and permits through one transport, before model imports."""
    if stdin:
        import sys
        source = PipePermit(sys.stdin.buffer)
    else:
        source = lambda: json.loads(retry_sharing(lambda: path.read_text(encoding='utf-8')))
    while True:
        try:
            permit = source()
            break
        except TelemetryStale:
            time.sleep(.05)
    settings = TrialSettings(**permit['settings'])
    gate = BatchGate(source, nonce, idle_ratio=settings.idle_ratio,
                     min_idle_s=settings.min_idle_s, stale_s=settings.stale_s)
    gate.wait()
    return gate, settings


def retry_sharing(operation):
    """Windows replacement can briefly conflict with a reader or scanner handle."""
    deadline = time.monotonic() + .25
    while True:
        try:
            return operation()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.01)


def write_json(path: Path, data: dict):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    retry_sharing(lambda: temporary.replace(path))


@dataclass(frozen=True)
class Sample:
    at: float
    commit_used_gib: float
    commit_limit_gib: float
    gpu_free_gib: float
    gpu_total_gib: float

    def validate(self):
        if (any(not math.isfinite(v) or v < 0 for v in asdict(self).values())
                or not 0 <= self.commit_used_gib <= self.commit_limit_gib
                or not 0 <= self.gpu_free_gib <= self.gpu_total_gib
                or self.commit_limit_gib == 0 or self.gpu_total_gib == 0):
            raise ValueError('invalid resource telemetry')


@dataclass(frozen=True)
class TrialSettings:
    idle_ratio: float = 3.
    min_idle_s: float = .25
    sample_s: float = .25
    stale_s: float = 1.

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('monitoring and pacing settings must be finite and positive')
        if self.sample_s >= self.stale_s:
            raise ValueError('sampling interval must be shorter than telemetry freshness')


class BatchGate:
    """Pace actual model forwards while preserving tokenizer/batch decisions."""

    def __init__(self, permit: Path, nonce: str, *, idle_ratio=3., min_idle_s=.25,
                 stale_s=1., clock=time.monotonic, sleep=time.sleep, synchronize=None):
        self.permit, self.nonce = permit, nonce
        self.idle_ratio, self.min_idle_s, self.stale_s = idle_ratio, min_idle_s, stale_s
        self.clock, self.sleep, self.synchronize = clock, sleep, synchronize
        self.idle_seconds = 0.
        self.monitor_wait_seconds = 0.
        self.forward_seconds = 0.
        self.input_shapes = []
        self.forwards = 0
        self.failed = False
        self._ready_at = 0.
        self._started_at = 0.
        self._handles = []

    def check(self):
        if self.failed:
            raise TrialAborted('a forward failed; evaluation recovery is prohibited')
        try:
            permit = (self.permit() if callable(self.permit) else
                      json.loads(retry_sharing(lambda: self.permit.read_text(encoding='utf-8'))))
            if permit['nonce'] != self.nonce:
                raise TrialAborted('guard identity changed')
            age = self.clock() - permit['at']
            if permit['allowed'] is not True:
                raise TrialAborted(permit.get('reason') or 'guard cancelled')
            if not math.isfinite(age) or age < 0:
                raise TrialAborted('guard telemetry time is invalid')
            if age > self.stale_s:
                raise TelemetryStale('guard telemetry is stale')
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise TrialAborted(f'guard permit unavailable: {error}') from error

    def wait(self):
        """Pause for a late owner; cancellation and broken permits still fail.

        Worker lifetime belongs to the supervisor's Job Object. If that owner
        dies or its telemetry reader fails, it terminates the waiting worker.
        Scheduling jitter alone must not discard completed benchmark queries.
        """
        while True:
            try:
                self.check()
                return
            except TelemetryStale:
                start = self.clock()
                try:
                    self.sleep(.05)
                finally:
                    self.monitor_wait_seconds += self.clock() - start

    def configure_cuda(self, torch):
        self.wait()
        if self.synchronize is None:
            self.synchronize = torch.cuda.synchronize

    def _before(self, module, args, kwargs):
        self.wait()
        start = self.clock()
        monitor_before = self.monitor_wait_seconds
        while True:
            remaining = self._ready_at - self.clock()
            if remaining <= 0:
                break
            self.sleep(min(.05, remaining))
            self.wait()
        self.idle_seconds += self.clock() - start - (self.monitor_wait_seconds - monitor_before)
        self.synchronize()
        self.wait()
        inputs = kwargs.get('input_ids')
        if inputs is None and args and isinstance(args[0], Mapping):
            inputs = args[0].get('input_ids')
        if inputs is not None:
            self.input_shapes.append(list(inputs.shape))
        self._started_at = self.clock()

    def _after(self, module, args, output):
        if output is None:
            # always_call also observes failed forwards. Do not mask the
            # original exception; prohibit any recovery forward or model load.
            self.failed = True
            return
        self.synchronize()
        elapsed = self.clock() - self._started_at
        self.forward_seconds += elapsed
        self.forwards += 1
        self._ready_at = self.clock() + max(self.min_idle_s, self.idle_ratio * elapsed)
        self.wait()

    def attach(self, model):
        self._handles.append(model.register_forward_pre_hook(self._before, with_kwargs=True))
        self._handles.append(model.register_forward_hook(self._after, always_call=True))

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


async def supervise(settings, output, sample, launch, lease, *, clock=time.monotonic):
    """Own monitoring, permit freshness, one worker, and its capacity lease.

    `completed` means resource monitoring and the worker both completed. Desktop
    usability and retrieval quality have their own gates and are never inferred.
    """
    output = Path(output)
    permit = output / 'permit.json'
    nonce = uuid.uuid4().hex
    worker = None
    result = {'status': 'refused', 'reason': None, 'nonce': nonce,
              'settings': asdict(settings), 'memory_policy': 'observe',
              'samples': 0, 'responsiveness': 'unvalidated',
              'worker_started': False}

    async def read():
        value = await asyncio.wait_for(asyncio.to_thread(sample), timeout=settings.stale_s)
        value.validate()
        age = clock() - value.at
        if not math.isfinite(age) or not 0 <= age <= settings.stale_s:
            raise TrialAborted('resource telemetry is stale')
        result['samples'] += 1
        result['last_sample'] = asdict(value)
        with (output / 'resources.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(asdict(value)) + '\n')
        return value

    def publish(allowed, reason=None):
        data = {'nonce': nonce, 'at': clock(), 'allowed': allowed,
                'reason': reason, 'settings': asdict(settings)}
        if worker is not None and hasattr(worker, 'publish'):
            worker.publish(data)
        else:
            write_json(permit, data)

    try:
        await read()
        async with lease:
            try:
                await read()
                publish(True)
                worker = launch(permit, nonce)
                result['worker_started'] = True
                result['status'] = 'running'
                renewed = clock()
                while True:
                    await read()  # Covers loading, indexing, queries, and exit.
                    if (output / 'cancel').exists():
                        raise TrialAborted('trial cancelled by operator')
                    if clock() - renewed >= 20:
                        await asyncio.wait_for(lease.renew(), timeout=settings.stale_s)
                        renewed = clock()
                    publish(True)
                    code = worker.poll()
                    if code is not None:
                        if code != 0:
                            raise TrialAborted(f'worker exited with status {code}')
                        report = json.loads((output / 'worker' / 'report.json').read_text(encoding='utf-8'))
                        if report.get('complete') is not True or report.get('guard_nonce') != nonce:
                            raise TrialAborted('worker report is incomplete or belongs to another run')
                        result['status'] = 'completed'
                        break
                    await asyncio.sleep(settings.sample_s)
            finally:
                # Revoke and stop BEFORE releasing capacity, including on cancellation.
                try:
                    publish(False, 'supervisor finished')
                finally:
                    if worker is not None:
                        worker.stop()
    except BaseException as error:
        result['status'] = 'aborted' if worker is not None else 'refused'
        result['reason'] = f'{type(error).__name__}: {error}'
    finally:
        write_json(output / 'guard-report.json', result)
    return result


class WindowsResources:
    """Read system commit and NVML memory without importing or initializing torch."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        import sys
        if sys.platform != 'win32':
            raise RuntimeError('live resource sampling currently requires Windows')
        import pynvml

        class PerformanceInfo(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
                'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
                'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')] + [
                    (n, wintypes.DWORD) for n in ('HandleCount', 'ProcessCount', 'ThreadCount')]

        self.ctypes, self.structure, self.nvml = ctypes, PerformanceInfo, pynvml
        self.get_performance = ctypes.WinDLL('psapi', use_last_error=True).GetPerformanceInfo
        self.get_performance.argtypes = [ctypes.POINTER(PerformanceInfo), wintypes.DWORD]
        self.get_performance.restype = wintypes.BOOL
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() != 1:
                raise RuntimeError('live probes currently require exactly one physical GPU')
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except BaseException:
            pynvml.nvmlShutdown()
            raise

    def __call__(self):
        info = self.structure()
        info.cb = self.ctypes.sizeof(info)
        if not self.get_performance(self.ctypes.byref(info), info.cb):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        gpu = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        return Sample(time.monotonic(), info.CommitTotal * info.PageSize / 2**30,
                      info.CommitLimit * info.PageSize / 2**30, gpu.free / 2**30, gpu.total / 2**30)

    def close(self):
        self.nvml.nvmlShutdown()
