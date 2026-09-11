import sys

import pytest

from mainframe.core import device


@pytest.mark.parametrize("message", ["[WinError 1455] synthetic failure", "allocation (os error 1455)"])
def test_host_capacity_is_not_a_cuda_context_fault(message):
    cause = OSError(message)
    wrapped = RuntimeError("CUDA error: allocation failed")
    wrapped.__cause__ = cause
    assert device.memory_pressure_guidance(wrapped)
    assert not device.is_device_failure(wrapped)


def test_native_error_code_works_without_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    error = OSError("localized system message")
    error.winerror = 1455
    assert device.memory_pressure_guidance(error)
    assert not device.is_device_failure(error)
    assert "torch" not in sys.modules


@pytest.mark.parametrize("message", ["model 1455 unavailable", "os error 14550", "CUDA error: unknown error"])
def test_unrelated_errors_do_not_receive_windows_capacity_advice(message):
    assert device.memory_pressure_guidance(RuntimeError(message)) is None
