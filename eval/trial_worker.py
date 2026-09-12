"""Windows worker ownership; no model imports and no process-name matching."""

import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


class OwnedWorker:
    """Keep the interpreter behind a pipe barrier until its Job Object owns it.

    Python's Windows venv launcher can spawn an interpreter before assignment.
    The bootstrap identifies that interpreter and waits for the parent. EOF
    exits without importing site/model packages; closing the non-inherited job
    handle kills assigned processes and their descendants, even after a crash.
    """

    def __init__(self, python, script, args, output):
        if sys.platform != 'win32':
            raise RuntimeError('owned live probes currently require Windows')
        self.process = self.job = self.log = None
        self._condition = threading.Condition()
        self._pending = None
        self._stopping = False
        self._pipe_error = None
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            'SetInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            'AssignProcessToJobObject': ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            'IsProcessInJob': ([wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)], wintypes.BOOL),
            'OpenProcess': ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            'TerminateJobObject': ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            'QueryInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                           wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = argtypes, restype

        class BasicLimits(ctypes.Structure):
            _fields_ = [('ProcessTime', ctypes.c_int64), ('JobTime', ctypes.c_int64),
                        ('Flags', wintypes.DWORD), ('MinWorkingSet', ctypes.c_size_t),
                        ('MaxWorkingSet', ctypes.c_size_t), ('ActiveProcesses', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t), ('Priority', wintypes.DWORD),
                        ('SchedulingClass', wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [('Basic', BasicLimits), ('Io', ctypes.c_uint64 * 6),
                        ('ProcessMemory', ctypes.c_size_t), ('JobMemory', ctypes.c_size_t),
                        ('PeakProcessMemory', ctypes.c_size_t), ('PeakJobMemory', ctypes.c_size_t)]

        self.job = self.api.CreateJobObjectW(None, None)
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = ExtendedLimits()
            # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. Ownership only; no memory cap.
            limits.Basic.Flags = 0x2000
            if not self.api.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            ready = Path(output) / 'worker.pid'
            bootstrap = (
                "import os,sys\nfrom pathlib import Path\n"
                "ready=Path(sys.argv.pop(1)); temporary=ready.with_suffix('.tmp')\n"
                "temporary.write_text(str(os.getpid())); temporary.replace(ready)\n"
                "if os.read(sys.stdin.fileno(), 1) != b'1': raise SystemExit(2)\n"
                "import site; site.main()\nimport runpy\n"
                "target=sys.argv.pop(1); sys.argv[0]=target\n"
                "runpy.run_path(target, run_name='__main__')\n")
            self.log = (Path(output) / 'worker.log').open('wb')
            self.process = subprocess.Popen(
                [str(python), '-S', '-u', '-c', bootstrap, str(ready), str(script), *map(str, args)],
                stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
                bufsize=0,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            self._assign(int(self.process._handle))
            deadline = time.monotonic() + 5
            while not ready.exists():
                if self.process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('worker bootstrap did not become ready')
                time.sleep(.01)
            # Assign the actual interpreter as well, if it predates the launcher assignment.
            child = self.api.OpenProcess(0x0100 | 0x0001 | 0x0400, False, int(ready.read_text()))
            if not child:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                self._assign(child)
            finally:
                self.api.CloseHandle(child)
            self.process.stdin.write(b'1')
            self.process.stdin.flush()
            threading.Thread(target=self._publish_loop, daemon=True, name='trial-permit-writer').start()
        except BaseException:
            self.stop()
            raise

    def _assign(self, handle):
        member = wintypes.BOOL()
        if not self.api.IsProcessInJob(handle, self.job, ctypes.byref(member)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not member.value and not self.api.AssignProcessToJobObject(self.job, handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def poll(self):
        return self.process.poll()

    def publish(self, permit):
        """Keep only the newest pending heartbeat; never block the monitor."""
        if self.poll() is not None:
            return
        with self._condition:
            if self._pipe_error is not None and self.poll() is None:
                raise RuntimeError(f'worker permit pipe failed: {self._pipe_error}')
            self._pending = json.dumps(permit).encode('utf-8') + b'\n'
            self._condition.notify()

    def _publish_loop(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopping or self._pending is not None)
                    if self._stopping:
                        return
                    data, self._pending = self._pending, None
                self.process.stdin.write(data)
                self.process.stdin.flush()
        except (OSError, ValueError) as error:
            self._pipe_error = str(error)

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify()
        if self.job:
            try:
                if not self.api.TerminateJobObject(self.job, 2):
                    raise ctypes.WinError(ctypes.get_last_error())

                class Accounting(ctypes.Structure):
                    _fields_ = [('Times', ctypes.c_int64 * 4), ('Faults', wintypes.DWORD),
                                ('Total', wintypes.DWORD), ('Active', wintypes.DWORD),
                                ('Terminated', wintypes.DWORD)]

                deadline = time.monotonic() + 5
                while True:
                    info = Accounting()
                    if not self.api.QueryInformationJobObject(self.job, 1, ctypes.byref(info), ctypes.sizeof(info), None):
                        raise ctypes.WinError(ctypes.get_last_error())
                    if not info.Active:
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError('owned processes did not exit within five seconds')
                    time.sleep(.01)
            finally:
                self.api.CloseHandle(self.job)
                self.job = None
        # Terminating readers first also releases a writer blocked in the pipe.
        if self.process is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except OSError:
                # Windows can report EINVAL as well as EPIPE after the reader
                # exits. The owned job is already stopped; no data remains owed.
                pass
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()
