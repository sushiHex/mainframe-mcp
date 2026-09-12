"""Lazy model ownership and bounded recovery.

One lock serializes loading, use, and release. Health readers see immutable
metadata snapshots without acquiring that lock or retaining model objects.
Device faults get one reload and retry; failed loads back off."""

import copy
import gc
import logging
import threading
import time
from dataclasses import dataclass

from mainframe.core.device import empty_cuda_cache, is_device_failure, memory_pressure_guidance

logger = logging.getLogger(__name__)

ROLES = ("embedder", "reranker")


@dataclass(frozen=True)
class _ModelState:
    name: str
    state: str = "absent"
    error: str | None = None
    retry_at: float | None = None
    guidance: str | None = None
    info: dict | None = None

    def retry_in(self, now: float) -> float:
        return max(0.0, self.retry_at - now) if self.retry_at is not None else 0.0

    def as_dict(self, now: float) -> dict:
        return {"name": self.name, "state": self.state, "error": self.error, "guidance": self.guidance,
                "retry_after_s": int(self.retry_in(now)), "info": copy.deepcopy(self.info)}


def _default_embedder(config):
    from mainframe.core.embedder import Embedder
    return Embedder(config)


def _default_reranker(config):
    from mainframe.core.reranker import Reranker
    return Reranker(config)


class Models:
    def __init__(self, config: dict, embedder_factory=None, reranker_factory=None, events=None, clock=time.monotonic):
        self.config = config
        self.events = events
        self.clock = clock
        # Acquire BEFORE looking up a model. Waiting callers hold no model
        # references, so unloading can finish before a replacement is built.
        self._lock = threading.Lock()
        self._factories = {"embedder": embedder_factory or _default_embedder,
                           "reranker": reranker_factory or _default_reranker}
        self._objects: dict = {}
        self._state = {r: _ModelState(config.get(r, {}).get("model", "")) for r in ROLES}
        self.retry_seconds = float(config.get("models", {}).get("retry_minutes", 15)) * 60

    def _set_state(self, role: str, state: str, *, error=None, retry_at=None, guidance=None, info=None):
        # Publish one complete snapshot. Readers never combine fields from
        # different transitions, and the retry deadline is stored only here.
        self._state[role] = _ModelState(self._state[role].name, state, error, retry_at, guidance, info)

    def _emit(self, kind: str, **detail):
        if self.events is not None:
            try:
                self.events.emit(kind, **detail)
            except Exception:
                pass

    def _get(self, role: str):
        """Resolve or load a model. The caller holds `_lock`."""
        obj = self._objects.get(role)
        if obj is not None:
            return obj
        status = self._state[role]
        wait = status.retry_in(self.clock())
        if wait > 0:
            raise RuntimeError(f"{role} unavailable: {status.error}; retry in {int(wait)}s")
        self._set_state(role, "loading")
        try:
            obj = self._factories[role](self.config)
        except Exception as e:
            guidance = memory_pressure_guidance(e)
            self._set_state(role, "failed", error=str(e), retry_at=self.clock() + self.retry_seconds,
                            guidance=guidance)
            self._emit("model.failed", role=role, error=str(e), guidance=guidance)
            logger.warning("%s failed to load: %s", role, str(e))
            if guidance:
                logger.warning("%s", guidance)
            raise
        self._objects[role] = obj
        try:
            model_info = getattr(obj, "info", None)
            info = copy.deepcopy(model_info) if isinstance(model_info, dict) else None
        except Exception:
            info = None
        self._set_state(role, "loaded", info=info)
        self._emit("model.loaded", role=role, name=self._state[role].name)
        return obj

    def invoke(self, role: str, operation):
        """Run `operation(model)`, reloading and retrying once on a device fault.

        One lock owns lookup, use, and release. Operations return results,
        never model references; waiting callers resolve the current model only
        after acquiring the lock. Load failures keep `_get`'s own backoff."""
        with self._lock:
            obj = self._get(role)
            try:
                return operation(obj)
            except Exception as e:
                if not is_device_failure(e):
                    raise
                # Buffered log records must not keep the exception's model
                # references alive through its traceback.
                logger.warning("%s: device failure (%s); reloading and retrying once", role, str(e))
                self._emit("model.device_failure", role=role, error=str(e))
            # Leave the handler (including chained tracebacks) and drop our
            # local BEFORE collecting cycles and clearing the allocator cache.
            del obj
            self._invalidate((role,))
            return operation(self._get(role))

    def _invalidate(self, roles):
        """Release all selected roles under `_lock`, then reclaim memory once."""
        if not roles:
            return
        for role in roles:
            self._objects.pop(role, None)
            self._set_state(role, "absent")
        gc.collect()
        empty_cuda_cache()
        for role in roles:
            self._emit("model.invalidated", role=role)

    def invalidate(self, role: str):
        """Wait for the current operation, then release this role's model."""
        with self._lock:
            self._invalidate((role,))

    def invalidate_all(self) -> list:
        """Release resident models and clear failed-load backoff in one turn."""
        with self._lock:
            roles = [r for r in ROLES if self._state[r].state in ("loaded", "failed")]
            self._invalidate(roles)
            return roles

    def loaded(self, role: str) -> bool:
        return role in self._objects

    def state(self) -> dict:
        now = self.clock()
        return {r: status.as_dict(now) for r, status in self._state.items()}

    def prewarm(self, background: bool = True):
        def _run():
            for role in ROLES:
                try:
                    with self._lock:
                        self._get(role)
                except Exception:
                    pass
        if background:
            threading.Thread(target=_run, name="prewarm", daemon=True).start()
        else:
            _run()
