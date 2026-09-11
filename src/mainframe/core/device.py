"""What we know about the GPU without importing torch.

Every place that reaches into an already-loaded torch lives here, so the two
modules that need it — the model registry, which decides whether to heal, and
the indexer, which decides whether a failure belongs to one document — can each
reach it without reaching each other. `core/indexer.py` deliberately knows
nothing about roles, locks or backoff (it takes an embedder by parameter), and
a predicate about the device is no reason to change that.

`torch` is always read out of `sys.modules`, never imported: only a torch that
is already loaded can have raised any of this, and a daemon that has never
loaded a model must not initialize CUDA to ask a question about it.
"""

import logging
import re
import sys

logger = logging.getLogger(__name__)

# Substrings of the messages torch reports for a GPU or CUDA-context fault.
# Every one of them NAMES the device stack. Bare phrases like "unknown error",
# "initialization error" or "invalid device" were here and are gone: they match
# ordinary application errors, and the cost of a false positive is dropping a
# healthy 8B model and reloading it for two minutes.
_DEVICE_MESSAGES = ("cuda error", "cuda driver", "cuda runtime", "cuda failure", "cuda-capable device",
                    "cudnn error", "cublas", "device-side assert", "illegal memory access",
                    "no kernel image is available")


def _causes(exc: BaseException):
    """`exc` and the chain it was explicitly raised `from`. Only `__cause__`:
    `__context__` picks up whatever happened to be in flight, which is not a
    claim about this failure."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__


def memory_pressure_guidance(exc: BaseException) -> str | None:
    """Recognize Windows commit exhaustion, including native-library wrappers."""
    for cause in _causes(exc):
        if (getattr(cause, "winerror", None) == 1455
                or re.search(r"\b(?:winerror|os error)\s*1455\b", str(cause), re.IGNORECASE)):
            return ("Windows system commit capacity is exhausted (error 1455). "
                    "Check Memory > Committed in Task Manager; close memory-heavy processes "
                    "or increase page file capacity before retrying. Free VRAM does not measure this limit.")
    return None


def is_device_failure(exc: BaseException) -> bool:
    """Did the GPU (or its context) go bad, as opposed to this operation being
    wrong?

    THE question that separates "reload and retry" from "report and move on",
    and the one that decides whether a failure belongs to one document or to
    the whole process.

    The match is fuzzy ON PURPOSE. torch has a dedicated type for a few of
    these (`torch.AcceleratorError` on builds that define it) but reports most
    driver- and context-level faults as a plain `RuntimeError` whose only
    distinguishing feature is its message — "CUDA error: unknown error" after a
    suspend, "CUDA driver ..." after a driver reset. So the type is checked
    where a type exists and the message where it does not.

    The whole cause chain is examined, because the layer that reports the fault
    is rarely the layer that raised it — a wrapper saying "embedding failed"
    `from` the real CUDA error is the ordinary shape.

    Out-of-memory is deliberately EXCLUDED. It is a capacity failure, not a
    dead context: reloading the same model cannot make room, and both the
    embedder and the reranker already degrade to one item at a time on it."""
    texts = [str(e).lower() for e in _causes(exc)]
    if any("out of memory" in t for t in texts) or memory_pressure_guidance(exc):
        return False
    torch = sys.modules.get("torch")
    device_error = getattr(torch, "AcceleratorError", None) if torch is not None else None
    if device_error is not None and any(isinstance(e, device_error) for e in _causes(exc)):
        return True
    return any(m in t for t in texts for m in _DEVICE_MESSAGES)


def empty_cuda_cache():
    """Best-effort. Dropping our references leaves the allocator holding the
    blocks, so the reload finds less free VRAM than it should."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.debug("empty_cache failed (non-fatal): %s", e)
