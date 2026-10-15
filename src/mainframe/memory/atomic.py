"""Durable file primitives: tmp → fsync → os.replace for rewrites;
tmp → fsync → os.link (never overwrites) with a collision counter for
immutable captures."""

import os
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def create_exclusive(directory: Path, base: str, suffix: str = ".md", text: str = "") -> Path:
    """Publish `text` as `<base><suffix>` (or `<base>-N<suffix>`): written and
    fsynced under a temporary name, then hard-linked into place. A link never
    overwrites, so this is race-safe between two writers, and the published
    name only ever holds the complete text: a process killed mid-write (a hook
    past its timeout) leaves at most a stray `.tmp`, never a short capture that
    a later writer would skip as "already captured". Returns the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".{base}{suffix}.{os.getpid()}.tmp"
    try:
        with open(tmp, "x", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        n = 1
        while True:
            path = directory / (f"{base}{suffix}" if n == 1 else f"{base}-{n}{suffix}")
            try:
                os.link(tmp, path)
                return path
            except FileExistsError:
                n += 1
            except PermissionError:
                if not path.is_dir():  # Windows reports an existing directory as EACCES, not EEXIST
                    raise
                n += 1
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
