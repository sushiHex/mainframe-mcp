"""Durable file primitives: tmp → fsync → os.replace for rewrites;
exclusive-create with a collision counter for immutable captures."""

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
    """Create `<base><suffix>` (or `<base>-N<suffix>`) with O_EXCL and write `text`
    through the same descriptor — a crash can never leave a zero-byte file that
    a later writer would skip as "already captured". Race-safe between two
    writers; never overwrites. Returns the created path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        name = f"{base}{suffix}" if n == 1 else f"{base}-{n}{suffix}"
        path = directory / name
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            n += 1
            continue
        except PermissionError:
            if path.is_dir():  # Windows reports an existing directory as EACCES, not EEXIST
                n += 1
                continue
            raise
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return path
