import os

from mainframe.memory.atomic import create_exclusive, write_atomic
from mainframe.memory.frontmatter import render, split


def test_frontmatter_roundtrip_and_flattening():
    fm = render({"type": "session", "project": "p", "cwd": "C:\\x\\y", "title": "multi\nline"})
    fields, body = split(fm + "# H\n\nbody\n")
    assert fields["type"] == "session" and fields["cwd"] == "C:/x/y" and fields["title"] == "multi line"
    assert body == "# H\n\nbody\n"


def test_split_without_frontmatter():
    fields, body = split("# H\nbody")
    assert fields == {} and body == "# H\nbody"


def test_write_atomic_leaves_no_tmp(tmp_path):
    p = tmp_path / "n.md"
    write_atomic(p, "one")
    write_atomic(p, "two")
    assert p.read_text(encoding="utf-8") == "two"
    assert not list(tmp_path.glob("*.tmp"))


def test_create_exclusive_never_overwrites(tmp_path):
    a = create_exclusive(tmp_path, "2026-01-01-000000-ab12cd34")
    b = create_exclusive(tmp_path, "2026-01-01-000000-ab12cd34")
    assert a.name == "2026-01-01-000000-ab12cd34.md" and b.name == "2026-01-01-000000-ab12cd34-2.md"
    assert a.exists() and b.exists() and os.path.getsize(a) == 0


def test_create_exclusive_skips_a_same_named_directory(tmp_path):
    (tmp_path / "2026-01-01-000000-ab12cd34.md").mkdir()
    p = create_exclusive(tmp_path, "2026-01-01-000000-ab12cd34")
    assert p.name == "2026-01-01-000000-ab12cd34-2.md" and p.is_file()


def test_write_atomic_cleans_tmp_on_failure(tmp_path, monkeypatch):
    p = tmp_path / "n.md"
    write_atomic(p, "one")

    def boom(fd):
        raise OSError("disk full")
    monkeypatch.setattr(os, "fsync", boom)
    import pytest
    with pytest.raises(OSError):
        write_atomic(p, "two")
    assert p.read_text(encoding="utf-8") == "one"      # target untouched
    assert not list(tmp_path.glob("*.tmp"))            # no leaked temp file


def test_create_exclusive_writes_through_fd(tmp_path):
    p = create_exclusive(tmp_path, "cap", text="# hello\n")
    assert p.read_text(encoding="utf-8") == "# hello\n"
    q = create_exclusive(tmp_path, "cap", text="# second\n")
    assert q.name == "cap-2.md" and q.read_text(encoding="utf-8") == "# second\n"


def test_create_exclusive_cleans_up_on_write_failure(tmp_path, monkeypatch):
    import pytest

    def boom(fd):
        raise OSError("disk full")
    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        create_exclusive(tmp_path, "cap", text="x")
    assert list(tmp_path.iterdir()) == []
