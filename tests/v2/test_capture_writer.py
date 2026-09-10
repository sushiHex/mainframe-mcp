from datetime import datetime

import pytest

from mainframe.memory.capture import KINDS, write_capture
from mainframe.memory.frontmatter import split
from mainframe.memory.lanes import ADHOC_PROJECT, Lanes, capture_id_of


def test_write_capture_layout_and_identity(cfg):
    L = Lanes(cfg)
    now = datetime(2026, 9, 2, 13, 4, 5)
    p = write_capture(L, "## Summary\nDecided X. api_key=example-body\n", "Big decision",
                      project="proj", session_id="abcdef1234", now=now, cwd="C:\\x\\proj", branch="main",
                      reason="api_key=example-reason")
    assert p.parent == L.capture_dir("proj") and p.name.startswith("2026-09-02-130405-abcdef12-")
    fields, body = split(p.read_text(encoding="utf-8"))
    assert fields["type"] == "session" and fields["project"] == "proj" and fields["session_id"] == "abcdef12"
    assert fields["capture_id"] == capture_id_of(p) and p.name.endswith(f"-{fields['capture_id'][:8]}.md")
    assert body.startswith("# Session 2026-09-02 — Big decision")
    assert "example-body" not in body and "example-reason" not in fields["reason"]
    assert fields["cwd"] == "C:/x/proj" and fields["git_branch"] == "main"


def test_write_capture_never_overwrites_and_adhoc(cfg):
    L = Lanes(cfg)
    now = datetime(2026, 9, 2, 0, 0, 0)
    a = write_capture(L, "same body", "t", project=None, now=now)
    b = write_capture(L, "same body", "t", project=None, now=now)
    assert a != b and a.parent == L.capture_dir(ADHOC_PROJECT) and a.read_text(encoding="utf-8")
    assert split(a.read_text(encoding="utf-8"))[0]["session_id"] == "adhoc"


def test_title_is_one_line_and_captured_by_is_scrubbed(cfg):
    L = Lanes(cfg)
    now = datetime(2026, 9, 2)
    p = write_capture(L, "body", "Big\n  decision\tnow", project="p",
                      captured_by="hook api_key=example-provenance", now=now)
    _, body = split(p.read_text(encoding="utf-8"))
    assert body.splitlines()[0] == "# Session 2026-09-02 — Big decision now"
    assert "example-provenance" not in p.read_text(encoding="utf-8")


def test_write_capture_rejects_bad_project_and_kind(cfg):
    with pytest.raises(ValueError):
        write_capture(Lanes(cfg), "x", "t", project="../evil")
    with pytest.raises(ValueError):
        write_capture(Lanes(cfg), "x", "t", project="p", kind="poem")
    assert KINDS == ("session", "decision", "note")
