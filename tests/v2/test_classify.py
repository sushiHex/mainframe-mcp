import pytest

from mainframe.core.classify import (CAPTURE_LANE, KIND_HEAD, KNOWLEDGE_LANE, PUBLISHED_KINDS,
                                     ROOT_CONTEXT_FILES, classify_source_type,
                                     declared_kind, declared_kind_of)

FM = "---\ntype: {t}\nproject: p\ncapture_id: abc\n---\n# T\n\nbody\n"


@pytest.mark.parametrize("lane,path,expected", [
    (CAPTURE_LANE, "c:/fixtures/.claude/mainframe/captures/example-project/2026-01-01-000000-ab12cd34-ef56ab78.md", "session"),
    (KNOWLEDGE_LANE, "c:/fixtures/repos/x/claude.md", "project"),
    (KNOWLEDGE_LANE, "c:/fixtures/repos/x/agents.md", "project"),
    (KNOWLEDGE_LANE, "c:/fixtures/repos/x/.hermes.md", "project"),
    (KNOWLEDGE_LANE, "c:/fixtures/repos/x/research/2026-01-01-notes.md", "research"),
    (KNOWLEDGE_LANE, "c:/fixtures/repos/x/research/memory/x.md", "research"),
    (KNOWLEDGE_LANE, "c:/fixtures/repos/x/docs/arch.md", "docs"),
    (KNOWLEDGE_LANE, "c:/fixtures/repos/x/machines/m1.md", "docs"),
    (KNOWLEDGE_LANE, "c:/fixtures/repos/x/playbooks/p.md", "docs"),
    (KNOWLEDGE_LANE, "c:/fixtures/.claude/mainframe/library/ref.md", "library"),
])
def test_classify(lane, path, expected):
    assert classify_source_type(lane, path) == expected


def test_root_context_files_are_case_insensitive_on_canonical_paths():
    # canonical() lower-cases on Windows; the classifier must not depend on case.
    assert classify_source_type(KNOWLEDGE_LANE, "/repos/x/CLAUDE.md") == "project"
    assert "SOUL.md" not in ROOT_CONTEXT_FILES


def test_declared_kind_is_an_allowlist():
    assert PUBLISHED_KINDS == {"decision", "note"}
    assert declared_kind(FM.format(t="decision")) == "decision"
    assert declared_kind(FM.format(t="note")) == "note"
    for t in ("session", "Decision", "brand-new-kind", ""):
        assert declared_kind(FM.format(t=t)) == "session"
    assert declared_kind("# no fence\n\ntype: decision\n") == "session"
    assert declared_kind("---\ntype: decision\nno closing fence\n") == "session"


def test_one_rule_for_text_and_file(tmp_path):
    crlf = FM.format(t="decision").replace("\n", "\r\n")
    assert declared_kind(crlf) == "decision"
    p = tmp_path / "c.md"
    p.write_bytes(crlf.encode())
    assert declared_kind_of(p) == "decision"
    long = "---\ntype: decision\npad: " + "x" * 5000 + "\n---\nbody\n"   # fence closes past KIND_HEAD
    assert declared_kind(long) == "session"
    p.write_text(long, encoding="utf-8")
    assert declared_kind_of(p) == "session"
    assert declared_kind_of(tmp_path / "missing.md") == "session"


def test_the_header_bound_counts_characters_not_bytes(tmp_path):
    # ~5,000 bytes but ~2,550 characters: the fence closes inside KIND_HEAD in characters.
    text = "---\npad: " + "é" * 2500 + "\ntype: decision\n---\nbody\n"
    assert len(text) < KIND_HEAD < len(text.encode("utf-8"))
    p = tmp_path / "c.md"
    p.write_bytes(text.encode("utf-8"))
    assert declared_kind(text) == "decision"
    assert declared_kind_of(p) == "decision"


def test_the_header_bound_is_measured_after_crlf_normalization(tmp_path):
    # LF form closes at KIND_HEAD - 20 characters; CRLF adds one byte per line, past KIND_HEAD raw.
    lines = ["pad%d: %s" % (i, "x" * 60) for i in range(60)]
    lf = "---\n" + "\n".join(lines) + "\ntype: decision\n---\nbody\n"
    lf_close = lf.index("\n---\n") + 5
    assert lf_close < KIND_HEAD
    crlf = lf.replace("\n", "\r\n")
    assert crlf.index("\r\n---\r\n") + 7 > KIND_HEAD                  # raw characters would cut the fence
    p = tmp_path / "c.md"
    p.write_bytes(crlf.encode("utf-8"))
    assert declared_kind(crlf) == "decision" == declared_kind_of(p)


def test_a_capture_row_takes_its_kind_and_knowledge_ignores_it():
    assert classify_source_type(CAPTURE_LANE, "c:/m/captures/p/x.md", "decision") == "decision"
    assert classify_source_type(CAPTURE_LANE, "c:/m/captures/p/x.md") == "session"
    assert classify_source_type(KNOWLEDGE_LANE, "c:/r/p/research/x.md", "decision") == "research"
