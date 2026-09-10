import pytest

from mainframe.core.classify import (CAPTURE_LANE, KNOWLEDGE_LANE, ROOT_CONTEXT_FILES,
                                     classify_source_type)


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
