"""The consolidator's prompt-budget math is the load-bearing memory-correctness
path (newer facts must survive; the old note gets truncated, never the new
reports) — previously GPU-only and tested exclusively through trivial fakes.
build_prompt() is pure: tokenizer injected, no model."""

from mainframe_mcp.consolidator import (
    CONTEXT_WINDOW,
    build_prompt,
    skeleton_tokens,
)


class WordTokenizer:
    """1 token per whitespace-delimited word — deterministic, GPU-free."""

    def encode(self, text):
        return text.split(" ")

    def decode(self, tokens):
        return " ".join(tokens)


def _mk(text="fact one", n=1):
    return [{"title": f"r{i}", "path": f"C:/r{i}.md", "text": text} for i in range(n)]


def _prompt(reports, existing="", max_new_tokens=4096):
    tok = WordTokenizer()
    upd, add = skeleton_tokens(tok)
    return build_prompt(reports, existing, tok, max_new_tokens, upd, add)


def test_fresh_merge_uses_add_prompt():
    p = _prompt(_mk("alpha beta gamma"))
    assert "knowledge consolidator" in p and "UPDATING" not in p
    assert "alpha beta gamma" in p


def test_update_path_puts_new_reports_before_existing():
    p = _prompt(_mk("newest finding wins"), existing="old canonical body")
    assert "UPDATING" in p
    assert p.index("newest finding wins") < p.index("old canonical body"), \
        "NEW reports must precede the existing note (priority order)"


def test_oversized_existing_is_truncated_reports_survive():
    huge_existing = " ".join(f"old{i}" for i in range(CONTEXT_WINDOW))
    p = _prompt(_mk("critical new fact"), existing=huge_existing, max_new_tokens=4096)
    assert "critical new fact" in p, "new reports must never be truncated away"
    assert "UPDATING" in p
    assert len(p.split(" ")) <= CONTEXT_WINDOW - 4096 - 512 + 8  # within budget (+format slack)


def test_reports_exceeding_update_budget_fall_back_to_fresh_merge():
    huge_reports = _mk(" ".join(f"new{i}" for i in range(CONTEXT_WINDOW)))
    p = _prompt(huge_reports, existing="old canonical body")
    assert "UPDATING" not in p, "no room for the old note -> plain merge"
    assert "old canonical body" not in p
    assert "new0" in p  # truncated but present


def test_oversized_fresh_merge_is_truncated_to_budget():
    huge_reports = _mk(" ".join(f"w{i}" for i in range(CONTEXT_WINDOW * 2)))
    p = _prompt(huge_reports)
    assert len(p.split(" ")) <= CONTEXT_WINDOW - 4096 - 512 + 8
