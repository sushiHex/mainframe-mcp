"""The kept chunker must produce byte-identical chunks to v1 — the retrieval
baseline depends on it. Runs over this repo's own markdown (no GPU)."""
from pathlib import Path

from mainframe.core.chunker import chunk_markdown as v2
from mainframe_mcp.chunker import chunk_markdown as v1

ROOT = Path(__file__).resolve().parents[2]


def test_chunks_identical_on_repo_docs():
    files = sorted(list((ROOT / "docs").rglob("*.md")) + [ROOT / "CLAUDE.md", ROOT / "README.md"])
    assert files, "no docs found"
    total = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        a = [(c.text, c.heading, c.chunk_index, c.char_start, c.char_end, c.token_count) for c in v1(text, 256, 0.35)]
        b = [(c.text, c.heading, c.chunk_index, c.char_start, c.char_end, c.token_count) for c in v2(text, 256, 0.35)]
        assert a == b, f"chunker drift in {f}"
        total += len(a)
    # Documentation size varies between checkouts; parity requires nonempty
    # coverage, not the size of a particular corpus.
    assert total > 0
