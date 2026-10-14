"""The chunker must produce byte-identical chunks to the ones the retrieval
baseline was measured under.

`fixtures/chunker_baseline.json` was generated from the original chunker (the
retired `mainframe_mcp.chunker`, whose output `mainframe.core.chunker` had
matched byte-for-byte on every repo document) at the moment that package was
removed. It pins what the eval composite score and the index fingerprint were
measured against: a change that moves any field here moves the baseline and
needs its own evaluation and an offline rebuild, not a regenerated golden."""
import json
from pathlib import Path

from mainframe.core.chunker import chunk_markdown

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_chunks_match_the_measured_baseline():
    text = (FIXTURES / "chunker_baseline.md").read_text(encoding="utf-8")
    expected = json.loads((FIXTURES / "chunker_baseline.json").read_text(encoding="utf-8"))
    actual = [{"chunk_index": c.chunk_index, "heading": c.heading, "char_start": c.char_start,
               "char_end": c.char_end, "token_count": c.token_count, "text": c.text}
              for c in chunk_markdown(text, 256, 0.35)]
    assert actual == expected, "chunker drift from the measured baseline"
    # The fixture must keep exercising overlap (one section spanning several
    # chunks) and tiny-section merging, or the golden proves less than it claims.
    headings = [c["heading"] for c in actual]
    assert headings.count("A long section") > 1
    assert "Tiny section" not in headings
