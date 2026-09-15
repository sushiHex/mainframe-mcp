from mainframe.core.indexer import UNCHANGED, VANISHED, DocFailure, EmbedderUnavailable, LaneFile, prepare_document
from mainframe.core.paths import canonical, doc_id, chunk_key
from mainframe.core.store import DocRows
from v2.helpers import write_md

CHUNK = {"chunk_size": 256, "overlap_ratio": 0.35}


def test_rows_carry_identity_and_classification(tmp_path, fake_embedder):
    # Each section is > 10 estimated tokens so the chunker keeps them separate
    # (sections under MIN_SECTION_TOKENS are merged into the next one).
    p = write_md(tmp_path / "repos" / "proj" / "docs" / "a.md",
                 "# A\n\nfirst section text with enough words to stand alone here.\n\n"
                 "## B\n\nsecond section text with enough words to stand alone too.\n")
    lf = LaneFile(path=canonical(p), lane="knowledge", project="proj")
    d = prepare_document(lf, CHUNK, fake_embedder.embed, now="2026-01-01T00:00:00")
    assert isinstance(d, DocRows) and d.doc_id == doc_id(lf.path) and len(d.rows) == 2
    r = d.rows[1]
    assert r["chunk_key"] == chunk_key(d.doc_id, 1) and r["chunk_index"] == 1
    assert r["lane"] == "knowledge" and r["source_type"] == "docs" and r["project"] == "proj"
    assert len(r["vector"]) == 64 and r["indexed_at"] == "2026-01-01T00:00:00"
    assert r["file_hash"] == d.rows[0]["file_hash"] and r["authored_at"]


def test_a_device_failure_propagates_but_an_ordinary_one_becomes_a_docfailure(tmp_path, fake_embedder):
    """A `DocFailure` says "THIS document is bad" — a dead CUDA context is not
    that, and reporting it as one told the caller a lie it acted on: the batch
    "succeeded" with every file marked failed, `Models.invoke` never saw the
    exception, and nothing reloaded. The predicate decides which is which."""
    import pytest

    p = write_md(tmp_path / "a.md", "# A\n\nbody text here.\n")
    lf = LaneFile(path=canonical(p), lane="knowledge", project="x")

    class Bad:
        def __init__(self, exc):
            self.exc = exc

        def embed(self, texts, batch_size=None):
            raise self.exc

    with pytest.raises(RuntimeError, match="CUDA error"):
        prepare_document(lf, CHUNK, Bad(RuntimeError("CUDA error: unknown error")).embed)

    ordinary = prepare_document(lf, CHUNK, Bad(ValueError("token id out of range")).embed)
    assert isinstance(ordinary, DocFailure) and "embed: token id out of range" in ordinary.error


def test_embedder_unavailable_propagates_unlike_an_ordinary_failure(tmp_path, fake_embedder):
    """`EmbedderUnavailable` is the `embed` callable's own way of saying the
    model was never resolved at all (see `core/pipeline.py`'s `embed`
    closure) — it must propagate exactly like a device fault, never becoming
    a `DocFailure`, even though its message matches no CUDA/driver pattern
    `is_device_failure` would recognize."""
    import pytest

    p = write_md(tmp_path / "a.md", "# A\n\nbody text here.\n")
    lf = LaneFile(path=canonical(p), lane="knowledge", project="x")

    def unavailable(texts):
        raise EmbedderUnavailable("embedder unavailable: no weights on disk; retry in 900s")

    with pytest.raises(EmbedderUnavailable, match="no weights on disk"):
        prepare_document(lf, CHUNK, unavailable)


def test_unchanged_when_hash_matches(tmp_path, fake_embedder):
    p = write_md(tmp_path / "a.md", "# A\n\nbody text here.\n")
    lf = LaneFile(path=canonical(p), lane="knowledge", project="x")
    first = prepare_document(lf, CHUNK, fake_embedder.embed)
    assert prepare_document(lf, CHUNK, fake_embedder.embed, known_hash=first.rows[0]["file_hash"]) is UNCHANGED


def test_empty_file_yields_no_rows(tmp_path, fake_embedder):
    p = write_md(tmp_path / "e.md", "   \n\n")
    d = prepare_document(LaneFile(canonical(p), "knowledge", "x"), CHUNK, fake_embedder.embed)
    assert isinstance(d, DocRows) and d.rows == []


def test_missing_file_is_vanished(tmp_path, fake_embedder):
    lf = LaneFile(canonical(tmp_path / "missing.md"), "knowledge", "x")
    assert prepare_document(lf, CHUNK, fake_embedder.embed) is VANISHED


def test_capture_lane_is_scrubbed_and_typed_session(tmp_path, fake_embedder):
    p = write_md(tmp_path / "c.md", "# S\n\n## Summary\nuse api_key=example-body now.\n")
    d = prepare_document(LaneFile(canonical(p), "capture", "proj"), CHUNK, fake_embedder.embed)
    assert d.rows[0]["source_type"] == "session" and d.rows[0]["lane"] == "capture"
    assert "example-body" not in d.rows[0]["text"] and "[REDACTED]" in d.rows[0]["text"]


def test_contextualizer_prefix_applied(tmp_path, fake_embedder):
    class Ctx:
        enabled = True
        def contextualize(self, doc, chunks):
            return ["CTX for chunk"] * len(chunks)
    p = write_md(tmp_path / "a.md", "# A\n\nbody text here.\n")
    d = prepare_document(LaneFile(canonical(p), "knowledge", "x"), CHUNK, fake_embedder.embed, contextualizer=Ctx())
    assert d.rows[0]["text"].startswith("CTX for chunk\n\n")


def test_raising_contextualizer_degrades_to_plain_chunks(tmp_path, fake_embedder):
    class Boom:
        enabled = True
        def contextualize(self, doc, chunks):
            raise RuntimeError("api down")
    p = write_md(tmp_path / "a.md", "# A\n\nbody text here.\n")
    d = prepare_document(LaneFile(canonical(p), "knowledge", "x"), CHUNK, fake_embedder.embed, contextualizer=Boom())
    assert isinstance(d, DocRows) and d.rows[0]["text"] == "# A\n\nbody text here."


def test_frontmatter_kept_by_default(tmp_path, fake_embedder, cfg):
    # chunker.strip_frontmatter defaults False: v1 (src/mainframe_mcp/) never
    # stripped frontmatter. Preserve that chunking behavior by default;
    # enabling stripping requires its own retrieval evaluation.
    text = "---\ntype: note\nproject: p\n---\n\n# T\n\nbody text here.\n"
    p = write_md(tmp_path / "n.md", text)
    lf = LaneFile(canonical(p), "knowledge", "p")
    d = prepare_document(lf, cfg["chunker"], fake_embedder.embed)
    assert isinstance(d, DocRows) and any("type: note" in r["text"] for r in d.rows)


def test_strip_frontmatter_flag_excludes_it_and_keeps_offsets_aligned(tmp_path, fake_embedder, cfg):
    text = "---\ntype: note\nproject: p\n---\n\n# T\n\nbody text here.\n"
    p = write_md(tmp_path / "n.md", text)
    lf = LaneFile(canonical(p), "knowledge", "p")
    chunk_cfg = dict(cfg["chunker"], strip_frontmatter=True)
    d = prepare_document(lf, chunk_cfg, fake_embedder.embed)
    assert isinstance(d, DocRows) and not any("type: note" in r["text"] for r in d.rows)
    # chunk_markdown stores each chunk's .strip()ped text but char_start/char_end
    # span the UNSTRIPPED section — a pre-existing chunker property, not
    # introduced by this flag (search.py's own line-span verifier compares
    # span.strip() to the stored text for the same reason) — so the alignment
    # check strips both sides rather than requiring exact equality.
    for r in d.rows:
        assert text[r["char_start"]:r["char_end"]].strip() == r["text"]
