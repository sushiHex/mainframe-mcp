from pathlib import Path

import pytest

from mainframe.core.hashing import file_hash
from mainframe.core.indexer import LaneFile, prepare_document
from mainframe.core.paths import canonical
from mainframe.core.search import MAX_RESULTS, SearchService, line_span
from v2.helpers import write_md

CHUNK = {"chunk_size": 256, "overlap_ratio": 0.35}


def _index(store, embedder, files):
    docs = [prepare_document(lf, CHUNK, embedder) for lf in files]
    store.upsert_batch(docs)
    store.optimize()


def _models(cfg, embedder, reranker):
    """SearchService holds the REGISTRY, not the models: every GPU call goes
    through `Models.invoke`, so a reload underneath it is invisible here."""
    from mainframe.core.models import Models
    return Models(cfg, embedder_factory=lambda c: embedder, reranker_factory=lambda c: reranker)


def _svc(cfg, store, fake_embedder, fake_reranker):
    return SearchService(_models(cfg, fake_embedder, fake_reranker), store, cfg)


def test_concise_shape_and_line_spans(cfg, store, fake_embedder, fake_reranker, tmp_path):
    p = write_md(tmp_path / "repos" / "p" / "docs" / "a.md",
                 "# Title\n\nintro line with enough words to stand on its own as a section here.\n\n"
                 "## Qwen reranker\n\nqwen3 reranker hybrid search details here.\n")
    _index(store, fake_embedder, [LaneFile(canonical(p), "knowledge", "p")])
    out = _svc(cfg, store, fake_embedder, fake_reranker).search("qwen3 reranker hybrid", limit=3)
    assert out["confidence"] == "high" and "guidance" not in out
    r = out["results"][0]
    assert set(r) == {"file", "heading", "line_start", "line_end", "rerank_score", "snippet"}
    assert r["heading"] == "Qwen reranker" and r["line_start"] == 5 and r["line_end"] == 7
    assert len(r["snippet"]) <= 200


def test_detailed_shape(cfg, store, fake_embedder, fake_reranker, tmp_path):
    p = write_md(tmp_path / "repos" / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    _index(store, fake_embedder, [LaneFile(canonical(p), "knowledge", "p")])
    r = _svc(cfg, store, fake_embedder, fake_reranker).search("qwen3", response_format="detailed")["results"][0]
    for k in ("text", "truncated", "char_start", "char_end", "score", "source_type", "project", "file", "rerank_score", "snippet"):
        assert k in r
    assert r["truncated"] is False and r["project"] == "p"


def test_line_spans_absent_when_file_changed(cfg, store, fake_embedder, fake_reranker, tmp_path):
    p = write_md(tmp_path / "repos" / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    _index(store, fake_embedder, [LaneFile(canonical(p), "knowledge", "p")])
    p.write_text("# T\n\nsomething else entirely now.\n", encoding="utf-8")
    r = _svc(cfg, store, fake_embedder, fake_reranker).search("qwen3")["results"][0]
    assert "line_start" not in r and "line_end" not in r


def test_merged_capture_citations_need_no_reindex(cfg, store, fake_embedder, fake_reranker, tmp_path):
    text = ("---\ntype: session\nproject: p\n---\n"
            "# Session\n\n# Orchid receipt\n\n"
            "Orchid receipts record immutable diagnostics for the memory lane.\n")
    p = write_md(tmp_path / "mainframe" / "captures" / "p" / "receipt.md", text)
    doc = prepare_document(LaneFile(canonical(p), "capture", "p"), CHUNK, fake_embedder)
    # Freeze the stored passage produced before this fix: each tiny section
    # adds a newline, so neither its text nor its end offset matches the file.
    legacy_text = text.replace("---\n# Session", "---\n\n# Session").replace(
        "# Session\n\n# Orchid", "# Session\n\n\n# Orchid").strip()
    assert len(doc.rows) == 1 and doc.rows[0]["text"] == legacy_text
    assert doc.rows[0]["char_end"] == len(text) + 2
    store.upsert_batch([doc])
    store.optimize()
    version = store.table.version
    svc = _svc(cfg, store, fake_embedder, fake_reranker)
    for fmt in ("concise", "detailed", "full"):
        r = svc.search("Orchid receipts", include_sessions=True, response_format=fmt)["results"][0]
        assert (r["line_start"], r["line_end"]) == (1, 9)
        if fmt != "concise":
            assert r["text"] == legacy_text
            assert text[r["char_start"]:r["char_end"]] == text.strip()
    assert store.table.version == version


@pytest.mark.parametrize("source,passage,start,end,expected", [
    ("# One\n\n# Two\n\nbody\n", "# One\n\n\n# Two\n\nbody", 0, 20, (0, 18)),
    # A split chunk can start after the inserted newline; both offsets drift.
    ("prefix\nunique body\ntail\n", "unique body", 8, 19, (7, 18)),
    # Exact offsets disambiguate repeated passages; recovery must not guess.
    ("same\nsame\n", "same", 5, 9, (5, 9)),
    ("same\nsame\n", "same", 1, 5, None),
    ("x\nx\nx", "x\n\nx", 0, 6, None),  # overlapping matches are ambiguous
    ("one\n\ntwo", "one\ntwo", 0, 7, None),  # never add missing source newlines
    ("if ready:\n    run()", "if ready:\nrun()", 0, 15, None),
    ("secret=original", "secret=[REDACTED]", 0, 16, None),
    ("original body", "context\n\noriginal body", 0, 13, None),
    ("body", " \n ", 0, 4, None),
])
def test_citation_recovery_verifies_unique_source(tmp_path, source, passage, start, end, expected):
    p = write_md(tmp_path / "source.md", source)
    row = {"doc_path": str(p), "file_hash": file_hash(source), "text": passage,
           "char_start": start, "char_end": end}
    out = {"char_start": start, "char_end": end}
    SearchService._add_line_spans([row], [(0, 1.0)], [out])
    if expected is None:
        assert "line_start" not in out and "line_end" not in out
    else:
        assert (out["char_start"], out["char_end"]) == expected
        assert (out["line_start"], out["line_end"]) == line_span(source, *expected)


def test_citation_recovery_still_requires_indexed_revision(tmp_path):
    original = "# Session\n\n# Receipt\n\nunique diagnostic receipt\n"
    p = write_md(tmp_path / "capture.md", original + "later edit\n")
    row = {"doc_path": str(p), "file_hash": file_hash(original),
           "text": original.replace("# Session\n\n", "# Session\n\n\n").strip(),
           "char_start": 0, "char_end": len(original) + 1}
    out = {}
    SearchService._add_line_spans([row], [(0, 1.0)], [out])
    assert not out


def test_empty_and_low_confidence_guidance(cfg, store, fake_embedder, tmp_path):
    class LowReranker:
        enabled = True; heading_inject = False; default_top_k = 3
        def rerank(self, q, docs, top_k=None, headings=None):
            return [(i, 0.1) for i in range(min(top_k or 3, len(docs)))]
    svc = SearchService(_models(cfg, fake_embedder, LowReranker()), store, cfg)
    out = svc.search("anything")
    assert out["confidence"] == "none" and "include_sessions" in out["guidance"]
    p = write_md(tmp_path / "repos" / "p" / "docs" / "a.md", "# T\n\nqwen3 reranker text.\n")
    _index(store, fake_embedder, [LaneFile(canonical(p), "knowledge", "p")])
    out = svc.search("qwen3")
    assert out["confidence"] == "low" and "rerank_score" in out["guidance"]


def test_limit_is_clamped_and_sessions_opt_in(cfg, store, fake_embedder, fake_reranker, tmp_path):
    k = write_md(tmp_path / "repos" / "p" / "docs" / "a.md", "# T\n\nqwen3 alpha.\n")
    c = write_md(tmp_path / "mainframe" / "captures" / "p" / "2026-01-01-000000-s-c.md", "# S\n\nqwen3 alpha capture.\n")
    _index(store, fake_embedder, [LaneFile(canonical(k), "knowledge", "p"), LaneFile(canonical(c), "capture", "p")])
    svc = _svc(cfg, store, fake_embedder, fake_reranker)
    assert len(svc.search("qwen3", limit=1000)["results"]) <= MAX_RESULTS
    files = {r["file"] for r in svc.search("qwen3", limit=10)["results"]}
    assert all("captures" not in f for f in files)
    files = {r["file"] for r in svc.search("qwen3", limit=10, include_sessions=True)["results"]}
    assert any("captures" in f for f in files)


def test_full_format_is_untruncated_and_unknown_format_is_rejected(cfg, store, fake_embedder, fake_reranker, tmp_path):
    filler = "filler padding words here to lengthen the chunk substantially. " * 12
    body = f"qwen3 reranker text. {filler}"
    p = write_md(tmp_path / "repos" / "p" / "docs" / "a.md", f"# T\n\n{body}\n")
    _index(store, fake_embedder, [LaneFile(canonical(p), "knowledge", "p")])
    svc = _svc(cfg, store, fake_embedder, fake_reranker)

    detailed = svc.search("qwen3", response_format="detailed")["results"][0]
    assert len(detailed["text"]) == 500 and detailed["truncated"] is True

    full = svc.search("qwen3", response_format="full")["results"][0]
    assert len(full["text"]) > 600 and full["truncated"] is False
    assert full["text"].startswith(detailed["text"])
    for k in ("char_start", "char_end", "score", "source_type", "project", "file", "rerank_score", "snippet"):
        assert k in full

    import pytest
    with pytest.raises(ValueError):
        svc.search("qwen3", response_format="bogus")


def test_line_span_helper():
    text = "a\nb\nc\nd\n"
    assert line_span(text, 2, 4) == (2, 2)      # "b\n" -> line 2 only
    assert line_span(text, 2, 6) == (2, 3)      # "b\nc\n" -> lines 2-3
