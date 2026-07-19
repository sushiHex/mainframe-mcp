"""API contextual retrieval at ingest (b): opt-in wiring, plain fallback,
scrubbing, and the disabled-by-default guarantee. The real API call lives in
contextualizer.Contextualizer; tests inject fakes via mf.contextualizer."""

from pathlib import Path

import pytest

from conftest import make_mainframe
from mainframe_mcp.contextualizer import Contextualizer


@pytest.fixture
def mainframe(tmp_path):
    return make_mainframe(tmp_path)


class FakeContextualizer:
    enabled = True

    def __init__(self, prefix="This chunk covers"):
        self.prefix = prefix
        self.calls = 0

    def contextualize(self, doc_text, chunk_texts):
        self.calls += 1
        return [f"{self.prefix} {t.split()[0]} within the doc." for t in chunk_texts]


def _write_doc(mf, body):
    d = Path(mf.config["paths"]["repos_dir"]) / "proj" / "docs"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "doc.md"
    f.write_text(body, encoding="utf-8")
    return f


def test_contexts_prepended_to_indexed_text(mainframe):
    mf = mainframe
    mf.contextualizer = FakeContextualizer()
    f = _write_doc(mf, "# Doc\n\nQwen3 reranker hybrid facts live here.\n")
    assert mf.ingest_file(str(f))["status"] == "ingested"
    df = mf.store.table.to_pandas()
    row = df[df["doc_path"] == str(f)].iloc[0]
    assert row["text"].startswith("This chunk covers")   # context prepended
    assert "Qwen3 reranker hybrid facts" in row["text"]  # original preserved


def test_disabled_by_default_leaves_chunks_plain(mainframe):
    mf = mainframe  # DEFAULTS: contextual.enabled = False -> real Contextualizer, disabled
    f = _write_doc(mf, "# Doc\n\nPlain chunk body without any context.\n")
    assert mf.ingest_file(str(f))["status"] == "ingested"
    df = mf.store.table.to_pandas()
    row = df[df["doc_path"] == str(f)].iloc[0]
    assert "Plain chunk body" in row["text"]
    assert not row["text"].startswith("This chunk")
    assert isinstance(mf.contextualizer, Contextualizer)
    assert mf.contextualizer.enabled is False


def test_context_failure_falls_back_to_plain(mainframe):
    mf = mainframe

    class HalfBroken(FakeContextualizer):
        def contextualize(self, doc_text, chunk_texts):
            return [None] * len(chunk_texts)  # API failed for every chunk

    mf.contextualizer = HalfBroken()
    f = _write_doc(mf, "# Doc\n\nResilient body text stays indexed anyway.\n")
    assert mf.ingest_file(str(f))["status"] == "ingested"
    df = mf.store.table.to_pandas()
    assert (df["doc_path"] == str(f)).any()


def test_context_is_scrubbed_before_indexing(mainframe):
    mf = mainframe

    class Leaky(FakeContextualizer):
        def contextualize(self, doc_text, chunk_texts):
            return ["Context mentioning api_key=SUPERSEKRET123456 here." for _ in chunk_texts]

    mf.contextualizer = Leaky()
    f = _write_doc(mf, "# Doc\n\nSome body content for the chunk.\n")
    assert mf.ingest_file(str(f))["status"] == "ingested"
    df = mf.store.table.to_pandas()
    row = df[df["doc_path"] == str(f)].iloc[0]
    assert "SUPERSEKRET123456" not in row["text"]


def test_contextualizer_disabled_without_config():
    # Real class, enabled=False path: never constructs a client, returns Nones
    c = Contextualizer({"contextual": {"enabled": False}})
    assert c.enabled is False
    assert c.contextualize("doc", ["a", "b"]) == [None, None]
