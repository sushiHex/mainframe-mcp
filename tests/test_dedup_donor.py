"""§9.5 — exact-dedup donor scoping: a session chunk can't evict a curated one."""

from pathlib import Path

# Title (<10 tokens) merges into the Intro section, so the Shared section below
# is an independent chunk whose text is byte-identical across both files.
SHARED_SECTION = (
    "## Shared\n"
    "Identical shared paragraph about Qwen3 bge-reranker on the RTX 3090 with "
    "256 token chunks appears verbatim in both files for the dedup donor test.\n"
)


def _doc(title_word, intro):
    return f"# Doc {title_word}\n\n## Intro\n{intro}\n\n{SHARED_SECTION}"


def test_session_chunk_does_not_evict_curated_chunk(mainframe):
    mf = mainframe
    repos = Path(mf.config["paths"]["repos_dir"])
    sess = repos / "p" / "research" / "sessions"
    sess.mkdir(parents=True)
    res = repos / "p" / "research"

    # Session captured FIRST (would be the donor under the old global dedup).
    sfile = sess / "2026-06-27-000000-abcd1234-s.md"
    sfile.write_text(
        _doc("S", "Intro for the session document with clearly more than ten tokens here."),
        encoding="utf-8",
    )
    assert mf.ingest_file(str(sfile))["status"] == "ingested"

    # Curated research file sharing the identical Shared chunk.
    rfile = res / "curated.md"
    rfile.write_text(
        _doc("R", "Intro for the curated research document, also well over ten tokens long."),
        encoding="utf-8",
    )
    assert mf.ingest_file(str(rfile))["status"] == "ingested"

    # The curated copy of the identical chunk must survive (not skipped as a dupe
    # of the earlier session chunk).
    df = mf.store.table.to_pandas()
    curated_shared = df[
        (df["source_type"] == "research")
        & (df["text"].str.contains("Identical shared paragraph"))
    ]
    assert len(curated_shared) >= 1
