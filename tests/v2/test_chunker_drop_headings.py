"""Corpus hygiene: `chunker.drop_headings` lets a config exclude generated
run-telemetry sections (see the drop-generated-headings brief / issue #25) at
CHUNK time, before hashing, embedding or FTS ever see them. Off by default
(empty list); GPU-free."""
import json
import re

import pytest

from mainframe.config import DEFAULTS, ConfigError, deep_copy, load_config
from mainframe.core.chunker import chunk_markdown
from mainframe.core.store import Store

FIXTURE = """# Investigation Report

## Findings

The retrieval regression traced back to three generated telemetry sections
outranking the real answer, which turned out to be corpus hygiene rather
than a reranker defect worth chasing any further down that path.

## Chain E — ERROR: Timed out after 480s

---
**Oracle SDK Execution Metrics**
- Total time: 685s
- Total tokens: 388,099 (in: 114,603 | out: 273,496)
- Model calls: 12
- Retries: 3

## Conclusion

Excluding the telemetry sections at chunk time restored the real answer to
the candidate pool without touching the reranker or the embedder at all.
"""

DROPPED_HEADING = "Chain E — ERROR: Timed out after 480s"
TELEMETRY_PATTERN = r"^Chain [A-Z] — ERROR: Timed out"


# ---------- chunk_markdown behavior ----------

def test_default_empty_list_is_identical_to_no_feature():
    baseline = chunk_markdown(FIXTURE, 256, 0.35)
    assert chunk_markdown(FIXTURE, 256, 0.35, drop_headings=[]) == baseline
    assert chunk_markdown(FIXTURE, 256, 0.35, drop_headings=None) == baseline


def test_one_pattern_drops_only_the_matching_sections():
    baseline = chunk_markdown(FIXTURE, 256, 0.35)
    filtered = chunk_markdown(FIXTURE, 256, 0.35, drop_headings=[TELEMETRY_PATTERN])

    assert DROPPED_HEADING in [c.heading for c in baseline]
    assert DROPPED_HEADING not in [c.heading for c in filtered]

    # Every other chunk is unchanged: same text, same heading, same order
    # (char offsets too, since surviving sections were never re-sliced).
    kept = [c for c in baseline if c.heading != DROPPED_HEADING]
    assert [c.text for c in filtered] == [c.text for c in kept]
    assert [c.heading for c in filtered] == [c.heading for c in kept]
    assert [c.char_start for c in filtered] == [c.char_start for c in kept]
    assert [c.char_end for c in filtered] == [c.char_end for c in kept]
    assert [c.token_count for c in filtered] == [c.token_count for c in kept]
    # chunk_index is contiguously renumbered over the survivors, not a gapped
    # carry-over of the baseline numbering.
    assert [c.chunk_index for c in filtered] == list(range(len(filtered)))


def test_a_pattern_matching_nothing_changes_nothing():
    baseline = chunk_markdown(FIXTURE, 256, 0.35)
    filtered = chunk_markdown(FIXTURE, 256, 0.35, drop_headings=[r"^Nothing Matches This Heading$"])
    assert filtered == baseline


# ---------- fingerprint_from_config ----------

def test_fingerprint_differs_by_pattern_set_not_by_order():
    a = deep_copy(DEFAULTS)
    a["chunker"]["drop_headings"] = [TELEMETRY_PATTERN, r"^Oracle SDK"]
    b = deep_copy(DEFAULTS)
    b["chunker"]["drop_headings"] = [r"^Oracle SDK", TELEMETRY_PATTERN]
    c = deep_copy(DEFAULTS)
    c["chunker"]["drop_headings"] = [TELEMETRY_PATTERN]

    fp_a = Store.fingerprint_from_config(a)
    fp_b = Store.fingerprint_from_config(b)
    fp_c = Store.fingerprint_from_config(c)

    assert fp_a == fp_b        # reordering a user's patterns is not a real change
    assert fp_a != fp_c        # a different pattern SET is a real change


def test_empty_drop_headings_fingerprints_identically_to_before_the_key_existed():
    """Load-bearing (brief point 4): an existing index's marker was written
    before `drop_headings` existed, so it has no such key at all. Upgrading
    to this feature with the default `[]` must not read as drift, or every
    installed index would demand an offline rebuild on the next restart."""
    with_key = deep_copy(DEFAULTS)
    assert with_key["chunker"]["drop_headings"] == []
    without_key = deep_copy(DEFAULTS)
    del without_key["chunker"]["drop_headings"]   # simulate a pre-upgrade config

    fp_with_empty = Store.fingerprint_from_config(with_key)
    fp_without_key = Store.fingerprint_from_config(without_key)

    assert fp_with_empty == fp_without_key
    assert "drop_headings" not in fp_with_empty


# ---------- config validation ----------

def test_invalid_regex_raises_config_error_naming_the_pattern(tmp_path):
    bad = tmp_path / "bad-drop-headings.json"
    bad.write_text(json.dumps({"chunker": {"drop_headings": ["(unclosed"]}}), encoding="utf-8")
    with pytest.raises(ConfigError, match=re.escape("(unclosed")):
        load_config(bad)
