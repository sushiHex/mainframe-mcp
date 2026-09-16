"""Guards skills/mainframe-retrieval/SKILL.md against drifting from the real
`search` contract (issue #22; re-pointed for `reranked`).

History: the skill once documented a `confidence` field ("high"/"low"/"none",
then just "unavailable"/"none" after PR #17 removed the absolute threshold).
An incident showed that field was worse than useless: its only two values
restated whether `results` was empty, and "unavailable" read as "the ranking
was unavailable" to an operator with no other way to check how a result was
ranked. It was replaced with `ranked_by` (the model NAME that ordered the
results), which a second review rejected before merge: the model value is
config-controlled and unvalidated, and in this project a model value can be a
local filesystem path (see `eval/.eval-lancedb.meta.json`'s `embedder_model`)
-- carrying it in every search response would leak a home directory into
every agent transcript. `reranked` replaces it: the caller's only real
question ("is `rerank_score` a ranking signal?") is a boolean, so the field
is one.

The PREVIOUS version of this guard imported `CONFIDENCE_VALUES` from
`mainframe.core.search` and checked the skill's documented values against it
-- a real code -> doc link. A later rewrite (during the `ranked_by` interlude)
imported only `re` and `Path`, so it verified the skill's prose against a
hardcoded regex and never touched the code at all; renaming the response key
in `core/search.py` would have left it green. This version imports the
exported key constant `RERANKED_KEY` and builds its pattern from that live
value, so a rename that isn't mirrored in SKILL.md fails here.
"""
import re
from pathlib import Path

from mainframe.core.search import RERANKED_KEY

SKILL = Path(__file__).resolve().parents[2] / "skills" / "mainframe-retrieval" / "SKILL.md"

# A JSON-key or backtick-quoted field reference to `key`, e.g. `"reranked":`
# or `` `reranked` ``. Deliberately narrower than the bare word: it must not
# forbid ordinary prose use of a retired field's name (e.g. "not a
# calibrated-confidence score", which CLAUDE.md/README.md/docs/V2.md/
# docs/RERANKER_COMPARISON.md all write deliberately).
def _field_pattern(key: str) -> re.Pattern:
    escaped = re.escape(key)
    return re.compile(rf'"{escaped}"\s*:|`{escaped}`')


_RERANKED_FIELD_RE = _field_pattern(RERANKED_KEY)
# The two field shapes `reranked` superseded. Either reappearing as a real
# field reference means the skill has drifted back toward a response shape
# the code no longer returns.
_CONFIDENCE_FIELD_RE = _field_pattern("confidence")
_RANKED_BY_FIELD_RE = _field_pattern("ranked_by")
# A numeric threshold compared against a rerank-style score, e.g. `>0.85` or `<0.3`.
_NUMERIC_THRESHOLD_RE = re.compile(r'[<>]\s*0\.\d+')


def _skill_text() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_skill_documents_the_reranked_key():
    text = _skill_text()
    assert _RERANKED_FIELD_RE.search(text), (
        f"expected SKILL.md to document mainframe.core.search.RERANKED_KEY "
        f"({RERANKED_KEY!r}) as a JSON key or backtick-quoted field reference"
    )


def test_skill_does_not_document_retired_field_forms():
    text = _skill_text()
    assert not _CONFIDENCE_FIELD_RE.search(text), (
        "SKILL.md still documents a `confidence` field -- that shape was "
        "removed (superseded by `ranked_by`, itself superseded by the "
        f"boolean `{RERANKED_KEY}`); the skill has drifted from the code again."
    )
    assert not _RANKED_BY_FIELD_RE.search(text), (
        "SKILL.md still documents a `ranked_by` field -- that shape was "
        f"replaced by the boolean `{RERANKED_KEY}` because a model name in "
        "the response can leak a filesystem path and invites model-card "
        "reasoning over using the ordering; the skill has drifted from the "
        "code again."
    )


def test_skill_states_rerank_score_is_not_a_confidence_score():
    """Item 4 of the ranked_by-review fix: the retired-field ban above must
    not block the caveat the rest of the repo writes deliberately
    (CLAUDE.md, README.md, docs/V2.md, docs/RERANKER_COMPARISON.md all say a
    rerank score is not correctness/calibrated confidence). Only the
    JSON/field-reference FORM of `confidence` is banned, not the word."""
    text = _skill_text()
    assert re.search(r"not\s+a\s+correctness\s+or\s+calibrated-confidence\s+score", text, re.IGNORECASE), (
        "expected SKILL.md to state that rerank_score is not a correctness "
        "or calibrated-confidence score"
    )
    assert not _CONFIDENCE_FIELD_RE.search(text)


def test_skill_has_no_numeric_rerank_threshold():
    text = _skill_text()
    matches = _NUMERIC_THRESHOLD_RE.findall(text)
    assert not matches, (
        f"SKILL.md still asserts a numeric rerank_score threshold {matches!r}; "
        "PR #17 removed the absolute confidence threshold because reranker "
        "backends emit incompatible scoring scales."
    )
