"""Guards skills/mainframe-retrieval/SKILL.md against drifting from the real
`search` contract (issue #22; re-pointed for `ranked_by`).

The skill previously documented a `confidence` field ("high"/"low"/"none",
then just "unavailable"/"none" after PR #17 removed the absolute threshold).
An incident showed that field was worse than useless: its only two values
restated whether `results` was empty, and "unavailable" read as "the ranking
was unavailable" to an operator who had no other way to check how a result
was ranked. The fix replaced `confidence` with `ranked_by` — the name of the
model that ordered the results, or `null` when nothing did — so a response
can answer that question about itself. This test now fails if the skill
still documents the retired `confidence` vocabulary, and fails if it does not
document `ranked_by`, so the skill and the code cannot drift apart again.
"""
import re
from pathlib import Path

SKILL = Path(__file__).resolve().parents[2] / "skills" / "mainframe-retrieval" / "SKILL.md"

_RANKED_BY_RE = re.compile(r"`?ranked_by`?")
_CONFIDENCE_RE = re.compile(r"confidence", re.IGNORECASE)
# A numeric threshold compared against a rerank-style score, e.g. `>0.85` or `<0.3`.
_NUMERIC_THRESHOLD_RE = re.compile(r'[<>]\s*0\.\d+')


def _skill_text() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_skill_documents_ranked_by_and_not_confidence():
    text = _skill_text()
    assert _RANKED_BY_RE.search(text), "expected SKILL.md to document `ranked_by`"
    assert not _CONFIDENCE_RE.search(text), (
        "SKILL.md still documents `confidence` -- that field was removed in "
        "favor of `ranked_by` (see the ranked_by incident); the skill has "
        "drifted from the code again."
    )


def test_skill_has_no_numeric_rerank_threshold():
    text = _skill_text()
    matches = _NUMERIC_THRESHOLD_RE.findall(text)
    assert not matches, (
        f"SKILL.md still asserts a numeric rerank_score threshold {matches!r}; "
        "PR #17 removed the absolute confidence threshold because reranker "
        "backends emit incompatible scoring scales."
    )
