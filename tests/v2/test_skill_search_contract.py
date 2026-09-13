"""Guards skills/mainframe-retrieval/SKILL.md against drifting from the real
`search` contract (issue #22).

The skill previously asserted a "high"/"low"/"none" `confidence` vocabulary
with numeric rerank_score thresholds (>0.85 strong, <0.3 likely noise). PR #17
deliberately removed that threshold: reranker backends emit model-native
ordering signals — some bounded probabilities, some unbounded logits — so an
absolute cutoff cannot mean the same thing across backends. The code's actual
vocabulary lives in `mainframe.core.search.CONFIDENCE_VALUES`; this test reads
the skill file and checks its claims against that one source of truth instead
of letting docs and code drift apart silently again.
"""
import re
from pathlib import Path

from mainframe.core.search import CONFIDENCE_VALUES

SKILL = Path(__file__).resolve().parents[2] / "skills" / "mainframe-retrieval" / "SKILL.md"

# The JSON-shaped contract declaration, e.g. `"confidence": "unavailable"|"none"`.
_CONTRACT_RE = re.compile(r'"confidence"\s*:\s*((?:"[a-z]+"\s*\|?\s*)+)')
# Standalone prose/bullet mentions, e.g. `` `confidence: "none"` ``.
_PROSE_RE = re.compile(r'confidence:?\s*`?"([a-z]+)"', re.IGNORECASE)
# A numeric threshold compared against a rerank-style score, e.g. `>0.85` or `<0.3`.
_NUMERIC_THRESHOLD_RE = re.compile(r'[<>]\s*0\.\d+')


def _skill_text() -> str:
    return SKILL.read_text(encoding="utf-8")


def _skill_confidence_values(text: str) -> set[str]:
    flat = " ".join(text.split())  # tolerate the contract line wrapping
    values: set[str] = set()
    contract = _CONTRACT_RE.search(flat)
    if contract:
        values |= set(re.findall(r'"([a-z]+)"', contract.group(1)))
    values |= set(m.lower() for m in _PROSE_RE.findall(flat))
    return values


def test_skill_confidence_vocabulary_matches_the_code():
    text = _skill_text()
    skill_values = _skill_confidence_values(text)
    assert skill_values, "expected SKILL.md to document at least one confidence value"
    assert skill_values == set(CONFIDENCE_VALUES), (
        f"SKILL.md documents confidence values {skill_values!r} but "
        f"mainframe.core.search.CONFIDENCE_VALUES is {set(CONFIDENCE_VALUES)!r} "
        "-- the skill has drifted from the code (see issue #22)."
    )


def test_skill_has_no_numeric_rerank_threshold():
    text = _skill_text()
    matches = _NUMERIC_THRESHOLD_RE.findall(text)
    assert not matches, (
        f"SKILL.md still asserts a numeric rerank_score threshold {matches!r}; "
        "PR #17 removed the absolute confidence threshold because reranker "
        "backends emit incompatible scoring scales."
    )
