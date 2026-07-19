"""Contradiction detection — model configurable via config.json."""

import logging
import re

import numpy as np
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

CONTRADICTION_LABEL = "contradiction"

_LIST_MARKER = re.compile(r"^\s*([-*+•]|\d+[.)])\s+")


def fact_units(text: str, cap: int = 120) -> list[str]:
    """Split a memory note / reports into atomic fact lines for NLI pairing —
    non-heading, non-trivial lines with their list marker stripped. If over
    `cap`, sample evenly across the whole text (head→tail, via linspace) so a
    long note's most-recent sections are still represented (not just the head)."""
    units = []
    for line in text.splitlines():
        s = _LIST_MARKER.sub("", line).strip()
        if not s or s.startswith("#") or len(s) < 20:
            continue
        units.append(s)
    if len(units) <= cap:
        return units
    idx = np.linspace(0, len(units) - 1, cap, dtype=int)
    return [units[i] for i in idx]


def find_contradictions_nn(detector, embedder, old_text: str, new_text: str,
                           gate: float = 0.4) -> list[dict]:
    """For each fact in `new_text`, NLI-check it ONLY against its nearest fact in
    `old_text` (cosine via the resident embedder — one NLI call per new fact, not
    O(n·m)). `gate` skips unrelated pairs (NaN-safe). Returns the flagged
    {prior, newer, confidence} pairs to SURFACE — never to mutate anything."""
    old_facts, new_facts = fact_units(old_text), fact_units(new_text)
    if not old_facts or not new_facts:
        return []
    # One batched embed pass; embedder returns unit vectors, so dot == cosine.
    embs = np.asarray(embedder.embed(old_facts + new_facts), dtype="float32")
    old, new = embs[: len(old_facts)], embs[len(old_facts):]
    sims = new @ old.T
    best = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(new_facts)), best]
    found = []
    for ni, nf in enumerate(new_facts):
        if not (best_sim[ni] >= gate):  # NaN-safe: skip unrelated/degenerate
            continue
        oi = int(best[ni])
        res = detector.check_contradiction(old_facts[oi], nf)
        if res.get("is_contradiction"):
            found.append({"prior": old_facts[oi], "newer": nf,
                          "confidence": round(res.get("confidence", 0.0), 3)})
    return found


class ContradictionDetector:
    def __init__(self, config: dict):
        cfg = config["nli"]
        self.model_name = cfg["model"]
        self.enabled = cfg.get("enabled", True)
        self.threshold = cfg.get("threshold", 0.7)

        if not self.enabled:
            logger.info("NLI model disabled in config.")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading NLI model {self.model_name} on {device}...")
        # Sequence-pair classification (premise, hypothesis) — NOT zero-shot.
        # zero-shot ignored text_b entirely, so it could never compare new-vs-
        # existing; text-classification with a text_pair feeds BOTH sequences.
        self.pipe = pipeline(
            "text-classification",
            model=self.model_name,
            device=0 if device == "cuda" else -1,
            top_k=None,  # return scores for all labels
        )
        logger.info("NLI model loaded.")

    def check_contradiction(self, text_a: str, text_b: str) -> dict:
        """Does the NEW claim (text_b) contradict the EXISTING one (text_a)?"""
        if not self.enabled:
            return {"is_contradiction": False, "confidence": 0.0, "label": "disabled"}

        # premise = existing (text_a), hypothesis = new (text_b). Batch input ->
        # list-of-lists output, so [0] always unwraps; `or` guards a None return.
        out = self.pipe([{"text": text_a, "text_pair": text_b}], top_k=None)
        rows = (out or [[]])[0] or []
        label_scores = {r["label"].lower(): float(r["score"]) for r in rows}
        contradiction_score = label_scores.get(CONTRADICTION_LABEL, 0.0)
        top_label = max(label_scores, key=label_scores.get) if label_scores else "unknown"

        return {
            "is_contradiction": contradiction_score >= self.threshold,
            "confidence": contradiction_score,
            "label": top_label,
            "all_scores": label_scores,
        }
