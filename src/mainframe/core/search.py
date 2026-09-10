"""The read path: embed → hybrid candidates → rerank → shape.
LLM-free; the only models touched are the embedder and the reranker."""

from pathlib import Path
import re

from mainframe.core.hashing import file_hash

MAX_RESULTS = 10
LOW_CONFIDENCE = 0.3
SNIPPET_CHARS = 200
TEXT_CHARS = 500

_EMPTY_GUIDANCE = ("No matches. Rephrase with SPECIFIC technical terms (function/class/proper "
                   "names, exact config keys, error strings) — natural-language questions rank "
                   "poorly. Or set include_sessions=true to also search raw session captures.")
_LOW_GUIDANCE = ("Best rerank_score < 0.3 (likely noise). Rephrase with more specific technical "
                 "terms. Trust rerank_score (higher = better), not score.")


def line_span(text: str, char_start: int, char_end: int) -> tuple[int, int]:
    """1-based inclusive line numbers for [char_start, char_end). A span that
    ends right after a newline belongs to the preceding line."""
    start = text.count("\n", 0, char_start) + 1
    end_pos = max(char_start, char_end - 1)
    return start, text.count("\n", 0, end_pos) + 1


def source_span(text: str, passage: str, start: int, end: int) -> tuple[int, int] | None:
    """Verify stored offsets, or recover one unambiguous source occurrence.

    Tiny-section merging adds newlines to the indexed passage. Permit only
    removal of those extra newlines when locating it; indentation and every
    other character must match. This also repairs old rows without re-embedding.
    """
    passage = passage.strip()
    if not passage:
        return None
    if 0 <= start < end <= len(text) and text[start:end].strip() == passage:
        return start, end
    parts = re.split(r"(\n+)", passage)
    pattern = re.compile("".join(r"\n{1,%d}" % len(part) if part.startswith("\n")
                                 else re.escape(part) for part in parts))
    match = pattern.search(text)
    # Search from the next character, not the end: overlapping occurrences
    # are ambiguous too. Never use a nearby occurrence as a guessed citation.
    if match is None or pattern.search(text, match.start() + 1) is not None:
        return None
    return match.span()


class SearchService:
    """Holds the model REGISTRY, never the models: every GPU call goes through
    `Models.invoke`, which owns the lock and reloads a dead one underneath us.
    Caching the objects here is what forced the caller to rebuild this service
    whenever the registry's identities changed."""

    def __init__(self, models, store, config: dict):
        self.models = models
        self.store = store
        self.config = config

    def search(self, query: str, limit: int = 3, include_sessions: bool = False,
               response_format: str = "concise") -> dict:
        if response_format not in ("concise", "detailed", "full"):
            raise ValueError(f"unknown response_format: {response_format!r} "
                             "(expected concise, detailed, or full)")
        limit = max(1, min(int(limit), MAX_RESULTS))
        pool = max(int(self.config["search"].get("candidate_pool", 20)), limit * 2)
        q = self.models.invoke("embedder", lambda e: e.embed_query(query))
        raw = self.store.search(q, top_k=pool, include_captures=include_sessions, query_text=query)
        if not raw:
            return {"results": [], "confidence": "none", "guidance": _EMPTY_GUIDANCE}
        texts = [r["text"] for r in raw]
        headings = [r.get("heading", "") for r in raw]
        ranked = self.models.invoke("reranker", lambda r: r.rerank(query, texts, top_k=limit, headings=headings))
        results = [self._shape(raw[i], float(s), response_format) for i, s in ranked]
        self._add_line_spans(raw, ranked, results)
        best = max((r["rerank_score"] for r in results), default=0.0)
        payload = {"results": results, "confidence": "high" if best >= LOW_CONFIDENCE else "low"}
        if best < LOW_CONFIDENCE:
            payload["guidance"] = _LOW_GUIDANCE
        return payload

    @staticmethod
    def _shape(row: dict, score: float, fmt: str) -> dict:
        text = row["text"]
        out = {"file": row["doc_path"], "heading": row.get("heading", ""),
               "rerank_score": round(score, 4),
               "snippet": " ".join(text.split())[:SNIPPET_CHARS]}
        if fmt in ("detailed", "full"):
            # "full" is "detailed" with the untruncated chunk text — callers that
            # score against the FULL text (e.g. eval's --daemon mode) must not be
            # penalized by the 500-char preview cap other detailed consumers want.
            shown_text = text if fmt == "full" else text[:TEXT_CHARS]
            out.update({"text": shown_text, "truncated": False if fmt == "full" else len(text) > TEXT_CHARS,
                        "char_start": int(row["char_start"]), "char_end": int(row["char_end"]),
                        "score": float(row.get("boosted_score", row.get("_distance", 0.0))),
                        "source_type": row.get("source_type", ""), "project": row.get("project", "")})
        return out

    @staticmethod
    def _add_line_spans(raw: list, ranked: list, results: list) -> None:
        """1-based line_start/line_end when the on-disk file still matches the
        indexed revision and its passage can be verified. Recovered character
        offsets replace stale ones in detailed/full results; no index writes."""
        cache = {}
        for (i, _), out in zip(ranked, results):
            row = raw[i]
            f = row["doc_path"]
            if f not in cache:
                try:
                    cache[f] = Path(f).read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    cache[f] = None
            text = cache[f]
            if text is None or file_hash(text) != row.get("file_hash"):
                continue
            span = source_span(text, row["text"], int(row["char_start"]), int(row["char_end"]))
            if span is None:
                continue
            if "char_start" in out:
                out["char_start"], out["char_end"] = span
            out["line_start"], out["line_end"] = line_span(text, *span)
