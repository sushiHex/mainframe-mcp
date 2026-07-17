"""API-based contextual retrieval at ingest (Anthropic's method) — OPT-IN.

The ONE deliberate exception to the 100%-local rule: at INGEST time only, each
chunk is situated within its whole document by claude-haiku-4-5 and the 1-2
sentence context is prepended to the chunk text before embedding + FTS indexing
(Anthropic's published numbers: contextual embeddings alone cut top-20 retrieval
failures ~35%; + contextual BM25 ~49%; + reranking ~67% — this stack already has
the BM25 + rerank halves). SERVING stays fully local; nothing leaves the box at
query time.

OFF by default (`contextual.enabled`) because enabling it sends document content
to the Anthropic API. Every failure — no credentials, rate limits, network,
refusal — degrades to plain chunks; contextualization must never block ingest.

The document is passed as a cache_control'd system block so the per-chunk calls
reuse the cached prefix (~90% input-cost cut on repeat calls; docs under the
model's ~4096-token minimum cacheable prefix silently don't cache, which is fine).
"""

import logging

logger = logging.getLogger(__name__)

_PROMPT = (
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Please give a short succinct context (1-2 sentences) to situate this chunk "
    "within the overall document for the purposes of improving search retrieval "
    "of the chunk. Answer only with the succinct context and nothing else."
)
# Haiku 4.5 has a 200K-token window; clamp the doc prefix for safety + cost.
_MAX_DOC_CHARS = 150_000


class Contextualizer:
    def __init__(self, config: dict):
        cfg = config.get("contextual", {})
        self.enabled = cfg.get("enabled", False)
        self.model = cfg.get("model", "claude-haiku-4-5")
        self._client = None
        if not self.enabled:
            return
        try:
            import anthropic
            # Zero-arg client: resolves ANTHROPIC_API_KEY / auth token / an
            # `ant auth login` profile from the environment.
            self._client = anthropic.Anthropic()
        except Exception as e:
            logger.warning(f"Contextualizer disabled (no anthropic client): {e}")
            self.enabled = False

    def contextualize(self, doc_text: str, chunk_texts: list[str]) -> list[str | None]:
        """One situating context (or None) per chunk. NEVER raises — any
        failure yields None for that chunk and ingest proceeds plain."""
        if not self.enabled or self._client is None:
            return [None] * len(chunk_texts)
        system = [{
            "type": "text",
            "text": f"<document>\n{doc_text[:_MAX_DOC_CHARS]}\n</document>",
            "cache_control": {"type": "ephemeral"},  # reused across the chunk loop
        }]
        out: list[str | None] = []
        for chunk in chunk_texts:
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=150,
                    system=system,
                    messages=[{"role": "user", "content": _PROMPT.format(chunk=chunk)}],
                )
                text = next((b.text for b in resp.content if b.type == "text"), "").strip()
                out.append(text or None)
            except Exception as e:  # rate-limit exhaustion, network, refusal, ...
                logger.warning(f"contextualize failed for a chunk (plain fallback): {e}")
                out.append(None)
        return out
