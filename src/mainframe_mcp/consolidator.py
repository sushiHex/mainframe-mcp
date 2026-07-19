"""Knowledge consolidation — model configurable via config.json."""

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Qwen2.5-3B-Instruct context window; prompt is truncated to leave room for the
# generated continuation so prompt+max_new_tokens never overflows it.
CONTEXT_WINDOW = 32768
# Cluster summaries take a deliberately tighter prompt budget (they summarize a
# few chunks, not full reports) and generate only ~256 tokens.
CLUSTER_MAX_PROMPT_TOKENS = 4000

# Shared so the two prompts can't drift (dedup/length/attribution/don't-invent).
_SHARED_CONSOLIDATION_RULES = """- Preserve all unique facts, numbers, and findings.
- When sources agree, state the fact once with "(confirmed by N sources)".
- Remove redundant phrasing — say each fact once, concisely.
- Use markdown with clear headings. Keep the output under 500 lines.
- Do not invent information not present in the sources."""

CONSOLIDATION_PROMPT = """You are a knowledge consolidator. Merge these research reports into ONE canonical reference document.

Rules:
{rules}
- When reports contradict, state both positions and note the disagreement.

Source reports:

{reports}

Write the consolidated reference document:"""

# Conflict-aware variant (mem0 "update phase"): fold new reports INTO an existing
# canonical note, letting newer facts supersede stale ones instead of blindly
# re-merging. NEW reports come FIRST (highest priority) — see consolidate()'s
# budgeting, which truncates the older `existing` block, never the new reports.
UPDATE_CONSOLIDATION_PROMPT = """You are a knowledge consolidator UPDATING a canonical reference document with newer findings. The NEW reports take priority.

Rules:
{rules}
- Start from the current canonical memory and fold in the new reports.
- When a new report corrects or contradicts the current memory, KEEP THE NEWER FACT and drop the superseded one (newer wins); briefly note the change if material.

## New session reports
{reports}

## Current canonical memory (may contain stale facts)
{existing}

Write the updated canonical reference document:"""


def skeleton_tokens(tokenizer) -> tuple[int, int]:
    """(update, add) template overhead in tokens — measured, not guessed, so
    the budget math tracks prompt edits automatically."""
    upd = len(tokenizer.encode(UPDATE_CONSOLIDATION_PROMPT.format(
        rules=_SHARED_CONSOLIDATION_RULES, reports="", existing="")))
    add = len(tokenizer.encode(CONSOLIDATION_PROMPT.format(
        rules=_SHARED_CONSOLIDATION_RULES, reports="")))
    return upd, add


def build_prompt(reports: list[dict], existing: str, tokenizer,
                 max_new_tokens: int, update_skeleton_tokens: int,
                 add_skeleton_tokens: int) -> str:
    """Assemble the consolidation prompt within the context-window budget.

    PURE (tokenizer injected, no model) — this is the load-bearing memory-
    correctness math: when space runs out, the EXISTING note is truncated and
    the NEW reports survive intact (newer facts must win); only when the new
    reports alone exceed the update budget does it fall back to a fresh merge.
    """
    reports_text = "\n\n".join(
        f"{'=' * 40}\n## Report: {r['title']}\nSource: {r['path']}\n{'=' * 40}\n\n{r['text']}"
        for r in reports
    )
    # Leave room for the generated continuation so prompt + max_new_tokens
    # stays within the context window (was a fixed 30000 that overflowed).
    max_prompt = max(1024, CONTEXT_WINDOW - max_new_tokens - 512)
    ebody = existing.strip()

    if ebody:
        # UPDATE path: NEW reports have priority. Budget reports first, give
        # `ebody` only the remainder, and truncate `ebody` (not a blind
        # tail-cut of the whole prompt — that would eject the new reports AND
        # the generation instruction once the note grows large).
        rtok = tokenizer.encode(reports_text)
        if len(rtok) > max_prompt - update_skeleton_tokens:
            # No room for the old note -> fall through to the plain merge.
            # Size reports against the (smaller) ADD skeleton so the full
            # window is used and the ADD instruction still fits.
            logger.warning(f"Reports alone ({len(rtok)} tok) exceed update budget — merging fresh")
            reports_text = tokenizer.decode(rtok[: max_prompt - add_skeleton_tokens])
            ebody = ""
        else:
            etok = tokenizer.encode(ebody)
            ebudget = max_prompt - update_skeleton_tokens - len(rtok)
            if len(etok) > ebudget:
                logger.warning(f"Existing note ({len(etok)} tok) truncated to {ebudget} to keep new reports intact")
                ebody = tokenizer.decode(etok[:ebudget]) if ebudget > 0 else ""

    if ebody:
        return UPDATE_CONSOLIDATION_PROMPT.format(
            rules=_SHARED_CONSOLIDATION_RULES, existing=ebody, reports=reports_text)
    prompt = CONSOLIDATION_PROMPT.format(rules=_SHARED_CONSOLIDATION_RULES, reports=reports_text)
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_prompt:
        logger.warning(f"Consolidation input is {len(tokens)} tokens — truncating to {max_prompt}")
        prompt = tokenizer.decode(tokens[:max_prompt])
    return prompt


class Consolidator:
    def __init__(self, config: dict):
        cfg = config["consolidator"]
        self.model_name = cfg["model"]
        self.enabled = cfg.get("enabled", True)
        self.max_new_tokens = cfg.get("max_new_tokens", 4096)
        self.quantize = cfg.get("quantize", True)

        if not self.enabled:
            logger.info("Consolidator disabled in config.")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        cache_dir = config["paths"]["model_cache"]

        logger.info(f"Loading consolidation model {self.model_name} on {device}...")

        load_kwargs = {"device_map": device, "cache_dir": cache_dir, "trust_remote_code": True}
        if self.quantize and device == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, **load_kwargs)
        logger.info("Consolidation model loaded.")

        # Measure each template's fixed overhead once so the budget math tracks
        # prompt edits automatically (vs a hardcoded guess that silently drifts).
        self._update_skeleton_tokens, self._add_skeleton_tokens = skeleton_tokens(self.tokenizer)

    def summarize_cluster(self, texts: list[str], headings: list[str], doc_path: str = "") -> str:
        """Generate a retrievable summary from a cluster of related chunks.

        Unlike consolidate() which merges full reports, this creates a concise
        summary optimized for vector retrieval — it should capture the key
        topics, terms, and facts so semantic search can find it.
        """
        if not self.enabled:
            return ""

        # Build context from chunk texts
        chunks_text = ""
        for i, (text, heading) in enumerate(zip(texts, headings)):
            h = f" ({heading})" if heading and heading != "(no heading)" else ""
            chunks_text += f"\n--- Chunk {i+1}{h} ---\n{text}\n"

        doc_name = Path(doc_path).name

        prompt = f"""Summarize these related text chunks from "{doc_name}" into ONE paragraph.

Rules:
- Capture ALL key topics, terms, names, and facts mentioned.
- Include specific technical terms, project names, and concepts — these help search.
- Be concise but information-dense. 2-4 sentences.
- Do not add information not in the source chunks.

Source chunks:
{chunks_text}

Summary:"""

        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > CLUSTER_MAX_PROMPT_TOKENS:
            prompt = self.tokenizer.decode(tokens[:CLUSTER_MAX_PROMPT_TOKENS])

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, temperature=0.3, do_sample=True, top_p=0.9)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def consolidate(self, reports: list[dict], topic: str = "", existing: str = "") -> str:
        if not self.enabled:
            return "ERROR: Consolidator disabled in config."

        prompt = build_prompt(reports, existing, self.tokenizer, self.max_new_tokens,
                              self._update_skeleton_tokens, self._add_skeleton_tokens)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, temperature=0.3, do_sample=True, top_p=0.9)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
