"""Cross-encoder reranking — model configurable via config.json.

Two scoring backends:
- "cross-encoder" (default): sentence-transformers CrossEncoder (bge et al.).
- "qwen3-logit": native Qwen3-Reranker scoring — the model is a CAUSAL LM that
  judges yes/no; the score is P("yes") from the last-position logits. Loading it
  through CrossEncoder would silently mis-score it (no sequence-classification
  head), which is why the seq-cls conversions exist — but the native head +
  instruction is the higher-ceiling path (eval 2026-07-01b).
"""

import io
import logging
import os
import sys
import warnings

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Native Qwen3-Reranker prompt scaffold (from the model card — do not reword;
# the model was trained on these exact tokens).
_QWEN3_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on '
    'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN3_DEFAULT_INSTRUCTION = (
    "Given a technical documentation search query, retrieve the passage that best answers it"
)
# Per-category instructions, keyed by the embedder's query classifier — the
# SAME classification that already routes embedding prefixes was previously
# discarded before it reached the reranker. Qwen's model card credits
# instruction customization with 1-5%. 'general' is intentionally absent ->
# instance default. Overridable via config reranker.instructions.
CATEGORY_INSTRUCTIONS = {
    "code": ("Given a programming question, retrieve the code documentation or "
             "API reference passage that best answers it"),
    "research": ("Given a research question, retrieve the research finding or "
                 "analysis passage that best answers it"),
    "architecture": ("Given a system design question, retrieve the architecture "
                     "documentation or design-decision passage that best answers it"),
    "legal": ("Given a legal or compliance question, retrieve the legal, privacy, "
              "or compliance passage that best answers it"),
    "config": ("Given a setup or deployment question, retrieve the configuration "
               "or installation passage that best answers it"),
}
# Clamp each doc before assembly so tokenizer truncation can never eat the
# suffix (the assistant tag the score is read at). 256-tok chunks are ~1.5K
# chars; only pathological chunks hit this. DO NOT trim these for speed:
# maxlen 1024 + clamp 2400 measured 0.3707 vs 0.4187 (eval 2026-07-01f).
_QWEN3_DOC_CHAR_CLAMP = 3000
# The query gets the same treatment — unbounded, it could push the assembled
# prompt past max_length and truncate away the assistant suffix.
_QWEN3_QUERY_CHAR_CLAMP = 1000
_QWEN3_MAX_LENGTH = 2048


def detect_backend(model_name: str, cfg_backend: str | None = None) -> str:
    """Pick the scoring backend. Config `reranker.backend` wins; otherwise
    native Qwen3-Reranker checkpoints (NOT the *-seq-cls conversions, which are
    real CrossEncoders) use the yes/no-logit path."""
    if cfg_backend:
        return cfg_backend
    name = model_name.lower()
    if "qwen3-reranker" in name and "seq-cls" not in name:
        return "qwen3-logit"
    return "cross-encoder"


def qwen3_pair_text(query: str, doc: str, instruction: str) -> str:
    """The <Instruct>/<Query>/<Doc> body the Qwen3 reranker scores (model-card
    format). Pure — unit-tested without the model."""
    return (f"<Instruct>: {instruction}\n<Query>: {query[:_QWEN3_QUERY_CHAR_CLAMP]}\n"
            f"<Doc>: {doc[:_QWEN3_DOC_CHAR_CLAMP]}")


class Reranker:
    def __init__(self, config: dict):
        cfg = config["reranker"]
        self.model_name = cfg["model"]
        self.enabled = cfg.get("enabled", True)
        self.heading_inject = cfg.get("heading_inject", False)
        self.default_top_k = config["search"].get("rerank_top_k", 5)
        # Set unconditionally (before the disabled early-return) so
        # resolve_instruction is a plain attribute read — a getattr default
        # here would fail silently forever on an attribute-name typo. The gate
        # lives HERE (not in callers) so every caller passes its classification
        # unconditionally. OFF by default: measured 0.3760 vs 0.4187.
        self.category_enabled = cfg.get("category_instructions", False)

        if not self.enabled:
            logger.info("Reranker disabled in config.")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._backend = detect_backend(self.model_name, cfg.get("backend"))
        self.instruction = cfg.get("instruction", _QWEN3_DEFAULT_INSTRUCTION)
        # Per-category instructions (qwen3-logit only); config entries override
        # the defaults key-by-key.
        self.category_instructions = {**CATEGORY_INSTRUCTIONS, **cfg.get("instructions", {})}
        self.quantize = cfg.get("quantize", True)  # qwen3-logit only: INT8 vs fp16

        logger.info(f"Loading reranker {self.model_name} ({self._backend}) on {device}...")
        old_stdout, old_stderr = sys.stdout, sys.stderr
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        warnings.filterwarnings("ignore")
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            if self._backend == "qwen3-logit":
                self._load_qwen3(device)
            else:
                # mxbai-rerank is missing score.weight — PyTorch randomly inits it.
                # Fix seed so the init is reproducible across process loads.
                torch.manual_seed(42)
                self.model = CrossEncoder(self.model_name, device=device, trust_remote_code=True)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            os.environ.pop("TRANSFORMERS_VERBOSITY", None)
        logger.info("Reranker loaded.")

    def _load_qwen3(self, device: str):
        """Native Qwen3-Reranker: a causal LM scored via yes/no logits.

        `reranker.quantize` (default true) picks INT8 (~4.5GB for the 4B —
        VRAM-lean but bitsandbytes decomposition costs 2-3x LATENCY on Ampere)
        vs fp16 (~8GB, fastest). fp16 also lifts the INT8 kernel grid limit,
        so the whole candidate pool scores in one forward pass."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", trust_remote_code=True)
        kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16, "device_map": device}
        self._quantized = self.quantize and device == "cuda"
        if self._quantized:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self.tokenizer.convert_tokens_to_ids("no")

    def resolve_instruction(self, category: str | None) -> str:
        """Per-category instruction for the qwen3-logit backend; the config
        gate (category_instructions, default off) and unknown/absent categories
        both fall back to the instance default."""
        if category and self.category_enabled:
            return self.category_instructions.get(category, self.instruction)
        return self.instruction

    def _score_qwen3(self, query: str, texts: list[str],
                     instruction: str | None = None) -> list[float]:
        """Score = P('yes') at the last position, softmaxed over {yes, no}.

        INT8: batch 8 (bitsandbytes grid: batch*seq < 65535) in INPUT ORDER —
        LLM.int8's per-batch outlier decomposition makes scores batch-
        composition-dependent, and re-batching measurably moved the eval
        (0.4187 -> 0.3707, 2026-07-01f); the validated arrangement stays
        bit-stable. fp16 is batch-invariant, so it gets length-sorted batches
        (similar-length rows share padding) purely for speed.

        A CUDA OOM mid-batch degrades to item-at-a-time scoring (parity with
        the embedder's fallback) instead of killing the search."""
        instruction = instruction or self.instruction
        quantized = getattr(self, "_quantized", True)
        step = 8 if quantized else 32
        order = (list(range(len(texts))) if quantized
                 else sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True))
        scores = [0.0] * len(texts)

        def _forward(idx: list[int]) -> None:
            batch = [
                _QWEN3_PREFIX + qwen3_pair_text(query, texts[j], instruction) + _QWEN3_SUFFIX
                for j in idx
            ]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=_QWEN3_MAX_LENGTH,
                                    return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                logits = self.model(**inputs).logits[:, -1, :]
            yes_no = torch.stack([logits[:, self._yes_id], logits[:, self._no_id]], dim=1)
            probs = torch.softmax(yes_no.float(), dim=1)[:, 0].cpu().tolist()
            for j, p in zip(idx, probs):
                scores[j] = p

        for i in range(0, len(order), step):
            idx = order[i:i + step]
            try:
                _forward(idx)
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"Reranker OOM at batch={len(idx)}; retrying item-at-a-time")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for j in idx:
                    _forward([j])
        return scores

    @property
    def info(self) -> dict:
        """What is actually loaded — backends span 0.33-0.42 composite quality
        and 2-8s/query, so status() must be able to say which one is live."""
        return {
            "model": self.model_name,
            "backend": getattr(self, "_backend", "disabled" if not self.enabled else "?"),
            "quantized": getattr(self, "_quantized", None),
            "enabled": self.enabled,
        }

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        headings: list[str] | None = None,
        category: str | None = None,
    ) -> list[tuple[int, float]]:
        if not self.enabled or not documents:
            return [(i, 0.0) for i in range(min(top_k or self.default_top_k, len(documents)))]

        k = top_k or self.default_top_k

        # Heading injection: prepend section heading to give reranker context
        if self.heading_inject and headings:
            texts = []
            for doc, heading in zip(documents, headings):
                if heading and heading != "(no heading)":
                    texts.append(f"## {heading}\n{doc}")
                else:
                    texts.append(doc)
        else:
            texts = documents

        if getattr(self, "_backend", "cross-encoder") == "qwen3-logit":
            scores = self._score_qwen3(query, texts,
                                       instruction=self.resolve_instruction(category))
        else:
            pairs = [(query, doc) for doc in texts]
            with torch.inference_mode():
                scores = self.model.predict(pairs, show_progress_bar=False)
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return indexed_scores[:k]
