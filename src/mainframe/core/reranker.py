"""Cross-encoder reranking — model configurable via config.json.

Two scoring backends:
- "cross-encoder" (default): sentence-transformers CrossEncoder (bge et al.).
- "qwen3-logit": native Qwen3-Reranker scoring — the model is a CAUSAL LM that
  judges yes/no; the score is P("yes") from the last-position logits. Loading it
  through CrossEncoder would silently mis-score it (no sequence-classification
  head), which is why the seq-cls conversions exist — but the native head +
  instruction follows the native model scoring interface.

v2 uses one shared reranker instruction across query categories.
"""

import io
import logging
import os
import sys
import warnings

import torch
from sentence_transformers import CrossEncoder

from mainframe_mcp.qwen import tokenize_with_suffix

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
# Keep the existing query bound. Documents use the available TOKEN budget:
# preserved code blocks and long paragraphs can exceed 3,000 characters while
# still fitting the context. tokenize_with_suffix reserves the scoring suffix.
# Changes to these limits require a retrieval evaluation.
_QWEN3_QUERY_CHAR_CLAMP = 1000
_QWEN3_MAX_LENGTH = 2048
_QWEN3_BATCH_SIZE = 8


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
    """Mainframe's measured <Instruct>/<Query>/<Doc> scoring body.

    Qwen's published template uses <Document>; changing this requires evaluation.
    """
    return (f"<Instruct>: {instruction}\n<Query>: {query[:_QWEN3_QUERY_CHAR_CLAMP]}\n"
            f"<Doc>: {doc}")


class Reranker:
    def __init__(self, config: dict):
        cfg = config["reranker"]
        self.model_name = cfg["model"]
        self.enabled = cfg.get("enabled", True)
        self.heading_inject = cfg.get("heading_inject", False)
        self.default_top_k = config["search"].get("rerank_top_k", 5)

        if not self.enabled:
            logger.info("Reranker disabled in config.")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._backend = detect_backend(self.model_name, cfg.get("backend"))
        self.instruction = cfg.get("instruction", _QWEN3_DEFAULT_INSTRUCTION)
        self.revision = cfg.get("revision")
        self.quantize = cfg.get("quantize", True)  # qwen3-logit only: NF4 on CUDA vs BF16

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
                self.model = CrossEncoder(self.model_name, device=device, trust_remote_code=True,
                                          revision=self.revision)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            os.environ.pop("TRANSFORMERS_VERBOSITY", None)
        logger.info("Reranker loaded.")

    def _load_qwen3(self, device: str):
        """Load the measured Qwen contract: NF4 on CUDA, BF16 otherwise."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        identity = {"trust_remote_code": True}
        if self.revision:
            identity["revision"] = self.revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", **identity)
        kwargs = {**identity, "dtype": torch.bfloat16, "attn_implementation": "sdpa",
                  "device_map": device}
        self._nf4 = self.quantize and device == "cuda"
        if self._nf4:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            kwargs["device_map"] = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if self._nf4 and not getattr(self.model, "is_loaded_in_4bit", False):
            raise RuntimeError("Qwen NF4 load did not produce a 4-bit model")
        self.model.eval()
        self._precision = "nf4-bf16" if self._nf4 else "bf16"
        self._yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self.tokenizer.convert_tokens_to_ids("no")

    def _score_qwen3(self, query: str, texts: list[str]) -> list[float]:
        """Score Qwen yes/no logits in fixed, saved batches of eight.

        A CUDA OOM mid-batch degrades to item-at-a-time scoring instead of
        killing the search."""
        instruction = self.instruction
        step = _QWEN3_BATCH_SIZE
        order = list(range(len(texts)))
        scores = [0.0] * len(texts)

        def _forward(idx: list[int]) -> None:
            batch = [
                _QWEN3_PREFIX + qwen3_pair_text(query, texts[j], instruction) + _QWEN3_SUFFIX
                for j in idx
            ]
            inputs = tokenize_with_suffix(self.tokenizer, batch, _QWEN3_SUFFIX,
                                          _QWEN3_MAX_LENGTH).to(self.model.device)
            with torch.inference_mode():
                # Only the final token judges yes/no. Keep the full attention
                # context, but avoid projecting every token into the vocabulary
                # or retaining a cache that this one-shot score never reuses.
                logits = self.model(
                    **inputs, logits_to_keep=1, use_cache=False
                ).logits[:, -1, :]
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
            "revision": getattr(self, "revision", None),
            "precision": getattr(self, "_precision", None),
            "batch_size": (_QWEN3_BATCH_SIZE if getattr(self, "_backend", None)
                           == "qwen3-logit" else None),
            "enabled": self.enabled,
        }

    def rerank(self, query: str, documents: list[str], top_k: int | None = None,
               headings: list[str] | None = None) -> list[tuple[int, float]]:
        if not self.enabled or not documents:
            return [(i, 0.0) for i in range(min(top_k or self.default_top_k, len(documents)))]
        k = top_k or self.default_top_k
        if self.heading_inject and headings:
            texts = [f"## {h}\n{d}" if h and h != "(no heading)" else d
                     for d, h in zip(documents, headings)]
        else:
            texts = documents
        if getattr(self, "_backend", "cross-encoder") == "qwen3-logit":
            scores = self._score_qwen3(query, texts)
        else:
            pairs = [(query, doc) for doc in texts]
            with torch.inference_mode():
                scores = self.model.predict(pairs, show_progress_bar=False)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return indexed[:k]
