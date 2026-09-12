"""GPU-accelerated embedding — model configurable via config.json.

Loads INT8 quantized by default (~7.6GB VRAM for Qwen3-8B).
Swap models by changing config.json or MAINFRAME_EMBEDDER_MODEL env var.
"""

import gc
import logging
from pathlib import Path

import torch

from mainframe.core.device import empty_cuda_cache, memory_pressure_guidance

logger = logging.getLogger(__name__)

# The bitsandbytes INT8 matmul kernel (ops.cu) launches a grid sized by batch*seq
# with a 16-bit dimension, so batch*seq must stay under this or it aborts the
# process. (This is a bitsandbytes kernel limit, not a general CUDA one.)
_BNB_INT8_GRID_MAX = 65535


class Embedder:
    encoding: dict | None = None

    def __init__(self, config: dict):
        from mainframe.core import encoding
        self.encoding = encoding.native_contract(config)
        if self.encoding:
            encoding.require_native_runtime()
        cfg = config["embedder"]
        self.model_name = cfg["model"]
        self.batch_size = cfg.get("batch_size", 8)
        self.query_prefix = cfg.get("query_prefix", "")
        # `or 2048` also handles an explicit null in config.json (which would
        # bypass the .get default and yield None).
        self.max_seq_length = cfg.get("max_seq_length") or 2048
        self._use_manual = False

        cache_dir = Path(config["paths"]["model_cache"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if cfg.get("quantize", True) and device == "cuda":
            self._load_quantized(self.model_name, cache_dir)
        else:
            self._load_fp(self.model_name, cache_dir, device)

        # Pin the sequence length and keep the INT8 kernel grid (batch * seq)
        # under the bitsandbytes limit. Without this, a single chunk near
        # max_seq_length at batch_size=8 makes 8*8192=65536 overflow, and
        # bitsandbytes aborts the whole process rather than raising. (The manual
        # path enforces the same bound via the tokenizer max_length below.)
        # Derive _safe_batch from the model's EFFECTIVE seq length: if pinning
        # silently fails, the model keeps its (larger) default and we must cap
        # against that, not the value we tried to set.
        effective_seq = self.max_seq_length
        if not self._use_manual:
            pin_error = None
            try:
                self.model.max_seq_length = self.max_seq_length
            except Exception as e:
                pin_error = str(e)
                logger.warning(f"Could not pin model.max_seq_length: {e}")
            effective_seq = getattr(self.model, "max_seq_length", None) or self.max_seq_length
            if self.encoding and (pin_error or effective_seq != self.max_seq_length):
                self.model = None
                gc.collect()
                empty_cuda_cache()
                raise ValueError('native model cannot enforce the recorded context limit')
        self._safe_batch = max(1, _BNB_INT8_GRID_MAX // max(1, effective_seq))
        if not self.encoding and self.batch_size > self._safe_batch:
            logger.info(
                f"Capping embed batch_size {self.batch_size} -> {self._safe_batch} "
                f"(max_seq_length={self.max_seq_length}, INT8 grid limit)"
            )
            self.batch_size = self._safe_batch

    def _load_quantized(self, model_name: str, cache_dir: Path):
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading {model_name} (INT8 via sentence-transformers)...")
            self.model = SentenceTransformer(
                model_name,
                model_kwargs={"quantization_config": quantization_config, "device_map": "auto"},
                tokenizer_kwargs={"padding_side": "left"},
                cache_folder=str(cache_dir),
                trust_remote_code=True,
            )
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded: dim={self.dimension}, VRAM={torch.cuda.memory_allocated()/1024**3:.1f}GB")
            return
        except Exception as e:
            logger.warning(f"sentence-transformers quantized load failed: {e}")
            guidance = memory_pressure_guidance(e)
            if guidance:
                logger.warning("%s", guidance)

        # Leave the handler first: its traceback may own a partial model.
        # Release both assigned models and constructor cycles before fallback.
        self.model = None
        gc.collect()
        empty_cuda_cache()
        self._load_manual_quantized(model_name, cache_dir, quantization_config)

    def _load_manual_quantized(self, model_name, cache_dir, quantization_config):
        from transformers import AutoTokenizer, AutoModel
        logger.info(f"Loading {model_name} (INT8 via transformers direct)...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", cache_dir=str(cache_dir), trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_name, quantization_config=quantization_config, device_map="auto",
            cache_dir=str(cache_dir), trust_remote_code=True)
        self._use_manual = True
        self.dimension = self._model.config.hidden_size
        logger.info(f"Loaded: dim={self.dimension}")

    def _load_fp(self, model_name, cache_dir, device):
        from sentence_transformers import SentenceTransformer
        kwargs = {'trust_remote_code': True}
        if self.encoding:
            from transformers import AutoConfig
            native_config = AutoConfig.from_pretrained(
                model_name, revision=self.encoding['revision'], cache_dir=str(cache_dir), trust_remote_code=False)
            if (getattr(native_config, 'auto_map', None) or {}).get('AutoModel'):
                raise ValueError('native custom model code requires a reviewed adapter; automatic architecture fallback is refused')
            kwargs = {'revision': self.encoding['revision'], 'trust_remote_code': False,
                      'model_kwargs': {'dtype': torch.bfloat16, 'attn_implementation': 'sdpa'}}
        logger.info("Loading %s (%s on %s)...", model_name, 'native BF16' if self.encoding else 'default precision', device)
        self.model = SentenceTransformer(
            model_name, cache_folder=str(cache_dir), device=device, **kwargs)
        self.dimension = self.model.get_sentence_embedding_dimension()

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def embed(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        if not texts:
            return []
        # Clip even an explicit caller batch_size to the grid-safe cap.
        bs = batch_size or self.batch_size
        if not self.encoding:
            bs = min(bs, self._safe_batch)
        try:
            return self._embed_at(texts, bs)
        except torch.cuda.OutOfMemoryError:
            # A transient VRAM spike (other GPU processes) shouldn't crash the
            # whole ingest — free the cache and retry one chunk at a time.
            torch.cuda.empty_cache()
            if bs > 1:
                logger.warning(f"CUDA OOM at batch_size={bs}; retrying at batch_size=1")
                return self._embed_at(texts, 1)
            raise

    def _embed_at(self, texts: list[str], bs: int) -> list[list[float]]:
        if self._use_manual:
            return self._embed_manual(texts, bs)
        with torch.inference_mode():
            if self.encoding:
                return self.model.encode_document(
                    texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True,
                    convert_to_tensor=True).float().cpu().tolist()
            embeddings = self.model.encode(texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def _embed_manual(self, texts, batch_size):
        import torch.nn.functional as F
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_dict = self._tokenizer(batch, padding=True, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
            batch_dict = {k: v.to(self._model.device) for k, v in batch_dict.items()}
            with torch.no_grad():
                outputs = self._model(**batch_dict)
            embs = self._last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
            all_embs.extend(embs.cpu().tolist())
        return all_embs

    @staticmethod
    def _classify_query(query: str) -> str:
        """Classify query for category-specific Instruct prefix."""
        q = query.lower()
        if any(w in q for w in ["code", "function", "class", "import", "api", "eslint",
                                 "typescript", "python", "swift", "graphql", "sql"]):
            return "code"
        if any(w in q for w in ["research", "study", "analysis", "comparison", "benchmark"]):
            return "research"
        if any(w in q for w in ["architecture", "design", "pattern", "structure", "schema"]):
            return "architecture"
        if any(w in q for w in ["legal", "privacy", "compliance", "license", "coppa"]):
            return "legal"
        if any(w in q for w in ["config", "setup", "install", "deploy", "docker"]):
            return "config"
        return "general"

    CATEGORY_PREFIXES = {
        "code": "Instruct: Find code documentation or API reference for this programming question\nQuery: ",
        "research": "Instruct: Find research findings or analysis about this topic\nQuery: ",
        "architecture": "Instruct: Find architecture documentation or design decisions about this system\nQuery: ",
        "legal": "Instruct: Find legal, privacy, or compliance documentation\nQuery: ",
        "config": "Instruct: Find configuration, setup, or deployment documentation\nQuery: ",
        "general": "",  # filled from config at runtime
    }

    def embed_query(self, query: str) -> list[float]:
        if self.encoding:
            prompt = self.encoding['query_prompt']
            with torch.inference_mode():
                return self.model.encode_query(
                    [query], show_progress_bar=False, normalize_embeddings=True, convert_to_tensor=True,
                    **({'prompt_name': prompt} if prompt else {})).float().cpu().tolist()[0]
        category = self._classify_query(query)
        prefix = self.CATEGORY_PREFIXES.get(category) or self.query_prefix
        if category == "general":
            prefix = self.query_prefix
        text = f"{prefix}{query}" if prefix else query
        return self.embed([text])[0]

    @property
    def info(self) -> dict:
        info = {"model": self.model_name, "dimension": self.dimension, "device": self.device}
        if self.encoding:
            info['encoding'] = dict(self.encoding)
        return info
