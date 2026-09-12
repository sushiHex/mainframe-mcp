"""Exercise both production scoring paths with a local, deterministic tokenizer."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from mainframe_mcp.qwen import tokenize_with_suffix


class RecordingByteTokenizer:
    """A byte-level tokenizer double with a deliberately wide vocab token."""

    def __init__(self):
        self.calls = []
        self.get_vocab_calls = 0

    def get_vocab(self):
        self.get_vocab_calls += 1
        return {"x" * 16: 0, "S": 1}

    def encode(self, text, add_special_tokens=False):
        return [1] if text == "S" else [0] * len(text)

    def __call__(self, prompts, **kwargs):
        self.calls.append((list(prompts), kwargs))
        limit = kwargs.get("max_length", 1_000_000)
        return {"input_ids": [[0] * min(len(prompt), limit) for prompt in prompts]}

    def pad(self, rows, **kwargs):
        width = max(map(len, rows["input_ids"]))
        ids = [([0] * (width - len(row))) + row for row in rows["input_ids"]]
        mask = [([0] * (width - len(row))) + ([1] * len(row)) for row in rows["input_ids"]]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def test_huge_prompt_is_bounded_before_tokenization_and_keeps_suffix():
    tokenizer = RecordingByteTokenizer()
    out = tokenize_with_suffix(tokenizer, ["x" * 100_000 + "S"], "S", max_length=8)

    prompts, kwargs = tokenizer.calls[0]
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == 7
    assert len(prompts[0]) <= 7 * 16
    assert out["input_ids"].tolist() == [[0] * 7 + [1]]


def test_vocab_width_is_cached_on_the_resident_tokenizer():
    tokenizer = RecordingByteTokenizer()

    tokenize_with_suffix(tokenizer, ["xS"], "S", max_length=8)
    tokenize_with_suffix(tokenizer, ["xS"], "S", max_length=8)

    assert tokenizer.get_vocab_calls == 1


@pytest.fixture(params=["mainframe_mcp.reranker", "mainframe.core.reranker"])
def scorer(request):
    module = importlib.import_module(request.param)
    vocabulary = ["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>",
                  "ANSWER", "routine", "context", "query", "document"]
    backend = Tokenizer(models.WordLevel(dict(zip(vocabulary, range(len(vocabulary)))),
                                         unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"], padding_side="left",
        model_max_length=100_000, model_input_names=["input_ids", "attention_mask"])
    r = module.Reranker.__new__(module.Reranker)
    r.instruction = "Find relevant evidence"
    r._quantized = True
    r._yes_id, r._no_id = 0, 1
    r.tokenizer = tokenizer
    seen = []

    class Model:
        device = "cpu"

        def __call__(self, input_ids, attention_mask, logits_to_keep):
            seen.append({"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone()})
            logits = torch.zeros((len(input_ids), logits_to_keep, 2))
            logits[:, -1, 0] = (input_ids == vocabulary.index("ANSWER")).any(dim=1).float() * 5
            return SimpleNamespace(logits=logits)

    r.model = Model()
    return module, r, seen


@pytest.mark.parametrize("fenced", [False, True], ids=["paragraph", "code-block"])
def test_evidence_beyond_old_character_limit_reaches_model(scorer, fenced):
    from mainframe.core.chunker import chunk_markdown

    module, r, seen = scorer
    paragraph = "routine context " * 260 + "ANSWER closes the ticket.\n"
    source = "# Handbook\n\n" + (f"```text\n{paragraph}```\n" if fenced else paragraph)
    chunk = next(c for c in chunk_markdown(source) if "ANSWER" in c.text)
    assert chunk.text.index("ANSWER") > 3950
    assert source[chunk.char_start:chunk.char_end].strip() == chunk.text
    scores = r._score_qwen3("query", ["document", chunk.text])
    assert scores[1] > scores[0]
    prompt = module._QWEN3_PREFIX + module.qwen3_pair_text("query", chunk.text, r.instruction) + module._QWEN3_SUFFIX
    expected = r.tokenizer.encode(prompt)
    assert len(expected) < module._QWEN3_MAX_LENGTH
    assert seen[0]["input_ids"][1].tolist() == expected


def test_token_overflow_preserves_suffix_context_bound_and_left_padding(scorer):
    module, r, seen = scorer
    r._score_qwen3("query", ["document", "routine context " * 3000])
    batch = seen[0]
    assert batch["input_ids"].shape == (2, module._QWEN3_MAX_LENGTH)
    suffix = r.tokenizer.encode(module._QWEN3_SUFFIX, add_special_tokens=False)
    for row in batch["input_ids"]:
        assert row[-len(suffix):].tolist() == suffix
    assert batch["attention_mask"][0, 0] == 0
    assert batch["attention_mask"][0, -1] == 1
    assert batch["attention_mask"][1].all()


def test_ordinary_prompt_tokens_and_int8_batch_composition_are_unchanged(scorer):
    module, r, seen = scorer
    docs = ["routine " * i for i in range(1, 12)]
    r._score_qwen3("query", docs)
    assert [len(batch["input_ids"]) for batch in seen] == [8, 3]
    for offset, batch in zip((0, 8), seen):
        prompts = [module._QWEN3_PREFIX + module.qwen3_pair_text("query", doc, r.instruction)
                   + module._QWEN3_SUFFIX for doc in docs[offset:offset + 8]]
        expected = r.tokenizer(prompts, padding=True, truncation=True,
                               max_length=module._QWEN3_MAX_LENGTH, return_tensors="pt")
        assert torch.equal(batch["input_ids"], expected["input_ids"])
        assert torch.equal(batch["attention_mask"], expected["attention_mask"])
