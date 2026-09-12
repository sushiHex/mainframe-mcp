"""Exercise both production scoring paths with a local, deterministic tokenizer."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


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
