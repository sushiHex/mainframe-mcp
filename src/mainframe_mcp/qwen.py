"""Shared token budgeting for the native Qwen rerankers."""

_WIDEST_VOCAB_SPELLING = "_mainframe_qwen_widest_vocab_spelling"


def _prefix_char_budget(tokenizer, token_budget: int) -> int:
    """Return a safe pre-tokenization bound for Qwen's byte-level BPE.

    A Qwen vocabulary token is a non-empty sequence of byte-level characters.
    Therefore a prefix that fits ``token_budget`` tokens cannot be longer than
    ``token_budget`` times the widest vocabulary spelling. This is a bound from
    the loaded vocabulary, not a heuristic character cutoff. It lets us reject
    a pathological single chunk before the tokenizer has to materialize all of
    its IDs, while leaving every potentially fitting prompt untouched.
    """
    widest = getattr(tokenizer, _WIDEST_VOCAB_SPELLING, None)
    if widest is None:
        vocab = tokenizer.get_vocab()
        widest = max(map(len, vocab), default=0)
        setattr(tokenizer, _WIDEST_VOCAB_SPELLING, widest)
    if widest <= 0:
        raise ValueError("native Qwen tokenizer has no vocabulary spellings")
    return token_budget * widest


def tokenize_with_suffix(tokenizer, prompts: list[str], suffix: str, max_length: int):
    """Preserve prompts that fit; reserve the scoring suffix when they overflow.

    Qwen's assistant suffix begins with a special token, so its token boundary
    is stable. Tokenize the preceding prompt to its reserved budget and append
    the complete suffix IDs. Padding happens afterward, using the model's
    left-padding tokenizer.
    """
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    budget = max_length - len(suffix_ids)
    if budget <= 0:
        raise ValueError("reranker context must leave room before the scoring suffix")
    if not suffix or any(not prompt.endswith(suffix) for prompt in prompts):
        raise ValueError("native Qwen prompts must end with the scoring suffix")
    char_budget = _prefix_char_budget(tokenizer, budget)
    prefixes = [prompt[:min(len(prompt) - len(suffix), char_budget)] for prompt in prompts]
    rows = tokenizer(prefixes, padding=False, truncation=True, max_length=budget,
                     return_attention_mask=False)["input_ids"]
    rows = [ids + suffix_ids for ids in rows]
    return tokenizer.pad({"input_ids": rows}, padding=True,
                         return_attention_mask=True, return_tensors="pt")
