"""Shared token budgeting for the native Qwen rerankers."""


def tokenize_with_suffix(tokenizer, prompts: list[str], suffix: str, max_length: int):
    """Preserve prompts that fit; reserve the scoring suffix when they overflow.

    Tokenize complete prompts first to preserve ordinary token boundaries.
    Padding happens after budgeting, using the model's left-padding tokenizer.
    """
    rows = tokenizer(prompts, padding=False, truncation=False,
                     return_attention_mask=False)["input_ids"]
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    budget = max_length - len(suffix_ids)
    if budget <= 0:
        raise ValueError("reranker context must leave room before the scoring suffix")
    rows = [ids if len(ids) <= max_length else ids[:budget] + suffix_ids for ids in rows]
    return tokenizer.pad({"input_ids": rows}, padding=True,
                         return_attention_mask=True, return_tensors="pt")
