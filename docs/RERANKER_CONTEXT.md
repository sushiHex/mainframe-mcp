# Native reranker context

The v2 default reranker is [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/e61197ed45024b0ed8a2d74b80b4d909f1255473), loaded in BF16 with its pinned default technical-documentation instruction and `<Document>` marker. It uses the native suffix-preserving yes/no-logit adapter, a 2,048-token joint query/document budget, and saved-order batches of eight. The default response returns three passages.

## Input contract

- Bound the query and document prompt by the tokenizer-derived budget, reserving the complete assistant scoring suffix.
- Preserve token IDs for fitting prompts. For overflow, retain the leading prompt tokens that fit and preserve the scoring suffix.
- Apply left padding after bounding. The score comes from the final assistant-response position; unused decoder caching remains disabled.
- Keep candidate order stable through batching and retry an individual item only after a real device out-of-memory error.

The reranker score is a relative ranking signal for passages in the same response. It is not a calibrated probability that a passage is correct, current, or sufficient to answer the query.

## Evaluation scope

The selected contract came from a sealed 36-question source-grounded evaluation with fixed 20-candidate pools, complete-fact rubrics, and masked agent passage review. It tied the NF4 control on reviewed top-three complete facts (33/36) and covered all eight critical facts, while the control covered seven. It also used less model memory on that same task. See [the compact comparison](COMPACT_RERANKERS.md) for the full scope, historical measurements, and limitations.

The earlier 187-query comparisons and token-budget regression checks remain historical evidence. Their literal source/passage labels are not answer-sufficiency judgments, and known challenges are not untouched holdouts. Do not infer a general quality guarantee from either the historical labels or the corrected validation set.

## Operational effect

Changing only the reranker changes the ordering of existing first-stage candidates. It does not change stored embeddings, chunks, or the native index identity, so it does not require re-embedding or rebuilding the index. An embedding-contract change does require the normal offline rebuild procedure.
