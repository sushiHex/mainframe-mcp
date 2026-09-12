# Reranker comparison

The selected evaluation contract is [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/e61197ed45024b0ed8a2d74b80b4d909f1255473), revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`, in BF16. It uses the native Qwen suffix-preserving yes/no-logit adapter, the default technical-documentation instruction with `<Document>`, a 2,048-token joint budget, saved-order batches of eight, and a default of three returned passages.

This choice came from a sealed 36-question, source-grounded comparison with fixed 20-candidate pools and masked agent passage review. Qwen matched the 4B NF4 control on reviewed complete-fact coverage at top three (33/36) and covered all eight predeclared critical facts (control: 7/8). On the same task it measured 1.109735 GiB loaded, 1.395288 GiB peak allocated, and 1.523438 GiB peak reserved, compared with 2.492583, 3.232769, and 3.603516 GiB for the NF4 control.

The full candidate contracts, historical frozen-label results, validation method, reviewed results, and model-card license terms are in [Compact reranker evaluation](COMPACT_RERANKERS.md). Historical label coverage is retained there as historical evidence; a matching label or source does not prove answer sufficiency. The relative yes/no ranking score is not a correctness confidence.

Top three is the default context policy. Top five reuses the same scores but adds two passages, and is reported only as a context-cost tradeoff. No prompt or cutoff was tuned on the sealed validation. A fixed CPU passage-to-answer check and integration checks are separate from the selection evidence.

Changing the reranker only changes candidate ordering. It does not require re-embedding or rebuilding the index; changing the embedding contract does.
