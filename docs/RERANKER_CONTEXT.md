# Native reranker context

Both v1 and v2 use the same token-budget helper for native Qwen reranking.
The earlier document cutoff discarded everything after 3,000 characters.
An observed answer started at character 3,950 of a 4,719-character paragraph,
although the entire prompt occupied only 1,401 tokens in the pinned tokenizer.
The answer was present in both embedding models' candidate pools.

## Input contract

- Keep the existing prompt scaffold, 1,000-character query bound, and
  2,048-token maximum context.
- Before tokenization, cap only an impossible-to-fit prompt at the remaining
  token budget times the widest spelling in the loaded Qwen byte-level BPE
  vocabulary. This is a tokenizer-derived safety bound, not a fixed character
  cutoff: every prompt that can fit remains untouched.
- Prompts that fit retain their exact token IDs. The tokenizer then truncates
  the bounded pre-suffix prompt to the remaining token budget.
- For overflow, retain the leading tokens that fit and reserve the complete
  assistant suffix. The score must come from the assistant response position.
- Apply left padding afterward. Keep saved-order batches of eight for both the
  selected NF4 path and BF16 fallback, and retain item-at-a-time retry on a
  real CUDA out-of-memory error.

The production scaffold labels the document with `<Doc>`. The pinned
Qwen3 Reranker 0.6B and 4B model cards and chat templates use `<Document>`. A
separately preregistered 187-query comparison changed only this marker. It
improved some aggregate counts for Qwen 0.6B but retained a true root-cause
mechanism loss. The selected Qwen 4B configuration keeps the measured `<Doc>`
prompt. See the [reranker comparison](RERANKER_COMPARISON.md).

This does not expand the context or introduce sliding windows, extra model
calls, new chunks, or index migrations. Token-dense documents can still exceed
the context; their tail is omitted. A pathological unsplittable chunk cannot
make tokenizer input grow without bound. Quantized scores can change for other
candidates sharing a batch with an expanded document.

## Regression evidence

GPU-free tests exercise both production scoring paths with a deterministic local
tokenizer. They cover late evidence in oversized paragraphs and fenced blocks,
suffix preservation on overflow, left padding, unchanged ordinary token IDs,
the historical INT8 batch composition, and pre-tokenization bounding for a huge
single prompt. The late-evidence regression failed against both original
backends before the fix.

The bounded-tokenization follow-up was checked on the same frozen candidate
pools using the cached tokenizer, without loading model weights. All **7,480
prompts across 1,122 batches** retained identical token IDs and attention masks,
including 41 long fitting prompts and four over-context prompts. A synthetic
million-character prompt verified the input bound and complete scoring suffix;
a separate regression verifies that each tokenizer's vocabulary is scanned
only once. This preserves measured inputs; no new quality or latency gain is
claimed.

Live comparisons retain the original question labels and source snapshots.
The previous fresh challenge is now a known regression set; it must not be
presented as an unseen holdout after using it to develop this fix. Record
retrieval changes and resource measurements before claiming a quality gain.

## Measured follow-up (2026-09-11)

The frozen candidate pools from the original public benchmark, 47-question
working sample, and 20-question challenge were checked for both embedding
models: **374 query/pool combinations**. Embeddings and candidate selection
were held fixed. Complete token batches were identical for 314 reused results;
60 queries, including controls, were scored through the real GPU reranker.
All 33 replayed original-score controls reproduced their previous rankings and
rounded scores exactly. Source and manifest hashes, unchanged labels, and every
result's metrics were independently verified. This is a reranking regression
check, not a new end-to-end embedding benchmark.

Expected-passage counts below are **first place / top three**:

| Frozen set | Qwen pools, before → after | Harrier pools, before → after |
|---|---|---|
| Mainframe benchmark, 20 questions | 17/19 → 17/19 | 17/19 → 17/19 |
| Working sample, 47 questions | 23/34 → 23/34 | 27/37 → 26/37 |
| Known challenge, 20 questions | 16/19 → 15/19 | 16/19 → 15/20 |

SciFact's 100 document-label questions retained all metrics: document hits at
first place/top three stayed 79/90 for Qwen and 81/91 for Harrier. The working
sample's Harrier document hits changed from 30/41 to 29/41; MRR@3 changed from
0.7411 to 0.7305 and nDCG@3 from 0.7614 to 0.7535. Other document metrics stayed
unchanged. Its affected first-place result swapped with a related result at
rank two; the labeled passage remained available.

The formerly hidden critical answer now appears for both models. Harrier retains
every baseline top-three passage and reaches **5/5 critical matches**. Qwen
remains at 4/5 exact critical matches: another expected excerpt leaves its top
three, although manual review found the same explanation in its first two
results. The frozen score was not changed. Two challenge answers move below
first place for each model, while the repaired answer gains first place.
These are measured tradeoffs, not an across-the-board quality improvement.

The replay used Qwen3-Reranker-4B INT8, revision
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`, on an RTX 3090 with PyTorch
2.11.0+cu128, Transformers 5.7.0, and Sentence Transformers 5.4.1. It loaded
only the reranker: peak CUDA allocation/reservation was **7.951/9.789 GiB**,
returning to 0.008 GiB allocated after release. Those numbers are not a full
embedding-stack footprint. CPU tests overlapped the final scoring phase; no
latency improvement is claimed.

The private helper omitted the final report's run identifier, so its supervisor
rejected the report format after scoring completed. The helper was corrected;
an independent check verified all 374 saved records, source identities, labels,
and metrics without rewriting the original report or claiming that supervisor
check passed. The runtime regression suite passed **542 tests, with 1 skipped**.
Raw sources, queries, and reports remain outside Git. These measurements
preceded Harrier default promotion; the accepted ranking tradeoffs remain
recorded here. See the [current default stack](../README.md#models-and-hardware).

The installed Harrier wheel also passed an unpaced daemon check across two
explicitly configured projects: 18 documents, 303 chunks, authenticated searches,
private observation recording, shutdown, and restart. The unchanged restart
reused every row. Startup and search peaked at **6.886 GiB allocated / 7.504 GiB
reserved**; this small scope is not comparable to the broader replay peak.
The supervisor accepted this run's completion report. The operator described
the validation session as smooth; displayed-frame timing was not measured.
Two earlier user questions were recorded with historical provenance, separately
from scripted probes. They are not new independent evaluation questions.

## KV-cache probe

A later 23-query probe disabled the Qwen reranker's decoder cache and included
the largest batches from the comparison pool. Its worker and supervisor both
completed. All **460 raw scores, full rankings, and metrics** matched the cached
control exactly. The probe reached **5.701 GiB allocated / 7.016 GiB reserved**,
while the separate full 187-query cached control reached **7.951 / 9.283 GiB**.
This was a targeted subset, not a matched full-run memory experiment, so it does
not establish the full-run memory saving. Paired active time was **75.610 seconds
without cache** and **73.569 seconds with cache**; no speed improvement is
claimed.

The later [reranker comparison](RERANKER_COMPARISON.md) selects Qwen 4B NF4 as
the preferred shared-GPU configuration. The selection keeps the same pinned
weights and `<Doc>` prompt, uses BF16 compute and disables the cache. Its
production scoring contract reproduced all 3,740 scores and 187 full rankings
exactly. An installed two-document synthetic daemon smoke also passed startup,
search, mutation, restart, status-contract, and cleanup checks.
