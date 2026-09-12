# Native reranker context

Both v1 and v2 use the same token-budget helper for native Qwen reranking.
The earlier document cutoff discarded everything after 3,000 characters.
An observed answer started at character 3,950 of a 4,719-character paragraph,
although the entire prompt occupied only 1,401 tokens in the pinned tokenizer.
The answer was present in both embedding models' candidate pools.

## Input contract

- Keep the existing prompt scaffold, 1,000-character query bound, and
  2,048-token maximum context.
- Tokenize complete prompts. Prompts that fit retain their exact token IDs.
- For overflow, retain the leading tokens that fit and reserve the complete
  assistant suffix. The score must come from the assistant response position.
- Apply left padding afterward. Keep INT8 batches of eight in candidate order
  and retain item-at-a-time retry on a real CUDA out-of-memory error.

This does not expand the context or introduce sliding windows, extra model
calls, new chunks, or index migrations. Token-dense documents can still exceed
the context; their tail is omitted. Larger inputs within the existing bound
can increase attention work and memory use. Quantized scores can change for
other candidates sharing a batch with an expanded document.

## Regression evidence

GPU-free tests exercise both production scoring paths with a deterministic local
tokenizer. They cover late evidence in oversized paragraphs and fenced blocks,
suffix preservation on overflow, left padding, unchanged ordinary token IDs,
and the original INT8 batch composition. The late-evidence regression failed
against both original backends before the fix.

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
Raw sources, queries, and reports remain outside Git. Model defaults are unchanged;
the change is available for review and scoped experimental use.

The installed Harrier wheel also passed an unpaced daemon check across two
explicitly configured projects: 18 documents, 303 chunks, authenticated searches,
private observation recording, shutdown, and restart. The unchanged restart
reused every row. Startup and search peaked at **6.886 GiB allocated / 7.504 GiB
reserved**; this small scope is not comparable to the broader replay peak.
The supervisor accepted this run's completion report. The operator described
the validation session as smooth; displayed-frame timing was not measured.
Two earlier user questions were recorded with historical provenance, separately
from scripted probes. They are not new independent evaluation questions.
