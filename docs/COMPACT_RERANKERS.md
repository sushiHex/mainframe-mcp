# Compact reranker evaluation

> Status: Qwen3-Reranker-0.6B BF16, using the pinned default `<Document>`
> contract, is the selected compact default. Quality, memory, answer generation,
> and production verification are reported separately below.

This note records the completed shared-GPU comparisons behind the fixed
first-stage retrieval setup. Each trial reranked the same frozen pool of 20
candidates, using batch size eight and a 2,048-token joint pair budget unless
noted below. Results are useful for choosing what to validate; they are not a
general retrieval benchmark.

## Completed candidates and contracts

| Candidate | Native scoring contract | Weight license / release gate |
|---|---|---|
| [Qwen3 Reranker 4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B/tree/22e683669bc0f0bd69640a1354a6d0aebcfeede5) NF4 control | Production Qwen yes/no-logit contract, NF4 weights with BF16 compute | Apache-2.0; reviewed control |
| [Qwen3 Reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/e61197ed45024b0ed8a2d74b80b4d909f1255473) | Native Qwen suffix-preserving yes/no logit. Default, evidence, and released-generic instructions were separate trials. | Apache-2.0; compact candidate |
| [mxbai-rerank-base-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2/tree/3ea9d4dffa7d12a4f366be8e275c349de9fc9865) | Sentence Transformers CrossEncoder with the model's Transformer + LogitScore graph, FP16 | Apache-2.0 |
| [ettin-reranker-150m-v1](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1/tree/025501c4e0f9bbeb4c5b198318e0089ff061cc14) | Native Sentence Transformers CrossEncoder modules, BF16 | Apache-2.0; compact candidate |
| [ettin-reranker-400m-v1](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1/tree/5dca36282a5d85f368d2544002513a29159b4c9e) | Native Sentence Transformers CrossEncoder modules and released scalar head, BF16; 512-token document window | Apache-2.0; compact candidate |
| [SkillReason-reranker-0.6b](https://huggingface.co/donghongjiang/SkillReason-reranker-0.6b/tree/ec55bc8069585862ddae905aeaa312ec39266909) | Native causal yes-minus-no logit gap, BF16 | Apache-2.0 |
| [KaLM-Reranker-V1-Nano](https://huggingface.co/KaLM-Embedding/KaLM-Reranker-V1-Nano/tree/8042f04d3258ee3f5de42b735187dbf9d05ef9d9) | Encoder-decoder reranker, BF16; 1,024-token document encoder, 512-token query decoder, and P(yes) score | Apache-2.0; [Gemma terms](https://ai.google.dev/gemma/terms) also apply upstream |
| KaLM document-window repair (same pinned checkpoint) | Encoder-decoder reranker, BF16; 2,048-token document encoder, 512-token query decoder, and P(yes) score | Same checkpoint terms |

The listed license summaries record the pinned model-card terms; they do not
establish quality or integration suitability.

## Historical development results

The historical suite has 187 queries: public scientific document labels,
source-document passage labels, working-material passage labels, and known
challenge passage labels. Counts below are frozen-label coverage at ranks 3 / 5
across that suite; source identity and literal passage labels are not
answer-sufficiency judgments.

| Candidate | Pair budget | Science (100) | Source docs (20) | Working material (47) | Known challenges (20) | Total (187) |
|---|---|---:|---:|---:|---:|---:|
| Qwen 4B NF4 control | 2,048 joint | 88 / 94 | 19 / 20 | 37 / 40 | 19 / 20 | 163 / 174 |
| Qwen 0.6B, evidence instruction | 2,048 joint | 89 / 91 | 19 / 20 | 34 / 39 | 20 / 20 | 162 / 170 |
| Qwen 0.6B, default instruction | 2,048 joint | 88 / 91 | 18 / 20 | 36 / 40 | 19 / 20 | 161 / 171 |
| Qwen 0.6B, released generic instruction | 2,048 joint | 86 / 91 | 18 / 20 | 37 / 38 | 16 / 19 | 157 / 168 |
| mxbai-rerank-base-v2 | 2,048 joint | 88 / 91 | 18 / 20 | 38 / 40 | 19 / 20 | 163 / 171 |
| ettin-reranker-150m-v1 | 2,048 joint | 89 / 92 | 19 / 20 | 36 / 38 | 18 / 18 | 162 / 168 |
| ettin-reranker-400m-v1, 512-token window | 512 tokens | 88 / 93 | 19 / 20 | 37 / 40 | 20 / 20 | 164 / 173 |
| SkillReason-reranker-0.6b | 2,048 joint | 85 / 91 | 18 / 19 | 36 / 38 | 19 / 20 | 158 / 168 |
| KaLM-Reranker-V1-Nano | 1,024 doc / 512 query | 87 / 90 | 19 / 19 | 36 / 39 | 18 / 19 | 160 / 167 |
| KaLM document-window repair | 2,048 doc / 512 query | 87 / 90 | 19 / 19 | 36 / 39 | 19 / 20 | 161 / 168 |

A separate 24-query synthetic-public source-lookup probe was used for limited
development diagnostics. Its generic wording and literal anchors are too weak
to headline as answer quality, so it is excluded from the table and from model
selection.

## Corrected validation results

The locked corrected validation set contains 36 source-grounded questions, of
which eight have predeclared critical rationales. Every finalist reranked the
same frozen pool of 20 candidates per question. The table reports the final
masked agent passage-review result: a passage set covers a question only when
it supplies all required facts, with equivalent evidence accepted. The
conservative automated source-receipt confirmation was retained in private
records: it only confirms that the audited supporting excerpt was returned
verbatim from its frozen source, and does not decide fact sufficiency.

| Candidate | Top-3 complete facts (36) | Top-3 critical facts (8) | Top-5 complete facts (36) | Top-5 critical facts (8) |
|---|---:|---:|---:|---:|
| Qwen 4B NF4 control | 33 / 36 | 7 / 8 | 34 / 36 | 8 / 8 |
| Qwen 0.6B default instruction | 33 / 36 | 8 / 8 | 33 / 36 | 8 / 8 |
| SkillReason 0.6B | 32 / 36 | 8 / 8 | 33 / 36 | 8 / 8 |
| Ettin 400M, 512-token window | 30 / 36 | 7 / 8 | 32 / 36 | 8 / 8 |
| KaLM document-window repair | 27 / 36 | 7 / 8 | 31 / 36 | 8 / 8 |

Top-three remains the default context policy. Top-five uses the same reranker
scores but adds two passages, so it is reported as a context-cost tradeoff and
not treated as a top-three recovery. These are masked agent passage reviews;
no human review or answer generation was used.

The Qwen 0.6B default instruction preserves all five predeclared historical
critical target passages within its top three. It also moves one known
noncritical implementation root-cause passage from rank two to rank five
relative to the control. That retained top-five evidence is a tradeoff under
the default top-three policy, not a universal quality claim.

The sealed evaluation selects the pinned Qwen 0.6B default `<Document>`
contract in BF16. It matches the control's reviewed top-three fact count,
covers one additional reviewed critical fact, and uses less observed model
memory on the same frozen validation task. No prompt or cutoff was tuned on
validation. The fixed CPU passage-to-answer check and integration checks are
separate evidence; this small source-grounded set does not establish general
quality or universal parity.

## Passage-to-answer check

Twelve questions were selected before validation rankings. Each was answered
once with the selected model's top three passages and once with its top five,
using the same Phi4-mini Q4_K_M generator on CPU: temperature 0, fixed seed,
8,192-token context, and 256-token output budget. This generator is an
evaluation caller, not a new Mainframe runtime default. An agent reviewer judged
actual answers and citations before the randomized arm labels were revealed;
visible passage counts prevented complete cutoff blinding.

For Qwen 0.6B, top three was preferred on **6/12 questions**, top five on **2/12**,
with **4 ties**. All 24 contexts supplied the required facts, yet only 7/12
top-three answers and 3/12 top-five answers stated all required facts correctly.
Misread evidence and unjustified abstentions were generator errors. One output
at each cutoff reached the output budget; both remain in the counts.
Mean prompt size rose from 1,137 to 1,713 tokens at five.

The same scheduled probe on the provisional Ettin window model preferred three
on 3 questions and five on 4, with 5 ties; both cutoffs supplied complete facts
for 11/12 questions. These small, single-sample probes establish neither
universal cutoff superiority nor normal-corpus answer accuracy. Keep three as
the default and request more evidence when needed.

## Production verification

Both production rerankers reproduced all **4,220 scores and 211 full rankings**
from the frozen development replay exactly. The only later source edits were
descriptions; code comparison confirmed unchanged scoring.

The installed wheel passed a paced, two-document daemon smoke: **36 searches,
nine lifecycle checks**, no failed jobs, and no health-probe failures. It covered
startup, capture, watcher additions/edits/renames/deletions, authentication,
restart, and shutdown. Restart upserted zero unchanged rows. Both Harrier and
Qwen executed paced forwards in both phases and were released afterward.
All 55 installed Python modules matched the wheel; dependency versions stayed
unchanged. This synthetic lifecycle check does not measure normal-corpus quality.

An earlier smoke was rejected because its hooks missed Harrier's direct forward
calls. The instrumentation was corrected and regression-tested before the
successful repeat. The GPU-free repository suite passed 595 tests with one
skip; a clean wheel installation also passed outside the checkout.

## Resource measurements

These are observed PyTorch allocator values under the shared GPU. They exclude
the first-stage embedder and are not total system or driver memory guarantees.
The first table is the matched corrected-validation task and is the appropriate
comparison for the selected contract.

| Candidate | Loaded allocation (GiB) | Warm peak allocated (GiB) | Warm peak reserved (GiB) |
|---|---:|---:|---:|
| Qwen 4B NF4 control | 2.492583 | 3.232769 | 3.603516 |
| Qwen 0.6B default instruction | 1.109735 | 1.395288 | 1.523438 |

Earlier development measurements remain useful as historical ranges, but were
not a joint run with the corrected validation task:

| Candidate | Loaded allocation (GiB) | Warm peak allocated / reserved (GiB) |
|---|---:|---:|
| Qwen 4B NF4 control | 2.493 | 3.738 / 6.154 historical run |
| Qwen 0.6B variants | 1.110 | 1.395–1.621 / 1.689–2.170 |
| mxbai-rerank-base-v2 | 0.928 | 1.711 / 2.625 |
| ettin-reranker-150m-v1 | 0.288 | 0.712 / 1.088 |
| ettin-reranker-400m-v1, 512-token window | 0.737 | 0.877 / 1.025 |
| SkillReason-reranker-0.6b | 1.110 | 1.621 / 2.172 |
| KaLM-Reranker-V1-Nano | 1.477 | 1.911 / 2.357 |
| KaLM document-window repair | 1.477 | 1.918 / 2.355 |

The 512-token Ettin 400M window recovered two pre-identified historical
critical anchors into its top-three passages. This is a development diagnostic,
not a validation outcome or an answer-sufficiency result.

## First-stage ablation

A CPU-only hybrid-first-stage ablation retained the saved candidate order and
applied no reranker. Its frozen-label coverage was 148 / 163 at ranks 3 / 5,
versus 163 / 174 for the NF4 control on the historical suite. This shows that
reranking adds value over the same candidates on the frozen label metric; it
does not measure answer sufficiency.

## Interpretation

The historical comparisons measure frozen source and passage labels. They do
not run an answer generator, and matching a source or related mechanism does
not prove that a passage supplies the requested fact. The corrected validation
therefore adds complete-fact rubrics and masked agent passage review. Its
results select the Qwen 0.6B default contract under the fixed top-three policy.
Training was unnecessary under the frozen promotion criteria. Any future
tuning informed by these results needs a new, unseen final validation set.
